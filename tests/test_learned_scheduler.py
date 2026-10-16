"""scheduler="learned" (mac.MacLink.learned_prepare, docs/learned-scheduler.md).

L1 zero scorer and zero external log-weight: bitwise the PF engine (outputs and every UL state tensor), both timescales,
   slot and per-RBG PF updates.
L2 the log-weight moves resources: a robot with a large weight takes the carrier, with a small one it starves.
L3 group-indexed scorer parameters and the external input reach lw exactly; features are sane.
L4 a partial reset leaves the other envs bitwise unaffected and clears only the reset envs' last message class.
L5 config: unused-field report, one cell only, triton refuses it.
GPU: the graph backend is bitwise equal to the reference while the trainer rewrites l_ext and the scorer parameters
     between steps, and no new graph is captured for those writes.
"""
import math

import pytest
import torch

from isaac_net import NRConfig, Requests, make_engine
from isaac_net.core import multicell
from isaac_net.core.mac import L_FEAT
from isaac_net.core.nr_fast import TritonUnsupported, state_dict

E, R = 4, 6


def _drive(nets, steps, seed=0, logw=None, params=None, reset_at=None, dev="cpu"):
    """Same workload on every engine; logw(t) -> [E,R] or None; params(t) -> (w1,b1,w2,b2) or None."""
    g = torch.Generator().manual_seed(seed)
    outs = [[] for _ in nets]
    for t in range(steps):
        send = (torch.rand(E, R, generator=g) < 0.5).long() * torch.randint(1, 3, (E, R), generator=g)
        snr = 5 + 25 * torch.rand(E, R, generator=g)
        lw = None if logw is None else logw(t)
        pr = None if params is None else params(t)
        for i, n in enumerate(nets):
            if reset_at is not None and t == reset_at[0] and i in reset_at[1]:
                n.reset(torch.tensor([0], device=dev))
            if lw is not None:
                n.set_sched_logw(lw.to(dev))
            if pr is not None:
                n.set_sched_params(*[x.to(dev) for x in pr])
            n.submit(None, Requests(send.to(dev)))
            outs[i].append(n.step(None, snr.to(dev)))
    return outs


def _same(x, y):
    if x.is_floating_point():
        return bool(((x == y) | (torch.isnan(x) & torch.isnan(y))).all())
    return torch.equal(x, y)


def _eq_out(a, b, envs=slice(None)):
    for k in a:
        if torch.is_tensor(a[k]) and a[k].dim() >= 1 and a[k].shape[0] == E:
            assert _same(a[k][envs], b[k][envs]), k


UL_STATE = ("sent", "floor", "olla", "avg", "bsr", "sr_t", "csi", "h_state", "h_lo", "h_hi", "h_ready", "h_ntx",
            "h_mcs", "h_comb")


@pytest.mark.parametrize("extra", [dict(), dict(learned_timescale="slot"), dict(pf_update="rbg", pf_avg_idle="freeze"),
                                   dict(n_harq=1, eff_sinr="mean_db", n_prb=50, rbg_size=10)])
def test_l1_zero_weights_are_bitwise_pf(extra):
    pf_extra = {k: v for k, v in extra.items() if not k.startswith("learned_")}
    pf = make_engine("L2", E, R, "cpu", NRConfig(**pf_extra), seed=3)
    ln = make_engine("L2", E, R, "cpu", NRConfig(scheduler="learned", **extra), seed=3)
    oa, ob = _drive([pf, ln], 40)
    for a, b in zip(oa, ob):
        _eq_out(a, b)
    for n in UL_STATE:
        assert torch.equal(getattr(pf.net.ul, n), getattr(ln.net.ul, n)), n
    assert torch.equal(pf.net.ul.q.fin, ln.net.ul.q.fin)


def _served(net, steps, lw):
    """All robots backlogged with large frames at equal SNR; delivered bytes per robot."""
    tot = torch.zeros(E, R)
    sizes = torch.tensor(net.config.msg_sizes)
    for t in range(steps):
        net.set_sched_logw(lw)
        net.submit(None, Requests(torch.full((E, R), 2, dtype=torch.long)))
        o = net.step(None, torch.full((E, R), 20.0))
        tot += (o["delivered"] * sizes[(o["cls"] - 1).clamp(min=0)]).sum(-1) if "cls" in o else o["delivered"].sum(-1)
    return tot


def test_l2_weight_moves_resources():
    cfg = NRConfig(scheduler="learned", msg_sizes=(4000.0, 30000.0))
    lw = torch.zeros(E, R)
    lw[:, 0] = 6.0
    hi = _served(make_engine("L2", E, R, "cpu", cfg, seed=1), 30, lw)
    lw[:, 0] = -6.0
    lo = _served(make_engine("L2", E, R, "cpu", cfg, seed=1), 30, lw)
    base = _served(make_engine("L2", E, R, "cpu", cfg, seed=1), 30, torch.zeros(E, R))
    share = lambda x: (x[:, 0] / x.sum(-1).clamp(min=1)).mean().item()
    # robot 0's own offered load caps its share near 0.3; a large weight brings it close to that cap
    assert share(hi) > 1.3 * share(base) and share(base) > 5 * share(lo) and share(lo) < 0.05


def test_l3_groups_params_ext_and_features():
    cfg = NRConfig(scheduler="learned", learned_groups=2, learned_hidden=4, learned_logw_clip=5.0)
    net = make_engine("L2", E, R, "cpu", cfg, seed=0)
    net.set_sched_groups(torch.tensor([0, 1, 0, 1]))
    z = lambda *s: torch.zeros(*s)
    net.set_sched_params(z(2 * L_FEAT, 4), z(4), z(4), torch.tensor(1.5), group=1)
    ext = torch.zeros(E, R)
    ext[:, 1] = 9.0                     # clamped to 5
    net.set_sched_logw(ext)
    net.set_sched_hint(torch.full((E, R), 0.5))
    f0 = net.sched_features(torch.full((E, R), 10.0))
    assert f0.shape == (E, R, L_FEAT) and torch.equal(f0[..., 0], torch.zeros(E, R))   # empty queues
    send = torch.zeros(E, R, dtype=torch.long)
    send[:, 2] = 2
    net.submit(None, Requests(send))
    net.step(None, torch.full((E, R), -5.0))        # bad channel: the 30 kB frame stays queued
    st = net.sched_state()
    want = torch.where(torch.tensor([0, 1, 0, 1]).bool()[:, None], 1.5, 0.0).expand(E, R).clone()
    want[:, 1] = (want[:, 1] + 9.0).clamp(max=5.0)
    assert torch.allclose(st["logw"], want)
    f = net.sched_features(torch.full((E, R), -5.0))
    assert (f[:, 2, 0] > 0).all() and (f[:, 0, 0] == 0).all()                  # backlog only where sent
    assert (f[:, 2, 1] > 0).all()                                                # head-of-line age grows
    assert torch.allclose(f[:, 2, 5], torch.tensor(1.0)) and (f[:, 0, 5] == 0).all()   # last class 2 of 2
    assert torch.allclose(f[..., 6], torch.tensor(0.5))
    p = net.sched_params()
    assert p["b2"].tolist() == [0.0, 1.5] and p["w1"].shape == (2, 2 * L_FEAT, 4)


def test_l4_partial_reset_isolation():
    cfg = NRConfig(scheduler="learned")
    a, b = (make_engine("L2", E, R, "cpu", cfg, seed=7) for _ in range(2))
    lw = lambda t: torch.sin(torch.arange(E * R, dtype=torch.float32).view(E, R) + t)
    oa, ob = _drive([a, b], 30, logw=lw, reset_at=(12, {0}))
    for x, y in zip(oa, ob):
        _eq_out(x, y, slice(1, None))
    assert torch.equal(a.net.ul.l_cls[1:], b.net.ul.l_cls[1:])


def test_l5_config_and_backends():
    assert "learned_hidden" in NRConfig(learned_hidden=8).unused_fields("L2")
    assert "learned_hidden" not in NRConfig(scheduler="learned", learned_hidden=8).unused_fields("L2")
    with pytest.raises(ValueError):
        make_engine("L2", E, R, "cpu", multicell(3, scheduler="learned"), seed=0)
    with pytest.raises(AssertionError):
        NRConfig(scheduler="learned", learned_timescale="tti")
    from isaac_net.core.nr_fast import NRTritonEngine
    assert any("learned" in f for f, _ in NRTritonEngine.refusals(NRConfig(scheduler="learned")))
    with pytest.raises(ValueError):
        make_engine("L2", E, R, "cpu", NRConfig(), seed=0).set_sched_logw(torch.zeros(E, R))
    assert "learned" in NRConfig(scheduler="learned").describe()
    assert abs(math.exp(NRConfig().learned_logw_clip) - 16.0) < 1e-9          # default weight range [1/16, 16]


def test_l6_weight_range_and_replace_ablation():
    net = make_engine("L2", E, R, "cpu", NRConfig(scheduler="learned"), seed=0)
    net.set_sched_logw(torch.full((E, R), 10.0))
    net.submit(None, Requests(torch.ones(E, R, dtype=torch.long)))
    net.step(None, torch.full((E, R), 10.0))
    assert torch.allclose(net.net.ul.l_wexp, torch.tensor(16.0))
    a = make_engine("L2", E, R, "cpu", NRConfig(scheduler="learned", learned_metric="weight"), seed=0)
    _drive([a], 10)          # runs; the channel-blind variant is an ablation


@pytest.mark.gpu
@pytest.mark.parametrize("extra", [dict(), dict(learned_timescale="slot", learned_groups=3),
                                   dict(n_harq=1, eff_sinr="mean_db", n_prb=50, rbg_size=10, control_step_ms=20.0,
                                        timeout_steps=25, msg_sizes=(300.0, 1200.0))])
def test_gpu_graph_bitwise_with_live_parameter_writes(extra):
    dev = torch.device("cuda")
    cfg = NRConfig(scheduler="learned", **extra)
    G, H = cfg.learned_groups, cfg.learned_hidden
    ref = make_engine("L2", E, R, dev, cfg, backend="reference", seed=11)
    gr = make_engine("L2", E, R, dev, cfg, backend="graph", seed=11)
    grp = torch.arange(E) % G
    for n in (ref, gr):
        n.set_sched_groups(grp.to(dev))
    gen = torch.Generator().manual_seed(5)

    def params(t):
        if t % 7:
            return None
        return (0.3 * torch.randn(G, 2 * L_FEAT, H, generator=gen), 0.1 * torch.randn(G, H, generator=gen),
                torch.randn(G, H, generator=gen), 0.2 * torch.randn(G, generator=gen))
    logw = lambda t: torch.cos(torch.arange(E * R, dtype=torch.float32).view(E, R) * 0.7 + t)
    oa, ob = _drive([ref, gr], 12, logw=logw, params=params, dev=dev)
    n_graphs = len(gr._graphs)
    oa2, ob2 = _drive([ref, gr], 40, seed=1, logw=logw, params=params, dev=dev)
    assert len(gr._graphs) == n_graphs, "parameter writes must not capture new graphs"
    for x, y in zip(oa + oa2, ob + ob2):
        _eq_out(x, y)
    sa, sb = state_dict(ref), state_dict(gr)
    for k in sa:
        assert _same(sa[k], sb[k]), k
    assert ref.net.ul.lw.abs().sum() > 0


# ---------------- learned_pregrant: proactive UL grants per robot (docs/learned-scheduler.md) ----------------
# P1 grants on with a zero floor: bitwise the PF engine.  P2 a grant cuts the delay of a sporadic robot.
# P3 an unused grant with hold 0 changes nothing; with hold 1 it takes RBGs from the others.  P4 config / API refusals.
NS = dict(n_harq=1, eff_sinr="mean_db", n_prb=50, rbg_size=10, sr_period_slots=5, sr_grant_delay_slots=10,
          ul_harq_rtt_slots=20, control_step_ms=20.0, timeout_steps=25, msg_sizes=(300.0, 1200.0, 1200.0))


@pytest.mark.parametrize("extra", [dict(), NS, dict(learned_timescale="slot")])
def test_p1_zero_pregrant_is_bitwise_pf(extra):
    pf_extra = {k: v for k, v in extra.items() if not k.startswith("learned_")}
    pf = make_engine("L2", E, R, "cpu", NRConfig(**pf_extra), seed=3)
    ln = make_engine("L2", E, R, "cpu", NRConfig(scheduler="learned", learned_pregrant=True, **extra), seed=3)
    ln.set_sched_pregrant(torch.zeros(E, R))
    oa, ob = _drive([pf, ln], 40)
    for a, b in zip(oa, ob):
        _eq_out(a, b)
    for n in UL_STATE:
        assert torch.equal(getattr(pf.net.ul, n), getattr(ln.net.ul, n)), n


def _sporadic(pg, hold=1, steps=60, idle_robot=False, big=False):
    ns = dict(NS, msg_sizes=(300.0, 20000.0, 1200.0)) if big else NS        # big: robots 1.. saturate the carrier
    cfg = NRConfig(scheduler="learned", learned_pregrant=True, learned_pregrant_hold_rbg=hold, **ns)
    net = make_engine("L2", E, R, "cpu", cfg, seed=2)
    x = torch.zeros(E, R)
    x[:, 0] = pg
    net.set_sched_pregrant(x)
    dly, n, others = 0.0, 0, 0.0
    for t in range(steps):
        send = torch.full((E, R), 2, dtype=torch.long)                    # robots 1.. backlogged with 1200 B frames
        send[:, 0] = 0 if idle_robot else (1 if t % 5 == 0 else 0)      # robot 0: a 300 B frame every 100 ms
        net.submit(None, Requests(send))
        o = net.step(None, torch.full((E, R), 15.0))
        d0 = o["delivered"][:, 0]
        dly += float(torch.where(d0, o["delay"][:, 0], torch.zeros_like(o["delay"][:, 0])).sum())
        n += int(d0.sum())
        others += float(o["delivered"][:, 1:].sum())
    return dly / max(1, n), others


def test_p2_pregrant_cuts_sporadic_delay():
    d_off, _ = _sporadic(0)
    d_on, _ = _sporadic(300)
    assert d_on < d_off


def test_p3_unused_grant_cost():
    _, base = _sporadic(0, idle_robot=True, big=True)
    _, free = _sporadic(300, hold=0, idle_robot=True, big=True)
    _, held = _sporadic(300, hold=1, idle_robot=True, big=True)
    assert free == base and held < base


def test_p5_configured_grant_period():
    """A periodic grant (every 4th TDD period) sits between no grant and an always-on grant for a sporadic robot."""
    def run(per, pg):
        cfg = NRConfig(scheduler="learned", learned_pregrant=True, learned_pregrant_period_slots=per, **NS)
        net = make_engine("L2", E, R, "cpu", cfg, seed=2)
        x = torch.zeros(E, R)
        x[:, 0] = pg
        net.set_sched_pregrant(x)
        dly, n = 0.0, 0
        for t in range(60):
            send = torch.full((E, R), 2, dtype=torch.long)
            send[:, 0] = 1 if t % 5 == 0 else 0
            net.submit(None, Requests(send))
            o = net.step(None, torch.full((E, R), 15.0))
            d0 = o["delivered"][:, 0]
            dly += float(torch.where(d0, o["delay"][:, 0], torch.zeros_like(o["delay"][:, 0])).sum())
            n += int(d0.sum())
        return dly / max(1, n)
    off, cg, aon = run(0, 0), run(20, 300), run(0, 300)
    assert aon <= cg <= off and aon < off
    with pytest.raises(AssertionError):
        NRConfig(scheduler="learned", learned_pregrant=True, learned_pregrant_period_slots=7)


def test_p4_pregrant_config():
    with pytest.raises(ValueError):
        NRConfig(scheduler="learned", learned_pregrant=True, ul_grant_model="bsr")
    with pytest.raises(ValueError):
        make_engine("L2", E, R, "cpu", NRConfig(scheduler="learned"), seed=0).set_sched_pregrant(0)
    assert "learned_pregrant" in NRConfig(learned_pregrant=True).unused_fields("L2")
    assert "proactive" in NRConfig(scheduler="learned", learned_pregrant=True).describe()


@pytest.mark.gpu
def test_gpu_graph_bitwise_with_live_pregrant_writes():
    dev = torch.device("cuda")
    cfg = NRConfig(scheduler="learned", learned_pregrant=True, **NS)
    ref = make_engine("L2", E, R, dev, cfg, backend="reference", seed=11)
    gr = make_engine("L2", E, R, dev, cfg, backend="graph", seed=11)
    logw = lambda t: torch.cos(torch.arange(E * R, dtype=torch.float32).view(E, R) * 0.7 + t)

    def drive(steps, seed):
        g = torch.Generator().manual_seed(seed)
        outs = [[], []]
        for t in range(steps):
            send = (torch.rand(E, R, generator=g) < 0.4).long() * torch.randint(1, 4, (E, R), generator=g)
            snr = 5 + 25 * torch.rand(E, R, generator=g)
            pg = (torch.rand(E, R, generator=g) < 0.3).float() * 1200
            for i, n in enumerate((ref, gr)):
                n.set_sched_logw(logw(t).to(dev))
                n.set_sched_pregrant(pg.to(dev))
                n.submit(None, Requests(send.to(dev)))
                outs[i].append(n.step(None, snr.to(dev)))
        return outs
    oa, ob = drive(12, 0)
    n_graphs = len(gr._graphs)
    oa2, ob2 = drive(40, 1)
    assert len(gr._graphs) == n_graphs, "pregrant writes must not capture new graphs"
    for x, y in zip(oa + oa2, ob + ob2):
        _eq_out(x, y)
    sa, sb = state_dict(ref), state_dict(gr)
    for k in sa:
        assert _same(sa[k], sb[k]), k
