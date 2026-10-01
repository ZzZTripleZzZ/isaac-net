"""AdaptiveEngine (core/adaptive.py): per-env routing between a cheap and an expensive level.

Limits: up_threshold = 0 is the expensive level bitwise and up_threshold = inf the cheap level bitwise, in both
layouts, through partial resets, with SNR or poses. The subbatch layout equals the mask layout bitwise while envs
switch (budget >= E). Handoffs move queues exactly, set the documented MAC entry state, conserve frames and keep
exact timeouts; delay levels take the conditional residual delay. The budget caps the active rows and serves the
highest indicator first. Static mode, set_fraction nesting and the curriculum callback. Partial resets leave other
envs bitwise unaffected. GPU (`gpu`): graph backends equal the reference adaptive engine bitwise, subbatch == mask on
graph, no host sync in submit / step, triton at the limits equals plain triton.
"""
import math

import pytest
import torch

from isaac_net.core import EdgeConfig, EdgeLoop, NRConfig, make_engine, multicell
from isaac_net.core.adaptive import FIFO, AdaptiveEngine, FidelityConfig, FidelityCurriculum, make_adaptive
from isaac_net.core.proto import netsim as ns
from isaac_net.core.proto.netsim import Requests

CFG = NRConfig(seed=11)
SEED = 11


def _params(level):
    """Monotone synthetic L05 / L05Q quantile tables that differ per bin (None for the other levels)."""
    if level not in ("L05", "L05Q"):
        return None
    g = torch.Generator().manual_seed(0)
    bins = [len(ns.NACT_EDGES) + 1, len(ns.SNR_EDGES) + 1] + ([len(ns.OWNQ_EDGES) + 1] if level == "L05Q" else [])
    bins.append(2)
    scale = 0.05 + 3.0 * torch.rand(*bins, 1, generator=g)
    return {"q": scale * torch.linspace(0.0, 1.0, 101) ** 2 * 4 + 0.02, "pdrop": 0.1 * torch.rand(*bins, generator=g)}


def _inputs(E, R, steps, seed=1, loads=(0.05, 0.9), dev="cpu", poses=False):
    """Per-env send probability in loads (uniform), classes 1/2, SNR random walk (or random-walk poses)."""
    g = torch.Generator().manual_seed(seed)
    p = loads[0] + (loads[1] - loads[0]) * torch.rand(E, 1, generator=g)
    snr = 5.0 + 20.0 * torch.rand(E, R, generator=g)
    pos = 20.0 + 100.0 * torch.rand(E, R, 2, generator=g)
    out = []
    for _ in range(steps):
        send = (torch.rand(E, R, generator=g) < p).long() * (1 + (torch.rand(E, R, generator=g) < 0.4).long())
        det = torch.rand(E, R, generator=g) < 0.2
        hid = torch.randint(0, 3, (E,), generator=g)
        snr = (snr + torch.randn(E, R, generator=g)).clamp(-5.0, 35.0)
        pos = (pos + torch.randn(E, R, 2, generator=g)).clamp(1.0, 150.0)
        x = pos.clone() if poses else snr.clone()
        out.append((send.to(dev), det.to(dev), hid.to(dev), x.to(dev)))
    return out


def _run(net, ins, resets=None, snr_feature=False):
    resets = resets or {}
    outs = []
    for k, (send, det, hid, x) in enumerate(ins):
        if k in resets:
            net.reset(resets[k])
        acc = net.submit(None, Requests(send, det, hid), x if snr_feature and x.dim() == 2 else None)
        o = net.step(None, x)
        o = {kk: v.detach().cpu().clone() for kk, v in o.items()}
        o["accepted"] = acc.cpu().clone()
        outs.append(o)
    return outs


def _same(a, b):
    if a.dtype.is_floating_point:
        return torch.equal(torch.isnan(a), torch.isnan(b)) and torch.equal(a.nan_to_num(0.0), b.nan_to_num(0.0))
    return torch.equal(a, b)


def _assert_equal_runs(A, B, keys=None, envs=None):
    for k, (a, b) in enumerate(zip(A, B)):
        for key in keys or [kk for kk in b if kk in a and not kk.startswith("fidelity")]:
            x, y = a[key], b[key]
            if envs is not None:
                x, y = x[envs], y[envs]
            assert _same(x, y), f"step {k} key {key}"


def _adaptive(E, R, fid, dev="cpu", cfg=CFG):
    return make_adaptive(E, R, dev, cfg, fid, seed=SEED)


def _fid(cheap="L1", expensive="L2-legacy", **kw):
    return FidelityConfig(cheap=cheap, expensive=expensive, cheap_params=_params(cheap),
                          expensive_params=_params(expensive), **kw)


PAIRS = [("L1", "L2-legacy"), ("L0", "L2-legacy"), ("L05Q", "L2-legacy"), ("L0DR", "L1")]
LAYOUTS = [dict(layout="mask"), dict(layout="subbatch", active_budget=8)]


# ------------------------------------------------------------------ the two limits
@pytest.mark.parametrize("cheap,expensive", PAIRS)
@pytest.mark.parametrize("layout", LAYOUTS, ids=["mask", "subbatch"])
@pytest.mark.parametrize("poses", [False, True], ids=["snr", "poses"])
def test_threshold_zero_is_the_expensive_level(cheap, expensive, layout, poses):
    E, R = 8, 5
    ins = _inputs(E, R, 30, poses=poses)
    resets = {9: [1, 6], 20: torch.tensor([0, 1, 2]), 25: None}
    ref = _run(make_engine(expensive, E, R, "cpu", CFG, params=_params(expensive), seed=SEED), ins, resets)
    net = _adaptive(E, R, _fid(cheap, expensive, mode="load", up_threshold=0.0, **layout))
    ada = _run(net, ins, resets)
    _assert_equal_runs(ada, ref)
    assert all(bool(o["fidelity"].all()) for o in ada)
    assert net.fidelity_stats()["up"] == 0 and net.fidelity_stats()["down"] == 0


@pytest.mark.parametrize("cheap,expensive", PAIRS)
@pytest.mark.parametrize("layout", LAYOUTS, ids=["mask", "subbatch"])
@pytest.mark.parametrize("poses", [False, True], ids=["snr", "poses"])
def test_threshold_inf_is_the_cheap_level(cheap, expensive, layout, poses):
    E, R = 8, 5
    ins = _inputs(E, R, 30, poses=poses)
    resets = {9: [1, 6], 20: torch.tensor([0, 1, 2])}
    ref = _run(make_engine(cheap, E, R, "cpu", CFG, params=_params(cheap), seed=SEED), ins, resets)
    ada = _run(_adaptive(E, R, _fid(cheap, expensive, mode="load", up_threshold=math.inf, **layout)), ins, resets)
    _assert_equal_runs(ada, ref)
    assert not any(bool(o["fidelity"].any()) for o in ada)


# ------------------------------------------------------------------ switching
SWITCHY = dict(mode="load", indicator="backlog", up_threshold=2500.0, down_threshold=800.0, min_dwell_steps=2)
# L0 (median delay 0.05 steps) holds almost no backlog: switch on the offered load instead
SWITCHY_OFFERED = dict(mode="load", indicator="offered", up_threshold=7000.0, down_threshold=4000.0, min_dwell_steps=2)


def _switchy(cheap):
    return SWITCHY_OFFERED if cheap == "L0" else SWITCHY


def _switch_stats(outs):
    f = torch.stack([o["fidelity"] for o in outs])
    return int((f[1:] > f[:-1]).sum()), int((f[1:] < f[:-1]).sum())


@pytest.mark.parametrize("cheap,expensive", PAIRS)
def test_subbatch_equals_mask_while_switching(cheap, expensive):
    E, R = 8, 5
    ins = _inputs(E, R, 60, loads=(0.02, 0.8))
    resets = {17: [3], 33: torch.tensor([0, 5, 7])}
    sw = _switchy(cheap)
    mask = _run(_adaptive(E, R, _fid(cheap, expensive, layout="mask", **sw)), ins, resets)
    sub = _run(_adaptive(E, R, _fid(cheap, expensive, layout="subbatch", active_budget=E, **sw)), ins, resets)
    up, down = _switch_stats(mask)
    assert up > 0 and down > 0, (up, down)
    _assert_equal_runs(sub, mask, keys=[k for k in mask[0]])


def _loaded_envs_inputs(E, R, envs, steps=50):
    g = torch.Generator().manual_seed(4)
    p = torch.zeros(E, 1)
    p[list(envs)] = torch.tensor([0.9, 0.7, 0.8])[: len(envs), None]
    return [((torch.rand(E, R, generator=g) < p).long() * 2, det, hid, x) for (_, det, hid, x) in _inputs(E, R, steps)]


def test_subbatch_with_a_small_budget_matches_mask_while_under_budget():
    """Budget 3 of 8, load on envs 5-7 only (slots 2, 0, 1). On the CPU the frozen engine's transcendental
    functions round differently in the vectorized body and the scalar tail of a tensor, so a row's position and the
    tensor size change the last bit: the run agrees with the mask layout up to float rounding (discrete outputs
    equal here). The GPU test below checks bitwise equality for the same case."""
    E, R = 8, 5
    ins = _loaded_envs_inputs(E, R, (5, 6, 7))
    mask = _run(_adaptive(E, R, _fid(layout="mask", **SWITCHY)), ins)
    net = _adaptive(E, R, _fid(layout="subbatch", active_budget=3, **SWITCHY))
    sub = _run(net, ins)
    assert net.fidelity_stats()["denied"] == 0 and net.fidelity_stats()["up"] > 0
    for a, b in zip(sub, mask):
        for key in b:
            x, y = a[key], b[key]
            if x.dtype.is_floating_point:
                assert torch.allclose(x.nan_to_num(0), y.nan_to_num(0), rtol=1e-4, atol=1e-3), key
            else:
                assert torch.equal(x, y), key


def test_allocator_prefers_home_slots_and_keeps_the_priority_order():
    E, R, M = 10, 3, 4
    net = _adaptive(E, R, _fid(layout="subbatch", active_budget=M, mode="load", up_threshold=1.0))
    req = torch.tensor([1, 0, 1, 0, 0, 1, 1, 0, 1, 1], dtype=torch.bool)
    prio = torch.tensor([5.0, 0, 9, 0, 0, 1, 7, 0, 8, 2])
    net.slot_env.fill_(-1)
    net.slot_env[3] = 4                                      # slot 3 busy
    grant, slot = net._allocate(req, prio)
    assert grant.tolist() == [0, 0, 1, 0, 0, 0, 1, 0, 1, 0]      # the 3 free slots go to prio 9, 8, 7
    # env 2 -> home 2; env 8 -> home 0 (8 mod 4); env 6 (home 2, taken by the better-ranked env 2) -> free slot 1
    assert int(slot[2]) == 2 and int(slot[8]) == 0 and int(slot[6]) == 1


def test_budget_caps_active_rows_and_serves_the_highest_indicator():
    E, R, M = 10, 4, 3
    ins = _inputs(E, R, 40, loads=(0.3, 0.95))
    net = _adaptive(E, R, _fid(layout="subbatch", active_budget=M, mode="load", indicator="backlog",
                               up_threshold=1000.0, down_threshold=500.0, min_dwell_steps=0))
    for send, det, hid, x in ins:
        net.submit(None, Requests(send, det, hid))
        before = net.active.clone()
        o = net.step(None, x)
        ind = o["fidelity_indicator"]
        assert int(net.active.sum()) <= M
        assert int((net.slot_env >= 0).sum()) == int(net.active.sum())
        # table consistency: env_slot and slot_env are inverse maps on the active envs
        act = net.active.nonzero().squeeze(-1)
        assert torch.equal(net.slot_env[net.env_slot[act]], act)
        newly = net.active & ~before
        waiting = ~net.active & (ind >= 1000.0)
        if newly.any() and waiting.any():
            assert float(ind[newly].min()) >= float(ind[waiting].max())
    assert net.fidelity_stats()["denied"] > 0


# ------------------------------------------------------------------ handoff rules
def _state(level_obj, rows):
    return {n: level_obj.get(n)[rows].clone() for n in FIFO}


def test_handoff_moves_queues_exactly_and_sets_the_mac_entry_state():
    E, R = 6, 5
    ins = _inputs(E, R, 12, loads=(0.6, 0.9))
    net = _adaptive(E, R, _fid(mode="cheap", layout="mask"))
    _run(net, ins)
    before = _state(net.lc, slice(None))
    assert int((before["cap"] >= 0).sum()) > 0
    served = net.served.clone()
    net.set_fraction(1.0)                                   # every env moves up now
    after = _state(net.lx, slice(None))
    for n in FIFO:
        assert torch.equal(before[n], after[n]), n
    assert int((net.cheap.cap >= 0).sum()) == 0                              # the cheap rows are empty
    ex = net.exp
    assert torch.equal(ex.bsr, before["rem"].sum(-1))
    assert torch.equal(ex.sr_t, torch.full_like(ex.sr_t, -1))
    assert torch.equal(ex.avg, (served / ex.K).clamp(min=ns.PF_AVG_MIN))
    assert int(ex.wait.max()) == 0 and float(ex.hcnt.max()) == 0.0
    assert torch.equal(ex.olla, torch.full_like(ex.olla, -3.0))            # no robot was on L2 yet: the prior
    assert torch.equal(ex.clock, net.cheap.clock)
    assert abs(float(ex.h.pow(2).sum(-1).mean()) - 1.0) < 0.35             # CN(0,1): E|h|^2 = 1
    # and back: the expensive rows (partly served, HARQ in flight) move to the cheap level exactly
    _run(net, _inputs(E, R, 7, seed=3, loads=(0.6, 0.9)))
    before = _state(net.lx, slice(None))
    olla = net.exp.olla.clone()
    assert float((olla + 3.0).abs().max()) > 0                                # OLLA moved while on L2
    net.set_fraction(0.0)
    after = _state(net.lc, slice(None))
    for n in FIFO:
        assert torch.equal(before[n], after[n]), n
    assert int((net.exp.cap >= 0).sum()) == 0
    avg = net.exp.avg.clone()
    net.set_fraction(1.0)                                   # back up: each robot gets its own OLLA from its last exit
    assert torch.equal(net.exp.olla, olla)
    assert torch.allclose(net.exp.avg, avg, rtol=1e-6)     # and its PF average (continued estimate)
    net.set_fraction(0.0)
    net.reset([0])                                          # a reset (on L1) forgets it: back to the prior
    net.olla_mem[1, 2] = float("nan")                       # a robot without memory: its env's mean
    net.set_fraction(1.0)
    assert torch.equal(net.exp.olla[2:], olla[2:]) and torch.equal(net.exp.olla[1, [0, 1, 3, 4]], olla[1, [0, 1, 3, 4]])
    assert torch.allclose(net.exp.olla[1, 2], olla[1, [0, 1, 3, 4]].mean())
    assert torch.equal(net.exp.olla[0], torch.full((R,), -3.0))


@pytest.mark.parametrize("cheap,expensive", PAIRS)
@pytest.mark.parametrize("layout", LAYOUTS, ids=["mask", "subbatch"])
def test_conservation_and_exact_timeouts_across_handoffs(cheap, expensive, layout):
    E, R = 8, 5
    ins = _inputs(E, R, 70, loads=(0.02, 0.95))
    net = _adaptive(E, R, _fid(cheap, expensive, **_switchy(cheap), **layout))
    acc = torch.zeros(E, R, dtype=torch.long)
    gone = torch.zeros(E, R, dtype=torch.long)
    for send, det, hid, x in ins:
        acc += net.submit(None, Requests(send, det, hid)).long()
        o = net.step(None, x)
        assert not bool((o["delivered"] & o["timed_out"]).any())
        gone += (o["delivered"] | o["timed_out"]).sum(-1)
        t = o["t"][:, None, None]
        cap = o["cap"]
        age = t + 1 - cap
        assert not bool((o["timed_out"] & (age != net.cheap.TIMEOUT)).any())       # exactly at the deadline
        assert not bool((o["delivered"] & (o["delay"] < 0)).any())
        still = (cap >= 0) & ~o["delivered"] & ~o["timed_out"]
        assert not bool((still & (age >= net.cheap.TIMEOUT)).any())
    assert torch.equal(acc, gone + net.queued())
    st = net.fidelity_stats()
    assert st["up"] > 0 and st["down"] > 0


@pytest.mark.parametrize("cheap", ["L0", "L05Q"])
def test_delay_level_entry_draws_the_conditional_residual_delay(cheap):
    """Frames that waited w enter a delay level with dlv >= now and a delay distributed as delay | delay > w."""
    E, R = 64, 8
    fid = _fid(cheap, "L2-legacy", mode="expensive", layout="mask")
    cfg = NRConfig(seed=5, l0_delay_median_steps=3.0, l0_delay_log_sigma=0.8, l0_loss=0.1)
    net = _adaptive(E, R, fid, cfg=cfg)
    snr = torch.full((E, R), -10.0)                 # poor link: large frames wait on the expensive level
    for k in range(4):
        net.submit(None, Requests(torch.full((E, R), 2, dtype=torch.long) * (k == 0)))
        net.step(None, snr)
    src = _state(net.lx, slice(None))
    valid = src["cap"] >= 0
    assert int(valid.sum()) > 200
    net.set_fraction(0.0)
    now = net.cheap.clock[:, None, None].float()
    dlv = net.cheap.dlv
    fin = torch.isfinite(dlv) & valid
    assert bool((dlv[fin] >= now.expand_as(dlv)[fin]).all())
    if cheap == "L0":
        w = (now - src["cap"].float())[valid]
        assert float(w.min()) == float(w.max()) == 4.0
        mu, sig, p = math.log(3.0), 0.8, 0.1
        Fw = 0.5 * (1 + math.erf((math.log(4.0) - mu) / (sig * math.sqrt(2))))
        d = (dlv - src["cap"].float())[fin]
        med = math.exp(mu + sig * math.sqrt(2) * torch.erfinv(torch.tensor(2 * (Fw + 0.5 * (1 - Fw)) - 1)).item())
        assert abs(float(d.median()) - med) / med < 0.12
        lost = float((~torch.isfinite(dlv) & valid).sum()) / float(valid.sum())
        pc = p / (p + (1 - p) * (1 - Fw))
        assert abs(lost - pc) < 0.06


# ------------------------------------------------------------------ static mode and curriculum
def test_static_mode_fixed_subset_and_nested_fractions():
    E, R = 20, 4
    net = _adaptive(E, R, _fid(mode="static", fraction=0.25))
    assert net.subbatch and net.M == 5 and int(net.active.sum()) == 5
    first = net.active.clone()
    outs = _run(net, _inputs(E, R, 10, loads=(0.3, 0.9)), resets={4: [0, 1, 2, 3, 4, 5, 6, 7]})
    for o in outs:
        assert torch.equal(o["fidelity"].bool(), first)
    mask = _adaptive(E, R, _fid(mode="static", fraction=0.25, layout="mask"))
    assert torch.equal(mask.active, first)
    m = _adaptive(E, R, _fid(mode="static", fraction=0.2, layout="mask"))
    sets = []
    for f in (0.2, 0.5, 0.8):
        m.set_fraction(f)
        sets.append(m.active.clone())
        assert int(m.active.sum()) == round(f * E)
    assert bool((sets[1] | ~sets[0]).all()) and bool((sets[2] | ~sets[1]).all())


def test_curriculum_callback():
    E, R = 6, 4
    net = _adaptive(E, R, _fid(mode="cheap", layout="mask"))
    cur = FidelityCurriculum.step_at(net, 3)
    ins = _inputs(E, R, 6, loads=(0.3, 0.8))
    seen = []
    for it in range(6):
        cur(it)
        _run(net, ins[it:it + 1])
        seen.append(int(net.active.sum()))
    assert seen == [0, 0, 0, 6, 6, 6]
    ramp = FidelityCurriculum(net, lambda it: min(1.0, it / 4))
    assert [ramp.fraction(i) for i in range(6)] == [0.0, 0.25, 0.5, 0.75, 1.0, 1.0]
    ramp(1)
    assert int(net.active.sum()) == round(0.25 * E)


# ------------------------------------------------------------------ resets, API, config
@pytest.mark.parametrize("layout", LAYOUTS, ids=["mask", "subbatch"])
def test_partial_reset_leaves_other_envs_bitwise_unaffected(layout):
    E, R = 8, 5
    ins = _inputs(E, R, 40, loads=(0.02, 0.8))
    kw = dict(**SWITCHY, **layout)
    a = _run(_adaptive(E, R, _fid(**kw)), ins)
    b = _run(_adaptive(E, R, _fid(**kw)), ins, resets={11: [2, 5], 27: torch.tensor([0])})
    others = torch.tensor([1, 3, 4, 6, 7])
    _assert_equal_runs(b, a, keys=[k for k in a[0] if k != "det_env"], envs=others)


def test_api_legacy_calls_config_and_rejections():
    E, R = 4, 3
    fid = _fid(mode="load", up_threshold=0.0, layout="mask")
    net = make_adaptive(E, R, "cpu", NRConfig(seed=SEED, fidelity=fid))
    assert isinstance(net, AdaptiveEngine) and net.fid is fid
    ref = make_engine("L2-legacy", E, R, "cpu", CFG, seed=SEED)
    for send, det, hid, x in _inputs(E, R, 8):
        for n in (net, ref):
            n.add_frames(None, send, det, hid, x)
        a, b = net.step(None, x, hid), ref.step(None, x, hid)
        assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    assert "fidelity" in NRConfig(fidelity=fid).unused_fields("L2-legacy")
    edge = make_adaptive(E, R, "cpu", NRConfig(seed=SEED, edge=EdgeConfig()), fid)
    assert isinstance(edge, EdgeLoop) and isinstance(edge.engine, AdaptiveEngine)
    o = edge.step(None, torch.full((E, R), 10.0))
    assert "act_age" in o and "fidelity" in o
    with pytest.raises(ValueError, match="rng"):
        make_adaptive(E, R, "cpu", NRConfig(rng="global"), fid)
    with pytest.raises(ValueError, match="single legacy cell"):
        make_adaptive(E, R, "cpu", multicell(3), fid)
    with pytest.raises(ValueError):
        FidelityConfig(cheap="L2-legacy")
    with pytest.raises(ValueError):
        FidelityConfig(cheap="L1", expensive="L1")


# ------------------------------------------------------------------ the NR engine as the expensive level
def _seeded_run(net_fn, ins, resets=None):
    torch.manual_seed(123)                     # the NR engine's stepping draws use the global RNG
    return _run(net_fn(), ins, resets)


@pytest.mark.parametrize("cheap", ["L1", "L05Q"])
@pytest.mark.parametrize("layout", LAYOUTS, ids=["mask", "subbatch"])
def test_nr_expensive_limits(cheap, layout):
    E, R = 8, 5
    ins = _inputs(E, R, 20)
    resets = {7: [1, 6], 14: torch.tensor([0, 1, 2])}
    ref = _seeded_run(lambda: make_engine("L2", E, R, "cpu", CFG, seed=SEED), ins, resets)
    ada = _seeded_run(lambda: _adaptive(E, R, _fid(cheap, "L2", mode="load", up_threshold=0.0, **layout)), ins,
                      resets)
    _assert_equal_runs(ada, ref)
    ref = _run(make_engine(cheap, E, R, "cpu", CFG, params=_params(cheap), seed=SEED), ins, resets)
    ada = _seeded_run(lambda: _adaptive(E, R, _fid(cheap, "L2", mode="load", up_threshold=math.inf, **layout)),
                      ins, resets)
    _assert_equal_runs(ada, ref)


def test_nr_handoff_moves_the_queue_and_conserves_frames():
    E, R = 6, 5
    net = _adaptive(E, R, _fid("L1", "L2", mode="cheap", layout="mask"))
    _run(net, _inputs(E, R, 10, loads=(0.6, 0.9)))
    before = {n: v.clone() for n, v in net.lc.fifo().items()}
    assert int((before["cap"] >= 0).sum()) > 0
    net.set_fraction(1.0)
    after = net.lx.fifo()
    for n in FIFO:
        if n == "rem":
            assert float((after[n] - before[n]).abs().max()) <= 1.0          # air-byte rounding, under 1 byte
        else:
            assert torch.equal(after[n], before[n]), n
    ul = net.exp.net.ul
    assert torch.equal(ul.bsr, ul.q.enq) and int((ul.h_state != 0).sum()) == 0 and float(ul.olla.abs().max()) == 0
    assert torch.equal(net.exp.clock, net.cheap.clock)
    ins = _inputs(E, R, 60, seed=9, loads=(0.02, 0.9))
    net = _adaptive(E, R, _fid("L1", "L2", layout="subbatch", active_budget=4, **SWITCHY))
    acc = torch.zeros(E, R, dtype=torch.long)
    gone = torch.zeros(E, R, dtype=torch.long)
    for send, det, hid, x in ins:
        acc += net.submit(None, Requests(send, det, hid)).long()
        o = net.step(None, x)
        gone += (o["delivered"] | o["timed_out"] | o["dropped"]).sum(-1)
        age = o["t"][:, None, None] + 1 - o["cap"]
        assert not bool((o["timed_out"] & (age != CFG.timeout_steps)).any())
    assert torch.equal(acc, gone + net.queued())
    st = net.fidelity_stats()
    assert st["up"] > 0 and st["down"] > 0


# ------------------------------------------------------------------ GPU
GPU_PAIRS = [("L1", "L2-legacy"), ("L05Q", "L2-legacy")]


@pytest.mark.gpu
@pytest.mark.parametrize("cheap,expensive", GPU_PAIRS)
@pytest.mark.parametrize("layout", LAYOUTS, ids=["mask", "subbatch"])
def test_gpu_graph_backends_equal_the_reference_adaptive_engine(cheap, expensive, layout):
    E, R = 8, 5
    dev = torch.device("cuda")
    ins = _inputs(E, R, 50, loads=(0.02, 0.8), dev=dev)
    resets = {13: [2], 31: torch.tensor([0, 4, 6], device=dev)}
    kw = dict(**SWITCHY, **layout)
    ref = _run(_adaptive(E, R, _fid(cheap, expensive, **kw), dev=dev), ins, resets)
    fast = _run(_adaptive(E, R, _fid(cheap, expensive, cheap_backend="graph", expensive_backend="graph", **kw),
                          dev=dev), ins, resets)
    up, down = _switch_stats(ref)
    assert up > 0 and down > 0
    _assert_equal_runs(fast, ref, keys=[k for k in ref[0]])


@pytest.mark.gpu
@pytest.mark.parametrize("cheap,cb,xb", [("L1", "graph", "graph"), ("L1", "triton", "triton"),
                                         ("L0DR", "graph", "triton"), ("L05Q", "graph", "graph")])
@pytest.mark.parametrize("layout", LAYOUTS, ids=["mask", "subbatch"])
def test_gpu_graph_mode_equals_eager_mode(cheap, cb, xb, layout):
    """FidelityConfig.graph: submit, step and reset captured in one CUDA graph each (after two eager calls; reset
    as a masked reset of both levels) equal the same engine run eagerly (the levels' own resets), bitwise, through
    switches, partial resets every few steps, poses input (radio reset) and mode changes (re-capture)."""
    E, R = 16, 5
    dev = torch.device("cuda")
    ins = _inputs(E, R, 45, loads=(0.02, 0.8), dev=dev, poses=True)
    resets = {k: torch.tensor([(3 * k) % E, (5 * k + 1) % E], device=dev) for k in range(2, 45, 3)}
    resets[20] = None
    kw = dict(cheap_backend=cb, expensive_backend=xb, **_switchy(cheap), **layout)
    kw["decision_period"] = 2 if cheap == "L1" else 1          # two captured steps, with and without switching

    def run(graph):
        net = _adaptive(E, R, _fid(cheap, "L2-legacy", graph=graph, **kw), dev=dev)
        assert net.graph == graph
        a = _run(net, ins[:15], resets)
        net.set_fraction(0.5)                               # static mode: the step graph is captured again
        b = _run(net, ins[15:30], {k - 15: v for k, v in resets.items() if 15 <= k < 30})
        net.set_mode("load")
        c = _run(net, ins[30:], {k - 30: v for k, v in resets.items() if k >= 30})
        return a + b + c, net.fidelity_stats()

    eager, st_e = run(False)
    graph, st_g = run(True)
    assert st_e == st_g and st_e["up"] > 0 and st_e["down"] > 0
    _assert_equal_runs(graph, eager, keys=[k for k in eager[0]])


@pytest.mark.gpu
def test_gpu_subbatch_equals_mask_on_graph_backends():
    E, R = 16, 5
    dev = torch.device("cuda")
    ins = _inputs(E, R, 50, loads=(0.02, 0.8), dev=dev)
    kw = dict(cheap_backend="graph", expensive_backend="graph", **SWITCHY)
    mask = _run(_adaptive(E, R, _fid(layout="mask", **kw), dev=dev), ins, {20: [3, 9]})
    sub = _run(_adaptive(E, R, _fid(layout="subbatch", active_budget=E, **kw), dev=dev), ins, {20: [3, 9]})
    _assert_equal_runs(sub, mask, keys=[k for k in mask[0]])


@pytest.mark.gpu
def test_gpu_subbatch_away_from_home_slots_equals_mask():
    """Load on envs 5-7 with budget 3: they run in slots 2, 0, 1. On the GPU every element runs the same code, so the
    graph subbatch run equals the mask layout bitwise."""
    E, R = 8, 5
    dev = torch.device("cuda")
    ins = [tuple(x.to(dev) for x in step) for step in _loaded_envs_inputs(E, R, (5, 6, 7))]
    kw = dict(cheap_backend="graph", expensive_backend="graph", **SWITCHY)
    mask = _run(_adaptive(E, R, _fid(layout="mask", **kw), dev=dev), ins)
    net = _adaptive(E, R, _fid(layout="subbatch", active_budget=3, **kw), dev=dev)
    sub = _run(net, ins)
    assert net.fidelity_stats()["up"] > 0 and set(net.slot_env.tolist()) <= {-1, 5, 6, 7}
    _assert_equal_runs(sub, mask, keys=[k for k in mask[0]])


@pytest.mark.gpu
@pytest.mark.parametrize("layout", LAYOUTS, ids=["mask", "subbatch"])
@pytest.mark.parametrize("backend", ["graph", "triton"])
def test_gpu_submit_and_step_have_no_host_sync(layout, backend):
    E, R = 16, 5
    dev = torch.device("cuda")
    ins = _inputs(E, R, 12, loads=(0.02, 0.9), dev=dev)
    net = _adaptive(E, R, _fid(cheap_backend=backend, expensive_backend=backend, **SWITCHY, **layout), dev=dev)
    ids = torch.tensor([1, 7], device=dev)
    _run(net, ins[:3], resets={0: ids, 1: ids, 2: ids})      # warm up: graph capture, triton compile
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        for send, det, hid, x in ins[3:]:
            net.reset(ids)                                    # graph mode: the reset is sync-free too
            net.submit(None, Requests(send, det, hid))
            net.step(None, x)
    finally:
        torch.cuda.set_sync_debug_mode(0)
    assert net.fidelity_stats()["up"] >= 0


@pytest.mark.gpu
@pytest.mark.parametrize("layout", LAYOUTS, ids=["mask", "subbatch"])
def test_gpu_triton_limits_equal_plain_triton(layout):
    E, R = 8, 5
    dev = torch.device("cuda")
    ins = _inputs(E, R, 25, dev=dev)
    resets = {10: [1, 6]}
    ref = _run(make_engine("L2-legacy", E, R, dev, CFG, "triton", seed=SEED), ins, resets)
    ada = _run(_adaptive(E, R, _fid(expensive_backend="triton", cheap_backend="triton", mode="load",
                                    up_threshold=0.0, **layout), dev=dev), ins, resets)
    _assert_equal_runs(ada, ref)
    ref = _run(make_engine("L1", E, R, dev, CFG, "triton", seed=SEED), ins, resets)
    ada = _run(_adaptive(E, R, _fid(expensive_backend="triton", cheap_backend="triton", mode="load",
                                    up_threshold=math.inf, **layout), dev=dev), ins, resets)
    _assert_equal_runs(ada, ref)
