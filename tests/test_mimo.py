"""Two-layer SU-MIMO rank model of the NR engine (NRConfig.n_layers_max; docs/configurability.md "MIMO rank"), CPU.

  (a) n_layers_max = 1 is bitwise the pre-feature engine: the live NRNet against the frozen engine of tests/nr_frozen/
      with the rank fields set (they are not read), one cell and with a partial reset, and at three cells against the
      live defaults; the off engine has no rank state and no rank step keys; tbs_38214 with layers=1 (int or tensor)
      is bitwise the one-layer TBS
  (b) the TBS of a rank-2 TB is about twice the rank-1 TBS at the same MCS and allocation (engine TBs and the PHY
      table); at high SINR rank 2 raises the saturated DL throughput by more than 20 % and by less than 2x (layer
      penalty); at low SINR every TB stays rank 1 and the run is bitwise the rank-off run; UL rank 2 only with ul_mimo
  (c) rank_rule="los" with Rician K from the radio's LOS state: links with K >= rank_k_max_db (LOS, high K) get rank 1,
      NLOS links (K = 0) rank 2; without Rician fading (no K) every link counts as rich scattering
  (d) a HARQ process keeps the rank of its TB over every retransmission while the rank of new TBs changes; byte
      conservation (new bytes = decoded bytes + bytes in undecoded processes)
  (e) env 0 is independent of E, of other envs' resets and of a partial reset, one and three cells; a partial reset
      returns the reset envs' rank state to 1
  (f) unused_fields gates the rank fields by their switches; the triton backend refuses rank 2; the nr_equiv MIMO
      config runs on the reference with its rank state in nr_fast.state_dict, and the graph backend's capture path
      (CPU stand-in, nr_fast._EagerReplay) equals the reference bitwise on it
GPU: the nr_equiv MIMO config (MIMO_CFGS) is in the G1 and G7 lists of test_nr_fast.py (graph bitwise); triton
refuses it, so it is not in G2.
"""
import math

import pytest
import torch

import nr_equiv
from isaac_net.core import NRConfig, Requests, make_engine, multicell
from isaac_net.core.mac import MacLink
from isaac_net.core.nr_engine import NRNet
from isaac_net.core.nr_fast import NRGraphEngine, NRTritonEngine, state_dict
from isaac_net.core.phy import PHY, tbs_38214
from isaac_net.core.radio import RadioMC
from nr_frozen.nr_engine import NRNet as FrozenNRNet

SIZES = (4000.0, 30000.0)
FAST = dict(control_step_ms=10.0)
OFF_FIELDS = dict(n_layers_max=1, rank_rule="los", rank_sinr_min_db=-5.0, rank_k_max_db=20.0,
                  rank_layer_penalty_db=1.0, ul_mimo=True, dl_mimo=False)


# ---------------------------------------------------------------------------------------------------------- (a)
def _run_single(cls, cfg, E=3, R=4, steps=8, reset_at=4):
    torch.manual_seed(11)
    net = cls(E, R, "cpu", SIZES, cfg, generator=torch.Generator().manual_seed(3))
    g = torch.Generator().manual_seed(5)
    outs = []
    for t in range(steps):
        if t == reset_at:
            net.reset(torch.tensor([1]))
        send = torch.randint(0, 3, (E, R), generator=g)
        snr = -5 + 35 * torch.rand(E, R, generator=g)
        hid = torch.randint(0, 3, (E,), generator=g)
        net.add_frames(t, send, torch.rand(E, R, generator=g) < 0.3, hid, snr)
        if cfg.dl:
            net.add_dl_frames(t, torch.where(send > 0, net.sizes[(send - 1).clamp(min=0)], torch.zeros_like(snr)))
        o = net.step(t, snr, hid, full=True) if t % 2 == 0 else net.step_rx(t, snr - 113.0, hid, full=True)
        outs.append({k: v.clone() for k, v in o.items()})
        for lk in (net.ul, net.dl):
            if lk is not None:
                outs[-1].update({f"{lk.dir}.{n}": getattr(lk, n).clone()
                                 for n in ("olla", "avg", "csi", "sent", "h_mcs", "h_tbs")})
    return outs


def _equal(a, b, skip=()):
    for t, (x, y) in enumerate(zip(a, b)):
        assert x.keys() - set(skip) == y.keys() - set(skip)
        for k in x.keys() - set(skip):
            assert torch.equal(x[k].nan_to_num(-7.0), y[k].nan_to_num(-7.0)), (t, k)


def test_a_off_equals_frozen_engine():
    cfg = NRConfig(dl=True, rng="global", control_step_ms=20.0, **OFF_FIELDS)    # the frozen engine: global RNG
    assert cfg.mimo_dirs == ()
    _equal(_run_single(FrozenNRNet, cfg), _run_single(NRNet, cfg))
    net = NRNet(2, 2, "cpu", SIZES, cfg, seed=0)
    assert not net.ul.mimo and not net.dl.mimo
    assert not any(hasattr(lk, n) for lk in (net.ul, net.dl) for n in ("h_rank", "last_rank"))
    eng = make_engine("L2", 2, 3, "cpu", cfg.with_(rng="engine"), seed=1)
    assert not any(k.endswith(("h_rank", "last_rank")) for k in state_dict(eng))
    eng.submit(None, torch.ones(2, 3, dtype=torch.long))
    assert not any("rank" in k for k in eng.step(None, torch.full((2, 3), 20.0)))


def _drive_poses(cfg, E=3, R=4, steps=5, resets=None, seed=4, emax=6):
    """Inputs drawn for emax envs (an engine with E envs uses the first E rows), resets = {step: env ids}."""
    eng = make_engine("L2", E, R, "cpu", cfg, seed=seed)
    g = torch.Generator().manual_seed(1)
    pos = torch.rand(emax, R, 2, generator=g) * 150
    outs = []
    for k in range(steps):
        send = torch.randint(0, 3, (emax, R), generator=g)
        eng.submit(None, Requests(send[:E], None, torch.full((E,), k)))
        if cfg.dl:
            eng.add_dl_frames(None, send[:E].float() * 3000)
        pos = (pos + 3 * (2 * torch.rand(emax, R, 2, generator=g) - 1)).clamp(0, 150)
        o = eng.step(None, pos[:E])
        outs.append({n: v.clone() for n, v in o.items() if torch.is_tensor(v) and v.dim() >= 1 and v.shape[0] == E})
        if resets and k in resets:
            eng.reset(torch.tensor(resets[k]))
    return outs, eng


def test_a_three_cells_off_equals_defaults():
    base = multicell(3, dl=True, **FAST)
    a, _ = _drive_poses(base)
    b, _ = _drive_poses(base.with_(**OFF_FIELDS))
    _equal(a, b)


def test_a_tbs_layers_one_is_the_one_layer_tbs():
    n = torch.arange(0, 274, dtype=torch.float32)[:, None]
    qm, r = torch.tensor([2.0, 4.0, 6.0, 8.0]), torch.tensor([0.12, 0.37, 0.62, 0.93])
    base = tbs_38214(qm, r, n, 12)
    assert torch.equal(base, tbs_38214(qm, r, n, 12, layers=1))
    assert torch.equal(base, tbs_38214(qm, r, n, 12, layers=torch.ones(274, 1, dtype=torch.long)))
    phy = PHY("dl", 1, "cpu")
    nprb = torch.tensor([[10.0, 51.0]])
    assert torch.equal(phy.tbs_all(nprb, 12, 12), phy.tbs_all(nprb, 12, 12, 0, torch.ones(1, 2, dtype=torch.long)))


# ---------------------------------------------------------------------------------------------------------- (b)
def test_b_rank2_tbs_about_doubles_at_the_same_mcs():
    phy = PHY("dl", 1, "cpu")
    for nprb in (10.0, 51.0, 106.0):
        n = torch.tensor([nprb])
        t1 = phy.tbs_all(n, 12, 12).double()
        t2 = phy.tbs_all(n, 12, 12, 0, torch.full((1,), 2)).double()
        ratio = t2 / t1
        assert float(ratio.min()) > 1.9 and float(ratio.max()) < 2.15, (nprb, ratio)
    # lena TBS mode: the rank multiplies the payload
    pl = PHY("dl", 1, "cpu", tbs_mode="lena")
    r = pl.tbs_all(torch.tensor([51.0]), 12, 12, 0, torch.full((1,), 2)).double() / pl.tbs_all(
        torch.tensor([51.0]), 12, 12).double()
    assert float(r.min()) > 1.9 and float(r.max()) < 2.1


def _saturated(n_layers, snr, steps=12, R=2, **kw):
    """One cell, no fading, R robots with 1 MB UL and DL frames every step (saturated), input SNR snr."""
    cfg = NRConfig(dl=True, fading=False, msg_sizes=(1e6,), n_layers_max=n_layers, rank_rule="sinr",
                   control_step_ms=20.0, frame_buffer=4, **kw)
    net = NRNet(1, R, "cpu", cfg.msg_sizes, cfg, seed=0)
    calls = []
    for lk in (net.ul, net.dl):
        orig = lk.phy.select_mcs

        def rec(*a, _o=orig, _d=lk.dir, **k):
            m, tb = _o(*a, **k)
            calls.append((_d, m, tb, a[3], a[4], k.get("layers", 1)))
            return m, tb
        lk.phy.select_mcs = rec
    outs = []
    one = torch.ones(1, R, dtype=torch.long)
    s = torch.full((1, R), float(snr))
    for t in range(steps):
        net.add_frames(t, one, torch.zeros(1, R, dtype=torch.bool), torch.zeros(1, dtype=torch.long), s)
        net.add_dl_frames(t, torch.full((1, R), 1e6))
        outs.append({k: v.clone() for k, v in net.step(t, s, full=True).items()})
    return net, outs, calls


def test_b_rank2_raises_saturated_throughput_by_less_than_2x():
    n1, _, _ = _saturated(1, 30.0)
    n2, o2, calls = _saturated(2, 30.0)
    assert bool((o2[-1]["dl_rank"] == 2).all()) and bool((o2[-1]["rank"] == 1).all())     # ul_mimo off by default
    b1, b2 = n1.counters()["dl"]["bytes_ok"], n2.counters()["dl"]["bytes_ok"]
    assert 1.2 < b2 / b1 < 2.0, b2 / b1
    assert n1.counters()["ul"]["bytes_ok"] == n2.counters()["ul"]["bytes_ok"]           # UL untouched
    # every new DL TB: rank 2, and its TBS about twice the one-layer TBS of the same MCS and allocation
    phy = n2.dl.phy
    nd = 0
    for d, m, tb, n_prb, nsym, lay in calls:
        if d != "dl":
            continue
        assert torch.is_tensor(lay) and bool((lay == 2).all())
        t1 = phy.tbs_all(n_prb, nsym, n2.cfg.dmrs_re_per_prb, n2.cfg.overhead_re_per_prb).gather(-1, m[..., None])[..., 0]
        sel = t1 > 0
        ratio = tb[sel].double() / t1[sel].double()
        assert float(ratio.min()) > 1.9 and float(ratio.max()) < 2.15
        nd += int(sel.sum())
    assert nd > 0
    # below the MCS cap the rank-2 MCS is lower than the rank-1 MCS (per-layer SINR 3 dB + penalty lower), and rank 2
    # still gains (DL SINR 15 dB)
    m1, _, _ = _saturated(1, 5.0)
    m2, _, _ = _saturated(2, 5.0)
    assert int(m2.dl.h_mcs.max()) < int(m1.dl.h_mcs.max())
    assert 1.05 < m2.counters()["dl"]["bytes_ok"] / m1.counters()["dl"]["bytes_ok"] < 2.0


def test_b_low_sinr_stays_rank1_and_bitwise():
    _, o1, _ = _saturated(1, -12.0)                 # DL SINR = -12 + dl_snr_offset_db 10 = -2 dB < 10 dB
    n2, o2, _ = _saturated(2, -12.0)
    assert all(bool((o["dl_rank"] == 1).all()) for o in o2)
    assert bool((n2.dl.h_rank == 1).all())
    _equal(o1, o2, skip=("rank", "dl_rank"))
    assert n2.counters()["dl"]["tb_new"] > 0


def test_b_ul_rank2_only_with_ul_mimo():
    n0, _, _ = _saturated(2, 30.0)
    n1, o1, _ = _saturated(2, 30.0, ul_mimo=True)
    assert not n0.ul.mimo and n1.ul.mimo
    assert bool((o1[-1]["rank"] == 2).all())
    assert n1.counters()["ul"]["bytes_ok"] > 1.2 * n0.counters()["ul"]["bytes_ok"]


# ---------------------------------------------------------------------------------------------------------- (c)
@pytest.fixture
def fake_los(monkeypatch):
    """RadioMC.los_state(): LOS iff x < 75 m (deterministic in the poses), no blockage state (test_rician.py)."""
    orig = RadioMC.pathgain_db

    def pathgain_db(self, pos, *a):
        self._fake_pos = pos
        return orig(self, pos, *a)

    def los_state(self):
        return (self._fake_pos[..., 0] < 75.0)[..., None].expand(-1, -1, self.C)

    monkeypatch.setattr(RadioMC, "pathgain_db", pathgain_db)
    monkeypatch.setattr(RadioMC, "los_state", los_state, raising=False)
    monkeypatch.setattr(RadioMC, "blocked_state", lambda self: None, raising=False)


def test_c_los_rule_rank1_on_high_k_rank2_on_nlos(fake_los):
    E, R = 3, 4
    cfg = NRConfig(dl=True, n_layers_max=2, rank_rule="los", rank_k_max_db=-10.0, fading_rician=True,
                   channel="tr38901_umi", rician_k_ramp_slots=0, msg_sizes=(2e5,), **FAST)
    eng = make_engine("L2", E, R, "cpu", cfg, seed=2)
    pos = torch.zeros(E, R, 2)
    pos[..., 0] = torch.tensor([20.0, 40.0, 110.0, 140.0])        # two LOS robots, two NLOS robots per env
    pos[..., 1] = 30.0
    for k in range(4):
        eng.submit(None, Requests(torch.ones(E, R, dtype=torch.long), None, torch.full((E,), k)))
        eng.add_dl_frames(None, torch.full((E, R), 2e5))
        out = eng.step(None, pos)
    k_lin = eng.net.k_lin
    los = pos[..., 0] < 75
    assert bool((k_lin[los] >= 10 ** (-10.0 / 10)).all()) and bool((k_lin[~los] == 0).all())  # UMi K ~ N(9, 5) dB
    tb = eng.net.dl.last_rank
    assert bool((tb[los] == 1).all()) and bool((tb[~los] == 2).all())
    assert torch.equal(out["dl_rank"], tb)
    assert int(eng.net.counters()["dl"]["tb_new"]) > 0


def test_c_no_rician_counts_as_rich_scattering():
    net, o, _ = _saturated(2, 30.0)
    assert net.dl.rank_k() is None and getattr(net, "k_lin", None) is None
    cfg = NRConfig(dl=True, n_layers_max=2, rank_rule="los", msg_sizes=(1e6,), **FAST)
    net = NRNet(2, 2, "cpu", cfg.msg_sizes, cfg, seed=0)
    s = torch.full((2, 2), -12.0)
    for t in range(3):
        net.add_dl_frames(t, torch.full((2, 2), 1e6))
        o = net.step(t, s, full=True)
    assert bool((o["dl_rank"] == 2).all())               # the "los" rule ignores the SINR


# ---------------------------------------------------------------------------------------------------------- (d)
def test_d_harq_retransmissions_keep_the_rank(monkeypatch):
    gen = torch.Generator().manual_seed(9)

    def rand_rank(self, est):                       # flip the rank of new TBs at random, slot by slot
        return 1 + torch.randint(0, 2, (self.E, self.R), generator=gen)
    monkeypatch.setattr(MacLink, "_rank", rand_rank)
    cfg = NRConfig(dl=True, ul_mimo=True, n_layers_max=2, msg_sizes=(4000.0, 30000.0), harq_fail="rlc_am",
                   discard="none", timeout_steps=1000, frame_buffer=64, bler_target=0.3, **FAST)
    E, R = 3, 4
    net = NRNet(E, R, "cpu", cfg.msg_sizes, cfg, seed=4)
    stats = {"retx": 0, "new": 0}
    for lk in (net.ul, net.dl):
        orig = lk.slot

        def slot(*a, _o=orig, _lk=lk, **k):
            r0, n0, s0 = _lk.h_rank.clone(), _lk.h_ntx.clone(), _lk.h_state.clone()
            _o(*a, **k)
            retx = (_lk.h_ntx == n0 + 1) & (n0 >= 1) & (s0 == 1)          # this slot retransmitted the process
            new = (_lk.h_ntx == 1) & ((n0 != 1) | (s0 != 1) | (_lk.h_rank != r0))
            assert torch.equal(_lk.h_rank[retx], r0[retx])
            assert bool(((_lk.h_rank == r0) | new).all())                # the rank changes only with a new TB
            stats["retx"] += int(retx.sum())
            stats["new"] += int((new & (_lk.h_rank != r0)).sum())
        lk.slot = slot
    g = torch.Generator().manual_seed(1)
    z = torch.zeros(E, dtype=torch.long)
    for t in range(25):
        send = torch.randint(0, 3, (E, R), generator=g)
        snr = -2 + 30 * torch.rand(E, R, generator=g)
        net.add_frames(t, send, torch.zeros(E, R, dtype=torch.bool), z, snr)
        net.add_dl_frames(t, send.float() * 3000)
        net.step(t, snr, z, full=True)
    assert stats["retx"] > 20 and stats["new"] > 20, stats
    for lk in (net.ul, net.dl):                        # byte conservation per link
        c = lk.ctr
        own = ((lk.h_hi - lk.h_lo) * (lk.h_state == 1)).sum()
        assert float(c["bytes_new"]) == float(c["bytes_ok"]) + float(own), lk.dir
        assert int(lk.sent.sum()) == float(c["bytes_new"])


# ---------------------------------------------------------------------------------------------------------- (e)
E_CASES = {"c1": NRConfig(dl=True, n_layers_max=2, ul_mimo=True, fading_rician=True, channel="tr38901_inf_sh",
                          rank_sinr_min_db=5.0, **FAST),
           "c3": multicell(3, dl=True, n_layers_max=2, ul_mimo=True, fading_rician=True, rank_sinr_min_db=5.0,
                           **FAST)}


def _rows_equal(a, b, rows):
    for x, y in zip(a, b):
        for name in x:
            u, v = x[name][rows].nan_to_num(-7.0), y[name][rows].nan_to_num(-7.0)
            if u.is_floating_point():
                assert torch.allclose(u, v, rtol=1e-5, atol=1e-5), name
            else:
                assert torch.equal(u, v), name


@pytest.mark.parametrize("case", list(E_CASES))
def test_e_env_independence_and_partial_reset(case, fake_los):
    cfg = E_CASES[case]
    a, _ = _drive_poses(cfg, 3, steps=6, resets={3: [1]})
    b, _ = _drive_poses(cfg, 6, steps=6, resets={3: [1]})
    assert "dl_rank" in a[0] and "rank" in a[0]
    _rows_equal(a, b, slice(0, 3))
    assert any(bool((o["dl_rank"] == 2).any()) for o in a) and any(bool((o["dl_rank"] == 1).any()) for o in a)
    c, _ = _drive_poses(cfg, 4, steps=6)
    d, eng = _drive_poses(cfg, 4, steps=6, resets={5: [1]})
    _rows_equal(c, d, [0, 2, 3])
    for lk in (eng.net.ul, eng.net.dl):              # the reset env's rank state is back to rank 1
        assert bool((lk.h_rank[1] == 1).all()) and bool((lk.last_rank[1] == 1).all())
        assert bool((lk.last_rank != 1).any())


# ---------------------------------------------------------------------------------------------------------- (f)
def test_f_config_gating():
    assert NRConfig(n_layers_max=2, dl=True).unused_fields("L2") == []
    assert NRConfig(n_layers_max=2, dl=True, ul_mimo=True, rank_layer_penalty_db=1.0).unused_fields("L2") == []
    assert NRConfig(rank_rule="los", ul_mimo=True).unused_fields("L2") == ["rank_rule", "ul_mimo"]
    assert NRConfig(n_layers_max=2, dl=True, rank_rule="sinr", rank_k_max_db=1.0).unused_fields("L2") == [
        "rank_k_max_db"]
    assert NRConfig(n_layers_max=2, dl=True, rank_rule="los", rank_sinr_min_db=5.0).unused_fields("L2") == [
        "rank_sinr_min_db"]
    # no direction may use rank 2 (DL off, UL MIMO off): the rule is unread
    assert NRConfig(n_layers_max=2, rank_sinr_min_db=5.0, dl_mimo=False).unused_fields("L2") == [
        "dl_mimo", "rank_sinr_min_db"]
    assert NRConfig(n_layers_max=2).unused_fields("L2-legacy") == ["n_layers_max"]
    assert NRConfig(n_layers_max=2, dl=True).mimo_dirs == ("dl",)
    assert NRConfig(n_layers_max=2, dl=True, ul_mimo=True).mimo_dirs == ("ul", "dl")
    assert NRConfig(n_layers_max=2).mimo_dirs == () and NRConfig(dl=True).mimo_dirs == ()
    for bad in (dict(n_layers_max=3), dict(rank_rule="pmi"), dict(rank_layer_penalty_db=-1.0)):
        with pytest.raises(AssertionError):
            NRConfig(**bad)


def test_f_triton_refuses_rank2():
    assert any("n_layers_max" in f for f, _ in NRTritonEngine.refusals(NRConfig(n_layers_max=2, dl=True)))
    assert not NRTritonEngine.refusals(NRConfig(n_layers_max=2))       # no rank-2 direction: nothing to refuse
    assert not NRTritonEngine.refusals(NRConfig(dl=True, **OFF_FIELDS))
    with pytest.raises(NotImplementedError, match="rank-2 MIMO"):
        make_engine("L2", 2, 2, "cpu", NRConfig(n_layers_max=2, dl=True), "triton")


@pytest.mark.parametrize("name", nr_equiv.MIMO_CFGS)
def test_f_equiv_config_runs_on_the_reference(name):
    cfg = nr_equiv.CFGS[name]().with_(msg_sizes=SIZES, **FAST)
    assert cfg.mimo_dirs == ("ul", "dl")
    eng = make_engine("L2", 4, 3, "cpu", cfg, "reference", seed=3)
    ranks = []
    for t, d in nr_equiv.Workload(cfg, 4, 3, 8, seed=4, p_reset=0.3, device="cpu", phase_offset=25):
        out = nr_equiv.drive(eng, d)
        if isinstance(out, dict) and "dl_rank" in out:
            ranks.append(out["dl_rank"])
    assert len(ranks) == 8 and bool((torch.stack(ranks) == 2).any())
    sd = state_dict(eng)
    assert {"ul.h_rank", "ul.last_rank", "dl.h_rank", "dl.last_rank"} <= set(sd)
    assert math.isfinite(float(eng.net.counters()["dl"]["bytes_ok"]))


class CPUGraph(NRGraphEngine):
    """The graph engine on CPU: captures and replays through nr_fast._EagerReplay (test_limits_closed.py)."""
    _require_cuda = False


@pytest.mark.parametrize("name", nr_equiv.MIMO_CFGS)
def test_f_graph_capture_path_equals_reference(name):
    cfg = nr_equiv.CFGS[name]().with_(msg_sizes=SIZES, **FAST)
    ref = make_engine("L2", 3, 4, "cpu", cfg, "reference", seed=3)
    gr = CPUGraph(3, 4, "cpu", cfg, seed=3)
    for t, d in nr_equiv.Workload(cfg, 3, 4, 10, seed=4, p_reset=0.3, device="cpu", phase_offset=25):
        oa, ob = nr_equiv.drive(ref, d), nr_equiv.drive(gr, d)
        assert set(oa) == set(ob) and "dl_rank" in oa, t
        for k in oa:
            if torch.is_tensor(oa[k]):
                assert torch.equal(oa[k].nan_to_num(-7.0), ob[k].nan_to_num(-7.0)), (t, k)
    sa, sb = state_dict(ref), state_dict(gr)
    for k in sa:
        assert torch.equal(sa[k].nan_to_num(-7.0), sb[k].nan_to_num(-7.0)), k
    assert ref.counters() == gr.counters() and gr.n_replays == 10
