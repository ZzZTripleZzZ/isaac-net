"""Radio link failure (RLF) and re-establishment, and A3 target admission (NRConfig.rlf, a3_min_target_rsrp_dbm; the
state machine is in core/radio.CellAssociation, the hookup in NRNet.step_cells; docs/multicell.md).

  (a) rlf=False is bitwise the pre-feature association and step_cells (frozen copy in tests/rlf_frozen.py) on a
      3-cell L2 run with poses, handovers and a partial reset: every output, per-env state tensor and counter
  (b) a robot driven to -15 dB serving SINR starts T310 after N310 out-of-sync indications, declares RLF at the
      T310 expiry slot, is never scheduled until the re-establishment completes reest_delay later on the best cell;
      its queue is kept (rlf_rlc="carry") or dropped (flush)
  (c) hysteresis: an SINR between Qout and Qin gives no indication; N311 in-sync indications in a row stop T310,
      an out-of-sync one in between restarts their count
  (d) RLF on: partial reset clears the RLF state of the reset envs only, and env rows do not depend on E
  (e) A3 admission: a neighbour below a3_min_target_rsrp_dbm is never a handover target, and the RLF cell search
      skips it too (T311 expires, the robot goes idle and its queue is dropped)
  (f) dead link reclaim: a robot at -10 dB on every cell stops sending TBs once in RLF (tb_new / tb_fail drop)
  (g) config: gating of the new fields in unused_fields, one-cell refusal, step-dict and counters keys
  (h) GPU: the graph backend captures the RLF state machine (bitwise equal to the reference, with partial resets)
"""
import math

import pytest
import torch

import rlf_frozen
from isaac_net.core import NRConfig, Requests, make_engine, multicell

# serving SINR under control: no interference, so the N+I estimate is the noise floor and sinr = pg + ue_tx - noise;
# 20 ms control steps (40 slots) keep the CPU tests short
QUIET = dict(ul_interference=False, dl_interference=False, control_step_ms=20.0)


def _pg(cfg, sinr):
    """Path gain [.., C] that gives these full-power serving-link SINRs (dB) without interference."""
    return torch.as_tensor(sinr, dtype=torch.float32) - cfg.ue_tx_dbm + cfg.subband_noise_dbm


def _tx_slots(trace, N, r=0, e=0):
    return sorted({round(frac * N) - 1 for frac, tx, *_ in trace if tx[e, r]})


# ---------------------------------------------------------------- (a) rlf=False is the pre-feature engine
def _per_env(net):
    owners = {"": net, "ul.": net.ul, "ul.q.": net.ul.q, "assoc.": net.assoc}
    if net.dl is not None:
        owners.update({"dl.": net.dl, "dl.q.": net.dl.q})
    return {p + n: v.clone() for p, o in owners.items() for n, v in vars(o).items()
            if torch.is_tensor(v) and v.dim() >= 1 and v.shape[0] == net.E}


def _drive_poses(eng, E, R, steps, reset_at, dl):
    g = torch.Generator().manual_seed(8)
    pos = torch.rand(E, R, 2, generator=g) * 150
    outs, states = [], []
    for t in range(steps):
        if t == reset_at:
            eng.reset(torch.tensor([1, 3]))
        u = torch.rand(E, R, generator=g)
        send = (u < 0.1).long() * 2 + ((u >= 0.1) & (u < 0.6)).long()
        pos = (pos + 4.0 * (2 * torch.rand(E, R, 2, generator=g) - 1)).clamp(0, 150)
        eng.submit(None, send)
        if dl:
            eng.add_dl_frames(None, torch.where(send > 0, eng.net.sizes[(send - 1).clamp(min=0)], torch.zeros(E, R)))
        o = eng.step(None, pos)
        outs.append({k: v.clone() for k, v in o.items() if torch.is_tensor(v)})
        states.append(_per_env(eng.net))
    return outs, states


@pytest.mark.parametrize("dl", [False, True])
def test_a_rlf_off_bitwise_equals_pre_feature(dl):
    E, R, steps = 4, 6, 20
    cfg = multicell(3, dl=dl, a3_ttt_ms=40.0, a3_hyst_db=1.0, cell_isd_m=60.0, control_step_ms=20.0)
    assert not cfg.rlf and cfg.a3_min_target_rsrp_dbm is None
    new = make_engine("L2", E, R, "cpu", cfg, seed=3)
    old = make_engine("L2", E, R, "cpu", cfg, seed=3)
    rlf_frozen.install(old.net)
    a = _drive_poses(new, E, R, steps, steps // 2, dl)
    b = _drive_poses(old, E, R, steps, steps // 2, dl)
    for t, (oa, ob, sa, sb) in enumerate(zip(a[0], b[0], a[1], b[1])):
        assert set(oa) == set(ob) and "rlf" not in oa
        for k in oa:
            assert torch.equal(oa[k].nan_to_num(-7.0), ob[k].nan_to_num(-7.0)), (t, k)
        assert set(sa) == set(sb)
        for k in sa:
            assert torch.equal(sa[k].nan_to_num(-7.0) if sa[k].is_floating_point() else sa[k],
                               sb[k].nan_to_num(-7.0) if sb[k].is_floating_point() else sb[k]), (t, k)
    assert new.counters() == old.counters() and "rlf" not in new.counters()
    assert int(old.net.assoc.n_ho.sum()) > 0 and not hasattr(new.net.assoc, "rlf_active")


# ---------------------------------------------------------------- (b) RLF and re-establishment
def _fail_run(rlf_rlc, steps=22, **kw):
    """Robot 0 of every env: 20 dB on cell 0 for 5 steps, then -15 dB on cell 0 (its serving cell) and 5 dB on
    cell 1; robots 1 and 2 stay at 20 dB on cell 0. A3 is off (huge TTT) so only RLF can move robot 0."""
    cfg = multicell(3, rlf=True, a3_ttt_ms=1e6, t310_ms=200.0, reest_delay_ms=50.0, frame_buffer=64,
                    timeout_steps=200, rlf_rlc=rlf_rlc, **QUIET, **kw)
    E, R = 2, 3
    eng = make_engine("L2", E, R, "cpu", cfg, seed=0)
    eng.net.ul.trace = []
    rec = []
    for t in range(steps):
        s0 = [20.0, 10.0, 0.0] if t < 5 else [-15.0, 5.0, 0.0]
        pg = _pg(cfg, [s0, [20.0, 0.0, 0.0], [20.0, 0.0, 0.0]])[None].expand(E, R, 3).clone()
        eng.submit(None, torch.ones(E, R, dtype=torch.long))
        o = eng.step(None, pathgain_db=pg)
        rec.append({k: v.clone() for k, v in o.items() if torch.is_tensor(v)})
    return eng, cfg, rec


def test_b_rlf_declared_unschedulable_reestablished_carry():
    eng, cfg, rec = _fail_run("carry")
    N, asc = cfg.slots_per_step, eng.net.assoc
    assert (asc.t310, asc.reest) == (round(cfg.t310_ms / cfg.slot_ms), round(cfg.reest_delay_ms / cfg.slot_ms))
    # first out-of-sync indication at step 5 (n310 = 1): T310 from slot 5N, RLF at 5N + T310, back at + reest
    g_rlf = 5 * N + asc.t310
    g_back = g_rlf + asc.reest
    rlf = [bool(o["rlf"][0, 0]) for o in rec]
    assert rlf == [g_rlf <= t * N + N - 1 < g_back for t in range(len(rec))], rlf
    assert not any(bool(o["rlf"][:, 1:].any()) for o in rec)
    cells = [int(o["serving_cell"][0, 0]) for o in rec]
    assert cells == [0 if t * N + N - 1 < g_back else 1 for t in range(len(rec))], cells    # best cell: 1
    tx = _tx_slots(eng.net.ul.trace, N)
    assert not any(g_rlf <= g < g_back for g in tx)
    ul = [g for g in range(len(rec) * N) if cfg.slot_symbols(g)[1] > 0]
    assert next(g for g in ul if g >= g_back) in tx                    # served at the first UL slot after it
    # carry: the backlog survives the outage (nothing dropped) and is served afterwards
    assert all(int(o["dropped"].sum()) == 0 for o in rec)
    t_rlf = g_rlf // N
    assert int(rec[t_rlf]["queue_len"][0, 0]) >= int(rec[t_rlf - 1]["queue_len"][0, 0]) > 0
    c = eng.counters()["rlf"]
    assert c == {"rlf": 2.0, "reest": 2.0, "rlf_idle": 0.0, "t310_start": 2.0, "t310_stop": 0.0}
    assert (asc.n_rlf[:, 0] == 1).all() and (asc.n_rlf[:, 1:] == 0).all() and (asc.n_ho == 0).all()


def test_b_rlf_flush_drops_the_queue():
    eng, cfg, rec = _fail_run("flush")
    N, asc = cfg.slots_per_step, eng.net.assoc
    t_rlf = (5 * N + asc.t310) // N
    q_before = int(rec[t_rlf - 1]["queue_len"][0, 0])
    assert q_before > 0
    dropped = [int(o["dropped"][:, 0].sum()) for o in rec]
    assert dropped[t_rlf] > 0 and sum(dropped) == dropped[t_rlf]
    assert all(int(o["dropped"][:, 1:].sum()) == 0 for o in rec)
    # rlf_rlc=None follows ho_rlc
    assert multicell(3, ho_rlc="flush").rlf_flush and not multicell(3, ho_rlc="flush", rlf_rlc="carry").rlf_flush


def test_b_n310_needs_consecutive_out_of_sync():
    eng, cfg, rec = _fail_run("carry", steps=8, n310=3)
    N, asc = cfg.slots_per_step, eng.net.assoc
    assert int(asc.t310_end[0, 0]) == 7 * N + asc.t310     # indications at steps 5, 6, 7


# ---------------------------------------------------------------- (c) hysteresis
def _sinr_run(seq, fail=False, **kw):
    cfg = multicell(3, rlf=True, a3_ttt_ms=1e6, **QUIET, **kw)
    E, R = 1, 2
    eng = make_engine("L2", E, R, "cpu", cfg, seed=1)
    t310 = []
    for s in seq:
        pg = _pg(cfg, [[s, -30.0, -30.0], [20.0, -30.0, -30.0]])[None].clone()
        eng.submit(None, torch.ones(E, R, dtype=torch.long))
        o = eng.step(None, pathgain_db=pg)
        assert fail or not bool(o["rlf"].any())
        t310.append(int(eng.net.assoc.t310_end[0, 0]))
    return eng, t310


def test_c_hysteresis_n311_stops_t310():
    # 20 x3, OOS (T310 starts), between, IS, OOS (restarts the IS count), IS, IS (N311 = 2 -> stop)
    seq = [20.0] * 3 + [-10.0, -7.0, -5.0, -10.0, -5.0, -5.0, 20.0]
    eng, t310 = _sinr_run(seq, n311=2)
    N = eng.config.slots_per_step
    run = [x >= 0 for x in t310]
    assert run == [False] * 3 + [True] * 5 + [False] * 2, t310
    assert len({x for x in t310 if x >= 0}) == 1 and t310[3] == 3 * N + eng.net.assoc.t310
    assert eng.counters()["rlf"] == {"rlf": 0.0, "reest": 0.0, "rlf_idle": 0.0, "t310_start": 1.0, "t310_stop": 1.0}
    _, t310 = _sinr_run(seq, n311=1)                      # the first in-sync indication stops it
    assert [x >= 0 for x in t310] == [False] * 3 + [True] * 2 + [False] + [True] * 1 + [False] * 3


def test_c_between_qout_and_qin_neither_starts_nor_stops():
    _, t310 = _sinr_run([-7.0] * 8)                        # no indication at all
    assert all(x < 0 for x in t310)
    eng, t310 = _sinr_run([-10.0] + [-7.0] * 12, fail=True, t310_ms=100.0)
    assert t310[0] >= 0 and len(set(t310[:5])) == 1          # in-between SINR does not stop T310
    assert eng.counters()["rlf"]["rlf"] == 1.0 and eng.counters()["rlf"]["t310_stop"] == 0.0


# ---------------------------------------------------------------- (d) partial reset and E-independence, RLF on
RD, EMAX, STEPS = 4, 6, 14


def _rlf_cfg():
    return multicell(3, rlf=True, t310_ms=40.0, t311_ms=100.0, reest_delay_ms=30.0, rlf_rlc="flush",
                     control_step_ms=20.0)


def _inputs():
    """Per-step (send [EMAX,RD], path gain [EMAX,RD,3]) with serving SINRs from -20 to 20 dB."""
    cfg = _rlf_cfg()
    g = torch.Generator().manual_seed(2)
    return [(torch.randint(0, 3, (EMAX, RD), generator=g),
             _pg(cfg, -20 + 40 * torch.rand(EMAX, RD, 3, generator=g))) for _ in range(STEPS)]


def _drive(eng, E, resets=()):
    rs = dict(resets)
    outs = []
    for k, (send, pg) in enumerate(_inputs()):
        eng.submit(None, Requests(send[:E], None, torch.full((E,), k, dtype=torch.long)))
        o = eng.step(None, pathgain_db=pg[:E])
        outs.append({n: v.clone() for n, v in o.items() if torch.is_tensor(v) and v.dim() >= 1 and v.shape[0] == E})
        outs[-1].update({n: v for n, v in _per_env(eng.net).items() if n.startswith("assoc.")})
        if k in rs:
            eng.reset(torch.tensor(rs[k]))
    return outs


def _rows_equal(a, b, rows_a, rows_b=None):
    rows_b = rows_a if rows_b is None else rows_b
    for k, (x, y) in enumerate(zip(a, b)):
        assert x.keys() == y.keys()
        for n in x:
            u, v = x[n][rows_a], y[n][rows_b]
            if u.is_floating_point():
                u, v = u.nan_to_num(-7.0), v.nan_to_num(-7.0)
            assert torch.equal(u, v), (k, n)


def test_d_rlf_env_rows_independent_of_E():
    a = _drive(make_engine("L2", 3, RD, "cpu", _rlf_cfg(), seed=5), 3, {6: [1]})
    eng = make_engine("L2", EMAX, RD, "cpu", _rlf_cfg(), seed=5)
    b = _drive(eng, EMAX, {6: [1]})
    _rows_equal(a, b, slice(0, 3))
    c = eng.counters()["rlf"]
    assert c["rlf"] > 0 and c["reest"] > 0 and c["t310_stop"] > 0, c


def test_d_partial_reset_clears_rlf_state_of_reset_envs_only():
    cfg = _rlf_cfg()
    a = _drive(make_engine("L2", 4, RD, "cpu", cfg, seed=5), 4)
    eng = make_engine("L2", 4, RD, "cpu", cfg, seed=5)
    b = _drive(eng, 4, {7: [1, 2]})
    _rows_equal(a, b, [0, 3])
    assert any(bool(o["rlf"][1:3].any()) or bool((o["assoc.t310_end"][1:3] >= 0).any()) for o in a[:8])
    # the reset rows equal a fresh engine's association state
    eng2 = make_engine("L2", 4, RD, "cpu", cfg, seed=5)
    for k in range(8):
        send, pg = _inputs()[k]
        eng2.submit(None, Requests(send[:4], None, torch.full((4,), k, dtype=torch.long)))
        eng2.step(None, pathgain_db=pg[:4])
    before = _per_env(eng2.net)
    eng2.reset(torch.tensor([1, 2]))
    fresh = _per_env(make_engine("L2", 4, RD, "cpu", cfg, seed=5).net)
    after = _per_env(eng2.net)
    for n in after:
        if n.startswith("assoc."):
            assert torch.equal(after[n][1:3], fresh[n][1:3]), n
            assert torch.equal(after[n][[0, 3]], before[n][[0, 3]]), n
    assert {"assoc.rlf_active", "assoc.t310_end", "assoc.t311_end", "assoc.reest_end", "assoc.oos_cnt",
            "assoc.is_cnt", "assoc.reest_cell", "assoc.n_rlf"} <= set(after)


# ---------------------------------------------------------------- (e) A3 target admission
def _rsrp(cfg, pg):
    return pg + cfg.gnb_tx_dbm - 10 * math.log10(12 * cfg.nprb)


def _a3_run(floor, steps=8):
    cfg = multicell(3, a3_ttt_ms=0.0, a3_min_target_rsrp_dbm=floor, **QUIET)
    E, R = 1, 2
    eng = make_engine("L2", E, R, "cpu", cfg, seed=0)
    # robot 0 attaches to cell 0 (10 dB), then cell 1 becomes 10 dB stronger than cell 0 (A3 with 3 dB hysteresis)
    pg = [_pg(cfg, [[10.0, 0.0, -20.0], [20.0, 0.0, 0.0]])[None]] + [_pg(cfg, [[5.0, 15.0, -20.0],
                                                                                 [20.0, 0.0, 0.0]])[None]] * steps
    for p in pg:
        eng.submit(None, torch.ones(E, R, dtype=torch.long))
        o = eng.step(None, pathgain_db=p.clone())
    return eng, cfg, o, _rsrp(cfg, pg[-1][0, 0, 1])


def test_e_a3_target_below_floor_is_not_admitted():
    eng, cfg, o, rsrp1 = _a3_run(None)
    assert int(o["serving_cell"][0, 0]) == 1 and int(eng.net.assoc.n_ho[0, 0]) == 1
    eng, _, o, _ = _a3_run(float(rsrp1) + 1.0)
    assert int(o["serving_cell"][0, 0]) == 0 and int(eng.net.assoc.n_ho.sum()) == 0
    eng, _, o, _ = _a3_run(float(rsrp1) - 1.0)
    assert int(o["serving_cell"][0, 0]) == 1 and int(eng.net.assoc.n_ho[0, 0]) == 1
    # the floor also gates NetSlotMC's A3 (shared CellAssociation) and is read only with several cells
    assert eng.net.assoc.floor_rx == pytest.approx(float(rsrp1) - 1.0 - float(_rsrp(cfg, 0.0)) + cfg.ue_tx_dbm)


def test_e_rlf_search_skips_cells_below_floor_then_idles():
    """Robot 0 fails on cell 0; cell 1 is decodable but below the RSRP floor, so no suitable cell exists: T311
    expires, the robot goes idle (queue dropped although rlf_rlc="carry") and connects once cell 2 rises."""
    base = multicell(3)
    floor = float(_rsrp(base, _pg(base, 8.0)))            # RSRP of a link at 8 dB SINR
    cfg = multicell(3, rlf=True, a3_ttt_ms=1e6, t310_ms=40.0, t311_ms=60.0, reest_delay_ms=10.0,
                    a3_min_target_rsrp_dbm=floor, rlf_rlc="carry", frame_buffer=64, timeout_steps=200, **QUIET)
    E, R = 1, 2
    eng = make_engine("L2", E, R, "cpu", cfg, seed=0)
    N, asc = cfg.slots_per_step, eng.net.assoc
    rec = []
    for t in range(16):
        s0 = [20.0, 0.0, -20.0] if t < 2 else ([-15.0, 5.0, -20.0] if t < 11 else [-15.0, 5.0, 12.0])
        pg = _pg(cfg, [s0, [20.0, 0.0, 0.0]])[None].clone()
        eng.submit(None, torch.ones(E, R, dtype=torch.long))
        o = eng.step(None, pathgain_db=pg)
        rec.append({k: v.clone() for k, v in o.items() if torch.is_tensor(v)})
    g_rlf = 2 * N + asc.t310                               # step 4
    t_idle = -(-(g_rlf + asc.t311) // N)                   # first evaluation at or after T311 expiry
    rlf = [bool(o["rlf"][0, 0]) for o in rec]
    assert rlf[g_rlf // N] and all(rlf[g_rlf // N:11]) and not rlf[-1], rlf
    dropped = [int(o["dropped"][0, 0].sum()) for o in rec]
    assert dropped[t_idle] > 0 and sum(dropped) == dropped[t_idle], dropped
    assert int(rec[-1]["serving_cell"][0, 0]) == 2                       # cell 1 (5 dB, below floor) never
    assert all(int(o["serving_cell"][0, 0]) == 0 for o in rec[:11])
    c = eng.counters()["rlf"]
    assert c["rlf"] == 1.0 and c["rlf_idle"] == 1.0 and c["reest"] == 1.0, c


# ---------------------------------------------------------------- (f) dead link reclaim
def _dead_link(rlf, steps=60):
    cfg = multicell(3, rlf=rlf, a3_ttt_ms=1e6, t310_ms=200.0, **QUIET)
    E, R = 2, 4
    eng = make_engine("L2", E, R, "cpu", cfg, seed=0)
    eng.net.ul.trace = []
    pg = _pg(cfg, [[-10.0, -12.0, -14.0]] + [[15.0, 0.0, 0.0]] * (R - 1))[None].expand(E, R, 3).clone()
    for _ in range(steps):
        eng.submit(None, torch.ones(E, R, dtype=torch.long))
        eng.step(None, pathgain_db=pg)
    return eng, cfg


def test_f_dead_link_stops_burning_prbs():
    off, cfg = _dead_link(False)
    on, _ = _dead_link(True)
    N = cfg.slots_per_step
    n_off = len(_tx_slots(off.net.ul.trace, N))
    tx_on = _tx_slots(on.net.ul.trace, N)
    g_rlf = on.net.assoc.t310                              # first evaluation at slot 0
    assert n_off > 100 and tx_on and max(tx_on) < g_rlf, (n_off, len(tx_on))
    co, cn = off.counters()["ul"], on.counters()["ul"]
    assert cn["tb_new"] < co["tb_new"] and cn["tb_fail"] < co["tb_fail"], (cn, co)
    assert len(tx_on) < 0.25 * n_off
    # the healthy robots keep their service
    assert bool(on.net.assoc.rlf_active[:, 0].all()) and not bool(on.net.assoc.rlf_active[:, 1:].any())


# ---------------------------------------------------------------- (g) config and API
def test_g_config_gating_and_refusals():
    assert multicell(3, rlf=True, t310_ms=500.0, n310=2, rlf_rlc="flush").unused_fields("L2") == []
    assert multicell(3, t310_ms=500.0).unused_fields("L2") == ["t310_ms"]             # rlf off
    assert NRConfig(rlf=True).unused_fields("L2") == ["rlf"]                         # one cell
    assert "rlf" in multicell(3, rlf=True).unused_fields("L2-legacy")                # NetSlotMC has no RLF
    assert multicell(3, a3_min_target_rsrp_dbm=-110.0).unused_fields("L2") == []
    assert "a3_min_target_rsrp_dbm" not in multicell(3, a3_min_target_rsrp_dbm=-110.0).unused_fields("L2-legacy")
    assert "gnb_tx_dbm" not in multicell(3, a3_min_target_rsrp_dbm=-110.0, gnb_tx_dbm=30.0).unused_fields("L2")
    assert NRConfig(a3_min_target_rsrp_dbm=-110.0).unused_fields("L2") == ["a3_min_target_rsrp_dbm"]
    with pytest.raises(ValueError, match="n_cells > 1"):
        make_engine("L2", 2, 2, "cpu", NRConfig(rlf=True))
    with pytest.raises(AssertionError):
        NRConfig(rlf_qin_db=-10.0)
    with pytest.raises(AssertionError):
        NRConfig(rlf_rlc="drop")


def test_g_step_dict_and_counters_only_when_on():
    E, R = 2, 3
    for rlf in (False, True):
        eng = make_engine("L2", E, R, "cpu", multicell(3, rlf=rlf), seed=0)
        eng.submit(None, torch.ones(E, R, dtype=torch.long))
        o = eng.step(None, torch.rand(E, R, 2) * 150)
        assert ("rlf" in o) == rlf and ("rlf" in eng.counters()) == rlf
        if rlf:
            assert o["rlf"].shape == (E, R) and o["rlf"].dtype == torch.bool


# ---------------------------------------------------------------- graph backend (GPU)
@pytest.mark.gpu
def test_graph_backend_with_rlf_bitwise_equals_reference():
    """The RLF state machine is captured: the graph backend equals the reference bitwise, partial resets included."""
    cfg = _rlf_cfg()
    E = EMAX
    ref = make_engine("L2", E, RD, "cuda", cfg, seed=5)
    gra = make_engine("L2", E, RD, "cuda", cfg, seed=5, backend="graph")
    for k, (send, pg) in enumerate(_inputs()):
        outs = []
        for eng in (ref, gra):
            eng.submit(None, Requests(send.cuda(), None, torch.full((E,), k, dtype=torch.long, device="cuda")))
            o = eng.step(None, pathgain_db=pg.cuda())
            outs.append({n: v.clone() for n, v in o.items() if torch.is_tensor(v)})
            if k == 6:
                eng.reset(torch.tensor([1, 4], device="cuda"))
        assert outs[0].keys() == outs[1].keys() and "rlf" in outs[0]
        for n in outs[0]:
            assert torch.equal(outs[0][n].nan_to_num(-7.0), outs[1][n].nan_to_num(-7.0)), (k, n)
    assert ref.counters() == gra.counters() and ref.counters()["rlf"]["rlf"] > 0
