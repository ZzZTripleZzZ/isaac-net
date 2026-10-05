"""Access state machine of the NR engine (core/access.py): RACH / connection setup and connected-mode DRX.

  (a) features off, or on with nothing to do, leave the engine's outputs bitwise unchanged;
  (b) 32 robots powering on together in one cell collide on 64 preambles at the rate of the analytical model and
      all connect within a bounded time;
  (c) a released (idle) robot reconnects on new UL data with the RO wait + RAR + Msg3 delay;
  (d) DRX: a robot sleeps after the inactivity timer, wakes at the on-duration, DL (and UL with
      drx_ul_wake="on_duration") deliveries wait for it, and energy drops with drx_sleep_power_w < idle_power_w;
  (e) env 0 is independent of E, of other envs' resets and of sharding (engine counter RNG);
  (f) unused_fields / make_engine report the new fields honestly.
"""
import pytest
import torch

from isaac_net.core import NRConfig, Requests, make_engine
from isaac_net.core.access import CONNECTED, DORMANT, IDLE
from isaac_net.core.config import multicell
from isaac_net.core.energy import EnergyConfig
from isaac_net.core.sharded import set_env_offset
from isaac_net.core.traffic import TrafficModel as TM

R, EMAX, STEPS, SEED = 3, 6, 8, 5

ACCESS = dict(rach=True, rach_initial="idle", rach_release_after_ms=150.0, drx=True, drx_inactivity_ms=20.0,
              drx_cycle_ms=80.0, drx_on_ms=10.0, drx_short_cycle_ms=40.0)


def _tensors(o, E):
    return {n: v.clone() for n, v in o.items() if torch.is_tensor(v) and v.dim() >= 1 and v.shape[0] == E}


def _inputs(steps=STEPS, seed=0):
    g = torch.Generator().manual_seed(seed)
    out = []
    for k in range(steps):
        send = torch.randint(0, 3, (EMAX, R), generator=g) * (torch.rand(EMAX, R, generator=g) < 0.4)
        out.append((send, torch.rand(EMAX, R, 2, generator=g) * 150,
                    (torch.rand(EMAX, R, generator=g) < 0.3).float() * 2000))
    return out


def _drive(net, E, resets=(), steps=STEPS, dl=False):
    rs = dict(resets)
    outs = []
    for k, (send, pos, dlb) in enumerate(_inputs(steps)):
        net.submit(None, Requests(send[:E], None, torch.full((E,), k, dtype=torch.long)))
        if dl:
            net.add_dl_frames(None, dlb[:E])
        outs.append(_tensors(net.step(None, pos[:E]), E))
        if k in rs:
            net.reset(torch.tensor(rs[k]))
    return outs


def _rows_equal(a, b, rows_a, rows_b=None, keys=None):
    rows_b = rows_a if rows_b is None else rows_b
    n = 0
    for k, (x, y) in enumerate(zip(a, b)):
        names = x.keys() if keys is None else keys
        for name in names:
            u, v = x[name][rows_a], y[name][rows_b]
            if u.is_floating_point():
                u, v = u.nan_to_num(-7.0), v.nan_to_num(-7.0)
            assert torch.equal(u, v), (k, name)
            n += 1
    assert n > 0


# ---------------------------------------------------------------------------------------------- (a) off = before
@pytest.mark.parametrize("base", ["one_cell_dl", "mc3"])
def test_on_with_nothing_to_do_is_bitwise_off(base):
    """rach on but every robot connected and never released, DRX on but never asleep: the gate is all True, so the
    engine's own outputs equal the feature-off engine bitwise (the off engine has no access hooks at all)."""
    cfg = NRConfig(dl=True, control_step_ms=20.0) if base == "one_cell_dl" else multicell(3, dl=True, control_step_ms=20.0)
    off = make_engine("L2", 4, R, "cpu", cfg, seed=SEED)
    assert off.access is None and "slot" not in vars(off.net.ul)
    on = make_engine("L2", 4, R, "cpu", cfg.with_(rach=True, drx=True, drx_inactivity_ms=1e6), seed=SEED)
    a = _drive(off, 4, {3: [1]}, dl=True)
    b = _drive(on, 4, {3: [1]}, dl=True)
    _rows_equal(a, b, slice(None), keys=a[0].keys())
    assert set(b[0]) - set(a[0]) == {"access_state", "access_sleep_frac", "rach_attempts"}
    assert all((o["access_state"] == CONNECTED).all() for o in b)
    assert on.counters()["access"]["rach_attempts"] == 0


def test_off_keys_and_counters_unchanged():
    net = make_engine("L2", 2, R, "cpu", NRConfig(), seed=SEED)
    o = net.step(None, torch.full((2, R), 20.0))
    assert "access_state" not in o and "access" not in net.counters()


# ---------------------------------------------------------------------------------------------- (b) power-on
def test_power_on_collisions_match_analytical_rate():
    """E = 256 independent envs (env-keyed draws: 256 seeds) of 32 robots that all power on at t = 0 and attempt at
    the same first RO. A robot's first preamble succeeds iff none of the other 31 picked it: (63/64)^31."""
    E, n = 256, 32
    cfg = NRConfig(rach=True, rach_initial="idle", fading=False, control_step_ms=20.0)
    net = make_engine("L2", E, n, "cpu", cfg, seed=11)
    snr = torch.full((E, n), 20.0)
    att = torch.zeros(E, n, dtype=torch.long)
    connected_by = None
    for k in range(8):
        net.submit(None, Requests(torch.full((E, n), int(k == 0), dtype=torch.long)))
        o = net.step(None, snr)
        att += o["rach_attempts"]
        if connected_by is None and bool((o["access_state"] == CONNECTED).all()):
            connected_by = k
    p1 = (1 - 1 / cfg.rach_preambles) ** (n - 1)
    first_ok = (att == 1).float().mean().item()
    assert abs(first_ok - p1) < 0.03, (first_ok, p1)
    # expected first-RO collisions per env: n (1 - p1); collisions happen, nobody exhausts preambleTransMax
    c = net.counters()["access"]
    assert c["rach_collisions"] > 0.5 * E * n * (1 - p1) and c["rach_failures"] == 0
    assert c["rach_successes"] == E * n and c["rach_attempts"] == att.sum().item()
    assert 1.0 < att.float().mean().item() < 2.0
    # bounded: with RO = 10 ms, RAR + Msg3 = 10 ms and 20 ms backoff, everybody is in within 100 ms (5 steps)
    assert connected_by is not None and connected_by <= 4


def test_robots_wait_unschedulable_until_connected():
    """Idle robots queue their frames: nothing is delivered before the RO + RAR + Msg3 delay."""
    cfg = NRConfig(rach=True, rach_initial="idle", fading=False, control_step_ms=5.0, timeout_steps=100)
    net = make_engine("L2", 1, 1, "cpu", cfg, seed=0)
    net.submit(None, Requests(torch.ones(1, 1, dtype=torch.long)))
    states, got = [], None
    for k in range(10):
        o = net.step(None, torch.full((1, 1), 40.0))
        states.append(int(o["access_state"]))
        if got is None and bool(o["delivered"].any()):
            got = k
    # RO at slot 3 (first UL-capable slot of the 20-slot window), served from slot 23 = step 2 (10 slots per step)
    assert states[:2] == [1, 1] and states[2] == CONNECTED and got >= 2


# ---------------------------------------------------------------------------------------------- (c) reconnect
def _first_delay(cfg, k_send, steps):
    net = make_engine("L2", 1, 1, "cpu", cfg, seed=0)
    st = []
    for k in range(steps):
        net.submit(None, Requests(torch.full((1, 1), int(k in (0, k_send)), dtype=torch.long)))
        o = net.step(None, torch.full((1, 1), 40.0))
        st.append(int(o["access_state"]) if "access_state" in o else None)
        if k >= k_send and bool(o["delivered"].any()):
            return float(o["delay"][o["delivered"]][0]), st, net
    raise AssertionError("not delivered")


def test_idle_robot_reconnects_with_expected_delay():
    base = NRConfig(fading=False, control_step_ms=10.0, timeout_steps=100)
    cfg = base.with_(rach=True, rach_release_after_ms=50.0)
    k = 20
    d0, _, _ = _first_delay(base, k, 30)
    d1, st, net = _first_delay(cfg, k, 30)
    assert st[k - 1] == IDLE and net.counters()["access"]["rrc_releases"] == 1
    # arrival at slot 400 (20 slots per step); RO at 403; served from 403 + 10 + 10 = 423; first U slot 424. Without
    # RACH the SR at 403 is granted at 406 and the first U slot is 409: 15 slots (0.75 steps) more.
    N = cfg.slots_per_step
    g0 = k * N
    ro = (g0 - 3 + 19) // 20 * 20 + 3
    served = ro + cfg.rach_rar_window_slots + cfg.rach_msg3_slots
    first_u = lambda g: next(x for x in range(g, g + 5) if cfg.tdd_pattern[x % 5] == "U")     # noqa: E731
    sr = next(x for x in range(g0, g0 + 5) if cfg.ul_capable(x))
    extra = first_u(served) - first_u(sr + cfg.sr_delay)
    assert extra == 15
    assert d1 - d0 == pytest.approx(extra / N, abs=1e-9)


def test_dl_data_triggers_access():
    cfg = NRConfig(rach=True, rach_initial="idle", dl=True, fading=False, control_step_ms=5.0)
    net = make_engine("L2", 1, 2, "cpu", cfg, seed=0)
    net.add_dl_frames(None, torch.tensor([[0.0, 500.0]]))
    o = net.step(None, torch.full((1, 2), 30.0))
    assert o["access_state"].tolist() == [[IDLE, 1]] and o["rach_attempts"].tolist() == [[0, 1]]


# ---------------------------------------------------------------------------------------------- (d) DRX
def _drx_run(drx, wake="sr", sleep_w=None, steps=24):
    cfg = NRConfig(dl=True, fading=False, control_step_ms=5.0, timeout_steps=200, drx=drx, drx_inactivity_ms=10.0,
                   drx_cycle_ms=40.0, drx_on_ms=5.0, drx_ul_wake=wake, energy=EnergyConfig(drx_sleep_power_w=sleep_w))
    net = make_engine("L2", 1, 2, "cpu", cfg, seed=0)
    rec = {"state": [], "ul": {}, "dl": {}, "sleep": [], "energy": None}
    for k in range(steps):
        net.submit(None, Requests(torch.tensor([[int(k in (0, 13)), 0]])))
        if k == 13:
            net.add_dl_frames(None, torch.tensor([[0.0, 1000.0]]))
        o = net.step(None, torch.full((1, 2), 30.0))
        rec["state"].append(o["access_state"][0].tolist() if "access_state" in o else None)
        rec["sleep"].append(o["access_sleep_frac"][0].tolist() if "access_sleep_frac" in o else None)
        if bool(o["delivered"].any()):
            rec["ul"][k] = float(o["delay"][o["delivered"]][0])
        if int(o["dl_newest"][0, 1]) >= 0:
            rec["dl"][k] = int(o["dl_newest"][0, 1])
        rec["energy"] = o["energy_cum_j"][0].clone()
    return rec


def test_drx_sleeps_wakes_and_delays():
    off = _drx_run(False)
    on = _drx_run(True)
    late = _drx_run(True, "on_duration")
    # 5 ms steps, inactivity 10 ms, cycle 40 ms, on-duration 5 ms: on-durations are steps 0, 8, 16
    assert on["state"][4] == [DORMANT, DORMANT] and on["state"][8] == [CONNECTED, CONNECTED]
    assert on["sleep"][4] == [1.0, 1.0] and on["sleep"][8] == [0.0, 0.0]
    # DL message submitted at step 13 (dormant robot): delivered at 13 without DRX, at the on-duration (16) with it
    assert off["dl"] == {13: 13} and on["dl"] == {16: 13}
    # UL data at step 13: drx_ul_wake="sr" wakes the robot at once (same delay as without DRX) ...
    assert on["ul"][14] == off["ul"][14]
    assert on["state"][13][0] == CONNECTED
    # ... "on_duration" waits for step 16: 3 steps later, plus the SR / grant delay from the on-duration start
    k_late = min(k for k in late["ul"] if k > 1)
    assert k_late >= 16 and late["ul"][k_late] > off["ul"][14] + 2.0


def test_drx_sleep_power_lowers_energy():
    off = _drx_run(False)
    same = _drx_run(True)                       # drx_sleep_power_w None: idle power while asleep
    low = _drx_run(True, sleep_w=0.001)
    assert torch.equal(off["energy"], same["energy"])
    assert (low["energy"] < off["energy"]).all() and low["energy"][1] < 0.5 * off["energy"][1]     # robot 1: idle
    # the energy model: idle term = (idle (1 - f) + sleep f) * step
    ec = EnergyConfig()
    asleep = sum(s[1] for s in low["sleep"])
    want = (off["energy"][1] - (ec.idle_power_w - 0.001) * asleep * 5e-3).item()
    assert low["energy"][1].item() == pytest.approx(want, rel=1e-4)


def test_short_cycle():
    cfg = NRConfig(fading=False, control_step_ms=5.0, drx=True, drx_inactivity_ms=10.0, drx_cycle_ms=80.0,
                   drx_on_ms=5.0, drx_short_cycle_ms=20.0, drx_short_cycles=2)
    net = make_engine("L2", 1, 1, "cpu", cfg, seed=0)
    st = []
    for k in range(20):
        net.submit(None, Requests(torch.tensor([[int(k == 0)]])))
        st.append(int(net.step(None, torch.full((1, 1), 30.0))["access_state"]))
    on = [k for k, s in enumerate(st) if s == CONNECTED]
    # activity until about step 1, timer expires at step 3; short cycle (20 ms = 4 steps) on-durations at steps 4
    # and 8, then the long cycle (80 ms) at step 16
    assert {4, 8, 16} <= set(on) and 12 not in on


# ---------------------------------------------------------------------------------------------- (e) independence
# no traffic models here: TrafficGen keeps one generator per engine, so its draws depend on E (engine.py)
CASES = {"one_cell": NRConfig(dl=True, control_step_ms=20.0, **ACCESS),
         "mc3": multicell(3, dl=True, control_step_ms=20.0, **ACCESS)}


def _make(case, E):
    return make_engine("L2", E, R, "cpu", CASES[case], seed=SEED)


@pytest.mark.parametrize("case", list(CASES))
def test_env0_independent_of_E(case):
    resets = {3: [1]}
    a = _drive(_make(case, 3), 3, resets, dl=True)
    b = _drive(_make(case, 6), 6, resets, dl=True)
    _rows_equal(a, b, slice(0, 3))
    assert sum(int(o["rach_attempts"].sum()) for o in a) > 0


@pytest.mark.parametrize("case", list(CASES))
def test_other_resets_do_not_shift_env0(case):
    a = _drive(_make(case, 4), 4, {2: [0]}, dl=True)
    b = _drive(_make(case, 4), 4, {0: [1], 1: [1, 2], 2: [0]}, dl=True)
    _rows_equal(a, b, 0)
    _rows_equal(a, b, 3)


@pytest.mark.parametrize("case", list(CASES))
def test_partial_reset_returns_to_initial_state(case):
    a = _drive(_make(case, 4), 4, dl=True)
    net = _make(case, 4)
    b = _drive(net, 4, {5: [1]}, dl=True)
    _rows_equal(a, b, [0, 2, 3])
    acc = net.access
    assert (acc.st[1] == IDLE).all() and (acc.ra_next[1] >= 2 ** 62).all() and (acc.ra_att[1] == 0).all()


def test_two_shards_equal_one_engine():
    E, split = 6, (2, 4)
    resets = {2: [1, 2, 4], 4: [0, 5]}
    un = _drive(_make("one_cell", E), E, resets, dl=True)
    parts, off = [], 0
    for e in split:
        eng = _make("one_cell", e)
        set_env_offset(eng, off)
        rs = {k: [i - off for i in v if off <= i < off + e] for k, v in resets.items()}
        rs = {k: v for k, v in rs.items() if v}
        outs = []
        for k, (send, pos, dlb) in enumerate(_inputs()):
            eng.submit(None, Requests(send[off:off + e], None, torch.full((e,), k, dtype=torch.long)))
            eng.add_dl_frames(None, dlb[off:off + e])
            outs.append(_tensors(eng.step(None, pos[off:off + e]), e))
            if k in rs:
                eng.reset(torch.tensor(rs[k]))
        parts.append(outs)
        off += e
    cat = [{n: torch.cat([p[k][n] for p in parts]) for n in parts[0][k]} for k in range(STEPS)]
    _rows_equal(un, cat, slice(None))


def test_traffic_model_arrivals_trigger_access():
    """Generated messages (no submit) wake idle robots; a robot is served only after its RACH completes."""
    cfg = NRConfig(rach=True, rach_initial="idle", fading=False, control_step_ms=20.0, timeout_steps=50,
                   traffic=[TM.periodic(300, period_ms=40, phase="aligned").on((0,))])
    net = make_engine("L2", 2, 2, "cpu", cfg, seed=0)
    att, got = 0, 0
    for _ in range(6):
        o = net.step(None, torch.full((2, 2), 30.0))
        att += int(o["rach_attempts"][:, 0].sum())
        got += int(o["delivered"][:, 0].sum())
        assert int(o["rach_attempts"][:, 1].sum()) == 0 and (o["access_state"][:, 1] == IDLE).all()
    assert att >= 2 and got >= 4


def test_global_rng_mode_runs():
    net = make_engine("L2", 2, R, "cpu", NRConfig(rng="global", control_step_ms=20.0, **ACCESS), seed=SEED)
    assert net.access._own_rng
    _drive(net, 2, {2: [0]})


# ---------------------------------------------------------------------------------------------- (f) config honesty
def test_unused_fields_and_level_checks():
    c = NRConfig(rach_preambles=32, drx_cycle_ms=320.0)
    assert {"rach_preambles", "drx_cycle_ms"} <= set(c.unused_fields("L2"))
    c = c.with_(rach=True)
    assert "rach_preambles" not in c.unused_fields("L2") and "drx_cycle_ms" in c.unused_fields("L2")
    c = c.with_(drx=True, drx_short_cycles=4)
    assert "drx_cycle_ms" not in c.unused_fields("L2") and "drx_short_cycles" in c.unused_fields("L2")
    assert "drx_short_cycles" not in c.with_(drx_short_cycle_ms=40.0).unused_fields("L2")
    assert {"rach", "drx", "rach_preambles"} <= set(c.unused_fields("L1"))
    for level in ("L1", "L0", "L2-legacy"):
        with pytest.raises(ValueError, match="rach / drx"):
            make_engine(level, 1, 2, "cpu", NRConfig(rach=True))
    with pytest.raises(ValueError, match="triton"):
        make_engine("L2", 1, 2, "cpu", NRConfig(drx=True), backend="triton")
    with pytest.raises(ValueError, match="multiple of the TDD period"):
        make_engine("L2", 1, 2, "cpu", NRConfig(rach=True, rach_occasion_slots=12))
    make_engine("L2", 1, 2, "cpu", NRConfig(rach=True, drx=True), strict=True)


@pytest.mark.gpu
def test_graph_backend_bitwise_equal_reference():
    cfg = NRConfig(dl=True, control_step_ms=20.0, traffic=[TM.periodic(300, period_ms=30)], **ACCESS)
    ref = make_engine("L2", 4, R, "cuda", cfg, seed=SEED)
    gr = make_engine("L2", 4, R, "cuda", cfg, "graph", seed=SEED)
    a = _drive(ref, 4, {3: [1]}, dl=True)
    b = _drive(gr, 4, {3: [1]}, dl=True)
    _rows_equal(a, b, slice(None))
    assert ref.counters()["access"] == gr.counters()["access"]
