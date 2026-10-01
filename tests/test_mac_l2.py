"""L2 MAC invariants: OLLA bounds, HARQ counters and RLC wait, power-headroom subband cap."""
import math

import torch

from isaac_net.core.proto import netsim as ns
from engine_api import (S, UL_PER_STEP, InjectNoise, Workload, advance, make_fast, make_ref, queued, run,
                        slot_probe, submit)

HARQ_MAX, HARQ_RTT, RLC_EXTRA, SR_DELAY = ns.HARQ_MAX, ns.HARQ_RTT, ns.RLC_EXTRA, ns.SR_DELAY


def _single_robot(snr_db=-60.0, cls=1):
    """One env, one robot, one frame queued at t=0 (not yet stepped). Fading gain starts at 0 dB."""
    net = make_ref("L2", 1, 1, "cpu")
    net.h = torch.full_like(net.h, 2 ** -0.5)
    snr = torch.full((1, 1), snr_db)
    hid = torch.zeros(1, dtype=torch.long)
    submit(net, 0, torch.full((1, 1), cls), torch.zeros(1, 1, dtype=torch.bool), hid, snr)
    return net, snr, hid


def _step_forced(net, t, snr, hid, u_per_slot):
    """One step with frozen fading and BLER draws u_per_slot[k] (0 = success, 1 = failure).

    The injected innovation nz = h (1 - RHO) sqrt(2) / sqrt(1 - RHO^2) keeps h (and the gain) constant,
    so the success probability never underflows to 0 and u = 0 always decodes."""
    nz = (net.h * (1 - ns.RHO) * math.sqrt(2) / math.sqrt(1 - ns.RHO ** 2)).expand(UL_PER_STEP, -1, -1, -1, -1)
    u = torch.as_tensor(u_per_slot, dtype=torch.float32).view(UL_PER_STEP, 1, 1).expand(-1, net.E, net.R).clone()
    keys = ["g", "tx", "ok_tb", "fail", "self.hcnt", "self.wait", "self.olla"]
    with slot_probe(keys) as slots, InjectNoise(nz, u):
        advance(net, t, snr, hid)
    return slots


def test_harq_timeline_all_failures():
    """SR at slot 0, grant at slot SR_DELAY; each failure waits HARQ_RTT; the HARQ_MAX-th failure resets
    the counter and adds RLC_EXTRA slots of wait."""
    net, snr, hid = _single_robot()
    slots = _step_forced(net, 0, snr, hid, [1.0] * UL_PER_STEP)
    txg = [s["g"] for s in slots if bool(s["tx"])]
    exp, g, h = [], SR_DELAY, 0
    hc_exp = []
    while g < UL_PER_STEP:
        exp.append(g)
        hc_exp.append(h)
        h += 1
        if h >= HARQ_MAX:
            h, g = 0, g + HARQ_RTT + RLC_EXTRA
        else:
            g = g + HARQ_RTT
    assert txg == exp == [2, 6, 10, 14, 28, 32, 36]
    assert [int(s["self.hcnt"]) for s in slots if bool(s["tx"])] == hc_exp
    assert int(net.hcnt) == 3 and int(net.wait) == 40
    assert abs(float(net.olla) - (-0.45 * len(exp))) < 1e-5
    # next step continues the cycle: 4th failure at slot 40 triggers the RLC wait
    slots = _step_forced(net, 1, snr, hid, [1.0] * UL_PER_STEP)
    txg = [s["g"] for s in slots if bool(s["tx"])]
    assert txg == [40, 54, 58, 62, 66]
    assert int(net.hcnt) == 0 and int(net.wait) == 66 + HARQ_RTT + RLC_EXTRA
    assert int(queued(net).sum()) == 1                   # nothing served, frame still queued


def test_harq_counter_resets_on_success():
    net, snr, hid = _single_robot()
    u = [1.0] * 6 + [0.0] * (UL_PER_STEP - 6)            # fail at slot 2, succeed from slot 6 on
    slots = _step_forced(net, 0, snr, hid, u)
    tx = [s for s in slots if bool(s["tx"])]
    assert [s["g"] for s in tx[:3]] == [2, 6, 7]
    assert [int(s["self.hcnt"]) for s in tx[:3]] == [0, 1, 0]
    assert all(int(s["self.hcnt"]) == 0 for s in tx[2:])
    assert int(net.hcnt) == 0


def test_harq_counter_zero_when_queue_empty(seeded):
    E, R = 4, 6
    net = make_ref("L2", E, R, "cpu")
    wl = Workload(E, R, "cpu", seed=31, period=10, snr_lo=-15.0, snr_hi=25.0)
    n_nonzero = [0]

    def check(t, net, newest, det_env):
        h = net.hcnt
        assert (h == h.round()).all() and (h >= 0).all() and (h <= HARQ_MAX - 1).all()
        assert (h[queued(net) == 0] == 0).all()
        n_nonzero[0] += int((h > 0).sum())

    run(net, wl, steps=50, on_step=check)
    assert n_nonzero[0] > 0


def test_olla_saturates_at_bounds():
    net, snr, hid = _single_robot()
    for t in range(12):                                  # ~7 failures per step, 0.45 dB each
        _step_forced(net, t, snr, hid, [1.0] * UL_PER_STEP)
    assert float(net.olla) == -10.0
    net, snr, hid = _single_robot(snr_db=5.0, cls=2)
    for t in range(8):
        if t:
            submit(net, t, torch.full((1, 1), 2), torch.zeros(1, 1, dtype=torch.bool), hid, snr)
        slots = _step_forced(net, t, snr, hid, [0.0] * UL_PER_STEP)
        assert all(-10.0 <= float(s["self.olla"]) <= 10.0 for s in slots)
    assert float(net.olla) == 10.0


def test_olla_within_bounds_under_load(seeded):
    E, R = 4, 8
    net = make_ref("L2", E, R, "cpu")
    wl = Workload(E, R, "cpu", seed=32, period=10, snr_lo=-15.0, snr_hi=40.0)
    with slot_probe(["self.olla"]) as slots:
        run(net, wl, steps=40)
    ol = torch.stack([s["self.olla"] for s in slots])
    assert (ol >= -10).all() and (ol <= 10).all()
    assert ((net.olla >= -10) & (net.olla <= 10)).all()
    assert ol.min() < -1 and ol.max() > 0.5              # the loop actually moved


def _check_phr(slots):
    binding = 0
    for s in slots:
        won, n_max, snr = s["won"], s["n_max"], s["snr_db"]
        exp = torch.floor(10 ** ((snr - ns.PHR_MIN_DB) / 10)).clamp(1, S)
        assert torch.equal(n_max, exp)
        n = won.sum(-1)
        assert (n <= n_max).all(), "robot won more subbands than its power headroom allows"
        binding += int(((n == n_max) & (n_max < S) & (n > 1)).sum())
    return binding


def test_power_headroom_caps_subbands_reference(seeded):
    E, R = 4, 2
    net = make_ref("L2", E, R, "cpu")
    wl = Workload(E, R, "cpu", seed=33, period=10, snr_lo=0.0, snr_hi=12.0, p=[0.9], big=[1.0])
    with slot_probe(["won", "n_max", "snr_db"]) as slots:
        run(net, wl, steps=15)
    assert _check_phr(slots) > 0, "cap never bound; test is vacuous"


def test_power_headroom_caps_subbands_fast_eager(seeded):
    from isaac_net.core.proto import netsim_fast
    E, R = 4, 2
    net = make_fast(E, R, "cpu", backend="eager")
    wl = Workload(E, R, "cpu", seed=33, period=10, snr_lo=0.0, snr_hi=12.0, p=[0.9], big=[1.0])
    with slot_probe(["won", "n_max", "snr_db"], module=netsim_fast) as slots:
        run(net, wl, steps=15)
    assert _check_phr(slots) > 0
