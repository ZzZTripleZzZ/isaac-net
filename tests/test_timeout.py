"""Timeouts: an undelivered frame captured at step c is dropped at the end of step c + TIMEOUT - 1,
i.e. after exactly TIMEOUT control steps, never earlier and never later."""
import pytest
import torch

from engine_api import (LEVELS, TIMEOUT, StepProbe, Workload, advance, collect, enable_stats, make_blackhole,
                        make_ref, queued, run, submit)


@pytest.mark.parametrize("level", LEVELS)
def test_timeout_fires_after_exactly_timeout_steps(level, seeded):
    E, R = 2, 3
    net, snr = make_blackhole(level, E, R, "cpu")
    enable_stats(net)
    hid = torch.zeros(E, dtype=torch.long)
    det = torch.zeros(E, R, dtype=torch.bool)
    t_a, t_b = 3, 8                                   # robot 0 sends at t_a and t_b, others at t_a only
    sp = StepProbe(net)
    for t in range(t_b + TIMEOUT + 2):
        send = torch.zeros(E, R, dtype=torch.long)
        if t == t_a:
            send[:] = 1
        if t == t_b:
            send[:, 0] = 2
        submit(net, t, send, det, hid, snr)
        n_before = int(collect(net)["x_cls"].numel())
        newest, _ = advance(net, t, snr, hid)
        n_timed = int(collect(net)["x_cls"].numel()) - n_before
        assert (newest == -1).all(), "black-hole engine delivered a frame"
        rec = sp.last
        if rec.timed.any():
            assert ((t + 1 - rec.cap[rec.timed]) == TIMEOUT).all()
        exp_timed = (E * R if t == t_a + TIMEOUT - 1 else 0) + (E if t == t_b + TIMEOUT - 1 else 0)
        assert n_timed == exp_timed, (t, n_timed, exp_timed)
        q = queued(net)
        exp_q0 = int(t_a <= t < t_a + TIMEOUT - 1) + int(t_b <= t < t_b + TIMEOUT - 1)
        exp_q = int(t_a <= t < t_a + TIMEOUT - 1)
        assert (q[:, 0] == exp_q0).all() and (q[:, 1:] == exp_q).all(), (t, q)
    sp.detach()


@pytest.mark.parametrize("level", LEVELS)
def test_no_frame_outlives_its_deadline(level, seeded):
    """Under a loaded workload every timed-out frame has age exactly TIMEOUT and no queued frame is older."""
    E, R = 3, 6
    net = make_ref(level, E, R, "cpu")
    wl = Workload(E, R, "cpu", seed=7, period=15, snr_lo=-20.0, snr_hi=15.0)
    recs = run(net, wl, steps=60, probe=True)
    n_timed = 0
    for rec in recs:
        age = rec.t + 1 - rec.cap
        assert (age[rec.timed] == TIMEOUT).all()
        assert (age[(rec.cap >= 0) & ~rec.timed & ~rec.delivered] < TIMEOUT).all()
        n_timed += int(rec.timed.sum())
    assert n_timed > 0
