"""Delay accounting is the same at every fidelity level.

For each delivered frame: delay = finish time - capture step, the finish time lies inside the step
in which the frame is reported delivered, 0 < delay <= TIMEOUT, the logged delay equals that value,
and the returned newest-delivered capture step is the max capture step delivered in that step.
"""
import pytest
import torch

from engine_api import (LEVELS, SLOT_LEVELS, TIMEOUT, UL_PER_STEP, StepProbe, Workload, advance, collect,
                        enable_stats, make_ref, run, submit)


@pytest.mark.parametrize("level", LEVELS)
def test_delay_accounting_consistent(level, seeded):
    E, R = 4, 6
    net = make_ref(level, E, R, "cpu")
    enable_stats(net)
    wl = Workload(E, R, "cpu", seed=11, period=10)
    recs = run(net, wl, steps=50, probe=True)
    delays = []
    for rec in recs:
        t, fin, dm = rec.t, rec.fin, rec.delivered
        f = fin[dm]
        if level in SLOT_LEVELS:
            assert ((f > t) & (f <= t + 1)).all()
            slots = (f - t) * UL_PER_STEP                       # delivery happens at the end of a UL slot
            assert ((slots - slots.round()).abs() < 1e-3).all()
        else:
            assert ((f >= t) & (f < t + 1)).all()
        d = fin - rec.cap.float()
        assert ((d[dm] > 0) & (d[dm] <= TIMEOUT)).all()
        delays.append(d[dm])
        exp_newest = torch.where(dm, rec.cap, torch.full_like(rec.cap, -1)).max(-1).values
        assert torch.equal(rec.newest, exp_newest)
        assert torch.isinf(fin[(rec.cap >= 0) & ~dm]).all()    # undelivered frames have no finish time
    logged = collect(net)["delay"]
    assert torch.equal(logged, torch.cat(delays))
    assert logged.numel() > 20


@pytest.mark.parametrize("level", LEVELS)
def test_detection_flag_follows_delivered_frames(level, seeded):
    """det_env[e] is True iff a delivered frame carried a detection of the current hazard id."""
    E, R = 4, 5
    net = make_ref(level, E, R, "cpu")
    wl = Workload(E, R, "cpu", seed=12, period=10)
    sp = StepProbe(net)
    seen = 0
    for t in range(40):
        send, det, hid, snr = wl.inputs(t)
        submit(net, t, send, det, hid, snr)
        det_q, hid_q = net.det.clone(), net.hid.clone()
        _, det_env = advance(net, t, snr, hid)
        dm = sp.last.delivered
        exp = (dm & det_q & (hid_q == hid[:, None, None])).flatten(1).any(-1)
        assert torch.equal(det_env, exp)
        seen += int(exp.sum())
    sp.detach()
    assert seen > 0
