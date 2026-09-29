"""Conservation: every accepted frame is delivered once, times out once, or is still queued.

Bytes: submitted = served + (residual of timed-out frames) + (residual still queued), and a
delivered frame has had all of its bytes served.
"""
from collections import defaultdict

import pytest
import torch

from engine_api import (F, LEVELS, SIZES, SLOT_LEVELS, StepProbe, Workload, advance, collect, enable_stats,
                        make_ref, run, slot_probe, submit)

SZ = torch.tensor(SIZES, dtype=torch.float64)


def _ids(mask, cap):
    e, r, f = mask.nonzero(as_tuple=True)
    return list(zip(e.tolist(), r.tolist(), cap[e, r, f].tolist()))


@pytest.mark.parametrize("level", LEVELS)
def test_frame_conservation_and_no_double_delivery(level, seeded):
    E, R = 4, 6
    net = make_ref(level, E, R, "cpu")
    enable_stats(net)
    wl = Workload(E, R, "cpu", seed=5, period=12)
    recs = run(net, wl, steps=70, probe=True)

    n_acc = n_dlv = n_to = 0
    b_acc = b_dlv = b_to = 0.0
    delivered, timed_out, accepted = set(), set(), set()
    for rec in recs:
        acc = rec.accepted
        n_acc += int(acc.sum())
        b_acc += float(SZ[(rec.extra["send"] - 1).clamp(min=0)][acc].sum())
        e, r = acc.nonzero(as_tuple=True)
        accepted |= set(zip(e.tolist(), r.tolist(), [rec.t] * len(e)))
        assert not (rec.delivered & rec.timed).any()
        d_ids, x_ids = _ids(rec.delivered, rec.cap), _ids(rec.timed, rec.cap)
        assert not (set(d_ids) & delivered), "frame delivered twice"
        assert not (set(x_ids) & (timed_out | delivered)), "frame removed twice"
        delivered |= set(d_ids)
        timed_out |= set(x_ids)
        n_dlv += len(d_ids)
        n_to += len(x_ids)
        b_dlv += float(SZ[rec.cls[rec.delivered] - 1].sum())
        b_to += float(SZ[rec.cls[rec.timed] - 1].sum())
    live = net.cap >= 0
    n_q = int(live.sum())
    b_q = float(SZ[net.cls[live] - 1].sum())
    assert n_acc == n_dlv + n_to + n_q
    assert b_acc == b_dlv + b_to + b_q
    assert (delivered | timed_out) <= accepted and not (set(_ids(live, net.cap)) & (delivered | timed_out))
    assert n_dlv > 0 and n_to + n_q > 0            # the workload exercises both outcomes
    st = collect(net)
    assert st["delay"].numel() == n_dlv and st["x_cls"].numel() == n_to
    # overflow = frames offered to a full FIFO
    n_offered = sum(int((rec.extra["send"] > 0).sum()) for rec in recs)
    assert st["overflow"] == n_offered - n_acc


@pytest.mark.parametrize("level", SLOT_LEVELS)
def test_byte_conservation_slot_levels(level, seeded):
    """Per slot, serve_fifo removes min(granted, queued) bytes; per frame, a delivered frame had its
    whole size served; overall submitted = served + timed-out residual + queued residual."""
    E, R = 3, 5
    net = make_ref(level, E, R, "cpu")
    wl = Workload(E, R, "cpu", seed=6, period=10)
    served_per_frame = defaultdict(float)
    size_of = {}
    b_sub = b_served = b_to = 0.0
    max_untouched_drift = 0.0
    prev_end = None
    for t in range(60):
        send, det, hid, snr = wl.inputs(t)
        acc = submit(net, t, send, det, hid, snr)
        e, r = acc.nonzero(as_tuple=True)
        for ee, rr in zip(e.tolist(), r.tolist()):
            size_of[(ee, rr, t)] = SIZES[int(send[ee, rr]) - 1]
        b_sub += float(SZ[(send - 1).clamp(min=0)][acc].sum())
        # add_frames writes the new frame's full size and leaves queued frames untouched
        if prev_end is not None:
            live_prev = prev_end[0] >= 0
            assert torch.equal(net.rem[live_prev], prev_end[1][live_prev])
        sp = StepProbe(net)
        with slot_probe([]) as slots:
            advance(net, t, snr, hid)
        sp.detach()
        rec = sp.last
        cap = rec.cap
        for s in slots:
            rin, rout, b = s["rem_in"].double(), s["rem_out"].double(), s["b"].double()
            removed = rin - rout
            q = rin.sum(-1)
            tol = F * 1e-3 + 1e-6 * q.clamp(min=1.0) * F
            assert ((removed.sum(-1) - torch.minimum(b, q)).abs() <= tol).all()
            # frames that received no service may only move by float rounding of the cumsum
            untouched = removed.abs() < 0.5
            if untouched.any():
                max_untouched_drift = max(max_untouched_drift, float(removed[untouched].abs().max()))
            b_served += float(removed.sum())
            ee, rr, ff = (cap >= 0).nonzero(as_tuple=True)
            for a, bb, c, x in zip(ee.tolist(), rr.tolist(), cap[ee, rr, ff].tolist(), removed[ee, rr, ff].tolist()):
                served_per_frame[(a, bb, c)] += x
        for key in _ids(rec.delivered, cap):
            assert abs(served_per_frame[key] - size_of[key]) <= 0.5, (key, served_per_frame[key], size_of[key])
        b_to += float(rec.rem_after_tx[rec.timed].double().sum())
        prev_end = (net.cap.clone(), net.rem.clone())
    b_q = float(net.rem.double().sum())
    assert abs(b_sub - (b_served + b_to + b_q)) <= 1e-6 * b_sub
    assert max_untouched_drift <= 0.13, max_untouched_drift
