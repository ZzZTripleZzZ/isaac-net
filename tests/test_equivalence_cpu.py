"""The fast L2 engine's eager backend (the ops captured by the `graph` backend) is bitwise identical
to the reference NetSlot on CPU when both see the same random draws. Catches drift between
netsim.py and netsim_fast.py without a GPU."""
import torch

from engine_api import (FIFO_FIELDS, MAC_FIELDS, InjectNoise, Workload, advance, collect, enable_stats, make_fast,
                        make_ref, set_noise, submit)


def test_fast_eager_bitwise_equals_reference_cpu():
    E, R, steps = 4, 6, 60
    torch.manual_seed(0)
    ref = make_ref("L2", E, R, "cpu")
    fast = make_fast(E, R, "cpu", backend="eager", inject=True)
    fast.h.copy_(ref.h)
    enable_stats(ref)
    enable_stats(fast)
    wl = Workload(E, R, "cpu", seed=1, period=15)
    for t in range(steps):
        send, det, hid, snr = wl.inputs(t)
        nz, u = wl.noise()
        submit(ref, t, send, det, hid, snr)
        submit(fast, t, send, det, hid, snr)
        with InjectNoise(nz, u):
            n_r, d_r = advance(ref, t, snr, hid)
        set_noise(fast, nz, u)
        n_f, d_f = advance(fast, t, snr, hid)
        assert torch.equal(n_r, n_f) and torch.equal(d_r, d_f), f"outputs differ at step {t}"
        for n in FIFO_FIELDS + MAC_FIELDS:
            assert torch.equal(getattr(ref, n), getattr(fast, n)), f"{n} differs at step {t}"
    cr, cf = collect(ref), collect(fast)
    assert cr["overflow"] == cf["overflow"]
    for k in cr:
        if k != "overflow":
            assert torch.equal(cr[k], cf[k]), k
    assert cr["delay"].numel() > 50 and cr["x_cls"].numel() > 0     # delivered and timed-out frames
