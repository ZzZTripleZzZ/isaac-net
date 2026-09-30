"""GPU tests (marker `gpu`, skipped without CUDA). Kept small because the GPU is shared (E <= 64).

* graph backend == reference, bitwise, on a short run with identical random draws;
* triton backend statistically close to the reference (aggregate delivered frames, mean delay);
* seeded determinism of the graph and triton backends;
* smoke test of the example env with the fast backends.
"""
import gc

import pytest
import torch

from engine_api import (FIFO_FIELDS, MAC_FIELDS, InjectNoise, Workload, advance, collect, enable_stats, make_fast,
                        make_ref, run, set_noise, state, submit)

pytestmark = pytest.mark.gpu


@pytest.fixture(autouse=True)
def _free_gpu_memory():
    """The GPU is shared: release cached blocks and CUDA-graph pools between tests."""
    yield
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _has_triton():
    try:
        import triton  # noqa: F401
        return True
    except Exception:
        return False


needs_triton = pytest.mark.skipif(not _has_triton(), reason="triton not installed")


def test_graph_bitwise_equals_reference():
    E, R, steps, dev = 16, 16, 40, torch.device("cuda")
    torch.manual_seed(0)
    ref = make_ref("L2", E, R, dev)
    fast = make_fast(E, R, dev, backend="graph", inject=True)
    fast.h.copy_(ref.h)
    enable_stats(ref)
    enable_stats(fast)
    wl = Workload(E, R, dev, seed=1, period=10)
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
    assert all(torch.equal(cr[k], cf[k]) for k in cr if k != "overflow")
    assert cr["delay"].numel() > 100


def _aggregate(net, wl, steps, noise_to=None, inject_ref=False):
    """Run and return (delivered frames, mean delay, timed-out frames)."""
    enable_stats(net)
    for t in range(steps):
        send, det, hid, snr = wl.inputs(t)
        if noise_to is not None or inject_ref:
            nz, u = wl.noise()
        d = net.dev
        submit(net, t, send.to(d), det.to(d), hid.to(d), snr.to(d))
        if noise_to is not None:
            set_noise(net, nz.to(noise_to), u.to(noise_to))
            advance(net, t, snr.to(d), hid.to(d))
        elif inject_ref:
            with InjectNoise(nz, u):
                advance(net, t, snr, hid)
        else:
            advance(net, t, snr.to(d), hid.to(d))
    st = collect(net)
    return st["delay"].numel(), float(st["delay"].mean()), st["x_cls"].numel()


def _close(a, b, rel):
    return abs(a - b) <= rel * max(abs(a), abs(b))


@needs_triton
def test_triton_close_to_reference_same_draws():
    """Same injected draws: triton differs from the reference only by float rounding, so aggregates
    must agree tightly. The reference runs on CPU to keep the shared GPU free."""
    E, R, steps = 32, 16, 60
    torch.manual_seed(0)
    ref = make_ref("L2", E, R, "cpu")
    tri = make_fast(E, R, "cuda", backend="triton", inject=True)
    tri.h.copy_(ref.h)
    a = _aggregate(ref, Workload(E, R, "cpu", seed=2), steps, inject_ref=True)
    b = _aggregate(tri, Workload(E, R, "cpu", seed=2), steps, noise_to="cuda")
    print(f"\nsame draws: reference (delivered, mean delay, timeouts) = {a}, triton = {b}")
    assert a[0] > 1000
    assert _close(a[0], b[0], 0.02), (a, b)
    assert _close(a[1], b[1], 0.05), (a, b)
    assert abs(a[2] - b[2]) <= max(10, 0.1 * a[2]), (a, b)


@needs_triton
@pytest.mark.slow
def test_triton_close_to_reference_own_rng():
    """Independent random streams (triton's in-kernel Philox vs torch): statistical agreement only."""
    E, R, steps = 64, 16, 80
    torch.manual_seed(0)
    ref = make_ref("L2", E, R, "cpu")
    torch.manual_seed(1)
    tri = make_fast(E, R, "cuda", backend="triton")
    a = _aggregate(ref, Workload(E, R, "cpu", seed=3), steps)
    b = _aggregate(tri, Workload(E, R, "cpu", seed=3), steps)
    print(f"\nown RNG: reference (delivered, mean delay, timeouts) = {a}, triton = {b}")
    assert a[0] > 5000
    assert _close(a[0], b[0], 0.03), (a, b)
    assert _close(a[1], b[1], 0.08), (a, b)


@pytest.mark.parametrize("backend", ["graph", pytest.param("triton", marks=needs_triton)])
def test_fast_backend_seeded_determinism(backend):
    def go():
        torch.manual_seed(5)
        net = make_fast(16, 8, "cuda", backend=backend)
        outs = []
        run(net, Workload(16, 8, "cuda", seed=4), 15,
            on_step=lambda t, n, newest, det: outs.append(newest.clone()))
        return outs, state(net)

    (oa, sa), (ob, sb) = go(), go()
    assert all(torch.equal(x, y) for x, y in zip(oa, ob))
    assert all(torch.equal(sa[k], sb[k]) for k in sa)


@pytest.mark.parametrize("backend", ["graph", pytest.param("triton", marks=needs_triton)])
def test_example_env_smoke(backend):
    from isaaclab_net.examples.fleet_task import OBS_DIM, TASK_SIZES, FleetEnv
    E, R, dev = 16, 8, torch.device("cuda")
    torch.manual_seed(0)
    net = make_fast(E, R, dev, backend=backend, sizes=TASK_SIZES["T1"])
    env = FleetEnv(E, R, net, dev)
    env.T = 15
    env.reset()
    n_info = 0
    for _ in range(20):
        vel = torch.rand(E, R, 2, device=dev) * 2 - 1
        send = torch.randint(0, 3, (E, R), device=dev)
        obs, rew, done, info = env.step(vel, send)
        assert obs.shape == (E, R, OBS_DIM) and torch.isfinite(obs).all() and torch.isfinite(rew).all()
        n_info += info is not None
    assert n_info == 1
    assert int((env.last_cap >= 0).sum()) > 0          # frames were delivered
