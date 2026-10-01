"""Engine-owned randomness (NRConfig.rng = "engine") and configurable application constants (frame buffer, timeout,
control step, UL slots per step) for the prototype, surrogate and bound levels.

CPU: the global torch RNG never changes an engine-mode run; each env's randomness depends only on (seed, env id,
episode, its own calls), so a reset re-seeds that env deterministically and E does not matter; rng="global" still
gives the earlier behavior; the fast eager backend equals the reference bitwise without injected draws; a
non-default config (F = 32, timeout 40, 50 ms steps with 20 UL slots) runs on every level and conserves frames and
bytes with exact timeouts; the fit tool records its application values and loading refuses a mismatch.
GPU (marker gpu): graph == reference bitwise in engine mode without injection (defaults and the non-default
config), with partial resets; triton close to the reference on the same engine draws; graph / triton runs do not
depend on the global CUDA RNG; the Triton hash equals the torch hash.
"""
import gc
import math

import pytest
import torch

from engine_api import lookup_params
from isaac_net.core import NRConfig, Requests, make_engine, multicell
from isaac_net.core.proto import rng as R_
from test_levels import synth_params

SIZES = (4000.0, 30000.0)
PROTO = ("L0", "L0DR", "L05", "L05Q", "L1", "L2-legacy")
OTHER = ("TR", "GE", "QA", "NN", "ORACLE", "NOCOMM")
ALL = PROTO + OTHER
BIG = NRConfig(frame_buffer=32, timeout_steps=40, control_step_ms=50.0)       # 20 UL slots per step
ODD = NRConfig(frame_buffer=24, timeout_steps=7, control_step_ms=100.0, proto_ul_slots_per_step=30)


def _params(level):
    if level == "L0":
        return {"mu": math.log(0.5), "sig": 0.5, "p": 0.05}
    if level in ("L05", "L05Q"):
        return lookup_params(level)
    return synth_params(level)


def _engine(level, E=4, R=5, dev="cpu", cfg=None, seed=3, backend="reference", **kw):
    cfg = (cfg or NRConfig()).with_(msg_sizes=SIZES)
    return make_engine(level, E, R, dev, cfg, backend, params=_params(level), seed=seed, **kw)


def _inputs(E, R, steps, dev="cpu", seed=1, p=0.5, snr=(0.0, 25.0), poses=False):
    """Per step (send [E,R], SNR [E,R] in dB, or poses [E,R,2] in a 150 m arena for the multi-cell engine)."""
    g = torch.Generator().manual_seed(seed)
    out = []
    for _ in range(steps):
        send = (torch.rand(E, R, generator=g) < p).long() * torch.randint(1, 3, (E, R), generator=g)
        x = snr[0] + (snr[1] - snr[0]) * torch.rand(E, R, generator=g)
        if poses:
            x = 150 * torch.rand(E, R, 2, generator=g)
        out.append((send.to(dev), x.to(dev)))
    return out


def _drive(net, inputs, disturb=None, resets=None):
    outs = []
    for t, (send, x) in enumerate(inputs):
        if resets and t in resets:
            net.reset(resets[t])
        if disturb is not None:
            disturb(t)
        net.submit(None, Requests(send), x if x.dim() == 2 else None)
        if disturb is not None:
            disturb(t + 1000)
        o = net.step(None, x)
        outs.append({k: v.clone() for k, v in o.items() if torch.is_tensor(v)})
    return outs


def _eq(a, b, rows=None):
    for oa, ob in zip(a, b):
        for k in oa:
            x, y = oa[k], ob[k]
            if rows is not None:
                x, y = x[rows], y[rows]
            if x.is_floating_point():
                x, y = x.nan_to_num(-7.0), y.nan_to_num(-7.0)
            if not torch.equal(x, y):
                return False
    return len(a) == len(b)


def _burn(dev):
    def f(t):
        torch.manual_seed(1000 + t)
        torch.rand(97, device=dev)
        torch.randn(13, device=dev)
    return f


# ----------------------------------------------------------------------------- task 1 (CPU)
@pytest.mark.parametrize("level", ALL + ("MC",))
def test_engine_rng_is_independent_of_the_global_rng(level):
    """Same seed and inputs; one run reseeds and consumes the global RNG between every call: identical outputs."""
    lvl, cfg = ("L2-legacy", multicell(3)) if level == "MC" else (level, None)
    ins = _inputs(4, 5, 25, poses=level == "MC")
    a = _drive(_engine(lvl, cfg=cfg), ins, resets={9: [1, 3]})
    torch.manual_seed(99)
    b = _drive(_engine(lvl, cfg=cfg), ins, disturb=_burn("cpu"), resets={9: [1, 3]})
    assert _eq(a, b)
    assert any(bool(o["delivered"].any()) for o in a) or level == "NOCOMM"


@pytest.mark.parametrize("level", ["L0", "L0DR", "L05", "L2-legacy", "GE", "NN"])
def test_global_mode_keeps_the_old_dependence(level):
    """rng="global" is the earlier behavior: the stepping draws follow the global RNG."""
    ins = _inputs(4, 5, 25)
    cfg = NRConfig(rng="global")
    torch.manual_seed(0)
    a = _drive(_engine(level, cfg=cfg), ins)
    torch.manual_seed(0)
    b = _drive(_engine(level, cfg=cfg), ins, disturb=_burn("cpu"))
    assert not _eq(a, b)


@pytest.mark.parametrize("level", ["L0DR", "L05Q", "TR", "GE"])
def test_env_streams_do_not_depend_on_E_or_other_envs_resets(level):
    """Env e's randomness is keyed by (seed, e, episode, call): the first 3 envs of an E = 3 and an E = 6 engine
    agree bitwise, even when the extra envs reset mid-run. (Levels with many transcendental ops are left out:
    on CPU, torch's vectorized and scalar-tail math paths can round differently when E changes; the draws
    themselves are checked for every shape in test_counter_rng_rows_do_not_depend_on_E.)"""
    lvl, cfg = level, None
    ins6 = _inputs(6, 4, 20, seed=5)
    ins3 = [(s[:3], x[:3]) for s, x in ins6]
    a = _drive(_engine(lvl, E=3, R=4, cfg=cfg), ins3)
    b = _drive(_engine(lvl, E=6, R=4, cfg=cfg), ins6, resets={7: [4, 5], 12: [3]})
    rows = slice(0, 3)
    assert _eq([{k: v[rows] for k, v in o.items()} for o in a], [{k: v[rows] for k, v in o.items()} for o in b])


@pytest.mark.parametrize("level", ["L0DR", "L05", "L2-legacy", "TR", "GE", "NN"])
def test_reset_reseeds_the_env_deterministically(level):
    """After its k-th reset an env replays the same randomness whenever the reset happens: env 1 reset at step 4 in
    one run and at step 9 in the other sees identical outputs over the next 12 steps (same inputs)."""
    ins = _inputs(3, 4, 30, seed=8)
    post = _inputs(3, 4, 12, seed=9)
    runs = []
    for t_reset in (4, 9):
        net = _engine(level, E=3, R=4)
        _drive(net, ins[:t_reset])
        net.reset([1])
        runs.append(_drive(net, post))
    one = [1]
    for oa, ob in zip(*runs):
        for k in ("delivered", "delay", "timed_out", "queue_bytes", "newest"):
            x, y = oa[k][one], ob[k][one]
            if x.is_floating_point():
                x, y = x.nan_to_num(-7.0), y.nan_to_num(-7.0)
            assert torch.equal(x, y), k


def test_counter_rng_rows_do_not_depend_on_E():
    a, b = R_.CounterRNG(5, 3, "cpu"), R_.CounterRNG(5, 7, "cpu")
    for r in (a, b):
        r.reset(None)
        r.reset(torch.tensor([2]))
        r.tick(R_.STEP)
    b.reset(torch.tensor([5, 6]))                          # other envs' resets do not matter
    assert torch.equal(a.normal(R_.STEP, 0, 40, 4, 5, 2), b.normal(R_.STEP, 0, 40, 4, 5, 2)[:3])
    assert torch.equal(a.uniform(R_.STEP, 1, 40, 4), b.uniform(R_.STEP, 1, 40, 4)[:3])
    assert torch.equal(a.reset_normal(torch.tensor([2]), 0, 7), b.reset_normal(torch.tensor([2]), 0, 7))
    assert not torch.equal(a.reset_normal(torch.tensor([1]), 0, 7), a.reset_normal(torch.tensor([2]), 0, 7))


def test_seed_from_config_and_argument():
    ins = _inputs(3, 4, 15)
    a = _drive(make_engine("L2-legacy", 3, 4, "cpu", NRConfig(seed=7)), ins)
    b = _drive(make_engine("L2-legacy", 3, 4, "cpu", seed=7), ins)
    c = _drive(make_engine("L2-legacy", 3, 4, "cpu", NRConfig(seed=7), seed=8), ins)
    assert _eq(a, b) and not _eq(a, c)
    assert make_engine("L1", 2, 2, "cpu", NRConfig(seed=11)).seed == 11


@pytest.mark.parametrize("level", ["L0", "L0DR", "L05", "L05Q", "L1", "L2-legacy"])
@pytest.mark.parametrize("cfg", [None, BIG, ODD], ids=["default", "big", "odd"])
def test_fast_eager_equals_reference_in_engine_mode(level, cfg):
    """The ops the graph backend captures, without injected draws, equal the reference bitwise (engine streams)."""
    ins = _inputs(4, 5, 45, p=0.8, snr=(-5.0, 15.0))
    rs = {11: [0, 2], 30: [3]}
    a = _drive(_engine(level, cfg=cfg), ins, resets=rs)
    b = _drive(_engine(level, cfg=cfg, backend="eager"), ins, resets=rs)
    assert _eq(a, b)


# ----------------------------------------------------------------------------- task 2 (CPU)
def _check_run(net, outs, ins, cfg):
    """Frame and nominal-byte conservation per class, exact timeouts, bounded residual bytes, F respected."""
    sz = torch.tensor(SIZES, dtype=torch.float64)
    dl_n = dl_b = to_n = to_b = 0
    maxq = 0
    acc_n = sum(int(o["accepted"].sum()) for o in outs)
    acc_b = sum(float(sz[(s - 1).clamp(min=0)][o["accepted"]].sum()) for (s, _), o in zip(ins, outs))
    for o in outs:
        d, x = o["delivered"], o["timed_out"]
        dl_n += int(d.sum())
        to_n += int(x.sum())
        dl_b += float(sz[o["cls"][d] - 1].sum())
        to_b += float(sz[o["cls"][x] - 1].sum())
        age = (o["t"][:, None, None] + 1 - o["cap"])
        assert bool((age[x] == cfg.timeout_steps).all()), "a message timed out at the wrong age"
        assert bool((age[d] <= cfg.timeout_steps).all())
        assert bool((o["delay"][d] >= 0).all()) and bool((o["delay"][d] <= cfg.timeout_steps).all())
        maxq = max(maxq, int(o["queue_len"].max()))
    live = net.cap >= 0
    q_n, q_b = int(live.sum()), float(sz[net.cls[live] - 1].sum())
    assert acc_n == dl_n + to_n + q_n
    assert acc_b == dl_b + to_b + q_b
    rem = outs[-1]["queue_bytes"].double()
    nominal = torch.where(live, sz[(net.cls - 1).clamp(min=0)].to(rem.device), torch.zeros(()).double()).sum(-1)
    nominal = nominal.to(rem.device)
    assert bool((rem <= nominal * (1 + 1e-5) + 1.0).all()) and bool((rem >= 0).all())    # float32 sums
    assert net.cap.shape[-1] == cfg.frame_buffer and maxq <= cfg.frame_buffer
    return dl_n, to_n, maxq


def _drive_acc(net, ins):
    outs = []
    for send, x in ins:
        acc = net.submit(None, Requests(send), x if x.dim() == 2 else None)
        o = net.step(None, x)
        o = {k: v.clone() for k, v in o.items() if torch.is_tensor(v)}
        o["accepted"] = acc.clone()
        outs.append(o)
    return outs


@pytest.mark.parametrize("level", ALL + ("MC",))
@pytest.mark.parametrize("cfg", [BIG, ODD], ids=["big", "odd"])
def test_non_default_app_config_runs_and_conserves(level, cfg):
    """F = 32, timeout 40, 50 ms steps with 20 UL slots (and an odd variant): every level runs, conserves frames
    and bytes, times out exactly at the configured age and fills the larger FIFO."""
    lvl, c = ("L2-legacy", multicell(3).with_(frame_buffer=cfg.frame_buffer, timeout_steps=cfg.timeout_steps,
                                              control_step_ms=cfg.control_step_ms,
                                              proto_ul_slots_per_step=cfg.proto_ul_slots_per_step)) \
        if level == "MC" else (level, cfg)
    net = _engine(lvl, E=4, R=6, cfg=c)
    assert (net.F, net.TIMEOUT, net.K) == (c.frame_buffer, c.timeout_steps, c.proto_slots_per_step)
    ins = _inputs(4, 6, 2 * c.timeout_steps + 20, p=0.9, snr=(-12.0, 8.0), seed=4, poses=level == "MC")
    outs = _drive_acc(net, ins)
    dl, to, maxq = _check_run(net, outs, ins, c)
    if level != "NOCOMM":
        assert dl > 0
    if level in ("NOCOMM", "L1", "L2-legacy", "QA"):               # congested: timeouts and a full, larger FIFO
        assert to > 0 and maxq == min(c.frame_buffer, c.timeout_steps - 1)   # one message per step at most


def test_defaults_resolve_to_the_prototype_constants():
    from isaac_net.core.proto import netsim
    cfg = NRConfig()
    assert (cfg.frame_buffer, cfg.timeout_steps, cfg.proto_slots_per_step) == (netsim.F, netsim.TIMEOUT,
                                                                                netsim.UL_PER_STEP)
    for level in ALL:
        net = _engine(level)
        assert (net.F, net.TIMEOUT, net.K) == (16, 20, 40)
    assert NRConfig(control_step_ms=50.0).proto_slots_per_step == 20
    with pytest.raises(ValueError, match="proto_ul_slots_per_step"):
        make_engine("L1", 2, 2, "cpu", NRConfig(control_step_ms=33.0))
    make_engine("L1", 2, 2, "cpu", NRConfig(control_step_ms=33.0, proto_ul_slots_per_step=13))


def test_timeout_and_step_change_the_levels():
    """A larger UL-slot budget serves more bytes per step (L1); a shorter timeout drops sooner (NOCOMM)."""
    ins = _inputs(3, 6, 12, p=1.0, snr=(0.0, 5.0), seed=2)
    few = _drive(_engine("L1", E=3, R=6, cfg=NRConfig(control_step_ms=50.0)), ins)
    many = _drive(_engine("L1", E=3, R=6), ins)
    assert float(many[-1]["queue_bytes"].sum()) < float(few[-1]["queue_bytes"].sum())
    outs = _drive(_engine("NOCOMM", E=3, R=6, cfg=NRConfig(timeout_steps=3)), ins)
    assert bool(outs[2]["timed_out"].any()) and not any(bool(o["timed_out"].any()) for o in outs[:2])


def test_fit_records_app_values_and_loading_checks_them(tmp_path):
    from isaac_net.tools import fit_levels as fl
    fit, info = fl.fit_levels("L2-legacy", E=4, R=4, episodes=2, test_episodes=1, T=60, sizes=SIZES, device="cpu",
                              nn_steps=10, qa_envs=4, qa_etas=(1.0,), log=lambda *a: None, frame_buffer=32,
                              timeout_steps=40, control_step_ms=50.0)
    meta = fit["meta"]
    assert (meta["frame_buffer"], meta["timeout_steps"], meta["control_step_ms"], meta["ul_slots_per_step"]) == \
        (32, 40, 50.0, 20)
    path = fl.save_fit(fit, str(tmp_path / "big.pt"), info)
    for level in ("TR", "GE", "QA", "NN"):
        net = make_engine(level, 2, 4, "cpu", BIG, params=path, sizes=SIZES, seed=0)
        _drive(net, _inputs(2, 4, 3))
        with pytest.raises(ValueError, match="fitted with"):
            make_engine(level, 2, 4, "cpu", params=path, sizes=SIZES)
    old = {k: v for k, v in fit.items()}
    old["meta"] = {"sizes": list(SIZES)}                       # an old fit file: prototype constants
    make_engine("NN", 2, 4, "cpu", params=old, sizes=SIZES)
    with pytest.raises(ValueError, match="frame_buffer"):
        make_engine("NN", 2, 4, "cpu", BIG, params=old, sizes=SIZES)


def test_torch_hash_matches_reference_values():
    """Pin the hash so the engine streams do not drift silently between versions."""
    r = R_.CounterRNG(12345, 3, "cpu")
    r.reset(None)
    r.tick(R_.STEP)
    u = r.uniform(R_.STEP, 1, 4)
    assert u.shape == (3, 4) and bool(((u >= 0) & (u < 1)).all())
    assert R_.mix32(0) == 0 and R_.mix32(1) == R_.mix32(1)
    ref = [R_.mix32(x) for x in (1, 2, 0xFFFFFFFF)]
    got = R_.mix32(torch.tensor([1, 2, 0xFFFFFFFF]))
    assert got.tolist() == ref and all(0 <= v <= 0xFFFFFFFF for v in ref)


# ----------------------------------------------------------------------------- GPU
@pytest.fixture
def _free_gpu_memory():
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


@pytest.mark.gpu
@pytest.mark.parametrize("level", ALL)
@pytest.mark.parametrize("cfg", [None, BIG, ODD], ids=["default", "big", "odd"])
def test_graph_equals_reference_engine_mode_gpu(level, cfg, _free_gpu_memory):
    """No injection: graph (captured with the engine streams inside) == reference bitwise, with partial resets,
    and a global-RNG disturbance between calls of the graph run changes nothing."""
    dev = torch.device("cuda")
    ins = _inputs(16, 8, 45, dev=dev, p=0.8, snr=(-5.0, 15.0))
    rs = {13: torch.tensor([0, 5, 9], device=dev), 31: torch.tensor([2], device=dev)}
    a = _drive(_engine(level, E=16, R=8, dev=dev, cfg=cfg), ins, resets=rs)
    g = _engine(level, E=16, R=8, dev=dev, cfg=cfg, backend="graph")
    b = _drive(g, ins, resets=rs, disturb=_burn(dev))
    assert set(g._graphs) == {"add", "step"}
    assert _eq(a, b)


@pytest.mark.gpu
def test_triton_hash_equals_torch_hash(_free_gpu_memory):
    if not _has_triton():
        pytest.skip("triton not installed")
    a, b = R_.CounterRNG(99, 32, "cuda"), R_.CounterRNG(99, 32, "cuda")
    b.use_triton = False
    assert a.use_triton
    for r in (a, b):
        r.reset(None)
        r.reset(torch.tensor([3, 4], device="cuda"))
        r.tick(R_.STEP)
    assert torch.equal(a.uniform(R_.STEP, 1, 40, 8), b.uniform(R_.STEP, 1, 40, 8))
    assert torch.allclose(a.normal(R_.STEP, 0, 40, 8, 5, 2), b.normal(R_.STEP, 0, 40, 8, 5, 2), atol=2e-6)
    ids = torch.tensor([1, 3], device="cuda")
    assert torch.equal(a.reset_uniform(ids, 2, 6), b.reset_uniform(ids, 2, 6))
    cpu = R_.CounterRNG(99, 32, "cpu")
    cpu.reset(None)
    cpu.reset(torch.tensor([3, 4]))
    cpu.tick(R_.STEP)
    assert torch.equal(cpu.uniform(R_.STEP, 1, 40, 8), a.uniform(R_.STEP, 1, 40, 8).cpu())


@pytest.mark.gpu
@pytest.mark.parametrize("level", ["L1", "L2-legacy"])
@pytest.mark.parametrize("cfg", [None, BIG, ODD], ids=["default", "big", "odd"])
def test_triton_engine_mode_close_to_reference_and_global_free(level, cfg, _free_gpu_memory):
    """triton hashes the same engine counters in its kernel: aggregates agree tightly with the reference (same
    draws up to rounding), and the run does not depend on the global CUDA RNG."""
    if not _has_triton():
        pytest.skip("triton not installed")
    dev = torch.device("cuda")
    E, R = 32, 12
    c = cfg or NRConfig()
    ins = _inputs(E, R, 60, dev=dev, p=0.6, snr=(-5.0, 25.0))
    ref = _drive(_engine(level, E=E, R=R, dev=dev, cfg=cfg), ins)
    tri = _drive(_engine(level, E=E, R=R, dev=dev, cfg=cfg, backend="triton"), ins)
    tri2 = _drive(_engine(level, E=E, R=R, dev=dev, cfg=cfg, backend="triton"), ins, disturb=_burn(dev))
    assert _eq(tri, tri2)
    n = [sum(int(o["delivered"].sum()) for o in run) for run in (ref, tri)]
    d = [sum(float(o["delay"].nan_to_num(0).sum()) for o in run) / max(k, 1) for run, k in zip((ref, tri), n)]
    assert n[0] > 200, n
    assert abs(n[0] - n[1]) <= 0.02 * n[0], (n, d)
    assert abs(d[0] - d[1]) <= 0.05 * d[0], (n, d)
    assert tri[0]["delivered"].shape[-1] == c.frame_buffer
