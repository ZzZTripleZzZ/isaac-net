"""Engine RNG and fast backends of the NR engine (level L2).

CPU
  R1 rng="engine": a run does not depend on the global torch RNG, and is reproducible from the seed
  R2 rng="engine": a partial reset re-seeds only the reset envs (the other envs stay bitwise equal to a run
     without the reset), and E does not change an env's draws
  R3 the NR draws equal an independent scalar reimplementation of the key scheme (nr_rng.py on proto/rng.py)
  R4 the fast backends refuse rng="global" and a CPU device
GPU (marker gpu; tests/nr_equiv.py is the harness, its CLI runs the 300-step versions)
  The short runs start the workload's 100-step traffic cycle at the medium phase (phase_offset), so they cover the
  medium and the all-large burst phases instead of the idle phase (p = 0.03, under 15 frames in 12 steps).
  Expected duration on the lab 4090: G1 + G2 + G7 about 3-5 min (G1 is about 3x its former 12-step cost).
  G1 graph == reference bitwise: every output and every state tensor at every step, UL, UL+DL, three cells, per-robot
     Doppler, round robin / max C/I, traffic models with sub-step arrivals (one and three cells), Rician fading (fixed
     K, K from the LOS state, three cells), frequency-correlated fading (fixed delay spread, per-link delay spread
     with Rician from the LOS state), the QoS scheduler (two and three classes, per-RBG PF), mini-slot grants (2-symbol
     UL + DL, 4-symbol UL with the BSR pipeline), with random partial resets; statistics and counters at the end
  G2 triton, teacher forced: identical decisions from an identical state (UL, UL+DL, LENA-like UL, and the 5G-LENA
     MAC switches the kernel implements: per-RBG PF with frozen averages, TDMA UL retx, previous-PUSCH AMC; Rician
     fading with fixed K and with K from the LOS state; frequency-correlated fading, FCORR = 1 and 2; the QoS scheduler)
  G3 triton, free running: aggregates within a few percent of the reference
  G4 the CUDA (Triton) draws equal the torch path (uniforms bitwise, normals to rounding)
  G5 make_engine("L2", backend="graph" / "triton") through the Isaac NetModule
  G7 graph == reference bitwise with the 5G-LENA MAC switches on (lena_match_v2, per-RBG PF on UL + DL with the BSR
     pipeline, every switch at three cells, the switches without the BSR pipeline, Rician with the BSR pipeline,
     frequency-correlated fading with per-RBG PF, mini-slot grants); triton
     refuses the BSR pipeline and mini-slots
     (test_nr_loadfix.py L10)
"""
import math

import pytest
import torch

import nr_equiv
from isaac_net.core import NRConfig, make_engine
from isaac_net.core import nr_rng as RNG
from isaac_net.core.nr_engine import NRNet

SIZES = (4000.0, 30000.0)


def _run(E, R, steps=6, seed=3, reset=None, disturb=False, cfg=None):
    cfg = cfg or NRConfig(dl=True)
    net = NRNet(E, R, "cpu", SIZES, cfg, seed=seed)
    g = torch.Generator().manual_seed(9)
    outs = []
    z = torch.zeros(E, dtype=torch.long)
    for t in range(steps):
        if reset is not None and t == reset[0]:
            net.reset(reset[1])
        send = torch.randint(0, 3, (E, R), generator=g)
        snr = 25 * torch.rand(E, R, generator=g)
        if disturb:
            torch.rand(1000)                                 # a policy using the global RNG
        net.add_frames(t, send, torch.zeros(E, R, dtype=torch.bool), z, snr)
        net.add_dl_frames(t, send.float() * 2000)
        outs.append(net.step(t, snr, z, full=True))
    return net, outs


def _same(a, b, rows=slice(None)):
    return all(torch.equal(x[k][rows].nan_to_num(-7.0), y[k][rows].nan_to_num(-7.0)) for x, y in zip(a, b) for k in x)


def test_r1_independent_of_global_rng_and_reproducible():
    torch.manual_seed(0)
    a = _run(4, 5)[1]
    torch.manual_seed(123)
    b = _run(4, 5, disturb=True)[1]
    assert _same(a, b)
    c = _run(4, 5, seed=4)[1]
    assert not _same(a, c)


def test_r2_partial_reset_isolation_and_E_independence():
    ids = torch.tensor([1])
    na, a = _run(4, 5, steps=8)
    nb, b = _run(4, 5, steps=8, reset=(4, ids))
    keep = torch.tensor([0, 2, 3])
    assert _same(a, b, keep)
    assert not _same(a, b, ids)
    assert torch.equal(na.h[keep], nb.h[keep]) and not torch.equal(na.h[ids], nb.h[ids])
    # env e's fading draws do not depend on E
    x = NRNet(3, 5, "cpu", SIZES, NRConfig(), seed=5)
    y = NRNet(6, 5, "cpu", SIZES, NRConfig(), seed=5)
    assert torch.equal(x.h, y.h[:3])
    assert torch.equal(x.rng.step_normal(RNG.FADING, 7, 11), y.rng.step_normal(RNG.FADING, 7, 11)[:3])


def _mix32(x):
    x ^= x >> 16
    x = (x * 0x21F0AAAD) & 0xFFFFFFFF
    x ^= x >> 15
    x = (x * 0x735A2D97) & 0xFFFFFFFF
    return x ^ (x >> 15)


def _salt(v):
    return (_mix32(v & 0xFFFFFFFF) + 0x9E3779B9) & 0xFFFFFFFF


def test_r3_hash_matches_scalar_reference():
    seed, E = 2 ** 40 + 77, 3
    r = RNG.NRRng(seed, E, "cpu")
    r.reset_mask(None)
    r.tick_step()
    r.tick_step()
    u = r.step_uniform(RNG.BLER_UL, 5, 7)
    s0 = _mix32(_mix32(seed & 0xFFFFFFFF) ^ _salt(seed >> 32))
    for e in range(E):
        h = _mix32(s0 ^ _salt(e))
        h = _mix32(h ^ _salt(0))                     # episode 0
        h = _mix32(h ^ _salt(RNG.STEP))
        h = _mix32(h ^ _salt(2))                     # counter
        base = _mix32(h ^ _salt((RNG.BLER_UL << 16) | 5))
        for i in range(7):
            x = _mix32(_mix32((base + i * 0x9E3779B9) & 0xFFFFFFFF) ^ base)
            assert float(u[e, i]) == (x >> 8) / 16777216.0


def test_r4_fast_backends_need_engine_rng_and_cuda():
    with pytest.raises(ValueError):
        make_engine("L2", 2, 2, "cpu", NRConfig(), "graph")
    if torch.cuda.is_available():
        with pytest.raises(ValueError, match="rng"):
            make_engine("L2", 2, 2, "cuda", NRConfig(rng="global"), "graph")


# ------------------------------------------------------------------------------------------------------------ GPU
@pytest.mark.gpu
@pytest.mark.parametrize("cfg", ["ul", "ul_dl", "cells3", "ul_doppler", "ul_maxci_pc", "ul_lena", "traffic",
                                 "traffic_c3", "ul_rician", "ul_rician_los", "cells3_rician",
                                 "ul_fcorr", "ul_fcorr_rician", "qos", "qos_rbg", "ul_minislot2", "ul_minislot4"])
def test_g1_graph_bitwise(cfg):
    # steps 25..64 of the cycle: 25 medium steps (p = 0.4), then 15 all-large burst steps (p = 0.9). The reference
    # delivers 228 (ul_maxci_pc) to 12399 (traffic_c3) frames here, against 3 to 3800 for the old 12 idle steps.
    r = nr_equiv.run("graph", cfg, E=8, R=4, steps=40, seed=3, p_reset=0.3, phase_offset=25)
    assert r["resets"] > 0
    assert r["frames_delivered"][0] > 150, r
    assert r["bitwise"], r["first_mismatch"]


@pytest.mark.gpu
@pytest.mark.parametrize("cfg", ["ul", "ul_dl", "ul_lena", "ul_compat", "traffic", "ul_lena_sched", "ul_dl_pf",
                                 "ul_rician", "ul_rician_los", "ul_fcorr", "ul_fcorr_rician", "qos", "qos_rbg"])
def test_g2_triton_teacher_forced(cfg):
    # steps 40..59 of the cycle: 10 medium, then 10 burst steps (about 2000 active robot-steps instead of about 100)
    r = nr_equiv.run("triton", cfg, E=16, R=8, steps=20, seed=3, mode="teacher", p_reset=0.3, phase_offset=40)
    assert r["robot_steps_active"] > 1000
    assert r["robot_steps_mismatch"] <= max(1, r["robot_steps_active"] // 1000), r


@pytest.mark.gpu
@pytest.mark.parametrize("cfg", ["ul_lena_v2", "ul_dl_pf_rbg", "cells3_v2", "ul_lena_sched", "ul_dl_pf",
                                 "ul_dl_pf_rician", "ul_fcorr", "ul_minislot2", "ul_minislot4"])
def test_g7_graph_bitwise_lena_mac_switches(cfg):
    """graph == reference bitwise with the 5G-LENA MAC switches on (BSR pipeline state, per-RBG PF, TDMA retx)."""
    r = nr_equiv.run("graph", cfg, E=8, R=6, steps=40, seed=3, p_reset=0.2, phase_offset=25)
    assert r["resets"] > 0 and r["frames_delivered"][0] > 150
    assert r["bitwise"], r["first_mismatch"]


@pytest.mark.gpu
def test_g3_triton_free_running_aggregates():
    r = nr_equiv.run("triton", "ul", E=32, R=8, steps=40, seed=5, p_reset=0.0)
    a, b = r["frames_delivered"]
    assert a > 100 and abs(a - b) <= 0.05 * a, r
    assert abs(r["mean_delay"][0] - r["mean_delay"][1]) <= 0.1 * r["mean_delay"][0] + 0.05, r
    ok = r["ul_counters"]["tb_ok"]
    assert abs(ok[0] - ok[1]) <= 0.05 * ok[0], r


@pytest.mark.gpu
def test_g4_triton_hash_equals_torch():
    """The CUDA draws (proto/rng_triton kernel) against the torch path on the same device."""
    a = RNG.NRRng(99, 6, "cuda")
    b = RNG.NRRng(99, 6, "cuda")
    b.use_triton = False
    for r in (a, b):
        r.reset_mask(None)
        r.tick_step()
    assert a.use_triton
    assert torch.equal(a.step_uniform(RNG.BLER_DL, 3, 1000), b.step_uniform(RNG.BLER_DL, 3, 1000))
    na, nb = a.step_normal(RNG.FADING, 9, 4096), b.step_normal(RNG.FADING, 9, 4096)
    assert torch.allclose(na, nb, atol=1e-5, rtol=1e-5)
    assert abs(float(na.mean())) < 0.05 and abs(float(na.std()) - 1) < 0.05
    assert math.isfinite(float(na.abs().max()))


@pytest.mark.gpu
@pytest.mark.parametrize("backend", ["graph", "triton"])
def test_g5_netmodule_backend(backend):
    from isaac_net.isaac.net_module import NetModule, TrafficRequest
    E, R = 4, 3
    m = NetModule("L2", E, R, "cuda", NRConfig(msg_sizes=SIZES), backend, radio="engine", seed=1)
    ref = NetModule("L2", E, R, "cuda", NRConfig(msg_sizes=SIZES), "reference", radio="engine", seed=1)
    pos = torch.rand(E, R, 3, device="cuda") * 50
    for t in range(4):
        for mod in (m, ref):
            mod.submit(None, TrafficRequest(torch.ones(E, R, dtype=torch.long, device="cuda")))
        o, oref = m.step(None, pos), ref.step(None, pos)
        if t == 2:
            m.reset(torch.tensor([1], device="cuda"))
            ref.reset(torch.tensor([1], device="cuda"))
    assert o["queue_len"].shape == (E, R)
    if backend == "graph":
        for k in ("newest_cap", "queue_len", "queue_bytes", "sinr_db"):
            assert torch.equal(o[k], oref[k]), k


@pytest.mark.gpu
def test_g6_slot_tap_wrapper_on_graph():
    """The energy wrapper's per-slot tap (core/slot_tap.py, a SINR hook) is captured by graph: its per-robot counters
    and the engine outputs equal the reference bitwise. The fused triton kernel refuses such hooks."""
    from isaac_net.core.energy import EnergyConfig
    E, R = 4, 3
    cfg = NRConfig(msg_sizes=SIZES, energy=EnergyConfig(battery_j=500.0), dl=True)
    ref = make_engine("L2", E, R, "cuda", cfg, "reference", seed=2)
    fast = make_engine("L2", E, R, "cuda", cfg, "graph", seed=2)
    g = torch.Generator(device="cuda").manual_seed(0)
    for t in range(5):
        send = torch.randint(0, 3, (E, R), device="cuda", generator=g)
        snr = 25 * torch.rand(E, R, device="cuda", generator=g)
        oa, ob = [], []
        for eng, o in ((ref, oa), (fast, ob)):
            eng.submit(None, send)
            eng.add_dl_frames(None, send.float() * 1500)
            o.append(eng.step(None, snr))
        for k in oa[0]:
            assert torch.equal(oa[0][k].nan_to_num(-7.0), ob[0][k].nan_to_num(-7.0)), (t, k)
    assert float(oa[0]["energy_cum_j"].sum()) > 0
    tri = make_engine("L2", E, R, "cuda", cfg, "triton", seed=2)
    tri.submit(None, send)
    with pytest.raises(NotImplementedError, match="hooks"):
        tri.step(None, snr)
