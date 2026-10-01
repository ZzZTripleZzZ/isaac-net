"""Isaac layer contract without Isaac Lab (ported from isaac/demo/tests/test_net_module.py onto make_engine).

1. The Isaac-side NetModule equals the reference engine of make_engine driven directly with the same SNR, sends,
   tags and RNG stream, bitwise, through a mid-run partial reset: every level the mixin offers (L0, L0DR, L1,
   L2-legacy on the fast eager backend; the NR engine L2 on reference), on CPU and on the GPU. The freshness
   outputs (last_cap, AoI) and tag_delivered are checked against values rebuilt from the reference outputs.
2. graph (bitwise, injected draws) and triton (invariants) through NetModule on the GPU, with partial resets.
3. The radio: the engine radio's SNR for the same shadowing field; pose-chunk interpolation; blockage; DR
   parameters that survive a reset.
4. MessageHistory delivers the first capture of an episode; the mixin, mdp terms and the deprecated NetConfig
   alias work without Isaac Lab. Observation selection and IsaacNetCfg plumbing: tests/test_isaac_config.py.
"""
import math

import pytest
import torch

from isaac_net import NRConfig, Requests, make_engine
from isaac_net.core.proto.netsim import Radio
from isaac_net.isaac import (IsaacRadio, MessageHistory, NetConfig, NetModule, ParamRanges, TrafficRequest,
                                segment_sphere_blocked)

R, SIZES = 6, (4000.0, 30000.0)
CFG = NRConfig(msg_sizes=SIZES)


def _has_triton():
    try:
        import triton  # noqa: F401
        return True
    except Exception:
        return False


def _workload(E, t, g, dev):
    rate = 0.05 + 0.35 * (t % 50 < 12)
    send = (torch.rand(E, R, device=dev, generator=g) < rate).long() * torch.randint(1, 3, (E, R), device=dev,
                                                                                     generator=g)
    hid = torch.full((E,), t // 30, dtype=torch.long, device=dev)
    det = (send > 0) & (torch.rand(E, R, device=dev, generator=g) < 0.3)
    tag = torch.where(det, hid[:, None].expand(E, R), torch.full_like(send, -1))
    cur = torch.where(torch.arange(E, device=dev) % 4 == 3, torch.full_like(hid, -1), hid)   # some envs: no tag
    return send, tag, cur


def _expected_snr(net, prev, valid, pos):
    """The module's radio contract, rebuilt here: mean over chunks of the best-cell SNR at interpolated poses."""
    p0 = torch.where(valid[:, None, None], prev, pos)
    C = net.pose_chunks
    return sum(net.radio.snr_db(p0 + ((c + 0.5) / C) * (pos - p0)).max(-1).values / C for c in range(C))


def _module_vs_reference(level, backend, dev, E=12, T=80, t_reset=37, seed=11):
    dev = torch.device(dev)
    params = {"mu": math.log(0.5), "sig": 0.5, "p": 0.05} if level == "L0" else None
    net = NetModule(level, E, R, dev, CFG, backend, pose_chunks=2, params=params, seed=seed)
    ref = make_engine(level, E, R, dev, CFG, "reference", params=params, seed=seed)
    g = torch.Generator(device=dev).manual_seed(1)
    pos = torch.rand(E, R, 3, device=dev, generator=g) * 150
    prev, valid = torch.zeros_like(pos), torch.zeros(E, dtype=torch.bool, device=dev)
    last = torch.zeros(E, R, dtype=torch.long, device=dev)
    ids = torch.arange(0, E, 3, device=dev)
    get_state = torch.cuda.get_rng_state if dev.type == "cuda" else torch.get_rng_state
    set_state = torch.cuda.set_rng_state if dev.type == "cuda" else torch.set_rng_state
    delivered = tagged = 0
    for t in range(T):
        if t == t_reset:
            net.reset(ids)
            ref.reset(ids)
            valid[ids] = False
            last[ids] = 0
        send, tag, cur = _workload(E, t, g, dev)
        pos = (pos + 2.0 * torch.randn(E, R, 3, device=dev, generator=g)).clamp(0, 150)
        snr = _expected_snr(net, prev, valid, pos)
        st = get_state()
        net.submit(None, TrafficRequest(send=send, tag=tag))
        o = net.step(None, pos, cur_tag=cur)
        set_state(st)
        ref.submit(None, Requests(send, tag >= 0, tag.max(-1).values.clamp(min=0)))
        newest, det_env = ref.step(None, snr, cur.clamp(min=0))
        prev, valid = pos.clone(), torch.ones_like(valid)
        last = torch.maximum(last, newest)
        clock = ref.clock
        assert torch.equal(o["sinr_db"], snr), f"SNR differs at step {t}"
        assert torch.equal(o["newest_cap"], newest), f"newest differs at step {t}"
        assert torch.equal(o["tag_delivered"], det_env & (cur >= 0)), f"tag_delivered differs at step {t}"
        assert torch.equal(o["queue_len"], ref.queued()), f"queue differs at step {t}"
        assert torch.equal(o["last_cap"], last)
        assert torch.equal(o["t"] + 1, clock) and torch.equal(net.clock, clock)
        assert torch.allclose(o["aoi_s"], (clock[:, None] - last).float() * 0.1)
        if t == t_reset:
            assert bool((o["t"][ids] == 0).all()) and bool((o["aoi_s"][ids] - 0.1).abs().max() < 1e-6)
        delivered += int(o["delivered"].sum())
        tagged += int(o["tag_delivered"].sum())
    assert delivered > 0 and tagged > 0


LEVELS_CPU = [("L0", "eager"), ("L0DR", "eager"), ("L1", "eager"), ("L2-legacy", "eager"), ("L2", "reference")]


@pytest.mark.parametrize("level,backend", LEVELS_CPU)
def test_module_equals_reference_engine_cpu(level, backend, seeded):
    _module_vs_reference(level, backend, "cpu")


@pytest.mark.gpu
@pytest.mark.parametrize("level,backend", LEVELS_CPU)
def test_module_equals_reference_engine_gpu(level, backend, seeded):
    if level == "L2":      # the NR reference engine is launch-bound on a GPU: keep the run short
        _module_vs_reference(level, backend, "cuda", E=16, T=40, t_reset=21)
    else:
        _module_vs_reference(level, backend, "cuda", E=16, T=60, t_reset=31)


@pytest.mark.gpu
def test_graph_module_bitwise_with_injected_draws(seeded):
    """graph backend through NetModule == reference engine, bitwise, with the same injected slot draws."""
    from engine_api import InjectNoise
    from isaac_net.core.proto.netsim import S, UL_PER_STEP
    E, T, t_reset, dev = 16, 60, 29, torch.device("cuda")
    cfg = CFG.with_(rng="global")          # InjectNoise feeds the reference through the global-RNG draw sites
    net = NetModule("L2-legacy", E, R, dev, cfg, "graph", pose_chunks=1, seed=5, inject=True)
    ref = make_engine("L2-legacy", E, R, dev, cfg, "reference", seed=5)
    g = torch.Generator(device=dev).manual_seed(2)
    pos = torch.rand(E, R, 3, device=dev, generator=g) * 150
    ids = torch.arange(1, E, 4, device=dev)
    for t in range(T):
        if t == t_reset:
            net.reset(ids)
            ref.reset(ids)
        send, tag, cur = _workload(E, t, g, dev)
        nz = torch.randn(UL_PER_STEP, E, R, S, 2, device=dev, generator=g)
        u = torch.rand(UL_PER_STEP, E, R, device=dev, generator=g)
        snr = net.radio.snr_db(pos).max(-1).values
        net.eng.set_noise(nz, u)
        net.submit(None, TrafficRequest(send=send, tag=tag))
        o = net.step(None, pos, cur_tag=cur)
        ref.submit(None, Requests(send, tag >= 0, tag.max(-1).values.clamp(min=0)))
        with InjectNoise(nz, u):
            newest, det_env = ref.step(None, snr, cur.clamp(min=0))
        assert torch.equal(o["newest_cap"], newest) and torch.equal(o["tag_delivered"], det_env & (cur >= 0))
        for n in ("cap", "rem", "bsr", "sr_t", "avg", "olla", "wait", "hcnt", "h"):
            assert torch.equal(getattr(net.eng, n), getattr(ref, n)), f"{n} differs at step {t}"


@pytest.mark.gpu
@pytest.mark.parametrize("level,backend", [("L2-legacy", "graph"), ("L2-legacy", "triton"), ("L1", "triton")])
def test_fast_backend_partial_reset(level, backend, seeded):
    """A partial reset leaves the other envs untouched, and the reset envs come back empty with AoI = 1 step."""
    if backend == "triton" and not _has_triton():
        pytest.skip("triton not installed")
    E, T, t_reset, dev = 16, 60, 30, torch.device("cuda")
    m = NetModule(level, E, R, dev, CFG, backend, pose_chunks=2, seed=3)
    g = torch.Generator(device=dev).manual_seed(3)
    pos = torch.rand(E, R, 3, device=dev, generator=g) * 150
    ids, other = torch.arange(0, E, 2, device=dev), torch.arange(1, E, 2, device=dev)
    names = ["cap", "rem"] + (["bsr", "wait", "h"] if level == "L2-legacy" else [])
    delivered = 0.0
    for t in range(T):
        send, tag, cur = _workload(E, t, g, dev)
        if t == t_reset:
            before = {k: getattr(m.eng, k)[other].clone() for k in names}
            clock_before = m.clock[other].clone()
            m.reset(ids)
            assert all(torch.equal(before[k], getattr(m.eng, k)[other]) for k in names)
            assert torch.equal(m.clock[other], clock_before) and bool((m.clock[ids] == 0).all())
            assert bool((m.eng.cap[ids] < 0).all()) and bool((m.eng.rem[ids] == 0).all())
            send[ids] = 0
        m.submit(None, TrafficRequest(send=send, tag=tag))
        o = m.step(None, pos, cur_tag=cur)
        if t == t_reset:
            assert bool((o["queue_len"][ids] == 0).all()) and bool((o["newest_cap"][ids] == -1).all())
            assert float((o["aoi_s"][ids] - 0.1).abs().max()) < 1e-6
        delivered += o["delivered"].float().mean().item()
    assert delivered > 0


def test_radio_matches_engine_radio(seeded):
    """Default parameters, one gNB at the origin, 2-D poses: the engine radio's SNR for the same field."""
    E = 5
    rad = IsaacRadio(E, "cpu", [(0.0, 0.0, 0.0)], ParamRanges.from_config(NRConfig()), seed=1)
    eng = Radio(E, "cpu", generator=torch.Generator().manual_seed(2))
    rad.k.copy_(eng.k)
    rad.phi.copy_(eng.phi)
    pos = torch.rand(E, R, 2) * 150
    assert torch.allclose(rad.snr_db(pos)[..., 0], eng.snr_db(pos), atol=1e-4)
    assert ParamRanges.from_config(NRConfig()) == ParamRanges()


def test_radio_reset_blockage_and_params(seeded):
    E = 6
    rad = IsaacRadio(E, "cpu", [(0.0, 0.0, 6.0), (150.0, 150.0, 6.0)], ParamRanges(), seed=4)
    pos = torch.rand(E, R, 3) * 150
    s0 = rad.snr_db(pos)
    blk = torch.zeros(E, R, 2, dtype=torch.bool)
    blk[:, 0, 1] = True
    assert torch.allclose(rad.snr_db(pos, blk)[:, 0, 1], s0[:, 0, 1] - 20.0)
    rad.sample_params([1, 4], {"pl_exp": (3.0, 3.2), "noise_dbm": (-95.0, -85.0)})
    p = rad.params()
    assert bool(((p["pl_exp"][[1, 4]] >= 3.0) & (p["pl_exp"][[1, 4]] <= 3.2)).all())
    assert bool((p["pl_exp"][[0, 2, 3, 5]] == 3.5).all())
    k_before = rad.k.clone()
    rad.reset(torch.tensor([1]))
    assert torch.equal(rad.params()["pl_exp"], p["pl_exp"]), "reset must not touch parameters"
    assert not torch.equal(rad.k[1], k_before[1]) and torch.equal(rad.k[[0, 2, 3, 4, 5]], k_before[[0, 2, 3, 4, 5]])
    with pytest.raises(KeyError):
        rad.set_params(None, bg_load=0.5)
    # a robot directly between the gNB and another robot blocks it
    a = torch.zeros(1, 1, 1, 3)
    b = torch.tensor([[[[10.0, 0.0, 0.0]]]])
    c = torch.tensor([[[5.0, 0.2, 0.0], [5.0, 3.0, 0.0]]])
    assert bool(segment_sphere_blocked(a, b, c, 0.3)[0, 0, 0])
    assert not bool(segment_sphere_blocked(a, b, c[:, 1:], 0.3)[0, 0, 0])


def test_message_history_first_capture():
    """The first delivered capture (env clock 0) of an episode reaches the receiver (demo fix)."""
    h = MessageHistory(1, 2, 1, history_len=32, device="cpu")
    h.reset(torch.tensor([0]), torch.zeros(1, 2, 1))
    h.push(torch.tensor([0]), torch.tensor([[[7.0], [8.0]]]))
    seen = h.update(torch.tensor([[0, -1]]))
    assert seen[0, :, 0].tolist() == [7.0, 0.0]


def test_mixin_mdp_and_netconfig_need_no_isaac(seeded):
    from isaac_net.isaac.mdp import randomize_network
    from isaac_net.isaac.mixins import NetEnvMixin

    class Env(NetEnvMixin):
        num_envs, device = 3, "cpu"
        episode_length_buf = torch.ones(3, dtype=torch.long)

    env = Env()
    for level, backend in [("L1", "reference"), ("L2-legacy", "eager"), ("L2", "reference"), ("ORACLE", "reference")]:
        env.net_setup(level, 2, NRConfig(), backend, pose_chunks=1)
        out = env.net_step(torch.rand(3, 2, 3) * 50, torch.ones(3, 2, dtype=torch.long))
        assert out["queue_len"].shape == (3, 2) and env.net_obs().shape == (3, 2, 4)
        randomize_network(env, torch.tensor([0, 2]), {"pl_exp": (3.0, 3.5)})
        env.net_reset(torch.tensor([1]))
    env.net_setup("off")
    assert env.net_step(torch.zeros(3, 2, 3), torch.ones(3, 2, dtype=torch.long)) is None
    env.R = 2
    assert torch.equal(env.net_obs(), torch.zeros(3, 2, 4))
    # the deprecated NetConfig: rung "L2" is the slot-level NetSlot, i.e. level "L2-legacy"; no defaults of its own
    with pytest.warns(DeprecationWarning):
        nc = NetConfig(num_envs=3, num_robots=2, device="cpu", rung="L2", backend="eager", pose_chunks=1,
                       msg_sizes=(1500.0, 12000.0), gnb_pos=((0.0, 0.0, 6.0),))
    m = NetModule(nc)
    assert m.level == "L2-legacy" and m.config.msg_sizes == (1500.0, 12000.0) and m.gnb.tolist() == [[0.0, 0.0, 6.0]]
    with pytest.warns(DeprecationWarning):
        m = NetModule(NetConfig(num_envs=3, num_robots=2, device="cpu"))
    assert m.level == "L2-legacy" and m.backend == "reference" and m.config == NRConfig()
    assert m.pose_chunks == 4 and m.gnb.tolist() == [[0.0, 0.0, 0.0]]
    with pytest.raises(ValueError):
        NetModule("L2-legacy", 2, 2, "cpu", NRConfig(n_cells=3, cell_layout="hex"))
