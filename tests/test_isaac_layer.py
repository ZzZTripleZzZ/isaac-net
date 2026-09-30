"""Isaac layer contract (no Isaac Lab needed; merged from isaac/demo/tests/test_net_module.py).

1. NetModule with the fast engine (backend eager) == NetModule with the registry engine (backend ref), L2,
   through submit/step, with a partial reset in the middle (same fixed SNR and RNG stream).
2. graph backend (GPU): a partial reset leaves the other envs untouched and the captured graph sees the reset rows.
"""
import pytest
import torch

from isaaclab_net.isaac import NetConfig, NetModule, TrafficRequest

R, SIZES = 6, (4000.0, 30000.0)


def _make(E, backend, dev):
    return NetModule(NetConfig(num_envs=E, num_robots=R, device=dev, rung="L2", backend=backend, pose_chunks=1,
                               msg_sizes=SIZES, step_dt=0.1, slot_dt=0.0025, timeout_steps=20, frame_depth=16))


def _workload(E, t, g, dev):
    rate = 0.05 + 0.35 * (t % 50 < 12)
    send = (torch.rand(E, R, device=dev, generator=g) < rate).long() * torch.randint(1, 3, (E, R), device=dev, generator=g)
    hid = torch.full((E,), t // 30, dtype=torch.long, device=dev)
    det = (send > 0) & (torch.rand(E, R, device=dev, generator=g) < 0.3)
    tag = torch.where(det, hid[:, None].expand(E, R), torch.full_like(send, -1))
    return send, tag, hid


def _eager_vs_ref(E, T, t_reset, dev):
    snr_fixed = 5 + 25 * torch.rand(E, R, device=dev)
    fix = lambda pos, blocked=None: snr_fixed[..., None]
    ref, fast = _make(E, "ref", dev), _make(E, "eager", dev)
    ref.eng.snr_db = fix
    fast.radio.snr_db = fix
    fast.eng.h.copy_(ref.eng.h)
    pos = torch.zeros(E, R, 3, device=dev)
    g = torch.Generator(device=dev).manual_seed(1)
    ids = torch.arange(0, E, 3, device=dev)
    worst = {}
    get_state = torch.cuda.get_rng_state if dev == "cuda" else torch.get_rng_state
    set_state = torch.cuda.set_rng_state if dev == "cuda" else torch.set_rng_state
    for t in range(T):
        if t == t_reset:
            ref.reset(ids)
            fast.reset(ids)
            fast.eng.h[ids] = ref.eng.h[ids]
        send, tag, hid = _workload(E, t, g, dev)
        st = get_state()
        ref.submit(t, TrafficRequest(send=send, tag=tag))
        o1 = ref.step(t, pos, cur_tag=hid)
        set_state(st)
        fast.submit(t, TrafficRequest(send=send, tag=tag))
        o2 = fast.step(t, pos, cur_tag=hid)
        for k in ("newest_cap", "last_cap", "aoi_s", "queue_len", "queue_bytes", "delivered", "tag_delivered"):
            worst[k] = max(worst.get(k, 0.0), (o1[k].float() - o2[k].float()).abs().max().item())
    assert all(v < 1e-3 for v in worst.values()), worst


def test_fast_eager_equals_registry_engine_cpu(seeded):
    _eager_vs_ref(6, 90, 45, "cpu")


@pytest.mark.gpu
def test_fast_eager_equals_registry_engine_gpu(seeded):
    _eager_vs_ref(16, 160, 80, "cuda")


@pytest.mark.gpu
def test_graph_backend_partial_reset(seeded):
    E, T, T_RESET, dev = 16, 60, 30, "cuda"
    snr_fixed = 5 + 25 * torch.rand(E, R, device=dev)
    m = _make(E, "graph", dev)
    m.radio.snr_db = lambda pos, blocked=None: snr_fixed[..., None]
    pos = torch.zeros(E, R, 3, device=dev)
    g = torch.Generator(device=dev).manual_seed(3)
    ids, other = torch.arange(0, E, 2, device=dev), torch.arange(1, E, 2, device=dev)
    delivered = 0.0
    for t in range(T):
        send, tag, hid = _workload(E, t, g, dev)
        if t == T_RESET:
            before = {k: getattr(m.eng, k)[other].clone() for k in ("cap", "rem", "bsr", "wait", "h")}
            m.reset(ids)
            assert all(torch.equal(before[k], getattr(m.eng, k)[other]) for k in before)
            assert bool((m.eng.cap[ids] < 0).all()) and bool((m.eng.rem[ids] == 0).all())
            send[ids] = 0
        m.submit(t, TrafficRequest(send=send, tag=tag))
        o = m.step(t, pos, cur_tag=hid)
        if t == T_RESET:
            assert bool((o["queue_len"][ids] == 0).all()) and bool((o["newest_cap"][ids] == -1).all())
            assert float((o["aoi_s"][ids] - 0.1).abs().max()) < 1e-6
        delivered += o["delivered"].float().mean().item()
    assert delivered > 0


def test_mixin_and_mdp_need_no_isaac():
    from isaaclab_net.isaac.mdp import randomize_network
    from isaaclab_net.isaac.mixins import NetEnvMixin

    class Env(NetEnvMixin):
        num_envs, device = 3, "cpu"
        episode_length_buf = torch.ones(3, dtype=torch.long)

    env = Env()
    env.net_setup(NetConfig(num_envs=3, num_robots=2, device="cpu", rung="L1", backend="ref", pose_chunks=1))
    out = env.net_step(torch.rand(3, 2, 3) * 50, torch.ones(3, 2, dtype=torch.long))
    assert out["queue_len"].shape == (3, 2)
    assert env.net_obs().shape == (3, 2, 4)
    randomize_network(env, torch.tensor([0, 2]), {"pl_exp": (3.0, 3.5)})
    env.net_reset(torch.tensor([1]))
