"""MuJoCo Playground / MJX backend (marker `mjx`).

1. Interop: buffer_callback hands torch the XLA buffers themselves (same device pointer, no copy) and XLA's stream.
2. The network inside the MJX fleet env, stepped by a jitted lax.scan of the Brax-wrapped env (the PPO rollout
   path) with staggered Playground autoresets, equals a replay of the recorded poses, sends, tags, resets and RNG
   states through the torch NetModule driven directly: bitwise, for every NET_OUTPUTS entry, on the same fast
   backend (graph, triton) and against the reference engine for the eager backend.
3. Partial resets: the module resets exactly the envs that the autoreset wrapper restarted, before the step, and a
   reset env starts its episode with AoI = one step and last_cap = 0.
Run: XLA_PYTHON_CLIENT_PREALLOCATE=false python -m pytest -m mjx tests/mjx
"""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("mujoco_playground")
torch = pytest.importorskip("torch")

import jax.numpy as jnp  # noqa: E402

pytestmark = pytest.mark.mjx

E, R, T, EP = 24, 6, 70, 25


def _rollout(level, backend, impl="warp", seed=3):
    from mujoco_playground import wrapper

    from isaac_net.examples.mjx_fleet_env import MJXFleetEnv, default_config
    c = default_config()
    c.num_envs, c.num_robots, c.net_level, c.net_backend, c.net_seed, c.net_record, c.impl = \
        E, R, level, backend, seed, True, impl
    env = MJXFleetEnv(c)
    w = wrapper.wrap_for_brax_training(env, episode_length=EP)
    s = jax.jit(w.reset)(jax.random.split(jax.random.PRNGKey(seed), E))
    # staggered episode counters -> the autoreset wrapper restarts different envs at different steps
    s.info["steps"] = jax.random.randint(jax.random.PRNGKey(7), (E,), 0, EP).astype(jnp.float32)

    def body(s, k):
        a = jax.random.uniform(k, (E, env.action_size), minval=-1.0, maxval=1.0)
        fresh = s.data.time == 0.0
        s = w.step(s, a)
        return s, (s.info["net"], fresh)

    torch.cuda.manual_seed(seed)
    _, (net, fresh) = jax.jit(lambda s, k: jax.lax.scan(body, s, jax.random.split(k, T)))(s, jax.random.PRNGKey(1))
    jax.block_until_ready(net)
    return env, jax.tree.map(np.asarray, net), np.asarray(fresh)


def test_buffer_callback_is_zero_copy_on_xla_stream():
    from jax.experimental.buffer_callback import buffer_callback
    seen = {}

    def cb(ctx, out, x):
        with torch.cuda.stream(torch.cuda.ExternalStream(ctx.stream)):
            xt, ot = torch.from_dlpack(x), torch.from_dlpack(out)
            seen["zero_copy"] = (xt.data_ptr() == x.__cuda_array_interface__["data"][0] and
                                 ot.data_ptr() == out.__cuda_array_interface__["data"][0])
            seen["stream"] = torch.cuda.current_stream().cuda_stream == ctx.stream
            seen["shape"] = tuple(xt.shape)
            ot.copy_(xt * 2)

    f = buffer_callback(cb, jax.ShapeDtypeStruct((3,), jnp.float32), has_side_effect=True,
                        vmap_method="broadcast_all")
    y = jax.jit(jax.vmap(f))(jnp.ones((5, 3)))
    assert np.array_equal(np.asarray(y), np.full((5, 3), 2.0, np.float32))
    assert seen == {"zero_copy": True, "stream": True, "shape": (5, 3)}


@pytest.mark.parametrize("level,backend,ref_backend", [
    ("L2-legacy", "graph", "graph"),
    ("L2-legacy", "triton", "triton"),
    ("L2-legacy", "eager", "reference"),
    ("L0", "graph", "graph"),
    ("L1", "eager", "reference"),
])
def test_in_env_network_equals_direct_replay(level, backend, ref_backend):
    if backend == "triton":
        pytest.importorskip("triton")
    from isaac_net.mjx import NET_OUTPUTS, replay
    env, net, fresh = _rollout(level, backend)
    m = env.net
    assert m.calls == T and len(m.records) == T
    rep = replay(m, backend=ref_backend)
    for t in range(T):
        for k in NET_OUTPUTS:
            got, want = net[k][t], rep[t][k].cpu().numpy()
            assert np.array_equal(got, want.astype(got.dtype)), f"{k} differs at step {t} ({level}/{backend})"
    # resets: exactly the envs the wrapper restarted (data.time == 0 at step start), and some are partial
    for t in range(T):
        ids = m.records[t]["reset_ids"].cpu().numpy()
        assert np.array_equal(ids, np.nonzero(fresh[t])[0]), f"reset envs differ at step {t}"
    partial = [t for t in range(1, T) if 0 < fresh[t].sum() < E]
    assert len(partial) >= 5, partial
    # a reset env starts with the state at reset known (last_cap 0) and AoI = one control step
    f = fresh.astype(bool)
    assert np.all(net["last_cap"][f] == 0)
    assert np.allclose(net["aoi_s"][f], m.step_dt)
    assert net["delivered"].mean() > 0.02 and net["tag_delivered"].sum() > 0


def test_off_level_has_no_network():
    env, net, fresh = _rollout("off", "graph")
    assert env.net is None
    assert all(np.all(v == 0) for v in net.values())
