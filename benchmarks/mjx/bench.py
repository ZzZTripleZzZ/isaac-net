"""Throughput of the MJX fleet env (Playground wrappers, jitted lax.scan rollout) with the network off / on.

    XLA_PYTHON_CLIENT_PREALLOCATE=false python benchmarks/mjx/bench.py --num_envs 1024 --num_robots 16 \
        --level L2-legacy --backend triton --steps 200

Random actions (uniform in [-1,1]), so about 1/3 of robot-steps send nothing, 1/3 small, 1/3 large. The rollout is
the PPO acting path: jit(lax.scan(wrapped_env.step)) over --chunk steps, timed over --steps steps after warm-up.
Also times the torch NetModule alone (submit + step, the same engine the callback runs) for the network cost
without JAX. Appends one JSON line with env steps/s, robot-steps/s and device-wide GPU utilisation / memory
sampled before and during the run (nvidia-smi), which labels the GPU contention of the measurement.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import threading
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

p = argparse.ArgumentParser()
p.add_argument("--num_envs", type=int, default=256)
p.add_argument("--num_robots", type=int, default=16)
p.add_argument("--level", default="L2-legacy", help="off, or a make_engine level (L0, L0DR, L1, L2-legacy, ...)")
p.add_argument("--backend", default="triton")
p.add_argument("--impl", default="warp")
p.add_argument("--steps", type=int, default=200)
p.add_argument("--chunk", type=int, default=20, help="env steps per jitted scan call")
p.add_argument("--episode_length", type=int, default=300)
p.add_argument("--out", default="bench_mjx.jsonl")
p.add_argument("--tag", default="")
args = p.parse_args()


def gpu_query():
    try:
        o = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=20).stdout
        u, m = [float(x) for x in o.strip().splitlines()[0].split(",")]
        return u, m
    except Exception:
        return float("nan"), float("nan")


class Sampler(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.u, self.m, self.stop = [], [], False

    def run(self):
        while not self.stop:
            u, m = gpu_query()
            self.u.append(u)
            self.m.append(m)
            time.sleep(0.5)


def main():
    import jax
    import torch
    from mujoco_playground import wrapper

    from isaac_net.examples.mjx_fleet_env import MJXFleetEnv, default_config

    E, R = args.num_envs, args.num_robots
    u0, m0 = gpu_query()
    c = default_config()
    c.num_envs, c.num_robots, c.net_level, c.net_backend, c.impl = E, R, args.level, args.backend, args.impl
    t0 = time.time()
    env = MJXFleetEnv(c)
    w = wrapper.wrap_for_brax_training(env, episode_length=args.episode_length)
    rec = dict(E=E, R=R, level=args.level, backend=args.backend if env.net is not None else "-", impl=args.impl,
               tag=args.tag, gpu_util_before=u0, gpu_mem_before_mib=m0, jax=jax.__version__)

    def body(s, k):
        a = jax.random.uniform(k, (E, env.action_size), minval=-1.0, maxval=1.0)
        s = w.step(s, a)
        return s, s.reward.mean()

    roll = jax.jit(lambda s, k: jax.lax.scan(body, s, jax.random.split(k, args.chunk)))
    s = jax.jit(w.reset)(jax.random.split(jax.random.PRNGKey(0), E))
    key = jax.random.PRNGKey(1)
    s, r = roll(s, key)                                   # compile + warm-up
    jax.block_until_ready(r)
    rec["startup_s"] = time.time() - t0
    for _ in range(2):
        key, k = jax.random.split(key)
        s, r = roll(s, k)
    jax.block_until_ready(r)
    n = 0
    sm = Sampler()
    sm.start()
    t1 = time.perf_counter()
    while n < args.steps:
        key, k = jax.random.split(key)
        s, r = roll(s, k)
        n += args.chunk
    jax.block_until_ready(r)
    dt = time.perf_counter() - t1
    sm.stop = True
    sm.join()
    rec.update(timed_steps=n, wall_s=dt, env_steps_per_s=n * E / dt, robot_steps_per_s=n * E * R / dt,
               ms_per_step=dt / n * 1e3, gpu_util_during_mean=sum(sm.u) / max(len(sm.u), 1),
               gpu_mem_during_max_mib=max(sm.m) if sm.m else float("nan"), mean_reward=float(r.mean()),
               obs_shape=list(s.obs.shape), obs_finite=bool(jax.numpy.isfinite(s.obs).all()))
    if env.net is not None:
        from isaac_net.isaac import TrafficRequest
        net = env.net.net
        dev = net.dev
        p3 = torch.rand(E, R, 3, device=dev) * 150
        send = torch.randint(0, 3, (E, R), device=dev)
        tag = torch.full_like(send, -1)
        cur = torch.zeros(E, dtype=torch.long, device=dev)
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        for _ in range(20):
            net.submit(None, TrafficRequest(send=send, tag=tag))
            net.step(None, p3, cur_tag=cur)
        torch.cuda.synchronize()
        rec["net_only_ms_per_step"] = (time.perf_counter() - t2) / 20 * 1e3
        rec["torch_peak_alloc_mib"] = torch.cuda.max_memory_allocated() / 2 ** 20
        rec["net_calls"], rec["net_resets"] = env.net.calls, env.net.resets
    print("RESULT " + json.dumps(rec), flush=True)
    with open(args.out, "a") as f:
        f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
