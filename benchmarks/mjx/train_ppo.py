"""PPO smoke run of the MJX fleet env with the network in the loop, on Playground's Brax PPO (JAX).

    XLA_PYTHON_CLIENT_PREALLOCATE=false python benchmarks/mjx/train_ppo.py --num_envs 256 --num_robots 16 \
        --level L2-legacy --backend triton --iters 5

The network steps inside Brax's jitted rollout (lax.scan over the vmapped env step) through NetModuleMJX's
buffer_callback. The eval env is a second MJXFleetEnv with its own network for num_eval_envs envs. Prints one
JSON line (RESULT ...) with the per-eval metrics, wall time and the network's step and reset counts.
"""
from __future__ import annotations

import argparse
import functools
import json
import os
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

p = argparse.ArgumentParser()
p.add_argument("--num_envs", type=int, default=256)
p.add_argument("--num_robots", type=int, default=16)
p.add_argument("--level", default="L2-legacy")
p.add_argument("--backend", default="triton")
p.add_argument("--impl", default="warp")
p.add_argument("--iters", type=int, default=5, help="PPO training steps (each = batch_size x num_minibatches x "
               "unroll_length env steps)")
p.add_argument("--unroll_length", type=int, default=10)
p.add_argument("--num_minibatches", type=int, default=4)
p.add_argument("--episode_length", type=int, default=100)
p.add_argument("--num_eval_envs", type=int, default=128)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--out", default="")
args = p.parse_args()


def main():
    import jax
    from brax.training.agents.ppo import networks as ppo_networks
    from brax.training.agents.ppo import train as ppo
    from mujoco_playground import wrapper

    from isaaclab_net.examples.mjx_fleet_env import MJXFleetEnv, default_config

    def make(n):
        c = default_config()
        c.num_envs, c.num_robots, c.net_level, c.net_backend, c.impl, c.net_seed = \
            n, args.num_robots, args.level, args.backend, args.impl, args.seed
        c.episode_length = args.episode_length
        return MJXFleetEnv(c)

    t0 = time.time()
    env, eval_env = make(args.num_envs), make(args.num_eval_envs)
    build_s = time.time() - t0
    batch = args.num_envs                         # one minibatch = num_envs trajectories of unroll_length
    steps_per_iter = batch * args.num_minibatches * args.unroll_length
    evals, times = [], []

    def progress(step, metrics):
        times.append(time.time())
        evals.append({"step": int(step), **{k: float(v) for k, v in metrics.items()
                                            if k.startswith("eval/episode_") and not k.endswith("_std")}})
        print("EVAL", json.dumps(evals[-1]), flush=True)

    train = functools.partial(
        ppo.train, num_timesteps=steps_per_iter * args.iters, num_evals=args.iters + 1, reward_scaling=1.0,
        episode_length=args.episode_length, normalize_observations=True, action_repeat=1,
        unroll_length=args.unroll_length, num_minibatches=args.num_minibatches, num_updates_per_batch=2,
        discounting=0.97, learning_rate=3e-4, entropy_cost=1e-3, num_envs=args.num_envs, batch_size=batch,
        num_eval_envs=args.num_eval_envs, seed=args.seed,
        network_factory=functools.partial(ppo_networks.make_ppo_networks, policy_hidden_layer_sizes=(128, 128),
                                          value_hidden_layer_sizes=(128, 128)))
    t1 = time.time()
    make_policy, params, _ = train(environment=env, eval_env=eval_env, wrap_env_fn=wrapper.wrap_for_brax_training,
                                   progress_fn=progress)
    wall = time.time() - t1
    rec = dict(E=args.num_envs, R=args.num_robots, level=args.level, backend=args.backend if env.net else "-",
               impl=args.impl, iters=args.iters, env_steps=steps_per_iter * args.iters, build_s=build_s,
               train_wall_s=wall, first_eval_s=(times[0] - t1) if times else None,
               train_env_steps_per_s_after_compile=(steps_per_iter * args.iters / (times[-1] - times[0])
                                                    if len(times) > 1 else None),
               evals=evals, jax=jax.__version__)
    if env.net is not None:
        rec.update(net_calls=env.net.calls, net_resets=env.net.resets, eval_net_calls=eval_env.net.calls,
                   eval_net_resets=eval_env.net.resets)
    print("RESULT " + json.dumps(rec), flush=True)
    if args.out:
        with open(args.out, "a") as f:
            f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
