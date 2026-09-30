"""PPO smoke training (rsl_rl, Isaac Lab 3.0 wrapper) on NetFleetEnv with the network in the loop.

Centralised policy over the env's R robots: obs [E, R*12], actions [E, R*3]. The point is a working
training pipeline and its throughput, not a tuned policy.
usage: python benchmarks\\train_ppo.py --num_envs 1024 --num_robots 16 --rung L2 --backend triton --iters 30
"""
from __future__ import annotations

import argparse
import os
import sys
import time

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(conflict_handler="resolve")
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--num_robots", type=int, default=16)
parser.add_argument("--rung", default="L2", choices=["off", "L0", "L1", "L2"])
parser.add_argument("--backend", default="triton")
parser.add_argument("--iters", type=int, default=30)
add_launcher_args(parser)
args = parser.parse_args()
_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repo root
if _root not in sys.path:
    sys.path.insert(0, _root)                     # isaaclab_net without `pip install -e .`


def main():
    from isaaclab.utils import configclass
    from isaaclab_rl.rsl_rl import (RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg,
                                    RslRlVecEnvWrapper, create_rsl_rl_runner)

    from isaaclab_net.examples.isaac_fleet_env import NetFleetEnv, make_cfg

    @configclass
    class FleetPPOCfg(RslRlOnPolicyRunnerCfg):
        num_steps_per_env = 24
        max_iterations = args.iters
        save_interval = 1000
        experiment_name = "net_fleet"
        actor = RslRlMLPModelCfg(hidden_dims=[256, 128], activation="elu", obs_normalization=True,
                                 distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.8))
        critic = RslRlMLPModelCfg(hidden_dims=[256, 128], activation="elu", obs_normalization=True)
        algorithm = RslRlPpoAlgorithmCfg(value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2,
                                         entropy_coef=0.005, num_learning_epochs=4, num_mini_batches=4,
                                         learning_rate=3.0e-4, schedule="adaptive", gamma=0.99, lam=0.95,
                                         desired_kl=0.01, max_grad_norm=1.0)

    cfg = make_cfg(args.num_envs, args.num_robots, args.rung, device=getattr(args, "device", None) or "cuda:0",
                   backend=args.backend)
    agent = FleetPPOCfg()
    agent.device = cfg.sim.device
    with launch_simulation(cfg, args):
        env = RslRlVecEnvWrapper(NetFleetEnv(cfg), clip_actions=1.0)
        log_dir = os.path.abspath(os.path.join("logs", "rsl_rl", "net_fleet",
                                               f"{args.rung}_{args.backend}_E{args.num_envs}_R{args.num_robots}"))
        runner = create_rsl_rl_runner(env, agent, log_dir=log_dir)
        t0 = time.time()
        runner.learn(num_learning_iterations=args.iters, init_at_random_ep_len=True)
        print(f"TRAIN_DONE {args.iters} iters in {time.time() - t0:.1f} s", flush=True)
        env.close()


if __name__ == "__main__":
    main()
