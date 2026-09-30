"""Throughput benchmark for NetFleetEnv: one (E, R, rung) configuration per process.

Usage (from C:\\isaac5g\\demo with the Isaac venv active):
    python benchmarks\bench.py --num_envs 256 --num_robots 16 --rung L2 --backend graph --steps 300 --out results.jsonl
Random actions (uniform in [-1,1]), so about 1/3 of robot-steps send nothing, 1/3 small, 1/3 large.
Appends one JSON line with env steps/s, robot-steps/s, GPU memory and device-wide GPU utilisation.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(conflict_handler="resolve")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--num_robots", type=int, default=16)
parser.add_argument("--rung", default="L2", choices=["off", "L0", "L1", "L2"])
parser.add_argument("--steps", type=int, default=300)
parser.add_argument("--warmup", type=int, default=50)
parser.add_argument("--max_time", type=float, default=120.0, help="stop the timed loop after this many seconds")
parser.add_argument("--out", default="results.jsonl")
parser.add_argument("--backend", default="graph", choices=["ref", "eager", "graph", "compile", "triton"],
                    help="L2 engine backend (ignored for off/L0/L1, which use the ref engine)")
add_launcher_args(parser)
args = parser.parse_args()

_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repo root
if _root not in sys.path:
    sys.path.insert(0, _root)                     # isaaclab_net without `pip install -e .`


def gpu_query():
    """Device-wide (all processes, incl. WSL jobs) utilisation % and used MiB."""
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        u = pynvml.nvmlDeviceGetUtilizationRates(h).gpu
        m = pynvml.nvmlDeviceGetMemoryInfo(h).used / 2**20
        return float(u), float(m)
    except Exception:
        try:
            smi = r"C:\Windows\System32\nvidia-smi.exe" if os.name == "nt" else "nvidia-smi"
            o = subprocess.run([smi,
                                "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=20).stdout
            u, m = [float(x) for x in o.strip().split(",")]
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
    import torch
    from isaaclab_net.examples.isaac_fleet_env import NetFleetEnv, make_cfg

    u_before, m_before = gpu_query()
    cfg = make_cfg(args.num_envs, args.num_robots, args.rung, device=getattr(args, "device", None) or "cuda:0",
                   backend=args.backend)
    rec = dict(E=args.num_envs, R=args.num_robots, rung=args.rung, steps=args.steps,
               backend=args.backend if args.rung == "L2" else ("ref" if args.rung != "off" else "-"),
               gpu_util_before=u_before, gpu_mem_before_mib=m_before)
    with launch_simulation(cfg, args):
        t0 = time.time()
        env = NetFleetEnv(cfg)
        rec["startup_s"] = time.time() - t0
        env.reset()
        E, A = env.num_envs, cfg.action_space
        dev = env.device
        with torch.inference_mode():
            for _ in range(args.warmup):
                env.step(2 * torch.rand(E, A, device=dev) - 1)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            s = Sampler()
            s.start()
            t0 = time.perf_counter()
            rsum = torch.zeros((), device=dev)
            n = 0
            while n < args.steps:
                obs, rew, term, trunc, extras = env.step(2 * torch.rand(E, A, device=dev) - 1)
                rsum += rew.mean()
                n += 1
                if n % 10 == 0:
                    torch.cuda.synchronize()
                    if time.perf_counter() - t0 > args.max_time:
                        break
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            s.stop = True
            s.join()
        rec.update(
            timed_steps=n,
            wall_s=dt,
            env_steps_per_s=n * E / dt,
            robot_steps_per_s=n * E * args.num_robots / dt,
            iter_per_s=n / dt,
            torch_peak_alloc_mib=torch.cuda.max_memory_allocated() / 2**20,
            torch_reserved_mib=torch.cuda.memory_reserved() / 2**20,
            gpu_util_during_mean=sum(s.u) / max(len(s.u), 1),
            gpu_mem_during_max_mib=max(s.m) if s.m else float("nan"),
            mean_reward=float(rsum) / n,
            obs_shape=list(obs["policy"].shape),
            obs_finite=bool(torch.isfinite(obs["policy"]).all()),
        )
        if env.net is not None:
            # isolated network cost: submit + step on the live module (state keeps evolving; harmless at the end)
            from isaaclab_net.isaac import TrafficRequest
            with torch.inference_mode():
                p3 = torch.cat([env._pos_radio(), torch.full((E, args.num_robots, 1), 0.5, device=dev)], -1)
                send = torch.randint(0, 3, (E, args.num_robots), device=dev)
                tag = torch.full_like(send, -1)
                cur = torch.zeros(E, dtype=torch.long, device=dev)
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                for _ in range(20):
                    env.net.submit(None, TrafficRequest(send=send, tag=tag))
                    env.net.step(None, p3, cur_tag=cur)
                torch.cuda.synchronize()
                rec["net_only_ms_per_step"] = (time.perf_counter() - t1) / 20 * 1e3
            o = env.net_out
            rec["net_last_step"] = dict(delivered_frac=float(o["delivered"].float().mean()),
                                        mean_queue=float(o["queue_len"].float().mean()),
                                        mean_aoi_s=float(o["aoi_s"].mean()),
                                        mean_snr_db=float(o["sinr_db"].mean()))
        print("RESULT " + json.dumps(rec), flush=True)
        with open(args.out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        env.close()


if __name__ == "__main__":
    main()
