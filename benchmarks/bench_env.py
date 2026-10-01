"""Full FleetEnv step time (env + network) with NetSlot vs NetSlotFast, random actions.

usage: python bench_env.py --E 256 --R 16 --impls orig,graph,compile,triton
"""
import argparse
import json
import time

import subprocess
import threading

import torch


class UtilSampler:
    """Samples nvidia-smi GPU utilization (whole GPU, all processes) every 0.5 s during a timing window."""

    def __enter__(self):
        self.v, self.stop = [], threading.Event()
        def run():
            while not self.stop.is_set():
                try:
                    o = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader,nounits'],
                                       capture_output=True, text=True, timeout=5).stdout.split(',')
                    self.v.append((float(o[0]), float(o[1])))
                except Exception:
                    pass
                self.stop.wait(0.5)
        self.th = threading.Thread(target=run, daemon=True); self.th.start(); return self

    def __exit__(self, *a):
        self.stop.set(); self.th.join()

    def summary(self):
        if not self.v:
            return {}
        u = [x[0] for x in self.v]
        return {'gpu_util_mean': round(sum(u) / len(u), 1), 'gpu_util_max': max(u), 'gpu_mem_used_MiB_total': max(x[1] for x in self.v)}

import os  # noqa: E402
import sys  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # without pip install -e .
from isaac_net.core.proto.netsim import NetSlot  # noqa: E402
from isaac_net.core.proto.netsim_fast import NetSlotFast  # noqa: E402
from isaac_net.examples.fleet_task import TASK_SIZES, FleetEnv  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--E", type=int, default=256)
    ap.add_argument("--R", type=int, default=16)
    ap.add_argument("--impls", default="orig,graph,compile,triton")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--out", default="out/bench_env.jsonl")
    a = ap.parse_args()
    dev = "cuda"
    for impl in a.impls.split(","):
        torch.manual_seed(0)
        sizes = TASK_SIZES["T1"]
        net = NetSlot(a.E, a.R, dev, sizes) if impl == "orig" else NetSlotFast(a.E, a.R, dev, sizes, backend=impl)
        env = FleetEnv(a.E, a.R, net, dev)
        env.reset()
        g = torch.Generator(device=dev).manual_seed(1)

        def act():
            vel = torch.rand(a.E, a.R, 2, device=dev, generator=g) * 2 - 1
            p = torch.rand(a.E, a.R, device=dev, generator=g)
            send = (p < 0.15).long() + (p < 0.05).long()
            return vel, send

        for _ in range(5):
            env.step(*act())
        torch.cuda.synchronize()
        n = a.steps if impl != "orig" else max(10, a.steps // 5)
        with UtilSampler() as us:
            t0 = time.time()
            for _ in range(n):
                env.step(*act())
            torch.cuda.synchronize()
            dt = time.time() - t0
        r = {"impl": impl, "E": a.E, "R": a.R, "env_ms_per_step": round(dt / n * 1e3, 3), "steps": n} | us.summary()
        print(json.dumps(r), flush=True)
        with open(a.out, "a") as f:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
