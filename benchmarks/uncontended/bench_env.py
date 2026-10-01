"""Full step time of the pure-torch fleet task (examples/fleet_task.FleetEnv: env + network) on an idle GPU.

One configuration per process. The network is off (a stub that never delivers), or a make_engine level and
backend behind the task's legacy add_frames / step calls. Random actions (velocity uniform in [-1, 1], send
w.p. 0.15, a third of the sends large), 10 warm-up steps, then --repeats windows of about --window_s seconds
each. Prints one JSON line with the median and min / max of the window means and the GPU state before / during.

usage: python benchmarks/uncontended/bench_env.py --net L2 --backend triton --E 4096 --R 100 --out x.jsonl
  --net off | L0 | L2-legacy | L2 (NR engine, UL only)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench_net import SIZES, Sampler, wait_idle  # noqa: E402

from isaac_net.core import NRConfig, make_engine  # noqa: E402
from isaac_net.core.proto.netsim import TIMEOUT  # noqa: E402
from isaac_net.examples.fleet_task import FleetEnv  # noqa: E402


class NoNet:
    """Network off: nothing is queued and nothing is delivered."""

    def __init__(self, E, R, dev):
        self.E, self.R, self.dev = E, R, dev
        self.newest = torch.full((E, R), -TIMEOUT, dtype=torch.long, device=dev)
        self.det = torch.zeros(E, dtype=torch.bool, device=dev)
        self.q = torch.zeros(E, R, dtype=torch.long, device=dev)

    def reset(self, env_ids=None):
        pass

    def queued(self):
        return self.q

    def add_frames(self, t, send, det, hid, snr):
        pass

    def step(self, t, snr, hid):
        return self.newest, self.det


class Clocked:
    """The task passes its own step counter as t; the engines run on their per-env clocks (t=None)."""

    def __init__(self, eng):
        self.eng = eng

    def reset(self, env_ids=None):
        self.eng.reset(env_ids)

    def queued(self):
        return self.eng.queued()

    def add_frames(self, t, send, det, hid, snr):
        self.eng.add_frames(None, send, det, hid, snr)

    def step(self, t, snr, hid):
        return self.eng.step(None, snr, hid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", required=True)
    ap.add_argument("--backend", default="triton")
    ap.add_argument("--E", type=int, required=True)
    ap.add_argument("--R", type=int, required=True)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--window_s", type=float, default=2.0)
    ap.add_argument("--max_steps", type=int, default=200)
    ap.add_argument("--idle_util", type=float, default=5.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev, E, R = "cuda", a.E, a.R
    rec = {"net": a.net, "backend": a.backend if a.net != "off" else "-", "E": E, "R": R, "robots": E * R,
           "gpu": torch.cuda.get_device_name(), "torch": torch.__version__}
    rec |= wait_idle(a.idle_util)
    torch.manual_seed(0)
    if a.net == "off":
        net = NoNet(E, R, dev)
    else:
        net = Clocked(make_engine(a.net, E, R, dev, NRConfig(msg_sizes=SIZES), a.backend, seed=1))
    env = FleetEnv(E, R, net, dev)
    env.reset()
    g = torch.Generator(device=dev).manual_seed(1)

    def act():
        vel = torch.rand(E, R, 2, device=dev, generator=g) * 2 - 1
        p = torch.rand(E, R, device=dev, generator=g)
        return vel, (p < 0.15).long() + (p < 0.05).long()

    torch.cuda.reset_peak_memory_stats()
    dt = 0.0
    for _ in range(10):
        t1 = time.perf_counter()
        env.step(*act())
        torch.cuda.synchronize()
        dt = time.perf_counter() - t1
    n = max(5, min(a.max_steps, math.ceil(a.window_s / max(dt, 1e-4))))
    s = Sampler()
    s.start()
    wins = []
    for _ in range(a.repeats):
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        for _ in range(n):
            env.step(*act())
        torch.cuda.synchronize()
        wins.append((time.perf_counter() - t1) / n * 1e3)
    s.halt.set()
    s.join()
    med = statistics.median(wins)
    rec |= {"ms_median": round(med, 4), "ms_min": round(min(wins), 4), "ms_max": round(max(wins), 4),
            "spread_pct": round(100 * (max(wins) - min(wins)) / med, 1), "ms_windows": [round(w, 4) for w in wins],
            "steps_per_window": n, "peak_alloc_mib": round(torch.cuda.max_memory_allocated() / 2 ** 20, 1),
            "status": "ok"} | s.summary()
    print(json.dumps(rec), flush=True)
    with open(a.out, "a") as f:
        f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
