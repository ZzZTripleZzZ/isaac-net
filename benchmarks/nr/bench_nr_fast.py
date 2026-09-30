"""NR engine (level L2) backends: ms per control step and GPU memory, eager reference vs graph vs triton.

One row per (backend, E, R, config): submit + step with dict outputs, synchronized, over `steps` timed control steps
after `warm` warm-up steps (the graph / triton warm-up includes capturing the step's CUDA graphs). Peak memory is
torch.cuda.max_memory_allocated above the baseline before the engine was built, CUDA-graph pools included. The GPU
utilization reported by nvidia-smi (whole device, all processes) is sampled before and after each row, so every
row carries its own contention label.

Configs: ul (NRConfig(): mu = 1, 20 MHz, 13 RBGs, 16 HARQ, EESM, UL only), ul_dl (the same with the downlink),
ul_c3 / ul_dl_c3 (multicell(3) and multicell(3, dl=True): three cells with interference and handover; graph and
reference only).

Usage (lab box): python benchmarks/nr/bench_nr_fast.py --out benchmarks/nr/results/bench_nr_fast.csv
  [--E 256 1024 4096] [--R 16 64 100] [--cfgs ul ul_dl ul_c3 ul_dl_c3] [--backends reference graph triton]
"""
from __future__ import annotations

import argparse
import csv
import gc
import os
import subprocess
import time

import torch

from isaaclab_net.core import NRConfig, Requests, make_engine, multicell

SIZES = (4000.0, 30000.0)
CFGS = {"ul": lambda: NRConfig(), "ul_dl": lambda: NRConfig(dl=True),
        "ul_c3": lambda: multicell(3), "ul_dl_c3": lambda: multicell(3, dl=True)}
FIELDS = ["backend", "cfg", "cells", "dl", "E", "R", "robots", "ms_per_step", "ms_min", "steps_timed", "warm_s",
          "peak_mib", "gpu_util_before", "gpu_util_after", "gpu_mem_used_mib", "status"]


def gpu_util():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout.strip().split(",")
        return int(out[0]), int(out[1])
    except Exception:
        return -1, -1


class Drive:
    """Steady load: every robot sends a frame w.p. 0.3 (class 1 or 2) and, with a downlink, receives 6 kB w.p. 0.3;
    SNR per robot drifting around a draw in [0, 25] dB; poses random-walking in the 150 m arena (several cells)."""

    def __init__(self, cfg, E, R, dev, seed=0):
        self.cfg, self.E, self.R, self.dev = cfg, E, R, dev
        g = torch.Generator(device=dev).manual_seed(seed)
        self.g = g
        self.snr = 25 * torch.rand(E, R, device=dev, generator=g)
        self.pos = 150 * torch.rand(E, R, 2, device=dev, generator=g)

    def __call__(self, eng):
        E, R, g, d = self.E, self.R, self.g, self.dev
        u = torch.rand(E, R, device=d, generator=g)
        send = (u < 0.3).long() * (1 + (u < 0.1).long())
        eng.submit(None, Requests(send))
        if self.cfg.dl:
            eng.add_dl_frames(None, torch.where(torch.rand(E, R, device=d, generator=g) < 0.3, 6000.0, 0.0))
        if self.cfg.n_cells > 1:
            self.pos = (self.pos + 2 * torch.rand(E, R, 2, device=d, generator=g) - 1).clamp(0, 150)
            return eng.step(None, self.pos)
        self.snr = (self.snr + 0.5 * torch.randn(E, R, device=d, generator=g)).clamp(-5, 30)
        return eng.step(None, self.snr)


def bench(backend, cfg_name, E, R, warm, steps, max_s):
    dev = torch.device("cuda")
    cfg = CFGS[cfg_name]().with_(msg_sizes=SIZES)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    u0, _ = gpu_util()
    row = {"backend": backend, "cfg": cfg_name, "cells": cfg.n_cells, "dl": int(cfg.dl), "E": E, "R": R,
           "robots": E * R, "gpu_util_before": u0}
    try:
        eng = make_engine("L2", E, R, dev, cfg, backend, seed=1)
        drv = Drive(cfg, E, R, dev)
        t0 = time.time()
        for _ in range(warm):
            drv(eng)
        torch.cuda.synchronize()
        row["warm_s"] = round(time.time() - t0, 2)
        times = []
        t_start = time.time()
        for _ in range(steps):
            t1 = time.perf_counter()
            drv(eng)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t1)
            if time.time() - t_start > max_s:
                break
        row.update(ms_per_step=round(1e3 * sum(times) / len(times), 3), ms_min=round(1e3 * min(times), 3),
                   steps_timed=len(times), status="ok")
        row["peak_mib"] = round((torch.cuda.max_memory_allocated() - base) / 2 ** 20, 1)
        del eng, drv
    except torch.cuda.OutOfMemoryError as e:
        row["status"] = "oom: " + str(e).split("\n")[0][:80]
    except NotImplementedError as e:
        row["status"] = "n/a: " + str(e)[:80]
    u1, mem = gpu_util()
    row.update(gpu_util_after=u1, gpu_mem_used_mib=mem)
    return row


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--E", type=int, nargs="+", default=[256, 1024, 4096])
    ap.add_argument("--R", type=int, nargs="+", default=[16, 64, 100])
    ap.add_argument("--cfgs", nargs="+", default=["ul", "ul_dl", "ul_c3", "ul_dl_c3"])
    ap.add_argument("--backends", nargs="+", default=["reference", "graph", "triton"])
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--ref_steps", type=int, default=3)
    ap.add_argument("--max_s", type=float, default=120.0)
    ap.add_argument("--out", default="benchmarks/nr/results/bench_nr_fast.csv")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    new = not os.path.exists(a.out)
    with open(a.out, "a", newline="") as f:
        w = csv.DictWriter(f, FIELDS)
        if new:
            w.writeheader()
        for cfg_name in a.cfgs:
            for E in a.E:
                for R in a.R:
                    for b in a.backends:
                        if b == "triton" and CFGS[cfg_name]().n_cells > 1:
                            continue
                        ref = b == "reference"
                        row = bench(b, cfg_name, E, R, warm=1 if ref else 3, steps=a.ref_steps if ref else a.steps,
                                    max_s=a.max_s)
                        w.writerow(row)
                        f.flush()
                        print(row, flush=True)
