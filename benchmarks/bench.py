"""ms per control step (submit + step) for every fidelity level and backend, plus peak GPU memory.

usage: python bench.py --levels all --backends reference,graph,triton --grid 16x16,256x16 --out bench.jsonl
Backends: reference (eager netsim), eager, graph, compile, triton (L1/L2 only). Combinations that do not
exist are skipped. Workload: each robot sends with p=0.3 per step (half small, half large); the first --warm
steps fill the queues and are not timed. --reset_frac > 0 also partially resets that fraction of envs every
step inside the timed loop (Isaac Lab style episode ends), through reset(env_ids) with an index tensor.
GPU utilization (whole GPU, all processes, nvidia-smi) is sampled during each timing window, because the GPU
may be shared with other jobs.
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time

import torch

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)                                   # isaac_net without `pip install -e .`
sys.path.insert(0, os.path.join(_root, "tests", "scripts"))  # testlib
from isaac_net.core.proto import netsim  # noqa: E402
from isaac_net.core.proto.netsim import Requests  # noqa: E402
from isaac_net.core.proto.netsim_fast import NetFast  # noqa: E402
from testlib import SIZES, synthetic_params  # noqa: E402


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


def gpu_util_now():
    try:
        o = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                           capture_output=True, text=True, timeout=5).stdout
        return float(o.strip().split()[0])
    except Exception:
        return None


def build(level, backend, E, R, dev):
    params = synthetic_params(level, dev)
    if backend in ("reference", "orig"):
        return netsim.make_net(level, E, R, dev, SIZES, params, seed=0)
    return NetFast(level, E, R, dev, SIZES, params=params, backend=backend, seed=0)


def run(level, backend, E, R, steps, warm, reset_frac, api="new", dev="cuda"):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base_mem = torch.cuda.memory_allocated()
    torch.manual_seed(0)
    t_build = time.time()
    net = build(level, backend, E, R, dev)
    g = torch.Generator(device=dev).manual_seed(1)
    snr = -5 + 40 * torch.rand(E, R, device=dev, generator=g)
    hid = torch.zeros(E, dtype=torch.long, device=dev)
    n_reset = max(1, int(round(reset_frac * E))) if reset_frac > 0 else 0

    def one():
        tx = torch.rand(E, R, device=dev, generator=g) < 0.3
        lg = torch.rand(E, R, device=dev, generator=g) < 0.5
        send = tx.long() * (1 + lg.long())
        det = tx & (torch.rand(E, R, device=dev, generator=g) < 0.3)
        if n_reset:
            net.reset(torch.randint(0, E, (n_reset,), device=dev, generator=g))
        if api == "legacy":
            net.add_frames(None, send, det, hid, snr)
            return net.step(None, snr, hid)
        net.submit(None, Requests(send, det, hid), snr)
        return net.step(None, snr)

    for _ in range(warm):
        one()
    torch.cuda.synchronize()
    setup_s = time.time() - t_build
    with UtilSampler() as us:
        t0 = time.time()
        for _ in range(steps):
            one()
        torch.cuda.synchronize()
        ms = (time.time() - t0) / steps * 1e3
    peak = (torch.cuda.max_memory_allocated() - base_mem) / 2 ** 20
    q = float(net.queued().float().mean())
    del net
    return {"level": level, "backend": backend, "E": E, "R": R, "ms_per_step": round(ms, 3),
            "peak_mem_MiB": round(peak, 1), "setup_s": round(setup_s, 1), "mean_queue": round(q, 2),
            "steps": steps, "reset_frac": reset_frac, "api": api} | us.summary()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="16x16,256x16")
    ap.add_argument("--levels", default="all")
    ap.add_argument("--backends", default="reference,graph,triton")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--ref_steps", type=int, default=10, help="timed steps for the (slow) reference backend")
    ap.add_argument("--warm", type=int, default=5)
    ap.add_argument("--reset_frac", type=float, default=0.0)
    ap.add_argument("--api", default="new", choices=["new", "legacy"],
                    help="new: submit/step dict with per-message outputs; legacy: add_frames/step tuple")
    ap.add_argument("--out", default="bench.jsonl")
    a = ap.parse_args()
    levels = netsim.RUNGS if a.levels == "all" else a.levels.split(",")
    for cell in a.grid.split(","):
        E, R = map(int, cell.split("x"))
        for level in levels:
            for backend in a.backends.split(","):
                if backend == "triton" and level not in ("L1", "L2"):
                    continue
                steps = a.ref_steps if backend in ("reference", "orig") else a.steps
                try:
                    r = run(level, backend, E, R, steps, a.warm, a.reset_frac, a.api)
                except torch.OutOfMemoryError:
                    r = {"level": level, "backend": backend, "E": E, "R": R, "error": "OOM"}
                r["gpu"] = torch.cuda.get_device_name()
                print(json.dumps(r), flush=True)
                with open(a.out, "a") as f:
                    f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
