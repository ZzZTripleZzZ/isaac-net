"""ms per control step (add_frames + step) for NetSlot vs NetSlotFast, plus peak GPU memory.

usage: python bench.py --grid 16x16,256x16 --impls orig,graph,compile --out bench.jsonl
Workload: each robot sends with p=0.3 per step (half small, half large); the first --warm steps
fill the queues and are not timed.
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

from netsim import NetSlot
from netsim_fast import NetSlotFast

SIZES = (4000.0, 30000.0)


def run(impl, E, R, steps, warm, dev="cuda"):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base_mem = torch.cuda.memory_allocated()
    torch.manual_seed(0)
    t_build = time.time()
    net = NetSlot(E, R, dev, SIZES) if impl == "orig" else NetSlotFast(E, R, dev, SIZES, backend=impl)
    g = torch.Generator(device=dev).manual_seed(1)
    snr = -5 + 40 * torch.rand(E, R, device=dev, generator=g)
    hid = torch.zeros(E, dtype=torch.long, device=dev)

    def one(t):
        tx = torch.rand(E, R, device=dev, generator=g) < 0.3
        lg = torch.rand(E, R, device=dev, generator=g) < 0.5
        send = tx.long() * (1 + lg.long())
        det = tx & (torch.rand(E, R, device=dev, generator=g) < 0.3)
        net.add_frames(t, send, det, hid, snr)
        return net.step(t, snr, hid)

    for t in range(warm):
        one(t)
    torch.cuda.synchronize()
    setup_s = time.time() - t_build
    with UtilSampler() as us:
        t0 = time.time()
        for t in range(warm, warm + steps):
            one(t)
        torch.cuda.synchronize()
        ms = (time.time() - t0) / steps * 1e3
    peak = (torch.cuda.max_memory_allocated() - base_mem) / 2 ** 20
    q = float(net.queued().float().mean())
    del net
    return {"impl": impl, "E": E, "R": R, "ms_per_step": round(ms, 3), "peak_mem_MiB": round(peak, 1),
            "setup_s": round(setup_s, 1), "mean_queue": round(q, 2), "steps": steps} | us.summary()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="16x16,256x16")
    ap.add_argument("--impls", default="orig,graph,compile")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--orig_steps", type=int, default=10)
    ap.add_argument("--warm", type=int, default=5)
    ap.add_argument("--out", default="bench.jsonl")
    a = ap.parse_args()
    for cell in a.grid.split(","):
        E, R = map(int, cell.split("x"))
        for impl in a.impls.split(","):
            try:
                r = run(impl, E, R, a.orig_steps if impl == "orig" else a.steps, a.warm)
            except torch.OutOfMemoryError as ex:
                r = {"impl": impl, "E": E, "R": R, "error": "OOM"}
            r["gpu"] = torch.cuda.get_device_name()
            print(json.dumps(r), flush=True)
            with open(a.out, "a") as f:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
