"""Network-only cost of one engine configuration on an otherwise idle GPU: ms per control step and peak memory.

One configuration per process (fresh CUDA context and allocator). The script

1. waits until the GPU is idle (nvidia-smi utilization below --idle_util % in three samples 0.5 s apart and no
   other compute process visible), and records the utilization and device memory it saw;
2. pre-generates a pool of 16 input sets on the device (sends, detection flags, SNR drifting around a draw in
   [0, 25] dB, downlink messages, poses random-walking in the 150 m arena), so the timed loop is only
   submit (+ add_dl_frames) + step and the input generation is not timed;
3. builds the engine, runs the warm-up (graph capture, Triton compilation, queues filling up), and then times
   --repeats windows of n control steps each, synchronizing at the end of each window. n is chosen from the last
   warm-up step so each window lasts about --window_s seconds (at least 2 steps for the eager reference, 5
   otherwise, at most --max_steps);
4. prints one JSON line with the median and the min / max of the window means, the peak memory above the baseline
   (torch.cuda.max_memory_allocated, CUDA-graph pools included) and the GPU utilization before and during timing.

Cases (--case): a level of make_engine (L0, L0DR, L05, L05Q, L1, L2-legacy, L2, WIFI, TR, GE, QA, NN), or one of
  L2-legacy+edge / L2-legacy+energy (EdgeLoop / EnergyLoop over L2-legacy; --wrap_graph captures the wrapper's
  own step in a CUDA graph). --cfg picks the NR engine variant: ul, ul_dl, c3 (multicell(3)), c3_dl, ul_v2l (v2_lumped()).
Workload: every robot sends w.p. 0.3 per step (a third of them large, 30 kB; the rest 4 kB); with a downlink every
robot also receives 6 kB w.p. 0.3. L05 / L05Q / TR / GE / QA / NN use synthetic parameters (their cost does not
depend on the values).

usage: python benchmarks/uncontended/bench_net.py --case L2 --backend triton --cfg ul --E 4096 --R 100 --out x.jsonl
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import statistics
import subprocess
import sys
import threading
import time

import torch

_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in (_root, os.path.join(_root, "tests", "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from isaac_net.core import NRConfig, Requests, make_engine, multicell  # noqa: E402
from isaac_net.core.config import lena_validation_v2  # noqa: E402

SIZES = (4000.0, 30000.0)
POOL = 16


def smi():
    """(utilization %, used MiB, number of compute processes) of GPU 0, whole device, all processes."""
    try:
        o = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=10).stdout.strip().split(",")
        a = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=10).stdout.strip()
        return float(o[0]), float(o[1]), len([x for x in a.splitlines() if x.strip()])
    except Exception:
        return float("nan"), float("nan"), -1


def wait_idle(max_util, max_wait_s=1800):
    """Block until three samples 0.5 s apart show utilization < max_util; return what was seen."""
    t0 = time.time()
    while True:
        s = [smi() for _ in range(3) if not time.sleep(0.5)]
        u = max(x[0] for x in s)
        if u < max_util or time.time() - t0 > max_wait_s:
            return {"idle_util_max": u, "idle_mem_used_mib": max(x[1] for x in s),
                    "idle_compute_apps": max(x[2] for x in s), "idle_wait_s": round(time.time() - t0, 1),
                    "idle_ok": u < max_util}


class Sampler(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.v, self.halt = [], threading.Event()

    def run(self):
        while not self.halt.is_set():
            u, m, _ = smi()
            self.v.append((u, m))
            self.halt.wait(0.5)

    def summary(self):
        if not self.v:
            return {}
        u = [x[0] for x in self.v]
        return {"run_util_mean": round(sum(u) / len(u), 1), "run_mem_used_max_mib": max(x[1] for x in self.v)}


def synth(level):
    if level in ("L05", "L05Q", "L0"):
        from testlib import synthetic_params
        return synthetic_params(level, "cuda") if level != "L0" else None
    if level == "TR":                     # synthetic frames through the real fit (as tests/test_levels.py)
        from isaac_net.tools.fit_levels import fit_tr
        g = torch.Generator().manual_seed(0)
        n = 400
        delay = 0.05 + 3 * torch.rand(n, generator=g)
        delay[torch.rand(n, generator=g) < 0.2] = math.inf
        fr = {"ep": torch.randint(0, 2, (n,), generator=g), "env": torch.randint(0, 3, (n,), generator=g),
              "cls": torch.randint(1, 3, (n,), generator=g), "cap": torch.randint(0, 12, (n,), generator=g),
              "delay": delay}
        return fit_tr(fr, 3, 2, 12)[0]
    if level == "GE":
        K = 3
        return {"P": torch.tensor([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]]),
                "pi0": torch.full((K,), 1 / K), "mu": torch.zeros(K, 2), "sig": torch.full((K, 2), 0.5),
                "p": torch.tensor([[0.05, 0.1], [0.3, 0.3], [0.6, 0.7]]),
                "q": torch.stack([torch.linspace(0.05 * (k + 1), 1.5 * (k + 1), 101).expand(2, 101) for k in range(K)])}
    if level == "QA":
        return {"eta": 1.0, "pf": True}
    if level == "NN":
        from isaac_net.core.levels import DelayNet
        torch.manual_seed(0)
        net = DelayNet(13, 8, 16)
        return {"state": net.state_dict(), "xm": torch.zeros(13), "xs": torch.ones(13), "din": 13, "Q": 8, "h": 16}
    return None


def v2_lumped(**kw):
    """lena_validation_v2() without the SR / BSR grant pipeline (the lumped 40-slot SR-to-grant delay instead), with
    the fleet task's 16-frame buffer: the closest configuration to v2 that the triton kernel accepts
    (docs/fidelity-vs-lena.md, "Scale configurations")."""
    return lena_validation_v2(ul_grant_model="lumped", sr_grant_delay_slots=40, frame_buffer=16, **kw)


def config(cfg_name):
    return {"ul": lambda: NRConfig(), "ul_dl": lambda: NRConfig(dl=True), "c3": lambda: multicell(3),
            "c3_dl": lambda: multicell(3, dl=True), "ul_v2l": v2_lumped}[cfg_name]().with_(msg_sizes=SIZES)


def build(case, backend, cfg, E, R, wrap_graph):
    dev = "cuda"
    if case == "L2-legacy+edge":
        from isaac_net.core.config import EdgeConfig
        from isaac_net.core.edge import EdgeLoop
        return EdgeLoop(make_engine("L2-legacy", E, R, dev, cfg, backend, seed=1), EdgeConfig(), graph=wrap_graph)
    if case == "L2-legacy+energy":
        from isaac_net.core.energy import EnergyConfig, EnergyLoop
        return EnergyLoop(make_engine("L2-legacy", E, R, dev, cfg, backend, seed=1), EnergyConfig(),
                          graph=wrap_graph, seed=1, config=cfg)
    return make_engine(case, E, R, dev, cfg, backend, params=synth(case), seed=1)


class Inputs:
    def __init__(self, E, R, cfg, seed=0):
        d = "cuda"
        g = torch.Generator(device=d).manual_seed(seed)
        self.sends, self.dets, self.snrs, self.dls, self.poses = [], [], [], [], []
        snr = 25 * torch.rand(E, R, device=d, generator=g)
        pos = 150 * torch.rand(E, R, 2, device=d, generator=g)
        for _ in range(POOL):
            u = torch.rand(E, R, device=d, generator=g)
            self.sends.append((u < 0.3).long() * (1 + (u < 0.1).long()))
            self.dets.append(torch.rand(E, R, device=d, generator=g) < 0.1)
            snr = (snr + 0.5 * torch.randn(E, R, device=d, generator=g)).clamp(-5, 30)
            self.snrs.append(snr.clone())
            if cfg.dl:
                self.dls.append(torch.where(torch.rand(E, R, device=d, generator=g) < 0.3, 6000.0, 0.0))
            if cfg.n_cells > 1:
                pos = (pos + 2 * torch.rand(E, R, 2, device=d, generator=g) - 1).clamp(0, 150)
                self.poses.append(pos.clone())
        self.hid = torch.zeros(E, dtype=torch.long, device=d)
        self.k = 0

    def __call__(self, eng, cfg):
        k = self.k % POOL
        self.k += 1
        eng.submit(None, Requests(self.sends[k], self.dets[k], self.hid))
        if cfg.dl:
            eng.add_dl_frames(None, self.dls[k])
        return eng.step(None, self.poses[k] if cfg.n_cells > 1 else self.snrs[k])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--backend", default="reference")
    ap.add_argument("--cfg", default="ul")
    ap.add_argument("--E", type=int, required=True)
    ap.add_argument("--R", type=int, required=True)
    ap.add_argument("--wrap_graph", action="store_true")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--warm", type=int, default=None, help="warm-up steps (default 3 reference, 20 otherwise)")
    ap.add_argument("--window_s", type=float, default=2.0)
    ap.add_argument("--max_steps", type=int, default=100)
    ap.add_argument("--idle_util", type=float, default=5.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ref = a.backend in ("reference", "eager")
    level_uses_cfg = a.case in ("L2",)
    cfg = config(a.cfg) if level_uses_cfg else NRConfig().with_(msg_sizes=SIZES)
    rec = {"case": a.case, "backend": a.backend, "cfg": a.cfg if level_uses_cfg else "-",
           "wrap_graph": a.wrap_graph, "E": a.E, "R": a.R, "robots": a.E * a.R,
           "gpu": torch.cuda.get_device_name(), "torch": torch.__version__}
    rec |= wait_idle(a.idle_util)
    try:
        torch.manual_seed(0)
        inp = Inputs(a.E, a.R, cfg)
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        eng = build(a.case, a.backend, cfg, a.E, a.R, a.wrap_graph)
        warm = a.warm if a.warm is not None else (3 if ref else 20)
        dt = 0.0
        for i in range(warm):
            t1 = time.perf_counter()
            inp(eng, cfg)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t1
        rec["warm_s"] = round(time.time() - t0, 2)
        n = max(2 if ref else 5, min(a.max_steps, math.ceil(a.window_s / max(dt, 1e-4))))
        s = Sampler()
        s.start()
        wins = []
        for _ in range(a.repeats):
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            for _ in range(n):
                inp(eng, cfg)
            torch.cuda.synchronize()
            wins.append((time.perf_counter() - t1) / n * 1e3)
        s.halt.set()
        s.join()
        med = statistics.median(wins)
        rec |= {"ms_median": round(med, 4), "ms_min": round(min(wins), 4), "ms_max": round(max(wins), 4),
                "spread_pct": round(100 * (max(wins) - min(wins)) / med, 1), "ms_windows": [round(w, 4) for w in wins],
                "steps_per_window": n, "peak_mib": round((torch.cuda.max_memory_allocated() - base) / 2 ** 20, 1),
                "status": "ok"} | s.summary()
    except torch.cuda.OutOfMemoryError as e:
        rec["status"] = "oom: " + str(e).split("\n")[0][:100]
    except (NotImplementedError, ValueError) as e:
        rec["status"] = "n/a: " + str(e).split("\n")[0][:120]
    print(json.dumps(rec), flush=True)
    with open(a.out, "a") as f:
        f.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
