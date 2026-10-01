"""GPU kernel time per control step of the levels and the adaptive engine (torch.profiler, CUDA activity).

usage: python benchmarks/adaptive/kernel_time.py --E 4096 --R 16 --out kernel_time.jsonl [--cheap_params fit.pt]

On a GPU shared with other jobs, wall time per step mostly measures time-slice waits. The sum of the durations of this
process's CUDA kernels per step (submit + step, no resets, after warm-up and graph capture) is a contention-light
estimate of the GPU work an engine needs. It leaves out launch overhead, which graph mode removes and the eager
reference backends pay in full. Workload: sweep.workload "mixed" (load switching active).
"""
import argparse
import json
import os
import sys

import torch
from torch.profiler import ProfilerActivity, profile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _root)
from sweep import workload  # noqa: E402
from isaac_net.core import NRConfig, make_engine  # noqa: E402
from isaac_net.core.adaptive import FidelityConfig, make_adaptive  # noqa: E402
from isaac_net.core.proto.netsim import Requests  # noqa: E402


def kernel_ms(net, wl, warm=20, n=10):
    for send, pos, _ in wl[:warm]:
        net.submit(None, Requests(send))
        net.step(None, pos)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for send, pos, _ in wl[warm: warm + n]:
            net.submit(None, Requests(send))
            net.step(None, pos)
        torch.cuda.synchronize()
    us = sum(e.self_device_time_total for e in prof.key_averages())
    return us / 1e3 / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--E", type=int, default=4096)
    ap.add_argument("--R", type=int, default=16)
    ap.add_argument("--cheap_params", default=None)
    ap.add_argument("--nr_E", type=int, default=256)
    ap.add_argument("--nr_R", type=int, default=8)
    ap.add_argument("--out", default="kernel_time.jsonl")
    a = ap.parse_args()
    dev = torch.device("cuda")
    cfg = NRConfig(seed=7)
    fh = open(a.out, "a")
    l05q = torch.load(a.cheap_params, weights_only=True)["L05Q"] if a.cheap_params else None

    def ada(cheap, cb, xb, E, R, expensive="L2-legacy", **kw):
        return lambda: make_adaptive(E, R, dev, cfg, FidelityConfig(
            cheap=cheap, expensive=expensive, cheap_backend=cb, expensive_backend=xb,
            cheap_params=l05q if cheap == "L05Q" else None, **kw), seed=7)

    E, R = a.E, a.R
    load = dict(mode="load", up_threshold=1000.0)
    cases = [("L2-legacy triton", E, R, lambda: make_engine("L2-legacy", E, R, dev, cfg, "triton", seed=7)),
             ("L2-legacy graph", E, R, lambda: make_engine("L2-legacy", E, R, dev, cfg, "graph", seed=7)),
             ("L1 triton", E, R, lambda: make_engine("L1", E, R, dev, cfg, "triton", seed=7))]
    if l05q is not None:
        cases.append(("L05Q graph", E, R, lambda: make_engine("L05Q", E, R, dev, cfg, "graph", seed=7, params=l05q)))
    for xb in ("triton", "graph"):
        for lay, kw in (("mask", dict(layout="mask")), ("budget 50%", dict(active_budget=0.5)),
                        ("budget 25%", dict(active_budget=0.25)), ("budget 10%", dict(active_budget=0.1))):
            cases.append((f"L1 triton -> L2-legacy {xb}, load thr 1000, {lay}", E, R,
                          ada("L1", "triton", xb, E, R, **load, **kw)))
        cases.append((f"L1 triton -> L2-legacy {xb}, static 10%", E, R,
                      ada("L1", "triton", xb, E, R, mode="static", fraction=0.1)))
    nE, nR = a.nr_E, a.nr_R
    cases += [("L2 (NR) reference", nE, nR, lambda: make_engine("L2", nE, nR, dev, cfg, seed=7)),
              ("L1 triton -> L2 (NR), load thr 1000, mask", nE, nR,
               ada("L1", "triton", "reference", nE, nR, expensive="L2", layout="mask", **load)),
              ("L1 triton -> L2 (NR), load thr 1000, budget 25%", nE, nR,
               ada("L1", "triton", "reference", nE, nR, expensive="L2", active_budget=0.25, **load))]
    wls = {}
    for name, e, r, build in cases:
        if (e, r) not in wls:
            wls[(e, r)] = workload(e, r, 40, "mixed", dev)
        torch.manual_seed(7)
        ms = kernel_ms(build(), wls[(e, r)])
        row = {"case": name, "E": e, "R": r, "kernel_ms_per_step": ms, "gpu": torch.cuda.get_device_name()}
        fh.write(json.dumps(row) + "\n")
        fh.flush()
        print(json.dumps(row), flush=True)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
