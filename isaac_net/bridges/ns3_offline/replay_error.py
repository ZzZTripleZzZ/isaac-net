"""Quantify the open-loop replay flaw.

For each seed:
  closed A   : policy A (random, open-loop) with the ns-3 pool in the loop; its trace is recorded.
  closed B   : policy B (greedy, sends when its queue is empty) with the ns-3 pool in the loop (truth).
  offline A  : ns-3 run offline on A's trace (file mode, hold mobility; also waypoint mobility).
  replay A|A : policy A on ReplayNet(outcomes of offline A)      -> should equal closed A (sanity).
  replay A|A' : same, but offline A used another ns-3 RNG run    -> noise floor of one sample path.
  replay B|A : policy B on ReplayNet(outcomes of offline A)      -> error vs closed B = the flaw.
  replay A|B : policy A on ReplayNet(outcomes of offline B)      -> the reverse mismatch.
usage: python replay_error.py OUTDIR --E 8 --R 16 --seeds 0 1 2
"""
import argparse
import json
import os
import time

import torch

from ...examples.fleet_task import TASK_SIZES

from ..ns3_pool.poolnet import PoolNet
from .offline_ns3 import run_offline
from .replaynet import ReplayNet
from .rollout import frame_metrics, ks, rollout

SIZES = TASK_SIZES["T1"]


def summarize(r):
    fm, d = frame_metrics(r["frames"], r["T"])
    info = r["info"] or {}
    fm.update({"sends_per_robot_step": r["sends_per_robot_step"], "large_share": r["large_share"],
               "overflow_share": r["overflow_share"],
               "aoi_mean_steps": sum(r["aoi"]) / max(1, len(r["aoi"])),
               "expo": info.get("expo"), "ret": info.get("ret"), "goals": info.get("goals")})
    return fm, d


def compare(test, truth, dt, dtruth):
    out = {"ks_delay": ks(dt, dtruth)}
    for k in ("delivery_rate", "delay_p50_ms", "delay_p95_ms", "delay_mean_ms", "aoi_mean_steps",
              "sends_per_robot_step", "expo", "ret"):
        a, b = test.get(k), truth.get(k)
        if a is None or b is None:
            continue
        out[f"{k}_err"] = a - b
        out[f"{k}_relerr"] = (a - b) / abs(b) if b else None
    return out


def keyed_delay_err(fa, fb):
    """Mean |delay difference| (ms) over frames present in both runs with the same key."""
    A = {(e, r, t): d for e, r, t, c, d in fa}
    B = {(e, r, t): d for e, r, t, c, d in fb}
    both = [k for k in A if k in B]
    fin = [k for k in both if A[k] != float("inf") and B[k] != float("inf")]
    fate = sum((A[k] == float("inf")) != (B[k] == float("inf")) for k in both)
    err = [abs(A[k] - B[k]) * 100 for k in fin]      # delays are float32 step units in NetBase
    return {"common_frames": len(both), "fate_mismatch": fate,
            "mean_abs_delay_err_ms": sum(err) / len(err) if err else None,
            "same_frames_1ms": sum(e < 1.0 for e in err), "delivered_both": len(fin)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--E", type=int, default=8)
    ap.add_argument("--R", type=int, default=16)
    ap.add_argument("--T", type=int, default=300)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--L", type=float, default=150.0, help="arena side in m")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.set_num_threads(1)
    E, R, T = a.E, a.R, a.T
    rows = []
    for seed in a.seeds:
        sr = 1000 * seed + 1
        run_base = sr + E                 # PoolNet: episode 1 (first env.reset) uses seed_run + E
        res, dists, t0 = {}, {}, time.time()
        recs = {}
        for pol in ("random", "greedy"):
            net = PoolNet(E, R, "cpu", SIZES, seed_run=sr)
            r = rollout(pol, net, E, R, seed, T, a.L)
            net.close()
            r["ns3_run_base"] = run_base
            recs[pol] = r
            torch.save({k: v for k, v in r.items()}, f"{a.out}/closed_{pol}_s{seed}.pt")
            res[f"closed_{pol}"], dists[f"closed_{pol}"] = summarize(r)
        t_closed = time.time() - t0
        outc = {}
        # (policy, mobility, ns-3 run offset): offset 500 = another ns-3 sample path (noise floor)
        for pol, mob, off in (("random", "hold", 0), ("random", "hold", 500), ("random", "waypoint", 0),
                              ("greedy", "hold", 0)):
            t1 = time.time()
            tag = f"{mob}" + (f"_run+{off}" if off else "")
            oc, walls = run_offline(recs[pol], f"{a.out}/offline_s{seed}_{pol}_{tag}", run_base + off, mob)
            outc[(pol, tag)] = oc
            res[f"offline_{pol}_{tag}_wall_s"] = time.time() - t1
        cases = [("replay_random|random_hold", "random", ("random", "hold")),
                 ("replay_random|random_hold_run+500", "random", ("random", "hold_run+500")),
                 ("replay_random|random_waypoint", "random", ("random", "waypoint")),
                 ("replay_greedy|random_hold", "greedy", ("random", "hold")),
                 ("replay_greedy|random_waypoint", "greedy", ("random", "waypoint")),
                 ("replay_random|greedy_hold", "random", ("greedy", "hold"))]
        replays = {}
        for name, pol, key in cases:
            net = ReplayNet(E, R, "cpu", SIZES, outc[key])
            r = rollout(pol, net, E, R, seed, T, a.L)
            replays[name] = r
            res[name], dists[name] = summarize(r)
            n = sum(net.hits.values())
            res[name]["lookup"] = {k: v / max(1, n) for k, v in net.hits.items()}
        cmp = {}
        for name, pol, key in cases:
            truth = f"closed_{pol}"
            cmp[name] = compare(res[name], res[truth], dists[name], dists[truth])
            cmp[name].update(keyed_delay_err(replays[name]["frames"], recs[pol]["frames"]))
        row = {"seed": seed, "E": E, "R": R, "T": T, "L": a.L, "closed_wall_s": t_closed, "metrics": res, "vs_truth": cmp}
        rows.append(row)
        with open(f"{a.out}/replay_error.jsonl", "a") as f:
            f.write(json.dumps(row, default=float) + "\n")
        print(json.dumps({"seed": seed, "vs_truth": cmp}, default=float), flush=True)


if __name__ == "__main__":
    main()
