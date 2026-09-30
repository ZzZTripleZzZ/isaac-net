"""Correctness: static robots, same traffic, same seed -> the bridge must reproduce netslot-ref.

1. Run the unmodified netslot-ref binary (copied build of ns3ref) with a given CLI.
2. Replay its generated frame schedule (frames.csv: ue, fid, gen, bytes) through the bridge, one
   control step per call, robots static (positions NaN = keep the scenario's own drop).
3. Compare per-frame completion delay (exact match expected) and the delay distributions.
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))  # repo root
from isaaclab_net.bridges.ns3_lockstep import protocol as P  # noqa: E402
from isaaclab_net.bridges.ns3_lockstep.core import BRIDGE_ROOT, ENVP, Ns3Lockstep  # noqa: E402

REF_BIN = f"{BRIDGE_ROOT}/ns-3.48/build/scratch/netslot-ref/ns3.48-netslot-ref-optimized"


def run_ref(args, out):
    os.makedirs(out, exist_ok=True)
    env = dict(os.environ, LD_LIBRARY_PATH=f"{BRIDGE_ROOT}/ns-3.48/build/lib:{ENVP}/lib")
    cmd = [REF_BIN] + [f"--{k}={v}" for k, v in args.items()] + ["--macTraces=0", "--pktLog=0", f"--outDir={out}"]
    t = time.time()
    subprocess.run(cmd, check=True, env=env, stdout=subprocess.DEVNULL)
    wall = time.time() - t
    rows = list(csv.DictReader(open(os.path.join(out, "frames.csv"))))
    return rows, wall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nUe", type=int, default=8)
    ap.add_argument("--frameBytes", type=int, default=4000)
    ap.add_argument("--p", type=float, default=0.5)
    ap.add_argument("--trafficTime", type=float, default=10.0)
    ap.add_argument("--dists", default="20,30,40,50,60,70,80,90")
    ap.add_argument("--placement", default="dists")
    ap.add_argument("--shadowStd", type=float, default=0.0)
    ap.add_argument("--run", type=int, default=1)
    ap.add_argument("--transport", default="tcp")
    ap.add_argument("--extra", default="", help="extra k=v,k=v ns-3 args for both runs")
    ap.add_argument("--out", default=f"{BRIDGE_ROOT}/results/correctness")
    a = ap.parse_args()

    common = {"nUe": a.nUe, "placement": a.placement, "dists": a.dists, "shadowStd": a.shadowStd, "run": a.run}
    for kv in filter(None, a.extra.split(",")):
        k, v = kv.split("=")
        common[k] = v
    ref_args = dict(common, frameBytes=a.frameBytes, p=a.p, trafficTime=a.trafficTime)
    tag = f"n{a.nUe}_S{a.frameBytes}_p{a.p}_run{a.run}_{a.placement}_{a.transport}" + (
        "_" + a.extra.replace("=", "").replace(",", "_") if a.extra else "")
    out = os.path.join(a.out, tag)
    rows, ref_wall = run_ref(ref_args, os.path.join(out, "ref"))
    app_start, step_s = 0.5, 0.1
    n_ticks = int(np.floor(a.trafficTime * 1000.0 / 100.0 + 1e-9))
    sched = {}
    for r in rows:
        k = int(round((float(r["gen"]) - app_start) / step_s))
        sched.setdefault(k, []).append((int(r["ue"]), int(r["fid"]), int(r["bytes"])))

    ns3_args = {k: v for k, v in common.items() if k != "run"}
    ns3_args["flowmon"] = 1        # netslot-ref installs a FlowMonitor; mirror its event set exactly
    br = Ns3Lockstep(1, a.nUe, mode="procs", transport=a.transport, run=a.run, ns3_args=ns3_args,
                     log_dir=os.path.join(out, "logs"))
    pos = np.full((1, a.nUe, 3), np.nan, np.float32)
    done = {}
    n_steps = n_ticks + 22       # netslot-ref runs appStart + trafficTime + deadline + 0.2
    t_wall = time.time()
    for k in range(n_steps):
        fl = sched.get(k, [])
        fr = np.zeros(len(fl), P.FRAME_IN)
        for j, (u, f, b) in enumerate(fl):
            fr[j] = (0, u, f, b)
        res = br.step(pos, fr)
        for d in res["done"]:
            # netslot-ref prints times with 6 significant digits (ostream default): do the same
            done[(int(d["ue"]), int(d["fid"]))] = float(f"{app_start + (k + d['frac']) * step_s:g}")
    bridge_wall = time.time() - t_wall
    tim = {k: float(np.mean(v)) for k, v in br.timing.items()}
    br.close()

    ref_done, diffs, only_ref, only_br = {}, [], 0, 0
    for r in rows:
        key = (int(r["ue"]), int(r["fid"]))
        if int(r["rxpk"]) >= int(r["npk"]):
            ref_done[key] = float(r["last"])
    for key, tl in ref_done.items():
        if key in done:
            diffs.append(done[key] - tl)
        else:
            only_ref += 1
    only_br = sum(1 for key in done if key not in ref_done)
    gen = {(int(r["ue"]), int(r["fid"])): float(r["gen"]) for r in rows}
    d_ref = np.sort([ref_done[k] - gen[k] for k in ref_done])
    d_br = np.sort([done[k] - gen[k] for k in done])
    ks = 0.0
    if len(d_ref) and len(d_br):
        allv = np.concatenate([d_ref, d_br])
        ks = float(np.max(np.abs(np.searchsorted(d_ref, allv, "right") / len(d_ref)
                                 - np.searchsorted(d_br, allv, "right") / len(d_br))))
    q = [0.5, 0.9, 0.95, 0.99]
    summary = {
        "tag": tag, "frames_generated": len(rows), "ref_complete": len(ref_done), "bridge_complete": len(done),
        "matched": len(diffs), "only_ref": only_ref, "only_bridge": only_br,
        "max_abs_diff_s": float(np.max(np.abs(diffs))) if diffs else None,
        "exact_matches": int(np.sum(np.abs(diffs) < 1e-12)) if diffs else 0,
        "ks_distance": ks,
        "ref_delay_q_ms": [float(np.quantile(d_ref, x) * 1e3) for x in q] if len(d_ref) else None,
        "bridge_delay_q_ms": [float(np.quantile(d_br, x) * 1e3) for x in q] if len(d_br) else None,
        "ref_wall_s": ref_wall, "bridge_wall_s": bridge_wall, "bridge_steps": n_steps, "bridge_timing_mean": tim,
    }
    os.makedirs(out, exist_ok=True)
    json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=1)
    np.savetxt(os.path.join(out, "delay_ref.txt"), d_ref)
    np.savetxt(os.path.join(out, "delay_bridge.txt"), d_br)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
