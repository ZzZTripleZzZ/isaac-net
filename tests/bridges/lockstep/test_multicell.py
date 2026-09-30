"""Single-process multi-cell mode: are the E cells in one ns-3 process independent?

Env 0 replays a fixed frame schedule. Env 1 is either idle or saturated (every UE sends a 30 kB frame
every step). If the cells are isolated (own SpectrumChannel, own RNG streams), env 0's per-frame
completion times are bitwise identical in both runs. Also checks RESET (in-process rebuild):
after a RESET with the same run number, env 0 must reproduce itself.
"""
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))  # repo root
from isaaclab_net.bridges.ns3_lockstep import protocol as P  # noqa: E402
from isaaclab_net.bridges.ns3_lockstep.core import BRIDGE_ROOT, Ns3Lockstep  # noqa: E402

R = 8
ARGS = {"placement": "dists", "dists": "20,30,40,50,60,70,80,90", "shadowStd": 0}
REF = f"{BRIDGE_ROOT}/results/correctness/n8_S4000_p0.5_run1_dists_tcp/ref/frames.csv"


def schedule():
    s = {}
    for r in csv.DictReader(open(REF)):
        k = int(round((float(r["gen"]) - 0.5) / 0.1))
        s.setdefault(k, []).append((int(r["ue"]), int(r["fid"]), int(r["bytes"])))
    return s


def episode(br, sched, env1_load, steps, E=2):
    pos = np.full((E, R, 3), np.nan, np.float32)
    done0, done_other, fid1 = {}, 0, 0
    for k in range(steps):
        fl = [(0, u, f, b) for (u, f, b) in sched.get(k, [])]
        if env1_load:
            for e in range(1, E):
                fl += [(e, u, fid1, 30000) for u in range(R)]
            fid1 += 1
        fr = np.array(fl, P.FRAME_IN) if fl else np.zeros(0, P.FRAME_IN)
        res = br.step(pos, fr)
        for d in res["done"]:
            if d["env"] == 0:
                done0[(int(d["ue"]), int(d["fid"]))] = k + d["frac"]
            else:
                done_other += 1
    return done0, done_other


def main():
    sched = schedule()
    steps = 120
    out = {}
    br = Ns3Lockstep(2, R, mode="single", transport="unix", run=1, ns3_args=ARGS)
    a, a1 = episode(br, sched, True, steps)        # fresh process, env 1 saturated
    br.reset(runs=[1])
    b, b1 = episode(br, sched, False, steps)       # rebuilt in-process, env 1 idle
    br.reset(runs=[1])
    c, c1 = episode(br, sched, True, steps)        # rebuilt again, env 1 saturated (must equal a)
    br.close()
    br = Ns3Lockstep(2, R, mode="single", transport="unix", run=1, ns3_args=ARGS)
    b2, _ = episode(br, sched, False, steps)       # fresh process, env 1 idle (must equal b)
    br.close()
    same = lambda x, y: x.keys() == y.keys() and all(x[k] == y[k] for k in x)
    out["env0_frames_done"] = [len(a), len(b), len(c)]
    out["env1_frames_done"] = [a1, b1, c1]
    out["env0_identical_loaded_vs_idle_env1"] = same(a, b)
    out["env0_identical_after_reset_same_run"] = same(a, c)
    out["env0_identical_idle_fresh_vs_rebuilt"] = same(b, b2)
    common = [k for k in a if k in b]
    out["env0_loaded_vs_idle_max_abs_diff_ms"] = float(max(abs(a[k] - b[k]) for k in common) * 100) if common else None
    out["env0_loaded_vs_idle_only_one_side"] = len(set(a) ^ set(b))
    # distribution comparison of env 0 in the 2-cell process vs a 1-cell process (different RNG streams)
    br1 = Ns3Lockstep(1, R, mode="procs", transport="unix", run=1, ns3_args=ARGS)
    s, _ = episode(br1, sched, False, steps, E=1)
    br1.close()
    out["env0_frames_done_single_cell_process"] = len(s)
    out["env0_identical_vs_1cell_process"] = same(a, s)
    cm = [k for k in a if k in s]
    out["env0_2cell_vs_1cell_max_abs_diff_ms"] = float(max(abs(a[k] - s[k]) for k in cm) * 100) if cm else None
    out["env0_2cell_vs_1cell_only_one_side"] = len(set(a) ^ set(s))
    gen = {}
    for r in csv.DictReader(open(REF)):
        gen[(int(r["ue"]), int(r["fid"]))] = (float(r["gen"]) - 0.5) / 0.1
    da = np.array([a[k] - gen[k] for k in a]) * 100
    ds = np.array([s[k] - gen[k] for k in s]) * 100
    out["delay_p50_p95_ms_2cell"] = [float(np.quantile(da, 0.5)), float(np.quantile(da, 0.95))]
    out["delay_p50_p95_ms_1cell"] = [float(np.quantile(ds, 0.5)), float(np.quantile(ds, 0.95))]
    os.makedirs(f"{BRIDGE_ROOT}/results", exist_ok=True)
    json.dump(out, open(f"{BRIDGE_ROOT}/results/multicell.json", "w"), indent=1)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
