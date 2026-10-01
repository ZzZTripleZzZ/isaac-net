"""L0-emp next to L0, L2 and ns-3 at the three closed-loop loads (docs/closed-loop.md, "Empirical marginal").

L0-emp draws each message's delay i.i.d. from the empirical delay marginal of the L2 arm of the same run (same seeds,
same pooled frames as the lognormal L0). It is run on its own with `run_closedloop.py --arms L0,L0-emp --fit-from
<run folder>`, which also reruns L0 as a check that the earlier L0 numbers reproduce. This script joins those runs
with the earlier folders (no ns-3 or L2 rerun) and writes:

  closedloop/l0emp/comparison.csv           per load and arm: AoI p95, frames sent, AoI mean, delay p50/p95 with
                                            95% t intervals, their difference to ns-3, and W1 / KS to ns-3 (pooled
                                            and disjoint seeds) and to L2
  closedloop/l0emp/closedloop_delay_quantiles.csv  1,001 quantiles (inverted CDF) of every arm's delivered delays,
                                            same format as the paper's figure data
  closedloop/l0emp/closedloop_distances.csv every run's distances.csv plus the L0-emp rows and, for the light load,
                                            the disjoint-seed rows that run did not write
  closedloop/l0emp/aoi.csv                  per load and arm, from the AoI rebuilt out of frames.csv.gz (it equals the
                                            logged AoI mean of every seed): the share of robot-steps whose AoI exceeds
                                            ns-3's AoI p95 (rounded up to the 0.1 s step) with its seed interval and
                                            its paired per-seed difference to ns-3, the AoI KS distance to ns-3 over
                                            disjoint seeds with the split-half floors, and two statistics of the
                                            delays of one robot: the share of the delay variance between robots and
                                            the lag-1 correlation of successive delays around each robot's mean
The earlier folders are only read. The two figure files are drop-in replacements for the paper's figure data and
carry every earlier row unchanged.

usage: python benchmarks/closedloop/l0emp_compare.py   (after the three L0-emp runs, see docs/closed-loop.md)
"""
from __future__ import annotations

import csv
import gzip
import json
import math
import os
import sys
from statistics import mean, stdev

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from run_closedloop import STEP_S, ks, w1  # noqa: E402

RES = os.path.abspath(os.path.join(HERE, "..", "results"))
OUT = os.path.join(RES, "closedloop", "l0emp")
RUNS = [  # run id (as in the paper's figure data), earlier folder, label
    ("light_8x16", os.path.join(RES, "closedloop"), "one frame per 0.6 s, 8 x 16"),
    ("i02_8x16", os.path.join(RES, "closedloop_loaded", "interval1_8x16"), "one frame per 0.2 s, 8 x 16"),
    ("r32_8x32", os.path.join(RES, "closedloop_loaded", "r32_8x32"), "one frame per 0.6 s, 8 x 32"),
]
ARM_ORDER = ["ideal", "L2", "L0", "L0-emp", "L1", "L2-legacy", "ns3"]
TQ = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571}
METS = ["delivery_ratio", "delay_p50_ms", "delay_p95_ms", "delay_mean_ms", "aoi_mean_s", "aoi_p95_s",
        "task_return", "hazard_exposure", "goals_per_robot", "sent", "ms_per_step", "wall_s"]
CHECK = ["sent", "delivered", "aoi_mean_s", "aoi_p95_s", "delay_p50_ms", "delay_p95_ms", "task_return"]


def read_rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def write_rows(path, rows, keys):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def read_frames(path):
    """{arm: (seed [n], delay_ms [n])} of a frames.csv.gz."""
    out = {}
    with gzip.open(path, "rt") as f:
        for r in csv.DictReader(f):
            s, d = out.setdefault(r["arm"], ([], []))
            s.append(int(r["seed"]))
            d.append(float(r["delay_ms"]))
    return {a: (np.asarray(s), np.asarray(d)) for a, (s, d) in out.items()}


def frames_full(path):
    """[n, 5] array (seed, env, robot, capture step, delay ms) per arm of a frames.csv.gz."""
    out = {}
    with gzip.open(path, "rt") as f:
        r = csv.reader(f)
        next(r)
        for a, s, e, ro, c, d in r:
            out.setdefault(a, []).append((int(s), int(e), int(ro), int(c), float(d)))
    return {a: np.asarray(v, float) for a, v in out.items()}


def aoi_series(f, E, R, T, edge):
    """AoI (s) of every robot at every step [T, E, R] from the delivered frames f of one seed, as run_episode
    samples it: a frame is reported in the step that holds its completion time cap + delay, and the AoI after step t
    is t + 1 - the newest delivered capture. edge = 0 takes the stored delay as is, -1e-4 puts a completion time
    within float32 rounding of a step boundary into the earlier
    step (the NR engine), +1e-4 into the later one (L0-emp, whose delays are multiples of 2.5 ms); the caller keeps the one that
    reproduces the logged AoI mean of every seed."""
    step = np.floor(f[:, 3] + f[:, 4] / (STEP_S * 1000.0) + edge).astype(int)
    k = step < T
    lc = np.full((T, E, R), -1)
    np.maximum.at(lc, (step[k], f[k, 1].astype(int), f[k, 2].astype(int)), f[k, 3].astype(int))
    lc = np.maximum(np.maximum.accumulate(lc, axis=0), 0)
    return (np.arange(T)[:, None, None] + 1 - lc) * STEP_S


def robot_stats(F):
    """Share of the delay variance between robots (seed, env, robot) and the lag-1 correlation of successive
    delays of one robot around that robot's mean."""
    F = F[np.lexsort((F[:, 3], F[:, 2], F[:, 1], F[:, 0]))]
    key = F[:, 0] * 10 ** 6 + F[:, 1] * 10 ** 3 + F[:, 2]
    _, inv = np.unique(key, return_inverse=True)
    mu = np.bincount(inv, F[:, 4]) / np.bincount(inv)
    r = F[:, 4] - mu[inv]
    same = key[1:] == key[:-1]
    return 1 - r.var() / F[:, 4].var(), float(np.corrcoef(r[:-1][same], r[1:][same])[0, 1])


def ci(v):
    v = [float(x) for x in v if not math.isnan(float(x))]
    m = mean(v)
    return m, (TQ.get(len(v), 2.0) * stdev(v) / math.sqrt(len(v)) if len(v) > 1 else math.nan)


def main():
    os.makedirs(OUT, exist_ok=True)
    comp, qrows, drows, checks, arows = [], [], [], [], []
    for run, old, label in RUNS:
        new = os.path.join(OUT, run)
        old_seed = [r for r in read_rows(os.path.join(old, "per_seed.csv")) if r["arm"] != "L0-emp"]
        new_seed = read_rows(os.path.join(new, "per_seed.csv"))
        fr = read_frames(os.path.join(old, "frames.csv.gz"))
        frn = read_frames(os.path.join(new, "frames.csv.gz"))
        # 1) the rerun L0 must reproduce the earlier L0 seed by seed
        o0 = {r["seed"]: r for r in old_seed if r["arm"] == "L0"}
        for r in (r for r in new_seed if r["arm"] == "L0"):
            for k in CHECK:
                checks.append((run, r["seed"], k, float(o0[r["seed"]][k]), float(r[k])))
        fr["L0-emp"] = frn["L0-emp"]
        seeds = old_seed + [r for r in new_seed if r["arm"] == "L0-emp"]
        # 2) per-arm means and intervals, differences to ns-3
        stat = {}
        for arm in ARM_ORDER:
            rr = [r for r in seeds if r["arm"] == arm]
            stat[arm] = {m: ci([r[m] for r in rr]) for m in METS}
        n3 = stat["ns3"]
        s3, d3 = fr["ns3"]
        od3 = d3[s3 % 2 == 1]
        for arm in ARM_ORDER:
            st = stat[arm]
            row = dict(run=run, arm=arm)
            for m in ("aoi_p95_s", "sent", "aoi_mean_s", "delay_p50_ms", "delay_p95_ms", "delivery_ratio",
                      "task_return"):
                row[m], row[m + "_ci95"] = st[m]
                row[m + "_ci95_pct"] = 100 * st[m][1] / st[m][0] if st[m][0] else math.nan
            for m in ("aoi_p95_s", "sent", "aoi_mean_s", "delay_p95_ms"):
                row[m + "_vs_ns3_pct"] = 100 * (st[m][0] / n3[m][0] - 1)
            if arm not in ("ideal", "ns3"):
                s, d = fr[arm]
                ev = d[s % 2 == 0]
                row.update(w1_ns3_pooled=w1(d, d3), ks_ns3_pooled=ks(d, d3), w1_ns3_disjoint=w1(ev, od3),
                           ks_ns3_disjoint=ks(ev, od3))
                if arm != "L2":
                    row.update(w1_l2_pooled=w1(d, fr["L2"][1]), ks_l2_pooled=ks(d, fr["L2"][1]))
            comp.append(row)
        # 3) figure data: 1,001 quantiles per arm (inverted CDF, as the paper's figure data)
        qq = np.linspace(0, 1, 1001)
        for arm in ["L0", "L0-emp", "L1", "L2", "L2-legacy", "ns3"]:
            for q, v in zip(qq, np.quantile(fr[arm][1], qq, method="inverted_cdf")):
                qrows.append(dict(run=run, arm=arm, q=round(float(q), 3), delay_ms=float(v)))
        # 4) distances: the run's own rows, the new L0-emp rows, and disjoint rows where the run lacks them
        old_d = read_rows(os.path.join(old, "distances.csv"))
        old_d = [r for r in old_d if "L0-emp" not in r["a"]]
        have = {(r["a"], r["b"]) for r in old_d}
        add = []

        def pair(a, b, x, y):
            if (a, b) not in have:
                add.append(dict(a=a, b=b, n_a=len(x), n_b=len(y), w1_ms=f"{w1(x, y):.3f}", ks=f"{ks(x, y):.4f}"))

        pair("L0-emp", "L2", fr["L0-emp"][1], fr["L2"][1])
        pair("L0-emp", "ns3", fr["L0-emp"][1], d3)
        for arm in ("L2", "L0", "L0-emp", "L1", "L2-legacy"):
            s, d = fr[arm]
            pair(f"{arm} even seeds", "ns3 odd seeds", d[s % 2 == 0], od3)
        allrows = old_d + add
        drows += [dict(run=run, **r) for r in allrows]
        # 5) AoI tail, AoI distance and delay correlation, from the frames
        setup = json.load(open(os.path.join(old, "setup.json")))
        E, R, T = setup["E"], setup["R"], setup["T"]
        ff = frames_full(os.path.join(old, "frames.csv.gz"))
        ff["L0-emp"] = frames_full(os.path.join(new, "frames.csv.gz"))["L0-emp"]
        thr = math.ceil(round(stat["ns3"]["aoi_p95_s"][0] / STEP_S, 6)) * STEP_S
        logged = {(r["arm"], int(r["seed"])): float(r["aoi_mean_s"]) for r in seeds}
        aoi, tail = {}, {}
        for arm in ("L2", "L0", "L0-emp", "L1", "L2-legacy", "ns3"):
            for edge in (0.0, -1e-4, 1e-4):
                aoi[arm] = {s: aoi_series(ff[arm][ff[arm][:, 0] == s], E, R, T, edge).ravel() for s in range(5)}
                if all(abs(aoi[arm][s].mean() - logged[(arm, s)]) < 1e-5 for s in range(5)):
                    break
            else:
                raise SystemExit(f"{run} {arm}: the rebuilt AoI does not reproduce the logged AoI mean")
            tail[arm] = np.array([np.mean(aoi[arm][s] > thr + 1e-6) for s in range(5)])
        od3 = np.concatenate([aoi["ns3"][s] for s in (1, 3)])
        floors = {a: ks(np.concatenate([aoi[a][s] for s in (0, 2, 4)]), np.concatenate([aoi[a][s] for s in (1, 3)]))
                  for a in ("L2", "ns3")}
        for arm in ("L2", "L0", "L0-emp", "L1", "L2-legacy", "ns3"):
            m, h = ci(tail[arm])
            dm, dh = ci(tail["ns3"] - tail[arm]) if arm != "ns3" else (math.nan, math.nan)
            vb, lag1 = robot_stats(ff[arm])
            ev = np.concatenate([aoi[arm][s] for s in (0, 2, 4)])
            arows.append(dict(run=run, arm=arm, aoi_threshold_s=round(thr, 3), p_aoi_above=m, p_aoi_above_ci95=h,
                              p_aoi_above_per_seed=" ".join(f"{x:.4f}" for x in tail[arm]),
                              ns3_minus_arm_paired=dm, ns3_minus_arm_paired_ci95=dh,
                              aoi_ks_ns3_disjoint=ks(ev, od3) if arm != "ns3" else math.nan,
                              aoi_ks_floor_l2=floors["L2"], aoi_ks_floor_ns3=floors["ns3"],
                              delay_var_between_robots=vb, delay_lag1_within_robot=lag1))
    keys = list(dict.fromkeys(k for r in comp for k in r))
    write_rows(os.path.join(OUT, "comparison.csv"), comp, keys)
    write_rows(os.path.join(OUT, "closedloop_delay_quantiles.csv"), qrows, ["run", "arm", "q", "delay_ms"])
    write_rows(os.path.join(OUT, "closedloop_distances.csv"), drows, ["run", "a", "b", "n_a", "n_b", "w1_ms", "ks"])
    write_rows(os.path.join(OUT, "aoi.csv"), arows, list(arows[0]))
    bad = [c for c in checks if abs(c[3] - c[4]) > 1e-6 * max(1.0, abs(c[3]))]
    with open(os.path.join(OUT, "l0_rerun_check.txt"), "w") as f:
        f.write(f"L0 rerun vs earlier L0: {len(checks)} seed x metric values compared, {len(bad)} differ\n")
        for c in bad:
            f.write(f"{c}\n")
    print(f"L0 rerun check: {len(checks)} compared, {len(bad)} differ")
    for r in comp:
        if r["arm"] in ("L2", "L0", "L0-emp", "ns3"):
            print(r["run"], r["arm"], f"AoI p95 {r['aoi_p95_s']:.3f} ± {r['aoi_p95_s_ci95']:.3f} "
                  f"({r['aoi_p95_s_vs_ns3_pct']:+.1f}%), sent {r['sent']:.0f} ± {r['sent_ci95']:.0f} "
                  f"({r['sent_vs_ns3_pct']:+.1f}%), p95 {r['delay_p95_ms']:.0f}, "
                  f"KS disj {r.get('ks_ns3_disjoint', float('nan')):.3f}")
    for r in arows:
        print(r["run"], r["arm"], f"P(AoI > {r['aoi_threshold_s']} s) {r['p_aoi_above']:.4f} ± {r['p_aoi_above_ci95']:.4f},"
              f" ns3 - arm {r['ns3_minus_arm_paired']:.4f} ± {r['ns3_minus_arm_paired_ci95']:.4f},"
              f" AoI KS {r['aoi_ks_ns3_disjoint']:.3f} (floors {r['aoi_ks_floor_l2']:.3f} / {r['aoi_ks_floor_ns3']:.3f}),"
              f" between-robot {r['delay_var_between_robots']:.2f}, lag1 {r['delay_lag1_within_robot']:.2f}")


if __name__ == "__main__":
    main()
