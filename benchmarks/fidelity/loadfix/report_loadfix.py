"""Tables of the load-gap study (docs/fidelity-load-gap.md) from the compare.py outputs of every arm.

usage: python report_loadfix.py <results dir with per_run_<arm>.csv and lena_pipeline_per_run.csv> <out dir>
Writes <out>/arms_by_regime.csv (one row per arm and load regime, also per seed split for the hold-out check),
<out>/lena_pipeline_by_regime.csv (the 5G-LENA grant accounting by regime and frame size) and <out>/doc_tables.md
(the markdown tables of docs/fidelity-load-gap.md), and prints a markdown overview.
"""
import csv
import glob
import os
import sys

import numpy as np

REG = ("light", "moderate", "saturated")
COLS = [("p50_relerr", "p50", True), ("p95_relerr", "p95", True), ("p99_relerr", "p99", True),
        ("drop_diff", "drop", True), ("goodput_relerr", "goodput", False), ("prb_relerr", "prb", True),
        ("tbs_ratio", "tbs_ratio", True), ("ks", "ks", True), ("w1_ms", "w1_ms", True)]


def med(x):
    x = np.array([v for v in x if np.isfinite(v)])
    return float(np.median(x)) if len(x) else float("nan")


def rows_of(path):
    return list(csv.DictReader(open(path)))


def summarize(rows, arm, group):
    out = dict(arm=arm, group=group, runs=len(rows))
    for col, name, signed in COLS:
        v = [float(r[col]) for r in rows]
        if signed:
            out[f"{name}_med"] = med(v)
        out[f"{name}_absmed"] = med(np.abs(v))
    return out


def write(path, rows):
    keys = list(rows[0])
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})


def main(res, out):
    os.makedirs(out, exist_ok=True)
    arms = sorted(os.path.basename(p)[len("per_run_"):-4] for p in glob.glob(os.path.join(res, "per_run_*.csv")))
    arms = ["primary"] + [a for a in arms if a != "primary"]
    table = []
    for arm in arms:
        pr = rows_of(os.path.join(res, f"per_run_{arm}.csv"))
        table.append(summarize(pr, arm, "all"))
        for g in REG:
            table.append(summarize([r for r in pr if r["regime"] == g], arm, g))
        for g in REG:
            table.append(summarize([r for r in pr if r["regime"] == g and r["seed"] == "2"], arm, g + "_seed2"))
    write(os.path.join(out, "arms_by_regime.csv"), table)
    print("| Arm | Regime | Runs | p50 err | \\|p50\\| | p95 err | Drop Δ pp | PRB err | TB ratio | KS |")
    print("|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for t in table:
        if t["group"] in REG:
            print(f"| {t['arm']} | {t['group']} | {t['runs']} | {t['p50_med']:+.1%} | {t['p50_absmed']:.1%} | "
                  f"{t['p95_med']:+.1%} | {100 * t['drop_med']:+.2f} | {t['prb_med']:+.1%} | {t['tbs_ratio_med']:.2f} | "
                  f"{t['ks_med']:.3f} |")
    lp = os.path.join(res, "lena_pipeline_per_run.csv")
    if os.path.exists(lp):
        L = rows_of(lp)
        f = lambda r, c: float(r[c]) if r.get(c) not in ("", None, "nan") else 0.0
        acc = []
        for S in ("all", "4000", "30000"):
            for g in REG:
                rs = [r for r in L if r["regime"] == g and (S == "all" or r["S"] == S)]
                if not rs:
                    continue
                pf = lambda c: med([f(r, c) / float(r["frames"]) for r in rs])
                acc.append(dict(S=S, regime=g, runs=len(rs), pad_frac_rbg=med([f(r, "pad_frac_rbg") for r in rs]),
                                empty_frac_tbs=med([f(r, "empty_frac_tbs") for r in rs]),
                                util_data=med([f(r, "util_data") for r in rs]), util_pad=med([f(r, "util_pad") for r in rs]),
                                util_retx=med([f(r, "util_retx") for r in rs]),
                                util_retx_blocked=med([f(r, "util_retx_blocked") for r in rs]),
                                tbs_per_frame=pf("new_tbs"), boot_per_frame=pf("boot_tbs"),
                                tail_stalls_per_frame=pf("tail_stalls"), sr_per_frame=pf("n_sr"),
                                tail_gap_ms=med([f(r, "tail_gap_ms_med") for r in rs if f(r, "tail_gap_ms_med") > 0]),
                                retx_gap_ms=med([f(r, "retx_gap_ms_med") for r in rs if f(r, "retx_gap_ms_med") > 0])))
        write(os.path.join(out, "lena_pipeline_by_regime.csv"), acc)
    doc_tables(table, out)


SINGLE = [("primary", "none (the engine)"), ("pf_intra", "`pf_intra_slot`"), ("pf_active", "`pf_active_only`"),
          ("pf", "both PF switches"), ("retx", "`retx_tdma`"), ("amc", "`amc_prev_alloc`"),
          ("oh8", "`tb_overhead_bytes=8`"), ("pipe", "`grant_pipeline` + 8 B")]
COMBO = ["primary", "pf", "pipe", "pf_pipe", "all"]
LOO = ["all", "all_no_intra", "all_no_active", "all_no_pipe", "all_no_retx", "all_no_amc", "all_no_oh8"]


def doc_tables(table, out):
    by = {(t["arm"], t["group"]): t for t in table}
    pct = lambda v: f"{v:+.1%}"
    lines = ["<!-- ARM_TABLE -->"]
    for arm, sw in SINGLE:
        for g in REG:
            t = by.get((arm, g))
            if t:
                lines.append(f"| {arm} | {sw} | {g} | {pct(t['p50_med'])} | {t['p50_absmed']:.1%} | {pct(t['p95_med'])} | "
                             f"{100 * t['drop_med']:+.2f} | {pct(t['prb_med'])} | {t['tbs_ratio_med']:.2f} | {t['ks_med']:.3f} |")
    lines.append("<!-- ALL_TABLE -->")
    for arm in COMBO:
        for g in REG + tuple(x + "_seed2" for x in REG):
            t = by.get((arm, g))
            if t:
                lines.append(f"| {arm} | {g.replace('_seed2', ', seed 2')} | {t['runs']} | {pct(t['p50_med'])} | "
                             f"{t['p50_absmed']:.1%} | {pct(t['p95_med'])} | {t['p95_absmed']:.1%} | "
                             f"{100 * t['drop_med']:+.2f} | {t['goodput_absmed']:.1%} | {pct(t['prb_med'])} | "
                             f"{t['tbs_ratio_med']:.2f} | {t['ks_med']:.3f} | {t['w1_ms_med']:.1f} |")
    lines.append("<!-- LOO_TABLE -->")
    for arm in LOO:
        m, s_, lt = by.get((arm, "moderate")), by.get((arm, "saturated")), by.get((arm, "light"))
        if m and s_ and lt:
            lines.append(f"| {arm} | {pct(m['p50_med'])} | {m['p50_absmed']:.1%} | {100 * m['drop_med']:+.2f} | "
                         f"{pct(s_['p50_med'])} | {s_['p50_absmed']:.1%} | {100 * s_['drop_med']:+.2f} | "
                         f"{pct(lt['p50_med'])} | {pct(lt['p95_med'])} | {pct(lt['prb_med'])} |")
    with open(os.path.join(out, "doc_tables.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
