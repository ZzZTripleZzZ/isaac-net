"""Print the markdown tables of docs/fidelity-vs-lena.md from benchmarks/fidelity/results/*.csv.

usage: python report.py [results dir]
"""
import csv
import os
import sys

import numpy as np

RES = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def rows(name):
    p = os.path.join(RES, name)
    return list(csv.DictReader(open(p))) if os.path.exists(p) else []


def f(x, kind="pct"):
    x = float(x)
    if not np.isfinite(x):
        return "–"
    if kind == "pct":
        return f"{100 * x:+.1f}%"
    if kind == "apct":
        return f"{100 * x:.1f}%"
    if kind == "ms":
        return f"{x:.1f}"
    if kind == "pp":
        return f"{100 * x:+.2f} pp"
    return f"{x:.3f}"


def table(header, body):
    print("| " + " | ".join(header) + " |")
    print("|" + "|".join(":---" if i == 0 else "---:" for i in range(len(header))) + "|")
    for b in body:
        print("| " + " | ".join(str(x) for x in b) + " |")
    print()


STRAT_H = ["Subset", "Runs", "KS (floor)", "W1 ms", "p50 err", "p95 err", "p99 err", "Drop Δ", "Goodput \\|err\\|",
           "PRB err"]


def strat_row(r, label):
    return [label, r["runs"], f"{f(r['ks_med'], 'x')} ({f(r['self_ks_med'], 'x')})", f(r["w1_ms_med"], "ms"),
            f(r["p50_med"]), f(r["p95_med"]), f(r["p99_med"]), f(r["drop_diff_med"], "pp"), f(r["goodput_absmed"], "apct"),
            f(r["prb_med"])]


def stratified(name, keys=("all", "regime", "N", "load", "S")):
    s = rows(name)
    for k in keys:
        table(STRAT_H, [strat_row(r, f"{r['group']} = {r['value']}" if k != "all" else "all runs")
                        for r in s if r["group"] == k])


def main():
    print("## Headline (primary arm, all runs)\n")
    s = {r["group"] + "/" + r["value"]: r for r in rows("summary_primary.csv")}
    a = s["all/all"]
    body = []
    for key, name, kind in (("ks", "KS distance, delay of on-time frames", "x"), ("self_ks", "KS, engine replica vs replica", "x"),
                            ("w1_ms", "Wasserstein-1, delay (ms)", "ms"), ("p50", "p50 delay rel. error", "pct"),
                            ("p95", "p95 delay rel. error", "pct"), ("p99", "p99 delay rel. error", "pct"),
                            ("drop_diff", "drop rate NR − LENA", "pp"), ("goodput", "cell goodput rel. error", "pct"),
                            ("ue_goodput", "per-UE goodput \\|rel. error\\| (median over UEs)", "pct"),
                            ("bler_diff", "first-tx BLER NR − LENA", "pp"), ("retx_diff", "retx TB fraction NR − LENA", "pp"),
                            ("retx_tbs", "retx TB count rel. error (LENA ≥ 20 retx)", "pct"), ("prb", "PRB utilization rel. error", "pct")):
        body.append([name, a[f"{key}_n"], f(a[f"{key}_med"], kind),
                     f(a[f"{key}_absmed"], "apct" if kind in ("pct",) else ("ms" if kind == "ms" else ("pp" if kind == "pp" else "x"))),
                     f(a[f"{key}_abs90"], "apct" if kind in ("pct",) else ("ms" if kind == "ms" else ("pp" if kind == "pp" else "x")))])
    table(["Metric", "Runs", "Median (signed)", "Median \\|·\\|", "90th pct \\|·\\|"], body)
    print("## Stratified (primary)\n")
    stratified("summary_primary.csv")
    print("## N x regime (primary)\n")
    table(STRAT_H, [strat_row(r, r["value"]) for r in rows("summary_primary.csv") if r["group"] == "N_x_regime"])
    print("## Hold-out seed only (primary, seed 2)\n")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from compare import agg
    pr = [dict(r, seed=int(r["seed"])) for r in rows("per_run_primary.csv")]
    ho = [r for r in pr if r["seed"] == 2]
    body = [strat_row(agg(ho, "all", "", "all"), "all hold-out runs")]
    for g in ("light", "moderate", "saturated"):
        sub = [r for r in ho if r["regime"] == g]
        body.append(strat_row(agg(sub, g, "", g), f"regime = {g}"))
    for n in sorted({int(r["N"]) for r in ho}):
        sub = [r for r in ho if int(r["N"]) == n]
        body.append(strat_row(agg(sub, n, "", n), f"N = {n}"))
    table(STRAT_H, body)
    print("## SR -> grant fit / hold-out\n")
    fr = rows("sr_fit_holdout.csv")
    vals = sorted({int(r["sr_grant_delay_slots"]) for r in fr})
    body = []
    for v in vals:
        ft = next(r for r in fr if int(r["sr_grant_delay_slots"]) == v and r["part"] == "fit")
        ho = next(r for r in fr if int(r["sr_grant_delay_slots"]) == v and r["part"] == "holdout")
        body.append([f"{v} ({v / 2:.1f} ms)" + (" **chosen**" if ft["chosen_on_fit"] == "True" else ""),
                     f(ft["objective"], "x"), f(ft["p50_med"]), f(ft["p95_med"]), f(ft["ks_med"], "x"),
                     f(ho["p50_med"]), f(ho["p95_med"]), f(ho["p50_absmed"], "apct"), f(ho["p95_absmed"], "apct"),
                     f(ho["ks_med"], "x"), f(ho["w1_ms_med"], "ms")])
    table(["SR→PUSCH slots", "Fit objective", "Fit p50", "Fit p95", "Fit KS", "Hold-out p50", "Hold-out p95",
           "Hold-out \\|p50\\|", "Hold-out \\|p95\\|", "Hold-out KS", "Hold-out W1 ms"], body)
    print("## Edge / centre (primary)\n")
    body = []
    for r in rows("edge_centre_primary.csv"):
        body.append([r["cls"], r["ues"], f(r["snr1_db_med"], "ms"), f(r["ue_p50_relerr_med"]), f(r["ue_p95_relerr_med"]),
                     f(r["ue_p95_absrel_med"], "apct"), f(r["pooled_ks_med"], "x"), f(r["ue_goodput_absrel_90"], "apct"),
                     f"{int(r['lena_retx_tbs'])} / {float(r['nr_retx_tbs']):.0f}",
                     f"{float(r['nr_tbs']) / float(r['lena_tbs']):.2f}"])
    table(["Class", "UEs", "snr1 median dB", "UE p50 err", "UE p95 err", "UE \\|p95 err\\|", "Pooled KS",
           "UE goodput \\|err\\| 90th", "retx TBs LENA / NR", "TBs NR / LENA"], body)
    print("## Ablation (one switch each, all 153 runs)\n")
    ab = {r["group"]: r for r in rows("ablation.csv")}
    order = ["primary", "olla_on", "phr_cap_wholeband", "ul_power_alloc", "ul_power_alloc_phr", "harq1", "sr_default3",
             "sr_measured14", "bler_sionna", "legacy_graph"]
    body = []
    for k in order + [k for k in ab if k not in order and not k.startswith("sr_grid")]:
        if k not in ab:
            continue
        r = ab[k]
        body.append([k, f(r["ks_med"], "x"), f(r["w1_ms_med"], "ms"), f(r["p50_med"]), f(r["p50_absmed"], "apct"),
                     f(r["p95_med"]), f(r["p95_absmed"], "apct"), f(r["drop_diff_absmed"], "pp"),
                     f(r["goodput_absmed"], "apct"), f(r["bler_diff_med"], "pp"), f(r["prb_med"])])
    table(["Arm", "KS", "W1 ms", "p50 err", "\\|p50\\|", "p95 err", "\\|p95\\|", "\\|drop Δ\\|", "\\|goodput\\|",
           "BLER Δ", "PRB err"], body)
    print("## Legacy engine (L2-legacy, graph) stratified\n")
    stratified("summary_legacy_graph.csv", keys=("all", "regime", "N"))
    print("## Fading arm\n")
    for arm in ("fade_matched", "fade_olla", "legacy_graph"):
        print(f"### {arm}\n")
        stratified(f"fade_summary_{arm}.csv", keys=("all", "N", "load"))
    fr = rows("fade_per_run_fade_matched.csv")
    if fr:
        la = rows("lena_runs_fade.csv")
        print("LENA fading arm: drop median", np.median([float(r["drop"]) for r in la]),
              "first-tx BLER median", np.median([float(r["first_tx_bler"]) for r in la]), "\n")
        for arm in ("fade_matched", "fade_olla", "legacy_graph"):
            pr = rows(f"fade_per_run_{arm}.csv")
            print(arm, "NR drop median", np.median([float(r["nr_drop"]) for r in pr]),
                  "NR BLER median", np.nanmedian([float(r["nr_first_tx_bler"]) for r in pr]), "\n")
    print("## Speed\n")
    lt = rows("timing_lena.csv")
    nt = rows("timing_nr.csv")
    body = []
    for n in sorted({int(r["nUe"]) for r in lt}):
        for S in (4000, 30000):
            le = [float(r["wall_per_sim_s"]) for r in lt if int(r["nUe"]) == n and int(r["frameBytes"]) == S and r["macTraces"] == "0"]
            cpu1 = [float(r["wall_per_sim_s"]) for r in nt if int(r["N"]) == n and int(r["S"]) == S and r["device"] == "cpu"
                    and r["E"] == "1" and r["threads"] == "1"]
            cpu16 = [float(r["wall_per_sim_s_per_env"]) for r in nt if int(r["N"]) == n and int(r["S"]) == S
                     and r["device"] == "cpu" and r["E"] == "16" and r["threads"] == "1"]
            body.append([n, S, f"{le[0]:.3f}" if le else "–", f"{cpu1[0]:.3f}" if cpu1 else "–",
                         f"{cpu1[0] / le[0]:.1f}×" if le and cpu1 else "–", f"{cpu16[0]:.3f}" if cpu16 else "–"])
    table(["N", "S (B)", "LENA s/s (1 core, traces off)", "NR ref. s/s (CPU, 1 thread, E = 1)", "NR / LENA",
           "NR ref. s/s per env (CPU, 1 thread, E = 16)"], body)


if __name__ == "__main__":
    main()
