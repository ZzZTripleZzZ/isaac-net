"""Tables of docs/fidelity-vs-lena.md, "Scale configurations", from compare.py outputs.

usage: python report_scalecfg.py <results dir> [<out dir>]
Reads per_run_<arm>.csv of the GPU replay arms in <results dir>/scalecfg and of the CPU reference arms
(primary_eng = v1, v2) in <results dir>/loadfix_v2; writes <out>/scalecfg_by_regime.csv (default <results dir>/scalecfg)
and prints the markdown table: per arm and load regime the signed median of the per-run p50 and p95 relative errors,
the median absolute p50 error, the median drop-rate difference, the median KS distance and the median PRB-use error.
"""
import csv
import os
import sys

import numpy as np

REG = ("light", "moderate", "saturated")
ARMS = [("loadfix_v2", "primary_eng", "v1 (`lena_validation()`), CPU reference"),
        ("loadfix_v2", "v2", "v2 (`lena_validation_v2()`), CPU reference"),
        ("scalecfg", "v1_triton", "v1, triton"),
        ("scalecfg", "v2_graph", "v2, graph"),
        ("scalecfg", "nrconfig", "`NRConfig()`, triton"),
        ("scalecfg", "nrconfig_nofade", "`NRConfig(fading=False)`, triton"),
        ("scalecfg", "v2_lumped40", "v2 minus BSR, triton"),
        ("scalecfg", "v2_lumped40_fb16", "v2 minus BSR, 16-frame buffer, triton")]


def med(v, absval=False):
    x = np.array([float(a) for a in v], dtype=np.float64)
    x = x[np.isfinite(x)]
    if absval:
        x = np.abs(x)
    return float(np.median(x)) if len(x) else float("nan")


def main(res, out):
    rows = []
    for sub, arm, label in ARMS:
        path = os.path.join(res, sub, f"per_run_{arm}.csv")
        if not os.path.exists(path):
            continue
        pr = list(csv.DictReader(open(path)))
        for g in ("all",) + REG:
            rs = pr if g == "all" else [r for r in pr if r["regime"] == g]
            col = lambda c: [r[c] for r in rs]
            rows.append(dict(arm=arm, label=label, regime=g, runs=len(rs), p50_med=med(col("p50_relerr")),
                             p50_absmed=med(col("p50_relerr"), True), p95_med=med(col("p95_relerr")),
                             p95_absmed=med(col("p95_relerr"), True), drop_med=med(col("drop_diff")),
                             ks_med=med(col("ks")), prb_med=med(col("prb_relerr")), w1_ms_med=med(col("w1_ms")),
                             bler_diff_med=med(col("bler_diff"))))
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "scalecfg_by_regime.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})
    print("| Configuration | Regime | Runs | p50 err | \\|p50\\| | p95 err | Drop Δ pp | KS | PRB err |")
    print("|:---|:---|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        print(f"| {r['label']} | {r['regime']} | {r['runs']} | {r['p50_med']:+.1%} | {r['p50_absmed']:.1%} | "
              f"{r['p95_med']:+.1%} | {100 * r['drop_med']:+.2f} | {r['ks_med']:.3f} | {r['prb_med']:+.1%} |")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else os.path.join(sys.argv[1], "scalecfg"))
