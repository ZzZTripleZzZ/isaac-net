"""Five fidelity columns by load regime from compare.py's per_run_<arm>.csv (medians over runs).

Columns: relative p50 and p95 delay error (NR - LENA) / LENA, drop-rate difference in percentage points, KS and
W1 (ms) between the delay distributions. Regimes by 5G-LENA's drop rate: light < 1%, moderate 1-20%,
saturated >= 20%. Also prints the median first-transmission BLER on both sides.

usage: python summarize.py <per_run csv> [label] >> table.md
"""
import csv
import sys

import numpy as np


def main(path, label=""):
    rows = list(csv.DictReader(open(path)))
    print("| Set | Regime | Runs | p50 | p95 | Drop (pp) | KS | W1 (ms) | BLER LENA / NR (%) |")
    print("|:---|:---|---:|---:|---:|---:|---:|---:|---:|")
    for g in ("light", "moderate", "saturated", "all"):
        sub = [r for r in rows if g == "all" or r["regime"] == g]
        if not sub:
            continue
        med = lambda k: float(np.nanmedian([float(r[k]) for r in sub]))  # noqa: E731
        print(f"| {label} | {g} | {len(sub)} | {100 * med('p50_relerr'):+.1f}% | {100 * med('p95_relerr'):+.1f}% | "
              f"{100 * med('drop_diff'):+.2f} | {med('ks'):.2f} | {med('w1_ms'):.1f} | "
              f"{100 * med('lena_first_tx_bler'):.2f} / {100 * med('nr_first_tx_bler'):.2f} |")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "")
