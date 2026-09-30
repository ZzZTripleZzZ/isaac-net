"""Delay CDF of the closed-loop run (docs/img/closed-loop-delay-cdf.png) from frames.csv.gz.

usage: python benchmarks/closedloop/plot_cdf.py benchmarks/results/closedloop docs/img/closed-loop-delay-cdf.png
"""
import csv
import gzip
import os
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

STYLE = {"L2": ("#1f77b4", "-"), "ns3": ("#000000", "--"), "L0": ("#2ca02c", "-"),
         "L1": ("#d62728", "-"), "L2-legacy": ("#9467bd", "-")}
LABEL = {"L2": "L2 (NR engine, lena_validation_v2)", "ns3": "ns-3 5G-LENA (lockstep bridge)",
         "L0": "L0 (i.i.d., fitted to L2)", "L1": "L1 (fluid)", "L2-legacy": "L2-legacy"}


def main(res_dir, out_png):
    d = defaultdict(list)
    with gzip.open(os.path.join(res_dir, "frames.csv.gz"), "rt") as f:
        for r in csv.DictReader(f):
            d[r["arm"]].append(float(r["delay_ms"]))
    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=150)
    for arm in ("L0", "L1", "L2-legacy", "L2", "ns3"):
        if arm not in d:
            continue
        x = np.sort(np.asarray(d[arm]))
        c, ls = STYLE[arm]
        ax.plot(x, np.arange(1, len(x) + 1) / len(x), color=c, ls=ls, lw=1.8 if arm in ("L2", "ns3") else 1.3,
                label=f"{LABEL[arm]}, n = {len(x):,}")
    ax.set_xlim(0, 800)
    ax.set_ylim(0, 1)
    ax.set_xlabel("frame delay (ms), capture to delivery")
    ax.set_ylabel("CDF over delivered frames")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    ax.set_title("Same scripted controller, 8 envs x 16 robots x 300 steps, 5 seeds", fontsize=9)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
