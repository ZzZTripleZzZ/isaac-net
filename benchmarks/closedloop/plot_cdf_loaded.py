"""Two-panel delay CDF of the loaded closed-loop runs (docs/img/closed-loop-loaded-cdf.png) from frames.csv.gz.

usage: python benchmarks/closedloop/plot_cdf_loaded.py benchmarks/results/closedloop_loaded \
           docs/img/closed-loop-loaded-cdf.png
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

from plot_cdf import LABEL, STYLE  # noqa: E402

PANELS = (("interval1_8x16", "a) one frame per 0.2 s, 8 envs x 16 robots", 1000),
          ("r32_8x32", "b) one frame per 0.6 s, 8 envs x 32 robots", 1600))


def main(res_dir, out_png):
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0), dpi=150, sharey=True)
    for ax, (sub, title, xmax) in zip(axes, PANELS):
        d = defaultdict(list)
        with gzip.open(os.path.join(res_dir, sub, "frames.csv.gz"), "rt") as f:
            for r in csv.DictReader(f):
                d[r["arm"]].append(float(r["delay_ms"]))
        for arm in ("L0", "L1", "L2-legacy", "L2", "ns3"):
            if arm not in d:
                continue
            x = np.sort(np.asarray(d[arm]))
            c, ls = STYLE[arm]
            ax.plot(x, np.arange(1, len(x) + 1) / len(x), color=c, ls=ls, lw=1.8 if arm in ("L2", "ns3") else 1.3,
                    label=LABEL[arm])
        ax.set_xlim(0, xmax)
        ax.set_ylim(0, 1)
        ax.set_xlabel("frame delay (ms), capture to delivery")
        ax.grid(alpha=0.3)
        ax.set_title(title, fontsize=9)
    axes[0].set_ylabel("CDF over delivered frames, 5 seeds")
    axes[1].legend(loc="lower right", fontsize=8, frameon=False)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png)


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    main(sys.argv[1], sys.argv[2])
