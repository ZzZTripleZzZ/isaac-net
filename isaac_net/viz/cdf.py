"""Delay and age-of-information CDFs, one curve per level or configuration, in the paper's style.

    from isaac_net.viz.cdf import plot_delay_cdf, plot_aoi_cdf
    plot_delay_cdf(["runs/l2", "runs/legacy", "runs/l0"], by="level", path="delay_cdf.png")   # RecorderLoop dirs
    plot_delay_cdf("benchmarks/results/closedloop/frames.csv.gz", by="level")                # frame table, arm col.
    plot_delay_cdf({"L2": delays_ms, "ns3": lena_ms})                                         # samples in ms

Inputs (one or a list): a RecorderLoop output directory (or the RecorderLoop itself), whose delay_hist / aoi_hist
tables give the CDF to within one histogram bin (about 6%); a frame table (DataFrame, or a CSV / CSV.gz / Parquet
path) with a delay_ms or delay_s column (aoi_s / aoi_ms for AoI), grouped by the column `by` (by="level" falls
back to an "arm" column, by="config" to "label"); or a dict {name: samples in ms}. Several inputs with the same
group name are pooled. by="level" names a recorder run by its level, by="config" by its RecordConfig.label.
"""
from __future__ import annotations

import numpy as np

from . import _data
from . import style as S


def _plot_cdf(data, by, kind, ax, path, log, xlabel, unit, title, xlim, legend, quantiles):
    dists = _data.distributions(data, by, kind)
    if not dists:
        raise ValueError("no data to plot")
    scale = {"ms": 1.0, "s": 1e-3}[unit]
    with S.paper_style():
        fig, ax = S.new_axes(ax)
        sty = S.styles_for(list(dists))
        lo, hi = np.inf, 0.0
        for name, item in dists.items():
            x, y = _data.ecdf(item)
            if x.size == 0:
                continue
            x = x * scale
            c, ls = sty[name]
            strong = str(name).lower() in ("l2", "ns3", "ns-3", "lena", "5g-lena")
            ax.plot(x, y, color=c, ls=ls, lw=1.3 if strong else 1.1, drawstyle="steps-post" if item[0] == "samples"
                    else "default", label=f"{name} (n = {_data.count(item):,})", zorder=3 if strong else 2)
            pos = x[x > 0]
            if pos.size:
                lo = min(lo, np.quantile(pos, 0.002))
                hi = max(hi, pos.max())
            for q in quantiles:
                v = _data.quantile(item, q) * scale
                if np.isfinite(v) and v > 0:
                    ax.plot([v], [q], marker="o", ms=2.5, color=c, zorder=4)
        log = log and np.isfinite(lo) and hi > 0 and hi / max(lo, 1e-12) >= 20     # under 1.3 decades: linear
        if log:
            ax.set_xscale("log")
            ax.set_xlim(*(xlim or (lo * 0.8, hi * 1.25)))
        elif xlim:
            ax.set_xlim(*xlim)
        ax.set_ylim(0, 1.0)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_ylabel("CDF")
        ax.set_xlabel(f"{xlabel} ({unit}{', log scale' if log else ''})")
        if title:
            ax.set_title(title)
        S.grid(ax, "both")
        if legend:
            ax.legend(loc="best", handlelength=2.0, labelspacing=0.25)
        S.finish(fig, path)
    return ax


def plot_delay_cdf(frames_or_records, by: str = "level", *, ax=None, path=None, log=True, unit="ms", title=None,
                   xlim=None, legend=True, quantiles=(0.5, 0.95)):
    """CDF of the per-message delay (capture to delivery) of every group. quantiles: marked with dots."""
    return _plot_cdf(frames_or_records, by, "delay", ax, path, log, "Message delay", unit, title, xlim, legend,
                     quantiles)


def plot_aoi_cdf(frames_or_records, by: str = "level", *, ax=None, path=None, log=True, unit="s", title=None,
                 xlim=None, legend=True, quantiles=(0.5, 0.95)):
    """CDF of the age of information over robots and control steps of every group."""
    return _plot_cdf(frames_or_records, by, "aoi", ax, path, log, "Age of information", unit, title, xlim, legend,
                     quantiles)
