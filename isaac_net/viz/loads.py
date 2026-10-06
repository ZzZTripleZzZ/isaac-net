"""Load plots: delivered vs offered throughput, and per-cell PRB utilization over time.

    from isaac_net.viz.loads import plot_throughput_vs_load, plot_cell_utilization
    plot_throughput_vs_load(["runs/l2", "runs/l0"], by="level", path="tput.png")      # RecorderLoop dirs
    plot_throughput_vs_load("results/", by="baseline")                                # benchmark result files
    plot_cell_utilization("runs/multicell", path="cells.png")                          # cells x time heatmap
"""
from __future__ import annotations

import os

import numpy as np

from . import _data
from . import style as S


def _bench_frame(paths):
    """Benchmark result files (isaac_net.bench) as a DataFrame: one row per run with the group keys and metrics."""
    import pandas as pd
    from ..bench.report import load_results
    rows = []
    for r in load_results(paths if isinstance(paths, (list, tuple)) else [paths]):
        rows.append({**{k: r.get(k) for k in ("task", "variant", "level", "backend", "baseline", "seed", "label",
                                                 "preset", "traffic")}, **r["eval"]["metrics"]})
    return pd.DataFrame(rows)


def throughput_frame(data, by="level"):
    """(DataFrame with offered_mbps, delivered_mbps and a group column "group") from recorder dirs, benchmark
    results, or a table that already has the two columns."""
    import pandas as pd
    from ..core.record import read_meta, read_records
    frames = []
    for item in (list(data) if isinstance(data, (list, tuple)) else [data]):
        rd = _data.record_dir_of(item)
        if rd is not None:
            meta = read_meta(rd)
            st = read_records(rd).groupby("env")[["steps", "offered_bytes", "delivered_bytes"]].sum()
            sec = st.steps.to_numpy(float) * meta["control_step_ms"] * 1e-3
            frames.append(pd.DataFrame({"offered_mbps": st.offered_bytes.to_numpy(float) * 8e-6 / sec,
                                        "delivered_mbps": st.delivered_bytes.to_numpy(float) * 8e-6 / sec,
                                        "group": _data.run_name(meta, by, rd), "run": os.path.abspath(rd)}))
            continue
        if isinstance(item, (str, os.PathLike)) and (os.path.isdir(item) or str(item).endswith(".json")):
            df = _bench_frame(item)
        else:
            df = _data.read_table(item)
        g = _data.group_column(df, by)
        df = df.copy()
        df["group"] = df[g].astype(str) if g else "all"
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def plot_throughput_vs_load(data, by: str = "level", *, x="offered_mbps", y="delivered_mbps", bins=8, ax=None,
                            path=None, title=None, legend=True):
    """Delivered vs offered throughput per group, with the y = x line. Recorder directories give one point per run
    (whole-run rates, mean over envs, whiskers = 10th-90th percentile over envs), so a sweep of runs at several loads
    traces each group's curve. Tables with many points per group are binned by offered load (equal-count bins, mean
    of each bin, whiskers = 10th-90th percentile of y); a few points (benchmark runs) are drawn as they are."""
    df = throughput_frame(data, by)
    if df.empty or x not in df or y not in df:
        raise ValueError(f"no {x} / {y} values to plot")
    with S.paper_style():
        fig, ax = S.new_axes(ax)
        groups = list(dict.fromkeys(df["group"]))
        sty = S.styles_for(groups)
        top = 0.0
        for gname in groups:
            d = df[df.group == gname]
            xs, ys = d[x].to_numpy(float), d[y].to_numpy(float)
            ok = np.isfinite(xs) & np.isfinite(ys)
            xs, ys = xs[ok], ys[ok]
            if xs.size == 0:
                continue
            c, ls = sty[gname]
            if "run" in d and d["run"].notna().all():
                agg = d.groupby("run").agg(x=(x, "mean"), y=(y, "mean"), lo=(y, lambda v: v.quantile(0.1)),
                                           hi=(y, lambda v: v.quantile(0.9))).sort_values("x")
                ax_, ay, alo, ahi = (agg[k].to_numpy(float) for k in ("x", "y", "lo", "hi"))
                ax.errorbar(ax_, ay, yerr=np.vstack([ay - alo, ahi - ay]), color=c, ls=ls, marker="o", ms=3,
                            lw=1.1, elinewidth=0.6, capsize=1.5, label=str(gname))
            elif xs.size > 4 * bins:
                edges = np.unique(np.quantile(xs, np.linspace(0, 1, bins + 1)))
                k = np.clip(np.searchsorted(edges, xs, side="right") - 1, 0, max(edges.size - 2, 0))
                mx = np.array([xs[k == i].mean() for i in range(edges.size - 1) if (k == i).any()])
                my = np.array([ys[k == i].mean() for i in range(edges.size - 1) if (k == i).any()])
                lo = np.array([np.quantile(ys[k == i], 0.1) for i in range(edges.size - 1) if (k == i).any()])
                hi = np.array([np.quantile(ys[k == i], 0.9) for i in range(edges.size - 1) if (k == i).any()])
                ax.errorbar(mx, my, yerr=np.vstack([my - lo, hi - my]), color=c, ls=ls, marker="o", ms=2.5, lw=1.1,
                            elinewidth=0.6, capsize=1.5, label=str(gname))
            else:
                o = np.argsort(xs)
                ax.plot(xs[o], ys[o], color=c, ls=ls, marker="o", ms=3, lw=1.0, label=str(gname))
            top = max(top, np.nanmax(xs), np.nanmax(ys))
        ax.plot([0, top * 1.05], [0, top * 1.05], color=S.GREY, lw=0.6, ls=":", zorder=1, label="delivered = offered")
        ax.set_xlim(0, top * 1.05 if top > 0 else 1)
        ax.set_ylim(0, top * 1.05 if top > 0 else 1)
        ax.set_xlabel("Offered load (Mbit/s per env)" if x == "offered_mbps" else x)
        ax.set_ylabel("Delivered (Mbit/s per env)" if y == "delivered_mbps" else y)
        if title:
            ax.set_title(title)
        S.grid(ax, "both")
        if legend:
            ax.legend(loc="upper left", labelspacing=0.25)
        S.finish(fig, path)
    return ax


def plot_cell_utilization(records, *, env=None, include_background=True, ax=None, path=None, title=None,
                          cmap="viridis", vmax=1.0):
    """Heatmap of each cell's uplink PRB utilization over time from a RecorderLoop directory (cells table). env:
    one env id, or None for the mean over envs. include_background adds the background users' share (bg_util)
    to the robots' share (prb_util); without the NR engine's PRB counters (levels other than L2) only bg_util
    is available."""
    from ..core.record import read_meta, read_records
    rd = _data.record_dir_of(records)
    if rd is None:
        raise ValueError("plot_cell_utilization needs a RecorderLoop output directory (per_cell=True)")
    meta = read_meta(rd)
    ce = read_records(rd, "cells")
    if ce.empty:
        raise ValueError(f"{rd} has no cells table (RecordConfig(per_cell=True))")
    if env is not None:
        ce = ce[ce.env == int(env)]
    util = ce.prb_util.to_numpy(float)
    if include_background:
        bg = ce.bg_util.to_numpy(float)
        util = np.where(np.isfinite(util), util, 0.0) + np.where(np.isfinite(bg), bg, 0.0)
        util = np.where(np.isfinite(ce.prb_util.to_numpy(float)) | np.isfinite(bg), util, np.nan)
    ce = ce.assign(u=util)
    if not np.isfinite(util).any():
        raise ValueError("no PRB utilization in the cells table (level L2 records prb_util; background users bg_util)")
    tab = ce.groupby(["cell", "step"]).u.mean().unstack("step")
    steps = tab.columns.to_numpy(float)
    dt = meta["control_step_ms"] * 1e-3
    C = tab.shape[0]
    with S.paper_style():
        fig, ax = S.new_axes(ax, figsize=(S.COL_W, max(1.2, 0.35 * C + 0.8)))
        t1 = steps * dt
        t0 = np.r_[0.0, t1[:-1]]
        ext = (t0[0], t1[-1], -0.5, C - 0.5)
        im = ax.imshow(tab.to_numpy(float), aspect="auto", origin="lower", extent=ext, cmap=cmap, vmin=0.0,
                       vmax=vmax, interpolation="nearest")
        ax.set_yticks(range(C))
        ax.set_yticklabels([f"cell {int(c)}" for c in tab.index])
        ax.set_xlabel("Time (s)")
        what = "robots + background" if include_background else "robots"
        cb = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.05)
        cb.set_label("UL PRB utilization")
        cb.outline.set_linewidth(0.4)
        scope = f"env {env}" if env is not None else f"mean over {meta['E']} envs"
        ax.set_title(title or f"{scope}, {what}")
        S.finish(fig, path)
    return ax
