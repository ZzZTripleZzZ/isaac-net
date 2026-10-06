"""Engine vs ns-3 5G-LENA: delay CDF overlay and per-run quantile errors, from the fidelity study's CSV formats.

    from isaac_net.viz.compare import plot_vs_ns3
    # frame tables: one CSV with an "arm" column (benchmarks/results/closedloop/frames.csv.gz), or two CSVs
    plot_vs_ns3("benchmarks/results/closedloop/frames.csv.gz", engine_arm=["L2", "L0"],
                per_run_csv="benchmarks/fidelity/results/loadfix_v2/per_run_v2.csv", path="vs_ns3.png")
    plot_vs_ns3(None, per_run_csv="benchmarks/fidelity/results/loadfix_v2/per_run_v2.csv")   # error bars only

Frame tables have a delay_ms column (delay_s also works), one row per delivered frame. Per-run tables are the
per_run_<arm>.csv files of benchmarks/fidelity/compare.py: lena_<q>_ms, nr_<q>_ms and <q>_relerr per run, with a
regime column (light / moderate / saturated). The error panel shows, per regime and over all runs, the median
relative error (NR - LENA) / LENA of each quantile, with whiskers from the 25th to the 75th percentile over runs.
"""
from __future__ import annotations

import numpy as np

from . import _data
from . import style as S

LENA_ARMS = ("ns3", "ns-3", "lena", "5g-lena", "ns-3 5g-lena")
REGIMES = ("light", "moderate", "saturated")


def _is_per_run(df) -> bool:
    return any(c.endswith("_relerr") for c in df.columns) and any(c.startswith("lena_") for c in df.columns)


def _ks(a, b):
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    a, b = np.sort(a), np.sort(b)
    x = np.concatenate([a, b])
    return float(np.max(np.abs(np.searchsorted(a, x, "right") / len(a) - np.searchsorted(b, x, "right") / len(b))))


def _frames(engine_csv, lena_csv, engine_arm, lena_arm):
    """{name: samples in ms} with the LENA arm first under its own name."""
    if engine_csv is None:
        return {}, None
    eng = _data.read_table(engine_csv)
    col, scale = _data._value_column(eng, "delay")
    out, ref = {}, None
    if lena_csv is not None:
        le = _data.read_table(lena_csv)
        lc, ls = _data._value_column(le, "delay")
        ref = "ns-3 5G-LENA"
        out[ref] = le[lc].to_numpy(float) * ls
        arms = [engine_arm] if isinstance(engine_arm, str) else engine_arm
        if "arm" in eng.columns and arms:
            for a in arms:
                out[str(a)] = eng.loc[eng.arm == a, col].to_numpy(float) * scale
        else:
            out[str(engine_arm or "engine")] = eng[col].to_numpy(float) * scale
        return out, ref
    if "arm" not in eng.columns:
        raise ValueError("one frame table needs an 'arm' column to split engine and ns-3 rows; or pass lena_csv")
    names = list(dict.fromkeys(eng.arm.astype(str)))
    lena = [lena_arm] if lena_arm else [a for a in names if a.lower() in LENA_ARMS]
    if not lena:
        raise ValueError(f"no ns-3 arm among {names}; pass lena_arm=")
    ref = lena[0]
    out[ref] = eng.loc[eng.arm.astype(str) == ref, col].to_numpy(float) * scale
    if engine_arm is None:
        arms = ["L2"] if "L2" in names else [a for a in names if a != ref and a != "ideal"]
    else:
        arms = [engine_arm] if isinstance(engine_arm, str) else list(engine_arm)
    for a in arms:
        out[str(a)] = eng.loc[eng.arm.astype(str) == str(a), col].to_numpy(float) * scale
    return out, ref


def plot_vs_ns3(engine_csv, lena_csv=None, *, per_run_csv=None, engine_arm=None, lena_arm=None,
                quantiles=("p50", "p95", "p99"), axes=None, path=None, log=True, title=None):
    """Delay CDF of the engine arm(s) over 5G-LENA's (with the KS distance of each arm) and, with a per-run table,
    the per-run quantile errors by load regime. engine_csv may itself be a per-run table. Returns the Axes array
    (CDF panel first when drawn)."""
    if engine_csv is not None and per_run_csv is None:
        df = _data.read_table(engine_csv)
        if _is_per_run(df):
            engine_csv, per_run_csv = None, df
    frames, ref = _frames(engine_csv, lena_csv, engine_arm, lena_arm)
    pr = _data.read_table(per_run_csv) if per_run_csv is not None else None
    n = int(bool(frames)) + int(pr is not None)
    if n == 0:
        raise ValueError("nothing to plot: pass frame tables and / or per_run_csv")
    plt = S.pyplot()
    with S.paper_style():
        if axes is None:
            fig, axs = plt.subplots(1, n, figsize=(S.COL_W * (1.0 if n == 1 else 2.0), 2.0),
                                    constrained_layout=True, squeeze=False)
            axs = list(axs[0])
        else:
            axs = list(np.atleast_1d(axes))
            fig = axs[0].figure
        k = 0
        if frames:
            ax = axs[k]
            k += 1
            sty = S.styles_for(list(frames))
            notes = []
            for name, v in frames.items():
                v = v[np.isfinite(v)]
                if v.size == 0:
                    continue
                x = np.sort(v)
                y = np.arange(1, x.size + 1) / x.size
                c, ls = sty[name]
                is_ref = name == ref
                ax.step(np.r_[x[0], x], np.r_[0, y], where="post", color=c, ls=ls, lw=1.3 if is_ref else 1.1,
                        label=("5G-LENA" if is_ref and name.lower() in ("ns3", "ns-3") else name),
                        zorder=3 if name == "L2" else 2)
                if not is_ref:
                    notes.append(f"{name} {_ks(v, frames[ref][np.isfinite(frames[ref])]):.2f}")
            pos = np.concatenate([v[v > 0] for v in frames.values() if (v > 0).any()] or [np.zeros(0)])
            if log and pos.size and pos.max() / np.quantile(pos, 0.002) >= 20:
                ax.set_xscale("log")
                ax.set_xlim(np.quantile(pos, 0.002) * 0.8, pos.max() * 1.25)
            else:
                log = False
            ax.set_ylim(0, 1.0)
            ax.set_yticks([0, 0.5, 1])
            ax.set_xlabel(f"Message delay (ms{', log scale' if log else ''})")
            ax.set_ylabel("CDF")
            S.grid(ax, "both")
            ax.legend(loc="upper left", labelspacing=0.25)
            if notes:
                ax.text(0.98, 0.03, "KS vs 5G-LENA\n" + "\n".join(notes), transform=ax.transAxes, ha="right",
                        va="bottom", linespacing=1.05, bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.8))
        if pr is not None:
            ax = axs[k]
            groups = [g for g in REGIMES if "regime" in pr and (pr.regime == g).any()] + ["all"]
            qs = [q for q in quantiles if f"{q}_relerr" in pr]
            width = 0.8 / max(len(qs), 1)
            qcol = {"p50": S.BLUE, "p95": S.ORANGE, "p99": S.VERMILION}
            for j, q in enumerate(qs):
                med, lo, hi, xs = [], [], [], []
                for i, g in enumerate(groups):
                    v = pr[f"{q}_relerr"] if g == "all" else pr.loc[pr.regime == g, f"{q}_relerr"]
                    v = 100 * v.to_numpy(float)
                    v = v[np.isfinite(v)]
                    if not v.size:
                        continue
                    m = np.median(v)
                    med.append(m)
                    lo.append(m - np.quantile(v, 0.25))
                    hi.append(np.quantile(v, 0.75) - m)
                    xs.append(i - 0.4 + width * (j + 0.5))
                ax.errorbar(xs, med, yerr=np.vstack([lo, hi]), ls="none", marker="o", ms=3, color=qcol.get(q, S.CYCLE[j]),
                            elinewidth=0.8, capsize=1.5, label=q)
            ax.axhline(0, color=S.GREY, lw=0.5, zorder=0)
            counts = {g: (len(pr) if g == "all" else int((pr.regime == g).sum())) for g in groups}
            ax.set_xticks(range(len(groups)))
            ax.set_xticklabels([f"{g}\n({counts[g]})" for g in groups])
            ax.set_xlim(-0.6, len(groups) - 0.4)
            ax.set_ylabel("Delay error vs 5G-LENA (%)")
            S.grid(ax, "y")
            ax.legend(loc="best", ncol=len(qs), columnspacing=0.8, handletextpad=0.2)
        if title:
            fig.suptitle(title)
        S.finish(fig, path)
    return np.asarray(axs[:n])
