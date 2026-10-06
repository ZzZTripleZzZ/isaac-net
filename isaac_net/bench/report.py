"""Aggregate result files into tables: mean and 95% confidence interval over seeds.

    rows = aggregate(load_results(["results/"]))
    print(markdown(rows, metrics=["task", "delay_p95_ms", "aoi_mean_s"]))
    html(results, rows, "report.html")             # self-contained HTML with figures (viz extra: matplotlib)
    figs = figures(results, rows, "report_fig/")   # the same figures as PNGs, for markdown_figures(figs)

One group = (task, variant, sim, level, backend, preset, traffic, baseline, label). A file without a
sim, preset or traffic field counts as the default ("torch", "default", "policy"). Each result file contributes one
value per metric (its evaluation mean over envs and episodes); the table shows the mean over the seeds and the
half-width of the two-sided 95% Student t interval, with n = the number of seeds. The pseudo-metric "task"
stands for each task's own metric.
"""
from __future__ import annotations

import glob
import json
import math
import os
from typing import Iterable, Sequence

from .spec import RESULT_SCHEMA, mean_ci95

GROUP_KEYS = ("task", "variant", "sim", "level", "backend", "preset", "traffic", "baseline", "label")
_KEY_DEFAULTS = {"sim": "torch", "preset": "default", "traffic": "policy"}
# shown by markdown() next to its group_cols when they differ between rows, so no two lines look alike
_EXTRA_COLS = ("sim", "preset", "traffic", "label")
DEFAULT_METRICS = ("task", "return", "deliveries", "drops", "delay_p50_ms", "delay_p95_ms", "aoi_mean_s")


def load_results(paths: Iterable[str]) -> list:
    """Result dicts from files and directories (recursively, *.json); files of another schema are skipped."""
    files = []
    for p in paths:
        if os.path.isdir(p):
            files += sorted(glob.glob(os.path.join(p, "**", "*.json"), recursive=True))
        else:
            files.append(p)
    out = []
    for f in files:
        try:
            with open(f) as fh:
                r = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(r, dict) and r.get("schema") == RESULT_SCHEMA:
            out.append(r)
    return out


def aggregate(results: Sequence[dict]) -> list:
    """One row per group: the group keys, the task metric's key and direction, seeds, and {metric: (mean, ci, n)}
    for every metric found, plus the mean timings."""
    groups = {}
    for r in results:
        key = tuple(r.get(k) if r.get(k) is not None else _KEY_DEFAULTS.get(k, "") for k in GROUP_KEYS)
        groups.setdefault(key, []).append(r)
    rows = []
    for key, rs in sorted(groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        spec = rs[0]["task_spec"]["metric"]
        names = []
        for r in rs:
            for k in r["eval"]["metrics"]:
                if k not in names:
                    names.append(k)
        stats = {k: mean_ci95([r["eval"]["metrics"].get(k) for r in rs]) for k in names}
        stats["task"] = stats.get(spec["key"], (math.nan, math.nan, 0))
        tim = {k: mean_ci95([r["timing"][k] for r in rs])[0] for k in rs[0]["timing"]}
        rows.append({**dict(zip(GROUP_KEYS, key)), "metric_key": spec["key"],
                     "higher_is_better": spec["higher_is_better"], "seeds": sorted(r["seed"] for r in rs),
                     "stats": stats, "timing": tim})
    return rows


def _fmt(m, ci, n, digits=3):
    if n == 0 or m is None or not math.isfinite(m):
        return "n/a"
    s = f"{m:.{digits}g}"
    if math.isfinite(ci):
        s += f" ± {ci:.2g}"
    return s


def markdown(rows: Sequence[dict], metrics: Sequence[str] = DEFAULT_METRICS, timing: bool = True,
             group_cols: Sequence[str] = ("task", "variant", "level", "backend", "baseline")) -> str:
    """Markdown table: one line per group, mean ± 95% CI over seeds (n seeds in the "n" column). The columns sim,
    preset, traffic and label are added after group_cols when they are not the same in every row."""
    group_cols = list(group_cols) + [c for c in _EXTRA_COLS
                                     if c not in group_cols and len({str(r.get(c)) for r in rows}) > 1]
    head = list(group_cols) + ["n"] + [("task metric" if m == "task" else m) for m in metrics]
    if timing:
        head += ["train s", "eval s/step"]
    lines = ["| " + " | ".join(head) + " |", "|" + "|".join([":---"] * len(group_cols) + ["---:"] * (
        len(head) - len(group_cols))) + "|"]
    for r in rows:
        cells = [str(r[c]) for c in group_cols] + [str(len(r["seeds"]))]
        for m in metrics:
            st = r["stats"].get(m, (math.nan, math.nan, 0))
            c = _fmt(*st)
            if m == "task" and c != "n/a":
                c = f"{r['metric_key']} {c} {'↑' if r['higher_is_better'] else '↓'}"
            cells.append(c)
        if timing:
            cells += [f"{r['timing'].get('train_sec', 0.0):.0f}", f"{r['timing'].get('eval_sec_per_step', 0.0):.4f}"]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def to_json(rows: Sequence[dict]) -> list:
    out = []
    for r in rows:
        d = {k: r[k] for k in r if k != "stats"}
        d["stats"] = {k: {"mean": v[0], "ci95": v[1], "n": v[2]} for k, v in r["stats"].items()}
        out.append(d)
    return out


# ---------------------------------------------------------------------------------------------- figures and HTML
# Need the viz extra (matplotlib); the tables above do not.

def _group_name(r: dict, keys) -> str:
    return " / ".join(str(r.get(k) if r.get(k) is not None else _KEY_DEFAULTS.get(k, "")) for k in keys)


def _episode_values(results: Sequence[dict], metric: str) -> dict:
    """{baseline label: values} of one metric: the per env-episode evaluation rows when the result files keep them
    (run --keep_rows), else one value per result file (its evaluation mean)."""
    out = {}
    for r in results:
        rows = r["eval"].get("rows")
        vals = [x.get(metric) for x in rows] if rows else [r["eval"]["metrics"].get(metric)]
        vals = [v for v in vals if v is not None and math.isfinite(v)]
        name = r["baseline"] + (f" [{r['label']}]" if r.get("label") else "")
        out.setdefault(name, []).extend(vals)
    return out


def _panels(results: Sequence[dict]):
    """(title, results) per (task, variant, sim, level, backend, preset, traffic): one figure each."""
    keys = ("task", "variant", "sim", "level", "backend", "preset", "traffic")
    groups = {}
    for r in results:
        groups.setdefault(_group_name(r, keys), []).append(r)
    return sorted(groups.items())


def cdf_figure(results: Sequence[dict], title: str = ""):
    """Figure with the CDFs of per-episode delay p95 and mean AoI, one curve per baseline."""
    from ..viz import style as S
    from ..viz.cdf import plot_aoi_cdf, plot_delay_cdf
    plt = S.pyplot()
    with S.paper_style():
        fig, axs = plt.subplots(1, 2, figsize=(S.TEXT_W * 0.8, 2.2), constrained_layout=True)
    d = _episode_values(results, "delay_p95_ms")
    a = {k: [v * 1e3 for v in vs] for k, vs in _episode_values(results, "aoi_mean_s").items()}
    if any(d.values()):
        plot_delay_cdf({k: v for k, v in d.items() if v}, ax=axs[0], quantiles=())
        axs[0].set_xlabel("Per-episode delay p95 (ms" + (", log scale)" if axs[0].get_xscale() == "log" else ")"))
    if any(a.values()):
        plot_aoi_cdf({k: v for k, v in a.items() if v}, ax=axs[1], quantiles=())
        axs[1].set_xlabel("Per-episode mean AoI (s" + (", log scale)" if axs[1].get_xscale() == "log" else ")"))
    if title:
        with S.paper_style():
            fig.suptitle(title)
    return fig


def summary_figure(rows: Sequence[dict]):
    """Task metric (mean ± 95% CI over seeds) of every aggregated row, one bar per group."""
    from ..viz import style as S
    plt = S.pyplot()
    labels = [_group_name(r, ("task", "level", "backend", "baseline")) + (f" [{r['label']}]" if r.get("label") else "")
              for r in rows]
    with S.paper_style():
        fig, ax = plt.subplots(figsize=(S.TEXT_W * 0.8, 0.9 + 0.22 * len(rows)), constrained_layout=True)
        m = [r["stats"]["task"][0] for r in rows]
        ci = [r["stats"]["task"][1] if math.isfinite(r["stats"]["task"][1]) else 0.0 for r in rows]
        sty = S.styles_for(list(dict.fromkeys(r["baseline"] for r in rows)))
        colors = [sty[r["baseline"]][0] for r in rows]
        y = list(range(len(rows)))[::-1]
        ax.barh(y, [v if math.isfinite(v) else 0.0 for v in m], xerr=ci, color=colors, alpha=0.8,
                error_kw=dict(lw=0.7, capsize=1.5), height=0.6)
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        keys = sorted({f"{r['metric_key']} {'↑' if r['higher_is_better'] else '↓'}" for r in rows})
        ax.set_xlabel("task metric: " + ", ".join(keys) + " (mean ± 95% CI over seeds)")
        S.grid(ax, "x")
    return fig


def figures(results: Sequence[dict], rows: Sequence[dict], out_dir: str) -> list:
    """Write the report's figures as PNGs into out_dir; returns [(caption, path)]."""
    from ..viz import style as S
    plt = S.pyplot()
    os.makedirs(out_dir, exist_ok=True)
    out = []
    fig = summary_figure(rows)
    p = os.path.join(out_dir, "summary.png")
    S.finish(fig, p, dpi=200)
    plt.close(fig)
    out.append(("Task metric per configuration", p))
    for i, (title, rs) in enumerate(_panels(results)):
        fig = cdf_figure(rs, title)
        p = os.path.join(out_dir, f"cdf_{i:02d}.png")
        S.finish(fig, p, dpi=200)
        plt.close(fig)
        out.append((f"Delay and AoI CDFs per baseline: {title}", p))
    return out


def markdown_figures(figs: Sequence, rel_to: str = ".") -> str:
    """Markdown image lines for figures() output, with paths relative to rel_to."""
    return "\n\n".join(f"![{c}]({os.path.relpath(p, rel_to)})" for c, p in figs)


def html(results: Sequence[dict], rows: Sequence[dict], path: str, metrics: Sequence[str] = DEFAULT_METRICS,
         timing: bool = True, title: str = "isaac-net benchmark report") -> str:
    """Self-contained HTML report (inline PNGs): the Markdown table, a per-config summary figure and table, and the
    delay / AoI CDFs per baseline of every task configuration. Returns path."""
    from ..viz.report import html_document, img_tag, markdown_table_html, table_html, write_html
    blocks = ["<h2>Results</h2>", markdown_table_html(markdown(rows, metrics, timing=timing)),
              '<p class="sub">Mean ± half-width of the two-sided 95% Student t interval over seeds; n = seeds.</p>',
              "<h2>Per-configuration summary</h2>", img_tag(summary_figure(rows), "summary")]
    conf = []
    for r in rows:
        conf.append([r["task"], r["variant"], r["level"], r["backend"], r["baseline"], len(r["seeds"]),
                     ", ".join(map(str, r["seeds"])), _fmt(*r["stats"].get("deliveries", (math.nan, math.nan, 0))),
                     _fmt(*r["stats"].get("delivery_ratio", (math.nan, math.nan, 0))),
                     f"{r['timing'].get('eval_robot_steps_per_s', 0.0):.3g}"])
    blocks.append(table_html(["task", "variant", "level", "backend", "baseline", "n", "seeds", "deliveries",
                              "delivery ratio", "robot-steps/s"], conf, left=5))
    blocks.append("<h2>Delay and AoI per baseline</h2>")
    kept = any(r["eval"].get("rows") for r in results)
    blocks.append('<p class="sub">' + ("CDF over evaluation env-episodes (result files with rows)." if kept else
                  "CDF over result files (one value per seed); run with --keep_rows for per-episode CDFs.") + "</p>")
    for title_, rs in _panels(results):
        blocks.append(img_tag(cdf_figure(rs, title_), title_, title_))
    return write_html(path, html_document(title, blocks, f"{len(results)} result file(s), {len(rows)} group(s)"))
