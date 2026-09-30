"""Aggregate result files into tables: mean and 95% confidence interval over seeds.

    rows = aggregate(load_results(["results/"]))
    print(markdown(rows, metrics=["task", "delay_p95_ms", "aoi_mean_s"]))

One group = (task, variant, level, backend, preset, traffic, baseline, label). Each result file contributes one
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

GROUP_KEYS = ("task", "variant", "level", "backend", "preset", "traffic", "baseline", "label")
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
        key = tuple(r.get(k, "") for k in GROUP_KEYS)
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
    """Markdown table: one line per group, mean ± 95% CI over seeds (n seeds in the "n" column)."""
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
