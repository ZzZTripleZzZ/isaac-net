"""Input normalization shared by the plot functions: recorder directories, frame tables, dicts of samples."""
from __future__ import annotations

import os

import numpy as np


def is_record_dir(x) -> bool:
    return isinstance(x, (str, os.PathLike)) and os.path.isdir(x) and os.path.exists(os.path.join(x, "meta.json"))


def record_dir_of(x):
    """A RecorderLoop (flushed) or a path; returns the directory or None."""
    cfg = getattr(x, "cfg", None)
    if cfg is not None and hasattr(x, "flush") and hasattr(cfg, "out_dir"):
        x.flush()
        return cfg.out_dir
    return x if is_record_dir(x) else None


def read_table(x):
    """A DataFrame from a DataFrame, or a .csv / .csv.gz / .parquet path."""
    import pandas as pd
    if isinstance(x, pd.DataFrame):
        return x
    p = str(x)
    if p.endswith(".parquet") or os.path.isdir(p):
        return pd.read_parquet(p)
    return pd.read_csv(p)


def run_name(meta: dict, by: str, path) -> str:
    """Group name of a recorder directory: by="level" -> its level, "config"/"label" -> its label (else level)."""
    if by == "level":
        return str(meta.get("level") or os.path.basename(os.path.normpath(path)))
    if by in ("config", "label"):
        return str(meta.get("label") or meta.get("level") or os.path.basename(os.path.normpath(path)))
    return str(meta.get(by) or meta.get("label") or os.path.basename(os.path.normpath(path)))


def group_column(df, by):
    if by in df.columns:
        return by
    alias = {"level": ("arm", "label", "source"), "config": ("label", "config", "arm"),
             "label": ("config", "arm")}.get(by, ())
    for a in alias:
        if a in df.columns:
            return a
    return None


def _as_list(data):
    return list(data) if isinstance(data, (list, tuple)) else [data]


def distributions(data, by: str, kind: str) -> dict:
    """{group: ("hist", lo, hi, count) | ("samples", values)} in ms (kind "delay" or "aoi").

    data: a recorder directory or RecorderLoop (histogram tables), a frame table (DataFrame or CSV / Parquet path)
    with a delay_ms / delay_s (aoi_s / aoi_ms) column, a dict {name: samples in ms}, or a list of these."""
    import pandas as pd
    from ..core.record import read_meta, read_records
    out = {}

    def add(name, item):
        if name in out:              # same group from several inputs: pool
            a, b = out[name], item
            if a[0] == b[0] == "samples":
                out[name] = ("samples", np.concatenate([a[1], b[1]]))
            elif a[0] == b[0] == "hist":
                out[name] = ("hist",) + tuple(np.concatenate([x, y]) for x, y in zip(a[1:], b[1:]))
            else:
                out[name + " (2)"] = item
        else:
            out[name] = item

    for item in _as_list(data):
        rd = record_dir_of(item)
        if rd is not None:
            meta = read_meta(rd)
            h = read_records(rd, f"{kind}_hist")
            add(run_name(meta, by, rd), ("hist", h.lo_ms.to_numpy(float), h.hi_ms.to_numpy(float),
                                         h["count"].to_numpy(float)))
            continue
        if isinstance(item, dict):
            for k, v in item.items():
                add(str(k), ("samples", np.asarray(v, dtype=float)))
            continue
        df = read_table(item)
        if {"lo_ms", "hi_ms", "count"} <= set(df.columns):
            g = group_column(df, by)
            for name, d in (df.groupby(g) if g else [("all", df)]):
                add(str(name), ("hist", d.lo_ms.to_numpy(float), d.hi_ms.to_numpy(float), d["count"].to_numpy(float)))
            continue
        col, scale = _value_column(df, kind)
        g = group_column(df, by)
        for name, d in (df.groupby(g, sort=False) if g else [("all", df)]):
            v = pd.to_numeric(d[col], errors="coerce").to_numpy(float) * scale
            add(str(name), ("samples", v[np.isfinite(v)]))
    return out


def _value_column(df, kind):
    cands = {"delay": (("delay_ms", 1.0), ("delay_s", 1e3), ("lena_delay_ms", 1.0)),
             "aoi": (("aoi_ms", 1.0), ("aoi_s", 1e3), ("aoi_mean_s", 1e3))}[kind]
    for c, s in cands:
        if c in df.columns:
            return c, s
    raise ValueError(f"no {kind} column in the table (looked for {', '.join(c for c, _ in cands)}); "
                     f"columns: {', '.join(map(str, df.columns))}")


def ecdf(item):
    """(x, y) of an empirical CDF in ms: exact steps for samples, bin upper edges for histograms."""
    if item[0] == "samples":
        x = np.sort(item[1])
        if x.size == 0:
            return x, x
        return np.r_[x[0], x], np.r_[0.0, np.arange(1, x.size + 1) / x.size]
    _, lo, hi, cnt = item
    if cnt.sum() <= 0:
        return np.array([]), np.array([])
    # pool bins with the same edges (several envs / flushes)
    keys, inv = np.unique(hi, return_inverse=True)
    c = np.bincount(inv, weights=cnt)
    los = np.array([lo[inv == k].min() for k in range(keys.size)])
    y = np.cumsum(c) / c.sum()
    return np.r_[los[0], keys], np.r_[0.0, y]


def quantile(item, q):
    x, y = ecdf(item)
    if x.size == 0:
        return float("nan")
    return float(np.interp(q, y, x))


def count(item):
    return int(item[1].size) if item[0] == "samples" else int(item[3].sum())
