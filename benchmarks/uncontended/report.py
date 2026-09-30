"""Merge the three process repeats of the uncontended campaign into CSVs and print the markdown tables of the docs.

usage: python benchmarks/uncontended/report.py <res_r1> <res_r2> <res_r3> --out benchmarks/results/uncontended
Each <res_ri> holds net.jsonl and env.jsonl from run.sh. A case's value is the median over the three processes of
each process's median window; lo / hi are the smallest and largest process medians, spread = (hi - lo) / median.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics

NET_KEY = ("case", "backend", "cfg", "wrap_graph", "E", "R")
ENV_KEY = ("net", "backend", "E", "R")


def load(dirs, name, key):
    rows = {}
    for i, d in enumerate(dirs):
        for line in open(os.path.join(d, name)):
            r = json.loads(line)
            if "E" not in r:
                continue
            rows.setdefault(tuple(r[k] for k in key), [None] * len(dirs))[i] = r
    return rows


def merge(rows, key, mem_key):
    out = []
    for k, rs in rows.items():
        ok = [r for r in rs if r is not None and r.get("status", "ok") == "ok"]
        rec = dict(zip(key, k))
        rec["robots"] = k[key.index("E")] * k[key.index("R")]
        if not ok:
            rec["status"] = next((r.get("status") for r in rs if r), "missing")
            out.append(rec)
            continue
        ms = [r["ms_median"] for r in ok]
        med = statistics.median(ms)
        rec |= {"ms": round(med, 4), "ms_lo": min(ms), "ms_hi": max(ms),
                "spread_pct": round(100 * (max(ms) - min(ms)) / med, 1), "n_proc": len(ok),
                "ms_per_proc": " ".join(f"{x:g}" for x in ms),
                "max_window_spread_pct": max(r["spread_pct"] for r in ok),
                mem_key: max(r[mem_key] for r in ok),
                "idle_util_max": max(r["idle_util_max"] for r in ok),
                "idle_mem_used_mib": max(r["idle_mem_used_mib"] for r in ok),
                "run_util_mean": round(statistics.mean(r.get("run_util_mean", float("nan")) for r in ok), 1),
                "run_mem_used_max_mib": max(r.get("run_mem_used_max_mib", 0) for r in ok),
                "gpu": ok[0]["gpu"], "torch": ok[0]["torch"], "status": "ok"}
        out.append(rec)
    return out


def write(path, recs):
    fields = []
    for r in recs:
        fields += [f for f in r if f not in fields]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fields)
        w.writeheader()
        w.writerows(recs)


def fmt(x):
    if x is None:
        return ""
    if x >= 100:
        return f"{x:,.0f}"
    if x >= 10:
        return f"{x:.1f}"
    return f"{x:.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    net = merge(load(a.dirs, "net.jsonl", NET_KEY), NET_KEY, "peak_mib")
    env = merge(load(a.dirs, "env.jsonl", ENV_KEY), ENV_KEY, "peak_alloc_mib")
    write(os.path.join(a.out, "net_rtx4090.csv"), net)
    write(os.path.join(a.out, "env_rtx4090.csv"), env)
    idx = {(r["case"], r["backend"], r["cfg"], r["wrap_graph"], r["E"], r["R"]): r for r in net}
    sizes = [(256, 16), (1024, 32), (4096, 16), (4096, 100)]

    def cell(case, b, cfg="-", wg=False, E=0, R=0, mem=False):
        r = idx.get((case, b, cfg, wg, E, R))
        if r is None or r.get("status") != "ok":
            return ""
        s = fmt(r["ms"]) + (f" ±{r['spread_pct']:.0f}%" if r["spread_pct"] >= 10 else "")
        return s + (f" ({r['peak_mib']:,.0f})" if mem else "")

    print("## prototype, surrogate and Wi-Fi levels: ms (peak MiB)")
    print("| Level | Backend | " + " | ".join(f"{E} × {R}" for E, R in sizes) + " |")
    for lv in ("L0", "L0DR", "L05", "L05Q", "TR", "GE", "QA", "NN", "L1", "L2-legacy", "WIFI"):
        for b in ("reference", "graph", "triton"):
            cs = [cell(lv, b, E=E, R=R, mem=True) for E, R in sizes]
            if any(cs):
                print(f"| `{lv}` | {b} | " + " | ".join(cs) + " |")
    print("\n## NR engine: ms (peak MiB)")
    for cfg in ("ul", "ul_dl", "c3", "c3_dl"):
        for b in ("reference", "graph", "triton"):
            cs = [cell("L2", b, cfg, E=E, R=R, mem=True) for E, R in sizes]
            if any(cs):
                print(f"| `{cfg}` | {b} | " + " | ".join(cs) + " |")
    print("\n## wrappers over L2-legacy triton")
    for w in ("edge", "energy"):
        for wg in (False, True):
            cs = [cell(f"L2-legacy+{w}", "triton", wg=wg, E=E, R=R, mem=True) for E, R in sizes]
            print(f"| {w} | {'graph' if wg else 'eager'} | " + " | ".join(cs) + " |")
    print("\n## full env step (ms)")
    eidx = {(r["net"], r["backend"], r["E"], r["R"]): r for r in env}
    for n, b in (("off", "-"), ("L0", "reference"), ("L0", "graph"), ("L2-legacy", "triton"), ("L2", "triton")):
        cs = []
        for E, R in sizes:
            r = eidx.get((n, b, E, R))
            cs.append("" if r is None or r.get("status") != "ok" else
                      fmt(r["ms"]) + (f" ±{r['spread_pct']:.0f}%" if r["spread_pct"] >= 10 else ""))
        print(f"| {n} | {b} | " + " | ".join(cs) + " |")
    print("\nmax idle util", max(r.get("idle_util_max", 0) for r in net + env),
          "max idle mem", max(r.get("idle_mem_used_mib", 0) for r in net + env),
          "rows with spread >= 10%:", sum(1 for r in net + env if r.get("spread_pct", 0) >= 10), "of", len(net + env))


if __name__ == "__main__":
    main()
