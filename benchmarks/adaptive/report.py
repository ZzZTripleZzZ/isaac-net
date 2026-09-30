"""Markdown tables from sweep.py output: python benchmarks/adaptive/report.py adaptive.jsonl [more.jsonl ...]"""
import json
import sys
from collections import defaultdict


def label(r):
    c = r.get("cfg")
    if not c:
        x = r.get("expensive", "L2-legacy")
        return {"ref": f"pure {x} (reference)", "ref-seed2": f"pure {x}, other seed"}.get(r["run"], f"pure {r['run']}")
    if c["mode"] == "static":
        return f"static, {c['fraction']:.0%} expensive"
    b = "mask" if c["layout"] == "mask" else f"budget {c['active_budget']:.0%}"
    dp = f", every {c['decision_period']} steps" if c.get("decision_period", 1) > 1 else ""
    return f"load, thr {c['up_threshold']:.0f} B, {b}{dp}"


def fmt(v, f="{:.3f}"):
    return "" if v is None else f.format(v)


def main(paths):
    rows = [json.loads(line) for p in paths for line in open(p) if line.strip()]
    by = defaultdict(list)
    for r in rows:
        by[(r["scenario"], r["E"], r["R"], r.get("cheap", "L1"), r["backend"], r.get("exp_backend", r["backend"]),
            r.get("expensive", "L2-legacy"))].append(r)
    for (sc, E, R, C, be, xb, X), rs in by.items():
        ref = next((r for r in rs if r["run"] == "ref"), None)
        print(f"\n### {sc}: {C} `{be}` and {X} `{xb}`, E = {E}, R = {R}\n")
        print("| Run | expensive share | W1 (steps) | KS | drop | Δdrop | p50 / p95 / p99 (steps) | ms / step | vs ref | "
              "ms / step, resets | switches up / down |")
        print("|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for r in rs:
            ms = r.get("ms")
            sp = f"{ref['ms'] / ms:.2f}×" if (ms and ref and ref.get("ms")) else ""
            sw = "" if r.get("switch_up") is None else f"{r['switch_up']} / {r['switch_down']}"
            print(f"| {label(r)} | {r['exp_share']:.0%} | {fmt(r['w1'])} | {fmt(r['ks'])} | {r['drop']:.3f} | "
                  f"{r['drop_diff']:+.3f} | {r['p50']:.2f} / {r['p95']:.2f} / {r['p99']:.2f} | {fmt(ms, '{:.2f}')} | "
                  f"{sp} | {fmt(r.get('ms_resets'), '{:.2f}')} | {sw} |")
        util = [r.get("gpu_util") for r in rs if r.get("gpu_util") is not None]
        if util:
            print(f"\nGPU utilization during the timing windows (all processes): {min(util):.0f}–{max(util):.0f}%.")


if __name__ == "__main__":
    main(sys.argv[1:])
