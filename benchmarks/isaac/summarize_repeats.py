"""Summarize the Isaac Lab scale runs with process repeats (run_repeats.ps1).

usage: python benchmarks/isaac/summarize_repeats.py <scale_repeats.jsonl> <run log(s)> --out <dir>
Writes <dir>/isaac_scale_repeats_rtx4090.csv (one row per process, with the CPU state the log recorded before it)
and <dir>/isaac_scale_repeats_summary_rtx4090.csv (per size and network: mean, min and max over the processes of
each process's median window; the on / off quantities from the means, the widest and narrowest off / on pairing
(added_ms_min / max) and the difference of each adjacent off / on pair (added_ms_paired)), and prints the rows.
A process's value is its median window, as in the single-process table.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics


def cpu_lines(logs):
    """[(E, level, repeat, cpu before)] in log order, from the '== E=... ' and 'CPU ...' lines."""
    out = []
    for p in logs:
        raw = open(p, "rb").read()
        txt = raw.decode("utf-16") if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else raw.decode("utf-8", "replace")
        cur = None
        for line in txt.splitlines():
            m = re.match(r"== E=(\d+) R=(\d+) (\S+) (\S+) repeat (\d+) (\S+) (.*)", line)
            if m:
                cur = dict(E=int(m[1]), R=int(m[2]), level=m[3], repeat=int(m[5]), start=m[6], idle=m[7])
                out.append(cur)
            elif line.startswith("CPU after ") and cur is not None:
                cur["cpu_after"] = line[len("CPU after "):]
            elif line.startswith("CPU ") and cur is not None:
                cur["cpu_before"] = line[4:]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl")
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    recs = [json.loads(x) for x in open(a.jsonl) if x.strip()]
    cpu = cpu_lines(a.logs)
    rows = []
    seen = {}
    for r in recs:
        k = (r["E"], r["R"], r["level"])
        seen[k] = seen.get(k, 0) + 1
        c = next((x for x in cpu if (x["E"], x["R"], x["level"]) == k and x["repeat"] == seen[k]), {})
        rows.append(dict(E=r["E"], R=r["R"], robots=r["E"] * r["R"], level=r["level"], backend=r["backend"],
                         nr_cfg=r.get("nr_cfg", "-"), repeat=seen[k], start=c.get("start", ""),
                         control_steps_per_s=round(r["iter_per_s"], 4),
                         windows=" ".join(f"{x:.3f}" for x in r["iter_per_s_windows"]),
                         step_ms=round(1000 / r["iter_per_s"], 1),
                         net_only_ms=round(r["net_only_ms_per_step"], 2) if "net_only_ms_per_step" in r else "",
                         net_only_windows=" ".join(f"{x:.1f}" for x in r.get("net_only_ms_windows", [])),
                         startup_s=round(r["startup_s"]), device_mem_max_mib=r["gpu_mem_during_max_mib"],
                         gpu_util_before=r["gpu_util_before"], gpu_mem_before_mib=r["gpu_mem_before_mib"],
                         gpu_util_during_mean=round(r["gpu_util_during_mean"], 1), idle=c.get("idle", ""),
                         cpu_before=c.get("cpu_before", ""), cpu_after=c.get("cpu_after", "")))
    os.makedirs(a.out, exist_ok=True)
    write(os.path.join(a.out, "isaac_scale_repeats_rtx4090.csv"), rows)
    summ = []
    for E, R in sorted({(r["E"], r["R"]) for r in rows}):
        off = [r for r in rows if (r["E"], r["R"], r["level"]) == (E, R, "off")]
        for lv in ("off", "L2"):
            rs = [r for r in rows if (r["E"], r["R"], r["level"]) == (E, R, lv)]
            if not rs:
                continue
            v = [r["control_steps_per_s"] for r in rs]
            s = dict(E=E, R=R, robots=E * R, level=lv, nr_cfg=rs[0]["nr_cfg"], n_proc=len(rs),
                     steps_per_s_mean=round(statistics.mean(v), 3), steps_per_s_min=min(v), steps_per_s_max=max(v),
                     step_ms_mean=round(1000 / statistics.mean(v), 1),
                     startup_s_mean=round(statistics.mean(r["startup_s"] for r in rs)),
                     device_mem_max_mib=max(r["device_mem_max_mib"] for r in rs),
                     gpu_util_during_mean=round(statistics.mean(r["gpu_util_during_mean"] for r in rs), 1))
            if lv != "off":
                n = [r["net_only_ms"] for r in rs]
                s.update(net_only_ms_mean=round(statistics.mean(n), 1), net_only_ms_min=min(n), net_only_ms_max=max(n))
                if off:
                    vo = [r["control_steps_per_s"] for r in off]
                    t_on, t_off = 1000 / statistics.mean(v), 1000 / statistics.mean(vo)
                    s.update(on_over_off=round(statistics.mean(v) / statistics.mean(vo), 3),
                             added_ms_mean=round(t_on - t_off, 1),
                             added_ms_min=round(1000 / max(v) - 1000 / min(vo), 1),
                             added_ms_max=round(1000 / min(v) - 1000 / max(vo), 1))
                    # paired: repeat k of the network run against repeat k of the off run (adjacent processes)
                    by = {r["repeat"]: r["control_steps_per_s"] for r in off}
                    pd = [1000 / r["control_steps_per_s"] - 1000 / by[r["repeat"]] for r in rs if r["repeat"] in by]
                    s.update(added_ms_paired=" ".join(f"{x:.1f}" for x in pd))
            summ.append(s)
    write(os.path.join(a.out, "isaac_scale_repeats_summary_rtx4090.csv"), summ)
    for s in summ:
        print(s)


def write(path, rows):
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
