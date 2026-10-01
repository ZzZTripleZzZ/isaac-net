"""Merge the three process repeats of the network-only runs of the scale configuration (run_v2l.sh) into a CSV.

usage: python benchmarks/uncontended/report_v2l.py <raw dir with net_v2l_r1..3.jsonl> --out <csv>
Same merge as report.py: a case's value is the median over the processes of each process's median window.
"""
from __future__ import annotations

import argparse
import json
import os

from report import NET_KEY, merge, write


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = {}
    for i in range(3):
        for line in open(os.path.join(a.raw, f"net_v2l_r{i + 1}.jsonl")):
            r = json.loads(line)
            rows.setdefault(tuple(r[k] for k in NET_KEY), [None] * 3)[i] = r
    recs = merge(rows, NET_KEY, "peak_mib")
    write(a.out, recs)
    for r in recs:
        print(r["cfg"], r["E"], r["R"], r["ms"], r["ms_per_proc"], r["spread_pct"], r["peak_mib"], r["idle_mem_used_mib"])


if __name__ == "__main__":
    main()
