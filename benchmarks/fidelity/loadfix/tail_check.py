"""Share of 5G-LENA frames completed by an RLC tail PDU (<= 64 B, sent >= 5 ms after the UE's previous PDU).

A frame counts if its completion time ('last' in frames.csv) lies within 1 ms of such a PDU's reception at the
gNB (NrUlRlcRxStats.txt). Reads the raw sweep traces read-only; prints one line per run and writes a CSV.

usage: python tail_check.py <ns3ref sweep/nofade> <out.csv>
"""
import csv
import glob
import os
import sys
from collections import defaultdict

import numpy as np


def main(sweep, out):
    res = []
    for d in sorted(glob.glob(os.path.join(sweep, "*"))):
        if not os.path.exists(os.path.join(d, "frames.csv")):
            continue
        rx = defaultdict(list)
        with open(os.path.join(d, "NrUlRlcRxStats.txt")) as f:
            f.readline()
            for line in f:
                c = line.split()
                rx[int(c[2])].append((float(c[0]), int(c[4])))
        tail = []
        for lst in rx.values():
            lst.sort()
            tail += [t1 for (t0, _), (t1, b1) in zip(lst, lst[1:]) if b1 <= 64 and t1 - t0 >= 5e-3]
        tail = np.sort(np.array(tail))
        fr = [r for r in csv.DictReader(open(os.path.join(d, "frames.csv"))) if int(r["rxpk"]) >= int(r["npk"])]
        last = np.array([float(r["last"]) for r in fr])
        frac = float("nan")
        if len(last) and len(tail):
            k = np.searchsorted(tail, last)
            near = np.minimum(np.abs(last - tail[np.clip(k - 1, 0, len(tail) - 1)]),
                              np.abs(tail[np.clip(k, 0, len(tail) - 1)] - last))
            frac = float(np.mean(near < 1e-3))
        res.append(dict(run=os.path.basename(d), complete_frames=len(last), tail_completed_frac=frac))
        print(res[-1]["run"], f"{frac:.3f}", flush=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(res[0]))
        w.writeheader()
        w.writerows(res)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
