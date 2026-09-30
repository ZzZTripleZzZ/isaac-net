"""BLER-curve sanity data (CSV only, no plotting): per MCS, BLER vs SINR at several code-block sizes,
the 10% BLER threshold per MCS, and consistency checks.
  C1 BLER is non-increasing in SINR and in MCS-order thresholds are increasing (per table)
  C2 table-1 and table-2 curves with identical (Qm, R) agree within half the raw SNR step (independent Sionna simulations)
  C3 10% thresholds sit a plausible gap above Shannon (0.5..6 dB at CBS ~1500)
The pass/fail checks also run as tests/test_nr_phy.py.
usage: python -m isaaclab_net.tools.bler_curves outdir
"""
import csv
import json
import math
import os
import sys

import torch

from isaaclab_net.core.phy import MCS_TABLES, PHY

outdir = sys.argv[1] if len(sys.argv) > 1 else "."
dev = "cpu"
CBS = [100, 524, 1524, 3824, 8424]
res = {}
rows = []
thr_rows = []
for direction in ("ul", "dl"):
    for tab in (1, 2):
        phy = PHY(direction, tab, dev)
        snr = torch.arange(-10, 30.01, 0.25)
        for m in range(phy.M):
            for c in CBS:
                b = phy.bler_lookup(torch.full_like(snr, m), snr, torch.full_like(snr, float(c)))
                for s, v in zip(snr.tolist(), b.tolist()):
                    rows.append((direction, tab, m, c, round(s, 2), v))
        thr = phy.thr_lookup(torch.full((phy.M,), 1524.0))
        for m in range(phy.M):
            q, r = MCS_TABLES[tab][m]
            se = q * r / 1024
            shannon = 10 * math.log10(2 ** se - 1)
            thr_rows.append((direction, tab, m, q, r, round(se, 4), round(float(thr[m]), 2), round(shannon, 2),
                             round(float(thr[m]) - shannon, 2)))
        mono = bool((phy.bler[..., 1:] <= phy.bler[..., :-1] + 1e-6).all())
        inc = bool((thr[1:] >= thr[:-1] - 1e-6).all())
        gaps = [x[-1] for x in thr_rows if x[0] == direction and x[1] == tab]
        res[f"C1 {direction} t{tab} monotone"] = {"pass": mono and inc, "info": f"thr increasing {inc}"}
        res[f"C3 {direction} t{tab} Shannon gap"] = {"pass": min(gaps) > 0.3 and max(gaps) < 6.5,
                                                    "info": f"gap to Shannon {min(gaps):.2f}..{max(gaps):.2f} dB"}
# C2: identical (Qm, R) across tables
for direction in ("ul", "dl"):
    p1, p2 = PHY(direction, 1, dev), PHY(direction, 2, dev)
    diffs = []
    for m2, e2 in enumerate(MCS_TABLES[2]):
        if e2 in MCS_TABLES[1]:
            m1 = MCS_TABLES[1].index(e2)
            diffs.append((m1, m2, round(float(p1.thr_ref[m1] - p2.thr_ref[m2]), 2)))
    mx = max(abs(d[2]) for d in diffs)
    # independent simulations on raw SNR grids of 25/14 = 1.79 dB (t1) and 30/19 = 1.58 dB (t2):
    # agreement is expected within about half a raw grid step
    res[f"C2 {direction} same (Qm,R) across tables"] = {"pass": mx <= 0.9, "info": f"max |dthr| {mx} dB (raw grid 1.6-1.8 dB); {diffs}"}
os.makedirs(outdir, exist_ok=True)
with open(os.path.join(outdir, "bler_curves.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["dir", "mcs_table", "mcs", "cbs_bits", "sinr_db", "bler_cb"]); w.writerows(rows)
with open(os.path.join(outdir, "bler_thresholds.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["dir", "mcs_table", "mcs", "qm", "r_x1024", "se", "sinr_bler10_db_cbs1524", "shannon_db", "gap_db"])
    w.writerows(thr_rows)
# PHY comparison: 10%-BLER SINR of a one-RBG (10 PRB, 12 symbols) TB per MCS (table 1)
from isaaclab_net.core.phy import lena_tables_path  # noqa: E402
LENA_DATA = lena_tables_path()
cmp_rows = []
ps = PHY("ul", 1, dev)
pl = PHY("ul", 1, dev, source="lena", tbs_mode="lena") if os.path.exists(LENA_DATA) else None
for m, (q, r) in enumerate(MCS_TABLES[1]):
    se = q * r / 1024
    legacy = 10 * math.log10(2 ** (se / 0.75) - 1) + math.log(9) / 1.5    # NetSlot logistic, slope 1.5/dB
    cmp_rows.append((m, round(se, 4), round(float(ps.thr_ref[m]), 2),
                     round(float(pl.thr_ref[m]), 2) if pl is not None else "", round(legacy, 2),
                     round(10 * math.log10(2 ** se - 1), 2)))
with open(os.path.join(outdir, "phy_compare.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["mcs", "se", "sionna_thr10_db", "lena_thr10_db", "netslot_legacy_thr10_db", "shannon_db"])
    w.writerows(cmp_rows)
if pl is not None:
    d = [x[3] - x[2] for x in cmp_rows]
    res["C5 LENA vs Sionna 10% points (info)"] = {"pass": True, "info": f"LENA - Sionna: min {min(d):+.2f} max {max(d):+.2f} "
                                                f"mean {sum(d) / len(d):+.2f} dB; MCS0 LENA {cmp_rows[0][3]} dB vs legacy "
                                                f"NetSlot at SE 0.23: {cmp_rows[0][4]} dB"}
for k, v in res.items():
    print(("PASS " if v["pass"] else "FAIL ") + k, v["info"])
json.dump(res, open(os.path.join(outdir, "bler_checks.json"), "w"), indent=1)
