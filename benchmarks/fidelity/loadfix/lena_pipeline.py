"""Grant-pipeline accounting of the 5G-LENA primary-arm runs, from the raw MAC / RLC / PHY traces (read-only).

For every run directory of the ns3ref sweep this joins
  NrUlMacStats.txt       every UL grant the scheduler issued (PUSCH slot, RNTI, HARQ id, ndi, rv, MCS, TB size)
  RxPacketTrace.txt      every decoded UL TB (PUSCH time, corrupt flag), de-duplicated
  NrUlRlcTxStats.txt     every RLC PDU the UE built into a grant (time = UL DCI reception at the UE, bytes)
  TxedUeMacCtrlMsgsTrace every SR the UE sent
  gnb_slots.csv          RBG / symbol use of every UL slot
and splits the UL resource use into the parts the NR engine does or does not model:

  data_rbg      RBGs a TB needed for the RLC bytes it carried (smallest RBG count whose TB fits them + 8 B of
                MAC subheader and short BSR)
  pad_rbg       RBGs granted beyond that (BSR quantization and the stale-BSR over-grant), incl. empty TBs
  empty_tbs     new-data grants that carried no RLC byte at all
  boot_tbs      grants whose TB carried at most 30 RLC bytes (the SR bootstrap grant and tails)
  retx_tbs      HARQ retransmissions; retx_slots_blocked = UL slots whose retransmission occupied every data
                symbol, so no other UE could be scheduled (5G-LENA schedules UL retransmissions TDMA)
  tail_stalls   RLC PDUs of <= 64 B sent >= 5 ms after the UE's previous RLC PDU while its frame was incomplete
                (the RLC header residue that only the 10 ms RLC buffer-status timer reports)

The RBG count of a grant is recovered by inverting the 5G-LENA TB size formula (phy.tbs_lena) at 13 symbols.

usage: python lena_pipeline.py <ns3ref sweep/nofade> <data dir of lena_extract (lena_runs.csv)> <out dir>
"""
import csv
import glob
import os
import sys
from collections import defaultdict

import numpy as np

# 5G-LENA MCS table 1 (Qm, R*1024), as phy.MCS_T1
MCS_T1 = [(2, 120), (2, 157), (2, 193), (2, 251), (2, 308), (2, 379), (2, 449), (2, 526), (2, 602),
          (2, 679), (4, 340), (4, 378), (4, 434), (4, 490), (4, 553), (4, 616), (4, 658), (6, 438),
          (6, 466), (6, 517), (6, 567), (6, 616), (6, 666), (6, 719), (6, 772), (6, 822), (6, 873),
          (6, 910), (6, 948)]
RBG_PRB, N_RBG, NSYM, REF_SC = 10, 5, 13, 1
HDR = 8          # MAC subheader (3) + short BSR (5) per TB, from NrUeMac::SendNewData


def tbs_lena_bytes(mcs, nprb):
    """5G-LENA NrAmc::CalculateTbSize in bytes (phy.tbs_lena, scalar)."""
    qm, r = MCS_T1[mcs][0], MCS_T1[mcs][1] / 1024
    p = int(np.floor((12 - REF_SC) * nprb * NSYM * qm * r / 8))
    tb = p - 3 if p >= 3 else p
    bits = p * 8
    bg2 = bits <= 292 or r <= 0.25 or (bits <= 3824 and r <= 0.67)
    cbmax = 480 if bg2 else 1056
    c = int(np.ceil(tb / cbmax))
    if tb > cbmax:
        tb = p - 3 * c
    return max(tb, 0)


TBS = np.array([[tbs_lena_bytes(m, RBG_PRB * k) for k in range(N_RBG + 1)] for m in range(len(MCS_T1))])


def n_rbg(mcs, tbs):
    k = np.nonzero(TBS[mcs] == tbs)[0]
    return int(k[0]) if len(k) else -1


def need_rbg(mcs, nbytes):
    if nbytes <= 0:
        return 0
    k = np.nonzero(TBS[mcs] >= nbytes + HDR)[0]
    return int(k[0]) if len(k) else N_RBG


def read_tsv(path, skip=1):
    with open(path) as f:
        for _ in range(skip):
            f.readline()
        for line in f:
            yield line.split()


def analyze(d):
    meta = {}
    for line in open(os.path.join(d, "meta.txt")):
        if line.strip():
            k, v = line.split(None, 1)
            meta[k] = v.strip()
    app, T = float(meta.get("appStart", 0.5)), float(meta["trafficTime"])
    t_lo, t_hi = app, app + T
    # grants: key (rnti, frame, sf, slot)
    grants = []
    for c in read_tsv(os.path.join(d, "NrUlMacStats.txt")):
        t, rnti, fr, sf, sl, _, nsym, hid, ndi, rv, mcs, tbs = (float(c[0]), int(c[4]), int(c[5]), int(c[6]),
                                                                  int(c[7]), int(c[8]), int(c[9]), int(c[10]),
                                                                  int(c[11]), int(c[12]), int(c[13]), int(c[14]))
        grants.append((t, rnti, fr, sf, sl, nsym, hid, ndi, rv, mcs, tbs))
    # PUSCH decode time per (rnti, frame, sf, slot)
    pusch = {}
    seen = set()
    with open(os.path.join(d, "RxPacketTrace.txt")) as f:
        f.readline()
        for line in f:
            if line in seen:
                continue
            seen.add(line)
            c = line.split()
            if c[1] != "UL":
                continue
            pusch[(int(c[9]), int(c[2]), int(c[3]), int(c[4]))] = (float(c[0]), int(c[16]))
    # RLC PDUs at the UE: (rnti, time, bytes)
    rlc = defaultdict(list)
    for c in read_tsv(os.path.join(d, "NrUlRlcTxStats.txt")):
        rlc[int(c[2])].append((float(c[0]), int(c[4])))
    sr = defaultdict(list)
    with open(os.path.join(d, "TxedUeMacCtrlMsgsTrace.txt")) as f:
        f.readline()
        for line in f:
            c = line.rstrip("\n").split("\t")
            if c[-1] == "SR":
                sr[int(c[7])].append(float(c[0]))
    # attach RLC bytes to grants: RLC PDUs built at DCI reception precede the PUSCH by < 3 ms
    by_rnti = defaultdict(list)
    for g in grants:
        p = pusch.get((g[1], g[2], g[3], g[4]))
        if p is None:
            continue
        by_rnti[g[1]].append((p[0], g, p[1]))
    acc = defaultdict(float)
    lead, gap_rv = [], []
    tb_rows = []
    for rnti, lst in by_rnti.items():
        lst.sort(key=lambda x: x[0])
        tp = np.array([x[0] for x in lst])
        data = np.zeros(len(lst))
        for (t, b) in rlc.get(rnti, []):
            k = int(np.searchsorted(tp, t, side="right"))
            if k < len(lst) and tp[k] - t < 3e-3:
                data[k] += b
        last_rv0 = {}
        for i, (tpu, g, bad) in enumerate(lst):
            t, _, _, _, _, nsym, hid, ndi, rv, mcs, tbs = g
            lead.append(tpu - t)
            k = n_rbg(mcs, tbs)
            inwin = t_lo <= tpu < t_hi
            if rv == 0:
                last_rv0[hid] = tpu
            elif hid in last_rv0 and rv == 1:
                gap_rv.append(tpu - last_rv0[hid])
            tb_rows.append((tpu, rnti, rv, k, int(data[i]), nsym))
            if not inwin:
                continue
            if rv > 0:
                acc["retx_tbs"] += 1
                acc["retx_rbg"] += max(k, 0)
                continue
            acc["new_tbs"] += 1
            acc["tb_bytes"] += tbs
            acc["rlc_bytes"] += data[i]
            acc["rbg_new"] += max(k, 0)
            nk = need_rbg(mcs, data[i])
            acc["data_rbg"] += nk
            acc["pad_rbg"] += max(k, 0) - nk
            acc["unknown_rbg"] += k < 0
            if data[i] == 0:
                acc["empty_tbs"] += 1
                acc["empty_rbg"] += max(k, 0)
            elif data[i] <= 30:
                acc["boot_tbs"] += 1
    # slots: UL slots in the traffic window and retx-blocked slots
    rows = list(csv.DictReader(open(os.path.join(d, "gnb_slots.csv"))))
    ul_slots_sched = sum(1 for r in rows if t_lo <= float(r["t"]) < t_hi)
    acc["ul_slots_sched"] = ul_slots_sched
    acc["ul_slots"] = int(round(T * 1000 / 2.5))
    acc["rbg_used_slots"] = sum(float(r["usedReg"]) / (RBG_PRB * 13) for r in rows if t_lo <= float(r["t"]) < t_hi)
    by_slot = defaultdict(list)
    for tpu, rnti, rv, k, dat, nsym in tb_rows:
        if t_lo <= tpu < t_hi:
            by_slot[round(tpu * 1e4)].append((rv, k))
    blocked = 0
    lost = 0
    for v in by_slot.values():
        if any(rv > 0 for rv, _ in v):
            blocked += 1
            lost += N_RBG - sum(max(k, 0) for _, k in v)
    acc["retx_slots"] = blocked
    acc["retx_lost_rbg"] = lost
    # tail stalls: small RLC PDU after >= 5 ms of silence of that UE
    stalls, stall_gap = 0, []
    for rnti, lst in rlc.items():
        lst.sort()
        for (t0, _), (t1, b1) in zip(lst, lst[1:]):
            if b1 <= 64 and t1 - t0 >= 5e-3 and t_lo <= t1 < t_hi:
                stalls += 1
                stall_gap.append(t1 - t0)
    acc["tail_stalls"] = stalls
    acc["tail_gap_ms_med"] = float(np.median(stall_gap) * 1e3) if stall_gap else float("nan")
    acc["n_sr"] = sum(1 for v in sr.values() for t in v if t_lo <= t < t_hi)
    acc["lead_ms_med"] = float(np.median(lead) * 1e3) if lead else float("nan")
    acc["retx_gap_ms_med"] = float(np.median(gap_rv) * 1e3) if gap_rv else float("nan")
    acc["retx_gap_n"] = len(gap_rv)
    return dict(acc), tb_rows


def main(sweep, data, out):
    os.makedirs(out, exist_ok=True)
    runs = {r["run"]: r for r in csv.DictReader(open(os.path.join(data, "lena_runs.csv")))}
    rows = []
    for d in sorted(glob.glob(os.path.join(sweep, "*"))):
        name = os.path.basename(d)
        if name not in runs:
            continue
        r = runs[name]
        a, _ = analyze(d)
        drop = float(r["drop"])
        row = dict(run=name, N=int(r["N"]), S=int(r["S"]), load=float(r["load"]), seed=int(r["seed"]),
                   regime="light" if drop < 0.01 else ("saturated" if drop >= 0.2 else "moderate"),
                   frames=int(r["frames"]), lena_drop=drop, **a)
        nt = max(a.get("new_tbs", 0), 1)
        row["tbs_per_frame"] = a.get("new_tbs", 0) / max(int(r["frames"]), 1)
        row["pad_frac_bytes"] = 1 - a.get("rlc_bytes", 0) / max(a.get("tb_bytes", 1), 1)
        row["pad_frac_rbg"] = a.get("pad_rbg", 0) / max(a.get("rbg_new", 1), 1)
        row["empty_frac_tbs"] = a.get("empty_tbs", 0) / nt
        cap = a["ul_slots"] * N_RBG
        row["util_new"] = a.get("rbg_new", 0) / cap
        row["util_data"] = a.get("data_rbg", 0) / cap
        row["util_pad"] = a.get("pad_rbg", 0) / cap
        row["util_retx"] = a.get("retx_rbg", 0) / cap
        row["util_retx_blocked"] = a.get("retx_lost_rbg", 0) / cap
        row["util_slots_trace"] = a.get("rbg_used_slots", 0) / cap
        rows.append(row)
        print(name, f"pad {row['pad_frac_rbg']:.3f} empty {row['empty_frac_tbs']:.3f} util data/pad/retx/blk "
              f"{row['util_data']:.3f}/{row['util_pad']:.3f}/{row['util_retx']:.3f}/{row['util_retx_blocked']:.3f}",
              flush=True)
    keys = list(rows[0])
    for r in rows[1:]:
        keys += [k for k in r if k not in keys]
    with open(os.path.join(out, "lena_pipeline_per_run.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})


if __name__ == "__main__":
    main(*sys.argv[1:4])
