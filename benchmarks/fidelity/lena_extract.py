"""Extract per-frame, per-UE and per-TB reference data from the raw 5G-LENA sweep traces.

Reads every run directory of the primary (no-fading) arm of the ns3ref sweep, read-only, and writes
  <out>/lena/<run>.npz        per-frame arrays: ue, step (100 ms send index), bytes, status, delay_ms
  <out>/lena_runs.csv         one row per run: traffic, drop, delay quantiles, HARQ, PRB, goodput
  <out>/lena_per_ue.csv       one row per UE: link budget, frames, ok, goodput, p50/p95, TBs, retx, lost
  <out>/lena_sr_dci.csv       one row per run: SR -> next UL DCI delay at the UE (ms)
Frame status follows ns3ref parse_run.py: 0 = ok (all packets received, delay <= 2 s), 1 = late
(complete, delay > 2 s), 2 = incomplete (a packet never arrived: RLC UM loss, PDCP discard, end of run).

Per-UE TB counts are recomputed here because the per-UE TB columns of ns3ref per_ue.csv are attributed
to UE 0 only (the nodeId column of ue_mac_buf.csv is constant, so its RNTI -> UE map collapses). With
fading off and whole-band power, every UL TB of a UE is received at exactly that UE's whole-band SNR
(ues.csv snr_bw_db), so RNTIs are mapped to UEs by matching the per-RNTI median PHY SINR to snr_bw_db.

usage: python lena_extract.py <ns3ref sweep/nofade dir> <out dir>
"""
import csv
import glob
import json
import os
import sys

import numpy as np

DEADLINE_S = 2.0


def read_meta(d):
    meta = {}
    for line in open(os.path.join(d, "meta.txt")):
        if line.strip():
            k, v = line.split(None, 1)
            meta[k] = v.strip()
    return meta


def parse_frames(d, app_start):
    rows = list(csv.DictReader(open(os.path.join(d, "frames.csv"))))
    ue = np.array([int(r["ue"]) for r in rows], dtype=np.int32)
    step = np.array([int(round((float(r["gen"]) - app_start) / 0.1)) for r in rows], dtype=np.int32)
    nbytes = np.array([int(r["bytes"]) for r in rows], dtype=np.int32)
    delay = np.array([float(r["delay"]) * 1e3 for r in rows])
    complete = np.array([int(r["rxpk"]) >= int(r["npk"]) for r in rows])
    status = np.where(~complete, 2, np.where(delay > DEADLINE_S * 1e3, 1, 0)).astype(np.int8)
    delay = np.where(complete, delay, np.nan)
    return ue, step, nbytes, status, delay


def parse_phy(d):
    """De-duplicated UL TB rows of RxPacketTrace: per RNTI TB count, rv histogram, corrupt, SINR, MCS."""
    path = os.path.join(d, "RxPacketTrace.txt")
    per = {}
    with open(path) as f:
        hdr = f.readline().split()
        ix = {h.lower(): i for i, h in enumerate(hdr)}
        seen = set()
        for line in f:
            if line in seen:          # 5G-LENA writes each UL TB row twice
                continue
            seen.add(line)
            c = line.split()
            if len(c) < len(hdr) or c[ix["direction"]] != "UL":
                continue
            rn = int(c[ix["rnti"]])
            p = per.setdefault(rn, {"rv": np.zeros((4, 2), dtype=np.int64), "sinr": [], "mcs0": []})
            rv = min(int(c[ix["rv"]]), 3)
            bad = int(c[ix["corrupt"]])
            p["rv"][rv, 0] += 1
            p["rv"][rv, 1] += bad
            p["sinr"].append(float(c[ix["sinr(db)"]]))
            if rv == 0:
                p["mcs0"].append(int(c[ix["mcs"]]))
    return per


def map_rnti(per, snr_bw):
    """Greedy nearest match of per-RNTI median SINR to the UEs' whole-band SNR; returns {rnti: ue}, max error."""
    pairs = sorted((abs(float(np.median(v["sinr"])) - s), rn, u) for rn, v in per.items() for u, s in enumerate(snr_bw))
    used_r, used_u, m, err = set(), set(), {}, 0.0
    for e, rn, u in pairs:
        if rn in used_r or u in used_u:
            continue
        used_r.add(rn)
        used_u.add(u)
        m[rn] = u
        err = max(err, e)
    return m, err


def sr_to_dci(d):
    """UE-side SR transmission -> next UL DCI reception at the same UE (by the trace's nodeId column), in ms."""
    sr, dci = {}, {}
    for name, want, dst in (("TxedUeMacCtrlMsgsTrace.txt", "SR", sr), ("RxedUeMacCtrlMsgsTrace.txt", "UL_DCI", dci)):
        with open(os.path.join(d, name)) as f:
            f.readline()
            for line in f:
                c = line.rstrip("\n").split("\t")
                if c[-1] == want:
                    dst.setdefault(c[6], []).append(float(c[0]))
    out = []
    for node, ts in sr.items():
        g = np.sort(np.array(dci.get(node, [])))
        if not len(g):
            continue
        t = np.array(ts)
        k = np.searchsorted(g, t, side="left")
        ok = k < len(g)
        out.append((g[k[ok]] - t[ok]) * 1e3)
    return np.concatenate(out) if out else np.zeros(0)


def main(sweep, out):
    os.makedirs(os.path.join(out, "lena"), exist_ok=True)
    runs, ues_rows, sr_rows = [], [], []
    for d in sorted(glob.glob(os.path.join(sweep, "*"))):
        if not os.path.exists(os.path.join(d, "frames.csv")):
            continue
        name = os.path.basename(d)
        meta = read_meta(d)
        summ = json.load(open(os.path.join(d, "summary.json")))
        app = float(meta.get("appStart", 0.5))
        T = float(meta["trafficTime"])
        ue_rows = list(csv.DictReader(open(os.path.join(d, "ues.csv"))))
        N = int(meta["nUe"])
        snr1 = np.array([float(r["snr1_db"]) for r in ue_rows])
        snrbw = np.array([float(r["snr_bw_db"]) for r in ue_rows])
        ue, step, nbytes, status, delay = parse_frames(d, app)
        np.savez_compressed(os.path.join(out, "lena", name + ".npz"), ue=ue, step=step, bytes=nbytes, status=status,
                            delay_ms=delay, snr1_db=snr1, snr_bw_db=snrbw)
        per = parse_phy(d)
        r2u, merr = map_rnti(per, snrbw)
        rv = sum((v["rv"] for v in per.values()), np.zeros((4, 2), dtype=np.int64))
        ok = status == 0
        dl = delay[ok]
        q = (lambda x: float(np.percentile(dl, x)) if len(dl) else float("nan"))
        f = name.split("_f")[1].split("_")[0]
        runs.append(dict(run=name, N=N, S=int(meta["frameBytes"]), load=float(f), p=float(meta["p"]),
                         seed=int(meta["run"]), T=T, frames=len(ue), ok=int(ok.sum()), late=int((status == 1).sum()),
                         incomplete=int((status == 2).sum()), drop=1 - ok.mean() if len(ue) else float("nan"),
                         p50_ms=q(50), p95_ms=q(95), p99_ms=q(99), mean_ms=float(dl.mean()) if len(dl) else float("nan"),
                         offered_kbps=nbytes.sum() * 8 / T / 1e3, goodput_kbps=nbytes[ok].sum() * 8 / T / 1e3,
                         tbs=int(rv[:, 0].sum()), rv0=int(rv[0, 0]), rv1=int(rv[1, 0]), rv2=int(rv[2, 0]),
                         rv3=int(rv[3, 0]), corrupt=int(rv[:, 1].sum()), lost_tbs=int(rv[3, 1]),
                         first_tx_bler=rv[0, 1] / max(rv[0, 0], 1), retx_frac=rv[1:, 0].sum() / max(rv[:, 0].sum(), 1),
                         prb_util=summ["ul_prb"]["prb_util_all_ul_slots"], wall_per_sim_s=float(meta["wallPerSimSecond"]),
                         rnti_map_max_err_db=merr, rnti_unmapped=len(per) - len(r2u)))
        u2r = {u: rn for rn, u in r2u.items()}
        for u in range(N):
            m = ue == u
            okm = m & ok
            v = per.get(u2r.get(u, -1))
            rvu = v["rv"] if v is not None else np.zeros((4, 2), dtype=np.int64)
            du = delay[okm]
            ues_rows.append(dict(run=name, ue=u, snr1_db=snr1[u], snr_bw_db=snrbw[u], frames=int(m.sum()),
                                 ok=int(okm.sum()), goodput_kbps=nbytes[okm].sum() * 8 / T / 1e3,
                                 p50_ms=float(np.percentile(du, 50)) if len(du) else float("nan"),
                                 p95_ms=float(np.percentile(du, 95)) if len(du) else float("nan"),
                                 tbs=int(rvu[:, 0].sum()), rv0=int(rvu[0, 0]), retx_tbs=int(rvu[1:, 0].sum()),
                                 first_fail=int(rvu[0, 1]), lost_tbs=int(rvu[3, 1]),
                                 mcs_first_median=float(np.median(v["mcs0"])) if v is not None and v["mcs0"] else float("nan")))
        s = sr_to_dci(d)
        sr_rows.append(dict(run=name, N=N, S=runs[-1]["S"], load=runs[-1]["load"], n_sr=len(s),
                            p10_ms=float(np.percentile(s, 10)) if len(s) else float("nan"),
                            p50_ms=float(np.percentile(s, 50)) if len(s) else float("nan"),
                            mean_ms=float(s.mean()) if len(s) else float("nan"),
                            p90_ms=float(np.percentile(s, 90)) if len(s) else float("nan")))
        print(name, runs[-1]["frames"], f"drop {runs[-1]['drop']:.3f}", f"rnti err {merr:.3f} dB",
              f"SR->DCI p50 {sr_rows[-1]['p50_ms']:.2f} ms", flush=True)
    for fn, rows in (("lena_runs.csv", runs), ("lena_per_ue.csv", ues_rows), ("lena_sr_dci.csv", sr_rows)):
        with open(os.path.join(out, fn), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    print("runs", len(runs))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
