"""Tables of the OAI rfsim campaign (benchmarks/oai/campaign.py) -> benchmarks/oai/results/*.csv.

    python benchmarks/oai/analyze.py --work ~/oai_rfsim/campaign --out benchmarks/oai/results

Writes
    owd_grid.csv          per latency run: MAC config, frame size, rate, frames, delivery, virtual-time frame delay
                          quantiles, wall-clock p50 and the median rfsim speed (virtual s per wall s)
    pipeline.csv          per frame of the low-rate 100-byte runs: send -> first UL grant (DCI) -> first PUSCH with
                          data -> last PUSCH with data -> sink, from the T-tracer, with every event placed at the
                          air time of its slot (virtual time)
    pipeline_summary.csv  medians and quartiles of those components per MAC config
    harq.csv              per attenuation: UL MCS, SNR reports, HARQ transmissions per TB, first-transmission BLER,
                          residual loss, the HARQ round trip of inferred retransmission chains, frame delay
    contention.csv        per UE-count run and UE: offered and delivered goodput, delivery, delay quantiles, Jain
Every run is ingested with tools/measure/ingest.py (unified tables next to each run).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))

from isaac_net.tools.measure import ingest, owd  # noqa: E402
from isaac_net.tools.measure.calibrate import harq_chains  # noqa: E402

SLOT_MS = 0.5


def q(x, ps=(5, 25, 50, 75, 95, 99)):
    x = np.asarray([v for v in x if v == v], float)
    if x.size == 0:
        return {f"p{p}": float("nan") for p in ps}
    return {f"p{p}": round(float(np.percentile(x, p)), 3) for p in ps}


def write(rows, path):
    if not rows:
        return
    cols = list(rows[0])
    for r in rows[1:]:
        cols += [c for c in r if c not in cols]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)")


def runs_of(work):
    out = []
    for camp in sorted(os.listdir(work)):
        d = os.path.join(work, camp)
        if not camp.startswith("campaign_") or not os.path.isdir(d):
            continue
        for rd in ingest.find_runs(d):
            out.append((camp[len("campaign_"):], rd))
    return out


def owd_grid(runs, tabs):
    rows = []
    for cfg, rd, m in runs:
        if m["experiment"] not in ("a", "d"):
            continue
        fr = tabs[rd]["frames"]
        tr = m.get("traffic", {})
        # wall-clock delays from the raw logs, for contrast
        _, frw, _ = owd.from_probe_logs(os.path.join(rd, "rx.csv"), os.path.join(rd, "tx_ue1.csv"))
        rows.append({"mac_config": cfg, "run_id": m["run_id"], "profile": tr.get("profile"), "size_b": tr.get("size"),
                     "rate_hz": tr.get("rate_hz"), "holdout": m.get("holdout", False), "frames": len(fr),
                     "delivered": round(sum(f["complete"] for f in fr) / max(1, len(fr)), 4),
                     **{f"delay_{k}_ms": v for k, v in q([f["delay_ms"] for f in fr]).items()},
                     "wall_delay_p50_ms": q([f["delay_ms"] for f in frw])["p50"],
                     "speed_median": round(m.get("timing", {}).get("speed_median", float("nan")), 3)})
    return rows


def air_time(rows, rd):
    """Air time (virtual epoch s) of the slot of each T-tracer row: the clock samples map slot ticks to virtual time
    (vtime.csv: v = absolute slot index x 0.5 ms), so a row's frame / slot, unwrapped near its own mapped timestamp,
    gives the time of that slot on the same axis as the probe timestamps. The rows' own timestamps are the gNB's
    processing times, which lead the air time for scheduling decisions and trail it for decoding."""
    from isaac_net.bridges.oai.vclock import load_vtime_csv
    w, v = load_vtime_csv(os.path.join(rd, "vtime.csv"))
    w0, v0 = int(w[0]) / 1e9, float(v[0])
    cyc = 1024 * 20
    for r in rows:
        est = (v0 + (r["t_s"] - w0)) / (SLOT_MS / 1e3)
        raw = r["sfn"] * 20 + r["slot"]
        n = raw + cyc * round((est - raw) / cyc)
        r["_slot"] = n
        r["_air"] = w0 + (n * SLOT_MS / 1e3 - v0)
    return rows


def pipeline(runs, tabs):
    rows = []
    for cfg, rd, m in runs:
        tr = m.get("traffic", {})
        if m["experiment"] != "a" or tr.get("size") != 100 or tr.get("rate_hz", 99) > 10 or tr.get("profile") != "cbr":
            continue
        sched = air_time([dict(r) for r in tabs[rd]["sched"] if r["dir"] == "UL" and r["sfn"] >= 0], rd)
        dci = sorted((r["_air"], r["_slot"], r["tbs_bytes"]) for r in sched if r["event"] == "sched")
        pdu = sorted((r["_air"], r["_slot"], r["data_bytes"], r["tbs_bytes"]) for r in sched
                     if r["event"] == "rx" and r["crc"] == 1)
        # grant of every PUSCH: every DCI yields one PUSCH, in order, so match receptions (pc rows: every detected
        # PUSCH, CRC OK or not) to DCIs first in, first out
        # (skipped: when grants go unused the UE sends nothing, no pc row exists and the pairing would slip)
        grant_of, pend, j = {}, [], 0
        pcs = sorted((r for r in sched if r["event"] == "pc"), key=lambda r: r["_slot"])
        if len(pcs) < 0.98 * len(dci):
            pcs = []
        for r in pcs:
            while j < len(dci) and dci[j][1] < r["_slot"]:
                pend.append(dci[j])
                j += 1
            if pend:
                grant_of[r["_slot"]] = pend.pop(0)
        dt, pt = np.array([d[0] for d in dci]), np.array([p[0] for p in pdu])
        for f in tabs[rd]["frames"]:
            if not f["complete"]:
                continue
            t0, t1 = f["t_tx_first_s"], f["t_rx_last_s"]
            i = int(np.searchsorted(dt, t0))
            j0, j1 = int(np.searchsorted(pt, t0)), int(np.searchsorted(pt, t1))
            data = [p for p in pdu[j0:j1] if p[2] > 0]
            if i >= len(dci) or not data:
                continue
            g_data = grant_of.get(data[0][1])
            rows.append({"mac_config": cfg, "run_id": m["run_id"], "frame_id": f["frame_id"],
                         "send_to_first_grant_ms": round((dci[i][0] - t0) * 1e3, 3),
                         "send_to_first_data_pusch_ms": round((data[0][0] - t0) * 1e3, 3),
                         "grant_to_pusch_slots": data[0][1] - g_data[1] if g_data else -1,
                         "first_to_last_data_pusch_ms": round((data[-1][0] - data[0][0]) * 1e3, 3),
                         "last_pusch_to_sink_ms": round((t1 - data[-1][0]) * 1e3, 3),
                         "total_ms": round(f["delay_ms"], 3), "data_tbs": len(data),
                         "first_pusch_tbs_b": data[0][3]})
    summ = []
    by = defaultdict(list)
    for r in rows:
        by[r["mac_config"]].append(r)
    for cfg, rs in sorted(by.items()):
        s = {"mac_config": cfg, "frames": len(rs)}
        for k in ("send_to_first_grant_ms", "send_to_first_data_pusch_ms", "grant_to_pusch_slots",
                  "first_to_last_data_pusch_ms", "last_pusch_to_sink_ms", "total_ms", "data_tbs",
                  "first_pusch_tbs_b"):
            v = q([r[k] for r in rs if r[k] >= 0], (25, 50, 75))
            s.update({f"{k}_{p}": v[p] for p in v})
        summ.append(s)
    return rows, summ


def harq(runs, tabs):
    rows = []
    for cfg, rd, m in runs:
        if m["experiment"] != "c":
            continue
        t = tabs[rd]
        up = [x for x in t["ue_period"] if x["dir"] == "UL"]
        rk = [sum(x[f"tx_r{k}"] for x in up if x[f"tx_r{k}"] >= 0) for k in range(4)]
        fail = sum(x["n_fail"] for x in up if x["n_fail"] >= 0)
        ch = [c for c in harq_chains(t["sched"])]
        gaps = [g for c in ch for g in c["gaps"] if g > 0]
        pc = [r for r in t["sched"] if r["event"] == "pc"]
        rx = [r for r in t["sched"] if r["event"] == "rx"]
        fr = t["frames"]
        r0 = max(rk[0], 1)
        rows.append({"mac_config": cfg, "run_id": m["run_id"], "ul_atten_db": m["radio"]["atten_db"],
                     "ul_mcs_median": float(np.median([r["mcs"] for r in pc])) if pc else float("nan"),
                     "pusch_snr_T_median_db": float(np.median([r["sinr_db"] for r in pc])) if pc else float("nan"),
                     "pusch_snr_stats_median_db": float(np.median([x["snr_db"] for x in up])) if up else float("nan"),
                     "tb_first_tx": rk[0], "bler_first_tx": round(rk[1] / r0, 4), "p_3rd_tx": round(rk[2] / r0, 4),
                     "p_4th_tx": round(rk[3] / r0, 4), "residual_loss": round(fail / r0, 4),
                     "crc_fail_T": sum(1 for r in rx if r["crc"] == 0), "crc_ok_T": sum(1 for r in rx if r["crc"] == 1),
                     "harq_rtt_slots_median": float(np.median(gaps)) if gaps else float("nan"),
                     "harq_rtt_samples": len(gaps),
                     "frames": len(fr), "delivered": round(sum(f["complete"] for f in fr) / max(1, len(fr)), 4),
                     **{f"delay_{k}_ms": v for k, v in q([f["delay_ms"] for f in fr], (50, 95, 99)).items()}})
    return sorted(rows, key=lambda r: (r["mac_config"], r["ul_atten_db"]))


def contention(runs, tabs):
    rows = []
    for cfg, rd, m in runs:
        if m["experiment"] != "b":
            continue
        fr = tabs[rd]["frames"]
        if not fr:
            continue
        dur = max(f["t_tx_first_s"] for f in fr) - min(f["t_tx_first_s"] for f in fr)
        per = []
        for ue in sorted({f["ue"] for f in fr}):
            fu = [f for f in fr if f["ue"] == ue]
            good = sum(f["frame_bytes"] for f in fu if f["complete"]) * 8 / dur
            per.append({"mac_config": cfg, "run_id": m["run_id"], "n_ue": m["n_ue"], "ue": ue,
                        "offered_mbps": round(sum(f["frame_bytes"] for f in fu) * 8 / dur / 1e6, 4),
                        "goodput_mbps": round(good / 1e6, 4),
                        "delivered": round(sum(f["complete"] for f in fu) / len(fu), 4),
                        **{f"delay_{k}_ms": v for k, v in q([f["delay_ms"] for f in fu], (50, 95, 99)).items()},
                        "speed_median": round(m.get("timing", {}).get("speed_median", float("nan")), 3)})
        g = np.array([p["goodput_mbps"] for p in per])
        jain = float(g.sum() ** 2 / (len(g) * (g ** 2).sum())) if (g ** 2).sum() > 0 else float("nan")
        for p in per:
            p["jain"] = round(jain, 4)
        rows += per
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    ap.add_argument("--no-reparse", action="store_true")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    runs, tabs = [], {}
    for cfg, rd in runs_of(a.work):
        if a.no_reparse and os.path.isdir(os.path.join(rd, "unified")):
            from isaac_net.tools.measure.schema import TABLES, read_table
            tabs[rd] = {t: read_table(os.path.join(rd, "unified", f"{t}.csv"), t) for t in TABLES}
        else:
            tabs[rd], _ = ingest.ingest_run(rd)
        runs.append((cfg, rd, ingest.load_manifest(rd)))
    write(owd_grid(runs, tabs), os.path.join(a.out, "owd_grid.csv"))
    pl, ps = pipeline(runs, tabs)
    write(pl, os.path.join(a.out, "pipeline.csv"))
    write(ps, os.path.join(a.out, "pipeline_summary.csv"))
    write(harq(runs, tabs), os.path.join(a.out, "harq.csv"))
    write(contention(runs, tabs), os.path.join(a.out, "contention.csv"))
    with open(os.path.join(a.out, "runs.json"), "w") as f:
        json.dump([{"mac_config": c, **{k: m.get(k) for k in ("run_id", "experiment", "n_ue", "traffic", "radio",
                                                                "gnb", "timing")}} for c, _, m in runs], f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
