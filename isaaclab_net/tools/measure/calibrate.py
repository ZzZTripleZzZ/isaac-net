"""Calibration hooks: unified measurement tables -> fitted parameters -> an NRConfig preset file.

Outputs mirror the public-data calibration (docs/calibration-public-data.md), one set per stack (srsran, oai):

- ``params_latency.json``: frame-delay quantiles per run, the latency components read directly from the logs
  (SR-to-PUSCH delay, HARQ round trip, proactive grants) and, with ``--engine-fit``, the engine replay fit of the
  SR / proactive-grant knobs and the processing offset d0 (W1, KS per run, held-out runs scored separately).
- ``params_link.json``: per SINR position, MCS, first-transmission BLER, HARQ transmission counts; the link
  adaptation offset against the engine's PHY tables, a maximum-likelihood SINR shift of the engine BLER model on
  per-TB CRC data, and a logistic BLER slope.
- ``params_contention.json``: per UE count, per-UE goodput, Jain fairness, frame delay quantiles, and the
  capacity scale eta when the cell was saturated.
- ``preset_<stack>.json``: NRConfig overrides on top of srsran_like() / oai_like() (load with
  ``preset.load_preset``), with the provenance of every field.

  python -m isaaclab_net.tools.measure.calibrate CAMPAIGN_DIR --out calib_out [--engine-fit]

Experiments are tagged in each run manifest: ``a`` (single-UE latency grid), ``b`` (UE-count sweep), ``c``
(fixed SINR positions), ``d`` (robot traffic replay, held out from the latency fit and used to validate it).
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np

from .ingest import ingest_campaign, load_campaign, tdd_pattern, TDD_PERIOD_MS
from .preset import write_preset

nan = float("nan")


def _fin(x):
    return x is not None and not (isinstance(x, float) and math.isnan(x))


def _q(xs, ps=(0.05, 0.5, 0.9, 0.95, 0.99)):
    x = np.asarray([v for v in xs if _fin(v)], float)
    if x.size == 0:
        return {f"p{int(p * 100)}": nan for p in ps} | {"n": 0}
    return {f"p{int(p * 100)}": float(np.quantile(x, p)) for p in ps} | {"n": int(x.size), "mean": float(x.mean())}


# ------------------------------------------------------------------ configuration facts
def config_facts(g, stack):
    """NRConfig fields known from the gNB configuration recorded in the manifest ("gnb" block)."""
    cfg, prov = {}, {}
    mu = int(g.get("mu", 1))
    cfg["mu"] = mu
    if "bandwidth_mhz" in g:
        cfg["bandwidth_mhz"] = int(g["bandwidth_mhz"])
    if "tdd_pattern" in g:
        cfg["tdd_pattern"] = g["tdd_pattern"]
        if "special_split" in g:
            cfg["special_split"] = tuple(g["special_split"])
    elif "tdd" in g:
        t = g["tdd"]
        per = t.get("period_slots")
        if per is None:
            per = int(round(TDD_PERIOD_MS[int(t["periodicity_idx"])] * 2 ** mu))
        cfg["tdd_pattern"], cfg["special_split"] = tdd_pattern(per, t["nof_dl_slots"], t.get("nof_dl_symbols", 0),
                                                               t["nof_ul_slots"], t.get("nof_ul_symbols", 0))
    if "special_ul_data" in g:
        cfg["special_ul_data"] = bool(g["special_ul_data"])
    if g.get("sr_period_ms"):
        cfg["sr_period_slots"] = int(round(float(g["sr_period_ms"]) * 2 ** mu))
    if g.get("min_k2") is not None:
        cfg["k2"] = int(g["min_k2"])
    if g.get("mcs_table"):
        cfg["mcs_table"] = 2 if str(g["mcs_table"]).lower() in ("2", "qam256") else 1
    if g.get("ul_mcs_max") is not None:
        cfg["ul_mcs_max"] = int(g["ul_mcs_max"])
    if g.get("max_harq_tx") is not None:
        cfg["max_harq_tx"] = int(g["max_harq_tx"])
    if g.get("n_harq") is not None:
        cfg["n_harq"] = int(g["n_harq"])
    if g.get("olla_target_bler") is not None:                     # srsRAN pusch.olla_target_bler
        cfg["bler_target"] = float(g["olla_target_bler"])
    elif g.get("ul_bler_target_upper") is not None:              # OAI MACRLCs: the controller keeps BLER in band
        cfg["bler_target"] = 0.5 * (float(g["ul_bler_target_upper"]) + float(g.get("ul_bler_target_lower", 0)))
    for k in cfg:
        prov[k] = "gNB configuration (manifest)"
    return cfg, prov


# ------------------------------------------------------------------ helpers on sched rows
_TBS_CACHE = {}


def infer_mcs(rows, mcs_table=1):
    """Fill missing MCS from (TBS, PRBs, symbols, modulation) by exact TS 38.214 TBS inversion, trying 1..3 DMRS
    symbols of type 1 (12, 24, 36 REs per PRB). Rows where no MCS reproduces the TBS keep mcs = -1."""
    import torch
    from isaaclab_net.core.phy import MCS_TABLES, tbs_38214
    tab = MCS_TABLES[mcs_table]
    qm = torch.tensor([q for q, _ in tab], dtype=torch.float64)
    r = torch.tensor([c / 1024 for _, c in tab], dtype=torch.float64)
    n = 0
    for row in rows:
        if row["mcs"] >= 0 or row["tbs_bytes"] <= 0 or row["n_prb"] <= 0 or row["n_sym"] <= 0:
            continue
        key = (row["tbs_bytes"], row["n_prb"], row["n_sym"], row["qm"], mcs_table)
        if key not in _TBS_CACHE:
            hit = -1
            for dmrs in (12, 24, 36):
                tbs = tbs_38214(qm, r, torch.tensor(float(row["n_prb"])), torch.tensor(float(row["n_sym"])), dmrs)
                ok = (tbs == row["tbs_bytes"] * 8) & ((qm == row["qm"]) if row["qm"] > 0 else True)
                idx = torch.nonzero(ok).flatten().tolist()
                if idx:
                    hit = idx[0]
                    break
            _TBS_CACHE[key] = hit
        row["mcs"] = _TBS_CACHE[key]
        n += row["mcs"] >= 0
    return n


def harq_chains(rows):
    """Reconstruct UL HARQ chains from rx rows with CRC (srsRAN PHY log): per (run, ue, harq id), a chain starts
    at rv 0 / new data and ends at CRC OK or when the next chain starts. Returns dicts with ntx, success, the
    first transmission's MCS / SINR and the slot gaps between transmissions (the HARQ round trip)."""
    by = defaultdict(list)
    for r in rows:
        if r["dir"] == "UL" and r["event"] == "rx" and r["crc"] in (0, 1) and r["harq_id"] >= 0:
            by[(r["run_id"], r["ue"], r["harq_id"])].append(r)
    chains = []
    for k, rs in by.items():
        rs.sort(key=lambda r: (r["slot_abs"], r["t_s"]))
        cur = None
        for r in rs:
            start = r["newtx"] == 1 or r["rv"] == 0 or cur is None or cur["done"]
            if start:
                if cur is not None:
                    chains.append(cur)
                cur = {"run_id": k[0], "ue": k[1], "ntx": 0, "success": False, "done": False, "mcs": r["mcs"],
                       "sinr_db": r["sinr_db"], "tbs_bytes": r["tbs_bytes"], "gaps": [], "last": None}
            if cur["last"] is not None and r["slot_abs"] >= 0 and cur["last"] >= 0:
                cur["gaps"].append(r["slot_abs"] - cur["last"])
            cur["last"] = r["slot_abs"]
            cur["ntx"] += 1
            if r["crc"] == 1:
                cur["success"], cur["done"] = True, True
        if cur is not None:
            chains.append(cur)
    return chains


def proactive_grants(rows, cfg_facts):
    """Detect grants without data: UL PDUs (pcap rx rows) with no SDU bytes. Returns (label, evidence)."""
    ul = sorted((r for r in rows if r["dir"] == "UL" and r["event"] == "rx" and r["data_bytes"] >= 0
                 and r["slot_abs"] >= 0), key=lambda r: r["slot_abs"])
    if len(ul) < 20:
        return None, {"n_pdus": len(ul)}
    empty = [r for r in ul if r["data_bytes"] == 0]
    frac = len(empty) / len(ul)
    ev = {"n_pdus": len(ul), "frac_without_data": frac}
    if frac < 0.2 or len(empty) < 10:
        return "off", ev
    by = defaultdict(list)
    for r in empty:
        by[r["ue"]].append(r["slot_abs"])
    gaps = [b - a for s in by.values() for a, b in zip(s, s[1:]) if b > a]
    if not gaps:
        return None, ev
    g = float(np.median(gaps))
    P = len(cfg_facts.get("tdd_pattern", "DDDSU"))
    ev["median_gap_slots"] = g
    ev["tdd_period_slots"] = P
    if g <= P / max(1, cfg_facts.get("tdd_pattern", "DDDSU").count("U")) + 0.5:
        return "every_ul_slot", ev
    if abs(g - P) <= 1:
        return "per_period", ev
    ev["note"] = "periodic grants without data at a period the engine has no knob for"
    return "per_period" if g < 4 * P else "off", ev


# ------------------------------------------------------------------ latency
def latency_components(sched, ue_period, facts, mu):
    comp, prov = {}, {}
    srp = [r["sr_to_pusch_avg_ms"] for r in ue_period if r["dir"] == "UL" and _fin(r["sr_to_pusch_avg_ms"])
           and r["sr_to_pusch_avg_ms"] > 0]
    if srp:
        comp["sr_to_pusch_ms_median"] = float(np.median(srp))
        comp["sr_grant_delay_slots"] = int(round(np.median(srp) * 2 ** mu))
        prov["sr_grant_delay_slots"] = f"srsRAN metrics avg_sr_to_pusch_delay, median of {len(srp)} reports"
    chains = harq_chains(sched)
    gaps = [g for c in chains for g in c["gaps"] if g > 0]
    if len(gaps) >= 5:
        comp["ul_harq_rtt_slots"] = int(round(np.median(gaps)))
        comp["ul_harq_rtt_samples"] = len(gaps)
        prov["ul_harq_rtt_slots"] = f"median slot gap between retransmissions of one HARQ process ({len(gaps)})"
    crc = [r["crc_delay_avg_ms"] for r in ue_period if _fin(r["crc_delay_avg_ms"]) and r["crc_delay_avg_ms"] > 0]
    if crc:
        comp["crc_delay_ms_median"] = float(np.median(crc))
    pg, ev = proactive_grants(sched, facts)
    if pg is not None:
        comp["proactive_grant"] = pg
        comp["proactive_evidence"] = ev
        prov["proactive_grant"] = f"UL PDUs without data in the MAC pcap: {ev}"
    return comp, prov


def _arrivals(frames, run_id):
    fr = [f for f in frames if f["run_id"] == run_id]
    if not fr:
        return [], [], []
    ues = sorted({f["ue"] for f in fr})
    t0 = min(f["t_tx_first_s"] for f in fr)
    arr = [(f["t_tx_first_s"] - t0, ues.index(f["ue"]), f["frame_bytes"]) for f in fr]
    meas = [f["delay_ms"] for f in fr]
    return arr, meas, ues


def _run_snr(ue_period, run, default=25.0):
    s = [r["snr_db"] for r in ue_period if r["run_id"] == run["run_id"] and r["dir"] == "UL" and _fin(r["snr_db"])]
    if s:
        return float(np.median(s))
    return float(run.get("radio", {}).get("sinr_setpoint_db") or default)


def engine_latency_fit(runs, frames, ue_period, base_cfg, grid, replicas=4, max_s=20.0, log=print):
    """Grid search of the engine's latency knobs by replaying experiment a runs (fit) and d runs (held out).
    grid: {NRConfig field: [values]}. d0 is shared by all fit runs (median quantile difference)."""
    from .replay import quantiles, replay, w1_shift
    fit_runs = [r for r in runs if r.get("experiment") == "a" and not r.get("holdout")]
    ho_runs = [r for r in runs if r.get("experiment") == "d" or (r.get("experiment") == "a" and r.get("holdout"))]
    data = {}
    for r in fit_runs + ho_runs:
        arr, meas, ues = _arrivals(frames, r["run_id"])
        if arr:
            data[r["run_id"]] = (arr, meas, ues, _run_snr(ue_period, r))
    if not any(r["run_id"] in data for r in fit_runs):
        return None
    keys = sorted(grid)
    results = []
    for vals in itertools.product(*(grid[k] for k in keys)):
        cand = dict(zip(keys, vals))
        cfg = base_cfg.with_(**cand)
        sims = {}
        for rid, (arr, meas, ues, snr) in data.items():
            sims[rid] = replay(cfg, arr, len(ues), snr, replicas=replicas, max_s=max_s)
        keep = {rid: [i for i, a in enumerate(arr) if max_s is None or a[0] <= max_s]
                for rid, (arr, _, _, _) in data.items()}
        diffs = [quantiles([data[rid][1][i] for i in keep[rid]]) - quantiles(sims[rid][:, keep[rid]].ravel())
                 for rid in (r["run_id"] for r in fit_runs) if rid in data]
        d0 = max(0.0, float(np.nanmedian(np.concatenate(diffs))))
        per = {}
        for rid, (arr, meas, _, _) in data.items():
            w1, k, _ = w1_shift([meas[i] for i in keep[rid]], sims[rid][:, keep[rid]].ravel(), d0)
            per[rid] = {"w1_ms": w1, "ks": k}
        fit_ids = [r["run_id"] for r in fit_runs if r["run_id"] in per]
        ho_ids = [r["run_id"] for r in ho_runs if r["run_id"] in per]
        res = {"params": cand, "d0_ms": d0, "per_run": per,
               "fit_w1_ms": float(np.mean([per[i]["w1_ms"] for i in fit_ids])),
               "fit_ks": float(np.mean([per[i]["ks"] for i in fit_ids])),
               "holdout_w1_ms": float(np.mean([per[i]["w1_ms"] for i in ho_ids])) if ho_ids else nan,
               "holdout_ks": float(np.mean([per[i]["ks"] for i in ho_ids])) if ho_ids else nan}
        log(f"  {cand} d0={d0:.2f} ms  fit W1={res['fit_w1_ms']:.2f} KS={res['fit_ks']:.2f}  "
            f"held-out W1={res['holdout_w1_ms']:.2f}")
        results.append(res)
    results.sort(key=lambda x: x["fit_w1_ms"])
    return {"best": results[0], "grid": results, "fit_runs": [r["run_id"] for r in fit_runs],
            "holdout_runs": [r["run_id"] for r in ho_runs], "replay_max_s": max_s, "replicas": replicas}


def default_grid(facts, comp, stack, mu):
    P = len(facts.get("tdd_pattern", "DDDSU"))
    g = {"proactive_grant": [comp["proactive_grant"]] if comp.get("proactive_grant") else ["off", "per_period"]}
    if "sr_period_slots" not in facts:
        g["sr_period_slots"] = sorted({P, int(10 * 2 ** mu), int(20 * 2 ** mu)})
    sd = comp.get("sr_grant_delay_slots")
    g["sr_grant_delay_slots"] = [sd] if sd else sorted({P, 2 * P, 4 * P})
    return g


# ------------------------------------------------------------------ link and HARQ
def link_fit(sched, ue_period, runs, facts, log=print):
    import torch
    from isaaclab_net.core.phy import PHY
    table = facts.get("mcs_table", 1)
    target = facts.get("bler_target", 0.1)
    phy = PHY("ul", table, "cpu", bler_target=target)
    thr = phy.thr_ref.tolist()
    n_inf = infer_mcs(sched, table)
    out = {"mcs_table": table, "engine_bler_target": target, "mcs_inferred_from_tbs": int(n_inf), "positions": []}
    c_runs = [r for r in runs if r.get("experiment") == "c"] or runs
    chains = harq_chains(sched)
    la_pts = []                                                    # (sinr, mcs) points for the LA offset
    for r in c_runs:
        rid = r["run_id"]
        up = [x for x in ue_period if x["run_id"] == rid and x["dir"] == "UL"]
        ch = [c for c in chains if c["run_id"] == rid]
        pos = {"run_id": rid, "atten_db": r.get("radio", {}).get("atten_db"),
               "sinr_setpoint_db": r.get("radio", {}).get("sinr_setpoint_db")}
        snr = [x["snr_db"] for x in up if _fin(x["snr_db"])] + [c["sinr_db"] for c in ch if _fin(c["sinr_db"])]
        mcs = [x["mcs"] for x in up if _fin(x["mcs"]) and x["mcs"] >= 0] + [c["mcs"] for c in ch if c["mcs"] >= 0]
        pos["sinr_db"] = float(np.median(snr)) if snr else nan
        pos["mcs"] = float(np.median(mcs)) if mcs else nan
        if ch:
            n = len(ch)
            pos["n_tb"] = n
            pos["bler_first_tx"] = sum(c["ntx"] > 1 or not c["success"] for c in ch) / n
            pos["residual_loss"] = sum(not c["success"] for c in ch) / n
            mx = max(c["ntx"] for c in ch)
            pos["ntx_hist"] = [sum(c["ntx"] == k for c in ch) / n for k in range(1, mx + 1)]
            pos["source"] = "per-TB HARQ chains"
        else:
            r0 = sum(x["tx_r0"] for x in up if x["tx_r0"] > 0)
            if r0 > 0:                                             # OAI round counters
                rk = [sum(x[f"tx_r{k}"] for x in up if x[f"tx_r{k}"] >= 0) for k in range(4)]
                pos["n_tb"] = r0
                pos["bler_first_tx"] = rk[1] / r0
                pos["residual_loss"] = sum(x["n_fail"] for x in up if x["n_fail"] >= 0) / r0
                pos["ntx_hist"] = [(rk[k] - (rk[k + 1] if k + 1 < 4 else 0)) / r0 for k in range(4)]
                pos["source"] = "OAI ulsch_rounds counters"
            else:
                ok = sum(x["n_ok"] for x in up if x["n_ok"] >= 0)
                nok = sum(x["n_nok"] for x in up if x["n_nok"] >= 0)
                if ok + nok > 0:
                    pos["n_tx"] = ok + nok
                    pos["bler_per_tx"] = nok / (ok + nok)
                    pos["source"] = "srsRAN ul_nof_ok / ul_nof_nok (all transmissions)"
        if _fin(pos["sinr_db"]) and _fin(pos["mcs"]):
            la_pts.append((pos["sinr_db"], int(round(pos["mcs"]))))
        out["positions"].append(pos)
    # link-adaptation offset: measured SINR minus the engine's target-BLER threshold of the chosen MCS
    tb_pts = [(c["sinr_db"], c["mcs"]) for c in chains if _fin(c["sinr_db"]) and c["mcs"] >= 0]
    pts = tb_pts or la_pts
    if pts:
        off = [s - thr[m] for s, m in pts if 0 <= m < len(thr)]
        out["la_offset_db"] = float(np.median(off))
        out["la_offset_note"] = ("median of (measured SINR - engine 10-PRB threshold of the MCS the gNB chose); "
                                 "positive = the stack is more conservative than the engine's link adaptation")
    # BLER model: ML SINR shift of the engine TB error model on first transmissions with CRC
    first = [c for c in chains if c["mcs"] >= 0 and _fin(c["sinr_db"]) and c["tbs_bytes"] > 0]
    if len(first) >= 30:
        mcs_t = torch.tensor([c["mcs"] for c in first])
        sinr_t = torch.tensor([c["sinr_db"] for c in first], dtype=torch.float32)
        tbs_t = torch.tensor([c["tbs_bytes"] * 8 for c in first], dtype=torch.float32)
        fail = np.array([c["ntx"] > 1 or not c["success"] for c in first], float)
        best = None
        for d in np.arange(-12, 12.01, 0.25):
            p = phy.tb_error_prob(mcs_t, sinr_t + float(d), tbs_t).double().clamp(1e-6, 1 - 1e-6).numpy()
            nll = -float(np.sum(fail * np.log(p) + (1 - fail) * np.log(1 - p)))
            if best is None or nll < best[1]:
                best = (float(d), nll)
        out["bler_sinr_shift_db"] = best[0]
        out["bler_shift_nll"] = best[1]
        out["bler_n_tb"] = len(first)
        # logistic in x = SINR - threshold(MCS): P(fail) = 1 / (1 + exp(k (x - x0)))
        x = np.array([c["sinr_db"] - thr[c["mcs"]] for c in first])
        bestl = None
        for k in np.arange(0.1, 5.01, 0.05):
            for x0 in np.arange(-10, 10.01, 0.25):
                p = np.clip(1 / (1 + np.exp(k * (x - x0))), 1e-6, 1 - 1e-6)
                nll = -np.sum(fail * np.log(p) + (1 - fail) * np.log(1 - p))
                if bestl is None or nll < bestl[2]:
                    bestl = (float(k), float(x0), float(nll))
        out["logistic_slope_per_db"], out["logistic_x0_db"] = bestl[0], bestl[1]
    else:
        out["bler_note"] = f"{len(first)} first transmissions with MCS, SINR and CRC; need >= 30 for the BLER fit"
    # measured OLLA operating point -> engine bler_target
    b1 = [p["bler_first_tx"] for p in out["positions"] if _fin(p.get("bler_first_tx"))]
    if b1:
        out["bler_first_tx_median"] = float(np.median(b1))
    mx = [c["mcs"] for c in chains if c["mcs"] >= 0] + [int(x["mcs"]) for x in ue_period
                                                         if x["dir"] == "UL" and _fin(x["mcs"]) and x["mcs"] >= 0]
    if mx:
        out["ul_mcs_observed_max"] = int(np.max(mx))
    return out


# ------------------------------------------------------------------ contention
def contention_fit(frames, ue_period, runs, facts):
    from isaaclab_net.core.config import NRConfig
    import torch
    from isaaclab_net.core.phy import MCS_TABLES, tbs_38214
    b_runs = [r for r in runs if r.get("experiment") == "b"]
    out = {"per_n": []}
    for r in b_runs:
        rid = r["run_id"]
        fr = [f for f in frames if f["run_id"] == rid]
        if not fr:
            continue
        dur = max(f["t_tx_first_s"] for f in fr) - min(f["t_tx_first_s"] for f in fr)
        per_ue = {}
        for ue in sorted({f["ue"] for f in fr}):
            fu = [f for f in fr if f["ue"] == ue]
            good = sum(f["frame_bytes"] for f in fu if f["complete"]) * 8 / max(dur, 1e-9)
            offered = sum(f["frame_bytes"] for f in fu) * 8 / max(dur, 1e-9)
            per_ue[ue] = {"goodput_bps": good, "offered_bps": offered,
                          "delivery": sum(f["complete"] for f in fu) / len(fu),
                          "delay_ms": _q([f["delay_ms"] for f in fu])}
        g = np.array([v["goodput_bps"] for v in per_ue.values()])
        jain = float(g.sum() ** 2 / (len(g) * (g ** 2).sum())) if (g ** 2).sum() > 0 else nan
        row = {"run_id": rid, "n_ue": len(per_ue), "per_ue": per_ue, "cell_goodput_bps": float(g.sum()),
               "jain": jain}
        up = [x for x in ue_period if x["run_id"] == rid and x["dir"] == "UL"]
        mcs = [x["mcs"] for x in up if _fin(x["mcs"]) and x["mcs"] >= 0]
        cfg = NRConfig(**{k: v for k, v in facts.items() if k in ("mu", "bandwidth_mhz", "tdd_pattern",
                                                                   "special_split", "special_ul_data")})
        if mcs:
            m = int(round(float(np.median(mcs))))
            q, c = MCS_TABLES[facts.get("mcs_table", 1)][m]
            sym = sum(cfg.slot_symbols(i)[1] for i in range(len(cfg.tdd_pattern)))
            per_period = float(tbs_38214(torch.tensor(float(q)), torch.tensor(c / 1024), torch.tensor(float(cfg.nprb)),
                                         torch.tensor(float(cfg.ul_data_symbols)), 12)) * sym / cfg.ul_data_symbols
            cap = per_period / (len(cfg.tdd_pattern) * cfg.slot_ms / 1000)
            row["capacity_bps_at_median_mcs"] = cap
            row["utilization"] = float(g.sum()) / cap
        out["per_n"].append(row)
    sat = [x for x in out["per_n"] if _fin(x.get("utilization")) and
           sum(v["offered_bps"] for v in x["per_ue"].values()) > 1.2 * x["cell_goodput_bps"]]
    if sat:
        out["eta"] = float(np.median([x["utilization"] for x in sat]))
        out["eta_note"] = ("cell goodput / (38.214 TBS capacity of all PRBs at the median UL MCS) in runs whose "
                           "offered load exceeded the goodput by 20%; the same quantity as the ColO-RAN eta")
    else:
        out["eta_note"] = "no saturated run (offered load > 1.2 x goodput); eta not identified"
    return out


# ------------------------------------------------------------------ driver
def calibrate(campaign_dir, out_dir, engine_fit=False, grid=None, replicas=4, max_s=20.0, reparse=True, log=print):
    if reparse:
        tabs, runs = ingest_campaign(campaign_dir)
    else:
        tabs, runs = load_campaign(campaign_dir)
    os.makedirs(out_dir, exist_ok=True)
    written = {}
    for stack in sorted({r.get("stack", "") for r in runs}):
        sr = [r for r in runs if r.get("stack", "") == stack]
        ids = {r["run_id"] for r in sr}
        sub = {t: [x for x in rows if x["run_id"] in ids] for t, rows in tabs.items()}
        facts, prov = config_facts(sr[0].get("gnb", {}), stack)
        mu = facts["mu"]
        for r in sr[1:]:
            f2, _ = config_facts(r.get("gnb", {}), stack)
            if f2 != facts and r.get("experiment") in ("a", "d"):
                log(f"warning: run {r['run_id']} has a different gNB configuration; the preset uses {sr[0]['run_id']}")
        comp, cprov = latency_components(sub["sched"], sub["ue_period"], facts, mu)
        lat = {"stack": stack, "components": comp, "runs": []}
        for r in sr:
            if r.get("experiment") in ("a", "d", "b"):
                fr = [f for f in sub["frames"] if f["run_id"] == r["run_id"]]
                lat["runs"].append({"run_id": r["run_id"], "experiment": r.get("experiment"),
                                    "traffic": r.get("traffic", {}), "n_ue": r.get("n_ue"),
                                    "delay_ms": _q([f["delay_ms"] for f in fr]),
                                    "loss": (sum(1 - f["complete"] for f in fr) / len(fr)) if fr else nan})
        base = {"srsran": "srsran_like", "oai": "oai_like"}.get(stack)
        nrc = dict(facts)
        for k in ("sr_grant_delay_slots", "ul_harq_rtt_slots", "proactive_grant"):
            if k in comp:
                nrc[k] = comp[k]
                prov[k] = cprov[k]
        link = link_fit(sub["sched"], sub["ue_period"], sr, facts, log)
        knobs = {}
        if "bler_target" not in nrc and _fin(link.get("bler_first_tx_median")):
            nrc["bler_target"] = max(1e-3, link["bler_first_tx_median"])
            prov["bler_target"] = "median measured first-transmission BLER over the SINR positions"
        if "ul_mcs_max" not in nrc and link.get("ul_mcs_observed_max") is not None:
            nrc["ul_mcs_max"] = link["ul_mcs_observed_max"]
            prov["ul_mcs_max"] = "highest UL MCS observed"
        for k in ("la_offset_db", "bler_sinr_shift_db", "logistic_slope_per_db"):
            if k in link:
                knobs[k] = link[k]
        cont = contention_fit(sub["frames"], sub["ue_period"], sr, facts)
        if "eta" in cont:
            knobs["eta_capacity"] = cont["eta"]
        if engine_fit:
            from isaaclab_net.core import config as C
            base_cfg = getattr(C, base)(**nrc) if base else C.NRConfig(**nrc)
            if not any(r.get("radio", {}).get("fading") for r in sr):
                base_cfg = base_cfg.with_(fading=False)
            g = grid or default_grid(facts, comp, stack, mu)
            log(f"[{stack}] engine replay grid {g}")
            fit = engine_latency_fit(sr, sub["frames"], sub["ue_period"], base_cfg, g, replicas, max_s, log)
            if fit is not None:
                lat["engine_fit"] = fit
                b = fit["best"]
                nrc.update(b["params"])
                nrc["proc_offset_ms"] = round(b["d0_ms"], 3)
                for k in list(b["params"]) + ["proc_offset_ms"]:
                    prov[k] = (f"engine replay fit on experiment a (W1 {b['fit_w1_ms']:.2f} ms, KS {b['fit_ks']:.2f}; "
                               f"held out W1 {b['holdout_w1_ms']:.2f} ms)")
        else:
            prov["proc_offset_ms"] = "not fitted in this call (run with --engine-fit); base preset value kept"
        for name, obj in (("params_latency", lat), ("params_link", link), ("params_contention", cont)):
            p = os.path.join(out_dir, f"{name}_{stack}.json")
            with open(p, "w") as f:
                json.dump(obj, f, indent=1, default=float)
            written[f"{name}_{stack}"] = p
        pp = os.path.join(out_dir, f"preset_{stack}.json")
        write_preset(pp, f"{os.path.basename(os.path.normpath(campaign_dir))}_{stack}", base,
                     {k: (list(v) if isinstance(v, tuple) else v) for k, v in nrc.items()}, knobs, prov,
                     {"latency_runs": len(lat["runs"]), "link_positions": len(link["positions"]),
                      "contention_runs": len(cont["per_n"])})
        written[f"preset_{stack}"] = pp
        log(f"[{stack}] preset -> {pp}")
    return written


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("campaign")
    ap.add_argument("--out", default=None, help="output directory (default <campaign>/calib)")
    ap.add_argument("--engine-fit", action="store_true", help="replay experiment a/d arrivals through the engine")
    ap.add_argument("--grid", default=None, help='JSON {"NRConfig field": [values]} for the engine fit')
    ap.add_argument("--replicas", type=int, default=4)
    ap.add_argument("--max-s", type=float, default=20.0, help="seconds of each run to replay")
    ap.add_argument("--no-reparse", action="store_true", help="use <campaign>/unified as is")
    a = ap.parse_args(argv)
    calibrate(a.campaign, a.out or os.path.join(a.campaign, "calib"), a.engine_fit,
              json.loads(a.grid) if a.grid else None, a.replicas, a.max_s, not a.no_reparse)
    return 0


if __name__ == "__main__":
    sys.exit(main())
