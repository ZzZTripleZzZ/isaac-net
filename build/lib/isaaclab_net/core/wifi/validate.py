"""Validation of the Wi-Fi level: `python -m isaaclab_net.core.wifi.validate [--quick] [--out DIR] [--jobs N]`.

Tables (docs/wifi.md reports them):
  A  Bianchi's saturation throughput (his Table I FHSS parameters, basic access and RTS/CTS): the analytic model
     (exact root), the tensor solver after its fixed iterations, and the event-driven simulator.
  B  802.11ax 20 MHz, MCS 7, AC_BE, saturated stations: aggregate throughput and mean access delay of the
     mean-field model against the event-driven simulator, over the number of stations and the bytes per access.
  C  EDCA: AC_VO and AC_BE stations together, saturated: throughput per AC.
  D  The full WIFI engine (sub-steps, Poisson accesses, FIFO, timeouts) against the event-driven simulator driven
     with the same traffic: every robot submits one message per 100 ms control step, synchronized as in an RL env.
     Mean, median and 95th-percentile message delay and the delivered fraction.
  E  Sub-step length and access_noise ablation on two D scenarios.
  F  ns-3's wifi-bianchi example (802.11ax, no aggregation, saturated): recorded ns-3 throughputs against the
     mean-field model and the event-driven simulator with the same parameters.
Results go to DIR/wifi_validation.json and DIR/wifi_validation.md (default ~/.cache/isaaclab_net/wifi).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

from . import meanfield as mf
from .config import EDCA, WifiConfig
from .eventsim import Const, Station, run
from .phy import AccessTiming, mcs_table


# ------------------------------------------------------------------------------------------------ helpers
class WifiBusy:
    """Picklable busy time of a success (which=0) or a collision (which=1): the channel time of one access minus
    the AIFS, which the event simulator counts as idle slots."""

    def __init__(self, wc, rate, aifsn, which):
        self.tm, self.rate, self.aifsn, self.which = AccessTiming(wc), float(rate), float(aifsn), which
        self.aifs = wc.sifs_us + aifsn * wc.slot_us

    def __call__(self, nbytes):
        return float(self.tm.times(float(nbytes), self.rate, self.aifsn)[self.which]) - self.aifs


def _wifi_station(wc, ac, rate, B=None):
    tm = AccessTiming(wc)
    cw_min, cw_max, aifsn, txop = EDCA[ac]
    cap = float(tm.cap_bytes(torch.tensor([rate], dtype=torch.float64), torch.tensor([txop], dtype=torch.float64))[0])
    if B is not None:
        cap = float(B)
    return Station(W0=cw_min + 1, Wmax=cw_max + 1, aifsn=aifsn, cap=cap, busy_succ=WifiBusy(wc, rate, aifsn, 0),
                   busy_coll=WifiBusy(wc, rate, aifsn, 1), max_tx=wc.max_tx, fer=wc.frame_error_rate)


def meanfield_saturated(wc, groups):
    """groups: list of (n, ac, rate, B). Returns per-group (throughput Mb/s total, mean access us, p)."""
    tm = AccessTiming(wc)
    n_tot = sum(g[0] for g in groups)
    f = torch.float64
    rows = []
    for n, ac, rate, B in groups:
        cw_min, cw_max, aifsn, _ = EDCA[ac]
        rows += [(cw_min + 1, cw_max + 1, aifsn, rate, B)] * n
    W0 = torch.tensor([r[0] for r in rows], dtype=f)[None]
    Wm = torch.tensor([r[1] for r in rows], dtype=f)[None]
    aif = torch.tensor([r[2] for r in rows], dtype=f)[None]
    rate = torch.tensor([r[3] for r in rows], dtype=f)[None]
    B = torch.tensor([float(r[4]) for r in rows], dtype=f)[None]
    w = torch.ones(1, n_tot, dtype=f)
    view = mf.DomainView(torch.ones(1, n_tot, 1, dtype=f))
    amin = aif.min()
    Ts, Tc, _ = tm.times(B, rate, amin.expand_as(aif))
    Z = int(aif.max() - amin) + 1
    res = mf.solve(w, W0, Wm, wc.max_tx, Ts, Tc, wc.slot_us, view, fer=wc.frame_error_rate, d=aif - amin, Z=Z,
                   iters=300, damp=0.5)
    out, i = [], 0
    for n, ac, r, b in groups:
        mu = res["mu"][0, i:i + n]
        out.append({"thr_mbps": float((mu * b * 8).sum()), "access_us": float((1 / mu).mean()),
                    "p": float(res["p"][0, i:i + n].mean())})
        i += n
    return out


def _es(args):
    stations, sim_us, sigma, sifs, arrivals, seed, timeout, warm = args
    return run(stations, sim_us, sigma, sifs, arrivals, seed=seed, timeout_us=timeout, warmup_us=warm)


def _pool(jobs, tasks):
    if jobs <= 1:
        return [_es(t) for t in tasks]
    with ProcessPoolExecutor(jobs) as ex:
        return list(ex.map(_es, tasks))


def rel(a, b):
    return (a - b) / b if b != 0 else float("nan")


# ------------------------------------------------------------------------------------------------ A: Bianchi
def table_a(quick, jobs):
    prm = mf.BIANCHI_FHSS
    ns = (5, 10, 20, 50)
    cases = [("basic", 32, 3), ("rts", 32, 3), ("basic", 128, 3)]
    sim_us = 20e6 if quick else 200e6
    tasks, keys = [], []
    for access, W, m in cases:
        ts, tc, P = mf.bianchi_times(access, prm)
        st = Station(W0=W, Wmax=W * 2 ** m, aifsn=2, cap=1.0, busy_succ=Const(ts - prm["difs"]),
                     busy_coll=Const(tc - prm["difs"]), max_tx=10 ** 9)
        for n in ns:
            tasks.append(([st] * n, sim_us, prm["slot"], prm["sifs"], None, 1000 + n, math.inf, 0.0))
            keys.append((access, W, m, n, ts, tc, P))
    sims = _pool(jobs, tasks)
    rows = []
    for (access, W, m, n, ts, tc, P), r in zip(keys, sims):
        ana = mf.saturation_scalar(n, W, W * 2 ** m, 400, ts, tc, prm["slot"], P)["S"]
        f = torch.float64
        w = torch.ones(1, n, dtype=f)
        res = mf.solve(w, torch.tensor(float(W), dtype=f), torch.tensor(float(W * 2 ** m), dtype=f), 400,
                       torch.full_like(w, ts), torch.full_like(w, tc), prm["slot"],
                       mf.DomainView(torch.ones(1, n, 1, dtype=f)), iters=WifiConfig().fp_iters * 4, damp=0.6)
        s_mf = float(res["mu"].sum()) * P
        s_ev = float(r["n_succ"].sum()) * P / r["sim_us"]
        rows.append({"access": access, "W": W, "m": m, "n": n, "S_bianchi": ana, "S_solver": s_mf, "S_event": s_ev,
                     "err_solver": rel(s_mf, ana), "err_event": rel(s_ev, ana)})
    return rows


# ------------------------------------------------------------------------------------------------ B: 802.11ax saturation
def table_b(quick, jobs):
    wc = WifiConfig(standard="ax", bandwidth_mhz=20)
    rates, _ = mcs_table("ax", 20)
    rate = rates[7]
    ns = (1, 2, 5, 10, 20, 50)
    sizes = [(1500, 0), (8000, 65535), (30000, 65535)]
    sim_us = 2e6 if quick else 10e6
    tasks, keys = [], []
    for B, agg in sizes:
        w = wc.with_(max_ampdu_bytes=agg)
        st = _wifi_station(w, "BE", rate, B=min(B, w.msdu_payload_bytes) if agg == 0 else B)
        for n in ns:
            tasks.append(([st] * n, sim_us, w.slot_us, w.sifs_us, None, 2000 + n, math.inf, 0.0))
            keys.append((B, agg, n, st.cap))
    sims = _pool(jobs, tasks)
    rows = []
    for (B, agg, n, cap), r in zip(keys, sims):
        w = wc.with_(max_ampdu_bytes=agg)
        m = meanfield_saturated(w, [(n, "BE", rate, cap)])[0]
        thr_ev = float(r["throughput_bps"].sum()) / 1e6
        p_ev = float(r["n_coll"].sum() / max(r["n_att"].sum(), 1))
        drop_ev = float(r["dropped_bytes"].sum() / max(r["dropped_bytes"].sum() + r["delivered_bytes"].sum(), 1))
        rows.append({"bytes_per_access": cap, "n": n, "thr_mf": m["thr_mbps"], "thr_event": thr_ev,
                     "err_thr": rel(m["thr_mbps"], thr_ev), "p_mf": m["p"], "p_event": p_ev,
                     "drop_mf": m["p"] ** w.max_tx, "drop_event": drop_ev})
    return rows


# ------------------------------------------------------------------------------------------------ C: EDCA mix
def table_c(quick, jobs):
    wc = WifiConfig(standard="ax", bandwidth_mhz=20, max_ampdu_bytes=0)
    rate = mcs_table("ax", 20)[0][7]
    mixes = [(0, 5), (0, 10), (5, 0), (10, 0), (2, 5), (2, 10), (5, 5), (5, 20)]
    sim_us = 2e6 if quick else 10e6
    tasks = []
    for nvo, nbe in mixes:
        sts = [_wifi_station(wc, "VO", rate)] * nvo + [_wifi_station(wc, "BE", rate)] * nbe
        tasks.append((sts, sim_us, wc.slot_us, wc.sifs_us, None, 3000 + nvo + nbe, math.inf, 0.0))
    sims = _pool(jobs, tasks)
    rows = []
    for (nvo, nbe), r in zip(mixes, sims):
        cap = float(wc.msdu_payload_bytes)
        grp = [g for g in [(nvo, "VO", rate, cap), (nbe, "BE", rate, cap)] if g[0] > 0]
        m = meanfield_saturated(wc, grp)
        if nvo == 0:
            m = [{"thr_mbps": 0.0}] + m
        elif nbe == 0:
            m = m + [{"thr_mbps": 0.0}]
        thr = r["throughput_bps"] / 1e6
        ev_vo, ev_be = float(thr[:nvo].sum()), float(thr[nvo:].sum())
        rows.append({"n_vo": nvo, "n_be": nbe, "vo_mf": m[0]["thr_mbps"], "vo_event": ev_vo,
                     "be_mf": m[1]["thr_mbps"], "be_event": ev_be, "err_vo": rel(m[0]["thr_mbps"], ev_vo),
                     "err_be": rel(m[1]["thr_mbps"], ev_be),
                     "err_total": rel(m[0]["thr_mbps"] + m[1]["thr_mbps"], ev_vo + ev_be)})
    return rows


# ------------------------------------------------------------------------------------------------ D / E: engine vs event sim
def engine_periodic(n, size, steps, E, wc, snr_db=30.0, device="cpu", seed=0):
    """Run level WIFI: every robot submits one message of `size` bytes at every control step. Returns delays (ms)
    of delivered messages and counts."""
    from ..config import NRConfig
    from ..engine import make_engine
    from ..traffic import Requests
    cfg = NRConfig(wifi=wc, msg_sizes=(float(size),), frame_buffer=64, timeout_steps=20)
    net = make_engine("WIFI", E, n, device, cfg, seed=seed)
    d, sent, deliv = [], 0, 0
    warm = 2
    for s in range(steps):
        send = torch.ones(E, n, dtype=torch.long, device=device)
        acc = net.submit(None, Requests(send))
        o = net.step(None, torch.full((E, n), snr_db, device=device))
        cap = o["cap"]
        m = o["delivered"] & (cap >= warm) & (cap < steps - 20)
        d.append((o["delay"][m] * cfg.control_step_ms).cpu())
        if s >= warm and s < steps - 20:
            sent += int(acc.sum())
        deliv += int(m.sum())
    d = torch.cat(d).numpy()
    return d, deliv, sent


def event_periodic(n, size, steps, wc, snr_db=30.0, seed=0, step_ms=100.0):
    rates, thr = mcs_table(wc.standard, wc.bandwidth_mhz, wc.n_ss, wc.gi_us)
    mcs = max(i for i, t in enumerate(thr) if t + wc.ra_margin_db <= snr_db)
    st = _wifi_station(wc, wc.access_category, rates[mcs])
    times = np.arange(steps) * step_ms * 1000.0
    arr = [(times + 0.0, np.full(steps, float(size)))] * n
    return (([st] * n, steps * step_ms * 1000.0, wc.slot_us, wc.sifs_us, arr, seed, 20 * step_ms * 1000.0, 0.0),
            steps, step_ms)


def table_d(quick, jobs, cases=None, wc=None, E=None, steps=None, device="cpu"):
    wc = wc or WifiConfig()
    cases = cases or [(n, s) for s in (4000, 30000) for n in (5, 10, 20, 40)]
    E = E or (16 if quick else 64)
    steps = steps or (40 if quick else 60)
    reps = 2 if quick else 8
    tasks, meta = [], []
    for n, size in cases:
        for rep in range(reps):
            t, st, sm = event_periodic(n, size, steps, wc, seed=4000 + 97 * rep + n)
            tasks.append(t)
            meta.append((n, size))
    sims = _pool(jobs, tasks)
    ev = {}
    for (n, size), r in zip(meta, sims):
        e = ev.setdefault((n, size), {"d": [], "done": 0, "lost": 0})
        arr = np.concatenate(r["msg_arrival_us"]) / 1000.0 / 100.0
        dd = np.concatenate(r["msg_delay_us"]) / 1000.0
        e["d"].append(dd[(arr >= 2) & (arr < steps - 20)])
        e["done"] += int(r["msg_done"].sum())
        e["lost"] += int(r["msg_lost"].sum())
    rows = []
    for n, size in cases:
        t0 = time.time()
        d, deliv, sent = engine_periodic(n, size, steps, E, wc, device=device)
        t_eng = time.time() - t0
        e = ev[(n, size)]
        de = np.concatenate(e["d"])
        load = n * size * 8 / 0.1 / 1e6
        row = {"n": n, "size": size, "offered_mbps": load,
               "mean_engine_ms": float(d.mean()) if len(d) else float("nan"), "mean_event_ms": float(de.mean()),
               "p50_engine_ms": float(np.median(d)) if len(d) else float("nan"), "p50_event_ms": float(np.median(de)),
               "p95_engine_ms": float(np.percentile(d, 95)) if len(d) else float("nan"),
               "p95_event_ms": float(np.percentile(de, 95)),
               "deliv_frac_engine": deliv / max(sent, 1),
               "deliv_frac_event": e["done"] / max(e["done"] + e["lost"], 1), "engine_s": t_eng}
        for k in ("mean", "p50", "p95"):
            row["err_" + k] = rel(row[k + "_engine_ms"], row[k + "_event_ms"])
        rows.append(row)
    return rows


def table_e(quick, jobs):
    rows = []
    for sub in (0.5, 1.0, 2.0, 5.0):
        for noise in ("poisson", "mean"):
            wc = WifiConfig(substep_ms=sub, access_noise=noise)
            for r in table_d(quick, jobs, cases=[(10, 4000), (20, 30000)], wc=wc):
                r.update(substep_ms=sub, access_noise=noise)
                rows.append(r)
    return rows


# ------------------------------------------------------------------------------------------------ F: ns-3 reference
# ns-3.48 src/wifi/examples/wifi-bianchi.cc (built unmodified from a copy of the lab's ns-3.48 tree), 802.11ax,
# HeMcs7, 20 MHz, 800 ns GI, 1500 B packet-socket packets, no A-MPDU (maxMpdus=0), FrameRetryLimit 65535 (as set by
# the example), saturated (pktInterval 100 us), 10 s per point, 1 trial. Aggregate throughput in Mb/s.
# n = 50 runs as the example's ad-hoc ring: its infrastructure run stopped with "Not all stations got traffic!"
# (association of 50 stations); at n = 10 the two modes differ by 0.4 % (33.18 infrastructure, 33.04 ad hoc).
# Runs with A-MPDU (maxMpdus=16) reported 74-88 Mb/s, above the 86 Mb/s PHY rate at n = 50, so the example's
# byte count cannot be taken as aggregated goodput; they are not used.
NS3_WIFI_BIANCHI = {5: (34.8605, "infra"), 10: (33.1789, "infra"), 20: (31.2682, "infra"), 35: (29.3572, "infra"),
                    50: (28.2275, "adhoc")}


def table_f(quick, jobs):
    wc = WifiConfig(msdu_payload_bytes=1500, msdu_overhead_bytes=38, max_ampdu_bytes=0, max_tx=1000)
    rate = mcs_table("ax", 20)[0][7]
    ns = sorted(NS3_WIFI_BIANCHI)
    st = _wifi_station(wc, "BE", rate, B=1500)
    sims = _pool(jobs, [([st] * n, 2e6 if quick else 10e6, wc.slot_us, wc.sifs_us, None, 6000 + n, math.inf, 0.0)
                        for n in ns])
    wa = wc.with_(collision_time="ack")
    rows = []
    for n, r in zip(ns, sims):
        ref, mode = NS3_WIFI_BIANCHI[n]
        m = meanfield_saturated(wc, [(n, "BE", rate, 1500.0)])[0]["thr_mbps"]
        ma = meanfield_saturated(wa, [(n, "BE", rate, 1500.0)])[0]["thr_mbps"]
        ev = float(r["throughput_bps"].sum()) / 1e6
        rows.append({"n": n, "ns3_mode": mode, "thr_ns3": ref, "thr_mf": m, "thr_event": ev, "err_mf": rel(m, ref),
                     "err_event": rel(ev, ref), "thr_mf_ack": ma, "err_mf_ack": rel(ma, ref)})
    return rows


# ------------------------------------------------------------------------------------------------ report
def _md(rows, cols, fmt):
    head = "| " + " | ".join(cols) + " |\n|" + "|".join([":---"] * len(cols)) + "|\n"
    return head + "".join("| " + " | ".join(fmt.get(c, "{}").format(r[c]) for c in cols) + " |\n" for r in rows)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=os.path.join(os.path.expanduser("~"), ".cache", "isaaclab_net", "wifi"))
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--tables", default="ABCDEF")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    torch.set_num_threads(1)
    res = {}
    fns = {"A": table_a, "B": table_b, "C": table_c, "D": table_d, "E": table_e, "F": table_f}
    for k in a.tables:
        t0 = time.time()
        res[k] = fns[k](a.quick, a.jobs) if k != "D" else table_d(a.quick, a.jobs, device=a.device)
        print(f"table {k}: {time.time() - t0:.1f} s", flush=True)
        with open(os.path.join(a.out, "wifi_validation.json"), "w") as f:
            json.dump(res, f, indent=1)
    p = "{:+.1%}"
    g = "{:.3f}"
    fmt = {"S_bianchi": "{:.4f}", "S_solver": "{:.4f}", "S_event": "{:.4f}", "err_solver": "{:+.2%}",
           "err_event": "{:+.2%}", "thr_mf": "{:.2f}", "thr_event": "{:.2f}", "err_thr": p, "access_mf_us": "{:.0f}",
           "access_event_us": "{:.0f}", "err_access": p, "p_mf": g, "p_event": g, "drop_mf": "{:.4f}", "drop_event": "{:.4f}", "bytes_per_access": "{:.0f}",
           "vo_mf": "{:.2f}", "vo_event": "{:.2f}", "be_mf": "{:.2f}", "be_event": "{:.2f}", "err_vo": p,
           "err_be": p, "err_total": p, "offered_mbps": "{:.1f}", "mean_engine_ms": "{:.2f}",
           "mean_event_ms": "{:.2f}", "p50_engine_ms": "{:.2f}", "p50_event_ms": "{:.2f}",
           "p95_engine_ms": "{:.2f}", "p95_event_ms": "{:.2f}", "err_mean": p, "err_p50": p, "err_p95": p,
           "deliv_frac_engine": g, "deliv_frac_event": g, "engine_s": "{:.1f}", "substep_ms": "{}",
           "thr_ns3": "{:.2f}", "err_mf": p, "thr_mf_ack": "{:.2f}", "err_mf_ack": p}
    cols = {"A": ["access", "W", "m", "n", "S_bianchi", "S_solver", "S_event", "err_solver", "err_event"],
            "B": ["bytes_per_access", "n", "thr_mf", "thr_event", "err_thr", "p_mf", "p_event", "drop_mf", "drop_event"],
            "C": ["n_vo", "n_be", "vo_mf", "vo_event", "err_vo", "be_mf", "be_event", "err_be", "err_total"],
            "D": ["n", "size", "offered_mbps", "mean_engine_ms", "mean_event_ms", "err_mean", "p50_engine_ms",
                  "p50_event_ms", "p95_engine_ms", "p95_event_ms", "err_p95", "deliv_frac_engine",
                  "deliv_frac_event"],
            "E": ["substep_ms", "access_noise", "n", "size", "mean_engine_ms", "mean_event_ms", "err_mean",
                  "p95_engine_ms", "p95_event_ms", "err_p95", "engine_s"],
            "F": ["n", "ns3_mode", "thr_ns3", "thr_mf", "err_mf", "thr_event", "err_event", "thr_mf_ack",
                  "err_mf_ack"]}
    with open(os.path.join(a.out, "wifi_validation.md"), "w") as f:
        for k in res:
            f.write(f"## Table {k}\n\n" + _md(res[k], cols[k], fmt) + "\n")
    print(open(os.path.join(a.out, "wifi_validation.md")).read())


if __name__ == "__main__":
    main()
