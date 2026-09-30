"""Replay 5G-LENA sweep drops in NRNet with identical per-UE link budgets (fixed-SNR input mode).

For every run directory (ues.csv, meta.txt, summary.json, delay_cdf.csv) the per-UE snr1_db
(single-subband full-power SNR incl. shadowing) is fed directly as NRNet's UL SNR input, fading
off, UE power over the whole band, frames of S bytes every 100 ms in phase with probability p for
trafficTime, then a drain. All runs with the same N are batched along the env axis (drop x p x
replica). KPIs follow parse_run.py semantics: a frame is ok if complete within 2 s; drop rate =
1 - ok / frames (late, lost, discarded and unfinished frames all count as drops); delay quantiles
over ok frames; first-tx BLER from rv 0; PRB use over all UL slots.

usage: python -m isaaclab_net.bridges.ns3_offline.lena_replay <sweep_dir> <out.csv> [replicas] [preset overrides as k=v ...]
"""
import csv
import glob
import json
import os
import sys
import time

import numpy as np
import torch

from ...core.config import lena_validation
from ...core.nr_engine import NRNet


def load_run(d):
    meta = dict(l.split(None, 1) for l in open(os.path.join(d, "meta.txt")).read().splitlines() if l.strip())
    ues = list(csv.DictReader(open(os.path.join(d, "ues.csv"))))
    summ = json.load(open(os.path.join(d, "summary.json")))
    cdf = [(float(r["quantile"]), float(r["delay_ms"])) for r in csv.DictReader(open(os.path.join(d, "delay_cdf.csv")))]
    return dict(dir=d, name=os.path.basename(d), N=int(meta["nUe"]), S=int(meta["frameBytes"]), p=float(meta["p"]),
                T=float(meta["trafficTime"]), snr=[float(u["snr1_db"]) for u in ues], summ=summ, cdf=cdf)


def replay_group(runs, reps, dev, overrides, drain=25):
    N = runs[0]["N"]
    E = len(runs) * reps
    cfg = lena_validation(**overrides)
    net = NRNet(E, N, dev, (4000.0, 30000.0), cfg)
    net.log_stats = True
    snr = torch.tensor([r["snr"] for r in runs for _ in range(reps)], device=dev)
    p = torch.tensor([r["p"] for r in runs for _ in range(reps)], device=dev)
    cls = torch.tensor([1 if r["S"] == 4000 else 2 for r in runs for _ in range(reps)], device=dev)
    T = int(round(runs[0]["T"] * 10))
    zb = torch.zeros(E, N, dtype=torch.bool, device=dev)
    z = torch.zeros(E, dtype=torch.long, device=dev)
    sent = torch.zeros(E, dtype=torch.long, device=dev)
    t0 = time.time()
    for t in range(T + drain):
        if t < T:
            send = (torch.rand(E, N, device=dev) < p[:, None]).long() * cls[:, None]
            sent += (send > 0).sum(-1)
            net.add_frames(t, send, zb, z, snr)
        net.step(t, snr, z)
    wall = time.time() - t0
    st = net.collect()
    delay = st["delay"].numpy() * 100.0
    denv = st["d_env"].numpy()
    ul = net.ul
    out = []
    for k, r in enumerate(runs):
        envs = range(k * reps, (k + 1) * reps)
        m = np.isin(denv, list(envs))
        d = delay[m]
        ok = d[d < 2000.0]
        frames = int(sent[k * reps:(k + 1) * reps].sum())
        rv_tx = ul.rv_tx[k * reps:(k + 1) * reps].sum(0).cpu().numpy()
        rv_fail = ul.rv_fail[k * reps:(k + 1) * reps].sum(0).cpu().numpy()
        prb = float(ul.prb_used_env[k * reps:(k + 1) * reps].sum()) / (reps * cfg.nprb * cfg.ul_slots_per_step * (T + drain))
        lq, lx = np.array([c[0] for c in r["cdf"]]), np.array([c[1] for c in r["cdf"]])
        ks = float(np.max(np.abs(np.searchsorted(np.sort(ok), lx, side="right") / max(len(ok), 1) - lq))) if len(ok) else 1.0
        s = r["summ"]
        out.append(dict(
            run=r["name"], N=r["N"], S=r["S"], p=r["p"], reps=reps,
            lena_drop=s["drop_rate"], nr_drop=1 - len(ok) / max(frames, 1),
            lena_p50=s["p50_ms"], nr_p50=float(np.percentile(ok, 50)) if len(ok) else float("nan"),
            lena_p95=s["p95_ms"], nr_p95=float(np.percentile(ok, 95)) if len(ok) else float("nan"),
            lena_bler1=s["ul_phy"]["first_tx_bler"], nr_bler1=float(rv_fail[1] / max(rv_tx[1], 1)),
            lena_retx_frac=s["ul_phy"]["retx_fraction"], nr_retx_frac=float(rv_tx[2:].sum() / max(rv_tx[1:].sum(), 1)),
            lena_prb=s["ul_prb"]["prb_util_all_ul_slots"], nr_prb=prb, ks=ks,
            wall_s_per_sim_s_group=wall / ((T + drain) * 0.1)))
    return out


def main():
    sweep, out_csv = sys.argv[1], sys.argv[2]
    reps = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    ov = {}
    for kv in sys.argv[4:]:
        k, v = kv.split("=")
        for cast in (int, float):
            try:
                v = cast(v)
                break
            except ValueError:
                pass
        ov[k] = {"True": True, "False": False}.get(v, v) if isinstance(v, str) else v
    dev = "cuda" if torch.cuda.is_available() and os.environ.get("NRC_DEVICE") != "cpu" else "cpu"
    runs = [load_run(d) for d in sorted(glob.glob(os.path.join(sweep, "*"))) if os.path.exists(os.path.join(d, "ues.csv"))]
    if os.environ.get("NRC_N"):
        keep = {int(x) for x in os.environ["NRC_N"].split(",")}
        runs = [r for r in runs if r["N"] in keep]
    rows = []
    for N in sorted({r["N"] for r in runs}):
        grp = [r for r in runs if r["N"] == N]
        torch.manual_seed(N)
        res = replay_group(grp, reps, dev, ov)
        rows += res
        print(f"N={N:3d}: {len(grp)} runs x {reps} reps, "
              f"mean |drop diff| {np.mean([abs(x['nr_drop'] - x['lena_drop']) for x in res]):.3f}, "
              f"median KS {np.median([x['ks'] for x in res]):.3f}", flush=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote", out_csv, len(rows), "rows")


if __name__ == "__main__":
    main()
