"""nr_replay.py on the fast backends of the NR engine (make_engine("L2", ..., backend="graph" | "triton"), CUDA).

Same inputs, batching, seeds and output files as ../nr_replay.py, so compare.py reads the results unchanged. The
fast backends refuse debug traces, so the per-UE TB counters (tb_tx, tb_fail, tb_exh) are written as NaN; every
per-run metric of compare.py (delay, drop, goodput, first-transmission BLER from rv_tx / rv_fail, PRB use) is
computed as in the CPU replay. The engine seed is 12345 + N (the CPU replay seeds the global RNG with the same value
and draws the engine seed from it, so the two replays use different engine streams: compare them as distributions).

The configuration is the preset named by NRF_PRESET (a function of isaaclab_net.core.config, or "NRConfig" for the
bare defaults) with the k=v overrides of the command line.

usage: NRF_PRESET=<preset> python nr_replay_fast.py <data dir> <out dir> <arm> <reps> <N> <backend> [k=v ...]
"""
import csv
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nr_replay import DRAIN_STEPS, parse_overrides  # noqa: E402

from isaaclab_net.core import config as C  # noqa: E402
from isaaclab_net.core.engine import make_engine  # noqa: E402


def main():
    data, out, arm, reps, N, backend = (sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]),
                                        sys.argv[6])
    ov = parse_overrides(sys.argv[7:])
    runs = [r for r in csv.DictReader(open(os.path.join(data, "lena_runs.csv"))) if int(r["N"]) == N]
    if os.environ.get("NRF_LIMIT"):
        runs = runs[:int(os.environ["NRF_LIMIT"])]
    odir = os.path.join(out, arm)
    os.makedirs(odir, exist_ok=True)
    dev = torch.device("cuda")
    T = int(round(float(runs[0]["T"]) * 10))
    steps = T + DRAIN_STEPS
    E = len(runs) * reps
    snr = torch.zeros(E, N)
    sched = torch.zeros(steps, E, N, dtype=torch.long)
    for k, r in enumerate(runs):
        z = np.load(os.path.join(data, "lena", r["run"] + ".npz"))
        cls = 1 if int(r["S"]) == 4000 else 2
        s = torch.zeros(steps, N, dtype=torch.long)
        s[torch.from_numpy(z["step"].astype(np.int64)), torch.from_numpy(z["ue"].astype(np.int64))] = cls
        for j in range(reps):
            snr[k * reps + j] = torch.from_numpy(z["snr1_db"]).float()
            sched[:, k * reps + j] = s
    preset = os.environ.get("NRF_PRESET", "lena_validation")
    cfg = C.NRConfig(**ov) if preset == "NRConfig" else getattr(C, preset)(**ov)
    cfg = cfg.with_(msg_sizes=(4000.0, 30000.0))
    eng = make_engine("L2", E, N, dev, cfg, backend=backend, seed=12345 + N)
    snr_d = snr.to(dev)
    sched_d = sched.to(dev)
    any_send = (sched > 0).flatten(1).any(1).tolist()
    rec_e, rec_u, rec_d = [], [], []
    prb_win = None
    torch.cuda.synchronize()
    t0 = time.time()
    for t in range(steps):
        if any_send[t]:
            eng.submit(None, sched_d[t], snr_db=snr_d)   # refused frames (PDCP discard, buffer) count as drops
        o = eng.step(None, snr_d)
        dv = o["delivered"]
        e, u, _ = dv.nonzero(as_tuple=True)
        rec_e.append(e)
        rec_u.append(u)
        rec_d.append(o["delay"][dv].float() * cfg.control_step_ms)
        if t == T - 1:
            prb_win = eng.net.ul.prb_used_env.clone()
    torch.cuda.synchronize()
    wall = time.time() - t0
    e = torch.cat(rec_e).cpu().numpy()
    u = torch.cat(rec_u).cpu().numpy()
    dly = torch.cat(rec_d).cpu().numpy()
    sent = (sched > 0).sum(0).numpy()
    rv_tx = eng.net.ul.rv_tx.cpu().numpy()
    rv_fail = eng.net.ul.rv_fail.cpu().numpy()
    prb_win = prb_win.cpu().numpy()
    nan_ue = np.full((E, N), np.nan)
    for k, r in enumerate(runs):
        lo, hi = k * reps, (k + 1) * reps
        m = (e >= lo) & (e < hi)
        np.savez_compressed(
            os.path.join(odir, r["run"] + ".npz"), rep=(e[m] - lo).astype(np.int16), ue=u[m].astype(np.int16),
            delay_ms=dly[m].astype(np.float32), sent=sent[lo:hi], tb_tx=nan_ue[lo:hi], tb_fail=nan_ue[lo:hi],
            tb_exh=nan_ue[lo:hi], rv_tx=rv_tx[lo:hi], rv_fail=rv_fail[lo:hi], prb_win=prb_win[lo:hi],
            prb_win_norm=np.float64(cfg.nprb * cfg.ul_slots_per_step * T))
    gfile = os.path.join(odir, "groups.csv")
    new = not os.path.exists(gfile)
    with open(gfile, "a") as f:
        if new:
            f.write("N,runs,reps,E,steps,wall_s,wall_per_sim_s_per_env,backend,preset,overrides\n")
        f.write(f"{N},{len(runs)},{reps},{E},{steps},{wall:.2f},{wall / (steps * 0.1) / E:.5f},{eng.backend},"
                f"{preset},\"{' '.join(sys.argv[7:])}\"\n")
    print(f"{arm} N={N}: {len(runs)} runs x {reps} reps on {eng.backend} in {wall:.1f} s", flush=True)


if __name__ == "__main__":
    main()
