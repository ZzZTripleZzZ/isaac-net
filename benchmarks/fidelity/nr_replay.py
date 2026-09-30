"""Replay every primary-arm 5G-LENA run in the NR engine (level L2, reference backend) on identical inputs.

Inputs per run come from lena_extract.py: each UE's single-subband full-power SNR (ues.csv snr1_db, the
fixed-SNR input mode) and the exact frame schedule of the LENA run (which UE sent a frame of S bytes at
which 100 ms instant), so the NR engine sees the same link budgets and the same offered traffic, frame by
frame. The configuration is lena_validation() (= lena_match in the validation geometry) plus the
overrides given on the command line, which is how the ablation arms are produced.

All runs with the same N are batched along the env axis (runs x replicas); replicas differ only in the
engine's random draws (TB decoding). The torch seed depends on N only, so every arm uses the same random
stream (common random numbers). Outputs, per run, <out>/<arm>/<run>.npz with
  rep, ue, delay_ms          every delivered frame (on time or late)
  sent [reps, N]             frames offered per UE (identical to LENA's)
  tb_tx, tb_fail, tb_exh     per UE TB transmissions, failed decodes, HARQ exhaustions [reps, N]
  rv_tx, rv_fail [reps, 5]   TBs by transmission number (index 1 = first transmission)
  prb_win [reps]             PRBs granted during the 30 s traffic window (LENA's PRB-use window)
and <out>/<arm>/groups.csv with the wall time of every N group.

usage: python nr_replay.py <data dir from lena_extract> <out dir> <arm> <reps> <N> [k=v ...]
"""
import csv
import os
import sys
import time

import numpy as np
import torch

from isaaclab_net.core.config import lena_validation
from isaaclab_net.core.nr_engine import NRNet

DRAIN_STEPS = 22          # LENA: traffic 0.5..30.5 s, simulation ends at 32.7 s


class TbAcc:
    """Stands in for MacLink.trace: accumulates per-UE TB counts instead of storing every slot."""

    def __init__(self, E, R):
        self.tx = torch.zeros(E, R, dtype=torch.long)
        self.fail = torch.zeros(E, R, dtype=torch.long)
        self.exh = torch.zeros(E, R, dtype=torch.long)

    def append(self, item):
        _, tx, ok, _, _, exh = item
        self.tx += tx.long()
        self.fail += (tx & ~ok).long()
        self.exh += exh.long()


def parse_overrides(args):
    ov = {}
    for kv in args:
        k, v = kv.split("=", 1)
        if v in ("True", "False"):
            ov[k] = v == "True"
        elif v == "None":
            ov[k] = None
        else:
            for cast in (int, float, str):
                try:
                    ov[k] = cast(v)
                    break
                except ValueError:
                    pass
    return ov


def main():
    data, out, arm, reps, N = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
    ov = parse_overrides(sys.argv[6:])
    torch.set_num_threads(int(os.environ.get("NRF_THREADS", "2")))
    runs = [r for r in csv.DictReader(open(os.path.join(data, "lena_runs.csv"))) if int(r["N"]) == N]
    if os.environ.get("NRF_LIMIT"):
        runs = runs[:int(os.environ["NRF_LIMIT"])]
    odir = os.path.join(out, arm)
    os.makedirs(odir, exist_ok=True)
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
    cfg = lena_validation(**ov)
    torch.manual_seed(12345 + N)
    net = NRNet(E, N, "cpu", (4000.0, 30000.0), cfg)
    acc = TbAcc(E, N)
    net.ul.trace = acc
    zb = torch.zeros(E, N, dtype=torch.bool)
    zh = torch.zeros(E, dtype=torch.long)
    rec_e, rec_u, rec_d = [], [], []
    prb_win = None
    t0 = time.time()
    for t in range(steps):
        send = sched[t]
        if bool((send > 0).any()):
            net.add_frames(t, send, zb, zh, snr)     # refused frames (PDCP discard, buffer) count as drops
        o = net.step(t, snr, zh, full=True)
        dv = o["delivered"]
        if bool(dv.any()):
            e, u, _ = dv.nonzero(as_tuple=True)
            rec_e.append(e)
            rec_u.append(u)
            rec_d.append(o["delay"][dv] * cfg.control_step_ms)
        if t == T - 1:
            prb_win = net.ul.prb_used_env.clone()
    wall = time.time() - t0
    e = torch.cat(rec_e).numpy() if rec_e else np.zeros(0, dtype=np.int64)
    u = torch.cat(rec_u).numpy() if rec_u else np.zeros(0, dtype=np.int64)
    dly = torch.cat(rec_d).numpy() if rec_d else np.zeros(0)
    sent = (sched > 0).sum(0).numpy()
    for k, r in enumerate(runs):
        lo, hi = k * reps, (k + 1) * reps
        m = (e >= lo) & (e < hi)
        np.savez_compressed(
            os.path.join(odir, r["run"] + ".npz"), rep=(e[m] - lo).astype(np.int16), ue=u[m].astype(np.int16),
            delay_ms=dly[m].astype(np.float32), sent=sent[lo:hi], tb_tx=acc.tx[lo:hi].numpy(),
            tb_fail=acc.fail[lo:hi].numpy(), tb_exh=acc.exh[lo:hi].numpy(), rv_tx=net.ul.rv_tx[lo:hi].numpy(),
            rv_fail=net.ul.rv_fail[lo:hi].numpy(), prb_win=prb_win[lo:hi].numpy(),
            prb_win_norm=np.float64(cfg.nprb * cfg.ul_slots_per_step * T))
    gfile = os.path.join(odir, "groups.csv")
    new = not os.path.exists(gfile)
    with open(gfile, "a") as f:
        if new:
            f.write("N,runs,reps,E,steps,wall_s,wall_per_sim_s_per_env,threads,overrides\n")
        f.write(f"{N},{len(runs)},{reps},{E},{steps},{wall:.2f},{wall / (steps * 0.1) / E:.5f},"
                f"{torch.get_num_threads()},\"{' '.join(sys.argv[6:])}\"\n")
    print(f"{arm} N={N}: {len(runs)} runs x {reps} reps in {wall:.1f} s", flush=True)


if __name__ == "__main__":
    main()
