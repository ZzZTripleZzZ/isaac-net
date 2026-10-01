"""Replay the primary-arm 5G-LENA runs through LoadFixNet (isaac_net/core/nr_loadfix.py) for one arm.
LoadFixNet is now NRNet on the NRConfig switches that the prototype names map to (the shim in nr_loadfix.py); the
engine-integrated v2 preset is replayed by nr_replay.py with NRF_PRESET=lena_validation_v2 (run_v2.sh).

Same inputs, seeds and output format as benchmarks/fidelity/nr_replay.py (per-run .npz that compare.py reads), so
an arm here and the fidelity study's arms are directly comparable: identical per-UE snr1_db, the exact LENA frame
schedule, lena_validation() plus NRConfig overrides, common random numbers across arms (torch seed 12345 + N).
The arm is a named LoadFixConfig switch set (nr_loadfix.ARMS) with optional lf.<field>=<value> overrides.
Extra per-run outputs: tb_bytes, tb_empty, retx_block (grant accounting of the arm).

usage: python loadfix_replay.py <data dir from lena_extract> <out dir> <arm label> <arm> <reps> <N> [k=v | lf.k=v ...]
"""
import os
import sys
import time

import csv
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from nr_replay import DRAIN_STEPS, TbAcc, parse_overrides  # noqa: E402

from isaac_net.core.config import lena_validation  # noqa: E402
from isaac_net.core.nr_loadfix import LoadFixNet, make_arm  # noqa: E402


def main():
    data, out, label, arm, reps, N = (sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]),
                                      int(sys.argv[6]))
    ov_all = parse_overrides(sys.argv[7:])
    lf_ov = {k[3:]: v for k, v in ov_all.items() if k.startswith("lf.")}
    ov = {k: v for k, v in ov_all.items() if not k.startswith("lf.")}
    torch.set_num_threads(int(os.environ.get("NRF_THREADS", "2")))
    runs = [r for r in csv.DictReader(open(os.path.join(data, "lena_runs.csv"))) if int(r["N"]) == N]
    if os.environ.get("NRF_LIMIT"):
        runs = runs[:int(os.environ["NRF_LIMIT"])]
    odir = os.path.join(out, label)
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
    lf = make_arm(arm, **lf_ov)
    torch.manual_seed(12345 + N)
    net = LoadFixNet(E, N, "cpu", (4000.0, 30000.0), cfg, lf=lf)
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
            net.add_frames(t, send, zb, zh, snr)
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
    ctr = {k: float(v) for k, v in net.ul.ctr.items()}
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
            f.write("N,runs,reps,E,steps,wall_s,wall_per_sim_s_per_env,threads,arm,overrides,tb_new,tb_bytes,"
                    "tb_empty,bytes_new,prb_used,retx_block\n")
        f.write(f"{N},{len(runs)},{reps},{E},{steps},{wall:.2f},{wall / (steps * 0.1) / E:.5f},"
                f"{torch.get_num_threads()},{arm},\"{' '.join(sys.argv[7:])}\",{ctr['tb_new']:.0f},"
                f"{ctr.get('tb_bytes', float('nan')):.0f},{ctr.get('tb_empty', float('nan')):.0f},"
                f"{ctr['bytes_new']:.0f},{ctr['prb_used']:.0f},{ctr.get('retx_block', float('nan')):.0f}\n")
    print(f"{label} ({arm}) N={N}: {len(runs)} runs x {reps} reps in {wall:.1f} s", flush=True)


if __name__ == "__main__":
    main()
