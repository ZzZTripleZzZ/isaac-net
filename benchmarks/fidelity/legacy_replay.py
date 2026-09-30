"""Replay the 5G-LENA runs with the legacy slot-level engine (level "L2-legacy", graph backend).

Same inputs as nr_replay.py: each UE's snr1_db (the legacy engine's native input, the full-power SNR over
one 10-PRB subband) and the exact LENA frame schedule. The legacy engine is frozen, so its closest
configuration is its only configuration: 1 HARQ process with head-of-line blocking, OLLA, PHR cap, power
split over the grant, logistic BLER, AR(1) Rayleigh fading that cannot be switched off, a 16-frame FIFO and
the 2 s purge. The only adaptation is the frame size on the air: 4150 / 31100 bytes (LENA's 50 bytes per
1400-byte packet), since the legacy engine carries no header overhead of its own. It has no HARQ or PRB
counters, so the output npz carries delays and offered frames only (has_mac = False).

usage: python legacy_replay.py <data dir> <out dir> <arm> <reps> <device> [backend]
"""
import csv
import os
import sys
import time

import numpy as np
import torch

from isaaclab_net.core.engine import make_engine

DRAIN_STEPS = 22
AIR = (4150.0, 31100.0)


def main():
    data, out, arm, reps, dev = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
    backend = sys.argv[6] if len(sys.argv) > 6 else "graph"
    runs_all = list(csv.DictReader(open(os.path.join(data, "lena_runs.csv"))))
    odir = os.path.join(out, arm)
    os.makedirs(odir, exist_ok=True)
    for N in sorted({int(r["N"]) for r in runs_all}):
        runs = [r for r in runs_all if int(r["N"]) == N]
        T = int(round(float(runs[0]["T"]) * 10))
        steps = T + DRAIN_STEPS
        E = len(runs) * reps
        snr = torch.zeros(E, N)
        sched = torch.zeros(steps, E, N, dtype=torch.long)
        for k, r in enumerate(runs):
            z = np.load(os.path.join(data, "lena", r["run"] + ".npz"))
            s = torch.zeros(steps, N, dtype=torch.long)
            s[torch.from_numpy(z["step"].astype(np.int64)), torch.from_numpy(z["ue"].astype(np.int64))] = \
                1 if int(r["S"]) == 4000 else 2
            for j in range(reps):
                snr[k * reps + j] = torch.from_numpy(z["snr1_db"]).float()
                sched[:, k * reps + j] = s
        torch.manual_seed(12345 + N)
        net = make_engine("L2-legacy", E, N, dev, backend=backend, sizes=AIR, seed=12345 + N)
        snr, sched = snr.to(dev), sched.to(dev)
        rec_e, rec_u, rec_d = [], [], []
        if dev == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for t in range(steps):
            net.submit(t, sched[t], snr)
            o = net.step(t, snr)
            dv = o["delivered"]
            e, u, _ = dv.nonzero(as_tuple=True)
            rec_e.append(e.cpu())
            rec_u.append(u.cpu())
            rec_d.append((o["delay"][dv] * 100.0).cpu())
        if dev == "cuda":
            torch.cuda.synchronize()
        wall = time.time() - t0
        e, u, dly = torch.cat(rec_e).numpy(), torch.cat(rec_u).numpy(), torch.cat(rec_d).numpy()
        sent = (sched > 0).sum(0).cpu().numpy()
        for k, r in enumerate(runs):
            lo, hi = k * reps, (k + 1) * reps
            m = (e >= lo) & (e < hi)
            np.savez_compressed(os.path.join(odir, r["run"] + ".npz"), rep=(e[m] - lo).astype(np.int16),
                                ue=u[m].astype(np.int16), delay_ms=dly[m].astype(np.float32), sent=sent[lo:hi],
                                has_mac=False)
        gfile = os.path.join(odir, "groups.csv")
        new = not os.path.exists(gfile)
        with open(gfile, "a") as f:
            if new:
                f.write("N,runs,reps,E,steps,wall_s,wall_per_sim_s_per_env,device,backend\n")
            f.write(f"{N},{len(runs)},{reps},{E},{steps},{wall:.2f},{wall / (steps * 0.1) / E:.5f},{dev},{backend}\n")
        print(f"{arm} N={N}: {len(runs)} runs x {reps} reps in {wall:.1f} s", flush=True)


if __name__ == "__main__":
    main()
