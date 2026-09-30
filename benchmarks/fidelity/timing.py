"""Wall time per simulated second of the NR engine (level L2, reference backend) at the ns3ref timing points.

Mirrors ns3ref timing.csv: N in {1, ..., 64}, S in {4000, 30000} bytes, the same send probability p per
(N, S), 10 s of traffic plus 2.7 s (12.7 s simulated), lena_validation() at 20 dB per UE. Each point is
run with E environments on the given device; the table reports wall seconds per simulated second for the
whole batch and per environment. The reference backend is the readable eager engine, not a fast backend.

usage: python timing.py <ns3ref timing.csv> <out.csv> <device> <E> [threads]
"""
import csv
import os
import sys
import time

import torch

from isaaclab_net.core.config import lena_validation
from isaaclab_net.core.nr_engine import NRNet


def main():
    src, out, dev, E = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    if len(sys.argv) > 5:
        torch.set_num_threads(int(sys.argv[5]))
    pts = sorted({(int(r["nUe"]), int(r["frameBytes"]), float(r["p"])) for r in csv.DictReader(open(src))})
    new = not os.path.exists(out)
    f = open(out, "a")
    if new:
        f.write("device,E,threads,N,S,p,sim_s,wall_s,wall_per_sim_s,wall_per_sim_s_per_env,loadavg1\n")
    for N, S, p in pts:
        torch.manual_seed(N)
        net = NRNet(E, N, dev, (4000.0, 30000.0), lena_validation())
        snr = torch.full((E, N), 20.0, device=dev)
        zb = torch.zeros(E, N, dtype=torch.bool, device=dev)
        zh = torch.zeros(E, dtype=torch.long, device=dev)
        cls = 1 if S == 4000 else 2
        steps = 127
        if dev == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for t in range(steps):
            if t < 100:
                send = (torch.rand(E, N, device=dev) < p).long() * cls
                net.add_frames(t, send, zb, zh, snr)
            net.step(t, snr, zh)
        if dev == "cuda":
            torch.cuda.synchronize()
        wall = time.time() - t0
        sim = steps * 0.1
        f.write(f"{dev},{E},{torch.get_num_threads()},{N},{S},{p},{sim:.1f},{wall:.3f},{wall / sim:.4f},"
                f"{wall / sim / E:.5f},{os.getloadavg()[0]:.2f}\n")
        f.flush()
        print(dev, E, N, S, f"{wall / sim:.3f} s/s", flush=True)


if __name__ == "__main__":
    main()
