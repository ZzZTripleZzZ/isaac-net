"""Real-time emulation mode: how many robots can one ns-3 cell serve while holding wall-clock time?

A client (standing in for the Isaac GUI demo) streams frames in wall-clock time at 10 Hz per robot
(random policy A mix: small 4 kB with prob 0.25, large 30 kB with prob 0.05, plus a position
update every step) to netslot-bridge --io=rt:PORT (RealtimeSimulatorImpl, BestEffort), and reads
delivery reports as they happen. The worker samples lag = wall - sim every 10 ms of sim time.

usage: python rt_bench.py OUT.jsonl --R 1 2 4 8 12 16 24 32 --dur 20
"""
import argparse
import json
import math
import os
import random
import socket
import subprocess
import threading
import time

import sys  # noqa: E402
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # repo root
from isaac_net.bridges.ns3_pool.ns3pool import BIN, DEFAULT_ARGS  # noqa: E402


def run(R, dur, port, seed=0, p_small=0.25, p_large=0.05, hold_ms=20.0, L=60.0, move=True, lag_file=None):
    rng = random.Random(seed)
    pos = [(rng.uniform(0, L), rng.uniform(0, L)) for _ in range(R)]
    init = ",".join(f"{x:.3f}:{y:.3f}:-1" for x, y in pos)
    proc = subprocess.Popen([BIN, f"--nUe={R}", f"--io=rt:{port}", f"--init={init}", f"--run={seed + 1}",
                             "--outDir=/tmp"] + DEFAULT_ARGS + ([f"--rtLagFile={lag_file}"] if lag_file else []), stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    for _ in range(600):
        try:
            s = socket.create_connection(("127.0.0.1", port))
            break
        except OSError:
            time.sleep(0.02)
    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    f = s.makefile("rwb", buffering=0)
    assert f.readline().startswith(b"H")
    t0 = time.monotonic()                      # ~ worker's wall 0 (Run starts right after H)
    # lag stats cover sim >= 1.5 s (warm-up = attach + first channel generation); traffic starts at 0.6 s
    got, lag_line = [], [None]

    def reader():
        for ln in f:
            p = ln.split()
            if p[0] == b"D":
                got.append((int(p[1]), int(p[2]), float(p[3]), float(p[4]), time.monotonic() - t0))
            elif p[0] == b"L":
                lag_line[0] = [float(x) for x in p[1:]]
    th = threading.Thread(target=reader, daemon=True)
    th.start()
    sent = {}
    late_sends = 0
    n_steps = int(dur * 10)
    for k in range(n_steps):
        target = 0.6 + 0.1 * k                   # after attach (sim 0.5 s)
        now = time.monotonic() - t0
        if now < target:
            time.sleep(target - now)
        elif now - target > 0.01:
            late_sends += 1
        lines = []
        for r in range(R):
            x, y = pos[r]
            x = min(L, max(0, x + rng.uniform(-0.3, 0.3)))
            y = min(L, max(0, y + rng.uniform(-0.3, 0.3)))
            pos[r] = (x, y)
            if move:
                lines.append(f"P {r} {x:.3f} {y:.3f} -1\n")
            u = rng.random()
            if u < p_small + p_large:
                b = 4000 if u < p_small else 30000
                lines.append(f"F {r} {k} {b}\n")
                sent[(r, k)] = time.monotonic() - t0
        f.write("".join(lines).encode())
    time.sleep(2.0)                              # drain
    f.write(b"Q\n")
    th.join(timeout=60)
    proc.wait(timeout=60)
    n, p50, p99, mx, final, over10, warm = lag_line[0] or [0] + [math.nan] * 6
    sim_d = sorted(1e3 * (last - gen) for _, _, gen, last, _ in got)
    wall_d = sorted(1e3 * (w - sent[(u, fid)]) for u, fid, _, _, w in got if (u, fid) in sent)
    q = lambda v, x: v[min(len(v) - 1, int(x * len(v)))] if v else None
    return {"R": R, "dur_s": dur, "L": L, "move": move, "frames_sent": len(sent), "frames_delivered": len(got),
            "lag_samples": n, "lag_p50_ms": p50, "lag_p99_ms": p99, "lag_max_ms": mx, "lag_final_ms": final,
            "lag_frac_over_10ms": over10, "warmup_lag_max_ms": warm,
            "holds_realtime": bool(p99 < hold_ms and final < hold_ms),
            "sim_delay_p50_ms": q(sim_d, .5), "wall_delay_p50_ms": q(wall_d, .5),
            "wall_delay_p95_ms": q(wall_d, .95), "client_late_sends": late_sends,
            "loadavg_1m": os.getloadavg()[0]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--R", type=int, nargs="+", default=[1, 2, 4, 8, 12, 16, 24, 32])
    ap.add_argument("--dur", type=float, default=20.0)
    ap.add_argument("--L", type=float, default=60.0, help="robots in an L x L square, gNB at the corner")
    ap.add_argument("--static", action="store_true", help="no position updates")
    ap.add_argument("--lagdir", default=None, help="write each run's lag series here")
    a = ap.parse_args()
    for R in a.R:
        lf = f"{a.lagdir}/lag_R{R}.csv" if a.lagdir else None
        r = run(R, a.dur, 45000 + R, L=a.L, move=not a.static, lag_file=lf)
        print(json.dumps(r), flush=True)
        with open(a.out, "a") as fo:
            fo.write(json.dumps(r) + "\n")
