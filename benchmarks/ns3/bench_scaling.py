"""Scaling of the parallel ns-3 pool: env-steps/s vs number of workers W and UEs per cell R.

FleetEnv (T1 sizes) + PoolNet, random velocities, each robot sends a small (4 kB) frame with
probability p per step. One JSON line per point in OUT.

usage: python bench_scaling.py OUT.jsonl --W 1 2 4 8 12 --R 8 16 32 64 --p 0.1 0.5 --steps 40
"""
import argparse
import json
import os
import time

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # repo root
import torch
from isaac_net.examples.fleet_task import FleetEnv, TASK_SIZES

from isaac_net.bridges.ns3_pool.poolnet import PoolNet


def cpu_busy():
    """Fraction of the box's CPU time busy over a short window (all processes)."""
    def snap():
        v = list(map(int, open("/proc/stat").readline().split()[1:]))
        return sum(v), v[3] + v[4]
    a, ia = snap(); time.sleep(0.5); b, ib = snap()
    return 1 - (ib - ia) / max(1, b - a)


def run_point(W, R, p, steps, warm, seed=0, extra=()):
    torch.manual_seed(seed)
    net = PoolNet(W, R, "cpu", TASK_SIZES["T1"], extra_args=extra, seed_run=1 + seed * 100)
    env = FleetEnv(W, R, net, "cpu")
    net.attach_env(env)
    env.reset()
    load0, busy0 = os.getloadavg()[0], cpu_busy()
    t_loop, t_pool0 = 0.0, None
    nsent = 0
    for k in range(warm + steps):
        if k == warm:
            t_start = time.perf_counter()
            n_pool0 = len(net.pool.timing["step_s"])
        vel = torch.rand(W, R, 2) * 2 - 1
        send = (torch.rand(W, R) < p).long()
        if k >= warm:
            nsent += int(send.sum())
        env.step(vel, send)
    wall = time.perf_counter() - t_start
    tm = net.pool.timing
    sl = slice(n_pool0, None)
    step_s, wmax, wmean = tm["step_s"][sl], tm["worker_max_s"][sl], tm["worker_mean_s"][sl]
    rss = net.pool.rss_mb()
    res = {
        "W": W, "R": R, "p": p, "steps": steps, "extra": list(extra),
        "offered_bytes_per_cell_step": R * p * TASK_SIZES["T1"][0],
        "env_steps_per_s": W * steps / wall,
        "step_wall_ms": 1e3 * wall / steps,
        "pool_step_ms": 1e3 * sum(step_s) / steps,
        "worker_max_ms": 1e3 * sum(wmax) / steps,
        "worker_mean_ms": 1e3 * sum(wmean) / steps,
        "barrier_wait_ms": 1e3 * (sum(wmax) - sum(wmean)) / steps,
        "protocol_ms": 1e3 * (sum(step_s) - sum(wmax)) / steps,
        "env_python_ms": 1e3 * (wall - sum(step_s)) / steps,
        "startup_s": tm["startup_s"][0],
        "rss_mb_per_worker": sum(rss) / len(rss),
        "frames_sent": nsent,
        "loadavg_1m_before": load0, "loadavg_1m_after": os.getloadavg()[0],
        "cpu_busy_before": busy0, "cpu_busy_after": cpu_busy(),
    }
    net.close()
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--W", type=int, nargs="+", default=[1, 2, 4, 8, 12])
    ap.add_argument("--R", type=int, nargs="+", default=[8, 16, 32, 64])
    ap.add_argument("--p", type=float, nargs="+", default=[0.1, 0.5])
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--warm", type=int, default=5)
    ap.add_argument("--extra", default="", help="extra netslot-bridge args, space separated (e.g. --ueUeFilter=0)")
    a = ap.parse_args()
    torch.set_num_threads(1)
    for R in a.R:
        for p in a.p:
            for W in a.W:
                r = run_point(W, R, p, a.steps, a.warm, extra=a.extra.split())
                print(json.dumps(r), flush=True)
                with open(a.out, "a") as f:
                    f.write(json.dumps(r) + "\n")
