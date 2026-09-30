"""Wall time per 100 ms control step vs E and R, split into ns-3 simulation time and bridge overhead.

Robots random-walk (0.3 m/step) in 20-80 m of the gNB; each robot sends a frame with p = 0.3 per step
(70% 4 kB, 30% 30 kB). 5 warm-up steps, then --steps timed steps.
rtt      = Python send-all .. receive-all (the whole lockstep call as the robot simulator sees it)
ns3_max  = max over processes of the time inside Simulator::Run() for this step
overhead = rtt - ns3_max (serialization, transport, ns-3 input/output handling, Python parsing)
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))  # repo root
from isaaclab_net.bridges.ns3_lockstep import protocol as P  # noqa: E402
from isaaclab_net.bridges.ns3_lockstep.core import Ns3Lockstep  # noqa: E402


def bench(E, R, transport, mode, steps, seed=0, epp=None):
    rng = np.random.default_rng(seed)
    t_start = time.perf_counter()
    br = Ns3Lockstep(E, R, mode=mode, transport=transport, run=1, envs_per_proc=epp)
    startup = time.perf_counter() - t_start
    ang = rng.uniform(0.05, 1.5, (E, R))
    d = rng.uniform(20, 80, (E, R))
    pos = np.stack([d * np.cos(ang), d * np.sin(ang), np.full((E, R), np.nan)], -1).astype(np.float32)
    t_reset = time.perf_counter()
    br.reset(pos=pos)
    reset_s = time.perf_counter() - t_reset
    fid = np.zeros((E, R), np.int64)
    rec = []
    done = 0
    for k in range(steps + 5):
        pos[..., :2] += rng.normal(0, 0.3, (E, R, 2)).astype(np.float32)
        send = rng.random((E, R)) < 0.3
        e, r = np.nonzero(send)
        fr = np.zeros(len(e), P.FRAME_IN)
        fr["env"], fr["ue"], fr["fid"] = e, r, fid[e, r]
        fr["bytes"] = np.where(rng.random(len(e)) < 0.7, 4000, 30000)
        fid[e, r] += 1
        res = br.step(pos, fr)
        done += len(res["done"])
        if k >= 5:
            rec.append(res["timing"])
    br.close()
    rtt = np.array([x["rtt"] for x in rec]) * 1e3
    run = np.array([x["ns3_run_max"] for x in rec]) * 1e3
    run_sum = np.array([x["ns3_run_sum"] for x in rec]) * 1e3
    other = np.array([x["ns3_other_max"] for x in rec]) * 1e3
    return {"E": E, "R": R, "transport": transport, "mode": mode, "procs": br.G, "steps": steps,
            "rtt_ms_mean": float(rtt.mean()), "rtt_ms_p50": float(np.median(rtt)),
            "ns3_run_ms_mean": float(run.mean()), "ns3_run_sum_ms_mean": float(run_sum.mean()),
            "ns3_io_ms_mean": float(other.mean()),
            "overhead_ms_mean": float((rtt - run).mean()), "overhead_ms_p50": float(np.median(rtt - run)),
            "ms_per_env_step": float(rtt.mean() / E), "frames_done": done,
            "startup_s": startup, "reset_s": reset_s}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="tcp-procs,unix-procs,shm-procs,tcp-single,unix-single")
    ap.add_argument("--E", default="1,2,4,8")
    ap.add_argument("--R", default="8,16")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--out", default="/home/zzhang66/experiments/bridge_lockstep/results/bench.jsonl")
    a = ap.parse_args()
    fo = open(a.out, "a")
    for cfg in a.configs.split(","):
        tr, mode = cfg.split("-")
        for E in map(int, a.E.split(",")):
            for R in map(int, a.R.split(",")):
                r = bench(E, R, tr, mode, a.steps)
                print(json.dumps(r), flush=True)
                fo.write(json.dumps(r) + "\n")
                fo.flush()


if __name__ == "__main__":
    main()
