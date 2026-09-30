"""Closed loop: FleetEnv with the ns-3 bridge as its NetBase, one episode per (variant, E, R).

The policy is the pilot's trained L2 actor (pilot/T1_L2_s0.pt) if present, else random. For context
the same episode (same torch seed) also runs on NetSlot (L2 reference engine). Nothing in env.py or
train.py is edited: Ns3Net binds to the FleetEnv that calls add_frames.
"""
import argparse
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))  # repo root
from isaaclab_net.bridges.ns3_lockstep.lockstep_net import Ns3Net  # noqa: E402
REPO = os.environ["NS3BRIDGE_REPO"]   # training repo with train.py and pilot/ (required)

sys.path.insert(0, REPO)
from isaaclab_net.examples.fleet_task import FleetEnv, TASK_SIZES  # noqa: E402
from isaaclab_net.core.proto.netsim import TIMEOUT, make_net  # noqa: E402
from train import AC  # noqa: E402


def run_episode(net, E, R, ac, dev, seed, task):
    torch.manual_seed(seed)
    net.log_stats = True
    env = FleetEnv(E, R, net, dev)
    net.log_cap_max = env.T - TIMEOUT - 1
    obs = env.reset()
    info, steps, t0 = None, 0, time.time()
    step_times = []
    while info is None:
        ts = time.perf_counter()
        if ac is None:
            vel = torch.rand(E, R, 2, device=dev) * 2 - 1
            send = torch.randint(0, 3, (E, R), device=dev)
        else:
            dn, dc = ac.dist(obs)
            vel, send = dn.mean.clamp(-1, 1), dc.probs.argmax(-1)
        obs, _, done, info = env.step(vel, send)
        step_times.append(time.perf_counter() - ts)
        steps += 1
    st = net.collect()
    dl = st["delay"] * 100.0
    n_del, n_drop = dl.numel(), st["x_cls"].numel()
    out = dict(info)
    out.update(steps=steps, wall_s=time.time() - t0, step_ms_mean=1e3 * sum(step_times) / steps,
               frames_delivered=n_del, frames_timed_out=n_drop,
               delay_p50_ms=dl.quantile(0.5).item() if n_del else None,
               delay_p95_ms=dl.quantile(0.95).item() if n_del else None,
               drop_rate=n_drop / max(1, n_del + n_drop), overflow=st["overflow"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="tcp-procs,unix-procs,shm-procs,tcp-single")
    ap.add_argument("--E", default="1,2,4")
    ap.add_argument("--R", default="8,16")
    ap.add_argument("--policy", default=os.path.join(REPO, "pilot", "T1_L2_s0.pt"))
    ap.add_argument("--task", default="T1")
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--out", default="/home/zzhang66/experiments/bridge_lockstep/results/closed_loop.jsonl")
    a = ap.parse_args()
    dev = "cpu"
    torch.set_num_threads(2)          # shared box: keep the CPU torch reference light
    ac = None
    if os.path.exists(a.policy):
        sd = torch.load(a.policy, map_location=dev)
        sd = sd.get("ac", sd) if isinstance(sd, dict) else sd
        ac = AC()
        ac.load_state_dict(sd)
        ac.eval()
    pol = "trained:" + os.path.basename(a.policy) if ac is not None else "random"
    fo = open(a.out, "a")
    for E in map(int, a.E.split(",")):
        for R in map(int, a.R.split(",")):
            ref = make_net("L2", E, R, dev, TASK_SIZES[a.task])
            with torch.no_grad():
                r = run_episode(ref, E, R, ac, dev, a.seed, a.task)
            r.update(variant="NetSlot-L2-ref", E=E, R=R, policy=pol)
            print(json.dumps(r), flush=True)
            fo.write(json.dumps(r) + "\n")
            for v in a.variants.split(","):
                tr, mode = v.split("-")
                t0 = time.time()
                net = Ns3Net(E, R, dev, TASK_SIZES[a.task], mode=mode, transport=tr, run=1)
                t_start = time.time() - t0
                with torch.no_grad():
                    r = run_episode(net, E, R, ac, dev, a.seed, a.task)
                tm = net.core.timing
                r.update(variant=v, E=E, R=R, policy=pol, startup_s=t_start,
                         rtt_ms_mean=1e3 * sum(tm["rtt"]) / len(tm["rtt"]),
                         ns3_run_ms_mean=1e3 * sum(tm["ns3_run"]) / len(tm["ns3_run"]),
                         ul_sinr_db_last_mean=float(torch.tensor(net.last["sinr_db"]).nanmean()))
                net.close()
                print(json.dumps(r), flush=True)
                fo.write(json.dumps(r) + "\n")
                fo.flush()


if __name__ == "__main__":
    main()
