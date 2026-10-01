"""ShardedEngine: equality with the unsharded engine and throughput on one or several GPUs.

    python benchmarks/bench_sharded.py --level L2-legacy --backend graph --E 8192 --R 16 --devices cuda:0,cuda:1

Runs the same random inputs (with partial resets) through make_engine on devices[0] and through ShardedEngine over
--devices, checks that every output is bitwise equal (shard-invariant levels; the check is skipped for the others),
then times --steps steps of each and prints one JSON line. Two shards on one GPU (--devices cuda:0,cuda:0) check
correctness only; the speed-up needs distinct GPUs.
"""
from __future__ import annotations

import argparse
import json
import time

import torch

from isaac_net.core import NRConfig, Requests, make_engine
from isaac_net.core.sharded import ShardedEngine


def drive(net, E, R, dev, steps, seed=0, resets=True):
    g = torch.Generator().manual_seed(seed)
    outs = []
    for k in range(steps):
        send = torch.randint(0, 3, (E, R), generator=g).to(dev)
        pos = (torch.rand(E, R, 2, generator=g) * 100).to(dev)
        net.submit(None, Requests(send))
        outs.append(net.step(None, pos))
        if resets and k % 7 == 3:
            net.reset(torch.randint(0, E, (max(1, E // 16),), generator=g).to(dev))
    return outs


def equal(a, b):
    for x, y in zip(a, b):
        for k in y:
            u, v = x[k], y[k]
            if u.is_floating_point():
                u, v = u.nan_to_num(-7.0), v.nan_to_num(-7.0)
            if not torch.equal(u.cpu(), v.cpu()):
                return False
    return True


def timed(net, E, R, dev, steps, devices):
    g = torch.Generator().manual_seed(1)
    send = torch.randint(0, 3, (E, R), generator=g).to(dev)
    pos = (torch.rand(E, R, 2, generator=g) * 100).to(dev)
    for _ in range(3):
        net.submit(None, Requests(send))
        net.step(None, pos)
    for d in devices:
        torch.cuda.synchronize(d)
    t0 = time.perf_counter()
    for _ in range(steps):
        net.submit(None, Requests(send))
        net.step(None, pos)
    for d in devices:
        torch.cuda.synchronize(d)
    return (time.perf_counter() - t0) / steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", default="L2-legacy")
    ap.add_argument("--backend", default="graph")
    ap.add_argument("--E", type=int, default=4096)
    ap.add_argument("--R", type=int, default=16)
    ap.add_argument("--devices", default="cuda:0,cuda:1")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--check_steps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    devs = [torch.device(d) for d in a.devices.split(",")]
    cfg = NRConfig()
    E, R = a.E, a.R
    un = make_engine(a.level, E, R, devs[0], cfg, backend=a.backend, seed=a.seed)
    sh = ShardedEngine(a.level, E, R, devs, cfg, backend=a.backend, seed=a.seed)
    same = None
    if sh.shard_invariant:
        same = equal(drive(sh, E, R, devs[0], a.check_steps), drive(un, E, R, devs[0], a.check_steps))
    t_un = timed(un, E, R, devs[0], a.steps, devs[:1])
    t_sh = timed(sh, E, R, devs[0], a.steps, sorted(set(devs), key=str))
    print(json.dumps({"level": a.level, "backend": a.backend, "E": E, "R": R, "devices": a.devices,
                      "gpus": [torch.cuda.get_device_name(d) for d in sorted(set(devs), key=str)],
                      "shard_invariant": sh.shard_invariant, "bitwise_equal": same,
                      "ms_per_step_unsharded": 1e3 * t_un, "ms_per_step_sharded": 1e3 * t_sh,
                      "speedup": t_un / t_sh}))


if __name__ == "__main__":
    main()
