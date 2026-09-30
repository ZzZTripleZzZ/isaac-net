"""Distribution of per-step worker wall time for one pool worker (spikes vs mean)."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))  # repo root
import torch
from isaaclab_net.examples.fleet_task import FleetEnv, TASK_SIZES
from isaaclab_net.bridges.ns3_pool.poolnet import PoolNet
R, steps, extra = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3:]
torch.manual_seed(0)
net = PoolNet(1, R, "cpu", TASK_SIZES["T1"], extra_args=extra)
env = FleetEnv(1, R, net, "cpu"); env.L = 60.0; net.attach_env(env); env.reset()
for k in range(steps):
    send = (torch.rand(1, R) < 0.3).long()
    env.step(torch.rand(1, R, 2) * 2 - 1, send)
w = torch.tensor(net.pool.timing["worker_max_s"]) * 1e3
q = lambda x: w.quantile(x).item()
big = (w > 100).nonzero().flatten().tolist()
print(f"R={R} {extra} steps={steps}: mean {w.mean():.1f} p50 {q(.5):.1f} p90 {q(.9):.1f} p99 {q(.99):.1f} max {w.max():.1f} ms; steps >100 ms: {len(big)} at {big[:20]}")
net.close()
