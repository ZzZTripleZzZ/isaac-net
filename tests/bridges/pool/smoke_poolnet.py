"""Smoke: FleetEnv + PoolNet with a random policy."""
import sys
import time
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))  # repo root
import torch
from isaac_net.examples.fleet_task import FleetEnv, TASK_SIZES
from isaac_net.bridges.ns3_pool.poolnet import PoolNet

E, R, steps = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
torch.manual_seed(0)
net = PoolNet(E, R, "cpu", TASK_SIZES["T1"])
net.log_stats = True
env = FleetEnv(E, R, net, "cpu")
net.attach_env(env)
obs = env.reset()
t0 = time.time()
for k in range(steps):
    vel = torch.rand(E, R, 2) * 2 - 1
    send = torch.multinomial(torch.tensor([0.6, 0.35, 0.05]), E * R, replacement=True).view(E, R)
    obs, rew, done, info = env.step(vel, send)
wall = time.time() - t0
st = net.collect()
d = st["delay"] * 100
print(f"E={E} R={R} steps={steps} wall {wall:.1f}s ({E*steps/wall:.1f} env-steps/s), startup {net.pool.timing['startup_s']}")
print(f"delivered {d.numel()} timeouts {st['x_cls'].numel()} p50 {d.quantile(.5).item() if d.numel() else None} ms p95 {d.quantile(.95).item() if d.numel() else None} ms")
print("queued now", net.queued().float().mean().item(), "rss MB", net.pool.rss_mb())
net.close()
