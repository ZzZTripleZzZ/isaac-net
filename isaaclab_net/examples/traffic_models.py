"""Traffic models next to policy messages on the NR engine (level L2).

    python -m isaaclab_net.examples.traffic_models [--device cuda] [--envs 64] [--steps 50]

A fleet of R = 6 robots per env: robots 0-3 stream 200-byte telemetry every 10 ms (ten messages per 100 ms control
step), robot 4 streams 25 fps video with a 15-frame GOP, robot 5 is a bursty Markov on/off source, and every robot
sends a 4 kB alarm report in the step its alarm fires. The policy still submits its own 4 kB messages through
submit(). Per tag, the script prints the messages accepted, delivered and their median / p95 delay, measured from
each message's arrival slot.
"""
from __future__ import annotations

import argparse

import torch

from isaaclab_net.core import NRConfig, Requests, make_engine
from isaaclab_net.core.traffic import TrafficModel as TM

TAGS = {0: "policy", 1: "telemetry", 2: "video", 3: "bursty", 4: "alarm"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--envs", type=int, default=16)
    ap.add_argument("--steps", type=int, default=50)
    a = ap.parse_args()
    E, R = a.envs, 6
    cfg = NRConfig(
        bandwidth_mhz=40, frame_buffer=64,
        traffic=[
            TM.periodic(200, period_ms=10, jitter_ms=1).on(range(4)),               # tag 1
            TM.video(fps=25, mean_frame_bytes=6_000, gop=(30_000, 4_000, 15)).on(4),  # tag 2
            TM.bursty(1_400, rate_hz=40, burst_size=4, on_off=(0.5, 1.0)).on(5),   # tag 3
            TM.event(4_000, trigger="alarm", det=True, deadline_ms=50),             # tag 4
            TM.policy(),                                                            # tag 0, submit()
        ])
    net = make_engine("L2", E, R, a.device, cfg, seed=0)
    torch.manual_seed(0)
    snr = 5 + 20 * torch.rand(E, R, device=a.device)
    delays = {t: [] for t in TAGS}
    for k in range(a.steps):
        send = (torch.rand(E, R, device=a.device) < 0.1).long()          # the policy's own messages
        net.submit(None, Requests(send))
        alarm = torch.rand(E, R, device=a.device) < 0.02
        out = net.step(None, snr, triggers={"alarm": alarm})
        if k == a.steps // 2:
            net.reset(torch.arange(0, E, 2, device=a.device))            # partial reset of every other env
        d = out["delivered"]
        for t in TAGS:
            delays[t].append(out["delay"][d & (out["tag"] == t)].float() * cfg.control_step_ms)
    st = {k: int(v) for k, v in net.traffic_stats.items()}
    print(f"generated {st['generated']} messages ({st['generated_bytes']} B), accepted {st['accepted']}, "
          f"refused {st['refused']}, deferred {int(net.traffic.deferred)}")
    for t, name in TAGS.items():
        x = torch.cat(delays[t])
        if x.numel():
            q = torch.quantile(x.cpu(), torch.tensor([0.5, 0.95]))
            print(f"  {name:9s} delivered {x.numel():6d}   delay p50 {q[0]:6.2f} ms   p95 {q[1]:6.2f} ms")


if __name__ == "__main__":
    main()
