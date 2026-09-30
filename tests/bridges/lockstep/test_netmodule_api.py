"""Smoke test of the NetModule-style API (Ns3NetModule): per-env clocks, partial reset, blockage,
end-of-step poses with in-step interpolation, NetOutput fields."""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))  # repo root
from isaaclab_net.bridges.ns3_lockstep.netmodule_ns3 import NetConfig, Ns3NetModule, TrafficRequest  # noqa: E402


def main():
    torch.manual_seed(0)
    E, R = 2, 8
    out = {}
    for tr, mode in (("tcp", "procs"), ("unix", "single")):
        cfg = NetConfig(num_envs=E, num_robots=R, device="cpu", msg_sizes=(4000.0, 30000.0))
        net = Ns3NetModule(cfg, transport=tr, mode=mode)
        ang = torch.rand(E, R) * 1.4 + 0.05
        d = 15 + 30 * torch.rand(E, R)
        pos = torch.stack([d * torch.cos(ang), d * torch.sin(ang), torch.full((E, R), 1.5)], -1)
        net.reset()
        n_del, delays, t_env = 0, [], []
        for k in range(80):
            if k == 40:
                net.reset(torch.tensor([True, False]))      # env 0 only
            pos[..., :2] += 0.3 * torch.randn(E, R, 2)
            send = (torch.rand(E, R) < 0.4).long()
            blocked = torch.zeros(E, R, 1, dtype=torch.bool)
            blocked[:, :2] = True                            # two robots per env behind a wall
            o = net.step(pos, TrafficRequest(send), blocked=blocked)
            n_del += int(o.delivered.sum())
            dd = o.delay_s[torch.isfinite(o.delay_s)]
            delays += dd.tolist()
            t_env.append(net.t.tolist())
        out[f"{tr}-{mode}"] = {
            "robot_steps_with_delivery": n_del,
            "delay_p50_ms": float(np.median(delays) * 1e3) if delays else None,
            "env_clock_after": t_env[-1], "env_clock_at_39": t_env[39],
            "sinr_blocked_vs_clear_db": [float(o.sinr_db[:, :2].nanmean()), float(o.sinr_db[:, 2:].nanmean())],
            "fields": sorted(k for k in vars(o) if not k.startswith("_")),
        }
        net.close()
    print(json.dumps(out))


if __name__ == "__main__":
    main()
