"""Edge-offloaded control: robots stream their state to an edge server that returns velocity commands.

Every control step each robot sends a state frame (class 1). The edge controller computes a command from the
state the frame captured, v = gain * (target(cap) - pos(cap)), and returns it to the robot through the edge loop
(EdgeLoop: edge queue and service time, then the return path). The robot applies the newest command it holds.
When that command is older than `max_age` steps it is stale, and the robot either keeps applying it
(stale="hold") or stops (stale="zero"). The reward is the negative tracking error to a target on a circle.

    python -m isaac_net.examples.edge_control --level L1 --return_path delay --stale hold
    python -m isaac_net.examples.edge_control --level L2 --return_path nr_dl --device cpu --envs 8 --steps 60
                                                             # closes the loop through the NR downlink scheduler

The policy side of an RL task would observe out["act_age"] (and out["edge_queue_len"]) next to its own state.
"""
from __future__ import annotations

import argparse
import math

import torch

from ..core import EdgeConfig, NRConfig, Requests, make_engine, netslot_compat

H = 64            # history of captured states (steps); longer than any action age we act on


class EdgeControlTask:
    L = 100.0             # arena (m)
    VMAX = 2.0            # m per control step
    GAIN = 0.5

    def __init__(self, net, stale="hold", max_age=5):
        assert stale in ("hold", "zero")
        self.net, self.stale, self.max_age = net, stale, max_age
        self.E, self.R, self.dev = net.E, net.R, net.dev
        E, R, d = self.E, self.R, self.dev
        self.phase = torch.rand(E, R, device=d) * 2 * math.pi
        self.pos = torch.rand(E, R, 2, device=d) * self.L
        self.cmd = torch.zeros(E, R, 2, device=d)
        self.hist_pos = torch.zeros(E, R, H, 2, device=d)
        self.hist_tgt = torch.zeros(E, R, H, 2, device=d)

    def target(self, t):
        a = self.phase + 0.05 * t[:, None].float()
        return self.L / 2 + 30.0 * torch.stack([torch.cos(a), torch.sin(a)], -1)

    def reset(self, env_ids):
        self.net.reset(env_ids)
        self.pos[env_ids] = torch.rand(len(env_ids), self.R, 2, device=self.dev) * self.L
        self.cmd[env_ids] = 0.0

    def step(self):
        t = self.net.clock
        slot = (t % H)[:, None].expand(-1, self.R)
        e = torch.arange(self.E, device=self.dev)[:, None]
        r = torch.arange(self.R, device=self.dev)[None, :]
        self.hist_pos[e, r, slot] = self.pos                       # the state the frame captures
        self.hist_tgt[e, r, slot] = self.target(t)
        self.net.submit(None, Requests(torch.ones(self.E, self.R, dtype=torch.long, device=self.dev)))
        out = self.net.step(None, self.pos)
        # a new command: the edge's output for the captured state of act_cap
        new = out["act_new"]
        cs = (out["act_cap"].clamp(min=0) % H)
        v = self.GAIN * (self.hist_tgt[e, r, cs] - self.hist_pos[e, r, cs])
        self.cmd = torch.where(new[..., None], v, self.cmd)
        stale = out["act_age"].nan_to_num(float("inf")) > self.max_age      # no action yet counts as stale
        apply = self.cmd if self.stale == "hold" else torch.where(stale[..., None], 0.0, self.cmd)
        n = apply.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        self.pos = (self.pos + apply * (self.VMAX / n).clamp(max=1.0)).clamp(0, self.L)
        err = (self.pos - self.target(out["t"] + 1)).norm(dim=-1)
        return -err, out


def run(level="L1", return_path="delay", stale="hold", E=64, R=8, steps=200, device="cpu", service_ms=20.0,
        max_age=5, seed=0):
    torch.manual_seed(seed)
    edge = EdgeConfig(service_ms=service_ms, servers_per_env=2, queue_cap=2 * R, deadline_ms=300.0,
                      return_path=return_path, ret_fixed_ms=2.0, cmd_bytes=200)
    base = netslot_compat(dl=True) if level == "L2" else NRConfig()
    net = make_engine(level, E, R, device, base.with_(edge=edge), seed=seed)
    task = EdgeControlTask(net, stale=stale, max_age=max_age)
    rew, age, lat, stale_frac = [], [], [], []
    for k in range(steps):
        rwd, out = task.step()
        if k >= steps // 4:
            rew.append(rwd.mean().item())
            age.append(out["act_age"].nanmean().item())
            stale_frac.append((out["act_age"].nan_to_num(float("inf")) > max_age).float().mean().item())
            lat.append(out["act_latency"][out["act_new"]].float())
        done = torch.nonzero(torch.rand(E, device=device) < 0.01).squeeze(-1)
        if done.numel():
            task.reset(done)
    lat = torch.cat(lat) if lat else torch.zeros(0)
    c = net.counters()
    res = {"reward": sum(rew) / len(rew), "act_age": sum(age) / len(age), "stale_frac": sum(stale_frac) / len(age),
           "loop_ms_p50": float(lat.quantile(0.5)) * 100 if lat.numel() else math.nan,
           "loop_ms_p95": float(lat.quantile(0.95)) * 100 if lat.numel() else math.nan,
           "edge_drops": int((c["dropped_full"] + c["dropped_deadline"]).sum()), "completed": int(c["completed"].sum())}
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--level", default="L1")
    ap.add_argument("--return_path", default="delay", choices=["instant", "delay", "nr_dl"])
    ap.add_argument("--stale", default="both", choices=["hold", "zero", "both"])
    ap.add_argument("--envs", type=int, default=64)
    ap.add_argument("--robots", type=int, default=8)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--service_ms", type=float, default=20.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--threads", type=int, default=1, help="CPU threads (the tensors are small; more threads contend)")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    for stale in (("hold", "zero") if a.stale == "both" else (a.stale,)):
        r = run(a.level, a.return_path, stale, a.envs, a.robots, a.steps, a.device, a.service_ms)
        print(f"{a.level} {a.return_path} stale={stale}: " + ", ".join(
            f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items()))


if __name__ == "__main__":
    main()
