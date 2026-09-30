"""Policies, a rollout runner that records traces and per-frame outcomes, and summary metrics."""
from __future__ import annotations

import json

import torch

from ...core.proto.netsim import TIMEOUT, F
from ...examples.fleet_task import FleetEnv


# ---- policies (obs layout from env.FleetEnv._obs) --------------------------------------------
def policy_random(env, obs, gen):
    """Open-loop: random velocity, send none / small / large with prob 0.7 / 0.25 / 0.05."""
    E, R = env.E, env.R
    vel = torch.rand(E, R, 2, generator=gen) * 2 - 1
    send = torch.multinomial(torch.tensor([0.7, 0.25, 0.05]), E * R, replacement=True, generator=gen).view(E, R)
    return vel, send


def policy_greedy(env, obs, gen):
    """Closed-loop: head to the goal but flee a known hazard (within its radius + 5 m); send a
    frame whenever the own queue is empty, a large one while the siren is on, else a small one."""
    L = env.L
    g = obs[..., 2:4]
    vel = g / g.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    hrel, hr, known = obs[..., 4:6] * L, obs[..., 6] * L, obs[..., 11] > 0
    near = known & (hrel.norm(dim=-1) < hr + 5.0)
    away = -hrel / hrel.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    vel = torch.where(near[..., None], away, vel)
    empty = obs[..., 8] == 0
    siren = obs[..., 7] > 0
    send = torch.where(empty, torch.where(siren, 2, 1), 0).long()
    return vel, send


POLICIES = {"random": policy_random, "greedy": policy_greedy}


class FrameLog:
    """Wrap a NetBase so every frame's fate (delivered at fin / timed out) is recorded."""

    def __init__(self, net):
        self.net, self.rows = net, []
        orig = net._transmit

        def tx(t, snr_db):
            fin = orig(t, snr_db)
            cap, cls = net.cap, net.cls
            dl = (cap >= 0) & torch.isfinite(fin)
            tt = t[:, None, None] if torch.is_tensor(t) else t        # per-env clock [E]
            to = (cap >= 0) & ~dl & ((tt + 1 - cap) >= TIMEOUT)
            for m, kind in ((dl, 1), (to, 0)):
                e, r, i = m.nonzero(as_tuple=True)
                for ee, rr, ii in zip(e.tolist(), r.tolist(), i.tolist()):
                    c = int(cap[ee, rr, ii])
                    d = float(fin[ee, rr, ii]) - c if kind else float("inf")
                    self.rows.append((ee, rr, c, int(cls[ee, rr, ii]), d))
            return fin
        net._transmit = tx


def rollout(policy_name, net, E, R, seed, T=300, L=150.0):
    """One episode of E envs. Returns trace (inputs a trace-driven ns-3 run needs), frames, metrics."""
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed + 12345)
    pol = POLICIES[policy_name]
    env = FleetEnv(E, R, net, "cpu")
    env.L = L                         # arena side (gNB at the corner); 150 m is the example-task default
    if hasattr(net, "attach_env"):
        net.attach_env(env)
    log = FrameLog(net)
    obs = env.reset()
    trace = {"pos": [], "snr": [], "send": [], "policy_send": []}
    aoi, info = [], None
    for t in range(T):
        vel, send = pol(env, obs, gen)
        trace["pos"].append(env.pos.clone())
        trace["snr"].append(env.radio.snr_db(env.pos))
        # the network only sees frames that fit in the robot's buffer (NetBase drops overflow)
        trace["send"].append(torch.where(net.queued() < F, send, torch.zeros_like(send)))
        trace["policy_send"].append(send.clone())
        obs, rew, done, info = env.step(vel, send)
        if not done:
            aoi.append((env.t - env.last_cap).clamp(max=TIMEOUT * 5).float().mean().item())
    tr = {k: torch.stack(v) for k, v in trace.items()}     # [T, E, R, ...]
    sent = tr["policy_send"] > 0
    return {"trace": tr, "frames": log.rows, "aoi": aoi, "info": info,
            "sizes": list(net.sizes.tolist()), "E": E, "R": R, "T": T, "seed": seed, "L": L,
            "sends_per_robot_step": sent.float().mean().item(),
            "large_share": ((tr["policy_send"] == 2).sum() / sent.sum().clamp(min=1)).item(),
            "overflow_share": 1 - (tr["send"] > 0).sum().item() / max(1, sent.sum().item())}


def frame_metrics(frames, T, sent_mask=None):
    """Delay/drop metrics over frames captured at or before T - TIMEOUT - 1 (no censoring)."""
    keep = [f for f in frames if f[2] <= T - TIMEOUT - 1]
    d = torch.tensor([f[4] for f in keep if f[4] != float("inf")]) * 100.0      # ms
    n = len(keep)
    out = {"frames": n, "delivered": int(d.numel()), "delivery_rate": d.numel() / max(1, n)}
    for q in (0.5, 0.9, 0.95):
        out[f"delay_p{int(q * 100)}_ms"] = d.quantile(q).item() if d.numel() else None
    out["delay_mean_ms"] = d.mean().item() if d.numel() else None
    return out, d


def ks(a, b):
    if a.numel() == 0 or b.numel() == 0:
        return None
    x = torch.cat([a, b]).sort().values
    fa = torch.searchsorted(a.sort().values, x, right=True).float() / a.numel()
    fb = torch.searchsorted(b.sort().values, x, right=True).float() / b.numel()
    return (fa - fb).abs().max().item()


def save(path, obj):
    torch.save(obj, path)


def dumps(o):
    return json.dumps(o, default=float)
