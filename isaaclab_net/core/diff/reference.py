"""Discrete counterparts of the differentiable KPIs: run a real engine level (L1, QA, L2-legacy, ...) with
binary sends and per-robot message sizes and aggregate the same KPIs as fluid.rollout.

Per-robot continuous message sizes are passed to the engine as one traffic class per (env, robot): sizes =
msg_bytes.flatten() and robot (e, r) sends class e * R + r + 1.
"""
from __future__ import annotations

import torch

from ..config import NRConfig
from ..engine import make_engine


def make_discrete(level, msg_bytes, device="cpu", seed=0, config=None, params=None):
    """Engine of `level` whose traffic class e * R + r + 1 has the size msg_bytes[e, r] (constant over time)."""
    E, R = msg_bytes.shape
    cfg = (config or NRConfig()).with_(seed=int(seed))
    sizes = tuple(float(v) for v in msg_bytes.detach().flatten().cpu())
    return make_engine(level, E, R, device, cfg, sizes=sizes, params=params)


def discrete_rollout(net, T, send, snr_db, warmup=0, pos=None):
    """Run the engine T steps. send: [T,E,R] bool (or [E,R] constant); snr_db: [E,R] or [T,E,R] (ignored if pos
    [E,R,2] is given, which goes through the engine's Radio). Returns the rollout() KPI dict (no energy)."""
    E, R = net.E, net.R
    dev = net.dev
    cls = (torch.arange(E * R, device=dev).view(E, R) + 1)
    aoi = torch.zeros(E, R, device=dev)
    acc = {k: torch.zeros(E, R, device=dev) for k in ("offered", "delivered", "delay_sum", "timed_out", "overflow",
                                                      "aoi_sum")}
    for t in range(T):
        s = send[t] if send.dim() == 3 else send
        s = s.to(dev)
        x = pos if pos is not None else (snr_db[t] if snr_db.dim() == 3 else snr_db)
        accepted = net.submit(None, torch.where(s, cls, torch.zeros_like(cls)))
        out = net.step(None, x)
        dlv = out["delivered"]
        newest = out["newest"]
        aoi = torch.where(newest >= 0, (out["t"][:, None] + 1 - newest).float(), aoi + 1)
        if t < warmup:
            continue
        acc["offered"] += s.float()
        acc["overflow"] += (s & ~accepted).float()
        acc["delivered"] += dlv.sum(-1).float()
        acc["delay_sum"] += torch.where(dlv, out["delay"], torch.zeros_like(out["delay"])).sum(-1)
        acc["timed_out"] += out["timed_out"].sum(-1).float()
        acc["aoi_sum"] += aoi
    n = T - warmup
    eps = 1e-9
    res = {k: acc[k] for k in ("offered", "delivered", "timed_out", "overflow")}
    res["delivery"] = acc["delivered"] / (acc["offered"] + eps)
    res["delay"] = acc["delay_sum"] / (acc["delivered"] + eps)
    res["aoi"] = acc["aoi_sum"] / n
    res["mean_delay"] = acc["delay_sum"].sum() / (acc["delivered"].sum() + eps)
    res["mean_delivery"] = acc["delivered"].sum() / (acc["offered"].sum() + eps)
    res["mean_aoi"] = res["aoi"].mean()
    res["delay_sum"] = acc["delay_sum"]
    return res
