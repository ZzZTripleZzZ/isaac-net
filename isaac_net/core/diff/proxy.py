"""Recipe: a small neural proxy of the slot-level KPIs, fitted to L2-legacy rollouts and used as a differentiable
stand-in (L2D option b).

The engine's NN level (levels/surrogates.py, DelayNet) predicts per-frame delay quantiles from send-time features
and samples outcomes autoregressively, which is not differentiable in the decisions. This proxy is its sibling for
optimization: the same MLP body (3 x SiLU layers), but it maps per-robot decisions and env aggregates directly to
the per-robot episode KPIs of L2-legacy, so d(KPI) / d(decision) comes from autograd:

    features (per robot, N_FEAT): send probability p, log message size, SNR at the transmit power, own offered
      load p * bytes, env offered load, env mean p, env mean SNR, R / 16
    targets: log(1 + mean delay [control steps]), logit(delivery ratio), log(mean AoI [control steps])

Steps (benchmarks/diff/nn_proxy.py runs them at a useful scale):
    X, Y = collect(n_batches, E, R, T)      # L2-legacy with random per-robot (p, bytes, SNR), Bernoulli sends
    model = fit(X, Y)                        # masked MSE, Adam
    kpi = predict(model, p, msg_bytes, snr_db)   # dict of [E,R] tensors, differentiable in all three inputs

Limits: the proxy is only as good as its training distribution (stationary Bernoulli sends, the default L2-legacy
constants, one cell, the ranges in collect); it averages over fading and HARQ rather than modeling them, and its
gradients are the gradients of a regression fit, not of the simulator. Check them against finite differences of
the real engine before relying on them (nn_proxy.py does).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from .reference import discrete_rollout, make_discrete

N_FEAT = 8
CAP_REF = 1e5           # bytes per control step, order of the cell's uplink capacity (feature scale only)
RANGES = {"p": (0.05, 0.8), "bytes": (1000.0, 40000.0), "snr": (-5.0, 35.0)}


def features(p, msg_bytes, snr_db):
    """[E,R,N_FEAT] proxy inputs from per-robot decisions [E,R] (all differentiable)."""
    R = p.shape[-1]
    load = p * msg_bytes / CAP_REF
    cols = [p, torch.log(msg_bytes / 1e4), snr_db / 40.0, load,
            load.sum(-1, keepdim=True).expand_as(p), p.mean(-1, keepdim=True).expand_as(p),
            (snr_db / 40.0).mean(-1, keepdim=True).expand_as(p), torch.full_like(p, R / 16.0)]
    return torch.stack(cols, -1)


class ProxyNet(nn.Module):
    """MLP body as in levels.surrogates.DelayNet, three regression heads (delay, delivery, AoI)."""

    def __init__(self, din=N_FEAT, h=128):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(din, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU())
        self.head = nn.Linear(h, 3)
        self.register_buffer("xm", torch.zeros(din))
        self.register_buffer("xs", torch.ones(din))

    def forward(self, x):
        return self.head(self.body((x - self.xm) / self.xs))


@torch.no_grad()
def collect(n_batches=8, E=64, R=8, T=200, warmup=20, device="cpu", seed=0, level="L2-legacy"):
    """Per-robot (features, targets) from n_batches rollouts of `level` with random per-robot decisions.
    Returns X [n, N_FEAT] and Y [n, 3] (NaN where undefined: no delivered message has no delay)."""
    g = torch.Generator().manual_seed(seed)
    Xs, Ys = [], []
    for b in range(n_batches):
        u = lambda lo, hi: lo + (hi - lo) * torch.rand(E, R, generator=g)     # noqa: E731
        p = u(*RANGES["p"])
        B = torch.exp(u(math.log(RANGES["bytes"][0]), math.log(RANGES["bytes"][1])))
        snr = u(*RANGES["snr"])
        send = torch.rand(T, E, R, generator=g) < p
        net = make_discrete(level, B, device=device, seed=seed * 1000 + b)
        r = discrete_rollout(net, T, send.to(device), snr.to(device), warmup=warmup)
        dl = torch.where(r["delivered"] > 0, torch.log1p(r["delay"]), torch.full_like(r["delay"], float("nan")))
        dv = r["delivery"].clamp(1e-3, 1 - 1e-3)
        valid_dv = r["offered"] > 0
        y = torch.stack([dl, torch.where(valid_dv, torch.log(dv) - torch.log1p(-dv), torch.full_like(dv, float("nan"))),
                         torch.log(r["aoi"].clamp(min=1e-3))], -1)
        Xs.append(features(p, B, snr).reshape(-1, N_FEAT))
        Ys.append(y.cpu().reshape(-1, 3))
    return torch.cat(Xs), torch.cat(Ys)


def fit(X, Y, epochs=300, lr=2e-3, batch=1024, device="cpu", seed=0, h=128):
    """Masked-MSE fit of a ProxyNet to (X, Y); returns the model in eval mode."""
    torch.manual_seed(seed)
    model = ProxyNet(X.shape[1], h).to(device)
    model.xm.copy_(X.mean(0))
    model.xs.copy_(X.std(0).clamp(min=1e-3))
    X, Y = X.to(device), Y.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = X.shape[0]
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            pred, y = model(X[idx]), Y[idx]
            m = torch.isfinite(y)
            loss = ((pred - torch.nan_to_num(y)) ** 2 * m).sum() / m.sum().clamp(min=1)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model.eval()


def predict(model, p, msg_bytes, snr_db):
    """Per-robot KPIs {delay, delivery, aoi} [E,R] (control steps, ratio, control steps); differentiable."""
    o = model(features(p, msg_bytes, snr_db))
    return {"delay": torch.expm1(o[..., 0]), "delivery": torch.sigmoid(o[..., 1]), "aoi": torch.exp(o[..., 2])}
