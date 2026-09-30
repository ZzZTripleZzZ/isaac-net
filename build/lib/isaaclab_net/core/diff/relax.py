"""Smooth relaxations of the hard operations of the fluid levels, all controlled by one temperature `tau`.

tau = 0 selects the exact hard operation (min, max, step, floor), so the relaxed models contain the discrete
model as the tau = 0 case. For tau > 0 the operations are smooth. Two kinds of temperature appear:

  relative (log domain), for positive quantities in bytes or subband shares:
      smin(a, b)  = exp(-tau * logsumexp(-log a / tau, -log b / tau))       <= min(a, b), -> min as tau -> 0
      smax(a, b)  = exp( tau * logsumexp( log a / tau,  log b / tau))       >= max(a, b)
      ge(x, y)    = sigmoid((log x - log y) / tau)                          soft indicator of x >= y
    The error is relative: smin(a, a) = a * 2^-tau, so tau = 0.05 costs about 3.4 % at the crossing point and
    nothing far from it. Byte counts of very different size (a 4 kB and a 30 kB frame) see the same relative
    smoothing, which a fixed byte temperature would not give.

  additive, for integer-valued quantities (frame counts, ages in control steps):
      below(n, m) = sigmoid((m - 0.5 - n) / tau)                             soft indicator of n < m for integers
      floor_int(x, lo, hi) = lo + sum_{m = lo+1..hi} sigmoid((x - m) / tau)  soft floor clamped to [lo, hi]

Every function takes tau as a python float so that the hard branch is chosen outside the autograd graph.
"""
from __future__ import annotations

import torch

TINY = 1e-12            # floor inside log() for byte quantities that can be exactly zero (empty frames)


def _log(x):
    return torch.log(x.clamp(min=TINY))


def _as(b, a):
    return b if torch.is_tensor(b) else torch.as_tensor(b, dtype=a.dtype, device=a.device)


def smin(a, b, tau):
    """Soft minimum of two non-negative tensors (relative temperature). smin(0, b) = ~0 and smin(a, 0) = ~0."""
    if tau <= 0:
        return torch.minimum(a, _as(b, a))
    la, lb = _log(a), _log(_as(b, a))       # -tau * logsumexp(-la / tau, -lb / tau), in closed form
    return torch.exp(torch.minimum(la, lb) - tau * torch.nn.functional.softplus(-(la - lb).abs() / tau))


def smax(a, b, tau):
    """Soft maximum of two positive tensors (relative temperature)."""
    if tau <= 0:
        return torch.maximum(a, _as(b, a))
    la, lb = _log(a), _log(_as(b, a))
    return torch.exp(torch.maximum(la, lb) + tau * torch.nn.functional.softplus(-(la - lb).abs() / tau))


def ge(x, y, tau, atol=0.0):
    """Soft indicator of x >= y for non-negative x, y (relative temperature). Hard: x >= y - atol."""
    if tau <= 0:
        return (x >= y - atol).to(x.dtype)
    return torch.sigmoid((_log(x) - _log(_as(y, x))) / tau)


def positive(x, thr, tau):
    """Soft indicator of x > thr for non-negative x and a positive threshold thr (relative temperature)."""
    if tau <= 0:
        return (x > thr).to(x.dtype)
    return torch.sigmoid((_log(x) - _log(_as(thr, x))) / tau)


def below(n, m, tau):
    """Soft indicator of n < m for an integer-valued n and integer m (additive temperature, in counts)."""
    if tau <= 0:
        return (n < m - 0.5).to(n.dtype)
    return torch.sigmoid((m - 0.5 - n) / tau)


def floor_int(x, lo, hi, tau):
    """floor(x) clamped to [lo, hi] for integers lo < hi; soft version = lo + sum of sigmoid steps at lo+1..hi."""
    if tau <= 0:
        return torch.floor(x).clamp(lo, hi)
    steps = torch.arange(lo + 1, hi + 1, dtype=x.dtype, device=x.device)
    return lo + torch.sigmoid((x[..., None] - steps) / tau).sum(-1)


def harmonic(n):
    """Harmonic number H_n for real n >= 1 (digamma(n + 1) + Euler-gamma); equals 1 + 1/2 + ... + 1/n at integers."""
    return torch.digamma(n + 1.0) + 0.5772156649015329
