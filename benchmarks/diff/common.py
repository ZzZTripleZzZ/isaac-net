"""Shared scenario and KPI wrappers for the differentiable-model benchmarks.

A scenario is E envs x R robots with positions drawn uniformly in an annulus 20..120 m around the gNB at the
origin (SNR roughly -5..27 dB at 23 dBm through proto.netsim.Radio), one Radio per scenario, and a fixed set of
send uniforms u [T,E,R] so that Bernoulli sends (u < p) use common random numbers across parameter values.
"""
import math
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from isaac_net.core.diff import (DiffFluid, discrete_rollout, make_discrete, radio_snr_db,  # noqa: E402
                                    relaxed_bernoulli, rollout)
from isaac_net.core.proto.netsim import Radio  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


class Scenario:
    def __init__(self, E, R, T, device="cpu", seed=0, r_min=20.0, r_max=120.0):
        self.E, self.R, self.T, self.dev = E, R, T, torch.device(device)
        g = torch.Generator().manual_seed(seed)
        ang = 2 * math.pi * torch.rand(E, R, generator=g)
        rad = torch.sqrt(r_min ** 2 + (r_max ** 2 - r_min ** 2) * torch.rand(E, R, generator=g))
        self.pos = torch.stack([rad * torch.cos(ang), rad * torch.sin(ang)], -1).to(self.dev)
        self.u = torch.rand(T, E, R, generator=g).to(self.dev)
        rg = torch.Generator(device=self.dev).manual_seed(seed + 7)
        self.radio = Radio(E, self.dev, generator=rg)
        self.seed = seed

    def snr(self, pos=None, tx_dbm=None, dtype=torch.float32):
        p = (self.pos if pos is None else pos).to(dtype)
        s = radio_snr_db(self.radio, p)
        return s if tx_dbm is None else s + (tx_dbm - 23.0)

    # ------------------------------------------------------------------ KPIs
    def diff_kpis(self, p, B, tx_dbm=None, pos=None, mode="L1", tau=0.05, send="mean", lam=0.1, warmup=None,
                  dtype=torch.float32, deadline=None):
        """Differentiable KPIs. send = "mean" (mean-field weight p) or "concrete" (relaxed Bernoulli with the
        scenario's uniforms and temperature lam)."""
        E, R, T = self.E, self.R, self.T
        net = DiffFluid(E, R, self.dev, mode=mode, tau=tau, dtype=dtype)
        pos_ = self.pos.to(dtype) if pos is None else pos
        snr = radio_snr_db(self.radio, pos_)
        p = torch.as_tensor(p, device=self.dev, dtype=dtype).expand(E, R)
        if send == "mean":
            s, ta = p, None
        else:
            s, ta = relaxed_bernoulli(p.expand(T, E, R), self.u.to(dtype), lam), {"send"}
        return rollout(net, T, s, torch.as_tensor(B, device=self.dev, dtype=dtype).expand(E, R), snr_db=snr,
                       tx_dbm=tx_dbm, warmup=T // 10 if warmup is None else warmup, time_axis=ta, deadline=deadline)

    @torch.no_grad()
    def discrete_kpis(self, level, p, B, tx_dbm=None, pos=None, seed=None, warmup=None):
        """KPIs of a real engine level with Bernoulli sends u < p (common random numbers) and the same SNR."""
        E, R, T = self.E, self.R, self.T
        B = torch.as_tensor(B, dtype=torch.float32, device=self.dev).expand(E, R).contiguous()
        p = torch.as_tensor(p, dtype=torch.float32, device=self.dev).expand(E, R)
        snr = self.snr(pos=pos, tx_dbm=tx_dbm)
        net = make_discrete(level, B, device=self.dev, seed=self.seed if seed is None else seed)
        return discrete_rollout(net, T, self.u < p, snr, warmup=T // 10 if warmup is None else warmup)
