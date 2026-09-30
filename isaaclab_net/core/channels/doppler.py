"""Per-robot Doppler for the AR(1) Rayleigh fading of the NR engine.

The engine's fading is h <- rho h + sqrt(1 - rho^2) n per elapsed slot group, with one global rho per ms
(NRConfig.fading_rho_per_ms, or derived from ue_speed_mps). With NRConfig.fading_doppler = "per_robot" every robot
gets its own rho from its own speed: rho_ms = J0(2 pi f_D 2.5 ms)^(1 / 2.5), f_D = v f_c / c, the rule of
config.fading_rho_from_speed applied per robot. The speed comes from the velocity input of the step, or from
consecutive poses (|x_t - x_{t-1}| / control step), floored at doppler_min_speed_mps.

install_per_robot_fading(net) gives an NRNet a per-robot rho tensor net.fading_rho_ms [E,R] (initially the global
rho), which NRNet._evolve reads directly (the NR engine's per-robot fading input; the graph and triton backends take it
as a static-shape input). The random draws are the same calls in the same order, so with every robot at the global
speed the result equals the global model up to float rounding of rho.
"""
from __future__ import annotations

import math

import torch

C_LIGHT = 299_792_458.0     # as config.fading_rho_from_speed


def rho_per_ms_from_speed(speed_mps, carrier_ghz, anchor_ms=2.5):
    """Tensor version of config.fading_rho_from_speed: speed [...] m/s -> AR(1) correlation per ms [...]."""
    fd = speed_mps * (carrier_ghz * 1e9 / C_LIGHT)
    j0 = torch.special.bessel_j0(2 * math.pi * anchor_ms * 1e-3 * fd)
    return j0.clamp(0.0, 1.0) ** (1 / anchor_ms)


def install_per_robot_fading(net):
    """Give an NRNet per-robot fading correlation (net.fading_rho_ms [E,R], updated by the caller every step)."""
    if not (hasattr(net, "fading_rho_ms") and hasattr(net, "h") and hasattr(net, "last_g")):
        raise RuntimeError("per-robot Doppler needs the NR engine's AR(1) fading state (NRNet.h, fading_rho_ms)")
    net.fading_rho_ms = torch.full((net.E, net.R), float(net.cfg.fading_rho_per_ms), device=net.dev)
    return net
