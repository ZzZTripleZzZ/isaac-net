"""Spatially correlated random fields as sums of K plane waves, one field per (cell, env).

f(x) = amp * sum_k cos(k_k . x + phi_k), phi_k ~ U(0, 2 pi), directions uniform. With amp = sqrt(2 / K) the field has
unit variance at every point, and its autocorrelation is E_k[J0(|k| r)], set by the distribution of |k| (spectrum):

  "sos"  the legacy field: wavelengths uniform in [20, 60] m scaled by dcorr / 10.3. At dcorr = 10.3 m (the default
         shadow_dcorr_m, the 1/e distance the calibration measured on this field; analytically 10.19 m) the draws and
         the arithmetic are the legacy ones, bit for bit. The autocorrelation has a negative lobe (-0.18 at 2.1 dcorr).
  "exp"  exponential autocorrelation exp(-r / dcorr), the Gudmundson model of TR 38.901 Sec. 7.6.3.1. |k| is drawn
         from the radial density of the 2-D Fourier transform of exp(-r / d), p(k) = d^2 k (1 + k^2 d^2)^(-3/2), by
         inversion: |k| = sqrt((1 - u)^-2 - 1) / d. The ensemble autocorrelation is exact; a single field with few
         modes is less Gaussian than the legacy one (a few modes have short wavelengths).

Every draw has a fixed shape ([C, E, K]); reset(mask) draws all rows and keeps the masked ones, so partial resets
need no host sync, and evaluation is one einsum (graph-safe).

Randomness: a torch.Generator (the sequential stream; row e's draws then depend on E and on earlier draws), or an
engine's counter-based RNG (proto/rng.CounterRNG) with a base stream id s: the three uniform draws (direction, |k|,
phase) are the RESET draws of streams s, s + 1, s + 2 keyed by (seed, env id, episode), so env e's field depends only
on (seed, e, episode of e), not on E or on other envs' resets. The caller advances the env's episode before reset.
"""
from __future__ import annotations

import math

import torch

LEGACY_DCORR_M = 10.3      # NRConfig.shadow_dcorr_m default: the legacy 20-60 m band maps to itself
SPECTRA = ("sos", "exp")


def counter_uniform(rng, stream, C, K):
    """rand(i) for draw_plane_waves from an engine CounterRNG: RESET uniforms [E,C,K] of stream + i for every env row,
    keyed by (seed, env id, episode)."""
    return lambda i: rng.reset_uniform(None, stream + i, C, K)       # noqa: E731


def draw_plane_waves(n, C, K, device, generator=None, spectrum="sos", dcorr_m=LEGACY_DCORR_M, rand=None):
    """k [C,n,K,2] (rad/m), phi [C,n,K]. spectrum "sos" at dcorr_m == 10.3 is the legacy RadioMC draw, bit for bit.
    rand: optional rand(i) -> U[0, 1) [n,C,K] for draw i (0 direction, 1 |k|, 2 phase), e.g. counter_uniform(...);
    by default torch.rand from `generator` in that order."""
    g, d = generator, device
    if rand is None:
        rand = lambda i: torch.rand(n, C, K, device=d, generator=g)    # noqa: E731
    ang = rand(0) * 2 * math.pi
    if spectrum == "sos":
        wl = 20 + 40 * rand(1)
        if dcorr_m != LEGACY_DCORR_M:
            wl = wl * (dcorr_m / LEGACY_DCORR_M)
        k = torch.stack([torch.cos(ang), torch.sin(ang)], -1) * (2 * math.pi / wl)[..., None]
    elif spectrum == "exp":
        u = rand(1).clamp(max=1 - 1e-6)
        kmag = torch.sqrt((1 - u) ** -2 - 1) / dcorr_m
        k = torch.stack([torch.cos(ang), torch.sin(ang)], -1) * kmag[..., None]
    else:
        raise ValueError(f"unknown field spectrum {spectrum!r}; one of {SPECTRA}")
    phi = rand(2) * 2 * math.pi
    return k.permute(1, 0, 2, 3).contiguous(), phi.permute(1, 0, 2).contiguous()


def eval_plane_waves(pos, k, phi, amp):
    """pos [E,R,2], k [C,E,K,2], phi [C,E,K] -> amp * sum_k cos(k . x + phi) as [E,R,C] (the legacy op order)."""
    arg = torch.einsum("erx,cekx->cerk", pos, k) + phi[:, :, None, :]       # [C,E,R,K]
    return (amp * torch.cos(arg).sum(-1)).permute(1, 2, 0)


class PlaneWaveField:
    """One field per (cell, env) with standard deviation `sigma` (amp = sigma sqrt(2 / K)).

    rng / stream: draw from an engine CounterRNG (streams stream .. stream + 2, keyed by env id and episode) instead
    of `generator`."""

    def __init__(self, E, C, K, device, generator=None, spectrum="exp", dcorr_m=10.0, sigma=1.0, rng=None, stream=0):
        self.E, self.C, self.K, self.dev, self.gen = E, C, K, device, generator
        self.rng, self.stream = rng, int(stream)
        self.spectrum, self.dcorr = spectrum, float(dcorr_m)
        self.amp = float(sigma) * math.sqrt(2 / K)
        self.k, self.phi = self._draw()

    def _draw(self):
        rand = None if self.rng is None else counter_uniform(self.rng, self.stream, self.C, self.K)
        return draw_plane_waves(self.E, self.C, self.K, self.dev, self.gen, self.spectrum, self.dcorr, rand)

    def reset(self, m):
        """m: bool [E] mask of envs to redraw (fixed shape: draw all, keep masked rows)."""
        k, phi = self._draw()
        self.k = torch.where(m.view(1, -1, 1, 1), k, self.k)
        self.phi = torch.where(m.view(1, -1, 1), phi, self.phi)

    def __call__(self, pos):
        return eval_plane_waves(pos, self.k, self.phi, self.amp)


def sum_of_cosines_cdf_table(K, n=1 << 16, seed=0):
    """Sorted samples [n] (CPU, float32) of sqrt(2/K) sum_k cos(U_k), U_k ~ U(0, 2 pi): the exact marginal of a unit
    field at any point. uniform_from_field() maps the field through this empirical CDF, so a threshold u < p holds
    with probability p (to 1/n) whatever K, which a Gaussian CDF would only approximate."""
    g = torch.Generator().manual_seed(seed)
    z = math.sqrt(2 / K) * torch.cos(torch.rand(n, K, generator=g, dtype=torch.float64) * 2 * math.pi).sum(-1)
    return z.sort().values.float()


def uniform_from_field(z, table):
    """z [...] unit field values -> spatially correlated U(0, 1) values (empirical CDF of the table)."""
    n = table.numel()
    return (torch.searchsorted(table, z.contiguous()).float() + 0.5) / (n + 1)
