"""Engine-owned random streams of the NR engine (NRConfig.rng = "engine").

Every random number the NR engine draws is a pure function of

    (seed, env id, episode of that env, channel, step counter of that env, stream, element index)

  episode   counts the resets of the env (the constructor's full reset is episode 0)
  channel   STEP for the draws of a control step, RESET for the draws of reset(env_ids)
  counter   control steps of the env since its last reset (= the env's episode clock)
  stream    (site << 16) | slot, where slot is the slot index inside the control step and site one of
              FADING   fading innovation, one normal per element of the fading state h [R, (C,) S, 2]
              BLER_UL  UL transport-block decode, one uniform per robot
              BLER_DL  DL transport-block decode, one uniform per robot
              H0       (channel RESET, slot 0) initial fading state
  element   index into the flattened per-env draw (C-order over the per-env shape)

so the per-env substream of env e is keyed by (e, episode, slot clock = counter * slots_per_step + slot). Consequences:
a policy's own use of the global torch RNG never changes the network; env e's network in its k-th episode depends
only on (seed, e, k) and its own inputs, never on E or on which other envs reset when; the reference, graph and triton
backends see the same draws.

The hash is the one of the prototype levels' engine RNG (proto/rng.py on the protolevels branch): a 32-bit Weyl
sequence mixed by two rounds of "lowbias32" (Chris Wellons' hash prospector, constants 0x21f0aaad / 0x735a2d97).
Uniforms take the top 24 bits (as torch.rand does); normals use Box-Muller on two uniforms (cosine branch). The torch
implementation keeps every value in [0, 2^32) inside int64 tensors with multipliers below 2^31, so no product
overflows. On CUDA a Triton kernel (nr_triton.draw_kernel) computes the same integers in uint32, one launch per draw,
graph-capturable, with no generator state; its uniforms are bitwise equal to the torch path and its normals agree to
float rounding. The fused NR Triton kernel inlines the same helpers. The reference and graph backends on one device
use the same implementation, so they see bitwise identical draws. ISAACLAB_NET_RNG_TORCH=1 forces the torch path.

rng = "global" keeps the earlier behavior: step draws from the global torch RNG (torch.randn_like / rand_like), reset
draws from the engine's torch.Generator.
"""
from __future__ import annotations

import math
import os

import torch

M32 = 0xFFFFFFFF
C1, C2 = 0x21F0AAAD, 0x735A2D97
GOLD = 0x9E3779B9
INV24 = 1.0 / 16777216.0
TWO_PI = 2.0 * math.pi
STEP, RESET = 2, 3
FADING, BLER_UL, BLER_DL, H0 = 1, 2, 3, 4
BLER = {"ul": BLER_UL, "dl": BLER_DL}


def stream_id(site, slot=0):
    return (site << 16) | slot


def mix32(x):
    """lowbias32 on a Python int or an int64 tensor holding values in [0, 2^32)."""
    x = x ^ (x >> 16)
    x = (x * C1) & M32
    x = x ^ (x >> 15)
    x = (x * C2) & M32
    return x ^ (x >> 15)


def salt(v):
    """(mix32(v) + GOLD) mod 2^32."""
    return (mix32(v & M32) + GOLD) & M32


def seed_key(seed):
    """32-bit key of a 64-bit seed."""
    return mix32(mix32(seed & M32) ^ salt((seed >> 32) & M32))


def key_torch(s0, env, ep, channel, ctr, stream):
    """Base key [rows] (int64 in [0, 2^32)) of (seed key, env, episode, channel, counter, stream)."""
    h = mix32(s0 ^ salt(env & M32))
    h = mix32(h ^ salt(ep & M32))
    h = mix32(h ^ salt(channel))
    h = mix32(h ^ salt(ctr & M32))
    return mix32(h ^ salt(stream))


def elem(base, weyl):
    """Element hash: two lowbias32 rounds over base + i * GOLD (mod 2^32)."""
    return mix32(mix32((base + weyl) & M32) ^ base)


def uniform_bits(h):
    return (h >> 8).to(torch.float32) * INV24


def box_muller(h1, h2):
    u1 = ((h1 >> 8) + 1).to(torch.float32) * INV24          # (0, 1]
    u2 = (h2 >> 8).to(torch.float32) * INV24
    return torch.sqrt(-2.0 * torch.log(u1)) * torch.cos(TWO_PI * u2)


_TRITON = None


def triton_ok():
    global _TRITON
    if os.environ.get("ISAACLAB_NET_RNG_TORCH", "") == "1":
        return False
    if _TRITON is None:
        try:
            from . import nr_triton  # noqa: F401
            _TRITON = True
        except Exception:
            _TRITON = False
    return _TRITON


class NRRng:
    """Per-env counter-based streams of one NR engine. State: episode [E] and step counter ctr [E], long device
    buffers updated in place (captured CUDA graphs that read them stay valid)."""

    def __init__(self, seed, E, device):
        self.seed = int(seed)
        self.E, self.dev = E, torch.device(device)
        self.s0 = seed_key(self.seed)
        self.env = torch.arange(E, dtype=torch.long, device=self.dev)
        self.episode = torch.full((E,), -1, dtype=torch.long, device=self.dev)
        self.ctr = torch.zeros(E, dtype=torch.long, device=self.dev)
        self._zero = torch.zeros(E, dtype=torch.long, device=self.dev)
        self._weyl = {}
        self.use_triton = self.dev.type == "cuda" and triton_ok()

    def reset(self, mask=None):
        """New episode for the envs of the bool mask [E] (None = all): episode += 1, counter = 0. In place."""
        if mask is None:
            self.episode.add_(1)
            self.ctr.zero_()
        else:
            self.episode.add_(mask.long())
            self.ctr.masked_fill_(mask, 0)

    def tick(self):
        """One more control step for every env (in place; captured by the graph backend)."""
        self.ctr.add_(1)

    def normal(self, site, slot, n):
        """Step draws N(0, 1) [E, n] of `site` in slot `slot` of the current control step."""
        return self._draw(self.ctr, STEP, stream_id(site, slot), n, True)

    def uniform(self, site, slot, n):
        """Step draws U[0, 1) [E, n]."""
        return self._draw(self.ctr, STEP, stream_id(site, slot), n, False)

    def reset_normal(self, site, n):
        """Reset draws N(0, 1) [E, n] keyed by (env, episode); call after reset(mask) and keep the reset rows."""
        return self._draw(self._zero, RESET, stream_id(site), n, True)

    def _draw(self, ctr, channel, stream, n, normal):
        if self.use_triton:
            from .nr_triton import launch_draw
            return launch_draw(self.env, self.episode, ctr, self.s0, salt(channel), salt(stream), n, normal)
        base = key_torch(self.s0, self.env, self.episode, channel, ctr, stream)[:, None]
        if not normal:
            return uniform_bits(elem(base, self._weyl_idx(n)))
        w = self._weyl_idx(2 * n).view(n, 2)
        return box_muller(elem(base, w[:, 0]), elem(base, w[:, 1]))

    def _weyl_idx(self, n):
        w = self._weyl.get(n)
        if w is None:
            w = (torch.arange(n, dtype=torch.long, device=self.dev) * GOLD) & M32
            self._weyl[n] = w
        return w


def make_rng(cfg, E, device, seed):
    """NRRng for cfg.rng == "engine" (seed None = drawn from the global torch RNG), else None."""
    if cfg.rng != "engine":
        return None
    if seed is None:
        seed = cfg.seed if cfg.seed is not None else int(torch.randint(0, 2 ** 62, ()).item())
    return NRRng(seed, E, device)
