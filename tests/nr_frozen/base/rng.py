# FROZEN copy of isaac_net/core/proto/rng.py at 2bb77f8 (2bb77f8a7f866acbe7aa4fb188ec2c047775d302), written by tests/scripts/refreeze_nr.py.
# Do not edit: re-freeze from a commit instead (see that script). Import rewrites:
#   '^(\\s*)from \\. import rng_triton' -> '\\1from isaac_net.core.proto import rng_triton'
#   '^(\\s*)from \\.rng_triton import' -> '\\1from isaac_net.core.proto.rng_triton import'
"""Engine-owned random streams (NRConfig.rng = "engine").

Every random number an engine draws while it runs is a pure function of

    (seed, env id, episode of that env, channel, call counter of that env, stream, element index)

where the episode counts the resets of the env, the channel is SUBMIT, STEP or RESET, the call counter counts the
submits (or steps) of the env since its last reset, and the stream names one draw site (for example the fading
innovation or the BLER draw). Consequences:
  * a policy's own use of the global torch RNG never changes the network (no shared stream);
  * env e's network in its k-th episode depends only on (seed, e, k) and its own inputs, never on which other
    envs reset when, on E, or on the backend's order of draws;
  * reset(env_ids) re-seeds exactly those envs (their episode advances), deterministically.

The counter-based generator is a 32-bit integer hash (Weyl sequence + two rounds of the "lowbias32" mixer, Chris
Wellons' hash prospector constants 0x21f0aaad / 0x735a2d97). Uniforms take the top 24 bits (as torch.rand does);
normals use Box-Muller on two uniforms (cosine branch). The torch implementation keeps every value in [0, 2^32)
inside int64 tensors with multipliers below 2^31, so no product overflows. On CUDA a Triton kernel computes the
same integers in uint32 (one launch per draw, graph-capturable, no generator state); uniforms are then bitwise
equal to the torch path and normals agree to float rounding. The reference and fast backends on one device use
the same implementation, so they see bitwise identical draws. ISAAC_NET_RNG_TORCH=1 forces the torch path.

The legacy behavior (stepping draws from the global torch RNG, reset draws from the engine's torch.Generator) is
NRConfig.rng = "global"; engines then have rng = None.
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
SUBMIT, STEP, RESET = 1, 2, 3
MODES = ("engine", "global")


def mix32(x):
    """lowbias32 on a Python int or an int64 tensor holding values in [0, 2^32)."""
    x = x ^ (x >> 16)
    x = (x * C1) & M32
    x = x ^ (x >> 15)
    x = (x * C2) & M32
    return x ^ (x >> 15)


def salt(v):
    """(mix32(v) + GOLD) mod 2^32: what combine() xors into the running key."""
    return (mix32(v & M32) + GOLD) & M32


def combine(h, v):
    return mix32(h ^ salt(v))


def seed_key(seed):
    return combine(mix32(seed & M32), (seed >> 32) & M32)


def check_mode(rng):
    if rng not in MODES:
        raise ValueError(f"rng={rng!r}; one of {MODES}")
    return rng


class CounterRNG:
    """Per-env counter-based streams of one engine. State: episode [E] and one call counter [E] per channel, all
    long device buffers updated in place (so captured CUDA graphs that read them stay valid)."""

    def __init__(self, seed, E, device):
        self.seed = int(seed)
        self.E, self.dev = E, torch.device(device)
        self.s0 = seed_key(self.seed)
        self.env = torch.arange(E, dtype=torch.long, device=self.dev)     # env ids of the rows (key of the draws)
        self.env_offset = 0
        self.episode = torch.full((E,), -1, dtype=torch.long, device=self.dev)
        self.ctr = {SUBMIT: torch.zeros(E, dtype=torch.long, device=self.dev),
                    STEP: torch.zeros(E, dtype=torch.long, device=self.dev)}
        self._zero = torch.zeros(E, dtype=torch.long, device=self.dev)
        self._weyl = {}
        self.use_triton = self.dev.type == "cuda" and _triton_ok()

    # ------------------------------------------------------------------ counters
    def reset(self, ids):
        """New episode for env ids (None = all): episode += 1, call counters = 0. Eager, in place."""
        if ids is None:
            self.episode.add_(1)
            for c in self.ctr.values():
                c.zero_()
        else:
            self.episode.index_add_(0, ids, torch.ones_like(ids))
            for c in self.ctr.values():
                c.index_fill_(0, ids, 0)

    def set_env_offset(self, offset):
        """Key row i by env id offset + i instead of i (in place, so captured graphs stay valid). A shard of a
        larger batch (core/sharded.py) sets the global id of its first env, so an env draws the same numbers
        whichever shard holds it."""
        offset = int(offset)
        self.env.add_(offset - self.env_offset)
        self.env_offset = offset

    def tick(self, channel):
        """One more call of `channel` (SUBMIT / STEP) for every env. Run it outside captured graphs."""
        self.ctr[channel].add_(1)

    # ------------------------------------------------------------------ draws
    def uniform(self, channel, stream, *tail):
        """U[0, 1) draws [E, *tail] of the current call of `channel` (after tick)."""
        return self._draw(self.env, self.episode, self.ctr[channel], channel, stream, tail, False)

    def normal(self, channel, stream, *tail):
        return self._draw(self.env, self.episode, self.ctr[channel], channel, stream, tail, True)

    def reset_uniform(self, ids, stream, *tail):
        """Reset draws [n, *tail] for env ids (None = all), keyed by (env, episode) after reset(ids)."""
        env, ep, ctr = self._rows(ids)
        return self._draw(env, ep, ctr, RESET, stream, tail, False)

    def reset_normal(self, ids, stream, *tail):
        env, ep, ctr = self._rows(ids)
        return self._draw(env, ep, ctr, RESET, stream, tail, True)

    def _rows(self, ids):
        if ids is None:
            return self.env, self.episode, self._zero
        return self.env.index_select(0, ids), self.episode.index_select(0, ids), torch.zeros_like(ids)

    def _draw(self, env, ep, ctr, channel, stream, tail, normal):
        n = 1
        for d in tail:
            n *= int(d)
        rows = env.shape[0]
        if self.use_triton:
            out = draw_triton(env, ep, ctr, self.s0, salt(channel), salt(stream), n, normal)
        else:
            out = self._draw_torch(env, ep, ctr, channel, stream, n, normal)
        return out.view(rows, *tail)

    def _weyl_idx(self, n):
        w = self._weyl.get(n)
        if w is None:
            w = (torch.arange(n, dtype=torch.long, device=self.dev) * GOLD) & M32
            self._weyl[n] = w
        return w

    def _draw_torch(self, env, ep, ctr, channel, stream, n, normal):
        base = key_torch(self.s0, env, ep, channel, ctr, stream)[:, None]
        if not normal:
            return uniform_bits(elem(base, self._weyl_idx(n)))
        w = self._weyl_idx(2 * n).view(n, 2)
        h1, h2 = elem(base, w[:, 0]), elem(base, w[:, 1])
        return box_muller(h1, h2)


def key_torch(s0, env, ep, channel, ctr, stream):
    """Base key [rows] (int64 in [0, 2^32)) of (seed, env, episode, channel, counter, stream)."""
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


# ---------------------------------------------------------------------------------------------- Triton
_TRITON = None


def _triton_ok():
    global _TRITON
    if os.environ.get("ISAAC_NET_RNG_TORCH", "") == "1":
        return False
    if _TRITON is None:
        try:
            from isaac_net.core.proto import rng_triton  # noqa: F401
            _TRITON = True
        except Exception:
            _TRITON = False
    return _TRITON


def draw_triton(env, ep, ctr, s0, chs, sts, n, normal):
    from isaac_net.core.proto.rng_triton import launch_draw
    return launch_draw(env, ep, ctr, s0, chs, sts, n, normal)
