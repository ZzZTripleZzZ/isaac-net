"""Engine-owned random streams of the NR engine (NRConfig.rng = "engine"), on proto/rng.CounterRNG.

Every random number the NR engine draws is a pure function of (seed, env id, episode of that env, channel, counter,
stream, element index), the key scheme of proto/rng.py (same hash, same Triton kernel on CUDA, same torch path on
CPU):
  episode   counts the resets of the env (the constructor's full reset is episode 0)
  channel   STEP for the draws of a control step, RESET for the draws of reset(env_ids)
  counter   STEP: control steps of the env since its last reset (the env's episode clock), ticked at the end of
            every step; RESET: 0
  stream    (site << 16) | slot, where slot is the slot index inside the control step and site one of
              FADING   fading innovation, one normal per element of the fading state h [R, (C,) S, 2]
              BLER_UL  UL transport-block decode, one uniform per robot
              BLER_DL  DL transport-block decode, one uniform per robot
              H0       (channel RESET, slot 0) initial fading state
  element   index into the flattened per-env draw (C-order over the per-env shape)
So env e's substream is keyed by (e, episode, slot clock = counter * slots_per_step + slot): a policy's use of the
global torch RNG never changes the network, env e's network in its k-th episode depends only on (seed, e, k) and its
own inputs (not on E or on other envs' resets), and the reference, graph and triton backends see the same draws (the
fused kernel inlines proto/rng_triton's hash: uniforms bitwise, normals to float rounding).

rng = "global" keeps the earlier behavior: step draws from the global torch RNG (torch.randn_like / rand_like), reset
draws from the engine's torch.Generator.
"""
from __future__ import annotations

import torch

from .proto.rng import RESET, STEP, CounterRNG, salt  # noqa: F401  (salt: the kernel's channel key)

FADING, BLER_UL, BLER_DL, H0 = 1, 2, 3, 4
BLER = {"ul": BLER_UL, "dl": BLER_DL}


def stream_id(site, slot=0):
    return (site << 16) | slot


class NRRng(CounterRNG):
    """CounterRNG with the NR engine's draw sites (step draws per slot, reset draws for every env row)."""

    def reset_mask(self, mask=None):
        """New episode for the envs of the bool mask [E] (None = all). In place (fixed shape)."""
        if mask is None:
            self.reset(None)
            return
        self.episode.add_(mask.long())
        for c in self.ctr.values():
            c.masked_fill_(mask, 0)

    def step_normal(self, site, slot, n):
        """N(0, 1) [E, n] of `site` in slot `slot` of the current control step."""
        return self.normal(STEP, stream_id(site, slot), n)

    def step_uniform(self, site, slot, n):
        return self.uniform(STEP, stream_id(site, slot), n)

    def reset_normal_all(self, site, n):
        """Reset draws [E, n] for every env, keyed by (env, episode); keep the rows of the envs just reset."""
        return self.reset_normal(None, stream_id(site), n)

    def tick_step(self):
        self.tick(STEP)


def make_rng(cfg, E, device, seed):
    """NRRng for cfg.rng == "engine" (seed None = drawn from the global torch RNG), else None."""
    if cfg.rng != "engine":
        return None
    if seed is None:
        seed = cfg.seed if cfg.seed is not None else int(torch.randint(0, 2 ** 62, ()).item())
    return NRRng(seed, E, device)
