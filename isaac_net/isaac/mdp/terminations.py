"""TerminationTerm that steps the network (no Isaac imports): the manager-based twin of the Direct mixin's
net_step in _get_dones.

    from isaaclab.managers import TerminationTermCfg as DoneTerm
    net_step = DoneTerm(func=net_mdp.net_step_done)          # first in TerminationsCfg; never ends an episode

ManagerBasedRLEnv.step runs, after physics: termination terms -> reward terms -> resets of finished envs ("reset"
events) -> interval events -> observation terms. The termination manager is the first post-physics stage, so a
network step there sees the end-of-step poses and the pre-reset queues, the rewards read the fresh network output,
the resets clear it for finished envs, and the observations see it. That is the order of the Direct mixin
(_get_dones -> _get_rewards -> _reset_idx -> _get_observations).
"""
from __future__ import annotations

import torch

from ..runtime import get_runtime


def net_step_done(env) -> torch.Tensor:
    """Step env.isaac_net once and return all False [E] (no termination)."""
    rt = get_runtime(env)
    rt.step()
    return torch.zeros(rt.num_envs, dtype=torch.bool, device=rt.device)
