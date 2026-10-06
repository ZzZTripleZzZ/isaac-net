"""RewardTerm functions for the manager-based workflow (no Isaac imports).

    from isaaclab.managers import RewardTermCfg as RewTerm
    aoi = RewTerm(func=net_mdp.net_aoi_penalty, weight=-1.0)

The reward manager multiplies by the weight and the env step, so a negative weight penalizes stale information.
"""
from __future__ import annotations

import torch

from ..runtime import get_runtime
from .observations import feature


def net_aoi_penalty(env, asset_cfg=None) -> torch.Tensor:
    """[E] mean over robots of the normalized age of information (AoI / time scale, clamped to [0, 1]) after the
    last network step; 0 for envs that have not stepped since their reset. Use a negative weight."""
    rt = get_runtime(env)
    if rt.net is None:
        return torch.zeros(rt.num_envs, device=rt.device)
    net = rt.net
    x = feature(rt.out, "net_aoi", rt.num_envs, rt.R, net.F, net.isaac.time_scale_s(net.config), rt.fresh,
                net.dev)[..., 0]
    ids = getattr(asset_cfg, "body_ids", None) if asset_cfg is not None else None
    if isinstance(ids, (list, tuple)):
        x = x[:, list(ids)]
    return x.mean(-1)


def net_send_cost(env, asset_cfg=None) -> torch.Tensor:
    """[E] fraction of robots that submitted a message in the last network step (a transmit-energy cost; use a
    negative weight). Pairs with net_aoi_penalty in scheduling tasks: sending lowers AoI but costs energy and
    contends for the uplink."""
    rt = get_runtime(env)
    x = (rt.last_send > 0).float() * rt.fresh.view(-1, 1).float()
    ids = getattr(asset_cfg, "body_ids", None) if asset_cfg is not None else None
    if isinstance(ids, (list, tuple)):
        x = x[:, list(ids)]
    return x.mean(-1)
