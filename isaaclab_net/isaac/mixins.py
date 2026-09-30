"""DirectRLEnv mixin: puts a NetModule in the loop with four one-line hook calls.

    class MyEnv(NetEnvMixin, DirectRLEnv):
        def _setup_scene(self):            ...; self.net_setup(net_cfg)
        def _get_dones(self):              ...; out = self.net_step(poses_local, send, tag, cur_tag)
        def _reset_idx(self, env_ids):     super()._reset_idx(env_ids); ...; self.net_reset(env_ids)
        def _get_observations(self):       feats = self.net_obs()        # [E,R,4], zeros where no network

Placement (DirectRLEnv.step on release/3.0.0): _pre_physics_step -> decimation x (_apply_action, sim.step)
-> _get_dones -> _get_rewards -> _reset_idx -> interval events -> _get_observations.
net_step belongs in _get_dones, the first post-physics hook, so pre-reset poses and queues are used.
The module needs no Isaac imports; the only Isaac-specific piece is `rigid_positions_local`.
"""
from __future__ import annotations

from typing import Optional

import torch

from .net_module import NetConfig, NetModule, ParamRanges, TrafficRequest, net_features


def rigid_positions_local(collection, env_origins: torch.Tensor) -> torch.Tensor:
    """[E,R,3] env-local positions of a RigidObjectCollection (3.0 ProxyArray or 2.x torch tensor)."""
    p = collection.data.body_link_pos_w
    p = p.torch if hasattr(p, "torch") else p
    return p - env_origins[:, None, :]


class NetEnvMixin:
    net: Optional[NetModule] = None
    net_out: Optional[dict] = None

    def net_setup(self, cfg: Optional[NetConfig], ranges: Optional[ParamRanges] = None):
        """cfg None = network off (ideal link). Call inside _setup_scene (before the EventManager exists)."""
        self.net = NetModule(cfg, ranges) if cfg is not None else None
        self.net_out = None

    def net_step(self, poses_local: torch.Tensor, send: torch.Tensor, tag: Optional[torch.Tensor] = None,
                 cur_tag: Optional[torch.Tensor] = None, blocked_fn=None) -> Optional[dict]:
        if self.net is None:
            return None
        self.net.submit(None, TrafficRequest(send=send, tag=tag))
        self.net_out = self.net.step(None, poses_local, cur_tag=cur_tag, blocked_fn=blocked_fn)
        return self.net_out

    def net_reset(self, env_ids):
        if self.net is not None:
            self.net.reset(env_ids)

    def net_obs(self) -> torch.Tensor:
        E, R = self.num_envs, self.net.R if self.net is not None else getattr(self, "R", 1)
        if self.net is None or self.net_out is None:
            return torch.zeros(E, R, 4, device=self.device)
        f = net_features(self.net_out, self.net.cfg.step_dt, self.net.cfg.frame_depth)
        fresh = (self.episode_length_buf == 0)[:, None, None]          # reset this step: stats of the old episode
        return torch.where(fresh, torch.zeros_like(f), f)
