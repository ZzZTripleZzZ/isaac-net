"""DirectRLEnv mixin: puts a NetModule in the loop with four hook calls.

    class MyEnv(NetEnvMixin, DirectRLEnv):
        def _setup_scene(self):          ...; self.net_setup("L2-legacy", R, NRConfig(), backend="triton")
        def _pre_physics_step(self, a):  ...; capture frames at the START-of-step pose (task code)
        def _get_dones(self):            ...; out = self.net_step(poses_end, send, tag, cur_tag)
        def _reset_idx(self, env_ids):   super()._reset_idx(env_ids); ...; self.net_reset(env_ids)
        def _get_observations(self):     feats = self.net_obs()        # [E,R,4], zeros where no network

Placement (DirectRLEnv.step on release/3.0.0): _pre_physics_step -> decimation x (_apply_action, sim.step)
-> _get_dones -> _get_rewards -> _reset_idx -> interval events -> _get_observations.
net_step belongs in _get_dones, the first post-physics hook, so it sees the pre-reset queues and the END-of-step
poses. Frames are captured at the start-of-step pose (read it in _pre_physics_step), so a message captured at
env clock t reflects the world at t, and its delay and the AoI are not optimistic by one control step.
The module needs no Isaac imports; the only Isaac-specific piece is `rigid_positions_local`.
"""
from __future__ import annotations

from typing import Optional

import torch

from .net_module import NetConfig, NetModule, TrafficRequest, net_features


def rigid_positions_local(collection, env_origins: torch.Tensor) -> torch.Tensor:
    """[E,R,3] env-local positions of a RigidObjectCollection (3.0 ProxyArray or 2.x torch tensor)."""
    p = collection.data.body_link_pos_w
    p = p.torch if hasattr(p, "torch") else p
    return p - env_origins[:, None, :]


class NetEnvMixin:
    net: Optional[NetModule] = None
    net_out: Optional[dict] = None

    def net_setup(self, level, num_robots: Optional[int] = None, config=None, backend: str = "reference",
                  **kwargs):
        """Build the network for this env batch; call inside _setup_scene.

        level: None or "off" (ideal link, no network features), or a make_engine level ("L0", "L0DR", "L1",
        "L2-legacy", "L2", ...). config: NRConfig. backend: "reference" | "eager" | "graph" | "compile" |
        "triton". kwargs go to NetModule (pose_chunks, gnb_pos, radio, ranges, params, seed).
        A NetConfig (the demo's configuration) is also accepted as `level`.
        """
        self.net_out = None
        if level is None or level == "off":
            self.net = None
            return
        if isinstance(level, NetConfig):
            self.net = NetModule(level, kwargs.get("ranges"))
            return
        scene = getattr(self, "scene", None)
        E = scene.num_envs if scene is not None else self.num_envs
        self.net = NetModule(level, E, num_robots, str(self.device), config, backend, **kwargs)

    def net_step(self, poses_end: torch.Tensor, send: torch.Tensor, tag: Optional[torch.Tensor] = None,
                 cur_tag: Optional[torch.Tensor] = None, blocked_fn=None) -> Optional[dict]:
        """Submit this step's messages (captured at the start-of-step pose) and advance the network by one control
        step with the END-of-step poses [E,R,3]. Returns the NetModule output dict, or None without a network."""
        if self.net is None:
            return None
        self.net.submit(None, TrafficRequest(send=send, tag=tag))
        self.net_out = self.net.step(None, poses_end, cur_tag=cur_tag, blocked_fn=blocked_fn)
        return self.net_out

    def net_reset(self, env_ids):
        if self.net is not None:
            self.net.reset(env_ids)

    def net_obs(self) -> torch.Tensor:
        """[E,R,4] network features (AoI, SNR, queued frames, delivered); zeros without a network, and zeros for
        envs that reset this step (their last output belongs to the previous episode)."""
        E = self.num_envs
        R = self.net.R if self.net is not None else getattr(self, "R", 1)
        if self.net is None or self.net_out is None:
            return torch.zeros(E, R, 4, device=self.device)
        f = net_features(self.net_out, self.net.step_dt, self.net.F)
        fresh = (self.episode_length_buf == 0)[:, None, None]
        return torch.where(fresh, torch.zeros_like(f), f)
