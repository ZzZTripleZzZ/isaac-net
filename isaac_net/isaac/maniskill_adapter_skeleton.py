"""Second-backend sketch: the same NetModule inside ManiSkill3 (SAPIEN/PhysX GPU). NOT RUN.

Purpose: show the module is backend-agnostic. Only three touch points change:
  pose source  : ManiSkill Actor.pose.p (torch [E,3] per actor; stack R actors -> [E,R,3])
  reset hook   : _initialize_episode(env_idx, options)  (ManiSkill partial reset via
                 env.reset(options={"env_idx": ids}))
  step hook    : _after_control_step()  (runs once per control step after sim_steps_per_control
                 physics steps; sim_freq / control_freq plays the role of Isaac Lab decimation)
Hook names checked against mani_skill/envs/sapien_env.py on main, 2026-09-29.
Env origins: ManiSkill GPU sim keeps each sub-scene in its own frame, so poses are already
env-local (verify for the chosen scene builder).

MuJoCo Playground variant (JAX): wrap NetModule calls with jax.dlpack <-> torch.utils.dlpack
inside a host callback, or port the ~300 lines of NetModule to jnp; the reset contract maps to
`jnp.where(done[:, None], init, state)` in the auto-reset wrapper.
"""
from __future__ import annotations

import torch

from mani_skill.envs.sapien_env import BaseEnv

from isaac_net import NRConfig
from isaac_net.isaac import MessageHistory, NetModule, TrafficRequest, net_features

NUM_ROBOTS = 16


class NetFleetManiSkill(BaseEnv):
    """Fleet of R actors; obs = stale server view + network features, same as the Isaac env."""

    def _load_scene(self, options: dict):
        # build R simple actors (e.g. scene.create_actor_builder().add_cylinder...) per sub-scene
        self.robots = [...]   # list of R batched Actor objects, each with .pose.p [E,3]
        E, dev = self.num_envs, self.device
        cfg = NRConfig(control_step_ms=1000.0 / self.control_freq)   # L2 (NR engine) accepts any control step
        self.net = NetModule("L2", E, NUM_ROBOTS, str(dev), cfg)
        self.uplink = MessageHistory(E, NUM_ROBOTS, 3, history_len=32, device=str(dev))
        self._net_out = None

    def _poses(self) -> torch.Tensor:
        return torch.stack([a.pose.p for a in self.robots], dim=1)       # [E,R,3]

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        # ... place actors for env_idx ...
        self.net.sample_params(env_idx)          # DR at reset (no EventTerm system here)
        self.net.reset(env_idx)
        self.uplink.reset(env_idx, self._poses()[env_idx])

    def _before_control_step(self):
        self._pos0 = self._poses()                    # capture at the start-of-step pose

    def _after_control_step(self):
        self.uplink.push(self.net.clock, self._pos0)
        send = torch.ones(self.num_envs, NUM_ROBOTS, dtype=torch.long, device=self.device)
        self.net.submit(None, TrafficRequest(send=send))
        self._net_out = self.net.step(None, self._poses())      # END-of-step poses
        self.uplink.update(self._net_out["newest_cap"])

    def _get_obs_extra(self, info: dict):
        feats = (net_features(self._net_out, self.net.step_dt, self.net.F) if self._net_out is not None
                 else torch.zeros(self.num_envs, NUM_ROBOTS, 4, device=self.device))
        return dict(server_view=self.uplink.seen.flatten(1), net=feats.flatten(1))
