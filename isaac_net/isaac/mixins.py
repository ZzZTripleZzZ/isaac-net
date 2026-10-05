"""DirectRLEnv mixin: puts a NetModule in the loop with four hook calls.

    class MyEnv(NetEnvMixin, DirectRLEnv):
        def _setup_scene(self):          ...; self.net_setup("L2-legacy", R, NRConfig(), "triton", IsaacNetCfg(...))
        def _pre_physics_step(self, a):  ...; capture frames at the START-of-step pose (task code)
        def _get_dones(self):            ...; out = self.net_step(poses_end, send, tag, cur_tag)
        def _reset_idx(self, env_ids):   super()._reset_idx(env_ids); ...; self.net_reset(env_ids)
        def _get_observations(self):     feats = self.net_obs()        # [E,R,obs_dim], zeros without a network

Configuration: the NRConfig (the network, as make_engine takes it), the level and the backend, and an IsaacNetCfg
for the Isaac-specific settings. With IsaacNetCfg.pose_asset set, net_step reads the end-of-step poses from that
scene entity itself (poses_end=None). The env config sizes its observation space with IsaacNetCfg.obs_dim(nr).

Multi-rate: IsaacNetCfg.net_decimation = k steps the network every k env control steps (network step = k env
steps; the messages of the k env steps are merged into one per robot, keeping the largest class and its tag);
net_substeps = m runs m network steps per env control step (network step = env step / m; the messages go into the
first substep, and the poses are interpolated). net_setup checks the env step against NRConfig.control_step_ms.
The network steps on env steps 1, k + 1, 2k + 1, ... of the run; in between (k > 1) net_step returns the last
output with the per-step flags cleared and aoi_s advanced by one env step per skipped tick, and the observation
keeps its last value. The merge keeps the largest class; on equal class it takes a tagged message over an
untagged one, so a detection tag is not lost. net_reset(env_ids) also clears the held output of those envs
(last_cap 0, aoi_s 0, empty queue), since their next network step is up to k - 1 env steps away. With m > 1 the returned
delivered, newest_cap and tag_delivered cover all m substeps, the rest (and the observation, except
delay_history, which collects every substep) is that of the last substep.

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

from ..core.proto.netsim import env_index
from .config import IsaacNetCfg
from .net_module import NetConfig, NetModule, TrafficRequest

_FLAGS = ("delivered", "tag_delivered", "msg_delivered", "timed_out")


def rigid_positions_local(collection, env_origins: torch.Tensor, body_ids=None) -> torch.Tensor:
    """[E,R,3] env-local body positions of a RigidObjectCollection, Articulation or RigidObject (3.0 ProxyArray or
    2.x torch tensor). body_ids selects bodies (None = all)."""
    p = collection.data.body_link_pos_w
    p = p.torch if hasattr(p, "torch") else p
    if body_ids is not None:
        p = p[:, list(body_ids)]
    return p - env_origins[:, None, :]


class NetEnvMixin:
    net: Optional[NetModule] = None
    net_out: Optional[dict] = None

    def net_setup(self, level, num_robots: Optional[int] = None, config=None, backend: str = "reference",
                  isaac: Optional[IsaacNetCfg] = None, **kwargs):
        """Build the network for this env batch; call inside _setup_scene.

        level: None or "off" (ideal link, no network features), or a make_engine level ("L0", "L0DR", "L1",
        "L2-legacy", "L2", ...). config: NRConfig. backend: "reference" | "eager" | "graph" | "compile" |
        "triton". isaac: IsaacNetCfg. kwargs go to NetModule (params, seed, strict, ranges, and IsaacNetCfg
        fields as shortcuts). A NetConfig (deprecated) is also accepted as `level`.
        """
        self.net_out = None
        self._net_isaac = isaac if isaac is not None else IsaacNetCfg()
        self._net_R = num_robots
        self._net_cfg = config
        self._net_tick = 0
        if level is None or level == "off":
            self.net = None
            return
        if isaac is not None and getattr(isaac, "scene_map", None) is not None:
            from .scene_map import resolve_scene_map        # radio map from the stage (docs/scene-radio-map.md)
            config, isaac = resolve_scene_map(config, isaac)
            self._net_cfg = config
        if isinstance(level, NetConfig):
            self.net = NetModule(level, kwargs.get("ranges"))
        else:
            scene = getattr(self, "scene", None)
            E = scene.num_envs if scene is not None else self.num_envs
            self.net = NetModule(level, E, num_robots, str(self.device), config, backend, isaac=isaac, **kwargs)
        self._net_isaac = self.net.isaac
        self._net_check_rate()
        E, R = self.net.E, self.net.R
        self._pend_send = torch.zeros(E, R, dtype=torch.long, device=self.net.dev)
        self._pend_tag = torch.full((E, R), -1, dtype=torch.long, device=self.net.dev)

    def _net_check_rate(self):
        """network control step == env control step * net_decimation / net_substeps."""
        env_dt = getattr(self, "step_dt", None)
        if env_dt is None:
            return
        c = self.net.isaac
        env_ms, net_ms = float(env_dt) * 1000.0, self.net.config.control_step_ms
        if abs(env_ms * c.net_decimation - net_ms * c.net_substeps) > 1e-6 * max(env_ms, net_ms):
            raise ValueError(
                f"the env control step is {env_ms:g} ms and NRConfig.control_step_ms is {net_ms:g} ms, but "
                f"IsaacNetCfg has net_decimation={c.net_decimation}, net_substeps={c.net_substeps}. Set "
                "control_step_ms to the env step, or net_decimation = network step / env step, or net_substeps = "
                "env step / network step.")

    def net_poses(self) -> torch.Tensor:
        """[E,R,3] end-of-step poses in radio coordinates from IsaacNetCfg.pose_asset (plus pose_offset_m)."""
        c = self._net_isaac
        if c.pose_asset is None:
            raise ValueError("pass poses to net_step, or set IsaacNetCfg.pose_asset")
        p = rigid_positions_local(self.scene[c.pose_asset], self.scene.env_origins, c.pose_body_ids)
        return p + torch.as_tensor(c.pose_offset_m, dtype=p.dtype, device=p.device)

    def net_step(self, poses_end: Optional[torch.Tensor] = None, send: Optional[torch.Tensor] = None,
                 tag: Optional[torch.Tensor] = None, cur_tag: Optional[torch.Tensor] = None,
                 blocked_fn=None) -> Optional[dict]:
        """Submit this step's messages (captured at the start-of-step pose) and advance the network by one env
        control step with the END-of-step poses [E,R,3] (None: read from IsaacNetCfg.pose_asset). Returns the
        NetModule output dict, or None without a network."""
        if self.net is None:
            return None
        net, c = self.net, self.net.isaac
        if poses_end is None:
            poses_end = self.net_poses()
        if send is None:
            send = torch.zeros(net.E, net.R, dtype=torch.long, device=net.dev)
        if tag is None:
            tag = torch.full_like(send, -1)
        k, m = c.net_decimation, c.net_substeps
        if k > 1:
            # merge the messages of the window: one per robot, the largest class and its tag; on equal class a
            # tagged message replaces an untagged one (a detection must not be dropped by a same-size frame)
            take = (send > self._pend_send) | ((send == self._pend_send) & (tag >= 0) & (self._pend_tag < 0))
            self._pend_send = torch.where(take, send, self._pend_send)
            self._pend_tag = torch.where(take, tag, self._pend_tag)
            self._net_tick += 1
            if (self._net_tick - 1) % k != 0:              # the network steps on env steps 1, k + 1, 2k + 1, ...
                if self.net_out is not None:
                    held = dict(self.net_out)
                    for f in _FLAGS:
                        if f in held:
                            held[f] = torch.zeros_like(held[f])
                    held["newest_cap"] = torch.full_like(held["newest_cap"], -1)
                    held["aoi_s"] = held["aoi_s"] + net.step_dt / k     # one env step = network step / k
                    self.net_out = held
                return self.net_out
            send, tag = self._pend_send, self._pend_tag
            self._pend_send = torch.zeros_like(send)
            self._pend_tag = torch.full_like(tag, -1)
        if m == 1:
            net.submit(None, TrafficRequest(send=send, tag=tag))
            self.net_out = net.step(None, poses_end, cur_tag=cur_tag, blocked_fn=blocked_fn)
            return self.net_out
        # m network steps within this env step: messages go into the first, poses are interpolated
        if poses_end.shape[-1] == 2:
            poses_end = torch.cat([poses_end, torch.zeros_like(poses_end[..., :1])], -1)
        prev = torch.where(net._prev_valid[:, None, None], net._prev, poses_end)
        agg = None
        for i in range(m):
            p = prev + ((i + 1) / m) * (poses_end - prev)
            net.submit(None, TrafficRequest(send=send if i == 0 else torch.zeros_like(send),
                                            tag=tag if i == 0 else torch.full_like(tag, -1)))
            o = net.step(None, p, cur_tag=cur_tag, blocked_fn=blocked_fn)
            if agg is None:
                agg = o
            else:
                d, nc = agg["delivered"] | o["delivered"], torch.maximum(agg["newest_cap"], o["newest_cap"])
                td = (agg["tag_delivered"] | o["tag_delivered"]) if "tag_delivered" in o else None
                agg = dict(o, delivered=d, newest_cap=nc)
                if td is not None:
                    agg["tag_delivered"] = td
        self.net_out = agg
        return agg

    def net_reset(self, env_ids):
        if self.net is not None:
            self.net.reset(env_ids)
            if self.net.isaac.net_decimation > 1:
                self._pend_send[env_ids] = 0
                self._pend_tag[env_ids] = -1
                if self.net_out is not None:
                    # the held output of a reset env belongs to its previous episode; until its next network
                    # step it reports the reset state (as NetModule does: capture 0 known, empty queue)
                    ids = env_index(env_ids, self.net.E, self.net.dev)
                    m = torch.ones(self.net.E, dtype=torch.bool, device=self.net.dev)
                    if ids is not None:
                        m = torch.zeros_like(m).index_fill_(0, ids, True)
                    held = dict(self.net_out)
                    for f in ("last_cap", "aoi_s", "queue_len", "queue_bytes"):
                        if f in held:
                            x = held[f]
                            held[f] = torch.where(m.view(-1, *([1] * (x.dim() - 1))), torch.zeros_like(x), x)
                    self.net_out = held

    def net_obs(self) -> torch.Tensor:
        """[E,R,obs_dim] the IsaacNetCfg.obs_features of the last network step; zeros without a network, before the
        first step, and for envs that reset since (their last output belongs to the previous episode)."""
        if self.net is not None:
            return self.net.obs()
        E = self.num_envs
        R = self._net_R if getattr(self, "_net_R", None) is not None else getattr(self, "R", 1)
        return torch.zeros(E, R, self._net_isaac.obs_dim(getattr(self, "_net_cfg", None)), device=self.device)
