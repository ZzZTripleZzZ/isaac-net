"""The network of a manager-based Isaac Lab env: env.isaac_net, a NetRuntime around NetModule (no Isaac imports).

    rt = net_setup(env, "L2-legacy", num_robots=16, config=NRConfig(...), backend="graph",
                   isaac=IsaacNetCfg(pose_asset="robots"), markers=NetMarkersCfg())   # or lazily from env.cfg.isaac_net
    rt.send[:] = classes          # [E,R] messages of this step (the NetSendAction term writes it), else periodic traffic
    rt.step()                     # one network step at the end-of-step poses (the net_step term calls it)
    rt.reset(env_ids)             # the net_reset event term
    rt.out, rt.net                # the last NetModule.step dict, the NetModule

NetRuntime reuses NetEnvMixin (multi-rate, pose source, markers) with the env's scene, device and step_dt, so the
Direct and the manager-based workflows run the same code. The manager terms in isaac/mdp/ find it through
get_runtime(env): env.isaac_net if it exists, else it is built on first use from env.cfg.isaac_net (a NetManagerCfg,
isaac/manager_cfg.py). The first use is the observation manager probing its term shapes, after the scene is created
and the simulation reset, so the pose asset can be read.

Traffic: without a send action, every robot submits a message of class `traffic_class` every `traffic_period`
network steps of its env (periodic status updates), so AoI and delays are defined for any task. A NetSendAction
term (isaac/mdp/actions.py) or task code writes rt.send instead; rt.send is consumed (zeroed) by every step.
"""
from __future__ import annotations

from typing import Optional

import torch

from ..core.proto.netsim import env_index
from .config import IsaacNetCfg
from .mixins import NetEnvMixin


class NetRuntime(NetEnvMixin):
    """NetEnvMixin bound to a manager-based env (module docstring)."""

    def __init__(self, env, level="L2-legacy", num_robots: Optional[int] = None, config=None, backend: str = "graph",
                 isaac: Optional[IsaacNetCfg] = None, markers=None, traffic_period: int = 1, traffic_class: int = 1,
                 tag_fn=None, cur_tag_fn=None, **kwargs):
        self.env = env
        isc = isaac if isaac is not None else IsaacNetCfg(pose_asset="robots")
        if num_robots is None:
            num_robots = infer_num_robots(env, isc)
        self.traffic_period, self.traffic_class = int(traffic_period), int(traffic_class)
        self.tag_fn, self.cur_tag_fn = tag_fn, cur_tag_fn
        self.send_from_action = False
        self.steps = 0
        self._last_counter = None
        self.net_setup(level, num_robots, config, backend, isaac=isc, markers=markers, **kwargs)
        E, R = self.num_envs, num_robots
        self.send = torch.zeros(E, R, dtype=torch.long, device=self.device)
        self.last_send = torch.zeros_like(self.send)                          # what the last step submitted
        self.fresh = torch.zeros(E, dtype=torch.bool, device=self.device)     # stepped since the env's reset

    # env attributes NetEnvMixin reads
    @property
    def scene(self):
        return getattr(self.env, "scene", None)

    @property
    def sim(self):
        """The env's simulation context (NetMarkers asks it whether anything displays the stage)."""
        return getattr(self.env, "sim", None)

    @property
    def num_envs(self) -> int:
        return int(self.env.num_envs)

    @property
    def device(self):
        return self.env.device

    @property
    def step_dt(self):
        return getattr(self.env, "step_dt", None)

    @property
    def out(self) -> Optional[dict]:
        return self.net_out

    @property
    def R(self) -> int:
        return int(self._net_R)

    # ------------------------------------------------------------------ hooks for the manager terms
    def traffic(self) -> torch.Tensor:
        """[E,R] messages of this step: rt.send if a send action (or the task) wrote it, else periodic traffic."""
        if self.send_from_action or bool(getattr(self, "_send_written", False)):
            return self.send
        if self.net is None:
            return self.send
        due = (self.net.clock % self.traffic_period) == 0                    # [E] per-env network clock
        return torch.where(due[:, None], torch.full_like(self.send, self.traffic_class), torch.zeros_like(self.send))

    def write_send(self, send: torch.Tensor):
        """Messages of this step [E,R] long (0 = none, c = class c), e.g. from an action term."""
        self.send.copy_(send.to(self.send.device, torch.long))
        self._send_written = True

    def step(self, poses: Optional[torch.Tensor] = None) -> Optional[dict]:
        """One env step of the network at the end-of-step poses (None: IsaacNetCfg.pose_asset). Runs at most once
        per env step (env.common_step_counter), so placing net_step twice does not double-step."""
        counter = getattr(self.env, "common_step_counter", None)
        if counter is not None and counter == self._last_counter:
            return self.net_out
        self._last_counter = counter
        if self.net is None:
            return None
        send = self.traffic()
        tag = self.tag_fn(self.env) if self.tag_fn is not None else None
        cur = self.cur_tag_fn(self.env) if self.cur_tag_fn is not None else None
        out = self.net_step(poses, send, tag, cur_tag=cur)
        self.last_send.copy_(send)
        self.send.zero_()
        self._send_written = False
        self.fresh.fill_(True)
        self.steps += 1
        return out

    def reset(self, env_ids=None):
        """Partial reset of env_ids (None, slice(None), indices of any int dtype, or a bool mask [E])."""
        ids = env_index(env_ids, self.num_envs, self.fresh.device)
        if ids is not None and ids.numel() == 0:
            return
        self.net_reset(ids)
        if ids is None:
            self.fresh.fill_(False)
        else:
            self.fresh[ids] = False


def infer_num_robots(env, isaac: IsaacNetCfg) -> int:
    """Number of robots = bodies of IsaacNetCfg.pose_asset (or len(pose_body_ids))."""
    if isaac.pose_body_ids is not None:
        return len(isaac.pose_body_ids)
    if isaac.pose_asset is None:
        raise ValueError("pass num_robots, or set IsaacNetCfg.pose_asset so it can be read from the scene")
    p = env.scene[isaac.pose_asset].data.body_link_pos_w
    p = p.torch if hasattr(p, "torch") else p
    return int(p.shape[1]) if p.dim() == 3 else 1


def net_setup(env, level="L2-legacy", num_robots: Optional[int] = None, config=None, backend: str = "graph",
              isaac: Optional[IsaacNetCfg] = None, markers=None, **kwargs) -> NetRuntime:
    """Create env.isaac_net (a NetRuntime) unless it exists; returns it. Call it from the env's __init__ after
    super().__init__, or let the manager terms build it lazily from env.cfg.isaac_net."""
    rt = getattr(env, "isaac_net", None)
    if rt is None:
        rt = NetRuntime(env, level, num_robots, config, backend, isaac=isaac, markers=markers, **kwargs)
        env.isaac_net = rt
    return rt


def get_runtime(env) -> NetRuntime:
    """env.isaac_net, built on first use from env.cfg.isaac_net (a NetManagerCfg)."""
    rt = getattr(env, "isaac_net", None)
    if rt is not None:
        return rt
    mc = getattr(getattr(env, "cfg", None), "isaac_net", None)
    if mc is None:
        raise RuntimeError("no network on this env: call isaac_net.isaac.mdp.net_setup(env, ...) or add a "
                           "NetManagerCfg to the env cfg (NetManagerCfg(...).apply(env_cfg))")
    return net_setup(env, **mc.setup_kwargs())
