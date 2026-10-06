"""NetManagerCfg: put the network into a ManagerBasedRLEnvCfg with one call.

    from isaac_net import NRConfig
    from isaac_net.isaac import IsaacNetCfg, NetManagerCfg, NetMarkersCfg

    @configclass
    class MyEnvCfg(ManagerBasedRLEnvCfg):
        ...
        def __post_init__(self):
            ...                                         # decimation, sim.dt, ... first: the network step is checked
            NetManagerCfg(level="L2-legacy", backend="graph", nr=NRConfig(control_step_ms=100.0),
                          isaac=IsaacNetCfg(pose_asset="robots"), markers=NetMarkersCfg()).apply(self)

apply(env_cfg) stores the NetManagerCfg as env_cfg.isaac_net (the terms build env.isaac_net from it on first use,
isaac/runtime.py) and adds, without touching the terms already there:

    terminations.net_step     DoneTerm(net_step_done), moved to the front     (placement="termination", default)
      or events.net_step      EventTerm(net_step, mode="interval", interval_range_s=(step_dt, step_dt),
                              is_global_time=True)                            (placement="interval")
    events.net_reset          EventTerm(net_reset, mode="reset")
    observations.<group>.<t>  ObsTerm(t) for t in obs_terms (default net_aoi, net_sinr, net_queue, net_delivered)
    rewards.net_aoi           RewTerm(net_aoi_penalty, weight=aoi_weight)    (skipped with aoi_weight None)
    actions.net_send          NetSendActionCfg(asset_name=pose_asset)          (send_action=True)

Order inside ManagerBasedRLEnv.step: physics -> terminations -> rewards -> resets (reset events) -> interval events ->
observations. The network must step after physics (end-of-step poses) and before observations; as the first
termination term it also runs before the rewards and before finished envs reset, as the Direct mixin's net_step in
_get_dones does. With placement="interval" the reward sees the previous step and reset envs step once from their
new pose before their first observation (isaac/mdp/events.py).

term_specs() lists the terms as plain tuples, so the wiring is testable without Isaac Lab.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

from .config import IsaacNetCfg
from .marker_geometry import NetMarkersCfg

DEFAULT_OBS_TERMS = ("net_aoi", "net_sinr", "net_queue", "net_delivered")


@dataclass
class NetManagerCfg:
    level: str = "L2-legacy"
    backend: str = "graph"
    num_robots: Optional[int] = None              # None: the bodies of isaac.pose_asset
    nr: Optional[object] = None                   # NRConfig; None: NRConfig() with control_step_ms = env step
    isaac: IsaacNetCfg = field(default_factory=lambda: IsaacNetCfg(pose_asset="robots"))
    seed: Optional[int] = None
    markers: Optional[NetMarkersCfg] = None
    placement: str = "termination"                # "termination" | "interval"
    obs_terms: Sequence[str] = DEFAULT_OBS_TERMS
    obs_group: str = "policy"
    aoi_weight: Optional[float] = -0.1
    send_action: bool = False
    send_classes: int = 2
    traffic_period: int = 1                       # periodic traffic without a send action (isaac/runtime.py)
    traffic_class: int = 1

    def __post_init__(self):
        from .mdp.observations import OBS_TERMS
        assert self.placement in ("termination", "interval"), self.placement
        bad = [t for t in self.obs_terms if t not in OBS_TERMS]
        if bad:
            raise ValueError(f"unknown network observation terms {bad}; one of {sorted(OBS_TERMS)}")

    def setup_kwargs(self) -> dict:
        """Arguments of runtime.net_setup(env, ...)."""
        kw = dict(level=self.level, num_robots=self.num_robots, config=self.nr, backend=self.backend,
                  isaac=self.isaac, markers=self.markers, traffic_period=self.traffic_period,
                  traffic_class=self.traffic_class)
        if self.seed is not None:
            kw["seed"] = self.seed
        return kw

    def obs_dim(self, num_robots: int) -> int:
        """Width the network terms add to the observation group (all robots observed)."""
        from .mdp.observations import TERM_WIDTH
        return num_robots * sum(TERM_WIDTH[t] for t in self.obs_terms)

    def term_specs(self, step_dt: Optional[float] = None) -> list:
        """[(section, name, func, kwargs)] the terms apply() adds; section is "terminations", "events",
        "observations", "rewards" or "actions" (func None for the action term)."""
        from . import mdp
        specs = []
        if self.placement == "termination":
            specs.append(("terminations", "net_step", mdp.net_step_done, {"time_out": False}))
        else:
            assert step_dt is not None, "placement='interval' needs the env step (decimation * sim.dt)"
            specs.append(("events", "net_step", mdp.net_step, {"mode": "interval", "interval_range_s": (step_dt, step_dt),
                                                                "is_global_time": True}))
        specs.append(("events", "net_reset", mdp.net_reset, {"mode": "reset"}))
        for t in self.obs_terms:
            specs.append(("observations", t, mdp.OBS_TERMS[t], {}))
        if self.aoi_weight is not None:
            specs.append(("rewards", "net_aoi", mdp.net_aoi_penalty, {"weight": float(self.aoi_weight)}))
        if self.send_action:
            specs.append(("actions", "net_send", None, {"asset_name": self.isaac.pose_asset,
                                                        "num_classes": self.send_classes,
                                                        "num_robots": self.num_robots}))
        return specs

    def apply(self, env_cfg):
        """Add the network terms to a ManagerBasedRLEnvCfg (module docstring); returns env_cfg."""
        from isaaclab.managers import EventTermCfg, ObservationTermCfg, RewardTermCfg, TerminationTermCfg

        step_dt = float(env_cfg.decimation) * float(env_cfg.sim.dt)
        if self.nr is None:
            from ..core.config import NRConfig
            net_ms = step_dt * 1000.0 * self.isaac.net_decimation / self.isaac.net_substeps
            self.nr = NRConfig(control_step_ms=net_ms)
        env_cfg.isaac_net = self
        for section, name, func, kw in self.term_specs(step_dt):
            if section == "terminations":
                _put_first(env_cfg.terminations, name, TerminationTermCfg(func=func, **kw))
            elif section == "events":
                if env_cfg.events is None:
                    raise ValueError("env_cfg.events is None: give the env an EventCfg (it may be empty)")
                setattr(env_cfg.events, name, EventTermCfg(func=func, **kw))
            elif section == "observations":
                group = getattr(env_cfg.observations, self.obs_group)
                setattr(group, name, ObservationTermCfg(func=func))
            elif section == "rewards":
                setattr(env_cfg.rewards, name, RewardTermCfg(func=func, **kw))
            elif section == "actions":
                from .mdp.actions import NetSendActionCfg
                setattr(env_cfg.actions, name, NetSendActionCfg(**kw))
        return env_cfg


def _put_first(cfg_obj, name: str, term):
    """Set cfg_obj.name = term as the first attribute (the managers iterate cfg.__dict__ in order)."""
    rest = {k: v for k, v in vars(cfg_obj).items() if k != name}
    vars(cfg_obj).clear()
    setattr(cfg_obj, name, term)
    for k, v in rest.items():
        setattr(cfg_obj, k, v)
