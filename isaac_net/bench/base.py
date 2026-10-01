"""NetTask: the base of every benchmark task (pure torch, kinematic 2D robots, network through NetModule).

    task = make_task(TaskConfig(task="coop_map", level="L2-legacy", backend="graph", num_envs=64, seed=0), "cuda")
    obs = task.reset()                                   # [E, R, task.obs_spec.dim]
    obs, rew, done, info = task.step(cont, send)         # cont [E,R,cont_dim] in [-1,1], send [E,R] long
    info["episodes"]                                     # metric rows of the envs whose episode ended this step

A task owns its robots and task state as fixed-shape tensors with leading dims [E, R], draws every random number
from its own generator seeded by TaskConfig.seed, and talks to the network only through a NetModule
(isaac/net_module.py, no Isaac import): submit at the start-of-step pose, step with the end-of-step pose, and the
NetModule's selected observation features appended to the task features. Episodes have a fixed length and every
env ends at the same step; reset(mask) still resets any subset of envs and leaves the others untouched.

Subclasses set the class attributes below and implement
    _task_obs() -> [E, R, D_task]                  task features (their widths in TASK_BLOCKS)
    _task_reset(mask [E] bool)                     redraw the task state of the masked envs
    _step(cont, send) -> reward [E,R], task value [E], extras {name: [E]}
    heuristic() -> cont [E,R,A], send_ready [E,R], send_busy [E,R]   (the scripted baseline, privileged state)
and call self._net_step(...) inside _step for every network step.
"""
from __future__ import annotations

import math
from typing import Optional

import torch

from ..core.config import NRConfig
from ..core.edge import EdgeLoop
from ..isaac.config import IsaacNetCfg
from ..isaac.net_module import NetModule, TrafficRequest
from .metrics import EpisodeMetrics
from .spec import ActionSpec, MetricSpec, ObsSpec, TaskConfig, background_config, nr_preset, traffic_preset


class _Tap:
    """Engine proxy that keeps the raw step dict (EdgeLoop keys, energy) next to what NetModule returns."""

    def __init__(self, eng):
        object.__setattr__(self, "_eng", eng)
        object.__setattr__(self, "last", None)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_eng"), name)

    def __setattr__(self, name, value):
        setattr(object.__getattribute__(self, "_eng"), name, value)

    def step(self, *a, **kw):
        out = object.__getattribute__(self, "_eng").step(*a, **kw)
        object.__setattr__(self, "last", out)
        return out


class NetTask:
    NAME = "base"
    DESCRIPTION = ""
    MECHANISM = ""                     # the endogenous network mechanism the task exercises
    MSG_SIZES = (4000.0, 30000.0)      # bytes per message class
    SEND_CHOICES = ("none", "small", "large")
    CONT_NAMES = ("vx", "vy")
    TASK_BLOCKS = ()                   # ((name, width), ...)
    METRIC = MetricSpec("metric", "", True, "")
    METRIC_REDUCE = "mean"             # "mean": task value averaged over steps; "sum": summed per episode
    EPISODE_STEPS = 300
    CONTROL_STEP_MS = 100.0            # network control step
    NET_SUBSTEPS = 1                   # network steps per task step
    TIMEOUT_STEPS = 20                 # application timeout in network steps
    ARENA = 150.0
    UE_HEIGHT = 0.0
    BLOCKAGE = False                   # the task passes a blocked_fn (coverage holes)
    BLOCKAGE_DB = 20.0
    EDGE = None                        # an EdgeConfig: the task runs through EdgeLoop
    LIGHT_SCALE = 0.1                  # message-size multiplier of the "light" negative-control variant

    def __init__(self, cfg: TaskConfig, device="cpu"):
        self.cfg = cfg
        self.E, self.R = cfg.num_envs, cfg.num_robots
        self.dev = torch.device(device)
        self.T = cfg.episode_steps or self.EPISODE_STEPS
        self.gen = torch.Generator(device=self.dev)
        self.gen.manual_seed(int(cfg.seed) * 7919 + 17)
        self.policy_gen = torch.Generator(device=self.dev)       # draws of the scripted heuristic only
        self.policy_gen.manual_seed(int(cfg.seed) * 7919 + 23)
        self.sizes = tuple(float(s) * cfg.size_scale for s in self.MSG_SIZES)
        self.nr = self.build_nr_config()
        multi = self.nr.n_cells > 1
        if multi and self.BLOCKAGE:
            raise ValueError(f"task {self.NAME} draws coverage holes on the Isaac radio; multi-cell presets use the "
                             "engine's radio. Use a single-cell preset.")
        self.isaac = IsaacNetCfg(radio="engine" if multi else "isaac", blockage=self.BLOCKAGE,
                                 blockage_db=self.BLOCKAGE_DB, obs_features=cfg.net_obs, obs_history=cfg.obs_history,
                                 pose_chunks=1)
        edge = self.nr.edge
        self.net = NetModule(cfg.level, self.E, self.R, self.dev, self.nr.with_(edge=None), cfg.backend,
                             isaac=self.isaac, params=cfg.level_params, seed=int(cfg.seed) * 1000 + 1)
        if edge is not None:
            # the edge stage runs as its own CUDA graph on a GPU (its event loop is launch-bound otherwise)
            graph = self.dev.type == "cuda" and edge.return_path != "nr_dl"
            self.net.eng = EdgeLoop(self.net.eng, edge, graph=graph)
        self.net.eng = _Tap(self.net.eng)
        self.obs_spec = ObsSpec(tuple(self.TASK_BLOCKS) + tuple(
            (f"net:{k}", w) for k, w in self.net.obs_features.dims.items()))
        self.action_spec = ActionSpec(len(self.CONT_NAMES), tuple(self.CONT_NAMES), tuple(self.SEND_CHOICES))
        self.metrics = EpisodeMetrics(self.E, self.R, len(self.SEND_CHOICES), self.dev,
                                      self.nr.control_step_ms / 1000.0, self.sizes)
        self.t = torch.zeros(self.E, dtype=torch.long, device=self.dev)        # task steps since the env's reset
        self.pos = torch.zeros(self.E, self.R, 2, device=self.dev)
        self.last_out: Optional[dict] = None
        self.queue_len = torch.zeros(self.E, self.R, dtype=torch.long, device=self.dev)
        self._all = torch.ones(self.E, dtype=torch.bool, device=self.dev)

    # ------------------------------------------------------------------ configuration
    def build_nr_config(self) -> NRConfig:
        """The preset's NRConfig with the task's application fields, the variant's overrides and the traffic."""
        c = self.cfg
        nr = nr_preset(c.preset).with_(msg_sizes=self.sizes, control_step_ms=float(self.CONTROL_STEP_MS),
                                       timeout_steps=int(self.TIMEOUT_STEPS), seed=int(c.seed) * 1000 + 1)
        tr = traffic_preset(c.traffic)
        if tr is not None:
            nr = nr.with_(traffic=tr)
        over = dict(c.nr_overrides)
        if c.variant == "background":
            over.setdefault("background", background_config(self.NAME))
        if self.EDGE is not None:
            over.setdefault("edge", self.EDGE)
        return nr.with_(**over) if over else nr

    @property
    def step_s(self) -> float:
        return self.CONTROL_STEP_MS * self.NET_SUBSTEPS / 1000.0

    def describe(self) -> dict:
        return {"task": self.NAME, "description": self.DESCRIPTION, "mechanism": self.MECHANISM,
                "metric": {"key": self.METRIC.key, "unit": self.METRIC.unit,
                           "higher_is_better": self.METRIC.higher_is_better, "description": self.METRIC.description},
                "obs": self.obs_spec.as_dict(), "action": self.action_spec.as_dict(),
                "msg_sizes": list(self.sizes), "episode_steps": self.T, "task_step_s": self.step_s,
                "net_control_step_ms": self.CONTROL_STEP_MS, "net_substeps": self.NET_SUBSTEPS,
                "timeout_steps": self.TIMEOUT_STEPS, "edge": repr(self.EDGE) if self.EDGE is not None else None}

    # ------------------------------------------------------------------ random helpers (task generator)
    def rand(self, *shape):
        return torch.rand(*shape, device=self.dev, generator=self.gen)

    def randn(self, *shape):
        return torch.randn(*shape, device=self.dev, generator=self.gen)

    def randint(self, hi, shape):
        return torch.randint(0, hi, shape, device=self.dev, generator=self.gen)

    def prand(self, *shape):
        """Uniform draws for heuristic() (its own generator, so the task's draws do not depend on the policy)."""
        return torch.rand(*shape, device=self.dev, generator=self.policy_gen)

    def where_env(self, mask, new, old):
        """old with the rows of the masked envs replaced by new (mask [E] bool, any trailing dims)."""
        return torch.where(mask.view(-1, *([1] * (old.dim() - 1))), new, old)

    # ------------------------------------------------------------------ API
    def reset(self, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Reset the masked envs (None = all): task state, network, metrics. Returns the observation [E,R,D]."""
        m = self._all if mask is None else mask.to(self.dev)
        self.t = torch.where(m, torch.zeros_like(self.t), self.t)
        self._task_reset(m)
        self.net.reset(None if mask is None else m.nonzero(as_tuple=True)[0])
        self.metrics.reset(m)
        self.queue_len = torch.where(m[:, None], torch.zeros_like(self.queue_len), self.queue_len)
        return self.obs()

    def obs(self) -> torch.Tensor:
        return torch.cat([self._task_obs(), self.net.obs()], -1)

    def step(self, cont: torch.Tensor, send: torch.Tensor):
        """One task step for every env. cont [E,R,cont_dim] (clamped to [-1,1]), send [E,R] long.
        Returns obs [E,R,D], reward [E,R], done [E] bool, info {"episodes": [row, ...]} (ended envs, reset)."""
        cont = cont.to(self.dev).clamp(self.action_spec.low, self.action_spec.high)
        send = send.to(self.dev).long().clamp(0, self.action_spec.n_send - 1)
        self.metrics.add_send(send)
        rew, value, extra = self._step(cont, send)
        self.metrics.add_task_step(rew, value, extra)
        self.t = self.t + 1
        done = self.t >= self.T
        info = {"episodes": []}
        if bool(done.any()):
            info["episodes"] = self.metrics.rows(done, self.METRIC.key, self.METRIC_REDUCE,
                                                 [c.replace(" ", "") for c in self.SEND_CHOICES])
            obs = self.reset(done if not bool(done.all()) else None)
        else:
            obs = self.obs()
        return obs, rew, done, info

    # ------------------------------------------------------------------ network
    def _net_step(self, cls: torch.Tensor, poses_end: torch.Tensor, tag=None, cur_tag=None, blocked_fn=None) -> dict:
        """Submit one message of class cls [E,R] (0 = none) captured now, advance the network one step with the
        end-of-step positions [E,R,2]; returns the NetModule step dict (plus the raw engine keys it drops)."""
        acc = self.net.submit(None, TrafficRequest(cls.long(), tag))
        self.metrics.add_submit(cls, acc)
        p3 = torch.cat([poses_end, torch.full_like(poses_end[..., :1], self.UE_HEIGHT)], -1)
        out = self.net.step(None, p3, cur_tag=cur_tag, blocked_fn=blocked_fn)
        raw = self.net.eng.last
        for k in ("dropped", "energy_j"):
            if k in raw and k not in out:
                out[k] = raw[k]
        for k in raw:
            if k.startswith("act_") or k.startswith("edge_") or k == "cmd_dropped":
                out[k] = raw[k]
        self.metrics.add_net_step(out)
        self.queue_len = out["queue_len"]
        self.last_out = out
        return out

    # ------------------------------------------------------------------ to implement
    def _task_obs(self) -> torch.Tensor:
        raise NotImplementedError

    def _task_reset(self, mask: torch.Tensor):
        raise NotImplementedError

    def _step(self, cont, send):
        raise NotImplementedError

    def heuristic(self):
        raise NotImplementedError

    # ------------------------------------------------------------------ shared kinematics
    def move(self, vel: torch.Tensor, vmax: float, speed: Optional[torch.Tensor] = None) -> torch.Tensor:
        """New positions after a velocity command vel [E,R,2] in [-1,1] (x vmax m per step, x speed [E,R])."""
        v = vel * vmax if speed is None else vel * vmax * speed[..., None]
        return (self.pos + v).clamp(0.0, self.ARENA)

    @staticmethod
    def unit(v: torch.Tensor) -> torch.Tensor:
        return v / v.norm(dim=-1, keepdim=True).clamp(min=1e-6)


def nan_mean(xs):
    v = [x for x in xs if x is not None and math.isfinite(x)]
    return sum(v) / len(v) if v else math.nan
