"""Isaac Lab variant of the benchmark (TaskConfig(sim="isaac")): Fleet-Alert on the Isaac Lab fleet env.

The torch tasks are the reference implementation of the suite. This adapter runs the same Fleet-Alert task inside
Isaac Lab 3.0 (examples/isaac_fleet_env.py, wired through NetEnvMixin) and exposes it with the NetTask interface,
so the runner, the baselines (random and PPO; the heuristic needs privileged task state and is torch-only) and the
metrics work unchanged. It needs a running Isaac Sim app: launch it with isaaclab.app.AppLauncher before
make_task, as the scripts in benchmarks/isaac/ do. Only fleet_alert exists in Isaac Lab so far.
Status: written against the fleet env's interface, not yet run inside Isaac Lab.
"""
from __future__ import annotations

import torch

from .metrics import EpisodeMetrics
from .spec import ActionSpec, ObsSpec, TaskConfig
from .tasks.fleet_alert import FleetAlert


class IsaacFleetAlert:
    NAME = FleetAlert.NAME
    METRIC = FleetAlert.METRIC
    METRIC_REDUCE = "mean"
    SEND_CHOICES = FleetAlert.SEND_CHOICES

    def __init__(self, cfg: TaskConfig, device="cuda:0"):
        from isaaclab_net.examples.isaac_fleet_env import (TASK_OBS, NetFleetEnv, fleet_isaac_cfg, make_cfg,
                                                           net_config)
        self.cfg = cfg
        self.E, self.R = cfg.num_envs, cfg.num_robots
        isaac = fleet_isaac_cfg(obs_features=cfg.net_obs, obs_history=cfg.obs_history)
        nr = net_config(0.1).with_(msg_sizes=tuple(s * cfg.size_scale for s in FleetAlert.MSG_SIZES),
                                   seed=int(cfg.seed) * 1000 + 1)
        ecfg = make_cfg(self.E, self.R, cfg.level, str(device), cfg.backend, isaac=isaac, nr=nr)
        ecfg.seed = int(cfg.seed)
        ecfg.net_seed = int(cfg.seed) * 1000 + 1
        self.env = NetFleetEnv(ecfg)
        self.dev = torch.device(self.env.device)
        self.T = int(self.env.max_episode_length)
        self.sizes = nr.msg_sizes
        net_dims = isaac.obs_dims(nr)
        blocks = [b for b in FleetAlert.TASK_BLOCKS]
        assert sum(w for _, w in blocks) == TASK_OBS
        self.obs_spec = ObsSpec(tuple(blocks) + tuple((f"net:{k}", w) for k, w in net_dims.items()))
        self.action_spec = ActionSpec(2, FleetAlert.CONT_NAMES, FleetAlert.SEND_CHOICES)
        self.metrics = EpisodeMetrics(self.E, self.R, 3, self.dev, 0.1, self.sizes)
        self.queue_len = torch.zeros(self.E, self.R, dtype=torch.long, device=self.dev)
        self.t = torch.zeros(self.E, dtype=torch.long, device=self.dev)
        self._send_val = torch.tensor([-1.0, 0.0, 1.0], device=self.dev)

    def describe(self) -> dict:
        return {"task": self.NAME, "sim": "isaac", "metric": {"key": self.METRIC.key, "unit": self.METRIC.unit,
                "higher_is_better": self.METRIC.higher_is_better, "description": self.METRIC.description},
                "obs": self.obs_spec.as_dict(), "action": self.action_spec.as_dict(), "msg_sizes": list(self.sizes),
                "episode_steps": self.T}

    def heuristic(self):
        raise NotImplementedError("the heuristic baseline reads privileged task state; it runs on sim='torch'")

    def reset(self, mask=None):
        obs, _ = self.env.reset()
        self.metrics.reset(torch.ones(self.E, dtype=torch.bool, device=self.dev))
        self.t.zero_()
        return obs["policy"].view(self.E, self.R, -1)

    def step(self, cont, send):
        E, R = self.E, self.R
        send = send.long().clamp(0, 2)
        act = torch.cat([cont.clamp(-1, 1), self._send_val[send][..., None]], -1).reshape(E, R * 3)
        self.metrics.add_send(send)
        env = self.env
        h_on, h_pos, radius = env.h_on.clone(), env.h_pos.clone(), None
        obs, rew, term, trunc, _ = env.step(act)
        out = env.net_out
        if out is not None:
            self.metrics.add_submit(send, None)
            self.metrics.add_net_step(out)
            self.queue_len = out["queue_len"]
        pos = env._pos_radio()
        radius = ((self.t - env.h_start + 1).float() * FleetAlert.H_GROW).clamp(max=FleetAlert.H_R) * h_on.float()
        inside = h_on[:, None] & ((pos - h_pos[:, None, :]).norm(dim=-1) < radius[:, None])
        rew_r = rew[:, None].expand(E, R)
        self.metrics.add_task_step(rew_r, inside.float().mean(-1))
        self.t += 1
        done = term | trunc
        info = {"episodes": []}
        if bool(done.any()):
            info["episodes"] = self.metrics.rows(done, self.METRIC.key, "mean", list(self.SEND_CHOICES))
            self.metrics.reset(done)
            self.t = torch.where(done, torch.zeros_like(self.t), self.t)
        return obs["policy"].view(E, R, -1), rew_r, done, info


def make_isaac_task(cfg: TaskConfig, device="cuda:0"):
    if cfg.task != "fleet_alert":
        raise NotImplementedError(f"sim='isaac' has only fleet_alert so far, not {cfg.task!r}")
    return IsaacFleetAlert(cfg, device)
