"""EdgeControl: edge-offloaded tracking control. Robots upload their state; the edge computes the velocity command."""
from __future__ import annotations

import math

import torch

from ...core.config import EdgeConfig
from ..base import NetTask
from ..spec import MetricSpec


class EdgeControl(NetTask):
    NAME = "edge_control"
    DESCRIPTION = (
        "Each robot tracks its own moving target (Ornstein-Uhlenbeck velocity, 2 m/s per axis, walls reflect). The "
        "policy acts every 100 ms; the robot, the network and the edge run on 20 ms steps. The policy picks the "
        "state upload rate (10, 25 or 50 Hz) and two controller parameters carried in every state message, the gain "
        "k in [0.5, 5] 1/s and the target prediction horizon h in [0, 0.5] s. The edge (2 servers per env, 2 ms "
        "per message, FIFO) computes u = v_tgt + k (p_tgt + h v_tgt - p_robot), clipped to 4 m/s, from the "
        "delivered state, and the command returns over the downlink (2 ms plus the transmission time at the "
        "robot's SINR). The robot holds its newest command, also after it is older than 0.5 s (hold-last).")
    MECHANISM = ("Closed-loop latency under the team's own load: a higher upload rate gives fresher commands until "
                 "the uplink and the shared edge servers queue up, and then every robot's loop latency grows.")
    MSG_SIZES = (300.0,)
    SEND_CHOICES = ("10Hz", "25Hz", "50Hz")
    CONT_NAMES = ("gain", "horizon")
    TASK_BLOCKS = (("pos", 2), ("tracking_error", 2), ("command", 2), ("rate", 3), ("command_age", 1),
                   ("stale", 1))
    METRIC = MetricSpec("tracking_error_m", "m", False, "mean distance between robot and target (lower is better)")
    CONTROL_STEP_MS = 20.0
    NET_SUBSTEPS = 5
    TIMEOUT_STEPS = 25
    EDGE = EdgeConfig(servers_per_env=2, discipline="fifo", service_ms=2.0, queue_cap=64, deadline_ms=200.0,
                      return_path="delay", ret_fixed_ms=2.0, cmd_bytes=100)
    DT = 0.02
    VMAX = 4.0
    PERIODS = (5, 2, 1)
    CMD_TIMEOUT = 25
    STALE = "hold"
    HIST = 64
    TH, SIG = 0.5, 2.0

    def __init__(self, cfg, device="cpu"):
        super().__init__(cfg, device)
        E, R, d, H = self.E, self.R, self.dev, self.HIST
        self.periods = torch.tensor(self.PERIODS, device=d)
        self.tpos = torch.zeros(E, R, 2, device=d)
        self.tvel = torch.zeros(E, R, 2, device=d)
        self.cmd = torch.zeros(E, R, 2, device=d)
        self.phase = torch.zeros(E, R, dtype=torch.long, device=d)
        self.prev_rate = torch.zeros(E, R, dtype=torch.long, device=d)
        self.act_age = torch.full((E, R), float(self.CMD_TIMEOUT + 1), device=d)
        self.h = {k: torch.zeros(E, R, H, 2, device=d) for k in ("rp", "tp", "tv")}
        self.h_k = torch.ones(E, R, H, device=d)
        self.h_h = torch.zeros(E, R, H, device=d)

    def _task_reset(self, m):
        E, R, L = self.E, self.R, self.ARENA
        tpos = 20 + (L - 40) * self.rand(E, R, 2)
        self.tpos = self.where_env(m, tpos, self.tpos)
        self.tvel = self.where_env(m, self.SIG / math.sqrt(2 * self.TH) * self.randn(E, R, 2), self.tvel)
        self.pos = self.where_env(m, tpos + self.randn(E, R, 2), self.pos)
        self.cmd = self.where_env(m, torch.zeros_like(self.cmd), self.cmd)
        self.phase = torch.where(m[:, None], self.randint(10, (E, R)), self.phase)
        self.prev_rate = torch.where(m[:, None], torch.zeros_like(self.prev_rate), self.prev_rate)
        self.act_age = torch.where(m[:, None], torch.full_like(self.act_age, self.CMD_TIMEOUT + 1), self.act_age)

    def _task_obs(self):
        err = ((self.tpos - self.pos) / 5.0).clamp(-3, 3)
        age = self.act_age.clamp(max=self.CMD_TIMEOUT) / self.CMD_TIMEOUT
        oh = torch.nn.functional.one_hot(self.prev_rate, 3).float()
        stale = (self.act_age > self.CMD_TIMEOUT).float()
        return torch.cat([self.pos / self.ARENA, err, self.cmd / self.VMAX, oh, age[..., None], stale[..., None]], -1)

    def _step(self, cont, rate):
        E, R, d, H = self.E, self.R, self.dev, self.HIST
        k = 0.5 * 10 ** ((cont[..., 0] + 1) / 2)
        hz = (cont[..., 1] + 1) / 2 * 0.5
        period = self.periods[rate]
        e = torch.arange(E, device=d)[:, None]
        r = torch.arange(R, device=d)[None, :]
        err_sum = torch.zeros(E, R, device=d)
        stale_sum = torch.zeros(E, R, device=d)
        age_sum = torch.zeros(E, R, device=d)
        drops = torch.zeros(E, device=d)
        for _ in range(self.NET_SUBSTEPS):
            g = self.net.clock                                          # [E] network clock of each env
            s = (g % H)[:, None].expand(E, R)
            for key, v in (("rp", self.pos), ("tp", self.tpos), ("tv", self.tvel)):
                self.h[key][e, r, s] = v
            self.h_k[e, r, s] = k
            self.h_h[e, r, s] = hz
            send = ((g[:, None] + self.phase) % period == 0).long()
            stale = self.act_age > self.CMD_TIMEOUT
            v = self.cmd if self.STALE == "hold" else torch.where(stale[..., None], 0.0, self.cmd)
            new_pos = (self.pos + v * self.DT).clamp(0, self.ARENA)
            out = self._net_step(send, new_pos)
            self.pos = new_pos
            new = out["act_new"]
            cs = out["act_cap"].clamp(min=0) % H
            gi = lambda x: x[e, r, cs]
            tv = gi(self.h["tv"])
            u = tv + gi(self.h_k)[..., None] * (gi(self.h["tp"]) + gi(self.h_h)[..., None] * tv - gi(self.h["rp"]))
            u = u * (self.VMAX / u.norm(dim=-1, keepdim=True).clamp(min=self.VMAX))
            self.cmd = torch.where(new[..., None], u, self.cmd)
            self.act_age = out["act_age"].float().nan_to_num(float(self.CMD_TIMEOUT + 1))
            # target: OU velocity, reflecting walls at [10, L - 10]
            self.tvel = (self.tvel - self.TH * self.tvel * self.DT
                         + self.SIG * math.sqrt(self.DT) * self.randn(E, R, 2))
            self.tpos = self.tpos + self.tvel * self.DT
            lo, hi = self.tpos < 10, self.tpos > self.ARENA - 10
            self.tvel = torch.where(lo, self.tvel.abs(), torch.where(hi, -self.tvel.abs(), self.tvel))
            self.tpos = self.tpos.clamp(10, self.ARENA - 10)
            err_sum += (self.tpos - self.pos).norm(dim=-1)
            stale_sum += stale.float()
            age_sum += self.act_age * self.CONTROL_STEP_MS
            drops += out["edge_dropped"].float().sum(-1) + out["cmd_dropped"].float().sum(-1)
        self.prev_rate = rate
        K = float(self.NET_SUBSTEPS)
        err = err_sum / K
        extra = {"stale_frac": (stale_sum / K).mean(-1), "command_age_ms": (age_sum / K).mean(-1),
                 "edge_drops": drops / R * self.T}
        return -err / 2.0, err.mean(-1), extra

    def heuristic(self):
        """Gain 2 1/s, horizon 0.1 s; 50 Hz uploads when the queue is empty, else 10 Hz."""
        a0 = 2 * math.log10(2.0 / 0.5) - 1
        a1 = 0.1 / 0.5 * 2 - 1
        cont = torch.tensor([a0, a1], device=self.dev).expand(self.E, self.R, 2)
        ready = torch.full((self.E, self.R), 2, dtype=torch.long, device=self.dev)
        return cont, ready, torch.zeros_like(ready)
