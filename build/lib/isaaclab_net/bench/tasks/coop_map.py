"""CoopMap: cooperative mapping. Robots upload local map patches; the reward is the freshness of the edge map."""
from __future__ import annotations

import torch

from ..base import NetTask
from ..spec import MetricSpec


class CoopMap(NetTask):
    NAME = "coop_map"
    DESCRIPTION = (
        "The edge keeps a 15 x 15 map of 10 m cells over the 150 m arena. A delivered patch refreshes every cell "
        "within 10 m (small patch) or 20 m (large patch) of where the robot was when it captured the patch, stamped "
        "with the capture step. Cell age is capped at 10 s. The reward of each robot is minus the fleet mean cell "
        "age plus a credit for the age reduction its own patches caused.")
    MECHANISM = ("Freshness under contention and capture staleness: a patch describes the place and time of its "
                 "capture, so queueing delay ages the map it refreshes, and large patches cover more cells but load "
                 "the shared uplink and delay everyone's patches.")
    MSG_SIZES = (1200.0, 9000.0)
    SEND_CHOICES = ("none", "small", "large")
    LIGHT_SCALE = 0.025
    CONT_NAMES = ("vx", "vy")
    TASK_BLOCKS = (("pos", 2), ("probe_age", 4), ("neighbour_offset", 2), ("own_cell_age", 1), ("fleet_age", 1))
    METRIC = MetricSpec("map_age_s", "s", False, "mean age of the edge map over cells and steps (lower is better)")
    VMAX = 0.3
    G = 15
    AMAX = 100
    RAD = (10.0, 20.0)
    HIST = 32
    PROBE = 20.0

    def __init__(self, cfg, device="cpu"):
        super().__init__(cfg, device)
        E, R, d = self.E, self.R, self.dev
        assert self.HIST > self.TIMEOUT_STEPS
        c = (torch.arange(self.G, device=d).float() + 0.5) * (self.ARENA / self.G)
        self.cc = torch.stack(torch.meshgrid(c, c, indexing="ij"), -1).reshape(-1, 2)
        self.C = self.cc.shape[0]
        self.rad = torch.tensor(self.RAD, device=d)
        self.dirs = torch.tensor([[1, 0], [-1, 0], [0, 1], [0, -1]], device=d).float()
        self.stamp = torch.full((E, self.C), -float(self.AMAX), device=d)
        self.hist = torch.zeros(E, R, self.HIST, 2, device=d)
        self.heading = torch.zeros(E, R, 2, device=d)

    def _task_reset(self, m):
        E, R = self.E, self.R
        self.pos = self.where_env(m, self.rand(E, R, 2) * self.ARENA, self.pos)
        self.stamp = self.where_env(m, torch.full_like(self.stamp, -float(self.AMAX)), self.stamp)
        self.hist = self.where_env(m, torch.zeros_like(self.hist), self.hist)
        self.heading = self.where_env(m, self.unit(self.randn(E, R, 2)), self.heading)

    def _ages(self):
        return (self.t[:, None].float() - self.stamp).clamp(max=self.AMAX)

    def _cell(self, p):
        ij = (p / (self.ARENA / self.G)).floor().long().clamp(0, self.G - 1)
        return ij[..., 0] * self.G + ij[..., 1]

    def _probe_ages(self):
        E, R, L = self.E, self.R, self.ARENA
        age = self._ages() / self.AMAX
        probes = self.pos[:, :, None, :] + self.PROBE * self.dirs
        inside = ((probes >= 0) & (probes <= L)).all(-1)
        return age.gather(1, self._cell(probes).reshape(E, -1)).reshape(E, R, 4) * inside, age

    def _task_obs(self):
        R, L = self.R, self.ARENA
        pa, age = self._probe_ages()
        own = age.gather(1, self._cell(self.pos))
        dd = (self.pos[:, :, None] - self.pos[:, None]).norm(dim=-1) + 1e9 * torch.eye(R, device=self.dev)
        nn_ = dd.argmin(-1)
        nb = (self.pos.gather(1, nn_[..., None].expand(-1, -1, 2)) - self.pos) / L
        fleet = age.mean(-1, keepdim=True).expand(-1, R)
        return torch.cat([self.pos / L, pa, nb, own[..., None], fleet[..., None]], -1)

    def _step(self, cont, send):
        E, R = self.E, self.R
        # the patch captures the start-of-step position at the env's network clock
        clk = self.net.clock
        slot = (clk % self.HIST)[:, None, None, None].expand(E, R, 1, 2)
        self.hist = self.hist.scatter(2, slot, self.pos[:, :, None, :])
        new_pos = self.move(cont, self.VMAX)
        out = self._net_step(send, new_pos)
        dlv, cap, cls = out["msg_delivered"], out["cap"], out["cls"]
        idx = (cap.clamp(min=0) % self.HIST)[..., None].expand(-1, -1, -1, 2)
        fpos = self.hist.gather(2, idx)                                               # [E,R,F,2]
        rad = self.rad[(cls - 1).clamp(min=0, max=1)]
        cover = ((fpos[..., None, :] - self.cc).norm(dim=-1) < rad[..., None]) & dlv[..., None]
        val = torch.where(cover, cap[..., None].float(), torch.full_like(cover, -1e9, dtype=torch.float))
        best = val.max(2).values                                                      # [E,R,C]
        own_gain = (best - self.stamp[:, None]).clamp(min=0, max=self.AMAX).sum(-1)
        self.stamp = torch.maximum(self.stamp, best.max(1).values)
        self.pos = new_pos
        # ages are read at the end of the step: t + 1 in the env clock
        fleet = ((self.t[:, None] + 1).float() - self.stamp).clamp(max=self.AMAX).mean(-1)
        rew = -fleet[:, None] / self.AMAX + own_gain * R / (self.AMAX * self.C)
        return rew, fleet * self.step_s, {}

    def heuristic(self):
        """Persistent heading, turned toward the oldest of the four probe cells now and then; large patch when the
        queue is empty."""
        pa, _ = self._probe_ages()
        best = self.dirs[pa.argmax(-1)]                                               # [E,R,2]
        turn = (self.prand(self.E, self.R) < 0.1)[..., None]
        self.heading = torch.where(turn, best, self.heading)
        edge = (self.pos < 5) | (self.pos > self.ARENA - 5)
        self.heading = torch.where(edge, (self.ARENA / 2 - self.pos).sign() * self.heading.abs(), self.heading)
        ready = torch.full((self.E, self.R), 2, dtype=torch.long, device=self.dev)
        return self.unit(self.heading), ready, torch.zeros_like(ready)
