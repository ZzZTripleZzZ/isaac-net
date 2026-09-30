"""Fleet-Alert: hazard detection offloaded over the shared uplink (the example fleet task, ported to NetTask)."""
from __future__ import annotations

import torch

from ..base import NetTask
from ..spec import MetricSpec


class FleetAlert(NetTask):
    NAME = "fleet_alert"
    DESCRIPTION = (
        "R robots drive to random goals in a 150 m arena. Hazards appear near a random robot, grow for 5 s and last "
        "10 s. A camera frame detects a hazard with probability 0.6 within 25 m (small frame) or 1.0 within 50 m "
        "(large frame), and the whole fleet learns the hazard only when a detecting frame is delivered. Until then "
        "robots cannot see it and pay 1 per step inside it; progress to the goal and 2 per goal reached are rewarded.")
    MECHANISM = ("Detection latency under self-generated contention: large frames detect more, but when many robots "
                 "send them together the shared uplink queues build up, so the detection reaches the fleet late or "
                 "times out.")
    MSG_SIZES = (4000.0, 30000.0)
    SEND_CHOICES = ("none", "small", "large")
    LIGHT_SCALE = 0.01
    CONT_NAMES = ("vx", "vy")
    TASK_BLOCKS = (("pos", 2), ("goal_offset", 2), ("hazard_offset", 2), ("hazard_radius", 1), ("siren", 1),
                   ("hazard_known", 1))
    METRIC = MetricSpec("hazard_exposure", "fraction of robot-steps", False,
                        "share of robot-steps spent inside an active hazard (lower is better)")
    VMAX = 0.3
    H_R, H_GROW, H_LIFE, H_RATE, SIREN = 15.0, 0.3, 100, 1 / 80, 10
    RANGE = (25.0, 50.0)
    PDET = (0.6, 1.0)

    def __init__(self, cfg, device="cpu"):
        super().__init__(cfg, device)
        E, R, d = self.E, self.R, self.dev
        self.rng_t = torch.tensor(self.RANGE, device=d)
        self.pdet_t = torch.tensor(self.PDET, device=d)
        self.goal = torch.zeros(E, R, 2, device=d)
        self.h_on = torch.zeros(E, dtype=torch.bool, device=d)
        self.h_pos = torch.zeros(E, 2, device=d)
        self.h_id = torch.zeros(E, dtype=torch.long, device=d)
        self.h_start = torch.zeros(E, dtype=torch.long, device=d)
        self.known = torch.zeros(E, dtype=torch.bool, device=d)

    def _task_reset(self, m):
        E, R, L = self.E, self.R, self.ARENA
        self.pos = self.where_env(m, self.rand(E, R, 2) * L, self.pos)
        self.goal = self.where_env(m, self.rand(E, R, 2) * L, self.goal)
        self.h_on = self.h_on & ~m
        self.known = self.known & ~m
        self.h_id = torch.where(m, torch.zeros_like(self.h_id), self.h_id)
        self.h_start = torch.where(m, torch.zeros_like(self.h_start), self.h_start)

    def _radius(self):
        return ((self.t - self.h_start + 1).float() * self.H_GROW).clamp(max=self.H_R) * self.h_on.float()

    def _task_obs(self):
        L, R = self.ARENA, self.R
        kf = self.known.float()[:, None].expand(-1, R)
        hrel = (self.h_pos[:, None, :] - self.pos) / L * kf[..., None]
        hr = self._radius()[:, None] / L * kf
        siren = (self.h_on & (self.t - self.h_start < self.SIREN)).float()[:, None].expand(-1, R)
        return torch.cat([self.pos / L, (self.goal - self.pos) / L, hrel, hr[..., None], siren[..., None],
                          kf[..., None]], -1)

    def _step(self, cont, send):
        E, R, t = self.E, self.R, self.t
        end = self.h_on & (t - self.h_start >= self.H_LIFE)
        self.h_on = self.h_on & ~end
        self.known = self.known & ~end
        spawn = ~self.h_on & (self.rand(E) < self.H_RATE)
        j = self.randint(R, (E,))
        c = self.pos[torch.arange(E, device=self.dev), j] + 10 * self.randn(E, 2)
        self.h_pos = torch.where(spawn[:, None], c.clamp(0, self.ARENA), self.h_pos)
        self.h_id = self.h_id + spawn.long()
        self.h_start = torch.where(spawn, t, self.h_start)
        self.h_on = self.h_on | spawn
        self.known = self.known & ~spawn
        # frames captured at the start-of-step pose
        dist = (self.pos - self.h_pos[:, None, :]).norm(dim=-1)
        ci = (send - 1).clamp(min=0)
        det = (send > 0) & self.h_on[:, None] & (dist < self.rng_t[ci]) & (self.rand(E, R) < self.pdet_t[ci])
        tag = torch.where(det, self.h_id[:, None].expand(E, R), torch.full_like(send, -1))
        cur = torch.where(self.h_on, self.h_id, torch.full_like(self.h_id, -1))
        d_old = (self.goal - self.pos).norm(dim=-1)
        new_pos = self.move(cont, self.VMAX)
        out = self._net_step(send, new_pos, tag=tag, cur_tag=cur)
        self.known = self.known | (out["tag_delivered"] & self.h_on)
        self.pos = new_pos
        d_new = (self.goal - self.pos).norm(dim=-1)
        inside = self.h_on[:, None] & ((self.pos - self.h_pos[:, None, :]).norm(dim=-1) < self._radius()[:, None])
        reached = d_new < 3.0
        rew = (d_old - d_new) - inside.float() + 2.0 * reached.float()
        self.goal = torch.where(reached[..., None], self.rand(E, R, 2) * self.ARENA, self.goal)
        return rew, inside.float().mean(-1), {"goals": reached.float().sum(-1) / R * self.T}

    def heuristic(self):
        """Drive to the goal and away from a known hazard; send a large frame when the queue is empty."""
        to_goal = self.unit(self.goal - self.pos)
        away = self.pos - self.h_pos[:, None, :]
        near = (self.known[:, None] & (away.norm(dim=-1) < self._radius()[:, None] + 5.0))[..., None]
        vel = torch.where(near, self.unit(away), to_goal)
        ready = torch.full((self.E, self.R), 2, dtype=torch.long, device=self.dev)
        return vel, ready, torch.zeros_like(ready)
