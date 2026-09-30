"""CoverageNav: coverage-aware navigation under a remote supervisor that needs periodic status updates."""
from __future__ import annotations

import torch

from ..base import NetTask
from ..spec import MetricSpec


class CoverageNav(NetTask):
    NAME = "coverage_nav"
    DESCRIPTION = (
        "Robots drive to random goals (progress plus 2 per goal). A supervisor must hear a status update (any "
        "delivered message) from each robot at least every 1 s, or the robot is stopped and may only crawl at 25% "
        "speed (so a robot stranded in a coverage hole can still leave it). A robot drives at full "
        "speed only if its newest delivered camera image (large message) is at most 1 s old, else at 40%. Each env "
        "has 5 coverage holes of 15 m radius with 25 dB extra path loss, so the route decides the link quality. "
        "The supervisor's rule uses the deliveries of the previous step.")
    MECHANISM = ("Coverage-dependent reachability: inside a hole the link rate drops and retransmissions stretch "
                 "delays, so status and images miss their deadlines unless the robot routes around holes or sends "
                 "less, and many robots sending images load the cell for everyone.")
    MSG_SIZES = (250.0, 2500.0)
    SEND_CHOICES = ("none", "status", "image")
    CONT_NAMES = ("vx", "vy")
    TASK_BLOCKS = (("pos", 2), ("goal_offset", 2), ("probe_snr", 4), ("snr_here", 1), ("status_age", 1),
                   ("image_age", 1), ("stopped", 1), ("slow", 1))
    METRIC = MetricSpec("progress_m", "m per robot per episode", True,
                        "net distance travelled toward the goals, per robot and episode (higher is better)")
    METRIC_REDUCE = "sum"
    VMAX = 0.3
    SLOW = 0.4
    CRAWL = 0.25                  # speed of a stopped robot
    TAU_S = 10
    TAU_I = 10
    PROBE = 15.0
    N_HOLES, HOLE_R = 5, 15.0
    BLOCKAGE = True
    BLOCKAGE_DB = 25.0

    def __init__(self, cfg, device="cpu"):
        super().__init__(cfg, device)
        E, R, d = self.E, self.R, self.dev
        self.goal = torch.zeros(E, R, 2, device=d)
        self.holes = torch.zeros(E, self.N_HOLES, 2, device=d)
        self.last_any = torch.zeros(E, R, dtype=torch.long, device=d)
        self.last_img = torch.zeros(E, R, dtype=torch.long, device=d)
        self.dirs = torch.tensor([[1, 0], [-1, 0], [0, 1], [0, -1]], device=d).float()

    def _task_reset(self, m):
        E, R, L = self.E, self.R, self.ARENA
        self.pos = self.where_env(m, self.rand(E, R, 2) * L, self.pos)
        self.goal = self.where_env(m, self.rand(E, R, 2) * L, self.goal)
        self.holes = self.where_env(m, 20.0 + (L - 20.0) * self.rand(E, self.N_HOLES, 2), self.holes)
        z = torch.zeros_like(self.last_any)
        self.last_any = torch.where(m[:, None], z, self.last_any)
        self.last_img = torch.where(m[:, None], z, self.last_img)

    def in_hole(self, p: torch.Tensor) -> torch.Tensor:
        """[E,N] points p [E,N,2|3] inside a coverage hole of their env."""
        return ((p[:, :, None, :2] - self.holes[:, None]).norm(dim=-1) < self.HOLE_R).any(-1)

    def _blocked(self, p3):
        g = self.net.radio.G
        return self.in_hole(p3)[..., None].expand(-1, -1, g)

    def _snr(self, p):
        """SNR (dB, best gNB) at positions p [E,N,2] with the holes, from the task's radio."""
        p3 = torch.cat([p, torch.zeros_like(p[..., :1])], -1)
        return self.net.radio.snr_db(p3, self._blocked(p3)).max(-1).values

    def _state(self):
        s_age = self.t[:, None] - self.last_any
        i_age = self.t[:, None] - self.last_img
        stopped = s_age > self.TAU_S
        slow = ~stopped & (i_age > self.TAU_I)
        return s_age, i_age, stopped, slow

    def _task_obs(self):
        E, R, L = self.E, self.R, self.ARENA
        s_age, i_age, stopped, slow = self._state()
        probes = (self.pos[:, :, None, :] + self.PROBE * self.dirs).clamp(0, L).reshape(E, R * 4, 2)
        psnr = self._snr(probes).reshape(E, R, 4) / 40.0
        snr = self._snr(self.pos) / 40.0
        sa = s_age.clamp(max=20).float() / 20
        ia = i_age.clamp(max=20).float() / 20
        return torch.cat([self.pos / L, (self.goal - self.pos) / L, psnr] +
                         [x[..., None] for x in (snr, sa, ia, stopped.float(), slow.float())], -1)

    def _step(self, cont, send):
        E, R = self.E, self.R
        _, _, stopped, slow = self._state()
        speed = torch.where(stopped, self.CRAWL, 1.0 - (1.0 - self.SLOW) * slow.float())
        d_old = (self.goal - self.pos).norm(dim=-1)
        new_pos = self.move(cont, self.VMAX, speed)
        out = self._net_step(send, new_pos, blocked_fn=self._blocked)
        self.last_any = torch.maximum(self.last_any, out["newest_cap"])
        img = torch.where(out["msg_delivered"] & (out["cls"] == 2), out["cap"], torch.full_like(out["cap"], -1))
        self.last_img = torch.maximum(self.last_img, img.max(-1).values)
        self.pos = new_pos
        d_new = (self.goal - self.pos).norm(dim=-1)
        reached = d_new < 3.0
        rew = (d_old - d_new) + 2.0 * reached.float()
        self.goal = torch.where(reached[..., None], self.rand(E, R, 2) * self.ARENA, self.goal)
        extra = {"stopped_frac": stopped.float().mean(-1), "slow_frac": slow.float().mean(-1),
                 "in_hole_frac": self.in_hole(self.pos).float().mean(-1)}
        return rew, (d_old - d_new).mean(-1), extra

    def heuristic(self):
        """Drive to the goal, sidestep a hole ahead; send an image when the queue is empty, else nothing."""
        to_goal = self.unit(self.goal - self.pos)
        ahead = self.pos + self.PROBE * to_goal
        side = torch.stack([-to_goal[..., 1], to_goal[..., 0]], -1)
        vel = torch.where(self.in_hole(ahead)[..., None] & ~self.in_hole(self.pos)[..., None],
                          self.unit(to_goal + 1.5 * side), to_goal)
        ready = torch.full((self.E, self.R), 2, dtype=torch.long, device=self.dev)
        return vel, ready, torch.zeros_like(ready)
