"""Batched multi-robot fleet with hazard detection offloaded over a shared 5G uplink."""
import torch
from netsim import Radio, F, TIMEOUT

TASK_SIZES = {"T1": (4000.0, 30000.0), "T2": (500.0, 1500.0)}
OBS_DIM = 12


class FleetEnv:
    L = 150.0
    T = 300
    VMAX = 0.3           # m per 100 ms step
    H_R = 15.0           # final hazard radius, grows by H_GROW per step from spawn
    H_GROW = 0.3
    H_LIFE = 100
    H_RATE = 1 / 80
    SIREN = 10
    RANGE = (25.0, 50.0)  # small, large
    PDET = (0.6, 1.0)

    def __init__(self, E, R, net, device):
        self.E, self.R, self.net, self.dev = E, R, net, device
        self.rng = torch.tensor(self.RANGE, device=device)
        self.pdet = torch.tensor(self.PDET, device=device)

    def reset(self):
        E, R, d = self.E, self.R, self.dev
        self.t = 0
        self.pos = torch.rand(E, R, 2, device=d) * self.L
        self.goal = torch.rand(E, R, 2, device=d) * self.L
        self.radio = Radio(E, d)
        self.net.reset()
        self.h_on = torch.zeros(E, dtype=torch.bool, device=d)
        self.h_pos = torch.zeros(E, 2, device=d)
        self.h_id = torch.zeros(E, dtype=torch.long, device=d)
        self.h_start = torch.zeros(E, dtype=torch.long, device=d)
        self.known = torch.zeros(E, dtype=torch.bool, device=d)
        self.last_cap = torch.full((E, R), -TIMEOUT, dtype=torch.long, device=d)
        self.ep = {k: torch.zeros(E, R, device=d) for k in ("ret", "expo", "goals", "s1", "s2")}
        return self._obs()

    def _obs(self):
        L, R = self.L, self.R
        p = self.pos / L
        g = (self.goal - self.pos) / L
        kf = self.known.float()[:, None].expand(-1, R)
        hrel = (self.h_pos[:, None, :] - self.pos) / L * kf[..., None]
        hr = self._radius()[:, None] / L * kf
        siren = (self.h_on & (self.t - self.h_start < self.SIREN)).float()[:, None].expand(-1, R)
        qf = self.net.queued().float() / F
        aoi = (self.t - self.last_cap).clamp(max=50).float() / 50
        snr = self.radio.snr_db(self.pos) / 40
        return torch.cat([p, g, hrel] + [x[..., None] for x in (hr, siren, qf, aoi, snr, kf)], -1)

    def _radius(self):
        return ((self.t - self.h_start + 1).float() * self.H_GROW).clamp(max=self.H_R) * self.h_on.float()

    def step(self, vel, send):
        """vel [E,R,2] in [-1,1]; send [E,R] in {0,1,2}. Returns obs, reward [E,R], done, info."""
        E, R, d, t = self.E, self.R, self.dev, self.t
        # hazard lifecycle at step start
        end = self.h_on & (t - self.h_start >= self.H_LIFE)
        self.h_on &= ~end
        self.known &= ~end
        spawn = ~self.h_on & (torch.rand(E, device=d) < self.H_RATE)
        if spawn.any():
            j = torch.randint(0, R, (E,), device=d)
            c = self.pos[torch.arange(E, device=d), j] + 10 * torch.randn(E, 2, device=d)
            self.h_pos = torch.where(spawn[:, None], c.clamp(0, self.L), self.h_pos)
            self.h_id = self.h_id + spawn.long()
            self.h_start = torch.where(spawn, torch.full_like(self.h_start, t), self.h_start)
            self.h_on |= spawn
            self.known &= ~spawn
        # capture frames at the current position
        dist = (self.pos - self.h_pos[:, None, :]).norm(dim=-1)
        ci = (send - 1).clamp(min=0)
        det = (send > 0) & self.h_on[:, None] & (dist < self.rng[ci]) & (torch.rand(E, R, device=d) < self.pdet[ci])
        snr = self.radio.snr_db(self.pos)
        self.net.add_frames(t, send, det, self.h_id, snr)
        newest, det_env = self.net.step(t, snr, self.h_id)
        self.last_cap = torch.maximum(self.last_cap, newest)
        self.known |= det_env & self.h_on
        # move and score
        d_old = (self.goal - self.pos).norm(dim=-1)
        self.pos = (self.pos + vel * self.VMAX).clamp(0, self.L)
        d_new = (self.goal - self.pos).norm(dim=-1)
        inside = self.h_on[:, None] & ((self.pos - self.h_pos[:, None, :]).norm(dim=-1) < self._radius()[:, None])
        reached = d_new < 3.0
        rew = (d_old - d_new) - inside.float() + 2.0 * reached.float()
        self.goal = torch.where(reached[..., None], torch.rand(E, R, 2, device=d) * self.L, self.goal)
        ep = self.ep
        ep["ret"] += rew
        ep["expo"] += inside.float()
        ep["goals"] += reached.float()
        ep["s1"] += (send == 1).float()
        ep["s2"] += (send == 2).float()
        self.t += 1
        done = self.t >= self.T
        info = None
        if done:
            info = {k: v.mean().item() for k, v in ep.items()}
            self.reset()
        return self._obs(), rew, done, info
