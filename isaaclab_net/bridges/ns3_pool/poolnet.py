"""PoolNet: NetBase drop-in whose uplink is a pool of ns-3 / 5G-LENA processes (one per env).

Usage (same as any rung):
    net = PoolNet(E, R, device, sizes)          # E <= 12 on the shared lab box
    env = FleetEnv(E, R, net, device)
    net.attach_env(env)                         # optional: real robot positions for the fading geometry
    obs = env.reset(); ... env.step(vel, send)

Mapping to ns-3 (per env e, robot r = UE r of worker e):
  * large-scale loss: the env's own SNR (path loss + its correlated shadowing field) is passed as a
    per-UE path-loss override, loss = P_TX - NI - snr_db = 113 - snr_db, so ns-3 sees exactly the
    single-subband full-power SNR NetSlot sees. Fast fading, AMC, HARQ, scheduling come from 5G-LENA.
  * position: env.pos if attached, otherwise a point at 45 deg whose log-distance loss equals the
    override (only the 3GPP fast-fading geometry uses it).
  * frames: a frame captured at step t is sent by UE r at the start of step t (fid = t), exactly one
    per robot per step, as ceil(bytes/1400) UDP packets through RLC UM with the 2 s PDCP discard.
  * delivery: when its last packet reaches the remote host, fin = t_cap + delay/period (step units).
  * reset: ns-3 cannot rewind, so every NetBase.reset() restarts the worker processes; the new
    processes attach at the positions of the first step (lazy start). Worker e uses RNG run
    seed_run + e + episode * E, so episodes and envs are independent.
Everything else (FIFO buffer of F frames, overflow, the 2 s timeout, stats) is NetBase's.
"""
from __future__ import annotations

import math

import torch

from ...core.proto.netsim import NI_DBM, P_TX_DBM, NetBase
from .ns3pool import Ns3Pool, encode_step


def loss_from_snr(snr_db):
    return P_TX_DBM - NI_DBM - snr_db


def equiv_pos(snr_db):
    """Position whose 40 + 35 log10(d) loss equals the override (no shadowing)."""
    d = 10 ** ((loss_from_snr(snr_db) - 40.0) / 35.0)
    d = d.clamp(1.0, 2000.0)
    return torch.stack([d / math.sqrt(2), d / math.sqrt(2)], -1)


class PoolNet(NetBase):
    def __init__(self, E, R, device, sizes, extra_args=(), launcher="local", seed_run=1, log_dir=None,
                 send_positions=True):
        self.pool = Ns3Pool(E, R, extra_args, launcher, seed_run=seed_run, log_dir=log_dir)
        self.env = None
        self.seed_run = seed_run
        self.episode = -1
        self.send_positions = send_positions
        self.pending = [[] for _ in range(E)]
        super().__init__(E, R, device, sizes)

    def attach_env(self, env):
        self.env = env

    # ---- NetBase hooks ------------------------------------------------------------------------
    def _reset_state(self, ids):
        if ids is not None:
            raise NotImplementedError("PoolNet restarts its ns-3 workers on reset: full resets only")
        self.pool.close()
        self.started = False
        self.pending = [[] for _ in range(self.E)]
        self.episode += 1

    def _on_arrival(self, t, e, r, i, draws):
        by = self.sizes[self.cls[e, r, i] - 1].round().long()
        for ee, rr, bb, tt in zip(e.tolist(), r.tolist(), by.tolist(), t[e].tolist()):
            self.pending[ee].append((rr, tt, bb))

    def _positions(self, snr_db):
        if self.env is not None:
            return self.env.pos.detach().float().cpu()
        return equiv_pos(snr_db.detach().float().cpu())

    def _transmit(self, t, snr_db):
        t = int(t[0]) if torch.is_tensor(t) else int(t)       # one shared clock (full resets only)
        E, R = self.E, self.R
        pos = self._positions(snr_db)
        loss = loss_from_snr(snr_db.detach().float().cpu())
        if not self.started:
            init = [[(float(pos[e, r, 0]), float(pos[e, r, 1]), float(loss[e, r])) for r in range(R)]
                    for e in range(E)]
            self.pool.seed_run = self.seed_run + self.episode * E
            self.pool.start(init)
            self.started = True
            self.t0 = t
        k = t - self.t0                      # worker-local step index
        pl, ll = pos.tolist(), loss.tolist()
        lines = []
        for e in range(E):
            ues = [(r, pl[e][r][0], pl[e][r][1], ll[e][r]) for r in range(R)] if self.send_positions else []
            lines.append(encode_step(k, ues, self.pending[e]))
            self.pending[e] = []
        replies = self.pool.step(lines)
        fin_t = torch.full_like(self.rem, float("inf"))
        period = self.pool.period
        cap = self.cap.cpu()
        hit_e, hit_r, hit_i, hit_f = [], [], [], []
        for e, rep in enumerate(replies):
            for ue, fid, gen, last in rep.done:
                slot = (cap[e, ue] == fid).nonzero()
                if slot.numel() == 0:
                    continue                  # timed out or overflowed on the NetBase side
                f = fid + (last - gen) / period
                hit_e.append(e); hit_r.append(ue); hit_i.append(int(slot[0, 0])); hit_f.append(min(f, t + 1 - 1e-6))
        if hit_e:
            d = self.dev
            idx = (torch.tensor(hit_e, device=d), torch.tensor(hit_r, device=d), torch.tensor(hit_i, device=d))
            fin_t[idx] = torch.tensor(hit_f, dtype=fin_t.dtype, device=fin_t.device)
            self.rem[idx] = 0.0
        return fin_t

    def close(self):
        self.pool.close()
