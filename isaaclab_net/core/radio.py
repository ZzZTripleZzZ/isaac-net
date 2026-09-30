"""Radio: large-scale gain of every robot-cell link, cell association and handover (merged from multicell/).

    RadioMC          rx_dbm(pos [E,R,2|3]) -> [E,R,C] received power at every gNB with the full ue_tx_dbm on one
                     subband, no fast fading (= RSRP up to a constant). Path loss per link plus one sum-of-plane-waves
                     shadowing field per (env, cell). At C = 1 it draws the field in the prototype Radio's order, so
                     the same generator state gives the same field, and rx - ni_fixed_dbm equals Radio.snr_db bitwise.
    CellAssociation  serving cell per robot: max-RSRP attach after every (partial) reset, A3 with hysteresis and
                     time-to-trigger (exact trigger slot), handover interruption window.
    Radio            the prototype single-cell radio (gNB at the origin), used by the prototype levels.

Both classes are configured by NRConfig (cells and radio blocks), keep fixed-shape state and support partial
reset(env_ids) without host syncs. The NR engine (nr_engine.py) and the multi-cell legacy engine
(proto/netsim_mc.py) use both classes.
"""
from __future__ import annotations

import math

import torch

from .config import NRConfig
from .proto.netsim import Radio  # noqa: F401  (re-export: the prototype single-cell radio)
from .queues import env_mask, onehot, reset_where


def pick(x, idx):
    """x [E,R,C,...] at cell idx [E,R] -> [E,R,...]."""
    ix = idx.view(*idx.shape, 1, *([1] * (x.dim() - 3))).expand(*idx.shape, 1, *x.shape[3:])
    return x.gather(2, ix).squeeze(2)


class RadioMC:
    """Per-link large-scale radio for C = cfg.n_cells gNBs at cfg.gnb_xy()."""

    def __init__(self, cfg: NRConfig, E, device, generator=None):
        self.cfg, self.E, self.dev, self.gen = cfg, E, device, generator
        self.gnb = torch.tensor(cfg.gnb_xy(), dtype=torch.float32, device=device)     # [C,2]
        self.C = self.gnb.shape[0]
        self.amp = cfg.shadow_sigma_db * math.sqrt(2 / cfg.shadow_modes)
        self.k, self.phi = self._draw(E)

    def _draw(self, n):
        K, C, d, g = self.cfg.shadow_modes, self.C, self.dev, self.gen
        ang = torch.rand(n, C, K, device=d, generator=g) * 2 * math.pi
        wl = 20 + 40 * torch.rand(n, C, K, device=d, generator=g)
        k = torch.stack([torch.cos(ang), torch.sin(ang)], -1) * (2 * math.pi / wl)[..., None]
        phi = torch.rand(n, C, K, device=d, generator=g) * 2 * math.pi
        return k.permute(1, 0, 2, 3).contiguous(), phi.permute(1, 0, 2).contiguous()   # [C,E,K,2], [C,E,K]

    def reset(self, env_ids=None):
        """Redraw the shadowing fields of the given envs (fixed shape: draw all, keep masked rows)."""
        m = env_mask(self.E, env_ids, self.dev)
        k, phi = self._draw(self.E)
        self.k = torch.where(m.view(1, -1, 1, 1), k, self.k)
        self.phi = torch.where(m.view(1, -1, 1), phi, self.phi)

    @classmethod
    def from_radio(cls, radio, cfg: NRConfig, device):
        """Wrap an existing single-cell prototype Radio (same shadowing field) as a C = 1 RadioMC."""
        obj = cls.__new__(cls)
        obj.cfg, obj.E, obj.dev, obj.gen = cfg, radio.k.shape[0], device, None
        obj.gnb = torch.tensor(cfg.gnb_xy(), dtype=torch.float32, device=device)
        assert obj.gnb.shape[0] == 1
        obj.C, obj.amp = 1, radio.amp
        obj.k, obj.phi = radio.k[None].contiguous(), radio.phi[None].contiguous()
        return obj

    def rx_dbm(self, pos):
        """pos [E,R,2] or [E,R,3] (z ignored) -> [E,R,C] dBm."""
        cfg = self.cfg
        pos = pos[..., :2]
        d = (pos[:, :, None, :] - self.gnb).norm(dim=-1).clamp(min=1.0)                  # [E,R,C]
        pl = cfg.pl_const_db + (10 * cfg.pathloss_exp) * torch.log10(d)
        arg = torch.einsum("erx,cekx->cerk", pos, self.k) + self.phi[:, :, None, :]       # [C,E,R,K]
        sh = (self.amp * torch.cos(arg).sum(-1)).permute(1, 2, 0)                        # [E,R,C]
        return cfg.ue_tx_dbm - pl - sh

    def pathgain_db(self, pos):
        """Large-scale gain (negative dB, incl. shadowing) of every link [E,R,C]."""
        return self.rx_dbm(pos) - self.cfg.ue_tx_dbm


class CellAssociation:
    """Serving cell per robot with A3 handover.

    RSRP = rx_dbm (L3-filtered: large-scale only). Initial association (after every reset) = max RSRP.
    A3: best neighbour > serving + a3_offset + a3_hyst, held continuously toward the same target for
    ttt_slots. RSRP changes once per control step, so plan() evaluates the condition once per step and
    returns the exact slot inside the step at which the HO fires (at slot k the condition has held
    cnt + k + 1 slots, so k = ttt - cnt - 1). After a HO the robot cannot be scheduled for ho_int_slots.

    Slot unit: UL slots (NetSlotMC, cfg.ttt_slots / ho_int_slots) by default; slot_ms counts every slot
    of that duration instead (the NR engine passes cfg.slot_ms and slots_per_step).
    """

    INIT = {"serv": 0, "pending": True, "a3_cand": -1, "a3_cnt": 0, "ho_end": 0, "n_ho": 0}

    def __init__(self, cfg: NRConfig, E, R, C, device, slots_per_step, slot_ms=None):
        self.cfg, self.E, self.R, self.C, self.dev, self.K = cfg, E, R, C, device, slots_per_step
        if slot_ms is None:
            self.ttt, self.ho_int = cfg.ttt_slots, cfg.ho_int_slots
        else:
            self.ttt = int(round(cfg.a3_ttt_ms / slot_ms))
            self.ho_int = int(round(cfg.ho_interruption_ms / slot_ms))
        z = lambda dt, v: torch.full((E, R), v, dtype=dt, device=device)
        self.serv = z(torch.long, 0)
        self.pending = torch.ones(E, dtype=torch.bool, device=device)      # needs initial association
        self.a3_cand = z(torch.long, -1)
        self.a3_cnt = z(torch.long, 0)
        self.ho_end = z(torch.long, 0)
        self.n_ho = z(torch.long, 0)

    def reset(self, env_ids=None):
        m = env_mask(self.E, env_ids, self.dev)
        for n, v in self.INIT.items():
            setattr(self, n, reset_where(getattr(self, n), m, v))

    def associate(self, rx):
        """Initial max-RSRP association for envs flagged by reset (no host sync)."""
        self.serv = torch.where(self.pending[:, None], rx.argmax(-1), self.serv)
        self.pending = torch.zeros_like(self.pending)

    def geometry_db(self, rx):
        """Serving RSRP minus strongest other-cell RSRP [E,R]; small values = cell edge (+inf at C = 1)."""
        if self.C == 1:
            return torch.full(rx.shape[:2], float("inf"), device=rx.device)
        other = rx.masked_fill(onehot(self.serv, self.C), -float("inf"))
        return pick(rx, self.serv) - other.max(-1).values

    def plan(self, rx):
        """Returns (fire slot within this control step [E,R] or -1, target cell [E,R])."""
        cfg = self.cfg
        rs = pick(rx, self.serv)
        best, bc = rx.masked_fill(onehot(self.serv, self.C), -float("inf")).max(-1)
        cond = best > rs + cfg.a3_offset_db + cfg.a3_hyst_db
        cnt = torch.where(cond & (bc == self.a3_cand), self.a3_cnt, torch.zeros_like(self.a3_cnt))
        k_fire = (self.ttt - cnt - 1).clamp(min=0)
        fire = cond & (k_fire < self.K)
        self.a3_cnt = torch.where(cond & ~fire, cnt + self.K, torch.zeros_like(cnt))
        self.a3_cand = torch.where(cond & ~fire, bc, torch.full_like(bc, -1))
        return torch.where(fire, k_fire, torch.full_like(k_fire, -1)), bc

    def switch(self, ho, target, g):
        """g: slot of the switch (int, [E,1] or [E,R])."""
        self.serv = torch.where(ho, target, self.serv)
        self.ho_end = torch.where(ho, g + self.ho_int, self.ho_end)
        self.n_ho = self.n_ho + ho.long()

    def schedulable(self, g):
        return g >= self.ho_end
