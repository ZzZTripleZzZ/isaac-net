"""NetSlotMC: multi-cell legacy L2 uplink with inter-cell interference and handover (merged from multicell/).

Subclass of netsim.NetSlot. It reuses NetBase (frame FIFO, timeouts, stats, per-env clocks, partial resets) and
NetSlot's per-slot MAC (SR/BSR, greedy PF with power-headroom cap, fading, OLLA, per-TB BLER, HARQ), and adds a
cell dimension C, one PF scheduler per cell, same-slot uplink interference, optional fractional power control
and A3 handover (radio.CellAssociation). Configured by the shared NRConfig (cells block; see config.multicell()).

Shapes: per robot [E,R]; per link [E,R,C]; fading [E,R,C,S,2]; per gNB and subband [E,C,S].
Python loops: only NetSlot's constant loops (UL slots per step, subbands). No host syncs in step (except
NetBase's optional stats logging).

Interference timing: every cell schedules slot g independently. Link adaptation (MCS, PHR cap) at gNB c uses the
N+I it measured in slot g-1 (EWMA with li_alpha < 1); decoding in slot g uses the actual interference of the
transmissions the other cells scheduled in slot g. OLLA absorbs the mismatch.

Equivalence: with C = 1, noise_model="fixed" and power control off (the NRConfig defaults), every floating-point
operation matches NetSlot's order and the engine is bitwise identical to NetSlot under the same random draws
(tests/test_multicell.py).

Inputs: step(t, poses) runs the engine's RadioMC; step(t, None, rx_dbm=rx) takes the per-link received power
[E,R,C] directly; the legacy form step(t, rx, hid) (NetSlotMC's original API) returns (newest, det_env).
"""
from __future__ import annotations

import math

import torch

from ..config import NRConfig, multicell
from ..queues import onehot
from ..radio import CellAssociation, RadioMC, pick
from .netsim import (BYTES_PER_SE, HARQ_MAX, HARQ_RTT, PF_AVG_MIN, PF_T, PHR_MIN_DB, RHO, RLC_EXTRA, SR_DELAY,
                     UL_PER_STEP, NetSlot, S, fill_rows, req_db, se_from_snr_db, serve_fifo)


class NetSlotMC(NetSlot):
    """Multi-cell NetSlot. serving_snr_db(rx) gives the per-robot SNR that add_frames() and observations expect."""

    # every per-env state tensor and its reset value (h is redrawn); checked by tests/test_multicell.py
    FRAME_INIT = dict(NetSlot.INIT)
    MAC_INIT_MC = {**NetSlot.MAC_INIT, "ioN_sum": 0.0, "n_flushed": 0}

    def __init__(self, E, R, device, sizes, cfg: NRConfig | None = None, seed=None):
        self.cfg = cfg if cfg is not None else multicell()
        self.C = self.cfg.n_cells
        self.gnb_xy = self.cfg.gnb_xy()
        self.noise_mw = 10 ** (self.cfg.subband_noise_dbm / 10)
        self.fixed = self.cfg.noise_model == "fixed"
        self.use_int = (not self.fixed) and self.cfg.ul_interference and self.C > 1
        self.slot_trace = None          # robot index: per-slot (served, cell, in HO, queue) of that robot
        self.log_sinr = False           # per-TB SINR samples (sanity sweeps, small E)
        self._rx_in = None
        super().__init__(E, R, device, sizes, seed=seed)

    # ---- state and partial reset ------------------------------------------------------------------
    def _alloc_state(self):
        super()._alloc_state()
        E, R, C, d = self.E, self.R, self.C, self.dev
        self.h = torch.zeros(E, R, C, S, 2, device=d)
        self.ar_r = torch.arange(R, device=d)
        self.assoc = CellAssociation(self.cfg, E, R, C, d, UL_PER_STEP)
        # measured noise plus interference per gNB and subband (mW), used by link adaptation
        self.ni_meas = torch.full((E, C, S), self.noise_mw, device=d)
        self.ioN_sum = torch.zeros(E, C, S, device=d)
        self.n_flushed = torch.zeros(E, dtype=torch.long, device=d)
        self.n_slots = 0
        self.sinr_log = []
        self.trace = []

    def _reset_state(self, ids):
        for n, v in NetSlot.MAC_INIT.items():
            fill_rows(getattr(self, n), ids, v)
        n = self._nrows(ids)
        fill_rows(self.h, ids, torch.randn(n, self.R, self.C, S, 2, device=self.dev, generator=self.gen) / math.sqrt(2))
        fill_rows(self.ni_meas, ids, self.noise_mw)
        fill_rows(self.ioN_sum, ids, 0.0)
        fill_rows(self.n_flushed, ids, 0)
        self.assoc.reset(ids)

    @property
    def serv(self):
        return self.assoc.serv

    # ---- inputs -----------------------------------------------------------------------------------
    def rx_from_poses(self, pos):
        """Per-link received power [E,R,C] from poses [E,R,2|3] through the engine's RadioMC (made on first use,
        from the engine generator; reset(env_ids) redraws its shadowing rows)."""
        if self.radio is None:
            self.radio = RadioMC(self.cfg, self.E, self.dev, generator=self.gen)
        return self.radio.rx_dbm(pos)

    def step(self, t, x=None, cur_hid=None, rx_dbm=None):
        """step(t, poses) -> dict; step(t, None, rx_dbm=rx) -> dict; step(t, rx, cur_hid) -> (newest, det_env)."""
        if rx_dbm is None:
            rx_dbm = x if cur_hid is not None else self.rx_from_poses(x)
        self._rx_in = rx_dbm
        return super().step(t, rx_dbm, cur_hid)

    def _snr_from(self, x):
        return self._rx_in

    def _advance(self, t, x, cur_hid, full):
        out = super()._advance(t, x, cur_hid, full)
        snr = self.serving_snr_db(self._rx_in)
        self._last_snr = snr               # SNR feature of the next submit (NetBase stores the rx input)
        if full:
            out["sinr_db"] = snr
            out["serving_cell"] = self.serv.clone()
        return out

    # ---- helpers ----------------------------------------------------------------------------------
    def _gain_db(self):
        return 10 * torch.log10((self.h ** 2).sum(-1).clamp(min=1e-6))        # [E,R,C,S]

    def _ni_la_db(self):
        """N+I (dBm) that link adaptation assumes at the serving gNB: (per subband, wideband)."""
        if self.fixed:
            return self.cfg.ni_fixed_dbm, self.cfg.ni_fixed_dbm
        per = self.ni_meas.gather(1, self.serv[..., None].expand(-1, -1, S))            # [E,R,S]
        return 10 * torch.log10(per), 10 * torch.log10(per.mean(-1))

    def serving_snr_db(self, rx):
        """Full-power single-subband SNR on the serving link vs the wideband N+I estimate [E,R]."""
        self.assoc.associate(rx)
        _, wide = self._ni_la_db()
        return pick(rx, self.serv) - wide

    def geometry_db(self, rx):
        return self.assoc.geometry_db(rx)

    def _handover(self, ho, target, g):
        """Robots `ho` [E,R] move to `target`. MAC state at the target starts fresh (OLLA, HARQ
        combining, PF average, pending SR); the buffer status arrives with the HO-complete message.
        RLC: "carry" keeps every queued frame, including a partly sent head-of-line frame (lossless);
        "flush" discards the frames still pending and logs them as drops."""
        self.assoc.switch(ho, target, g)
        z = torch.zeros_like(self.olla)
        self.olla = torch.where(ho, z, self.olla)
        self.hcnt = torch.where(ho, z, self.hcnt)
        self.wait = torch.where(ho, torch.zeros_like(self.wait), self.wait)
        self.sr_t = torch.where(ho, torch.full_like(self.sr_t, -1), self.sr_t)
        self.avg = torch.where(ho, torch.full_like(self.avg, 100.0), self.avg)
        if self.cfg.ho_rlc == "flush":
            drop = ho[..., None] & (self.cap >= 0) & (self.rem > 0)
            self.n_flushed = self.n_flushed + drop.flatten(1).sum(-1)
            if self.log_stats:
                keep = drop & (self.cap <= self.log_cap_max)
                for f in self.FEATS:
                    self.stats["x_" + f].append(getattr(self, f)[keep].cpu())
            self.cap = torch.where(drop, torch.full_like(self.cap, -1), self.cap)
            self.rem = torch.where(drop, torch.zeros_like(self.rem), self.rem)
            self.det = self.det & ~drop
        self.bsr = torch.where(ho, self.rem.sum(-1), self.bsr)

    # ---- one control step of UL slots -------------------------------------------------------------
    def _transmit(self, t, rx):
        cfg, C, E, R = self.cfg, self.C, self.E, self.R
        self.assoc.associate(rx)
        fin_t = torch.full_like(self.rem, float("inf"))
        ar = self.ar
        if C > 1:
            k_ho, tgt = self.assoc.plan(rx)
        rx_serv = pick(rx, self.serv)
        member = onehot(self.serv, C).permute(0, 2, 1)                  # [E,C,R]
        gain_now = self._gain_db()
        if self.log_sinr:
            geo = self.geometry_db(rx).clamp(max=99.0)
        for k in range(UL_PER_STEP):
            g = (t * UL_PER_STEP + k)[:, None]                          # [E,1] env-clock slot index
            if C > 1:     # sync-free: robots whose A3 trigger falls in this slot switch now
                self._handover(k_ho == k, tgt, g)
                rx_serv = pick(rx, self.serv)
                member = onehot(self.serv, C).permute(0, 2, 1)
                if self.log_sinr:
                    geo = self.geometry_db(rx).clamp(max=99.0)
            q = self.rem.sum(-1)
            # scheduling request for newly backlogged robots unknown to the gNB
            need_sr = (q > 0) & (self.bsr <= 0) & (self.sr_t < 0)
            self.sr_t = torch.where(need_sr, g, self.sr_t)
            granted = (self.sr_t >= 0) & (g - self.sr_t >= SR_DELAY)
            self.bsr[granted] = torch.clamp(self.bsr[granted], min=1.0)
            self.sr_t[granted] = -1
            # fading on every link: estimate from the previous slot, transmission sees the new one
            gain_prev = gain_now
            self.h = RHO * self.h + math.sqrt(1 - RHO ** 2) * torch.randn_like(self.h) / math.sqrt(2)
            gain_now = self._gain_db()
            gp_s = pick(gain_prev, self.serv)                            # [E,R,S]
            gn_s = pick(gain_now, self.serv)
            bonus = 3.0 * self.hcnt
            ni_sb, ni_wide = self._ni_la_db()
            snr_la = (rx_serv - ni_sb)[..., None] if self.fixed else rx_serv[..., None] - ni_sb
            est_db = snr_la + gp_s + self.olla[..., None]
            rate_est = se_from_snr_db(est_db) * BYTES_PER_SE
            elig = (self.bsr > 0) & (g >= self.wait) & (q > 0)
            if C > 1:
                elig = elig & self.assoc.schedulable(g)
            need = torch.where(elig, torch.minimum(self.bsr, q), torch.zeros_like(q))
            n_max = torch.floor(10 ** (((rx_serv - ni_wide) - PHR_MIN_DB) / 10)).clamp(1, S)
            cnt = torch.zeros_like(q)
            won = torch.zeros(E, R, S, dtype=torch.bool, device=self.dev)
            # one greedy PF scheduler per cell over the same subbands, cells batched
            for s in range(S):
                m = torch.where((need > 0) & (cnt < n_max), rate_est[..., s] / self.avg,
                                torch.full_like(need, -1.0))
                if C == 1:          # NetSlot's code path
                    best, w = m.max(-1)
                    ok = best > 0
                    won[ar, w, s] = ok
                    need[ar, w] -= rate_est[ar, w, s] * ok
                    cnt[ar, w] += ok.float()
                else:
                    m3 = torch.where(member, m[:, None, :], torch.full_like(m[:, None, :], -1.0))
                    best, w = m3.max(-1)                                                  # [E,C]
                    win = ((self.ar_r == w[..., None]) & (best > 0)[..., None]).any(1)    # [E,R]
                    won[..., s] = win
                    need = need - rate_est[..., s] * win
                    cnt = cnt + win.float()
            n = won.sum(-1)
            tx = n > 0
            nf = n.clamp(min=1).float()
            split_db = 10 * torch.log10(nf)          # per-subband power backoff from ue_tx_dbm
            if cfg.ul_pc_on:
                pl_serv = cfg.ue_tx_dbm - rx_serv
                split_db = torch.maximum(split_db, cfg.ue_tx_dbm - (cfg.ul_pc_p0_dbm + cfg.ul_pc_alpha * pl_serv))
            mean_est = (est_db * won).sum(-1) / nf - split_db
            se = se_from_snr_db(mean_est)
            # actual SINR; interference = robots scheduled on the same subband by other cells now
            if self.use_int:
                p_rx = 10 ** ((rx[..., None] - split_db[..., None, None] + gain_now) / 10) * won[:, :, None, :]
                other = ~member.permute(0, 2, 1)                                          # [E,R,C]
                interf = (p_rx * other[..., None]).sum(1)                                 # [E,C,S] mW
                ni_now = self.noise_mw + interf
                ni_act = 10 * torch.log10(ni_now.gather(1, self.serv[..., None].expand(-1, -1, S)))
                act = (rx_serv[..., None] - ni_act) - split_db[..., None] + gn_s
                self.ni_meas = cfg.li_alpha * ni_now + (1 - cfg.li_alpha) * self.ni_meas
                self.ioN_sum = self.ioN_sum + interf / self.noise_mw
            else:
                ni0 = cfg.ni_fixed_dbm if self.fixed else cfg.subband_noise_dbm
                act = (rx_serv - ni0)[..., None] - split_db[..., None] + gn_s
            self.n_slots += 1
            act_eff = (act * won).sum(-1) / nf + bonus
            p_ok = torch.sigmoid(1.5 * (act_eff - req_db(se)))
            ok_tb = tx & (torch.rand_like(p_ok) < p_ok)
            fail = tx & ~ok_tb
            served = torch.minimum(n * se * BYTES_PER_SE * ok_tb, q)
            self.rem, fin = serve_fifo(self.rem, served)
            fin_t = torch.where(fin, self._finvals(t, k), fin_t)
            if self.log_sinr:
                ni0 = cfg.ni_fixed_dbm if self.fixed else cfg.subband_noise_dbm
                snr_eff = (((rx_serv - ni0)[..., None] - split_db[..., None] + gn_s) * won).sum(-1) / nf
                self.sinr_log.append(torch.stack([act_eff - bonus, snr_eff, n.float(), geo], -1)[tx])
            if self.slot_trace is not None:
                r0 = self.slot_trace
                self.trace.append(torch.stack([served[:, r0], self.serv[:, r0].float(),
                                               (~self.assoc.schedulable(g))[:, r0].float(), q[:, r0]], -1))
            # link adaptation, HARQ, buffer status, PF averages (as NetSlot)
            self.olla = (self.olla + 0.05 * ok_tb - 0.45 * fail).clamp(-10, 10)
            hc = self.hcnt + 1
            exhausted = fail & (hc >= HARQ_MAX)
            self.hcnt = torch.where(fail, torch.where(exhausted, torch.zeros_like(hc), hc),
                                    torch.where(tx, torch.zeros_like(hc), self.hcnt))
            self.wait = torch.where(fail, g + HARQ_RTT + RLC_EXTRA * exhausted.long(), self.wait)
            self.bsr = torch.where(tx, self.rem.sum(-1), self.bsr)
            self.avg = ((1 - 1 / PF_T) * self.avg + (1 / PF_T) * served).clamp(min=PF_AVG_MIN)
        return fin_t
