"""Uplink MAC of the NR engine: SR/BSR access, UE power split with optional
power-headroom cap, gNB CSI from the previous PUSCH/SRS, UL HARQ timing (retx after proc + K2)."""
from __future__ import annotations

import math

import torch

from .mac import MacLink


class UlMac(MacLink):
    """sinr_ref_db = SINR if the full UE power were spread over cfg.snr_ref_prbs PRBs (legacy env
    input: full-power SNR over one 10-PRB subband; 5G-LENA ues.csv snr1_db). ul_power="allocated":
    the UE splits its power over its allocation (NetSlot, LENA UniformPowerAllocUsed);
    "whole_band": fixed PSD over the carrier (LENA UniformPowerAllocBw), no PHR cap."""

    direction = "ul"

    def sr_step(self, g):
        """SR on PUCCH at an SR opportunity (any UL-capable slot, PUSCH or not)."""
        need_sr = (self.unsent() > 0) & (self.bsr <= 0) & (self.sr_t < 0)
        self.sr_t = torch.where(need_sr, torch.full_like(self.sr_t, g), self.sr_t)

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        cfg = self.cfg
        P = len(cfg.tdd_pattern)
        self._pg_pos = next(p for p in range(P) if cfg.slot_symbols(p)[1] > 0)

    def _pre_slot(self, g):
        cfg = self.cfg
        if cfg.proactive_grant == "every_ul_slot" or (
                cfg.proactive_grant == "per_period" and g % len(cfg.tdd_pattern) == self._pg_pos):
            self.bsr = self.bsr.clamp(min=1)                    # grant without SR (unused grants not modelled)
        granted = (self.sr_t >= 0) & (g - self.sr_t >= self.cfg.sr_delay)
        self.bsr = torch.where(granted, self.bsr.clamp(min=1), self.bsr)
        self.sr_t = torch.where(granted, torch.full_like(self.sr_t, -1), self.sr_t)

    def _need(self, unsent):
        return torch.minimum(self.bsr, unsent)

    def _sched_estimate(self, sinr_ref):
        cfg, w, S = self.cfg, self.sb_prb, self.S
        if cfg.ul_power == "whole_band":       # PSD fixed: same per-PRB SINR whatever the grant size
            est = sinr_ref - 10 * math.log10(cfg.nprb / cfg.snr_ref_prbs) + self.csi
            return est, torch.full((self.E, self.R), S, dtype=torch.long, device=self.dev)
        est = sinr_ref - 10 * torch.log10(w / cfg.snr_ref_prbs) + self.csi     # one RBG at full power
        if cfg.phr_cap:
            n_max_prb = cfg.snr_ref_prbs * 10 ** ((sinr_ref.mean(-1) - cfg.phr_min_db) / 10)
            n_max = torch.floor(n_max_prb / w[0]).clamp(1, S).long()
        else:
            n_max = torch.full((self.E, self.R), S, dtype=torch.long, device=self.dev)
        return est, n_max

    def _split(self, n_prb):
        if self.cfg.ul_power == "whole_band":
            return torch.full_like(n_prb, 10 * math.log10(self.cfg.nprb / self.cfg.snr_ref_prbs))
        return 10 * torch.log10((n_prb / self.cfg.snr_ref_prbs).clamp(min=1e-3))

    def _la_estimate(self, sinr_ref, n_prb, est):
        return sinr_ref - self._split(n_prb)[..., None] + self.csi

    def _rx_sinr(self, sinr_ref, n_prb, gain_now):
        return sinr_ref - self._split(n_prb)[..., None] + gain_now

    def _harq_times(self, g, ack_slot):
        return 0, g, g + self.cfg.ul_rtt          # gNB decodes: process free at once; retx after proc + K2

    def _post_slot(self, tx, gain_now):
        self.bsr = torch.where(tx, self.unsent(), self.bsr)     # BSR rides every PUSCH
        self.csi = gain_now                                     # PUSCH/SRS measurement for the next decision

    def compact(self, gone):
        super().compact(gone)
        self.bsr = torch.minimum(self.bsr, self.unsent())
