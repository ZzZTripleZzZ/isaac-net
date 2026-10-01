"""Uplink MAC of the NR engine: SR/BSR access, UE power split with optional
power-headroom cap, gNB CSI from the previous PUSCH/SRS, UL HARQ timing (retx after proc + K2).

Grant models (cfg.ul_grant_model):
- "lumped": an SR is granted sr_grant_delay_slots later and every PUSCH reports the exact buffer (BSR), so the
  scheduler grants exactly what the UE has.
- "bsr": the 5G-LENA pipeline (docs/fidelity-load-gap.md). A UE with data the gNB does not know about sends an SR at
  the next SR opportunity; sr_boot_slots later the scheduler owes it a bootstrap grant of sr_boot_bytes (one RBG in
  practice). Every PUSCH carries a short BSR with the buffer left after that TB plus bsr_hdr_bytes, rounded UP to the
  38.321 level table (BSR_LEVELS); it reaches the scheduler bsr_delay_slots after the PUSCH and then overwrites the
  gNB's estimate (plus bsr_est_hdr_bytes), which grants reduce as they are issued. RBGs are allocated until the TB
  covers the estimate, so grants issued while a report is in flight are counted twice and padded. The TB that drains
  new data leaves rlc_tail_bytes behind (RLC header bytes the UE MAC did not count); the UE reports them only when
  the RLC buffer-status timer (rlc_tail_timer_ms after that TB) expires or new data arrives, and then asks again
  with an SR. Extra per-robot state (BSR_STATE, fixed shape, partial reset through MacLink.STATE): est (gNB buffer
  estimate), boot (bootstrap grant owed), rep_v / rep_g (reports in flight: value, PUSCH slot), hid / hid_until
  (residue not yet reported), enq_seen / armed (new data since the last drain).
With ul_amc_alloc="previous", last_nprb (PRBs of the robot's last PUSCH) is state too.
"""
from __future__ import annotations

import math

import torch

from .mac import BIG, MacLink
from .queues import onehot

# 3GPP TS 38.321 Table 6.1.3.1-1 upper bounds (5G-LENA nr-common.cc BufferSizeLevelBsrTable, 64 levels)
BSR_LEVELS = (0, 10, 12, 14, 17, 19, 22, 26, 31, 36, 42, 49, 57, 67, 78, 91, 107, 125, 146, 171, 200, 234, 274,
              321, 376, 440, 515, 603, 706, 826, 967, 1132, 1326, 1552, 1817, 2127, 2490, 2915, 3413, 3995, 4677,
              5476, 6411, 7505, 8787, 10287, 12043, 14099, 16507, 19325, 22624, 26487, 31009, 36304, 42502, 49759,
              58255, 68201, 79846, 93749, 109439, 128125, 150000, 150000)
BSR_STATE = {
    "est": ((), torch.long, 0), "boot": ((), torch.bool, False),
    "rep_v": (("K",), torch.long, 0), "rep_g": (("K",), torch.long, -BIG),
    "hid": ((), torch.long, 0), "hid_until": ((), torch.long, -1),
    "enq_seen": ((), torch.long, 0), "armed": ((), torch.bool, False),
}
AMC_STATE = {"last_nprb": ((), torch.float32, 0.0)}


class UlMac(MacLink):
    """sinr_ref_db = SINR if the full UE power were spread over cfg.snr_ref_prbs PRBs (legacy env
    input: full-power SNR over one 10-PRB subband; 5G-LENA ues.csv snr1_db). ul_power="allocated":
    the UE splits its power over its allocation (NetSlot, LENA UniformPowerAllocUsed);
    "whole_band": fixed PSD over the carrier (LENA UniformPowerAllocBw), no PHR cap."""

    direction = "ul"

    def sr_step(self, g):
        """SR on PUCCH at an SR opportunity (any UL-capable slot, PUSCH or not)."""
        if self._bsr_model:
            return self._sr_step_bsr(g)
        need_sr = (self.unsent() > 0) & (self.bsr <= 0) & (self.sr_t < 0)
        self.sr_t = torch.where(need_sr, g, self.sr_t)

    def __init__(self, cfg, *a, **k):
        st = dict(MacLink.STATE)
        if cfg.ul_grant_model == "bsr":
            st.update(BSR_STATE)
        if cfg.ul_amc_alloc == "previous":
            st.update(AMC_STATE)
        if len(st) > len(MacLink.STATE):
            self.STATE = st
        self._bsr_model = cfg.ul_grant_model == "bsr"
        super().__init__(cfg, *a, **k)
        if self._bsr_model:
            self._bsr_levels = torch.tensor(BSR_LEVELS, dtype=torch.long, device=self.dev)
            self._tail_slots = cfg.rlc_tail_timer_slots
        self.pc_backoff = None     # [E,R] fractional power control: min backoff (dB) from full power over
                                   # snr_ref_prbs PRBs, ue_tx - (P0 + alpha PL); set per slot by NRNet
        self.phr_snr = None        # [E,R] SNR against noise only for the PHR cap (several cells: power headroom
                                   # does not depend on interference); None = sinr_ref.mean(-1)
        P = len(cfg.tdd_pattern)
        self._pg_pos = next(p for p in range(P) if cfg.slot_symbols(p)[1] > 0)

    def _pre_slot(self, g, gh):
        if self._bsr_model:
            return self._pre_slot_bsr(g)
        cfg = self.cfg
        if cfg.proactive_grant == "every_ul_slot" or (
                cfg.proactive_grant == "per_period" and gh % len(cfg.tdd_pattern) == self._pg_pos):
            self.bsr = self.bsr.clamp(min=1)                    # grant without SR (unused grants not modelled)
        granted = (self.sr_t >= 0) & (g - self.sr_t >= self.cfg.sr_delay)
        self.bsr = torch.where(granted, self.bsr.clamp(min=1), self.bsr)
        self.sr_t = torch.where(granted, torch.full_like(self.sr_t, -1), self.sr_t)

    def _need(self, unsent):
        if self._bsr_model:
            return self.est + self.cfg.sr_boot_bytes * self.boot.long()
        return torch.minimum(self.bsr, unsent)

    def _sched_estimate(self, sinr_ref):
        cfg, w, S = self.cfg, self.sb_prb, self.S
        pc = self.pc_backoff
        if cfg.ul_power == "whole_band":       # PSD fixed: same per-PRB SINR whatever the grant size
            wb = 10 * math.log10(cfg.nprb / cfg.snr_ref_prbs)
            est = sinr_ref - (wb if pc is None else pc.clamp(min=wb)[..., None]) + self.csi
            return est, torch.full((self.E, self.R), S, dtype=torch.long, device=self.dev)
        if pc is None:
            est = sinr_ref - 10 * torch.log10(w / cfg.snr_ref_prbs) + self.csi     # one RBG at full power
        else:                                  # one RBG at the power-control PSD
            est = sinr_ref - torch.maximum(10 * torch.log10(w / cfg.snr_ref_prbs), pc[..., None]) + self.csi
        if cfg.phr_cap:
            ref = sinr_ref.mean(-1) if self.phr_snr is None else self.phr_snr
            n_max_prb = cfg.snr_ref_prbs * 10 ** ((ref - cfg.phr_min_db) / 10)
            n_max = torch.floor(n_max_prb / w[0]).clamp(1, S).long()
        else:
            n_max = torch.full((self.E, self.R), S, dtype=torch.long, device=self.dev)
        return est, n_max

    def _split(self, n_prb):
        """Per-PRB power backoff (dB) from the full UE power over snr_ref_prbs PRBs."""
        if self.cfg.ul_power == "whole_band":
            sp = torch.full_like(n_prb, 10 * math.log10(self.cfg.nprb / self.cfg.snr_ref_prbs))
        else:
            sp = 10 * torch.log10((n_prb / self.cfg.snr_ref_prbs).clamp(min=1e-3))
        return sp if self.pc_backoff is None else torch.maximum(sp, self.pc_backoff)

    def _la_estimate(self, sinr_ref, n_prb, est):
        return sinr_ref - self._split(n_prb)[..., None] + self.csi

    def _rx_sinr(self, sinr_ref, n_prb, gain_now):
        return sinr_ref - self._split(n_prb)[..., None] + gain_now

    def _harq_times(self, g, ack_slot):
        return 0, g, g + self.cfg.ul_rtt          # gNB decodes: process free at once; retx after proc + K2

    def _post_slot(self, tx, gain_now, g=None, tx_new=None, tbs_new=None):
        if self._bsr_model:
            self._post_slot_bsr(g, tx, tx_new, tbs_new)
        else:
            self.bsr = torch.where(tx, self.unsent(), self.bsr)     # BSR rides every PUSCH
        self.csi = gain_now                                     # PUSCH/SRS measurement for the next decision

    def handover(self, ho, flush=False):
        """As MacLink.handover; the buffer status reaches the target with the handover-complete message, and a
        pending SR is dropped."""
        super().handover(ho, flush)
        self.sr_t = torch.where(ho, torch.full_like(self.sr_t, -1), self.sr_t)
        self.bsr = torch.where(ho, self.unsent(), self.bsr)
        if self._bsr_model:            # the target learns the (quantized) buffer; reports in flight are lost
            vis = self._visible()
            rep = torch.where(vis > 0, self._quant(vis + self.cfg.bsr_hdr_bytes) + self.cfg.bsr_est_hdr_bytes,
                              torch.zeros_like(vis))
            self.est = torch.where(ho, rep, self.est)
            self.boot = self.boot & ~ho
            self.rep_g = torch.where(ho[..., None], torch.full_like(self.rep_g, -BIG), self.rep_g)

    def compact(self, gone):
        super().compact(gone)
        self.bsr = torch.minimum(self.bsr, self.unsent())
        if self._bsr_model:
            self.hid = torch.minimum(self.hid, self.unsent())

    # ---------------- ul_grant_model="bsr": the 5G-LENA SR / BSR pipeline ----------------
    def _visible(self):
        """Bytes the UE MAC knows it has (the RLC residue is hidden until reported)."""
        return (self.unsent() - self.hid).clamp(min=0)

    def _quant(self, nbytes):
        """Short-BSR level upper bound for nbytes (0 stays 0)."""
        lv = self._bsr_levels
        i = torch.searchsorted(lv, nbytes.clamp(max=150000).contiguous(), right=False)
        return lv[i.clamp(max=len(BSR_LEVELS) - 1)]

    def _arrivals(self):
        """New data since the last call reveals the RLC residue (an RLC buffer-status report on SDU arrival)."""
        new = self.q.enq > self.enq_seen
        self.hid = torch.where(new, torch.zeros_like(self.hid), self.hid)
        self.hid_until = torch.where(new, torch.full_like(self.hid_until, -1), self.hid_until)
        self.armed = self.armed | new
        self.enq_seen = self.q.enq.clone()

    def _sr_step_bsr(self, g):
        self._arrivals()
        exp = (self.hid > 0) & (self.hid_until >= 0) & (g >= self.hid_until)     # residue reported by the RLC timer
        self.hid = torch.where(exp, torch.zeros_like(self.hid), self.hid)
        self.hid_until = torch.where(exp, torch.full_like(self.hid_until, -1), self.hid_until)
        inflight = (self.rep_g > -BIG).any(-1)
        need_sr = (self._visible() > 0) & (self.est <= 0) & ~self.boot & ~inflight & (self.sr_t < 0)
        self.sr_t = torch.where(need_sr, g, self.sr_t)

    def _pre_slot_bsr(self, g):
        cfg = self.cfg
        self._arrivals()
        owed = (self.sr_t >= 0) & (g - self.sr_t >= cfg.sr_boot_slots)
        self.boot = self.boot | owed
        self.sr_t = torch.where(owed, torch.full_like(self.sr_t, -1), self.sr_t)
        # matured reports overwrite the estimate (latest one wins); the owed bootstrap grant is dropped
        ready = (self.rep_g > -BIG) & (g - self.rep_g >= cfg.bsr_delay_slots)
        g_last = torch.where(ready, self.rep_g, torch.full_like(self.rep_g, -BIG)).max(-1)
        has = g_last.values > -BIG
        v = self.rep_v.gather(-1, g_last.indices[..., None]).squeeze(-1)
        val = torch.where(v > 0, self._quant(v) + cfg.bsr_est_hdr_bytes, torch.zeros_like(v))
        self.est = torch.where(has, val, self.est)
        self.boot = self.boot & ~has
        self.rep_g = torch.where(ready, torch.full_like(self.rep_g, -BIG), self.rep_g)

    def _tb_payload(self, cap_b, unsent, tx_new, g):
        if not self._bsr_model:
            return torch.minimum(cap_b, unsent)
        vis = self._visible()
        byt_new = torch.minimum(cap_b, vis)
        # the TB that drains the buffer leaves the RLC header residue behind (reported by the RLC timer)
        drain = tx_new & (vis > 0) & (cap_b >= vis) & (self.hid == 0) & self.armed
        hold = torch.minimum(torch.full_like(vis, self.cfg.rlc_tail_bytes), (vis - 1).clamp(min=0)) * drain
        self.hid = self.hid + hold
        self.armed = self.armed & ~drain
        self.hid_until = torch.where(hold > 0, g + self._tail_slots, self.hid_until)
        return byt_new - hold

    def _post_slot_bsr(self, g, tx, tx_new, tbs_new):
        """After the TBs of a slot: estimate reduced by the new grants, a BSR in flight on every PUSCH."""
        cfg = self.cfg
        self.est = torch.where(tx_new, (self.est - tbs_new // 8).clamp(min=0), self.est)
        self.boot = self.boot & ~tx_new
        vis = self._visible()
        rep = torch.where(vis > 0, vis + cfg.bsr_hdr_bytes, torch.zeros_like(vis))
        free = self.rep_g == -BIG
        k = torch.where(free.any(-1), free.long().argmax(-1), self.rep_g.argmin(-1))    # oldest slot if full
        oh = onehot(k, self.rep_g.shape[-1]) & tx[..., None]
        self.rep_v = torch.where(oh, rep[..., None], self.rep_v)
        self.rep_g = torch.where(oh, g, self.rep_g)
