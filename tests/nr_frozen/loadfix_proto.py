"""FROZEN copy of the load-gap prototype (isaac_net/core/nr_loadfix.py at main 2400ed9), kept as the bitwise
reference of the engine's 5G-LENA MAC switches (tests/test_nr_loadfix.py L0). Do not edit.

Prototype 5G-LENA uplink-pipeline models for the NR engine under load (see docs/fidelity-load-gap.md).

The NR engine matches 5G-LENA at light load but is optimistic in loaded cells. The diagnosis traced the gap to
5G-LENA scheduler and grant-pipeline behaviors that the engine does not have. This module adds them as switches
on top of the unchanged engine, as subclasses: `LoadFixUlMac` (a UlMac whose data slot carries the switches) and
`LoadFixNet` (an NRNet that builds it). With every switch off, LoadFixNet is bitwise NRNet.

Switches (LoadFixConfig):
- pf_intra_slot: 5G-LENA OFDMA PF (NrMacSchedulerOfdma::AssignULRBG with NrMacSchedulerUeInfoPF). The winner of
  an RBG has its average throughput updated at once, (1 - 1/w) avg + (1/w) TB(k), before the next RBG is ranked,
  so RBGs spread over the UEs with close metrics instead of going to one UE until its need is covered. The
  average is committed with the granted TB bytes (not the decoded bytes).
- pf_active_only: the PF average of a UE moves only in slots where it has data to schedule (5G-LENA updates
  only the UEs in the active list); an idle UE keeps its average instead of decaying to zero.
- retx_tdma: an UL retransmission occupies every data symbol of its slot and at most one is sent per slot, the
  oldest NACK first (NrMacSchedulerHarqRr::ScheduleUlHarq), so no new data is scheduled in that slot.
- amc_prev_alloc: the MCS is the highest one whose TB error rate is at most the target for the PRB count of the
  UE's previous PUSCH (NrAmc::CreateCqiFeedbackSiso runs on the SINR of the last PUSCH, whose PRBs set the TB
  size of the lookup); the TB size then uses the current allocation.
- grant_pipeline: SR bootstrap grant, quantized buffer status reports and their report delay, with the padding
  they cause, instead of the lumped SR-to-grant delay. A UE with data that the gNB does not know about sends an SR
  at the next SR opportunity; sr_boot_slots later the scheduler owes it a 17-byte bootstrap grant (one RBG). Every
  PUSCH carries a short BSR with the buffer left after that TB, quantized UP to the 38.321 level table
  (BufferSizeLevelBsr); it reaches the scheduler bsr_delay_slots after the PUSCH and then overwrites the gNB's
  estimate (plus 5 bytes of RLC and MAC header), which grants reduce as they are issued. The scheduler allocates
  RBGs until the TB covers the estimate, so the grants issued while a report is in flight are counted twice and
  padded. A UE whose buffer has drained leaves a residue of rlc_tail_bytes (the RLC header bytes its MAC did not
  account for) that the UE reports only when the RLC buffer-status timer (rlc_tail_timer_slots after the draining
  TB) expires or new data arrives, after which it asks again with an SR.
- tb_overhead_bytes (int or None): per-TB overhead (MAC subheader 3 + short BSR 5 in 5G-LENA); None keeps the
  NRConfig value.

Everything is fixed-shape [E, R, ...] state with partial reset(env_ids); single cell only (asserted).
Hooks: none. The subclass replaces UlMac.slot with a copy of MacLink.slot that carries the switches; after the
fast backends land, the switches belong in mac.py / mac_ul.py behind NRConfig fields (see the patch description
in docs/fidelity-load-gap.md).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from isaac_net.core.config import NRConfig
from isaac_net.core.mac import BIG
from isaac_net.core.mac_ul import UlMac
from isaac_net.core.nr_engine import NRNet
from isaac_net.core.queues import env_mask, onehot, reset_where

# 3GPP TS 38.321 Table 6.1.3.1-1 upper bounds (5G-LENA nr-common.cc BufferSizeLevelBsrTable, 64 levels)
BSR_LEVELS = (0, 10, 12, 14, 17, 19, 22, 26, 31, 36, 42, 49, 57, 67, 78, 91, 107, 125, 146, 171, 200, 234, 274,
              321, 376, 440, 515, 603, 706, 826, 967, 1132, 1326, 1552, 1817, 2127, 2490, 2915, 3413, 3995, 4677,
              5476, 6411, 7505, 8787, 10287, 12043, 14099, 16507, 19325, 22624, 26487, 31009, 36304, 42502, 49759,
              58255, 68201, 79846, 93749, 109439, 128125, 150000, 150000)


@dataclass
class LoadFixConfig:
    """5G-LENA uplink-pipeline switches of LoadFixNet (all off = the NR engine unchanged)."""
    pf_intra_slot: bool = False
    pf_active_only: bool = False
    retx_tdma: bool = False
    amc_prev_alloc: bool = False
    grant_pipeline: bool = False
    tb_overhead_bytes: int | None = None
    sr_boot_slots: int = 6              # SR -> first PUSCH slot the bootstrap grant may use
    boot_bytes: int = 17                # 5G-LENA srGrantSize: 12 + 2 (RLC) + 3 (MAC subheader)
    bsr_delay_slots: int = 10           # PUSCH carrying a BSR -> first PUSCH scheduled with it (2 UL slots of DDDSU)
    bsr_hdr_bytes: int = 8              # the UE adds 5 (short BSR) + 3 (subheader) to its reported buffer
    est_hdr_bytes: int = 5              # scheduler adds 2 (RLC) + 3 (MAC subheader) to the reported buffer
    rlc_tail_bytes: int = 16            # RLC header residue left when the buffer drains (5-17 B in the traces)
    rlc_tail_timer_slots: int = 20      # RLC UM BufferStatusReportTimer, 10 ms at mu = 1

    def __post_init__(self):
        assert self.sr_boot_slots >= 1 and self.bsr_delay_slots >= 1 and self.rlc_tail_timer_slots >= 1
        assert self.boot_bytes >= 1 and self.rlc_tail_bytes >= 0

    def any(self):
        return (self.pf_intra_slot or self.pf_active_only or self.retx_tdma or self.amc_prev_alloc
                or self.grant_pipeline or self.tb_overhead_bytes is not None)


# named switch sets used by the benchmark (benchmarks/fidelity/loadfix/)
ARMS = {
    "base": {},
    "pf": dict(pf_intra_slot=True, pf_active_only=True),
    "pf_intra": dict(pf_intra_slot=True),
    "pf_active": dict(pf_active_only=True),
    "retx": dict(retx_tdma=True),
    "amc": dict(amc_prev_alloc=True),
    "oh8": dict(tb_overhead_bytes=8),
    "pipe": dict(grant_pipeline=True, tb_overhead_bytes=8),
    "pf_pipe": dict(pf_intra_slot=True, pf_active_only=True, grant_pipeline=True, tb_overhead_bytes=8),
    "all": dict(pf_intra_slot=True, pf_active_only=True, retx_tdma=True, amc_prev_alloc=True,
                grant_pipeline=True, tb_overhead_bytes=8),
}


class LoadFixUlMac(UlMac):
    """UlMac with the LoadFixConfig switches. Extra per-robot state (fixed shape, partial reset):
    last_nprb (PRBs of the last PUSCH), est (gNB buffer estimate, bytes), boot (bootstrap grant owed),
    rep_v / rep_g (in-flight BSRs: value, PUSCH slot; depth K), hid / hid_until (RLC residue not yet reported),
    enq_seen / armed (new data since the last drain: only the drain of new data leaves a residue)."""

    K = 4

    def __init__(self, cfg: NRConfig, lf: LoadFixConfig, E, R, device, meta=()):
        self.lf = lf
        super().__init__(cfg, E, R, device, meta)
        assert self.n_cells == 1
        self._levels = torch.tensor(BSR_LEVELS, dtype=torch.long, device=device)
        self._x_init()

    def _x_init(self):
        E, R, K, d = self.E, self.R, self.K, self.dev
        self.last_nprb = torch.zeros(E, R, device=d)
        self.est = torch.zeros(E, R, dtype=torch.long, device=d)
        self.boot = torch.zeros(E, R, dtype=torch.bool, device=d)
        self.rep_v = torch.zeros(E, R, K, dtype=torch.long, device=d)
        self.rep_g = torch.full((E, R, K), -BIG, dtype=torch.long, device=d)
        self.hid = torch.zeros(E, R, dtype=torch.long, device=d)
        self.hid_until = torch.full((E, R), -1, dtype=torch.long, device=d)
        self.enq_seen = torch.zeros(E, R, dtype=torch.long, device=d)
        self.armed = torch.zeros(E, R, dtype=torch.bool, device=d)
        self.ctr["tb_bytes"] = torch.zeros((), device=d)
        self.ctr["tb_empty"] = torch.zeros((), device=d)
        self.ctr["retx_block"] = torch.zeros((), device=d)

    X_INIT = {"last_nprb": 0.0, "est": 0, "boot": False, "rep_v": 0, "rep_g": -BIG, "hid": 0, "hid_until": -1,
              "enq_seen": 0, "armed": False}

    def reset(self, env_ids=None):
        super().reset(env_ids)
        if not hasattr(self, "last_nprb"):
            return
        m = env_mask(self.E, env_ids, self.dev)
        for n, v in self.X_INIT.items():
            setattr(self, n, reset_where(getattr(self, n), m, v))
        if env_ids is None:
            for k in ("tb_bytes", "tb_empty", "retx_block"):
                self.ctr[k].zero_()

    # ---------------- grant pipeline: UE-visible buffer, SR, BSR ----------------
    def _visible(self):
        """Bytes the UE MAC knows it has (the RLC residue is hidden until reported)."""
        return (self.unsent() - self.hid).clamp(min=0)

    def _quant(self, nbytes):
        """Short-BSR level upper bound for nbytes (0 stays 0)."""
        i = torch.searchsorted(self._levels, nbytes.clamp(max=150000).contiguous(), right=False)
        return self._levels[i.clamp(max=len(BSR_LEVELS) - 1)]

    def _arrivals(self):
        """New data since the last call reveals the RLC residue (an RLC buffer-status report on SDU arrival)."""
        new = self.q.enq > self.enq_seen
        self.hid = torch.where(new, torch.zeros_like(self.hid), self.hid)
        self.hid_until = torch.where(new, torch.full_like(self.hid_until, -1), self.hid_until)
        self.armed = self.armed | new
        self.enq_seen = self.q.enq.clone()

    def sr_step(self, g):
        if not self.lf.grant_pipeline:
            return super().sr_step(g)
        self._arrivals()
        # residue reported by the RLC timer
        exp = (self.hid > 0) & (self.hid_until >= 0) & (g >= self.hid_until)
        self.hid = torch.where(exp, torch.zeros_like(self.hid), self.hid)
        self.hid_until = torch.where(exp, torch.full_like(self.hid_until, -1), self.hid_until)
        inflight = (self.rep_g > -BIG).any(-1)
        need_sr = (self._visible() > 0) & (self.est <= 0) & ~self.boot & ~inflight & (self.sr_t < 0)
        self.sr_t = torch.where(need_sr, torch.full_like(self.sr_t, g), self.sr_t)

    def _pre_slot(self, g, gh=None):
        if not self.lf.grant_pipeline:
            return super()._pre_slot(g, g if gh is None else gh)
        self._arrivals()
        lf = self.lf
        owed = (self.sr_t >= 0) & (g - self.sr_t >= lf.sr_boot_slots)
        self.boot = self.boot | owed
        self.sr_t = torch.where(owed, torch.full_like(self.sr_t, -1), self.sr_t)
        # matured reports overwrite the estimate (latest one wins); the owed bootstrap grant is dropped
        ready = (self.rep_g > -BIG) & (g - self.rep_g >= lf.bsr_delay_slots)
        g_last = torch.where(ready, self.rep_g, torch.full_like(self.rep_g, -BIG)).max(-1)
        has = g_last.values > -BIG
        v = self.rep_v.gather(-1, g_last.indices[..., None]).squeeze(-1)
        val = torch.where(v > 0, self._quant(v) + lf.est_hdr_bytes, torch.zeros_like(v))
        self.est = torch.where(has, val, self.est)
        self.boot = self.boot & ~has
        self.rep_g = torch.where(ready, torch.full_like(self.rep_g, -BIG), self.rep_g)

    def _need(self, unsent):
        if not self.lf.grant_pipeline:
            return super()._need(unsent)
        return self.est + self.lf.boot_bytes * self.boot.long()

    def _post_tb(self, g, tx, tx_new, tbs, byt_new):
        """Grant pipeline after the TBs of this slot: estimate, BSR in flight, RLC residue."""
        lf = self.lf
        self.est = torch.where(tx_new, (self.est - tbs // 8).clamp(min=0), self.est)
        self.boot = self.boot & ~tx_new
        vis = self._visible()
        rep = torch.where(vis > 0, vis + lf.bsr_hdr_bytes, torch.zeros_like(vis))
        free = (self.rep_g == -BIG)
        k = torch.where(free.any(-1), free.long().argmax(-1), self.rep_g.argmin(-1))    # oldest slot if full
        oh = onehot(k, self.K) & tx[..., None]
        self.rep_v = torch.where(oh, rep[..., None], self.rep_v)
        self.rep_g = torch.where(oh, torch.full_like(self.rep_g, g), self.rep_g)

    # ---------------- one data slot (MacLink.slot with the switches) ----------------
    def slot(self, g, frac, nsym, sinr_ref_db, gain_now, ack_slot=0, gh=None, rel=0):
        lf = self.lf
        if not lf.any():
            return super().slot(g, frac, nsym, sinr_ref_db, gain_now, ack_slot, gh=gh, rel=rel)
        cfg, E, R, S, P, d, phy = self.cfg, self.E, self.R, self.S, self.P, self.dev, self.phy
        w = self.sb_prb
        unsent = self.unsent()
        tb_oh = cfg.tb_overhead_bytes if lf.tb_overhead_bytes is None else lf.tb_overhead_bytes
        self.h_state = torch.where((self.h_state == 2) & (self.h_ready <= g), torch.zeros_like(self.h_state), self.h_state)
        self._pre_slot(g)
        # ---- candidates ----
        rx_el = (self.h_state == 1) & (self.h_ready <= g)
        rx_p = torch.where(rx_el, self.h_ready, torch.full_like(self.h_ready, BIG)).argmin(-1)
        has_rx = rx_el.any(-1)
        free = self.h_state == 0
        p_new = free.long().argmax(-1)
        need = self._need(unsent)
        new_el = free.any(-1) & (need > 0) & ~has_rx
        pending = (self.h_state == 1).any(-1)
        rx_nsb = self.h_nsb.gather(-1, rx_p[..., None]).squeeze(-1)
        est, n_max = self._sched_estimate(sinr_ref_db)
        est_o = est + (self.olla[..., None] if cfg.olla else 0.0)
        re_prb = float(min(12 * nsym - cfg.dmrs_re_per_prb - cfg.overhead_re_per_prb, 156))
        if cfg.pf_metric == "wideband":
            allm = torch.ones(E, R, S, dtype=torch.bool, device=d)
            wb = phy.eff_sinr_all(est_o, allm, cfg.eff_sinr, w)
            ok = wb >= phy.thr_ref
            m = (ok.long() * torch.arange(1, phy.M + 1, device=d)).max(-1).values.clamp(min=1) - 1
            rate_sb = (phy.se[m] * re_prb / 8)[..., None] * w
        else:
            rate_sb = phy.se_at(est_o) * re_prb * w / 8
        metric_sb = rate_sb / self.avg[..., None]
        # ---- retransmission admission ----
        if lf.retx_tdma:
            # one retx per slot (every data symbol), the oldest NACK first; it blocks new data in the slot
            rdy = torch.where(has_rx, self.h_ready.gather(-1, rx_p[..., None]).squeeze(-1), torch.full_like(rx_nsb, BIG))
            first = rdy.argmin(-1)
            admitted = onehot(first, R) & has_rx.any(-1, keepdim=True) & has_rx
            block = admitted.any(-1, keepdim=True)
            new_el = new_el & ~block
            self.ctr["retx_block"] += (block[:, 0] & (rx_nsb * admitted).sum(-1).lt(S)).sum()
        else:
            rkey = torch.where(has_rx, metric_sb.sum(-1), torch.full_like(metric_sb[..., 0], -1.0))
            order = rkey.argsort(-1, descending=True)
            cum = (rx_nsb * has_rx).gather(-1, order).cumsum(-1)
            adm_sorted = has_rx.gather(-1, order) & (cum <= S)
            admitted = torch.zeros_like(has_rx).scatter(-1, order, adm_sorted)
        has_rx = admitted
        want_cnt = torch.where(has_rx, rx_nsb, torch.where(new_el, n_max, torch.zeros_like(n_max)))
        prio = (1e9 if cfg.retx_priority else 0.0) * has_rx.float()
        # ---- PF allocation, RBG by RBG ----
        cnt = torch.zeros(E, R, dtype=torch.long, device=d)
        left = need.float()
        cols = []
        wwin = 1.0 / cfg.pf_window
        base = (1 - wwin) * self.avg
        got = torch.zeros(E, R, device=d)                      # bytes granted so far in this slot
        for s in range(S):
            want = (cnt < want_cnt) & (has_rx | (left > 0))
            if lf.pf_intra_slot:
                ms = rate_sb[..., s] / (base + wwin * got).clamp(min=1e-9)
            else:
                ms = metric_sb[..., s]
            m = torch.where(want, ms + prio, torch.full_like(left, -1.0))
            best, wi = m.max(-1)
            oh = onehot(wi, R) & (best >= 0)[:, None]
            cols.append(oh)
            cnt = cnt + oh.long()
            got = got + rate_sb[..., s] * oh
            left = left - rate_sb[..., s] * oh * ~has_rx
        won = torch.stack(cols, -1)
        n_sb = won.sum(-1)
        n_prb = (won * w).sum(-1)
        tx_rx = has_rx & (n_sb == rx_nsb) & (n_sb > 0)
        tx_new = new_el & (n_sb > 0)
        tx = tx_rx | tx_new
        # ---- link adaptation ----
        est_tx = self._la_estimate(sinr_ref_db, n_prb, est)
        off = self.olla if cfg.olla else torch.zeros_like(self.olla)
        if lf.amc_prev_alloc:
            ref_prb = torch.where(self.last_nprb > 0, self.last_nprb, n_prb)
            ref_won = torch.ones_like(won) if cfg.ul_power == "whole_band" else won
            mcs_new, _ = phy.select_mcs(est_tx, ref_won, off, ref_prb, nsym, cfg.dmrs_re_per_prb,
                                        cfg.overhead_re_per_prb, cfg.eff_sinr, w)
            tbs_new = phy.tbs_all(n_prb, nsym, cfg.dmrs_re_per_prb, cfg.overhead_re_per_prb).gather(
                -1, mcs_new[..., None]).squeeze(-1)
        else:
            mcs_new, tbs_new = phy.select_mcs(est_tx, won, off, n_prb, nsym, cfg.dmrs_re_per_prb,
                                              cfg.overhead_re_per_prb, cfg.eff_sinr, w)
        cap_b = (tbs_new // 8 - tb_oh).clamp(min=1)
        if lf.grant_pipeline:
            vis = self._visible()
            byt_new = torch.minimum(cap_b, vis)
            # the TB that drains the buffer leaves the RLC header residue behind (reported by the RLC timer)
            drain = tx_new & (vis > 0) & (cap_b >= vis) & (self.hid == 0) & self.armed
            hold = torch.minimum(torch.full_like(vis, lf.rlc_tail_bytes), (vis - 1).clamp(min=0)) * drain
            byt_new = byt_new - hold
            self.hid = self.hid + hold
            self.armed = self.armed & ~drain
            self.hid_until = torch.where(hold > 0, torch.full_like(self.hid_until, g + lf.rlc_tail_timer_slots),
                                         self.hid_until)
        else:
            byt_new = torch.minimum(cap_b, unsent)
        # ---- bind TBs to processes ----
        p_tx = torch.where(tx_rx, rx_p, p_new)
        ohp = onehot(p_tx, P) & tx[..., None]
        ohn = ohp & tx_new[..., None]
        self.h_lo = torch.where(ohn, self.sent[..., None], self.h_lo)
        self.h_hi = torch.where(ohn, (self.sent + byt_new)[..., None], self.h_hi)
        self.h_mcs = torch.where(ohn, mcs_new[..., None], self.h_mcs)
        self.h_tbs = torch.where(ohn, tbs_new[..., None], self.h_tbs)
        self.h_nsb = torch.where(ohn, n_sb[..., None], self.h_nsb)
        self.h_ntx = torch.where(ohn, torch.zeros_like(self.h_ntx), self.h_ntx)
        self.h_comb = torch.where(ohn, torch.zeros_like(self.h_comb), self.h_comb)
        self.h_lexp = torch.where(ohn, torch.full_like(self.h_lexp, -float("inf")), self.h_lexp)
        self.h_nrb = torch.where(ohn, torch.zeros_like(self.h_nrb), self.h_nrb)
        self.h_state = torch.where(ohn, torch.ones_like(self.h_state), self.h_state)
        self.sent = self.sent + byt_new * tx_new
        g1 = lambda x: x.gather(-1, p_tx[..., None]).squeeze(-1)
        mcs, tbs = g1(self.h_mcs), g1(self.h_tbs)
        ntx = g1(self.h_ntx) + 1
        # ---- decoding ----
        act = self._rx_sinr(sinr_ref_db, n_prb, gain_now)
        if self.sinr_hook is not None:
            act = self.sinr_hook(g, self.dir, won & tx[..., None], n_prb, act)
        mcs_eq = None
        if cfg.harq_combining == "ir_lena":
            lse, _ = phy.eesm_lse(act, won, mcs, w)
            lexp = torch.logaddexp(g1(self.h_lexp), lse)
            nrb = g1(self.h_nrb) + n_prb
            eff_used = 10 * torch.log10((-phy.beta[mcs] * (lexp - torch.log(nrb.clamp(min=1)))).clamp(min=1e-6))
            eff_used = eff_used.nan_to_num(nan=-30.0, posinf=60.0, neginf=-30.0)
            mcs_eq = phy.mcs_eq[mcs, ntx.clamp(max=phy.mcs_eq.shape[1] - 1)]
            self.h_lexp = torch.where(ohp, lexp[..., None], self.h_lexp)
            self.h_nrb = torch.where(ohp, nrb[..., None], self.h_nrb)
            comb = g1(self.h_comb)
        else:
            eff = phy.eff_sinr(act, won, mcs, cfg.eff_sinr, w)
            comb = g1(self.h_comb) + 10 ** (eff / 10) if cfg.harq_combining == "cc" else 10 ** (eff / 10)
            eff_used = 10 * torch.log10(comb.clamp(min=1e-9))
        p_err = phy.tb_error_prob(mcs, eff_used, tbs, mcs_eq)
        u = torch.rand_like(p_err) if self.rng is None else self.rng.step_uniform(self._bler_site, rel, R)
        ok = tx & (u >= p_err)
        fail = tx & ~ok
        exh = fail & (ntx >= cfg.max_harq_tx)
        # ---- HARQ state ----
        ohp_ok, ohp_fail, ohp_exh = ohp & ok[..., None], ohp & fail[..., None], ohp & exh[..., None]
        st_ok, rdy_ok, rdy_fail = self._harq_times(g, ack_slot)
        self.h_state = torch.where(ohp_ok, torch.full_like(self.h_state, st_ok), self.h_state)
        self.h_ready = torch.where(ohp_ok, torch.full_like(self.h_ready, rdy_ok), self.h_ready)
        self.h_ntx = torch.where(ohp, ntx[..., None], self.h_ntx)
        self.h_comb = torch.where(ohp, comb[..., None], self.h_comb)
        self.h_ready = torch.where(ohp_fail, torch.full_like(self.h_ready, rdy_fail), self.h_ready)
        lo_tx, hi_tx = g1(self.h_lo), g1(self.h_hi)
        if self.trace is not None:
            self.trace.append((frac, tx.cpu(), ok.cpu(), lo_tx.cpu(), hi_tx.cpu(), exh.cpu()))
        if cfg.harq_fail == "rlc_am":
            self.h_ntx = torch.where(ohp_exh, torch.zeros_like(self.h_ntx), self.h_ntx)
            self.h_comb = torch.where(ohp_exh, torch.zeros_like(self.h_comb), self.h_comb)
            self.h_lexp = torch.where(ohp_exh, torch.full_like(self.h_lexp, -float("inf")), self.h_lexp)
            self.h_nrb = torch.where(ohp_exh, torch.zeros_like(self.h_nrb), self.h_nrb)
            self.h_ready = torch.where(ohp_exh, torch.full_like(self.h_ready, g + cfg.rlc_retx_slots), self.h_ready)
        else:
            self.h_state = torch.where(ohp_exh, torch.zeros_like(self.h_state), self.h_state)
            q = self.q
            ov = exh[..., None] & (q.start < hi_tx[..., None]) & (q.end > lo_tx[..., None]) & (q.cap >= 0)
            self.ctr["lost_frames"] += (ov & ~q.lost).sum()
            q.lost = q.lost | ov
        if cfg.olla:
            self.olla = (self.olla + cfg.olla_up_db * ok - self.olla_dn * fail).clamp(-10, 10)
        served = (hi_tx - lo_tx) * ok
        if lf.grant_pipeline:
            self._post_tb(g, tx, tx_new, tbs_new, byt_new)
            self.csi = gain_now
        else:
            self._post_slot(tx, gain_now)
        if lf.amc_prev_alloc:
            self.last_nprb = torch.where(tx, n_prb, self.last_nprb)
        # ---- PF average ----
        upd = (tbs_new // 8).float() * tx_new if lf.pf_intra_slot else served.float()
        new_avg = (1 - wwin) * self.avg + wwin * upd
        if lf.pf_active_only:
            active = (need > 0) & ~has_rx
            self.avg = torch.where(active, new_avg, self.avg)
        else:
            self.avg = new_avg
        # ---- counters ----
        c = self.ctr
        c["tb_new"] += tx_new.sum(); c["tb_retx"] += tx_rx.sum(); c["tb_ok"] += ok.sum()
        c["tb_fail"] += fail.sum(); c["exhaust"] += exh.sum(); c["bytes_ok"] += served.sum()
        c["bytes_new"] += (byt_new * tx_new).sum(); c["new_while_pending"] += (tx_new & pending).sum()
        c["prb_used"] += (n_prb * tx).sum(); c["prb_avail"] += float(w.sum()) * E * self.n_cells
        c["tb_bytes"] += ((tbs_new // 8) * tx_new).sum(); c["tb_empty"] += (tx_new & (byt_new == 0)).sum()
        ohk = onehot(ntx.clamp(max=cfg.max_harq_tx), cfg.max_harq_tx + 1)
        self.ntx_hist += (ohk & ok[..., None]).sum((0, 1))
        self.rv_tx += (ohk & tx[..., None]).sum(1)
        self.rv_fail += (ohk & fail[..., None]).sum(1)
        self.prb_used_env += (n_prb * tx).sum(1)
        # ---- RLC in-order delivery ----
        ack = self.ack_ptr()
        q = self.q
        done = (q.cap >= 0) & (q.end <= ack[..., None]) & ~q.lost & torch.isinf(q.fin)
        q.fin = torch.where(done, torch.full_like(q.fin, frac + cfg.proc_offset_ms / cfg.control_step_ms), q.fin)

    def compact(self, gone):
        super().compact(gone)
        if self.lf.grant_pipeline:
            self.hid = torch.minimum(self.hid, self.unsent())


class LoadFixNet(NRNet):
    """NRNet whose uplink MAC is LoadFixUlMac. lf=None or all switches off: bitwise NRNet."""

    def __init__(self, E, R, device, sizes, cfg: NRConfig | None = None, generator=None,
                 lf: LoadFixConfig | None = None):
        self.lf = lf or LoadFixConfig()
        super().__init__(E, R, device, sizes, cfg, generator)
        assert self.C == 1, "LoadFixNet is single-cell"
        meta = [("cls", torch.long), ("det", torch.bool), ("hid", torch.long), ("f_nact", torch.long),
                ("f_snr", torch.float32), ("f_own", torch.long)]
        # a fresh MAC is already in its initial state; no second reset(), which would redraw the fading state
        self.ul = LoadFixUlMac(self.cfg, self.lf, E, R, device, meta)
        self.ul.rng = self.rng                     # engine RNG (rng="engine"), as NRNet gives its links


def make_arm(name, **extra):
    """LoadFixConfig of a named arm (ARMS) with overrides."""
    kw = dict(ARMS[name])
    kw.update(extra)
    return LoadFixConfig(**kw)
