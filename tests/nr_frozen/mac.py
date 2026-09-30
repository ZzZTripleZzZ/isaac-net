"""Shared per-slot MAC algorithm for one direction of one cell (NR engine).

`MacLink.slot` implements, for every [E, R] robot at once: HARQ bookkeeping, candidate selection
(retransmission first, else new data on a free process), PF allocation RBG by RBG, link
adaptation, TB binding to HARQ processes, decoding with EESM + HARQ combining, HARQ / OLLA / PF
updates, and the RLC in-order pointer. Direction-specific pieces (SR/BSR, power split, CSI,
feedback timing) are the hooks overridden in mac_ul.UlMac and mac_dl.DlMac.

Byte-stream model: a new TB takes stream bytes [sent, sent + TBS/8 - tb_overhead) and binds them
to a free HARQ process, so new data proceeds on other processes while a failed TB waits (no MAC
head-of-line blocking; n_harq = 1 restores it). The in-order pointer is the lowest byte still
owned by an undecoded process (or `sent`). HARQ exhaustion: "rlc_am" resends the range after
rlc_retx_slots; "drop" marks overlapping frames lost (RLC UM).
"""
from __future__ import annotations

import torch

from isaaclab_net.core.config import NRConfig
from isaaclab_net.core.phy import PHY
from isaaclab_net.core.queues import FrameQueue, env_mask, onehot, reset_where

BIG = 2 ** 62


class MacLink:
    # per-robot state: name -> (extra dims key, dtype, initial value); dims "P" = HARQ, "S" = subband
    STATE = {
        "sent": ((), torch.long, 0), "floor": ((), torch.long, 0),
        "olla": ((), torch.float32, 0.0), "avg": ((), torch.float32, 100.0),
        "bsr": ((), torch.long, 0), "sr_t": ((), torch.long, -1),
        "csi": (("S",), torch.float32, 0.0),
        "h_state": (("P",), torch.long, 0),     # 0 free, 1 owns undecoded bytes, 2 decoded, awaiting feedback
        "h_lo": (("P",), torch.long, 0), "h_hi": (("P",), torch.long, 0),
        "h_ready": (("P",), torch.long, 0),     # earliest slot for the next (re)transmission / reuse
        "h_ntx": (("P",), torch.long, 0), "h_mcs": (("P",), torch.long, 0),
        "h_tbs": (("P",), torch.long, 0), "h_nsb": (("P",), torch.long, 0),
        "h_comb": (("P",), torch.float32, 0.0),             # chase: accumulated linear effective SINR
        "h_lexp": (("P",), torch.float32, -float("inf")),   # IR: log sum_w exp(-SINR / beta)
        "h_nrb": (("P",), torch.float32, 0.0),              # IR: PRBs over all transmissions
    }
    direction = None

    def __init__(self, cfg: NRConfig, E, R, device, meta=()):
        self.cfg, self.E, self.R, self.dev = cfg, E, R, device
        self.dir = self.direction
        self.P = cfg.n_harq
        self.S = cfg.n_subbands
        self.sb_prb = torch.tensor(cfg.subband_prbs, dtype=torch.float32, device=device)
        self.phy = PHY(self.dir, cfg.mcs_table, device, cfg.bler_target, cfg.bler_source, cfg.tbs_mode,
                       cfg.lena_ref_sc_per_rb, max(cfg.max_harq_tx, 2))
        cap = cfg.ul_mcs_max if self.dir == "ul" else cfg.dl_mcs_max
        if cap is not None:
            self.phy.mcs_max = min(int(cap), self.phy.M - 1)
        self.q = FrameQueue(E, R, cfg.frame_buffer, device, meta)
        self.olla_dn = cfg.olla_up_db * (1 - cfg.bler_target) / cfg.bler_target
        self.trace = None          # debug: list of per-slot (frac, tx, ok, lo, hi, exhausted) host copies
        self.sinr_hook = None      # multicell merge point, see slot()
        dims = {"P": self.P, "S": self.S}
        for n, (ex, dt, v) in self.STATE.items():
            setattr(self, n, torch.full((E, R) + tuple(dims[k] for k in ex), v, dtype=dt, device=device))
        self.ctr = {k: torch.zeros((), device=device) for k in
                    ("tb_new", "tb_retx", "tb_ok", "tb_fail", "exhaust", "bytes_ok", "bytes_new", "lost_frames",
                     "new_while_pending", "prb_used", "prb_avail")}
        self.ntx_hist = torch.zeros(cfg.max_harq_tx + 1, device=device)   # transmissions per decoded TB
        self.rv_tx = torch.zeros(E, cfg.max_harq_tx + 1, device=device)   # per env: TBs sent as transmission k
        self.rv_fail = torch.zeros(E, cfg.max_harq_tx + 1, device=device) # of which failed
        self.prb_used_env = torch.zeros(E, device=device)                 # per env: PRBs granted

    def reset(self, env_ids=None):
        """Partial reset: every per-robot state tensor of the given envs back to its initial value.
        Counters are global statistics and are cleared only by a full reset."""
        m = env_mask(self.E, env_ids, self.dev)
        self.q.reset(env_ids)
        for n, (_, _, v) in self.STATE.items():
            setattr(self, n, reset_where(getattr(self, n), m, v))
        if env_ids is None:
            for v in self.ctr.values():
                v.zero_()
            self.ntx_hist.zero_()
            self.rv_tx.zero_()
            self.rv_fail.zero_()
            self.prb_used_env.zero_()

    def unsent(self):
        return self.q.enq - self.sent

    def ack_ptr(self):
        own = (self.h_state == 1) & (self.h_hi > self.floor[..., None])
        lo = torch.where(own, torch.maximum(self.h_lo, self.floor[..., None]), torch.full_like(self.h_lo, BIG))
        return torch.minimum(self.sent, lo.min(-1).values)

    # ---------------- direction hooks ----------------
    def _pre_slot(self, g):
        pass

    def _need(self, unsent):
        return unsent

    def _sched_estimate(self, sinr_ref):
        """Scheduler's per-RBG SINR estimate [E,R,S] and max RBGs per UE [E,R]."""
        raise NotImplementedError

    def _la_estimate(self, sinr_ref, n_prb, est):
        return est

    def _rx_sinr(self, sinr_ref, n_prb, gain_now):
        raise NotImplementedError

    def _harq_times(self, g, ack_slot):
        """(state after ACK, ready slot after ACK, ready slot after NACK)."""
        raise NotImplementedError

    def _post_slot(self, tx, gain_now):
        pass

    # ---------------- one data slot ----------------
    def slot(self, g, frac, nsym, sinr_ref_db, gain_now, ack_slot=0):
        """Process one data slot at absolute slot g. sinr_ref_db [E,R,S] without fast fading (see the
        direction class), gain_now [E,R,S] fast-fading gain (dB), frac = completion time of this
        slot in control steps, ack_slot = DL HARQ-ACK slot."""
        cfg, E, R, S, P, d, phy = self.cfg, self.E, self.R, self.S, self.P, self.dev, self.phy
        w = self.sb_prb
        unsent = self.unsent()
        # DL processes whose ACK has reached the gNB become free
        self.h_state = torch.where((self.h_state == 2) & (self.h_ready <= g), torch.zeros_like(self.h_state), self.h_state)
        self._pre_slot(g)
        # ---- candidates: one TB per UE per slot, retransmissions first ----
        rx_el = (self.h_state == 1) & (self.h_ready <= g)
        rx_p = torch.where(rx_el, self.h_ready, torch.full_like(self.h_ready, BIG)).argmin(-1)
        has_rx = rx_el.any(-1)
        free = self.h_state == 0
        p_new = free.long().argmax(-1)
        need = self._need(unsent)
        new_el = free.any(-1) & (need > 0) & ~has_rx
        pending = (self.h_state == 1).any(-1)
        rx_nsb = self.h_nsb.gather(-1, rx_p[..., None]).squeeze(-1)
        # ---- scheduler's per-RBG estimate, with OLLA ----
        est, n_max = self._sched_estimate(sinr_ref_db)
        est_o = est + (self.olla[..., None] if cfg.olla else 0.0)
        re_prb = float(min(12 * nsym - cfg.dmrs_re_per_prb - cfg.overhead_re_per_prb, 156))
        if cfg.pf_metric == "wideband":
            allm = torch.ones(E, R, S, dtype=torch.bool, device=d)
            wb = phy.eff_sinr_all(est_o, allm, cfg.eff_sinr, w)                        # [E,R,M]
            ok = wb >= phy.thr_ref
            m = (ok.long() * torch.arange(1, phy.M + 1, device=d)).max(-1).values.clamp(min=1) - 1
            rate_sb = (phy.se[m] * re_prb / 8)[..., None] * w
        else:
            rate_sb = phy.se_at(est_o) * re_prb * w / 8                                # bytes per RBG
        metric_sb = rate_sb / self.avg[..., None]
        # retransmission admission: rank pending retx per cell (PF metric) and admit them while their
        # RBG counts fit the carrier, so no retx is left with a partial (unusable) allocation
        rkey = torch.where(has_rx, metric_sb.sum(-1), torch.full_like(metric_sb[..., 0], -1.0))
        order = rkey.argsort(-1, descending=True)
        cum = (rx_nsb * has_rx).gather(-1, order).cumsum(-1)
        adm_sorted = has_rx.gather(-1, order) & (cum <= S)
        admitted = torch.zeros_like(has_rx).scatter(-1, order, adm_sorted)
        has_rx = admitted
        want_cnt = torch.where(has_rx, rx_nsb, torch.where(new_el, n_max, torch.zeros_like(n_max)))
        prio = (1e9 if cfg.retx_priority else 0.0) * has_rx.float()
        # ---- PF allocation, RBG by RBG (greedy; stops when the need is covered) ----
        cnt = torch.zeros(E, R, dtype=torch.long, device=d)
        left = need.float()
        cols = []
        for s in range(S):
            want = (cnt < want_cnt) & (has_rx | (left > 0))
            m = torch.where(want, metric_sb[..., s] + prio, torch.full_like(left, -1.0))
            best, wi = m.max(-1)
            oh = onehot(wi, R) & (best >= 0)[:, None]
            cols.append(oh)
            cnt = cnt + oh.long()
            left = left - rate_sb[..., s] * oh * ~has_rx
        won = torch.stack(cols, -1)
        n_sb = won.sum(-1)
        n_prb = (won * w).sum(-1)
        tx_rx = has_rx & (n_sb == rx_nsb) & (n_sb > 0)
        tx_new = new_el & (n_sb > 0)
        tx = tx_rx | tx_new
        # ---- link adaptation for new TBs ----
        est_tx = self._la_estimate(sinr_ref_db, n_prb, est)
        off = self.olla if cfg.olla else torch.zeros_like(self.olla)
        mcs_new, tbs_new = phy.select_mcs(est_tx, won, off, n_prb, nsym, cfg.dmrs_re_per_prb,
                                          cfg.overhead_re_per_prb, cfg.eff_sinr, w)
        cap_b = (tbs_new // 8 - cfg.tb_overhead_bytes).clamp(min=1)
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
        # ---- decoding on the actual channel ----
        act = self._rx_sinr(sinr_ref_db, n_prb, gain_now)
        if self.sinr_hook is not None:
            # multicell merge point: add same-slot inter-cell interference given this slot's
            # allocation (won [E,R,S], n_prb [E,R]); returns the per-subband SINR used for decoding
            act = self.sinr_hook(g, self.dir, won, n_prb, act)
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
        ok = tx & (torch.rand_like(p_err) >= p_err)
        fail = tx & ~ok
        exh = fail & (ntx >= cfg.max_harq_tx)
        # ---- HARQ state update ----
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
        # ---- OLLA, direction-specific updates, PF ----
        if cfg.olla:
            self.olla = (self.olla + cfg.olla_up_db * ok - self.olla_dn * fail).clamp(-10, 10)
        served = (hi_tx - lo_tx) * ok
        self._post_slot(tx, gain_now)
        self.avg = (1 - 1 / cfg.pf_window) * self.avg + (1 / cfg.pf_window) * served
        # ---- counters ----
        c = self.ctr
        c["tb_new"] += tx_new.sum(); c["tb_retx"] += tx_rx.sum(); c["tb_ok"] += ok.sum()
        c["tb_fail"] += fail.sum(); c["exhaust"] += exh.sum(); c["bytes_ok"] += served.sum()
        c["bytes_new"] += (byt_new * tx_new).sum(); c["new_while_pending"] += (tx_new & pending).sum()
        c["prb_used"] += (n_prb * tx).sum(); c["prb_avail"] += float(w.sum()) * E
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

    # ---------------- end of control step ----------------
    def end_step(self, t, timeout):
        """Masks of delivered, timed-out (purge mode only) and resolved-lost frames."""
        q = self.q
        valid = q.cap >= 0
        delivered = valid & (q.fin <= t + 1 + 1e-9)          # completion + processing offset reached
        if self.cfg.discard == "purge":
            timed = valid & torch.isinf(q.fin) & ((t + 1 - q.cap) >= timeout)
        else:
            timed = torch.zeros_like(valid)
        dropped = valid & q.lost & (q.end <= self.ack_ptr()[..., None]) & ~timed
        return delivered, timed, dropped

    def compact(self, gone):
        self.q.remove(gone)
        self.floor = self.q.floor()
        self.sent = torch.maximum(self.sent, self.floor)
