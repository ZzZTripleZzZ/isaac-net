"""Shared per-slot MAC algorithm for one direction of one cell (NR engine).

`MacLink.slot` implements, for every [E, R] robot at once: HARQ bookkeeping, candidate selection
(retransmission first, else new data on a free process), PF allocation RBG by RBG, link
adaptation, TB binding to HARQ processes, decoding with EESM + HARQ combining, HARQ / OLLA / PF
updates, and the RLC in-order pointer. Direction-specific pieces (SR/BSR, power split, CSI,
feedback timing) are the hooks overridden in mac_ul.UlMac and mac_dl.DlMac.

Multi-cell (set by nr_engine.NRNet when n_cells > 1; all None at one cell, which leaves the single-cell code
path untouched): `member` [E,C,R] makes retransmission admission and PF allocation run one scheduler per cell
over the same RBGs, `sched_ok` [E,R] masks robots inside a handover interruption, and handover() moves a robot's
MAC state to a new cell.

Byte-stream model: a new TB takes stream bytes [sent, sent + TBS/8 - tb_overhead) and binds them
to a free HARQ process, so new data proceeds on other processes while a failed TB waits (no MAC
head-of-line blocking; n_harq = 1 restores it). The in-order pointer is the lowest byte still
owned by an undecoded process (or `sent`). HARQ exhaustion: "rlc_am" resends the range after
rlc_retx_slots; "drop" marks overlapping frames lost (RLC UM).

Schedulers (cfg.scheduler): "pf" proportional fair on the per-RBG rate estimate over the EWMA throughput (wideband
rate with pf_metric="wideband"), "pf_wideband" the same with the wideband rate, "maxci" the rate estimate alone, "rr"
round robin (the robot whose last transmission is oldest first, channel-blind), "qos" the 5G-LENA QoS scheduler
(NrMacSchedulerOfdmaQos, see qos_prepare): the PF metric r^gamma / avg times a per-robot class weight. Every scheduler
keeps the same retransmission admission and the same greedy RBG-by-RBG filling up to the need or the power-headroom cap.

5G-LENA MAC behavior under load (docs/fidelity-load-gap.md; defaults reproduce the engine before them bitwise):
pf_update="rbg" updates the RBG winner's PF average with its granted bytes after every RBG (5G-LENA OFDMA PF, so RBGs
spread over backlogged robots) and pf_avg_idle="freeze" moves the average only for robots with new data to schedule
(both links); ul_retx_sched="tdma" sends one UL retransmission per slot and cell, oldest NACK first, alone in its slot;
ul_amc_alloc="previous" picks the UL MCS for the PRB count of the robot's previous PUSCH and sizes the TB for the
current grant; ul_grant_model="bsr" (mac_ul.py) replaces the lumped SR delay by the SR / BSR pipeline, whose payload
rule enters through _tb_payload.

SU-MIMO rank (NRConfig.n_layers_max = 2, the directions of cfg.mimo_dirs; docs/configurability.md "MIMO rank"): link
adaptation picks the rank of every new TB (_rank: wideband SINR and / or the link's Rician K), selects the MCS on the
per-layer SINR (PHY.layer_sinr_db: SINR - 10 log10(rank) - rank_layer_penalty_db) and sizes the TB over the layers
(tbs_38214 layers=rank). The HARQ process keeps the rank (h_rank) for its retransmissions, and decoding uses the
per-layer SINR of that rank: the layers form one TB (one codeword, one CRC) decoded jointly through the same EESM /
BLER lookup. CQI, OLLA and the scheduler's single-layer rate estimate are unchanged. Off (n_layers_max = 1), no rank
state exists and none of this code runs.

Time: g (slot) and frac (completion time) are Python numbers in the reference, or 0-dim device tensors (long,
float64) when the graph backend captures the step (nr_fast.py); gh is always the host slot index, used only for
decisions that are fixed by the TDD pattern. Random draws: torch.rand_like (cfg.rng="global") or the engine's
counter-based streams (self.rng, nr_rng.py).
"""
from __future__ import annotations

import math

import torch

from .config import NRConfig
from .nr_rng import BLER
from .phy import PHY
from .queues import FrameQueue, env_mask, onehot, reset_where

BIG = 2 ** 62
AVG_MIN = 1e-9             # floor of the PF average (bytes per slot), as 5G-LENA's PF metric max(1e-9, avg)
BSR_DEPTH = 4              # ul_grant_model="bsr": buffer status reports in flight per robot ("K" state dims)
# rank-2 SU-MIMO (a direction in cfg.mimo_dirs): rank of the TB of each HARQ process, rank of the robot's last new TB
MIMO_STATE = {"h_rank": (("P",), torch.long, 1), "last_rank": ((), torch.long, 1)}


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
        "last_tx": ((), torch.long, -1),                    # slot of the last transmission (round robin)
    }
    direction = None

    def __init__(self, cfg: NRConfig, E, R, device, meta=()):
        self.cfg, self.E, self.R, self.dev = cfg, E, R, device
        self.dir = self.direction
        self.mimo = self.dir in cfg.mimo_dirs          # rank-2 SU-MIMO in this direction (n_layers_max = 2)
        if self.mimo:
            self.STATE = {**self.STATE, **MIMO_STATE}
        self.rank_k = None         # callable -> Rician K (linear) [E,R] of the serving link, or None (set by NRNet)
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
        self.sinr_hook = None      # same-slot inter-cell interference, see slot()
        self.member = None         # [E,C,R] serving-cell membership (multi-cell only)
        self.sched_ok = None       # [E,R] schedulable (outside a handover interruption; multi-cell only)
        self.slot_nsym = 14        # data symbols of the slot being processed (set by slot(), read by SINR hooks)
        self.n_cells = 1
        self.rng = None            # nr_rng.NRRng (cfg.rng="engine"; set by NRNet)
        self._bler_site = BLER[self.dir]
        self._w_sum = float(self.sb_prb.sum())
        dims = {"P": self.P, "S": self.S, "K": BSR_DEPTH}
        for n, (ex, dt, v) in self.STATE.items():
            setattr(self, n, torch.full((E, R) + tuple(dims[k] for k in ex), v, dtype=dt, device=device))
        self.ctr = {k: torch.zeros((), device=device) for k in
                    ("tb_new", "tb_retx", "tb_ok", "tb_fail", "exhaust", "bytes_ok", "bytes_new", "lost_frames",
                     "new_while_pending", "prb_used", "prb_avail")}
        self._lena_mac = bool(cfg.lena_mac_switches())   # 5G-LENA MAC switches on: grant accounting counters
        if self._lena_mac:
            for k in ("tb_bytes", "tb_empty", "retx_block"):
                self.ctr[k] = torch.zeros((), device=device)
        self.ntx_hist = torch.zeros(cfg.max_harq_tx + 1, device=device)   # transmissions per decoded TB
        self.rv_tx = torch.zeros(E, cfg.max_harq_tx + 1, device=device)   # per env: TBs sent as transmission k
        self.rv_fail = torch.zeros(E, cfg.max_harq_tx + 1, device=device) # of which failed
        self.prb_used_env = torch.zeros(E, device=device)                 # per env: PRBs granted
        self.qos = cfg.scheduler == "qos"
        if self.qos:               # per-message class from the queue's priority extra; weights set by qos_prepare
            self.q.enable_extras()
            P = cfg.qos_priority
            self._qos_rank = torch.tensor(sorted(range(len(P)), key=lambda c: (P[c], c)), device=device).argsort()
            self.qw = torch.full((E, R), float(100 - max(P)), device=device)

    def qos_prepare(self, tf):
        """scheduler="qos", once per control step before its slots (NRNet.step / step_cells, and before the fused
        kernel of the triton backend), at step start time tf (control steps; a 0-dim float64 tensor when captured).

        Byte assignment by class (5G-LENA NrMacSchedulerLcQos): the robot's queued frames whose bytes are all unsent
        and visible are reordered by class priority, stably (FrameQueue.reorder), so a TB takes the bytes of the
        most important class first and each class stays in arrival order. Frames with bytes already in a HARQ
        process keep their place: one queue per robot, in-order RLC delivery, no pre-emption of sent bytes.

        Class weight qw [E,R] (5G-LENA NrMacSchedulerUeInfoQos), from the classes c with unsent visible bytes: DL
        (CalculateDlWeight) the sum over those classes of (100 - P_c) * D_c; UL (CompareUeWeightsUl, the gNB knows
        the logical-channel groups from the BSR) (100 - P_c) * D_c of the class with the lowest priority level P_c.
        D_c is the delay-budget factor of CalculateDelayBudgetFactor: PDB / (PDB - HOL) while the head-of-line age
        HOL of class c (age of its oldest frame with unsent bytes, ms, at tf) is below PDB, PDB / 0.1 from then on;
        D_c = 1 when qos_pdb_ms[c] is inf (5G-LENA applies D to DC-GBR flows only; the UL factor is an extension,
        5G-LENA UL has none). A robot with no such class (BSR padding grants) gets the lowest class weight."""
        cfg, q = self.cfg, self.q
        Q = cfg.qos_classes
        self.sent = q.reorder(self.sent, self._qos_rank[q.prio.clamp(0, Q - 1)], Q)
        cls = q.prio.clamp(0, Q - 1)                    # after the reorder (the frames moved)
        act_f = (q.cap >= 0) & (q.end > self.sent[..., None]) & (q.start < q.enq[..., None])
        age = ((tf - q.cap.double() - q.off) * cfg.control_step_ms).float()
        acc = torch.zeros(self.E, self.R, device=self.dev)
        best_p = torch.full((self.E, self.R), 100, dtype=torch.long, device=self.dev)
        for c in range(Q):
            m = act_f & (cls == c)
            act = m.any(-1)
            pc, pdb = cfg.qos_priority[c], cfg.qos_pdb_ms[c]
            if math.isfinite(pdb):
                hol = torch.where(m, age, torch.zeros_like(age)).max(-1).values
                wc = (100 - pc) * (pdb / torch.where(hol >= pdb, torch.full_like(hol, 0.1), pdb - hol))
            else:
                wc = torch.full_like(acc, float(100 - pc))
            if self.dir == "dl":
                acc = acc + torch.where(act, wc, torch.zeros_like(wc))
            else:
                better = act & (pc < best_p)
                best_p = torch.where(better, pc, best_p)
                acc = torch.where(better, wc, acc)
        none = ~act_f.any(-1)
        self.qw = torch.where(none, torch.full_like(acc, float(100 - max(cfg.qos_priority))), acc)

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
    def _pre_slot(self, g, gh):
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

    def _tb_payload(self, cap_b, unsent, tx_new, g):
        """Stream bytes a new TB of capacity cap_b carries (ul_grant_model="bsr" holds back the RLC tail)."""
        return torch.minimum(cap_b, unsent)

    def _post_slot(self, tx, gain_now, g=None, tx_new=None, tbs_new=None):
        pass

    # ---------------- one data slot ----------------
    def slot(self, g, frac, nsym, sinr_ref_db, gain_now, ack_slot=0, gh=None, rel=0):
        """Process one data slot at absolute slot g. sinr_ref_db [E,R,S] without fast fading (see the
        direction class), gain_now [E,R,S] fast-fading gain (dB), frac = completion time of this
        slot in control steps, ack_slot = DL HARQ-ACK slot, gh = host slot index (default g), rel = slot index
        inside the control step (engine RNG stream)."""
        cfg, E, R, S, P, d, phy = self.cfg, self.E, self.R, self.S, self.P, self.dev, self.phy
        gh = g if gh is None else gh
        self.slot_nsym = nsym
        w = self.sb_prb
        unsent = self.unsent()
        # DL processes whose ACK has reached the gNB become free
        self.h_state = torch.where((self.h_state == 2) & (self.h_ready <= g), torch.zeros_like(self.h_state), self.h_state)
        self._pre_slot(g, gh)
        # ---- candidates: one TB per UE per slot, retransmissions first ----
        rx_el = (self.h_state == 1) & (self.h_ready <= g)
        rx_p = torch.where(rx_el, self.h_ready, torch.full_like(self.h_ready, BIG)).argmin(-1)
        has_rx = rx_el.any(-1)
        free = self.h_state == 0
        p_new = free.long().argmax(-1)
        need = self._need(unsent)
        new_el = free.any(-1) & (need > 0) & ~has_rx
        if self.sched_ok is not None:
            has_rx = has_rx & self.sched_ok
            new_el = new_el & self.sched_ok
        pending = (self.h_state == 1).any(-1)
        rx_nsb = self.h_nsb.gather(-1, rx_p[..., None]).squeeze(-1)
        # ---- scheduler's per-RBG estimate, with OLLA ----
        est, n_max = self._sched_estimate(sinr_ref_db)
        est_o = est + (self.olla[..., None] if cfg.olla else 0.0)
        re_prb = float(min(12 * nsym - cfg.dmrs_re_per_prb - cfg.overhead_re_per_prb, 156))
        sched = cfg.scheduler
        if cfg.pf_metric == "wideband" or sched == "pf_wideband":
            allm = torch.ones(E, R, S, dtype=torch.bool, device=d)
            wb = phy.eff_sinr_all(est_o, allm, cfg.eff_sinr, w)                        # [E,R,M]
            ok = (wb >= phy.thr_ref) & (torch.arange(phy.M, device=d) <= phy.mcs_max)   # cap as phy.mcs_at
            m = (ok.long() * torch.arange(1, phy.M + 1, device=d)).max(-1).values.clamp(min=1) - 1
            rate_sb = (phy.se[m] * re_prb / 8)[..., None] * w
        else:
            rate_sb = phy.se_at(est_o) * re_prb * w / 8                                # bytes per RBG
        if sched == "maxci":
            metric_sb = rate_sb
        elif sched == "rr":                    # age of the last transmission, the same on every RBG
            metric_sb = (g - self.last_tx).float()[..., None].expand_as(rate_sb)
        elif sched == "qos":                   # 5G-LENA QoS: r^gamma / avg times the class weight (qos_prepare)
            rate_q = rate_sb if cfg.qos_gamma == 1.0 else rate_sb ** cfg.qos_gamma
            metric_sb = rate_q / self.avg.clamp(min=AVG_MIN)[..., None] * self.qw[..., None]
        else:
            metric_sb = rate_sb / self.avg.clamp(min=AVG_MIN)[..., None]
        # retransmission admission: rank pending retx per cell (PF metric) and admit them while their
        # RBG counts fit the carrier, so no retx is left with a partial (unusable) allocation
        M = self.member
        tdma = self.dir == "ul" and cfg.ul_retx_sched == "tdma"
        rkey = None if tdma else torch.where(has_rx, metric_sb.sum(-1), torch.full_like(metric_sb[..., 0], -1.0))
        if tdma:
            # 5G-LENA UL HARQ (NrMacSchedulerHarqRr::ScheduleUlHarq): one retransmission per slot and cell, the
            # oldest NACK first, on every data symbol, so no new data is scheduled in that slot
            rdy = torch.where(has_rx, self.h_ready.gather(-1, rx_p[..., None]).squeeze(-1), torch.full_like(rx_nsb, BIG))
            if M is None:
                first = rdy.argmin(-1)
                admitted = onehot(first, R) & has_rx.any(-1, keepdim=True) & has_rx
                block = admitted.any(-1, keepdim=True)
                n_block = (block[:, 0] & (rx_nsb * admitted).sum(-1).lt(S)).sum()
            else:
                rx_c = has_rx[:, None, :] & M
                first = torch.where(M, rdy[:, None, :], torch.full_like(M, BIG, dtype=rdy.dtype)).argmin(-1)
                adm_c = onehot(first, R) & rx_c
                admitted = adm_c.any(1)
                blk_c = adm_c.any(-1)
                block = (blk_c[..., None] & M).any(1)
                n_block = (blk_c & (rx_nsb[:, None, :] * adm_c).sum(-1).lt(S)).sum()
            new_el = new_el & ~block
            self.ctr["retx_block"] += n_block
        elif M is None:
            order = rkey.argsort(-1, descending=True)
            cum = (rx_nsb * has_rx).gather(-1, order).cumsum(-1)
            adm_sorted = has_rx.gather(-1, order) & (cum <= S)
            admitted = torch.zeros_like(has_rx).scatter(-1, order, adm_sorted)
        else:                   # the same ranking inside every cell: [E,C,R], then back to [E,R]
            rx_c = has_rx[:, None, :] & M
            order = torch.where(rx_c, rkey[:, None, :], torch.full_like(M, -1.0, dtype=rkey.dtype)).argsort(
                -1, descending=True)
            cum = (rx_nsb[:, None, :] * rx_c).gather(-1, order).cumsum(-1)
            adm_sorted = rx_c.gather(-1, order) & (cum <= S)
            admitted = torch.zeros_like(rx_c).scatter(-1, order, adm_sorted).any(1)
        has_rx = admitted
        want_cnt = torch.where(has_rx, rx_nsb, torch.where(new_el, n_max, torch.zeros_like(n_max)))
        lex = cfg.retx_priority
        prio = (1e9 if lex else 0.0) * has_rx.float()
        # ---- PF allocation, RBG by RBG (greedy; stops when the need is covered) ----
        # retx_priority: an admitted retransmission that still wants RBGs wins RBG s over all new data (lexicographic
        # priority: while one wants it, the new-data robots of its cell are masked out). Among retransmissions the key
        # stays metric + 1e9 in float32, as before, so whenever every metric is below 1e9 the choice is unchanged; the
        # mask only matters when a PF metric (rate / tiny average) reaches the 1e9 bonus, where a new-data robot used
        # to take RBGs an admitted retransmission needed, leaving it short of its RBG count and unable to send.
        cnt = torch.zeros(E, R, dtype=torch.long, device=d)
        left = need.float()
        cols = []
        rbg_pf = cfg.pf_update == "rbg"
        if rbg_pf:              # 5G-LENA OFDMA PF: the winner's average moves after every RBG, before the next
            wwin = 1.0 / cfg.pf_window
            base = (1 - wwin) * self.avg
            got = torch.zeros(E, R, device=d)                  # bytes granted so far in this slot
        for s in range(S):
            want = (cnt < want_cnt) & (has_rx | (left > 0))
            ms = self._rbg_metric(s, rate_sb, base, wwin, got) if rbg_pf else metric_sb[..., s]
            m = torch.where(want, ms + prio, torch.full_like(left, -1.0))
            wr = want & has_rx if lex else None
            if M is None:
                if lex:
                    m = torch.where(wr.any(-1, keepdim=True) & ~wr, torch.full_like(m, -1.0), m)
                best, wi = m.max(-1)
                oh = onehot(wi, R) & (best >= 0)[:, None]
            else:               # one PF scheduler per cell on RBG s
                mc = torch.where(M, m[:, None, :], torch.full_like(m[:, None, :], -1.0))               # [E,C,R]
                if lex:
                    wr_c = wr[:, None, :] & M
                    mc = torch.where(wr_c.any(-1, keepdim=True) & ~wr_c, torch.full_like(mc, -1.0), mc)
                best, wi = mc.max(-1)                                                                   # [E,C]
                oh = (onehot(wi, R) & (best >= 0)[..., None]).any(1)
            cols.append(oh)
            cnt = cnt + oh.long()
            if rbg_pf:
                got = got + rate_sb[..., s] * oh
            left = left - rate_sb[..., s] * oh * ~has_rx
        won = torch.stack(cols, -1)
        if not lex:
            # retx_priority=False: retransmissions compete on the PF metric like new data, so an admitted one may win
            # fewer RBGs than its TB needs and then cannot be sent. Its RBGs are released and offered, RBG by RBG, to
            # the new-data robots of its cell that still want RBGs (same metric, same greedy rule), instead of staying
            # empty. RBGs of the retransmissions that were sent are untouched.
            short = has_rx & (won.sum(-1) != rx_nsb)
            rel_rs = won & short[..., None]                                                      # released [E,R,S]
            won = won & ~short[..., None]
            cnt = won.sum(-1)
            cols = []
            for s in range(S):
                want = (cnt < want_cnt) & new_el & (left > 0)
                ms = self._rbg_metric(s, rate_sb, base, wwin, got) if rbg_pf else metric_sb[..., s]
                m = torch.where(want, ms, torch.full_like(left, -1.0))
                if M is None:
                    best, wi = m.max(-1)
                    oh = onehot(wi, R) & ((best >= 0) & rel_rs[..., s].any(-1))[:, None]
                else:
                    best, wi = torch.where(M, m[:, None, :], torch.full_like(m[:, None, :], -1.0)).max(-1)
                    av = (rel_rs[..., s][:, None, :] & M).any(-1)                                    # [E,C]
                    oh = (onehot(wi, R) & ((best >= 0) & av)[..., None]).any(1)
                cols.append(oh)
                cnt = cnt + oh.long()
                if rbg_pf:
                    got = got + rate_sb[..., s] * oh
                left = left - rate_sb[..., s] * oh
            won = won | torch.stack(cols, -1)
        n_sb = won.sum(-1)
        n_prb = (won * w).sum(-1)
        tx_rx = has_rx & (n_sb == rx_nsb) & (n_sb > 0)
        tx_new = new_el & (n_sb > 0)
        tx = tx_rx | tx_new
        # ---- link adaptation for new TBs ----
        est_tx = self._la_estimate(sinr_ref_db, n_prb, est)
        off = self.olla if cfg.olla else torch.zeros_like(self.olla)
        lay = 1
        if self.mimo:           # rank of the new TB, then the MCS on the per-layer SINR and the TBS over the layers
            rank_new = self._rank(est_tx)
            est_tx = phy.layer_sinr_db(est_tx, rank_new[..., None], cfg.rank_layer_penalty_db)
            lay = rank_new
        amc_prev = self.dir == "ul" and cfg.ul_amc_alloc == "previous"
        if amc_prev:            # 5G-LENA UL AMC: MCS for the PRBs of the previous PUSCH, TB for this allocation
            ref_prb = torch.where(self.last_nprb > 0, self.last_nprb, n_prb)
            ref_won = torch.ones_like(won) if cfg.ul_power == "whole_band" else won
            mcs_new, _ = phy.select_mcs(est_tx, ref_won, off, ref_prb, nsym, cfg.dmrs_re_per_prb,
                                        cfg.overhead_re_per_prb, cfg.eff_sinr, w, layers=lay)
            tbs_new = phy.tbs_all(n_prb, nsym, cfg.dmrs_re_per_prb, cfg.overhead_re_per_prb, lay).gather(
                -1, mcs_new[..., None]).squeeze(-1)
        else:
            mcs_new, tbs_new = phy.select_mcs(est_tx, won, off, n_prb, nsym, cfg.dmrs_re_per_prb,
                                              cfg.overhead_re_per_prb, cfg.eff_sinr, w, layers=lay)
        cap_b = (tbs_new // 8 - cfg.tb_overhead_bytes).clamp(min=1)
        byt_new = self._tb_payload(cap_b, unsent, tx_new, g)
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
        if self.mimo:           # the process keeps the rank of its TB for every retransmission
            self.h_rank = torch.where(ohn, rank_new[..., None], self.h_rank)
            self.last_rank = torch.where(tx_new, rank_new, self.last_rank)
        self.sent = self.sent + byt_new * tx_new
        g1 = lambda x: x.gather(-1, p_tx[..., None]).squeeze(-1)
        mcs, tbs = g1(self.h_mcs), g1(self.h_tbs)
        ntx = g1(self.h_ntx) + 1
        # ---- decoding on the actual channel ----
        act = self._rx_sinr(sinr_ref_db, n_prb, gain_now)
        if self.sinr_hook is not None:
            # same-slot inter-cell interference given this slot's transmissions (won [E,R,S] of the robots
            # that transmit, n_prb [E,R]); returns the per-subband SINR used for decoding
            act = self.sinr_hook(g, self.dir, won & tx[..., None], n_prb, act)
        if self.mimo:           # per-layer SINR of the TB's rank (both layers decoded jointly as one TB)
            act = phy.layer_sinr_db(act, g1(self.h_rank)[..., None], cfg.rank_layer_penalty_db)
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
        # ---- HARQ state update ----
        ohp_ok, ohp_fail, ohp_exh = ohp & ok[..., None], ohp & fail[..., None], ohp & exh[..., None]
        st_ok, rdy_ok, rdy_fail = self._harq_times(g, ack_slot)
        self.h_state = torch.where(ohp_ok, torch.full_like(self.h_state, st_ok), self.h_state)
        self.h_ready = torch.where(ohp_ok, rdy_ok, self.h_ready)
        self.h_ntx = torch.where(ohp, ntx[..., None], self.h_ntx)
        self.h_comb = torch.where(ohp, comb[..., None], self.h_comb)
        self.h_ready = torch.where(ohp_fail, rdy_fail, self.h_ready)
        lo_tx, hi_tx = g1(self.h_lo), g1(self.h_hi)
        if self.trace is not None:
            self.trace.append((frac, tx.cpu(), ok.cpu(), lo_tx.cpu(), hi_tx.cpu(), exh.cpu()))
        if cfg.harq_fail == "rlc_am":
            self.h_ntx = torch.where(ohp_exh, torch.zeros_like(self.h_ntx), self.h_ntx)
            self.h_comb = torch.where(ohp_exh, torch.zeros_like(self.h_comb), self.h_comb)
            self.h_lexp = torch.where(ohp_exh, torch.full_like(self.h_lexp, -float("inf")), self.h_lexp)
            self.h_nrb = torch.where(ohp_exh, torch.zeros_like(self.h_nrb), self.h_nrb)
            self.h_ready = torch.where(ohp_exh, g + cfg.rlc_retx_slots, self.h_ready)
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
        self._post_slot(tx, gain_now, g, tx_new, tbs_new)
        if amc_prev:
            self.last_nprb = torch.where(tx, n_prb, self.last_nprb)
        self._pf_update(g, served, tx, won, need, has_rx, tbs_new, tx_new)
        # ---- counters ----
        c = self.ctr
        c["tb_new"] += tx_new.sum(); c["tb_retx"] += tx_rx.sum(); c["tb_ok"] += ok.sum()
        c["tb_fail"] += fail.sum(); c["exhaust"] += exh.sum(); c["bytes_ok"] += served.sum()
        c["bytes_new"] += (byt_new * tx_new).sum(); c["new_while_pending"] += (tx_new & pending).sum()
        c["prb_used"] += (n_prb * tx).sum(); c["prb_avail"] += self._w_sum * E * self.n_cells
        if self._lena_mac:
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
        q.fin = torch.where(done, frac + cfg.proc_offset_ms / cfg.control_step_ms, q.fin)

    def _rank(self, est):
        """Rank (1 or 2) of a new TB per robot [E,R] from the link-adaptation estimate est [E,R,S] (per-PRB SINR at
        this grant's PSD, before OLLA and before the layer split) and the link's Rician K (rank_k, None or absent
        = K 0: Rayleigh, NLOS). rank_rule "sinr": wideband SINR (linear mean over the RBGs, PRB-weighted) >=
        rank_sinr_min_db; "los": K < rank_k_max_db (rich scattering); "sinr_los": both."""
        cfg = self.cfg
        two = torch.ones(self.E, self.R, dtype=torch.bool, device=self.dev)
        if cfg.rank_rule != "los":
            lin = (10 ** (est.clamp(-30, 60) / 10) * self.sb_prb).sum(-1) / self._w_sum
            two = two & (10 * torch.log10(lin) >= cfg.rank_sinr_min_db)
        if cfg.rank_rule != "sinr":
            k = None if self.rank_k is None else self.rank_k()
            if k is not None:
                two = two & (k < 10 ** (cfg.rank_k_max_db / 10))
        return 1 + two.long()

    def _rbg_metric(self, s, rate_sb, base, wwin, got):
        """pf_update="rbg": metric of RBG s with the average moved by the bytes granted so far in this slot (qos: the
        rate to the power qos_gamma, times the class weight)."""
        if not self.qos:
            return rate_sb[..., s] / (base + wwin * got).clamp(min=1e-9)
        r = rate_sb[..., s] if self.cfg.qos_gamma == 1.0 else rate_sb[..., s] ** self.cfg.qos_gamma
        return r / (base + wwin * got).clamp(min=1e-9) * self.qw

    def _pf_update(self, g, served, tx, won, need=None, has_rx=None, tbs_new=None, tx_new=None):
        """Scheduler state after a data slot: PF average and, for round robin, the slot of the last transmission.
        pf_update="slot": EWMA of the served bytes of every robot; "rbg": of the granted TB bytes of new TBs (the
        per-RBG updates of the slot, committed). pf_avg_idle="freeze": only robots with new data to schedule in this
        slot (need > 0, no retransmission admitted) update their average (5G-LENA's active list)."""
        cfg = self.cfg
        if cfg.pf_update == "slot" and cfg.pf_avg_idle == "decay":
            # floored at AVG_MIN: an idle robot's average decays geometrically and would underflow float32 to 0
            # (an infinite PF metric); the floor binds only after about 2,500 idle slots at pf_window=100, where
            # rate / avg already exceeded the 1e9 retransmission bonus for any RBG carrying a byte
            self.avg = ((1 - 1 / cfg.pf_window) * self.avg + (1 / cfg.pf_window) * served).clamp(min=AVG_MIN)
        else:
            wwin = 1.0 / cfg.pf_window
            upd = (tbs_new // 8).float() * tx_new if cfg.pf_update == "rbg" else served.float()
            new_avg = ((1 - wwin) * self.avg + wwin * upd).clamp(min=AVG_MIN)
            if cfg.pf_avg_idle == "freeze":
                self.avg = torch.where((need > 0) & ~has_rx, new_avg, self.avg)
            else:
                self.avg = new_avg
        if cfg.scheduler == "rr":
            self.last_tx = torch.where(tx, g, self.last_tx)

    # ---------------- handover ----------------
    def handover(self, ho, flush=False):
        """Robots ho [E,R] move to a new cell. The target starts OLLA, the PF average and the CSI afresh. HARQ
        processes that own undecoded bytes restart their combining at the target (the TB is sent again from its
        first transmission); under RLC UM (harq_fail="drop") or flush=True (ho_rlc="flush") they are lost
        instead, and flush also drops every queued frame not yet completed (the unsent bytes are skipped)."""
        cfg, q = self.cfg, self.q
        hp = ho[..., None]
        own = (self.h_state == 1) & hp
        if cfg.harq_fail == "drop" or flush:
            valid = (q.cap >= 0) & torch.isinf(q.fin) & ~q.lost
            ov = (own[..., None, :] & (q.start[..., None] < self.h_hi[..., None, :])
                  & (q.end[..., None] > self.h_lo[..., None, :])).any(-1) & valid
            if flush:
                ov = ov | (valid & ho[..., None])
                self.sent = torch.where(ho, q.enq, self.sent)
            self.ctr["lost_frames"] += ov.sum()
            q.lost = q.lost | ov
            self.h_state = torch.where(own, torch.zeros_like(self.h_state), self.h_state)
        else:
            self.h_ntx = torch.where(own, torch.zeros_like(self.h_ntx), self.h_ntx)
            self.h_comb = torch.where(own, torch.zeros_like(self.h_comb), self.h_comb)
            self.h_lexp = torch.where(own, torch.full_like(self.h_lexp, -float("inf")), self.h_lexp)
            self.h_nrb = torch.where(own, torch.zeros_like(self.h_nrb), self.h_nrb)
        for n in ("olla", "avg", "csi"):
            x, v = getattr(self, n), self.STATE[n][2]
            setattr(self, n, torch.where(ho.view(*ho.shape, *([1] * (x.dim() - 2))), torch.full_like(x, v), x))

    # ---------------- end of control step ----------------
    def end_step(self, t, timeout):
        """Masks of delivered, timed-out (purge mode only) and resolved-lost frames."""
        q = self.q
        valid = q.cap >= 0
        lim = t + 1 + 1e-9 if not torch.is_tensor(t) else (t + 1).double() + 1e-9
        delivered = valid & (q.fin <= lim)                   # completion + processing offset reached
        if self.cfg.discard == "purge":
            timed = valid & torch.isinf(q.fin) & ((t + 1 - q.cap) >= timeout)
        else:
            timed = torch.zeros_like(valid)
        dropped = valid & q.lost & (q.end <= self.ack_ptr()[..., None]) & ~timed
        return delivered, timed, dropped

    def compact(self, gone):
        """Remove the resolved frames gone [E,R,F], advance the stream floor, and free the HARQ processes whose
        bytes all left with them.

        A process whose TB carried stream bytes that all lie below the new floor (h_lo < h_hi <= floor) holds only
        purged data (discard="purge" timeouts): it returns to the fresh state (free, no transmissions, no combining),
        so it neither retransmits dead bytes at retransmission priority nor, under harq_fail="rlc_am", loops on RLC
        resends forever. A partially purged process (h_lo < floor < h_hi) is kept: its upper bytes belong to the
        head frame, which is still queued, and freeing it would let ack_ptr pass bytes that were never decoded (a
        false in-order delivery). A TB that carried no stream bytes (h_lo == h_hi, padding of the 5G-LENA BSR grant
        model) purged nothing and is left to its HARQ process as before. Without purging, frames leave only once
        every byte below them is resolved, so no busy process lies wholly below the floor and nothing changes."""
        self.q.remove(gone)
        self.floor = self.q.floor()
        self.sent = torch.maximum(self.sent, self.floor)
        dead = (self.h_state == 1) & (self.h_hi <= self.floor[..., None]) & (self.h_hi > self.h_lo)
        for n in ("h_state", "h_ready", "h_ntx", "h_comb", "h_lexp", "h_nrb"):
            x = getattr(self, n)
            setattr(self, n, torch.where(dead, torch.full_like(x, self.STATE[n][2]), x))
