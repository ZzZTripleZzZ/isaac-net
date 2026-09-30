"""Triton side of the NR engine: the engine RNG in uint32 (nr_rng.py) and the fused per-control-step kernel of the
triton backend (see the second half of this file and nr_fast.NRTritonEngine). Imported lazily; needs Triton and CUDA.
"""
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


# ---------------------------------------------------------------------------------------------- RNG (nr_rng.py)
@triton.jit
def mix32(x):
    x = x ^ (x >> 16)
    x = x * tl.full([], 0x21F0AAAD, tl.uint32)
    x = x ^ (x >> 15)
    x = x * tl.full([], 0x735A2D97, tl.uint32)
    return x ^ (x >> 15)


@triton.jit
def salt(v):
    return mix32(v) + tl.full([], 0x9E3779B9, tl.uint32)


@triton.jit
def u32(x):
    """Low 32 bits of an int64 / int32 value as uint32."""
    return (x.to(tl.int64) & 0xFFFFFFFF).to(tl.uint32)


@triton.jit
def rng_key(s0, env, ep, chs, ctr, sts):
    """Base key of (seed key s0, env, episode, salted channel, counter, salted stream); all uint32."""
    h = mix32(s0 ^ salt(env))
    h = mix32(h ^ salt(ep))
    h = mix32(h ^ chs)
    h = mix32(h ^ salt(ctr))
    return mix32(h ^ sts)


@triton.jit
def rng_elem(base, i):
    w = u32(i) * tl.full([], 0x9E3779B9, tl.uint32)
    return mix32(mix32(base + w) ^ base)


@triton.jit
def rng_uniform(base, i):
    return (rng_elem(base, i) >> 8).to(tl.float32) * 5.9604644775390625e-08


@triton.jit
def rng_normal(base, i):
    h1 = rng_elem(base, 2 * i)
    h2 = rng_elem(base, 2 * i + 1)
    u1 = ((h1 >> 8) + 1).to(tl.float32) * 5.9604644775390625e-08
    u2 = (h2 >> 8).to(tl.float32) * 5.9604644775390625e-08
    return libdevice.sqrt(-2.0 * libdevice.log(u1)) * libdevice.cos(6.283185307179586 * u2)


@triton.jit(do_not_specialize=["s0", "chs", "sts"])
def _draw_kernel(out_ptr, env_ptr, ep_ptr, ctr_ptr, s0, chs, sts, N, NORMAL: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    i = tl.program_id(1).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    m = i < N
    base = rng_key(u32(s0), u32(tl.load(env_ptr + row)), u32(tl.load(ep_ptr + row)), u32(chs),
                   u32(tl.load(ctr_ptr + row)), u32(sts))
    if NORMAL:
        z = rng_normal(base, i)
    else:
        z = rng_uniform(base, i)
    tl.store(out_ptr + row * N + i, z, mask=m)


def launch_draw(env, ep, ctr, s0, chs, sts, n, normal):
    rows = env.shape[0]
    out = torch.empty(rows, n, dtype=torch.float32, device=env.device)
    if rows == 0 or n == 0:
        return out
    BLOCK = 1024 if n >= 1024 else max(16, triton.next_power_of_2(n))
    _draw_kernel[(rows, triton.cdiv(n, BLOCK))](out, env, ep, ctr, s0, chs, sts, n, NORMAL=bool(normal), BLOCK=BLOCK)
    return out


# ---------------------------------------------------------------------------------------------- fused step kernel
# One program per env runs every scheduled slot of one control step (fading evolution, SR, UL and optionally DL data
# slots and CQI reports) with the robot x {HARQ process, frame, subband, MCS} state in registers. Semantics follow
# mac.MacLink.slot, mac_ul.UlMac, mac_dl.DlMac and nr_engine.NRNet.step at one cell; see NRTritonEngine in
# nr_fast.py for what runs outside the kernel and for the equivalence methodology.
NEG_INF = float("-inf")


@triton.jit
def _col(x, sidx, s):
    """Column s of x [RB, SB] -> [RB]."""
    return tl.sum(tl.where(sidx[None, :] == s, x, 0.0), axis=1)


@triton.jit
def _db(x):
    """phy._db: 10 log10(max(x, 1e-6)) with NaN -> -30, +inf -> 60, -inf -> -30."""
    y = 10.0 * libdevice.log10(tl.maximum(x, 1e-6))
    y = tl.where(x != x, -30.0, y)
    return tl.where(y == float("inf"), 60.0, y)


@triton.jit
def _lse_rows(lw):
    """logsumexp over the last axis of lw [RB, SB] (entries may be -inf) -> [RB]."""
    mx = tl.max(lw, axis=1)
    e = tl.where(lw == float("-inf"), 0.0, libdevice.exp(lw - mx[:, None]))
    return tl.where(mx == float("-inf"), float("-inf"), mx + libdevice.log(tl.sum(e, axis=1)))


@triton.jit
def _eff_m(lin, est, lw, lnw, beta, MODE: tl.constexpr):
    """phy.eff_sinr_all for one MCS (scalar beta): effective SINR (dB) over the subbands with lw > -inf [RB]."""
    if MODE == 1:          # mean_db: PRB-weighted mean of the dB values
        wgt = tl.where(lw == float("-inf"), 0.0, libdevice.exp(lw - lnw[:, None]))
        v = tl.sum(tl.where(wgt != 0.0, est * wgt, 0.0), axis=1)
        return tl.where(lnw == float("-inf"), 0.0, v)
    return _db(-beta * (_lse_rows(-lin / beta + lw) - lnw))


@triton.jit
def _eff_one(sinr, lw, lnw, beta, MODE: tl.constexpr):
    """phy.eff_sinr for one MCS per robot (beta [RB]) -> [RB] dB; and the EESM log-sum (for IR combining)."""
    lin = libdevice.exp10(tl.minimum(tl.maximum(sinr, -30.0), 60.0) / 10.0)
    lse = _lse_rows(-lin / beta[:, None] + lw)
    if MODE == 1:
        wgt = tl.where(lw == float("-inf"), 0.0, libdevice.exp(lw - lnw[:, None]))
        v = tl.sum(tl.where(wgt != 0.0, sinr * wgt, 0.0), axis=1)
        eff = tl.where(lnw == float("-inf"), 0.0, v)
    else:
        eff = _db(-beta * (lse - lnw))
    return eff, lse


@triton.jit
def _bler(tab_ptr, m, sinr, i0, wi, bg, M: tl.constexpr, C: tl.constexpr, G: tl.constexpr, S0, DS,
          STEP: tl.constexpr, mask):
    """phy.bler_lookup: code-block BLER of MCS m at SINR sinr (dB), code-block index i0 (+ weight wi), base graph."""
    x0 = tl.where(sinr != sinr, -30.0, sinr)
    base0 = ((bg * M + m) * C + i0) * G
    if STEP:
        j = tl.minimum(tl.maximum(libdevice.floor((x0 - S0) / DS + 1e-6), 0.0), G - 1.0).to(tl.int32)
        return tl.load(tab_ptr + base0 + j, mask=mask, other=1.0)
    x = tl.minimum(tl.maximum((x0 - S0) / DS, 0.0), G - 1.0)
    j0 = tl.minimum(libdevice.floor(x), G - 2.0)
    wj = x - j0
    j0i = j0.to(tl.int32)
    lo = tl.load(tab_ptr + base0 + j0i, mask=mask, other=1.0) * (1 - wj) + \
        tl.load(tab_ptr + base0 + j0i + 1, mask=mask, other=1.0) * wj
    hi = tl.load(tab_ptr + base0 + G + j0i, mask=mask, other=1.0) * (1 - wj) + \
        tl.load(tab_ptr + base0 + G + j0i + 1, mask=mask, other=1.0) * wj
    return lo * (1 - wi) + hi * wi


@triton.jit
def _cidx(cbs, cax_ptr, C0, DC, C: tl.constexpr, STEP: tl.constexpr):
    """phy._cidx: code-block-size index (and interpolation weight) of cbs [RB] (float32)."""
    if STEP:
        cnt = tl.zeros(cbs.shape, tl.int32)
        for c in tl.static_range(C):
            cnt += (tl.load(cax_ptr + c) <= cbs).to(tl.int32)
        return tl.maximum(cnt - 1, 0), tl.zeros(cbs.shape, tl.float32)
    x = tl.minimum(tl.maximum((cbs - C0) / DC, 0.0), C - 1.0)
    i0 = tl.minimum(libdevice.floor(x), C - 2.0)
    return i0.to(tl.int32), x - i0


@triton.jit
def _segment(tbs, r, lift_ptr, NL: tl.constexpr, STEP: tl.constexpr):
    """phy.segment (Sionna) or phy.segment_ldpc_k (LENA) of TBS tbs [RB] (int64) at code rate r (float32):
    (code-block size float32, number of code blocks float32, base graph int32). Float64 as in phy.py."""
    t = tbs.to(tl.float64)
    r64 = r.to(tl.float64)
    if STEP:
        bg2 = (t <= 292) | (r64 <= 0.25) | ((t <= 3824) & (r64 <= 0.67))
        b = t + 24
        kcb = tl.where(bg2, 3840.0, 8448.0)
        c = tl.where(b <= kcb, 1.0, libdevice.ceil(b / (kcb - 24)))
        b1 = tl.where(c > 1, b + 24 * c, b)
        k1 = libdevice.floor(b1 / c)
        kb2 = tl.where(b > 640, 10.0, tl.where(b > 560, 9.0, tl.where(b > 192, 8.0, 6.0)))
        kb = tl.where(bg2, kb2, 22.0)
        q = k1 / kb
        zi = tl.zeros(tbs.shape, tl.int32)
        for i in tl.static_range(NL):
            zi += (tl.load(lift_ptr + i) < q).to(tl.int32)
        zc = tl.load(lift_ptr + tl.minimum(zi, NL - 1))
        k = zc * tl.where(bg2, 10.0, 22.0)
        return k.to(tl.float32), c.to(tl.float32), bg2.to(tl.int32)
    bg2 = (t <= 292) | ((t <= 3824) & (r64 <= 0.67)) | (r64 <= 0.25)
    kcb = tl.where(bg2, 3840.0, 8448.0)
    b = t + tl.where(t > 3824, 24.0, 16.0)
    c = tl.where(b <= kcb, 1.0, libdevice.ceil(b / (kcb - 24)))
    bp = tl.where(c > 1, b + 24 * c, b)
    return (bp / c).to(tl.float32), c.to(tl.float32), tl.zeros(tbs.shape, tl.int32)


@triton.jit
def _gather_p(x, pidx, p):
    """x [RB, PB] at process p [RB] -> [RB]."""
    return tl.sum(tl.where(pidx[None, :] == p[:, None], x, tl.zeros_like(x)), axis=1)


@triton.jit
def _mac_slot(
        # per-robot state [RB]
        sent, floor, olla, avg, bsr, sr_t, last_tx, enq,
        # per-subband state / inputs [RB, SB]
        csi, ref, gain, pc,
        # HARQ [RB, PB]
        h_st, h_lo, h_hi, h_rdy, h_ntx, h_mcs, h_tbs, h_nsb, h_comb, h_lexp, h_nrb,
        # queue [RB, FB]
        cap, qstart, qend, lost, fin,
        # per-env accumulators
        cnt_acc, hist_ok, hist_tx, hist_fail,
        # slot
        g, fin_val, re_prb, nsi, pg_now, ack_slot, u,
        # indices / masks
        ridx, rm, sidx, sm, pidx, midx, fidx, hidx, w, lw_all,
        # tables
        tab_ptr, thr_ptr, se_ptr, beta_ptr, rate_ptr, eq_ptr, lift_ptr, cax_ptr,
        tbs_ptr, cbs_ptr, ncb_ptr, bg_ptr, ci0_ptr, cwi_ptr,
        S0, DS, C0, DC,
        # config
        DIR: tl.constexpr, S_: tl.constexpr, SB: tl.constexpr, M: tl.constexpr, MB: tl.constexpr,
        C: tl.constexpr, G: tl.constexpr, NL: tl.constexpr, EQW: tl.constexpr, NPRB: tl.constexpr,
        MODE: tl.constexpr, COMB: tl.constexpr, SCHED: tl.constexpr, WIDEBAND: tl.constexpr,
        HARQ_DROP: tl.constexpr, OLLA: tl.constexpr, PHR_CAP: tl.constexpr, WHOLE_BAND: tl.constexpr,
        PC: tl.constexpr, STEP: tl.constexpr, RETX_PRIO: tl.constexpr,
        MCS_MAX: tl.constexpr, MAX_TX: tl.constexpr, TARGET: tl.constexpr, TB_OH: tl.constexpr,
        SR_DELAY: tl.constexpr, UL_RTT: tl.constexpr, RLC_RETX: tl.constexpr, GNB_PROC: tl.constexpr,
        REF_PRBS: tl.constexpr, PHR_MIN: tl.constexpr, WB_DB: tl.constexpr, W0: tl.constexpr,
        OLLA_UP: tl.constexpr, OLLA_DN: tl.constexpr, PF_A: tl.constexpr, PF_B: tl.constexpr):
    BIG: tl.constexpr = 2 ** 62
    unsent = enq - sent
    # DL processes whose ACK has reached the gNB become free
    h_st = tl.where((h_st == 2) & (h_rdy <= g), 0, h_st)
    if DIR == 0:           # UL: grants
        if pg_now:
            bsr = tl.maximum(bsr, 1)
        granted = (sr_t >= 0) & (g - sr_t >= SR_DELAY)
        bsr = tl.where(granted, tl.maximum(bsr, 1), bsr)
        sr_t = tl.where(granted, -1, sr_t)
    # ---- candidates ----
    rx_el = (h_st == 1) & (h_rdy <= g)
    rx_p = tl.argmin(tl.where(rx_el, h_rdy, BIG), axis=1)
    has_rx = tl.max(rx_el.to(tl.int32), axis=1) > 0
    free = h_st == 0
    p_new = tl.argmax(free.to(tl.int32), axis=1)
    if DIR == 0:
        need = tl.minimum(bsr, unsent)
    else:
        need = unsent
    new_el = (tl.max(free.to(tl.int32), axis=1) > 0) & (need > 0) & (~has_rx) & rm
    has_rx = has_rx & rm
    pending = tl.max((h_st == 1).to(tl.int32), axis=1) > 0
    rx_nsb = _gather_p(h_nsb, pidx, rx_p).to(tl.int64)
    # ---- scheduler estimate ----
    if DIR == 0:
        if WHOLE_BAND:
            if PC:
                est = ref - tl.maximum(pc, WB_DB)[:, None] + csi
            else:
                est = ref - WB_DB + csi
            n_max = tl.full(sent.shape, S_, tl.int64)
        else:
            wdb = 10.0 * libdevice.log10(w / REF_PRBS)
            if PC:
                est = ref - tl.maximum(wdb[None, :], pc[:, None]) + csi
            else:
                est = ref - wdb[None, :] + csi
            if PHR_CAP:
                mref = tl.sum(tl.where(sm[None, :], ref, 0.0), axis=1) / S_
                nmp = REF_PRBS * libdevice.exp10((mref - PHR_MIN) / 10.0)
                n_max = tl.minimum(tl.maximum(libdevice.floor(nmp / W0), 1.0), S_ * 1.0).to(tl.int64)
            else:
                n_max = tl.full(sent.shape, S_, tl.int64)
    else:
        est = ref + csi
        n_max = tl.full(sent.shape, S_, tl.int64)
    if OLLA:
        est_o = est + olla[:, None]
    else:
        est_o = est
    # ---- rate per RBG ----
    if WIDEBAND:
        lw_full = tl.where(sm[None, :] & rm[:, None], lw_all[None, :], float("-inf"))
        lnw_full = _lse_rows(lw_full)
        lin_o = libdevice.exp10(tl.minimum(tl.maximum(est_o, -30.0), 60.0) / 10.0)
        mw = tl.zeros(sent.shape, tl.int32)
        for m in range(M):
            wbm = _eff_m(lin_o, est_o, lw_full, lnw_full, tl.load(beta_ptr + m), MODE)
            mw = tl.where(wbm >= tl.load(thr_ptr + m), m, mw)
        rate = (tl.load(se_ptr + mw) * re_prb / 8.0)[:, None] * w[None, :]
    else:
        mi = tl.zeros(est_o.shape, tl.int32)
        for m in tl.static_range(M):
            if m <= MCS_MAX:
                mi = tl.where(est_o >= tl.load(thr_ptr + m), m, mi)
        rate = tl.load(se_ptr + mi) * re_prb * w[None, :] / 8.0
    if SCHED == 1:
        metric = rate
    elif SCHED == 2:
        metric = (g - last_tx).to(tl.float32)[:, None] + 0.0 * rate
    else:
        metric = rate / avg[:, None]
    metric = tl.where(sm[None, :], metric, 0.0)
    # ---- retransmission admission: rank by the metric sum, admit while the RBG counts fit ----
    rkey = tl.where(has_rx, tl.sum(metric, axis=1), -1.0)
    rkey = tl.where(rm, rkey, float("-inf"))
    val = tl.where(has_rx, rx_nsb, 0)
    before = (rkey[None, :] > rkey[:, None]) | ((rkey[None, :] == rkey[:, None]) & (ridx[None, :] <= ridx[:, None]))
    cum = tl.sum(tl.where(before, val[None, :], 0), axis=1)
    has_rx = has_rx & (cum <= S_)
    want_cnt = tl.where(has_rx, rx_nsb, tl.where(new_el, n_max, 0))
    if RETX_PRIO:
        prio = 1e9 * has_rx.to(tl.float32)
    else:
        prio = 0.0 * has_rx.to(tl.float32)
    # ---- PF (or max C/I, RR) allocation, RBG by RBG ----
    cnt = tl.zeros(sent.shape, tl.int64)
    left = need.to(tl.float32)
    won = tl.zeros(ref.shape, tl.int32)
    for s in tl.static_range(S_):
        want = (cnt < want_cnt) & (has_rx | (left > 0)) & rm
        ms = tl.where(want, _col(metric, sidx, s) + prio, -1.0)
        best = tl.max(ms, axis=0)
        wi = tl.argmax(ms, axis=0)
        sel = (ridx == wi) & (best >= 0)
        won = tl.where((sidx[None, :] == s) & sel[:, None], 1, won)
        cnt += sel.to(tl.int64)
        left = left - _col(rate, sidx, s) * sel.to(tl.float32) * (~has_rx).to(tl.float32)
    n_sb = tl.sum(won, axis=1).to(tl.int64)
    n_prb = tl.sum(won.to(tl.float32) * w[None, :], axis=1)
    tx_rx = has_rx & (n_sb == rx_nsb) & (n_sb > 0)
    tx_new = new_el & (n_sb > 0)
    tx = tx_rx | tx_new
    lw = tl.where(won > 0, lw_all[None, :], float("-inf"))
    lnw = _lse_rows(lw)
    # ---- link adaptation for new TBs ----
    if DIR == 0:
        if WHOLE_BAND:
            split = tl.full(n_prb.shape, WB_DB, tl.float32)
        else:
            split = 10.0 * libdevice.log10(tl.maximum(n_prb / REF_PRBS, 1e-3))
        if PC:
            split = tl.maximum(split, pc)
        est_tx = ref - split[:, None] + csi
    else:
        est_tx = est
    # highest MCS whose TB error probability at the (OLLA-offset) effective SINR meets the target, one MCS at a time
    lin_tx = libdevice.exp10(tl.minimum(tl.maximum(est_tx, -30.0), 60.0) / 10.0)
    toff0 = (nsi * (NPRB + 1) + n_prb.to(tl.int32)) * MB
    mcs_new = tl.zeros(sent.shape, tl.int32)
    tbs_new = tl.load(tbs_ptr + toff0, mask=tx_new, other=0)
    for m in range(MCS_MAX + 1):           # MCS_MAX <= M - 1 (phy.mcs_max)
        effm = _eff_m(lin_tx, est_tx, lw, lnw, tl.load(beta_ptr + m), MODE)
        if OLLA:
            effm = effm + olla
        o = toff0 + m
        tb = tl.load(tbs_ptr + o, mask=tx_new, other=0)
        bm = _bler(tab_ptr, m, effm, tl.load(ci0_ptr + o, mask=tx_new, other=0),
                   tl.load(cwi_ptr + o, mask=tx_new, other=0.0), tl.load(bg_ptr + o, mask=tx_new, other=0),
                   M, C, G, S0, DS, STEP, tx_new)
        tbler = 1.0 - libdevice.pow(1.0 - bm, tl.load(ncb_ptr + o, mask=tx_new, other=1.0))
        okm = (tbler <= TARGET) & (tb > 0)
        mcs_new = tl.where(okm, m, mcs_new)
        tbs_new = tl.where(okm, tb, tbs_new)
    tbs_new = tbs_new.to(tl.int64)
    cap_b = tl.maximum(tbs_new // 8 - TB_OH, 1)
    byt_new = tl.minimum(cap_b, unsent)
    # ---- bind TBs to processes ----
    p_tx = tl.where(tx_rx, rx_p, p_new)
    ohp = (pidx[None, :] == p_tx[:, None]) & tx[:, None]
    ohn = ohp & tx_new[:, None]
    h_lo = tl.where(ohn, sent[:, None], h_lo)
    h_hi = tl.where(ohn, (sent + byt_new)[:, None], h_hi)
    h_mcs = tl.where(ohn, mcs_new[:, None], h_mcs)
    h_tbs = tl.where(ohn, tbs_new[:, None], h_tbs)
    h_nsb = tl.where(ohn, n_sb[:, None], h_nsb)
    h_ntx = tl.where(ohn, 0, h_ntx)
    h_comb = tl.where(ohn, 0.0, h_comb)
    h_lexp = tl.where(ohn, float("-inf"), h_lexp)
    h_nrb = tl.where(ohn, 0.0, h_nrb)
    h_st = tl.where(ohn, 1, h_st)
    sent = sent + byt_new * tx_new.to(tl.int64)
    mcs = _gather_p(h_mcs, pidx, p_tx)
    tbs = _gather_p(h_tbs, pidx, p_tx).to(tl.int64)
    ntx = _gather_p(h_ntx, pidx, p_tx) + 1
    # ---- decoding on the actual channel ----
    if DIR == 0:
        act = ref - split[:, None] + gain
    else:
        act = ref + gain
    beta = tl.load(beta_ptr + mcs, mask=rm, other=1.0)
    eff1, lse1 = _eff_one(act, lw, lnw, beta, MODE)
    m_look = mcs
    if COMB == 2:          # 5G-LENA IR: EESM over every transmission, equivalent MCS at R / ntx
        lexp_p = _gather_p(h_lexp, pidx, p_tx)
        mx = tl.maximum(lexp_p, lse1)
        mn = tl.minimum(lexp_p, lse1)
        lexp = tl.where(mx == float("-inf"), float("-inf"), mx + libdevice.log1p(libdevice.exp(mn - mx)))
        nrb = _gather_p(h_nrb, pidx, p_tx) + n_prb
        v = -beta * (lexp - libdevice.log(tl.maximum(nrb, 1.0)))
        eff_used = 10.0 * libdevice.log10(tl.maximum(v, 1e-6))
        eff_used = tl.where(v != v, -30.0, eff_used)
        eff_used = tl.where(eff_used == float("inf"), 60.0, eff_used)
        m_look = tl.load(eq_ptr + mcs * EQW + tl.minimum(ntx, EQW - 1), mask=rm, other=0)
        h_lexp = tl.where(ohp, lexp[:, None], h_lexp)
        h_nrb = tl.where(ohp, nrb[:, None], h_nrb)
        comb = _gather_p(h_comb, pidx, p_tx)
    elif COMB == 1:
        comb = _gather_p(h_comb, pidx, p_tx) + libdevice.exp10(eff1 / 10.0)
        eff_used = 10.0 * libdevice.log10(tl.maximum(comb, 1e-9))
    else:
        comb = libdevice.exp10(eff1 / 10.0)
        eff_used = 10.0 * libdevice.log10(tl.maximum(comb, 1e-9))
    r_m = tl.load(rate_ptr + mcs, mask=rm, other=1.0)
    cbs1, ncb1, bg1 = _segment(tbs, r_m, lift_ptr, NL, STEP)
    i01, wi1 = _cidx(cbs1, cax_ptr, C0, DC, C, STEP)
    b1 = _bler(tab_ptr, m_look, eff_used, i01, wi1, bg1, M, C, G, S0, DS, STEP, rm)
    p_err = 1.0 - libdevice.pow(1.0 - b1, ncb1)
    ok = tx & (u >= p_err)
    fail = tx & (~ok)
    exh = fail & (ntx >= MAX_TX)
    # ---- HARQ state update ----
    ohp_ok = ohp & ok[:, None]
    ohp_fail = ohp & fail[:, None]
    ohp_exh = ohp & exh[:, None]
    if DIR == 0:
        h_st = tl.where(ohp_ok, 0, h_st)
        h_rdy = tl.where(ohp_ok, g, h_rdy)
    else:
        h_st = tl.where(ohp_ok, 2, h_st)
        h_rdy = tl.where(ohp_ok, ack_slot + GNB_PROC, h_rdy)
    h_ntx = tl.where(ohp, ntx[:, None], h_ntx)
    h_comb = tl.where(ohp, comb[:, None], h_comb)
    if DIR == 0:
        h_rdy = tl.where(ohp_fail, g + UL_RTT, h_rdy)
    else:
        h_rdy = tl.where(ohp_fail, ack_slot + GNB_PROC, h_rdy)
    lo_tx = _gather_p(h_lo, pidx, p_tx)
    hi_tx = _gather_p(h_hi, pidx, p_tx)
    if HARQ_DROP:
        h_st = tl.where(ohp_exh, 0, h_st)
        ov = exh[:, None] & (qstart < hi_tx[:, None]) & (qend > lo_tx[:, None]) & (cap >= 0)
        cnt_lost = tl.sum((ov & (lost == 0)).to(tl.float32))
        lost = lost | ov
    else:
        h_ntx = tl.where(ohp_exh, 0, h_ntx)
        h_comb = tl.where(ohp_exh, 0.0, h_comb)
        h_lexp = tl.where(ohp_exh, float("-inf"), h_lexp)
        h_nrb = tl.where(ohp_exh, 0.0, h_nrb)
        h_rdy = tl.where(ohp_exh, g + RLC_RETX, h_rdy)
        cnt_lost = 0.0
    # ---- OLLA, direction-specific updates, PF ----
    okf = ok.to(tl.float32)
    failf = fail.to(tl.float32)
    if OLLA:
        olla = tl.minimum(tl.maximum(olla + OLLA_UP * okf - OLLA_DN * failf, -10.0), 10.0)
    served = (hi_tx - lo_tx) * ok.to(tl.int64)
    if DIR == 0:
        bsr = tl.where(tx, enq - sent, bsr)
        csi = gain
    avg = PF_A * avg + PF_B * served.to(tl.float32)
    if SCHED == 2:
        last_tx = tl.where(tx, g, last_tx)
    # ---- counters ----
    txf = tx.to(tl.float32)
    cnt_acc += tl.where(hidx == 0, tl.sum(tx_new.to(tl.float32)), 0.0)
    cnt_acc += tl.where(hidx == 1, tl.sum(tx_rx.to(tl.float32)), 0.0)
    cnt_acc += tl.where(hidx == 2, tl.sum(okf), 0.0)
    cnt_acc += tl.where(hidx == 3, tl.sum(failf), 0.0)
    cnt_acc += tl.where(hidx == 4, tl.sum(exh.to(tl.float32)), 0.0)
    cnt_acc += tl.where(hidx == 5, tl.sum(served.to(tl.float32)), 0.0)
    cnt_acc += tl.where(hidx == 6, tl.sum((byt_new * tx_new.to(tl.int64)).to(tl.float32)), 0.0)
    cnt_acc += tl.where(hidx == 7, cnt_lost, 0.0)
    cnt_acc += tl.where(hidx == 8, tl.sum((tx_new & pending).to(tl.float32)), 0.0)
    cnt_acc += tl.where(hidx == 9, tl.sum(n_prb * txf), 0.0)
    k = tl.minimum(ntx, MAX_TX)
    ohk = hidx[None, :] == k[:, None]
    hist_ok += tl.sum((ohk & ok[:, None]).to(tl.float32), axis=0)
    hist_tx += tl.sum((ohk & tx[:, None]).to(tl.float32), axis=0)
    hist_fail += tl.sum((ohk & fail[:, None]).to(tl.float32), axis=0)
    # ---- RLC in-order delivery ----
    own = (h_st == 1) & (h_hi > floor[:, None])
    lo_own = tl.where(own, tl.maximum(h_lo, floor[:, None]), BIG)
    ack = tl.minimum(sent, tl.min(lo_own, axis=1))
    done = (cap >= 0) & (qend <= ack[:, None]) & (lost == 0) & (fin == float("inf"))
    fin = tl.where(done, fin_val, fin)
    return (sent, olla, avg, bsr, sr_t, last_tx, csi,
            h_st.to(tl.int32), h_lo, h_hi, h_rdy, h_ntx.to(tl.int32), h_mcs.to(tl.int32), h_tbs.to(tl.int32),
            h_nsb.to(tl.int32), h_comb, h_lexp, h_nrb,
            lost, fin, cnt_acc, hist_ok, hist_tx, hist_fail)


@triton.jit
def _ld_link(e, R, sent_p, floor_p, olla_p, avg_p, bsr_p, srt_p, ltx_p, enq_p, csi_p, hst_p, hlo_p, hhi_p, hrdy_p,
             hntx_p, hmcs_p, htbs_p, hnsb_p, hcomb_p, hlexp_p, hnrb_p, cap_p, qs_p, qe_p, lost_p, fin_p,
             ridx, rm, sidx, sm, pidx, pm, fidx, fm, S_: tl.constexpr, P: tl.constexpr, F: tl.constexpr):
    er = e * R + ridx
    o_rs = er[:, None] * S_ + sidx[None, :]
    m_rs = rm[:, None] & sm[None, :]
    o_rp = er[:, None] * P + pidx[None, :]
    m_rp = rm[:, None] & pm[None, :]
    o_rf = er[:, None] * F + fidx[None, :]
    m_rf = rm[:, None] & fm[None, :]
    return (tl.load(sent_p + er, mask=rm, other=0), tl.load(floor_p + er, mask=rm, other=0),
            tl.load(olla_p + er, mask=rm, other=0.0), tl.load(avg_p + er, mask=rm, other=100.0),
            tl.load(bsr_p + er, mask=rm, other=0), tl.load(srt_p + er, mask=rm, other=-1),
            tl.load(ltx_p + er, mask=rm, other=-1), tl.load(enq_p + er, mask=rm, other=0),
            tl.load(csi_p + o_rs, mask=m_rs, other=0.0),
            tl.load(hst_p + o_rp, mask=m_rp, other=3).to(tl.int32), tl.load(hlo_p + o_rp, mask=m_rp, other=0),
            tl.load(hhi_p + o_rp, mask=m_rp, other=0), tl.load(hrdy_p + o_rp, mask=m_rp, other=0),
            tl.load(hntx_p + o_rp, mask=m_rp, other=0).to(tl.int32),
            tl.load(hmcs_p + o_rp, mask=m_rp, other=0).to(tl.int32),
            tl.load(htbs_p + o_rp, mask=m_rp, other=0).to(tl.int32),
            tl.load(hnsb_p + o_rp, mask=m_rp, other=0).to(tl.int32),
            tl.load(hcomb_p + o_rp, mask=m_rp, other=0.0), tl.load(hlexp_p + o_rp, mask=m_rp, other=float("-inf")),
            tl.load(hnrb_p + o_rp, mask=m_rp, other=0.0),
            tl.load(cap_p + o_rf, mask=m_rf, other=-1).to(tl.int32), tl.load(qs_p + o_rf, mask=m_rf, other=0),
            tl.load(qe_p + o_rf, mask=m_rf, other=0), tl.load(lost_p + o_rf, mask=m_rf, other=0).to(tl.int1),
            tl.load(fin_p + o_rf, mask=m_rf, other=float("inf")))


@triton.jit
def _st_link(e, R, sent_p, olla_p, avg_p, bsr_p, srt_p, ltx_p, csi_p, hst_p, hlo_p, hhi_p, hrdy_p,
             hntx_p, hmcs_p, htbs_p, hnsb_p, hcomb_p, hlexp_p, hnrb_p, lost_p, fin_p,
             sent, olla, avg, bsr, sr_t, last_tx, csi, h_st, h_lo, h_hi, h_rdy, h_ntx, h_mcs, h_tbs, h_nsb,
             h_comb, h_lexp, h_nrb, lost, fin,
             ridx, rm, sidx, sm, pidx, pm, fidx, fm, S_: tl.constexpr, P: tl.constexpr, F: tl.constexpr):
    er = e * R + ridx
    o_rs = er[:, None] * S_ + sidx[None, :]
    m_rs = rm[:, None] & sm[None, :]
    o_rp = er[:, None] * P + pidx[None, :]
    m_rp = rm[:, None] & pm[None, :]
    o_rf = er[:, None] * F + fidx[None, :]
    m_rf = rm[:, None] & fm[None, :]
    tl.store(sent_p + er, sent, mask=rm)
    tl.store(olla_p + er, olla, mask=rm)
    tl.store(avg_p + er, avg, mask=rm)
    tl.store(bsr_p + er, bsr, mask=rm)
    tl.store(srt_p + er, sr_t, mask=rm)
    tl.store(ltx_p + er, last_tx, mask=rm)
    tl.store(csi_p + o_rs, csi, mask=m_rs)
    tl.store(hst_p + o_rp, h_st, mask=m_rp)
    tl.store(hlo_p + o_rp, h_lo, mask=m_rp)
    tl.store(hhi_p + o_rp, h_hi, mask=m_rp)
    tl.store(hrdy_p + o_rp, h_rdy, mask=m_rp)
    tl.store(hntx_p + o_rp, h_ntx, mask=m_rp)
    tl.store(hmcs_p + o_rp, h_mcs, mask=m_rp)
    tl.store(htbs_p + o_rp, h_tbs, mask=m_rp)
    tl.store(hnsb_p + o_rp, h_nsb, mask=m_rp)
    tl.store(hcomb_p + o_rp, h_comb, mask=m_rp)
    tl.store(hlexp_p + o_rp, h_lexp, mask=m_rp)
    tl.store(hnrb_p + o_rp, h_nrb, mask=m_rp)
    tl.store(lost_p + o_rf, lost, mask=m_rf)
    tl.store(fin_p + o_rf, fin, mask=m_rf)


@triton.jit(do_not_specialize=["s0", "chs", "K"])
def nr_step_kernel(
        # UL link state
        u_sent, u_floor, u_olla, u_avg, u_bsr, u_srt, u_ltx, u_enq, u_csi, u_hst, u_hlo, u_hhi, u_hrdy, u_hntx,
        u_hmcs, u_htbs, u_hnsb, u_hcomb, u_hlexp, u_hnrb, u_cap, u_qs, u_qe, u_lost, u_fin,
        # DL link state
        d_sent, d_floor, d_olla, d_avg, d_bsr, d_srt, d_ltx, d_enq, d_csi, d_hst, d_hlo, d_hhi, d_hrdy, d_hntx,
        d_hmcs, d_htbs, d_hnsb, d_hcomb, d_hlexp, d_hnrb, d_cap, d_qs, d_qe, d_lost, d_fin,
        # inputs and fading
        h_ptr, uref_ptr, dref_ptr, pc_ptr, t_ptr, ep_ptr, ctr_ptr, s0, chs,
        # traffic arrival gate (per-message stream ends [E,R,NMSG], arrival slots, base [E,R]); see launch_step
        gate_e_ptr, gate_s_ptr, gate_b_ptr, NMSG,
        # per-env accumulators [E, 8 HB]: UL counters, hist ok / tx / fail, then the same for the DL
        acc_ptr,
        # schedule of this step: itab [K, 10] int64, ftab [K, 7] float64; per-robot fading rho per ms [E,R]
        itab_ptr, ftab_ptr, K, rho_ptr,
        # tables per direction (UL, DL)
        u_tab, u_thr, u_se, u_beta, u_rate, u_eq, u_tbs, u_cbs, u_ncb, u_bg, u_ci0, u_cwi,
        d_tab, d_thr, d_se, d_beta, d_rate, d_eq, d_tbs, d_cbs, d_ncb, d_bg, d_ci0, d_cwi,
        lift_ptr, cax_ptr, w_ptr, S0, DS, C0, DC, R, N,
        RB: tl.constexpr, S_: tl.constexpr, SB: tl.constexpr, P: tl.constexpr, PB: tl.constexpr,
        F: tl.constexpr, FB: tl.constexpr, M: tl.constexpr, MB: tl.constexpr, HB: tl.constexpr,
        C: tl.constexpr, G: tl.constexpr, NL: tl.constexpr, EQW: tl.constexpr, NPRB: tl.constexpr,
        UL: tl.constexpr, DL: tl.constexpr, FADING: tl.constexpr, GATE: tl.constexpr, MMB: tl.constexpr,
        RHO_R: tl.constexpr,
        MODE: tl.constexpr, COMB: tl.constexpr, SCHED: tl.constexpr, WIDEBAND: tl.constexpr,
        HARQ_DROP: tl.constexpr, OLLA: tl.constexpr, PHR_CAP: tl.constexpr, WHOLE_BAND: tl.constexpr,
        PC: tl.constexpr, STEP: tl.constexpr, RETX_PRIO: tl.constexpr,
        MCS_MAX_UL: tl.constexpr, MCS_MAX_DL: tl.constexpr, MAX_TX: tl.constexpr, TARGET: tl.constexpr,
        TB_OH: tl.constexpr, SR_DELAY: tl.constexpr, UL_RTT: tl.constexpr, RLC_RETX: tl.constexpr,
        GNB_PROC: tl.constexpr, REF_PRBS: tl.constexpr, PHR_MIN: tl.constexpr, WB_DB: tl.constexpr,
        W0: tl.constexpr, OLLA_UP: tl.constexpr, OLLA_DN: tl.constexpr, PF_A: tl.constexpr, PF_B: tl.constexpr):
    e = tl.program_id(0).to(tl.int64)
    ridx = tl.arange(0, RB)
    rm = ridx < R
    sidx = tl.arange(0, SB)
    sm = sidx < S_
    pidx = tl.arange(0, PB)
    pm = pidx < P
    fidx = tl.arange(0, FB)
    fm = fidx < F
    midx = tl.arange(0, MB)
    hidx = tl.arange(0, HB)
    w = tl.load(w_ptr + sidx, mask=sm, other=0.0)
    lw_all = tl.where(sm, libdevice.log(tl.maximum(w, 1e-30)), float("-inf"))
    er = e * R + ridx
    o_rs = er[:, None] * S_ + sidx[None, :]
    m_rs = rm[:, None] & sm[None, :]
    t = tl.load(t_ptr)
    ep = u32(tl.load(ep_ptr + e))
    ctr = u32(tl.load(ctr_ptr + e))
    base0 = mix32(u32(s0) ^ salt(u32(e)))
    base0 = mix32(base0 ^ salt(ep))
    base0 = mix32(base0 ^ u32(chs))
    base0 = mix32(base0 ^ salt(ctr))
    # fading state h [R, S, 2]
    o_h = er[:, None] * (S_ * 2) + sidx[None, :] * 2
    hr = tl.load(h_ptr + o_h, mask=m_rs, other=1.0)
    hi = tl.load(h_ptr + o_h + 1, mask=m_rs, other=0.0)
    jn = (ridx[:, None] * S_ + sidx[None, :]) * 2
    if RHO_R:
        rho_ms = tl.load(rho_ptr + er, mask=rm, other=0.5)
    zero_rs = tl.zeros([RB, SB], tl.float32)
    if UL:
        (u_s_sent, u_s_floor, u_s_olla, u_s_avg, u_s_bsr, u_s_srt, u_s_ltx, u_s_enq, u_s_csi, u_s_hst, u_s_hlo,
         u_s_hhi, u_s_hrdy, u_s_hntx, u_s_hmcs, u_s_htbs, u_s_hnsb, u_s_hcomb, u_s_hlexp, u_s_hnrb, u_s_cap, u_s_qs,
         u_s_qe, u_s_lost, u_s_fin) = _ld_link(
            e, R, u_sent, u_floor, u_olla, u_avg, u_bsr, u_srt, u_ltx, u_enq, u_csi, u_hst, u_hlo, u_hhi, u_hrdy,
            u_hntx, u_hmcs, u_htbs, u_hnsb, u_hcomb, u_hlexp, u_hnrb, u_cap, u_qs, u_qe, u_lost, u_fin,
            ridx, rm, sidx, sm, pidx, pm, fidx, fm, S_, P, F)
        uref = tl.load(uref_ptr + o_rs, mask=m_rs, other=0.0)
        if PC:
            pc = tl.load(pc_ptr + er, mask=rm, other=0.0)
        else:
            pc = tl.zeros([RB], tl.float32)
        u_cnt = tl.zeros([HB], tl.float32)
        u_hok = tl.zeros([HB], tl.float32)
        u_htx = tl.zeros([HB], tl.float32)
        u_hfl = tl.zeros([HB], tl.float32)
        if GATE:           # the UL stream opens message by message at the arrival slots (engine traffic models)
            mmi = tl.arange(0, MMB)
            o_rm = er[:, None] * NMSG + mmi[None, :]
            m_rm = rm[:, None] & (mmi[None, :] < NMSG)
            g_base = tl.load(gate_b_ptr + er, mask=rm, other=0)
            g_end = tl.load(gate_e_ptr + o_rm, mask=m_rm, other=0)
            g_slot = tl.load(gate_s_ptr + o_rm, mask=m_rm, other=2 ** 30)
    if DL:
        (d_s_sent, d_s_floor, d_s_olla, d_s_avg, d_s_bsr, d_s_srt, d_s_ltx, d_s_enq, d_s_csi, d_s_hst, d_s_hlo,
         d_s_hhi, d_s_hrdy, d_s_hntx, d_s_hmcs, d_s_htbs, d_s_hnsb, d_s_hcomb, d_s_hlexp, d_s_hnrb, d_s_cap, d_s_qs,
         d_s_qe, d_s_lost, d_s_fin) = _ld_link(
            e, R, d_sent, d_floor, d_olla, d_avg, d_bsr, d_srt, d_ltx, d_enq, d_csi, d_hst, d_hlo, d_hhi, d_hrdy,
            d_hntx, d_hmcs, d_htbs, d_hnsb, d_hcomb, d_hlexp, d_hnrb, d_cap, d_qs, d_qe, d_lost, d_fin,
            ridx, rm, sidx, sm, pidx, pm, fidx, fm, S_, P, F)
        dref = tl.load(dref_ptr + o_rs, mask=m_rs, other=0.0)
        d_cnt = tl.zeros([HB], tl.float32)
        d_hok = tl.zeros([HB], tl.float32)
        d_htx = tl.zeros([HB], tl.float32)
        d_hfl = tl.zeros([HB], tl.float32)
    for k in range(K):
        rel = tl.load(itab_ptr + k * 10 + 0)
        dls = tl.load(itab_ptr + k * 10 + 1)
        uls = tl.load(itab_ptr + k * 10 + 2)
        srf = tl.load(itab_ptr + k * 10 + 3)
        cqi = tl.load(itab_ptr + k * 10 + 4)
        ackr = tl.load(itab_ptr + k * 10 + 5)
        pgn = tl.load(itab_ptr + k * 10 + 6)
        nsi_u = tl.load(itab_ptr + k * 10 + 7)
        nsi_d = tl.load(itab_ptr + k * 10 + 8)
        rho = tl.load(ftab_ptr + k * 7 + 0).to(tl.float32)
        c1 = tl.load(ftab_ptr + k * 7 + 1).to(tl.float32)
        fr = tl.load(ftab_ptr + k * 7 + 2)
        re_u = tl.load(ftab_ptr + k * 7 + 3).to(tl.float32)
        re_d = tl.load(ftab_ptr + k * 7 + 4).to(tl.float32)
        fin_off = tl.load(ftab_ptr + k * 7 + 5)
        g = t * N + rel
        fin_val = (t.to(tl.float64) + fr) + fin_off
        if FADING:
            base = mix32(base0 ^ salt(u32((1 << 16) | rel)))
            zr = rng_normal(base, jn)
            zi = rng_normal(base, jn + 1)
            if RHO_R:      # per-robot Doppler: rho_r ** (dt * slot_ms), as NRNet._evolve with fading_rho_ms
                rr = libdevice.pow(rho_ms, tl.load(ftab_ptr + k * 7 + 6).to(tl.float32))[:, None]
                cr = libdevice.sqrt(1.0 - rr * rr)
                hr = rr * hr + cr * zr / 1.4142135623730951
                hi = rr * hi + cr * zi / 1.4142135623730951
            else:
                hr = rho * hr + c1 * zr / 1.4142135623730951
                hi = rho * hi + c1 * zi / 1.4142135623730951
            gain = 10.0 * libdevice.log10(tl.maximum(hr * hr + hi * hi, 1e-6))
        else:
            gain = zero_rs
        if DL:
            if cqi != 0:
                mi = tl.zeros([RB, SB], tl.int32)
                xs = dref + gain
                for m in tl.static_range(M):
                    if m <= MCS_MAX_DL:
                        mi = tl.where(xs >= tl.load(d_thr + m), m, mi)
                d_s_csi = tl.load(d_thr + mi) - dref
            if dls != 0:
                ub = mix32(base0 ^ salt(u32((3 << 16) | rel)))
                ud = rng_uniform(ub, ridx)
                (d_s_sent, d_s_olla, d_s_avg, d_s_bsr, d_s_srt, d_s_ltx, d_s_csi, d_s_hst, d_s_hlo, d_s_hhi,
                 d_s_hrdy, d_s_hntx, d_s_hmcs, d_s_htbs, d_s_hnsb, d_s_hcomb, d_s_hlexp, d_s_hnrb, d_s_lost,
                 d_s_fin, d_cnt, d_hok, d_htx, d_hfl) = _mac_slot(
                    d_s_sent, d_s_floor, d_s_olla, d_s_avg, d_s_bsr, d_s_srt, d_s_ltx, d_s_enq, d_s_csi, dref,
                    gain, tl.zeros([RB], tl.float32), d_s_hst, d_s_hlo, d_s_hhi, d_s_hrdy, d_s_hntx, d_s_hmcs,
                    d_s_htbs, d_s_hnsb, d_s_hcomb, d_s_hlexp, d_s_hnrb, d_s_cap, d_s_qs, d_s_qe, d_s_lost, d_s_fin,
                    d_cnt, d_hok, d_htx, d_hfl,
                    g, fin_val, re_d, nsi_d, 0, t * N + ackr, ud,
                    ridx, rm, sidx, sm, pidx, midx, fidx, hidx, w, lw_all,
                    d_tab, d_thr, d_se, d_beta, d_rate, d_eq, lift_ptr, cax_ptr,
                    d_tbs, d_cbs, d_ncb, d_bg, d_ci0, d_cwi, S0, DS, C0, DC,
                    1, S_, SB, M, MB, C, G, NL, EQW, NPRB, MODE, COMB, SCHED, WIDEBAND, HARQ_DROP, OLLA,
                    PHR_CAP, WHOLE_BAND, False, STEP, RETX_PRIO, MCS_MAX_DL, MAX_TX, TARGET, TB_OH, SR_DELAY,
                    UL_RTT, RLC_RETX, GNB_PROC, REF_PRBS, PHR_MIN, WB_DB, W0, OLLA_UP, OLLA_DN, PF_A, PF_B)
        if UL:
            if GATE:
                if (srf != 0) | (uls != 0):
                    vis = tl.max(tl.where(g_slot <= rel, g_end, g_base[:, None]), axis=1)
                    u_s_enq = tl.maximum(g_base, vis)
            if srf != 0:
                need_sr = (u_s_enq - u_s_sent > 0) & (u_s_bsr <= 0) & (u_s_srt < 0)
                u_s_srt = tl.where(need_sr, g, u_s_srt)
            if uls != 0:
                ub = mix32(base0 ^ salt(u32((2 << 16) | rel)))
                uu = rng_uniform(ub, ridx)
                (u_s_sent, u_s_olla, u_s_avg, u_s_bsr, u_s_srt, u_s_ltx, u_s_csi, u_s_hst, u_s_hlo, u_s_hhi,
                 u_s_hrdy, u_s_hntx, u_s_hmcs, u_s_htbs, u_s_hnsb, u_s_hcomb, u_s_hlexp, u_s_hnrb, u_s_lost,
                 u_s_fin, u_cnt, u_hok, u_htx, u_hfl) = _mac_slot(
                    u_s_sent, u_s_floor, u_s_olla, u_s_avg, u_s_bsr, u_s_srt, u_s_ltx, u_s_enq, u_s_csi, uref,
                    gain, pc, u_s_hst, u_s_hlo, u_s_hhi, u_s_hrdy, u_s_hntx, u_s_hmcs,
                    u_s_htbs, u_s_hnsb, u_s_hcomb, u_s_hlexp, u_s_hnrb, u_s_cap, u_s_qs, u_s_qe, u_s_lost, u_s_fin,
                    u_cnt, u_hok, u_htx, u_hfl,
                    g, fin_val, re_u, nsi_u, pgn != 0, 0, uu,
                    ridx, rm, sidx, sm, pidx, midx, fidx, hidx, w, lw_all,
                    u_tab, u_thr, u_se, u_beta, u_rate, u_eq, lift_ptr, cax_ptr,
                    u_tbs, u_cbs, u_ncb, u_bg, u_ci0, u_cwi, S0, DS, C0, DC,
                    0, S_, SB, M, MB, C, G, NL, EQW, NPRB, MODE, COMB, SCHED, WIDEBAND, HARQ_DROP, OLLA,
                    PHR_CAP, WHOLE_BAND, PC, STEP, RETX_PRIO, MCS_MAX_UL, MAX_TX, TARGET, TB_OH, SR_DELAY,
                    UL_RTT, RLC_RETX, GNB_PROC, REF_PRBS, PHR_MIN, WB_DB, W0, OLLA_UP, OLLA_DN, PF_A, PF_B)
    if FADING:
        tl.store(h_ptr + o_h, hr, mask=m_rs)
        tl.store(h_ptr + o_h + 1, hi, mask=m_rs)
    ao = e * (8 * HB)
    if UL:
        _st_link(e, R, u_sent, u_olla, u_avg, u_bsr, u_srt, u_ltx, u_csi, u_hst, u_hlo, u_hhi, u_hrdy,
                 u_hntx, u_hmcs, u_htbs, u_hnsb, u_hcomb, u_hlexp, u_hnrb, u_lost, u_fin,
                 u_s_sent, u_s_olla, u_s_avg, u_s_bsr, u_s_srt, u_s_ltx, u_s_csi, u_s_hst, u_s_hlo, u_s_hhi,
                 u_s_hrdy, u_s_hntx, u_s_hmcs, u_s_htbs, u_s_hnsb, u_s_hcomb, u_s_hlexp, u_s_hnrb, u_s_lost,
                 u_s_fin, ridx, rm, sidx, sm, pidx, pm, fidx, fm, S_, P, F)
        tl.store(acc_ptr + ao + hidx, u_cnt)
        tl.store(acc_ptr + ao + HB + hidx, u_hok)
        tl.store(acc_ptr + ao + 2 * HB + hidx, u_htx)
        tl.store(acc_ptr + ao + 3 * HB + hidx, u_hfl)
    if DL:
        _st_link(e, R, d_sent, d_olla, d_avg, d_bsr, d_srt, d_ltx, d_csi, d_hst, d_hlo, d_hhi, d_hrdy,
                 d_hntx, d_hmcs, d_htbs, d_hnsb, d_hcomb, d_hlexp, d_hnrb, d_lost, d_fin,
                 d_s_sent, d_s_olla, d_s_avg, d_s_bsr, d_s_srt, d_s_ltx, d_s_csi, d_s_hst, d_s_hlo, d_s_hhi,
                 d_s_hrdy, d_s_hntx, d_s_hmcs, d_s_htbs, d_s_hnsb, d_s_hcomb, d_s_hlexp, d_s_hnrb, d_s_lost,
                 d_s_fin, ridx, rm, sidx, sm, pidx, pm, fidx, fm, S_, P, F)
        tl.store(acc_ptr + ao + 4 * HB + hidx, d_cnt)
        tl.store(acc_ptr + ao + 5 * HB + hidx, d_hok)
        tl.store(acc_ptr + ao + 6 * HB + hidx, d_htx)
        tl.store(acc_ptr + ao + 7 * HB + hidx, d_hfl)


CTR_NAMES = ("tb_new", "tb_retx", "tb_ok", "tb_fail", "exhaust", "bytes_ok", "bytes_new", "lost_frames",
             "new_while_pending", "prb_used")
LINK_PTRS = ("sent", "floor", "olla", "avg", "bsr", "sr_t", "last_tx", "q.enq", "csi", "h_state", "h_lo", "h_hi",
             "h_ready", "h_ntx", "h_mcs", "h_tbs", "h_nsb", "h_comb", "h_lexp", "h_nrb", "q.cap", "q.start",
             "q.end", "q.lost", "q.fin")


def _link_args(link):
    out = []
    for n in LINK_PTRS:
        x = getattr(link.q, n[2:]) if n.startswith("q.") else getattr(link, n)
        assert x.is_contiguous(), n
        out.append(x)
    return out


def launch_step(eng, uref, dref, pc, itab, ftab, K, gate=None):
    """Run the fused kernel for one control step of NRTritonEngine eng, in place on the engine's buffers."""
    net, cfg, tb = eng.net, eng.config, eng._tables
    ul, dl = net.ul, net.dl
    E, R = eng.E, eng.R
    U = ul if cfg.ul else (dl if dl is not None else ul)
    D = dl if dl is not None else U
    tu, td = tb["ul"], tb["dl"] if dl is not None else tb["ul"]
    dummy = uref
    eng._kernel = nr_step_kernel[(E,)](
        *_link_args(U), *_link_args(D),
        net.h, uref if uref is not None else dref, dref if dref is not None else dummy,
        pc if pc is not None else dummy, eng._tdev, net.rng.episode, net.rng.ctr, net.rng.s0, eng._chs,
        *(gate if gate is not None else (dummy, dummy, dummy)), gate[0].shape[-1] if gate is not None else 1,
        eng._acc, itab, ftab, K, net.fading_rho_ms if net.fading_rho_ms is not None else net.h,
        tu["tab"], tu["thr"], tu["se"], tu["beta"], tu["rate"], tu["eq"], tu["tbs"], tu["cbs"], tu["ncb"], tu["bg"],
        tu["ci0"], tu["cwi"],
        td["tab"], td["thr"], td["se"], td["beta"], td["rate"], td["eq"], td["tbs"], td["cbs"], td["ncb"], td["bg"],
        td["ci0"], td["cwi"],
        tb["lift"], tb["cax"], tb["w"], tb["S0"], tb["DS"], tb["C0"], tb["DC"], R, cfg.slots_per_step,
        GATE=gate is not None, RHO_R=net.fading_rho_ms is not None, MMB=triton.next_power_of_2(gate[0].shape[-1]) if gate is not None else 1,
        **eng._const, num_warps=eng._num_warps)
