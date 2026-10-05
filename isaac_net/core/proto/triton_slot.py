"""Fused Triton kernels: the whole 40-UL-slot loop of NetSlot._transmit (L2) or NetFluid._transmit (L1)
in one launch.

One program per env; robots x subbands (and robots x frame slots) live in registers for all 40 slots,
so state is read and written once per control step instead of ~100 times per slot.
Semantics follow netsim.NetSlot._transmit (audited version). Math uses libdevice (same CUDA math library
as ATen), but reduction/scan orders differ, so results agree with NetSlot to float rounding, not bitwise.

The frame buffer F, the UL slots per step K and the random-number mode are compile-time constants; FB is F
rounded up to a power of two (tl.arange needs one) and the extra lanes are masked. RNG: 0 = injected draws,
1 = in-kernel Philox with the engine's _seed (rng="global"), 2 = the engine's counter-based streams of
proto/rng.py hashed in the kernel (rng="engine"; the same uniforms as the reference, normals to rounding).
"""
import triton
import triton.language as tl
from triton.language.extra import libdevice

from . import rng as _rng
from . import rng_triton as _rt


@triton.jit
def _se(x, SE_MIN: tl.constexpr, SE_MAX: tl.constexpr):
    v = 0.75 * libdevice.log2(1.0 + libdevice.exp10(x / 10.0))
    return tl.minimum(tl.maximum(v, SE_MIN), SE_MAX)


@triton.jit
def slot_loop_kernel(rem_ptr, bsr_ptr, srt_ptr, avg_ptr, olla_ptr, wait_ptr, hcnt_ptr, h_ptr,
                     snr_ptr, fin_ptr, nz_ptr, u_ptr, t_ptr, seed_ptr, ep_ptr, ctr_ptr, env_ptr, s0, chs, st_n, st_u,
                     E, R,
                     RB: tl.constexpr, FB: tl.constexpr, F_: tl.constexpr, SB: tl.constexpr, S_: tl.constexpr,
                     K: tl.constexpr, RNG: tl.constexpr,
                     RHO: tl.constexpr, C1: tl.constexpr, SQ2: tl.constexpr, BYTES: tl.constexpr,
                     SE_MIN: tl.constexpr, SE_MAX: tl.constexpr, SR_DELAY: tl.constexpr,
                     HARQ_RTT: tl.constexpr, HARQ_MAX: tl.constexpr, RLC_EXTRA: tl.constexpr,
                     PF_A: tl.constexpr, PF_B: tl.constexpr, PHR: tl.constexpr, PF_MIN: tl.constexpr):
    e = tl.program_id(0).to(tl.int64)
    r = tl.arange(0, RB)
    rm = r < R
    f = tl.arange(0, FB)
    s = tl.arange(0, SB)
    sm = s < S_
    er = e * R + r
    o_rf = er[:, None] * F_ + f[None, :]
    m_rf = rm[:, None] & (f[None, :] < F_)
    o_rs = er[:, None] * (S_ * 2) + s[None, :] * 2
    m_rs = rm[:, None] & sm[None, :]
    if RNG == 2:
        ep = _rt.u32(tl.load(ep_ptr + e))
        ctr = _rt.u32(tl.load(ctr_ptr + e))
        env = _rt.u32(tl.load(env_ptr + e))                 # global env id (rng.env: shard offset + row)
        base_n = _rt.key(_rt.u32(s0), env, ep, _rt.u32(chs), ctr, _rt.u32(st_n))
        base_u = _rt.key(_rt.u32(s0), env, ep, _rt.u32(chs), ctr, _rt.u32(st_u))
        i_rs = (r[:, None] * S_ + s[None, :]) * 2          # within-env index of (r, s, re) in [K,R,S,2]

    rem = tl.load(rem_ptr + o_rf, mask=m_rf, other=0.0)
    bsr = tl.load(bsr_ptr + er, mask=rm, other=0.0)
    sr_t = tl.load(srt_ptr + er, mask=rm, other=-1)
    avg = tl.load(avg_ptr + er, mask=rm, other=1.0)
    olla = tl.load(olla_ptr + er, mask=rm, other=0.0)
    wait = tl.load(wait_ptr + er, mask=rm, other=0)
    hcnt = tl.load(hcnt_ptr + er, mask=rm, other=0.0)
    hr = tl.load(h_ptr + o_rs, mask=m_rs, other=1.0)
    hi = tl.load(h_ptr + o_rs + 1, mask=m_rs, other=0.0)
    snr = tl.load(snr_ptr + er, mask=rm, other=0.0)
    t = tl.load(t_ptr + e)                  # per-env clock
    seed = tl.load(seed_ptr)
    fin = tl.full([RB, FB], float("inf"), tl.float32)
    n_max = tl.minimum(tl.maximum(tl.floor(libdevice.exp10((snr - PHR) / 10.0)), 1.0), S_ * 1.0)
    ER = E * R
    for k in range(K):
        g = t * K + k
        q = tl.sum(rem, axis=1)
        need_sr = (q > 0) & (bsr <= 0) & (sr_t < 0)
        sr_t = tl.where(need_sr, g, sr_t)
        granted = (sr_t >= 0) & (g - sr_t >= SR_DELAY)
        bsr = tl.where(granted, tl.maximum(bsr, 1.0), bsr)
        sr_t = tl.where(granted, -1, sr_t)
        gain_prev = 10.0 * libdevice.log10(tl.maximum(hr * hr + hi * hi, 1e-6))
        if RNG == 0:
            no = k * (ER * S_ * 2) + o_rs
            nr = tl.load(nz_ptr + no, mask=m_rs, other=0.0)
            ni = tl.load(nz_ptr + no + 1, mask=m_rs, other=0.0)
        elif RNG == 2:
            ik = k * (R * S_ * 2) + i_rs
            nr = _rt.normal(base_n, ik)
            ni = _rt.normal(base_n, ik + 1)
        else:
            no = (k * (ER * S_ * 2) + o_rs).to(tl.int32)
            nr = tl.randn(seed, no)
            ni = tl.randn(seed, no + 1)
        hr = RHO * hr + C1 * nr / SQ2
        hi = RHO * hi + C1 * ni / SQ2
        gain_now = 10.0 * libdevice.log10(tl.maximum(hr * hr + hi * hi, 1e-6))
        est = snr[:, None] + gain_prev + olla[:, None]
        rate = _se(est, SE_MIN, SE_MAX) * BYTES
        elig = (bsr > 0) & (g >= wait) & (q > 0)
        need = tl.where(elig, tl.minimum(bsr, q), 0.0)
        cnt = tl.zeros([RB], tl.float32)
        won = tl.zeros([RB, SB], tl.int32)
        for sb in tl.static_range(S_):
            r_s = tl.sum(tl.where(s[None, :] == sb, rate, 0.0), axis=1)
            m = tl.where((need > 0) & (cnt < n_max) & rm, r_s / avg, -1.0)
            best = tl.max(m, axis=0)
            w = tl.argmax(m, axis=0)
            sel = (r == w) & (best > 0)
            won = tl.where((s[None, :] == sb) & sel[:, None], 1, won)
            need = need - tl.where(sel, r_s, 0.0)
            cnt = cnt + sel.to(tl.float32)
        n = tl.sum(won, axis=1)
        tx = n > 0
        nf = tl.maximum(n, 1).to(tl.float32)
        split_db = 10.0 * libdevice.log10(nf)
        wf = won.to(tl.float32)
        mean_est = tl.sum(est * wf, axis=1) / nf - split_db
        se = _se(mean_est, SE_MIN, SE_MAX)
        act = snr[:, None] - split_db[:, None] + gain_now
        act_eff = tl.sum(act * wf, axis=1) / nf + 3.0 * hcnt
        req = 10.0 * libdevice.log10(libdevice.exp2(se / 0.75) - 1.0)
        p_ok = 1.0 / (1.0 + libdevice.exp(-(1.5 * (act_eff - req))))
        if RNG == 0:
            u = tl.load(u_ptr + k * ER + er, mask=rm, other=1.0)
        elif RNG == 2:
            u = _rt.uniform(base_u, k * R + r)
        else:
            u = tl.rand(seed, (K * ER * S_ * 2 + k * ER + er).to(tl.int32))
        ok_tb = tx & (u < p_ok)
        fail = tx & (~ok_tb)
        served = tl.minimum(n.to(tl.float32) * se * BYTES * ok_tb.to(tl.float32), q)
        # FIFO service: new[j] = max(cum[j]-b,0) - max(cum[j-1]-b,0)
        cum = tl.cumsum(rem, axis=1)
        newcum = tl.maximum(cum - served[:, None], 0.0)
        prev = tl.maximum(cum - rem - served[:, None], 0.0)
        new = newcum - prev
        finm = (rem > 0) & (new <= 1e-3)
        rem = tl.where(finm, 0.0, new)
        finv = (t.to(tl.float64) + (k + 1).to(tl.float64) / K).to(tl.float32)
        fin = tl.where(finm, finv, fin)
        okf = ok_tb.to(tl.float32)
        failf = fail.to(tl.float32)
        olla = tl.minimum(tl.maximum(olla + 0.05 * okf - 0.45 * failf, -10.0), 10.0)
        hc = hcnt + 1.0
        exhausted = fail & (hc >= HARQ_MAX)
        hcnt = tl.where(fail, tl.where(exhausted, 0.0, hc), tl.where(tx, 0.0, hcnt))
        wait = tl.where(fail, g + HARQ_RTT + RLC_EXTRA * exhausted.to(tl.int64), wait)
        bsr = tl.where(tx, tl.sum(rem, axis=1), bsr)
        avg = tl.maximum(PF_A * avg + PF_B * served, PF_MIN)

    tl.store(rem_ptr + o_rf, rem, mask=m_rf)
    tl.store(fin_ptr + o_rf, fin, mask=m_rf)
    tl.store(bsr_ptr + er, bsr, mask=rm)
    tl.store(srt_ptr + er, sr_t, mask=rm)
    tl.store(avg_ptr + er, avg, mask=rm)
    tl.store(olla_ptr + er, olla, mask=rm)
    tl.store(wait_ptr + er, wait, mask=rm)
    tl.store(hcnt_ptr + er, hcnt, mask=rm)
    tl.store(h_ptr + o_rs, hr, mask=m_rs)
    tl.store(h_ptr + o_rs + 1, hi, mask=m_rs)


def launch(net, inject):
    """Run the 40-slot loop in place on net's persistent buffers; writes net._fin."""
    import math
    from . import netsim as ns
    E, R = net.E, net.R
    RB = max(16, triton.next_power_of_2(R))
    nw = 8 if RB >= 128 else (4 if RB >= 64 else 2)
    dummy = net.bsr
    mode = 0 if inject else (1 if net.rng is None else 2)
    ep = net.rng.episode if mode == 2 else net.sr_t
    ctr = net.rng.ctr[_rng.STEP] if mode == 2 else net.sr_t
    # env ids by pointer, as nr_step_kernel reads them: set_env_offset updates rng.env in place after a capture
    env = net.rng.env if mode == 2 else net.sr_t
    s0 = net.rng.s0 if mode == 2 else 0
    slot_loop_kernel[(E,)](
        net.rem, net.bsr, net.sr_t, net.avg, net.olla, net.wait, net.hcnt, net.h,
        net._snr, net._fin, net._nz if inject else dummy, net._u if inject else dummy,
        net._t, net._seed, ep, ctr, env, s0, _rng.salt(_rng.STEP), _rng.salt(0), _rng.salt(1), E, R,
        RB=RB, FB=ns.next_pow2(net.F), F_=net.F, SB=8, S_=ns.S, K=net.K, RNG=mode,
        RHO=ns.RHO, C1=math.sqrt(1 - ns.RHO ** 2), SQ2=math.sqrt(2), BYTES=ns.BYTES_PER_SE,
        SE_MIN=ns.SE_MIN, SE_MAX=ns.SE_MAX, SR_DELAY=ns.SR_DELAY, HARQ_RTT=ns.HARQ_RTT,
        HARQ_MAX=ns.HARQ_MAX, RLC_EXTRA=ns.RLC_EXTRA, PF_A=1 - 1 / ns.PF_T, PF_B=1 / ns.PF_T, PF_MIN=ns.PF_AVG_MIN,
        PHR=ns.PHR_MIN_DB, num_warps=nw)


@triton.jit
def fluid_loop_kernel(rem_ptr, snr_ptr, fin_ptr, t_ptr, R,
                      RB: tl.constexpr, FB: tl.constexpr, F_: tl.constexpr, K: tl.constexpr, S_: tl.constexpr,
                      BYTES: tl.constexpr, SE_MAX: tl.constexpr, ETA: tl.constexpr):
    """L1 fluid model (netsim.NetFluid._transmit): equal subband share among backlogged robots."""
    e = tl.program_id(0).to(tl.int64)
    r = tl.arange(0, RB)
    rm = r < R
    f = tl.arange(0, FB)
    er = e * R + r
    o_rf = er[:, None] * F_ + f[None, :]
    m_rf = rm[:, None] & (f[None, :] < F_)
    rem = tl.load(rem_ptr + o_rf, mask=m_rf, other=0.0)
    snr = tl.load(snr_ptr + er, mask=rm, other=0.0)
    t = tl.load(t_ptr + e)
    fin = tl.full([RB, FB], float("inf"), tl.float32)
    for k in range(K):
        q = tl.sum(rem, axis=1)
        back = (q > 0) & rm
        nb = tl.maximum(tl.sum(back.to(tl.int32), axis=0), 1).to(tl.float32)
        share = S_ / nb
        split = tl.maximum(share, 1.0)
        snr_sb = snr - 10.0 * libdevice.log10(split)
        se = tl.minimum(0.75 * libdevice.log2(1.0 + libdevice.pow(10.0, snr_sb / 10.0)), SE_MAX) * ETA
        b = share * se * BYTES * back.to(tl.float32)
        cum = tl.cumsum(rem, axis=1)
        newcum = tl.maximum(cum - b[:, None], 0.0)
        prev = tl.maximum(cum - rem - b[:, None], 0.0)
        new = newcum - prev
        finm = (rem > 0) & (new <= 1e-3)
        rem = tl.where(finm, 0.0, new)
        finv = (t.to(tl.float64) + (k + 1).to(tl.float64) / K).to(tl.float32)
        fin = tl.where(finm, finv, fin)
    tl.store(rem_ptr + o_rf, rem, mask=m_rf)
    tl.store(fin_ptr + o_rf, fin, mask=m_rf)


def launch_fluid(net):
    """Run the 40-slot L1 loop in place on net.rem; writes net._fin."""
    from . import netsim as ns
    RB = max(16, triton.next_power_of_2(net.R))
    nw = 8 if RB >= 128 else (4 if RB >= 64 else 2)
    fluid_loop_kernel[(net.E,)](net.rem, net._snr, net._fin, net._t, net.R,
                                RB=RB, FB=ns.next_pow2(net.F), F_=net.F, K=net.K, S_=ns.S, BYTES=ns.BYTES_PER_SE,
                                SE_MAX=ns.SE_MAX, ETA=net._eta, num_warps=nw)
