"""NetFast: graph-capturable re-implementations of every netsim fidelity level.

Same API as netsim.NetBase: reset(env_ids) / submit(t, requests) / step(t, poses_or_snr) -> dict, plus the
legacy add_frames / step(t, snr, hid) wrappers, queued / collect / clear_stats / log_stats.

Levels and backends
  L2 (NetSlot)    eager | graph | compile | triton
  L1 (NetFluid)   eager | graph | compile | triton
  L0, L0DR, L05, L05Q (NetDelay)   eager | graph | compile
  "eager"   graph-friendly ops (no nonzero / boolean-mask writes / host syncs), no capture.
  "graph"   the eager ops captured in two CUDA graphs (submit, step). Bitwise identical to the reference.
  "compile" bodies compiled with torch.compile (Inductor), then captured in CUDA graphs. May differ from the
            reference by float rounding.
  "triton"  one fused kernel for the 40-slot loop (L2: triton_slot.slot_loop_kernel, L1: fluid_loop_kernel).
            Agrees with the reference to float rounding, not bitwise.

Randomness
  L2 draws, per UL slot k, randn [E,R,S,2] (fading innovation) and rand [E,R] (BLER). NetDelay draws, per
  submit, randn [E,R] and two rand [E,R] (L0/L0DR: delay, loss; L05/L05Q: quantile, loss) for every robot and
  uses the draws of robots that enqueue. All come from the default CUDA generator inside the graph
  (graph-safe Philox) or, with inject=True, from static buffers filled by set_noise(...) so that tests can
  feed the reference and this engine the same draws. Resets draw from the engine generator self.gen, exactly
  as netsim does (same seed -> same reset draws as the reference).

Partial resets
  reset(env_ids) runs eagerly between graph replays and only writes rows of the persistent buffers in place
  (index_put / fill_), so buffer pointers and the captured graphs stay valid and nothing is reallocated.

Design notes
  * All state lives in persistent buffers updated in place (copy_), so CUDA-graph pointers stay valid.
  * The per-env clock t [E] enters the graphs through a device buffer; slot time g = 40 t + k and the finish
    time t + (k+1)/40 are computed on device (float64 then rounded to float32, exactly as the reference).
  * submit writes the new frame into slot count[e,r] with an in-place gather/scatter (O(E*R)) instead of
    nonzero + advanced indexing.
  * Compaction uses a stable cumsum-based scatter that realizes the same permutation as NetBase._compact.
  * Greedy PF over subbands stays sequential (need[] couples the subbands), with one-hot selects instead of
    advanced-index writes.
"""
import math

import torch

from . import netsim as _ns

UL_PER_STEP, S, BYTES_PER_SE, F = _ns.UL_PER_STEP, _ns.S, _ns.BYTES_PER_SE, _ns.F
TIMEOUT, SR_DELAY, HARQ_RTT, HARQ_MAX = _ns.TIMEOUT, _ns.SR_DELAY, _ns.HARQ_RTT, _ns.HARQ_MAX
RLC_EXTRA, RHO, PF_T, PHR_MIN_DB, SE_MAX = _ns.RLC_EXTRA, _ns.RHO, _ns.PF_T, _ns.PHR_MIN_DB, _ns.SE_MAX
PF_AVG_MIN = _ns.PF_AVG_MIN
se_from_snr_db, req_db, serve_fifo = _ns.se_from_snr_db, _ns.req_db, _ns.serve_fifo
Requests, Radio, env_index, fill_rows = _ns.Requests, _ns.Radio, _ns.env_index, _ns.fill_rows
DELAY_RUNGS = ("L0", "L0DR", "L05", "L05Q")

INF = float("inf")


# ----------------------------------------------------------------------------------------------
# Pure functions (no Python-side data dependence, no host syncs). Op order mirrors netsim so the
# eager/graph backends are bitwise identical to the reference.
# ----------------------------------------------------------------------------------------------
def slot_body(rem, bsr, sr_t, avg, olla, wait, hcnt, h, fin_t, snr_db, g, finv, nz, u, arR):
    """One L2 UL slot. g [E,1] long slot index, finv [E,1,1] finish time."""
    q = rem.sum(-1)
    need_sr = (q > 0) & (bsr <= 0) & (sr_t < 0)
    sr_t = torch.where(need_sr, g, sr_t)
    granted = (sr_t >= 0) & (g - sr_t >= SR_DELAY)
    bsr = torch.where(granted, torch.clamp(bsr, min=1.0), bsr)
    sr_t = torch.where(granted, torch.full_like(sr_t, -1), sr_t)
    gain_prev = 10 * torch.log10((h ** 2).sum(-1).clamp(min=1e-6))
    h = RHO * h + math.sqrt(1 - RHO ** 2) * nz / math.sqrt(2)
    gain_now = 10 * torch.log10((h ** 2).sum(-1).clamp(min=1e-6))
    bonus = 3.0 * hcnt
    est_db = snr_db[..., None] + gain_prev + olla[..., None]
    rate_est = se_from_snr_db(est_db) * BYTES_PER_SE
    elig = (bsr > 0) & (g >= wait) & (q > 0)
    need = torch.where(elig, torch.minimum(bsr, q), torch.zeros_like(q))
    n_max = torch.floor(10 ** ((snr_db - PHR_MIN_DB) / 10)).clamp(1, S)
    cnt = torch.zeros_like(q)
    wons = []
    for s in range(S):
        r_s = rate_est[..., s]
        m = torch.where((need > 0) & (cnt < n_max), r_s / avg, torch.full_like(need, -1.0))
        best, w = m.max(-1)
        sel = (arR == w[:, None]) & (best > 0)[:, None]
        wons.append(sel)
        need = need - torch.where(sel, r_s, torch.zeros_like(r_s))
        cnt = cnt + sel.float()
    won = torch.stack(wons, -1)
    n = won.sum(-1)
    tx = n > 0
    nf = n.clamp(min=1).float()
    split_db = 10 * torch.log10(nf)
    mean_est = (est_db * won).sum(-1) / nf - split_db
    se = se_from_snr_db(mean_est)
    act = snr_db[..., None] - split_db[..., None] + gain_now
    act_eff = (act * won).sum(-1) / nf + bonus
    p_ok = torch.sigmoid(1.5 * (act_eff - req_db(se)))
    ok_tb = tx & (u < p_ok)
    fail = tx & ~ok_tb
    served = torch.minimum(n * se * BYTES_PER_SE * ok_tb, q)
    rem, fin = serve_fifo(rem, served)
    fin_t = torch.where(fin, finv, fin_t)
    olla = (olla + 0.05 * ok_tb - 0.45 * fail).clamp(-10, 10)
    hc = hcnt + 1
    exhausted = fail & (hc >= HARQ_MAX)
    hcnt = torch.where(fail, torch.where(exhausted, torch.zeros_like(hc), hc),
                       torch.where(tx, torch.zeros_like(hc), hcnt))
    wait = torch.where(fail, g + HARQ_RTT + RLC_EXTRA * exhausted.long(), wait)
    bsr = torch.where(tx, rem.sum(-1), bsr)
    avg = ((1 - 1 / PF_T) * avg + (1 / PF_T) * served).clamp(min=PF_AVG_MIN)
    return rem, bsr, sr_t, avg, olla, wait, hcnt, h, fin_t


def fluid_body(rem, fin_t, snr_db, finv, eta=_ns.L1_ETA):
    """One L1 UL slot (same ops as netsim.NetFluid._transmit); eta = goodput factor."""
    q = rem.sum(-1)
    back = q > 0
    nb = back.sum(-1, keepdim=True).clamp(min=1).float()
    share = S / nb
    split = share.clamp(min=1.0)
    snr_sb = snr_db - 10 * torch.log10(split)
    se = (0.75 * torch.log2(1 + 10 ** (snr_sb / 10))).clamp(max=SE_MAX) * eta
    b = share * se * BYTES_PER_SE * back
    rem, fin = serve_fifo(rem, b)
    fin_t = torch.where(fin, finv, fin_t)
    return rem, fin_t


def add_body(cap, cls, det, hid, rem, f_nact, f_snr, f_own, send, det_in, hid_in, snr_in, t, sizes):
    """Enqueue in place: each accepting robot writes its new frame into FIFO slot count[e,r].
    O(E*R) gather/scatter at the write slot (robots that do not enqueue write their old value back)."""
    count = (cap >= 0).sum(-1)
    new = (send > 0) & (count < F)
    overflow = ((send > 0) & (count >= F)).sum()
    idx = count.clamp(max=F - 1)[..., None]
    nact = ((count + new.long()) > 0).sum(-1)       # robots with a non-empty FIFO after this submit
    put_slot(cap, idx, new, t[:, None].expand_as(send))
    put_slot(cls, idx, new, send)
    put_slot(det, idx, new, det_in)
    put_slot(hid, idx, new, hid_in[:, None].expand_as(send))
    put_slot(rem, idx, new, sizes[(send - 1).clamp(min=0)])
    put_slot(f_nact, idx, new, nact[:, None].expand_as(send))
    put_slot(f_snr, idx, new, snr_in)
    put_slot(f_own, idx, new, count)
    return overflow, new, idx, count, nact


def put_slot(x, idx, new, v):
    """In place: x[e, r, idx[e, r]] = v[e, r] where new[e, r]."""
    cur = x.gather(-1, idx).squeeze(-1)
    x.scatter_(-1, idx, torch.where(new, v.to(x.dtype), cur)[..., None])


def arrival_body(mode, new, send, count, nact, snr_in, t, z, u1, u2, lvl):
    """NetDelay._on_arrival for every robot at once; returns the delivery time [E,R] (used where new).
    lvl: dict of level tensors (mu/sig/p floats for L0; mu/sig/p [E] for L0DR; q, pd, edges for L05/L05Q)."""
    if mode == "L0":
        delay = torch.exp(lvl["mu"] + lvl["sig"] * z)
        lost = u1 < lvl["p"]
    elif mode == "L0DR":
        delay = torch.exp(lvl["mu"][:, None] + lvl["sig"][:, None] * z)
        lost = u1 < lvl["p"][:, None]
    else:
        c = torch.where(new, send, torch.ones_like(send))
        key = _ns.lookup_key(mode, nact[:, None].expand_as(send), snr_in, count, c, lvl["edges"])
        qf = lvl["q"][key]                                             # [E,R,101]
        u = u1 * 100
        lo = u.floor().long().clamp(max=99)
        w = u - lo
        delay = qf.gather(-1, lo[..., None]).squeeze(-1) * (1 - w) + qf.gather(-1, (lo + 1)[..., None]).squeeze(-1) * w
        lost = u2 < lvl["pd"][key]
    d = t[:, None] + delay
    return torch.where(lost, torch.full_like(d, INF), d)


def finish_body(cap, cls, det, hid, rem, dlv, f_nact, f_snr, f_own, fin, t, cur_hid):
    delivered = (cap >= 0) & torch.isfinite(fin)
    capd = torch.where(delivered, cap, torch.full_like(cap, -1))
    newest = capd.max(-1).values
    det_env = (delivered & det & (hid == cur_hid[:, None, None])).flatten(1).any(-1)
    timed = (cap >= 0) & ~delivered & ((t[:, None, None] + 1 - cap) >= TIMEOUT)
    delay = torch.where(delivered, fin - cap.float(), torch.full_like(fin, float("nan")))
    cap_out, cls_out = cap, cls
    gone = delivered | timed
    cap = torch.where(gone, torch.full_like(cap, -1), cap)
    rem = torch.where(gone, torch.zeros_like(rem), rem)
    dlv = torch.where(gone, torch.full_like(dlv, INF), dlv)
    det = det & ~gone
    # stable compaction: valid frames to the front in original order, invalid ones behind in original
    # order. Same permutation as argsort((cap<0)*F + arange(F)) in NetBase._compact.
    vi = (cap >= 0).long()
    dest = torch.where(vi > 0, vi.cumsum(-1) - 1, vi.sum(-1, keepdim=True) + (1 - vi).cumsum(-1) - 1)
    sc = lambda x: torch.empty_like(x).scatter_(-1, dest, x)
    cap, cls, det, hid, rem, dlv, f_nact, f_snr, f_own = (
        sc(x) for x in (cap, cls, det, hid, rem, dlv, f_nact, f_snr, f_own))
    fields = (cap, cls, det, hid, rem, dlv, f_nact, f_snr, f_own)
    return fields, (newest, det_env, delivered, timed, delay, cap_out, cls_out)


def slot_epilogue(rem, bsr, hcnt):
    """netsim.NetSlot._after_step."""
    q = rem.sum(-1)
    bsr = torch.minimum(bsr, q)
    hcnt = torch.where(q > 0, hcnt, torch.zeros_like(hcnt))
    return bsr, hcnt


# ----------------------------------------------------------------------------------------------
class NetFast:
    FIELDS = _ns.NetBase.FIELDS
    FEATS = _ns.NetBase.FEATS
    INIT, DTYPE = _ns.NetBase.INIT, _ns.NetBase.DTYPE
    MAC = _ns.NetSlot.MAC
    OUTS = ["newest", "det_env", "delivered", "timed", "delay", "cap_out", "cls_out"]

    def __init__(self, rung, E, R, device, sizes, params=None, backend="graph", inject=False, seed=None):
        assert rung in _ns.RUNGS, rung
        assert backend in ("eager", "graph", "compile", "triton")
        if backend == "triton" and rung not in ("L1", "L2"):
            raise ValueError(f"no triton kernel for {rung}; use graph")
        self.rung, self.mode = rung, rung
        self.E, self.R, self.dev = E, R, torch.device(device)
        self.backend, self.inject, self.params = backend, inject, params
        self.sizes = torch.tensor(sizes, device=self.dev, dtype=torch.float32)
        self.log_stats = False
        self.log_cap_max = 10 ** 9      # only log frames captured at or before this step (avoid censoring)
        if seed is None:
            seed = int(torch.randint(0, 2 ** 62, ()).item())
        self.gen = torch.Generator(device=self.dev)
        self.gen.manual_seed(seed)
        self.radio = None
        d = self.dev
        z = lambda shape, dt, v: torch.full(shape, v, dtype=dt, device=d)
        # persistent state buffers
        for n in self.FIELDS:
            setattr(self, n, z((E, R, F), self.DTYPE[n], self.INIT[n]))
        self.clock = z((E,), torch.long, 0)
        self._last_snr = z((E, R), torch.float32, 0.0)
        self._last_hid = z((E,), torch.long, 0)
        self._lvl = {}
        self._eta = float((params or {}).get("eta", _ns.L1_ETA)) if rung == "L1" else _ns.L1_ETA
        if rung == "L2":
            self.bsr = z((E, R), torch.float32, 0.0)
            self.sr_t = z((E, R), torch.long, -1)
            self.avg = z((E, R), torch.float32, 100.0)
            self.olla = z((E, R), torch.float32, 0.0)
            self.wait = z((E, R), torch.long, 0)
            self.hcnt = z((E, R), torch.float32, 0.0)
            self.h = z((E, R, S, 2), torch.float32, 0.0)
        elif rung == "L0":
            self._lvl = {k: float(params[k]) for k in ("mu", "sig", "p")}
        elif rung == "L0DR":
            self._dr = _ns.l0dr_ranges(params)
            self.mu = z((E,), torch.float32, 0.0)
            self.sig = z((E,), torch.float32, 0.0)
            self.p = z((E,), torch.float32, 0.0)
            self._lvl = {"mu": self.mu, "sig": self.sig, "p": self.p}
        elif rung in ("L05", "L05Q"):
            self._lvl = {"q": params["q"].to(d), "pd": params["pdrop"].to(d), "edges": _ns.lookup_edges(d)}
        # static inputs / outputs of the captured graphs
        self._t = z((E,), torch.long, 0)
        self._send = z((E, R), torch.long, 0)
        self._det_in = z((E, R), torch.bool, False)
        self._hid_in = z((E,), torch.long, 0)
        self._snr_add = z((E, R), torch.float32, 0.0)
        self._snr = z((E, R), torch.float32, 0.0)
        self._cur_hid = z((E,), torch.long, 0)
        self._newest = z((E, R), torch.long, -1)
        self._det_env = z((E,), torch.bool, False)
        self._delivered = z((E, R, F), torch.bool, False)
        self._timed = z((E, R, F), torch.bool, False)
        self._delay = z((E, R, F), torch.float32, float("nan"))
        self._cap_out = z((E, R, F), torch.long, -1)
        self._cls_out = z((E, R, F), torch.long, 0)
        self._qlen = z((E, R), torch.long, 0)
        self._qbytes = z((E, R), torch.float32, 0.0)
        self._accepted = z((E, R), torch.bool, False)
        self._fin = z((E, R, F), torch.float32, INF)
        self._overflow = torch.zeros((), dtype=torch.long, device=d)
        self._kfrac = torch.tensor([(k + 1) / UL_PER_STEP for k in range(UL_PER_STEP)], dtype=torch.float64, device=d)
        self._arR = torch.arange(R, device=d)
        self._arF = torch.arange(F, device=d)
        if inject:
            if rung == "L2":
                self._nz = torch.zeros((UL_PER_STEP, E, R, S, 2), device=d)
                self._u = torch.zeros((UL_PER_STEP, E, R), device=d)
            elif rung in DELAY_RUNGS:
                self._z = torch.zeros((E, R), device=d)
                self._u1 = torch.zeros((E, R), device=d)
                self._u2 = torch.zeros((E, R), device=d)
        self._seed = torch.randint(0, 2 ** 30, (), dtype=torch.long).to(d)   # Philox seed for the triton backend
        if backend == "compile":
            # static shapes: one specialization per (E, R); allow many sizes in one process
            torch._dynamo.config.recompile_limit = max(torch._dynamo.config.recompile_limit, 256)
            kw = dict(fullgraph=True, dynamic=False)
            self._slot, self._fluid = torch.compile(slot_body, **kw), torch.compile(fluid_body, **kw)
            self._add, self._arrival = torch.compile(add_body, **kw), torch.compile(arrival_body, **kw)
            self._finish = torch.compile(finish_body, **kw)
        else:
            self._slot, self._fluid, self._add, self._arrival, self._finish = (
                slot_body, fluid_body, add_body, arrival_body, finish_body)
        self._graphs = {}
        self._pool = None
        self.clear_stats()
        self.reset()

    # ---------------------------------------------------------------- reset
    def reset(self, env_ids=None):
        """Re-initialize env_ids (None = all) in place, outside any captured graph. Same draws as netsim."""
        ids = env_index(env_ids, self.E, self.dev)
        if ids is not None and ids.numel() == 0:
            return
        for n in self.FIELDS:
            fill_rows(getattr(self, n), ids, self.INIT[n])
        fill_rows(self.clock, ids, 0)
        fill_rows(self._last_snr, ids, 0.0)
        fill_rows(self._last_hid, ids, 0)
        n = self.E if ids is None else ids.numel()
        kw = dict(device=self.dev, generator=self.gen)
        if self.rung == "L2":
            for name, v in _ns.NetSlot.MAC_INIT.items():
                fill_rows(getattr(self, name), ids, v)
            fill_rows(self.h, ids, torch.randn(n, self.R, S, 2, **kw) / math.sqrt(2))
        elif self.rung == "L0DR":
            mu, sig, p = _ns.l0dr_draw(n, self._dr, kw)
            fill_rows(self.mu, ids, mu)
            fill_rows(self.sig, ids, sig)
            fill_rows(self.p, ids, p)
        if self.radio is not None:
            self.radio.reset(ids)

    def clear_stats(self):
        self.stats = {"delay": [], "overflow": 0}
        for f in self.FEATS:
            self.stats["d_" + f] = []
            self.stats["x_" + f] = []

    def queued(self):
        return (self.cap >= 0).sum(-1)

    def set_noise(self, *draws):
        """Inject the draws of the next call. L2 (step): nz [40,E,R,S,2] ~ N(0,1), u [40,E,R] ~ U(0,1).
        NetDelay levels (submit): z [E,R] ~ N(0,1), u1 [E,R], u2 [E,R] ~ U(0,1)."""
        bufs = (self._nz, self._u) if self.rung == "L2" else (self._z, self._u1, self._u2)
        for b, x in zip(bufs, draws):
            b.copy_(x)

    # ---------------------------------------------------------------- API
    def _tvec(self, t):
        return _ns.NetBase._tvec(self, t)

    def attach_radio(self, radio):
        self.radio = radio

    def _snr_from(self, x):
        return _ns.NetBase._snr_from(self, x)

    def submit(self, t, requests, snr_db=None):
        t = self._tvec(t)
        req = requests if isinstance(requests, Requests) else Requests(send=requests)
        self._t.copy_(t)
        self._send.copy_(req.send)
        if req.det is None:
            self._det_in.zero_()
        else:
            self._det_in.copy_(req.det)
        if req.hid is None:
            self._hid_in.zero_()
        else:
            self._hid_in.copy_(req.hid)
        self._last_hid.copy_(self._hid_in)
        self._snr_add.copy_(self._last_snr if snr_db is None else snr_db)
        self._run("add")
        if self.log_stats:
            self.stats["overflow"] += int(self._overflow)
        return self._accepted.clone()

    def add_frames(self, t, send, det, hid, snr_db):
        self.submit(t, Requests(send, det, hid), snr_db)

    def step(self, t, x, cur_hid=None):
        """See netsim.NetBase.step (same outputs)."""
        if cur_hid is not None:
            out = self._advance(t, x, cur_hid, full=False)
            return out["newest"], out["det_env"]
        return self._advance(t, x, None, full=True)

    def _advance(self, t, x, cur_hid, full):
        t = self._tvec(t)
        snr = self._snr_from(x)
        pre = None
        if self.log_stats:  # snapshot fields that the stats read before cleanup/compaction
            pre = {f: getattr(self, f).clone() for f in ["cap"] + self.FEATS}
        self._t.copy_(t)
        self._snr.copy_(snr)
        self._last_snr.copy_(snr)
        self._cur_hid.copy_(self._last_hid if cur_hid is None else cur_hid)
        self._run("step")
        if self.log_stats:
            st, cap = self.stats, pre["cap"]
            keep = cap <= self.log_cap_max
            dk, tk = self._delivered & keep, self._timed & keep
            st["delay"].append((self._fin - cap.float())[dk].cpu())
            for f in self.FEATS:
                st["d_" + f].append(pre[f][dk].cpu())
                st["x_" + f].append(pre[f][tk].cpu())
        out = {"newest": self._newest.clone(), "det_env": self._det_env.clone()}
        if full:
            out.update(delivered=self._delivered.clone(), timed_out=self._timed.clone(), cap=self._cap_out.clone(),
                       cls=self._cls_out.clone(), delay=self._delay.clone(), queue_len=self._qlen.clone(),
                       queue_bytes=self._qbytes.clone(), sinr_db=self._snr.clone(), t=t.clone())
        self.clock.copy_(t + 1)
        return out

    def collect(self):
        st = self.stats
        cat = lambda k: torch.cat(st[k]) if st[k] else torch.zeros(0)
        return {k: cat(k) for k in st if k != "overflow"} | {"overflow": st["overflow"]}

    # ---------------------------------------------------------------- bodies
    def _add_region(self):
        overflow, new, idx, count, nact = self._add(
            self.cap, self.cls, self.det, self.hid, self.rem, self.f_nact, self.f_snr, self.f_own,
            self._send, self._det_in, self._hid_in, self._snr_add, self._t, self.sizes)
        self._overflow.copy_(overflow)
        self._accepted.copy_(new)
        if self.rung in DELAY_RUNGS:
            E, R, d = self.E, self.R, self.dev
            if self.inject:
                z, u1, u2 = self._z, self._u1, self._u2
            else:
                z = torch.randn(E, R, device=d)
                u1 = torch.rand(E, R, device=d)
                u2 = torch.rand(E, R, device=d)
            dl = self._arrival(self.mode, new, self._send, count, nact, self._snr_add, self._t, z, u1, u2, self._lvl)
            put_slot(self.dlv, idx, new, dl)

    def _step_region(self):
        t = self._t
        finvals = (t.double()[:, None] + self._kfrac[None, :]).float()        # [E,40]
        if self.rung == "L2":
            fin = self._transmit_l2(t, finvals)
        elif self.rung == "L1":
            if self.backend == "triton":
                from .triton_slot import launch_fluid
                launch_fluid(self)                                            # updates rem in place, writes _fin
                fin = self._fin
            else:
                rem, fin = self.rem, torch.full_like(self.rem, INF)
                for k in range(UL_PER_STEP):
                    rem, fin = self._fluid(rem, fin, self._snr, finvals[:, k, None, None], self._eta)
                self.rem.copy_(rem)
        else:
            ok = (self.cap >= 0) & (self.dlv < (t + 1)[:, None, None])
            fin = torch.where(ok, self.dlv, torch.full_like(self.dlv, INF))
        fields, outs = self._finish(self.cap, self.cls, self.det, self.hid, self.rem, self.dlv, self.f_nact,
                                    self.f_snr, self.f_own, fin, t, self._cur_hid)
        for name, v in zip(self.OUTS, outs):          # outputs first: cap_out/cls_out may alias self.cap/self.cls
            getattr(self, "_" + name).copy_(v)
        for name, v in zip(self.FIELDS, fields):
            getattr(self, name).copy_(v)
        if self.rung == "L2":
            bsr, hcnt = slot_epilogue(self.rem, self.bsr, self.hcnt)
            self.bsr.copy_(bsr)
            self.hcnt.copy_(hcnt)
        if fin is not self._fin:
            self._fin.copy_(fin)
        self._qlen.copy_((self.cap >= 0).sum(-1))
        self._qbytes.copy_(self.rem.sum(-1))

    def _transmit_l2(self, t, finvals):
        """Runs the 40 slots; writes MAC state and rem in place, returns finish times [E,R,F]."""
        E, R, d = self.E, self.R, self.dev
        if self.backend == "triton":
            from .triton_slot import launch
            self._seed.add_(1)
            launch(self, self.inject)                       # updates MAC state and rem in place, writes _fin
            return self._fin
        gbase = t * UL_PER_STEP
        st = (self.rem, self.bsr, self.sr_t, self.avg, self.olla, self.wait, self.hcnt, self.h,
              torch.full_like(self.rem, INF))
        for k in range(UL_PER_STEP):
            if self.inject:
                nz, u = self._nz[k], self._u[k]
            else:
                nz = torch.randn(E, R, S, 2, device=d)
                u = torch.rand(E, R, device=d)
            st = self._slot(*st, self._snr, (gbase + k)[:, None], finvals[:, k, None, None].contiguous(),
                            nz, u, self._arR)
        rem, bsr, sr_t, avg, olla, wait, hcnt, h, fin = st
        for name, v in zip(["rem", "bsr", "sr_t", "avg", "olla", "wait", "hcnt", "h"],
                           (rem, bsr, sr_t, avg, olla, wait, hcnt, h)):
            getattr(self, name).copy_(v)
        return fin

    def _run(self, which):
        region = self._add_region if which == "add" else self._step_region
        if self.backend == "eager":
            region()
            return
        key = which
        gr = self._graphs.get(key)
        if gr is None:
            gr = self._capture(region)
            self._graphs[key] = gr
        gr.replay()

    def _state(self):
        names = list(self.FIELDS)
        if self.rung == "L2":
            names += self.MAC
        return names

    def _io(self):
        return [self._overflow, self._seed, self._accepted, self._fin, self._qlen, self._qbytes] + \
               [getattr(self, "_" + n) for n in self.OUTS]

    def _capture(self, region):
        """Warm up (compiles for the compile/triton backends) and capture region into a CUDA graph.
        State is snapshotted and restored so capture has no side effect on the simulation."""
        snap = {n: getattr(self, n).clone() for n in self._state()}
        snap_io = [x.clone() for x in self._io()]
        cpu_rng = torch.get_rng_state()
        s = torch.cuda.Stream(self.dev)
        s.wait_stream(torch.cuda.current_stream(self.dev))
        with torch.cuda.stream(s):
            for _ in range(2):
                region()
                for n, v in snap.items():
                    getattr(self, n).copy_(v)
        torch.cuda.current_stream(self.dev).wait_stream(s)
        g = torch.cuda.CUDAGraph()
        if self._pool is None:
            self._pool = torch.cuda.graph_pool_handle()
        with torch.cuda.graph(g, pool=self._pool):
            region()
        torch.cuda.synchronize(self.dev)
        for n, v in snap.items():
            getattr(self, n).copy_(v)
        for dst, src in zip(self._io(), snap_io):
            dst.copy_(src)
        torch.set_rng_state(cpu_rng)
        return g


class NetSlotFast(NetFast):
    """L2 (kept for backward compatibility with the original constructor signature)."""

    def __init__(self, E, R, device, sizes, backend="compile", inject=False, seed=None):
        super().__init__("L2", E, R, device, sizes, backend=backend, inject=inject, seed=seed)


def make_net_fast(rung, E, R, device, sizes, params=None, backend="compile", inject=False, seed=None):
    """Any level on any backend. backend="reference" (or "orig") returns the eager netsim engine."""
    if backend in ("reference", "orig"):
        return _ns.make_net(rung, E, R, device, sizes, params, seed=seed)
    return NetFast(rung, E, R, device, sizes, params=params, backend=backend, inject=inject, seed=seed)
