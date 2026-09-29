"""NetSlotFast: a graph-capturable, fused re-implementation of netsim.NetSlot (rung L2).

Same API as NetSlot: reset / add_frames / step / queued / collect (+ clear_stats, log_stats).

Backends
  "eager"   graph-friendly ops (no nonzero / boolean-mask writes / host syncs), no capture.
  "graph"   the eager ops captured in two CUDA graphs (add_frames, step). Bitwise identical to NetSlot.
  "compile" slot body, add_frames and step epilogue compiled with torch.compile (Inductor/Triton),
            then the whole 40-slot step captured in a CUDA graph. Fastest; may differ from NetSlot
            by float rounding (see the equivalence report).

Randomness
  NetSlot draws, per UL slot k, randn_like(h) [E,R,S,2] (fading innovation) and rand_like(p_ok) [E,R,S]
  (BLER draw). NetSlotFast draws the same shapes in the same order from the default CUDA generator
  inside the graph (graph-safe Philox), or, with inject=True, reads them from static buffers filled by
  set_noise(nz [40,E,R,S,2], u [40,E,R,S]) so that the equivalence test can feed both engines the same draws.

Design notes
  * All state lives in persistent buffers updated in place (copy_), so CUDA-graph pointers stay valid.
  * The control step index t and the hazard id enter the graph through device scalars/buffers, never as
    Python constants; slot time g = 40 t + k and the finish time t + (k+1)/40 are computed on device
    (float64 then rounded to float32, exactly as Python does in NetSlot).
  * add_frames writes the new frame into slot count[e,r] with a one-hot mask instead of nonzero + scatter.
  * _compact uses a stable cumsum-based scatter that realizes the same permutation as NetSlot's argsort.
  * Greedy PF over subbands stays sequential (need[] couples the subbands); it is unrolled inside the
    compiled slot body, with one-hot selects instead of advanced-index writes.
"""
import math
import os
import sys

import torch

try:
    import netsim as _ns
except ImportError:  # running from fast/ inside the repo
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import netsim as _ns

UL_PER_STEP, S, BYTES_PER_SE, F = _ns.UL_PER_STEP, _ns.S, _ns.BYTES_PER_SE, _ns.F
TIMEOUT, SR_DELAY, HARQ_RTT, HARQ_MAX = _ns.TIMEOUT, _ns.SR_DELAY, _ns.HARQ_RTT, _ns.HARQ_MAX
RLC_EXTRA, RHO, PF_T, PHR_MIN_DB = _ns.RLC_EXTRA, _ns.RHO, _ns.PF_T, _ns.PHR_MIN_DB
se_from_snr_db, req_db, serve_fifo = _ns.se_from_snr_db, _ns.req_db, _ns.serve_fifo

INF = float("inf")


# ----------------------------------------------------------------------------------------------
# Pure functions (no Python-side data dependence, no host syncs). Op order mirrors NetSlot so the
# eager/graph backends are bitwise identical.
# ----------------------------------------------------------------------------------------------
def slot_body(rem, bsr, sr_t, avg, olla, wait, hcnt, h, fin_t, snr_db, g, finv, nz, u, arR):
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
    avg = (1 - 1 / PF_T) * avg + (1 / PF_T) * served
    return rem, bsr, sr_t, avg, olla, wait, hcnt, h, fin_t


def add_body(cap, cls, det, hid, rem, f_nact, f_snr, f_own, send, det_in, hid_in, snr_in, t, sizes, arF):
    count = (cap >= 0).sum(-1)
    new = (send > 0) & (count < F)
    overflow = ((send > 0) & (count >= F)).sum()
    sl = (arF == count[..., None]) & new[..., None]                     # [E,R,F] one-hot write slot
    cap = torch.where(sl, t, cap)
    cls = torch.where(sl, send[..., None], cls)
    det = torch.where(sl, det_in[..., None], det)
    hid = torch.where(sl, hid_in[:, None, None], hid)
    rem = torch.where(sl, sizes[(send - 1).clamp(min=0)][..., None], rem)
    nact = (cap >= 0).any(-1).sum(-1)
    f_nact = torch.where(sl, nact[:, None, None], f_nact)
    f_snr = torch.where(sl, snr_in[..., None], f_snr)
    f_own = torch.where(sl, count[..., None], f_own)
    return cap, cls, det, hid, rem, f_nact, f_snr, f_own, overflow


def finish_body(cap, cls, det, hid, rem, dlv, f_nact, f_snr, f_own, bsr, hcnt, fin, t, cur_hid):
    delivered = (cap >= 0) & torch.isfinite(fin)
    capd = torch.where(delivered, cap, torch.full_like(cap, -1))
    newest = capd.max(-1).values
    det_env = (delivered & det & (hid == cur_hid[:, None, None])).flatten(1).any(-1)
    timed = (cap >= 0) & ~delivered & ((t + 1 - cap) >= TIMEOUT)
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
    q = rem.sum(-1)
    bsr = torch.minimum(bsr, q)
    hcnt = torch.where(q > 0, hcnt, torch.zeros_like(hcnt))
    return cap, cls, det, hid, rem, dlv, f_nact, f_snr, f_own, bsr, hcnt, newest, det_env, delivered, timed


# ----------------------------------------------------------------------------------------------
class NetSlotFast:
    FIELDS = ["cap", "cls", "det", "hid", "rem", "dlv", "f_nact", "f_snr", "f_own"]
    FEATS = ["cls", "f_nact", "f_snr", "f_own"]
    MAC = ["bsr", "sr_t", "avg", "olla", "wait", "hcnt", "h"]

    def __init__(self, E, R, device, sizes, backend="compile", inject=False):
        assert backend in ("eager", "graph", "compile", "triton")
        self.E, self.R, self.dev = E, R, torch.device(device)
        self.backend, self.inject = backend, inject
        self.sizes = torch.tensor(sizes, device=self.dev, dtype=torch.float32)
        self.log_stats = False
        self.log_cap_max = 10 ** 9      # only log frames captured at or before this step (avoid censoring)
        d = self.dev
        # persistent state buffers
        z = lambda shape, dt, v: torch.full(shape, v, dtype=dt, device=d)
        self.cap = z((E, R, F), torch.long, -1)
        self.cls = z((E, R, F), torch.long, 0)
        self.det = z((E, R, F), torch.bool, False)
        self.hid = z((E, R, F), torch.long, -1)
        self.rem = z((E, R, F), torch.float32, 0.0)
        self.dlv = z((E, R, F), torch.float32, INF)
        self.f_nact = z((E, R, F), torch.long, 0)
        self.f_snr = z((E, R, F), torch.float32, 0.0)
        self.f_own = z((E, R, F), torch.long, 0)
        self.bsr = z((E, R), torch.float32, 0.0)
        self.sr_t = z((E, R), torch.long, -1)
        self.avg = z((E, R), torch.float32, 100.0)
        self.olla = z((E, R), torch.float32, 0.0)
        self.wait = z((E, R), torch.long, 0)
        self.hcnt = z((E, R), torch.float32, 0.0)
        self.h = z((E, R, S, 2), torch.float32, 0.0)
        # static inputs / outputs of the captured graphs
        self._t = torch.zeros((), dtype=torch.long, device=d)
        self._send = z((E, R), torch.long, 0)
        self._det_in = z((E, R), torch.bool, False)
        self._hid_in = z((E,), torch.long, 0)
        self._snr_add = z((E, R), torch.float32, 0.0)
        self._snr = z((E, R), torch.float32, 0.0)
        self._cur_hid = z((E,), torch.long, 0)
        self._newest = z((E, R), torch.long, -1)
        self._det_env = z((E,), torch.bool, False)
        self._fin = z((E, R, F), torch.float32, INF)
        self._overflow = torch.zeros((), dtype=torch.long, device=d)
        self._kfrac = torch.tensor([(k + 1) / UL_PER_STEP for k in range(UL_PER_STEP)], dtype=torch.float64, device=d)
        self._arR = torch.arange(R, device=d)
        self._arF = torch.arange(F, device=d)
        if inject:
            self._nz = torch.zeros((UL_PER_STEP, E, R, S, 2), device=d)
            self._u = torch.zeros((UL_PER_STEP, E, R), device=d)
        self._seed = torch.randint(0, 2 ** 30, (), dtype=torch.long).to(d)   # Philox seed for the triton backend
        if backend in ("compile", "triton"):
            # static shapes: one specialization per (E, R); allow many sizes in one process
            torch._dynamo.config.recompile_limit = max(torch._dynamo.config.recompile_limit, 256)
            kw = dict(fullgraph=True, dynamic=False)
            self._slot = torch.compile(slot_body, **kw)
            self._add = torch.compile(add_body, **kw)
            self._finish = torch.compile(finish_body, **kw)
        else:
            self._slot, self._add, self._finish = slot_body, add_body, finish_body
        self._graphs = {}
        self._pool = None
        self.clear_stats()
        self.reset()

    # ---------------------------------------------------------------- API
    def reset(self):
        self.cap.fill_(-1); self.cls.zero_(); self.det.zero_(); self.hid.fill_(-1)
        self.rem.zero_(); self.dlv.fill_(INF); self.f_nact.zero_(); self.f_snr.zero_(); self.f_own.zero_()
        self.bsr.zero_(); self.sr_t.fill_(-1); self.avg.fill_(100.0); self.olla.zero_()
        self.wait.zero_(); self.hcnt.zero_()
        self.h.copy_(torch.randn(self.E, self.R, S, 2, device=self.dev) / math.sqrt(2))

    def clear_stats(self):
        self.stats = {"delay": [], "overflow": 0}
        for f in self.FEATS:
            self.stats["d_" + f] = []
            self.stats["x_" + f] = []

    def queued(self):
        return (self.cap >= 0).sum(-1)

    def set_noise(self, nz, u):
        """Inject the per-slot draws of the next step: nz [40,E,R,S,2] ~ N(0,1), u [40,E,R] ~ U(0,1)."""
        self._nz.copy_(nz)
        self._u.copy_(u)

    def add_frames(self, t, send, det, hid, snr_db):
        self._t.fill_(t)
        self._send.copy_(send); self._det_in.copy_(det); self._hid_in.copy_(hid); self._snr_add.copy_(snr_db)
        self._run("add")
        if self.log_stats:
            self.stats["overflow"] += int(self._overflow)

    def step(self, t, snr_db, cur_hid):
        """Advance [t, t+1). Returns (newest delivered capture step [E,R], detection delivered [E])."""
        pre = None
        if self.log_stats:  # snapshot fields that the stats read before cleanup/compaction
            pre = {f: getattr(self, f).clone() for f in ["cap"] + self.FEATS}
        self._t.fill_(t)
        self._snr.copy_(snr_db); self._cur_hid.copy_(cur_hid)
        self._run("step")
        if self.log_stats:
            st, cap = self.stats, pre["cap"]
            keep = cap <= self.log_cap_max
            dk, tk = self._delivered & keep, self._timed & keep
            st["delay"].append((self._fin - cap.float())[dk].cpu())
            for f in self.FEATS:
                st["d_" + f].append(pre[f][dk].cpu())
                st["x_" + f].append(pre[f][tk].cpu())
        return self._newest.clone(), self._det_env.clone()

    def collect(self):
        st = self.stats
        cat = lambda k: torch.cat(st[k]) if st[k] else torch.zeros(0)
        return {k: cat(k) for k in st if k != "overflow"} | {"overflow": st["overflow"]}

    # ---------------------------------------------------------------- bodies
    def _add_region(self):
        out = self._add(self.cap, self.cls, self.det, self.hid, self.rem, self.f_nact, self.f_snr, self.f_own,
                        self._send, self._det_in, self._hid_in, self._snr_add, self._t, self.sizes, self._arF)
        for name, v in zip(["cap", "cls", "det", "hid", "rem", "f_nact", "f_snr", "f_own"], out[:8]):
            getattr(self, name).copy_(v)
        self._overflow.copy_(out[8])

    def _step_region(self):
        E, R, d = self.E, self.R, self.dev
        t = self._t
        gbase = t * UL_PER_STEP
        finvals = (t.double() + self._kfrac).float()
        st = (self.rem, self.bsr, self.sr_t, self.avg, self.olla, self.wait, self.hcnt, self.h,
              torch.full_like(self.rem, INF))
        if self.backend == "triton":
            from triton_slot import launch
            self._seed.add_(1)
            launch(self, self.inject)                       # updates MAC state and rem in place
            st = (self.rem, self.bsr, self.sr_t, self.avg, self.olla, self.wait, self.hcnt, self.h, self._fin)
        for k in range(UL_PER_STEP if self.backend != "triton" else 0):
            if self.inject:
                nz, u = self._nz[k], self._u[k]
            else:
                nz = torch.randn(E, R, S, 2, device=d)
                u = torch.rand(E, R, device=d)
            st = self._slot(*st, self._snr, gbase + k, finvals[k].clone(), nz, u, self._arR)
        rem, bsr, sr_t, avg, olla, wait, hcnt, h, fin = st
        out = self._finish(self.cap, self.cls, self.det, self.hid, rem, self.dlv, self.f_nact, self.f_snr,
                           self.f_own, bsr, hcnt, fin, t, self._cur_hid)
        for name, v in zip(self.FIELDS + ["bsr", "hcnt"], out[:11]):
            getattr(self, name).copy_(v)
        for name, v in zip(["sr_t", "avg", "olla", "wait", "h"], (sr_t, avg, olla, wait, h)):
            getattr(self, name).copy_(v)
        self._newest.copy_(out[11]); self._det_env.copy_(out[12]); self._fin.copy_(fin)
        if self.log_stats:
            self._delivered.copy_(out[13]); self._timed.copy_(out[14])

    def _run(self, which):
        region = self._add_region if which == "add" else self._step_region
        if self.log_stats and not hasattr(self, "_delivered"):
            self._delivered = torch.zeros(self.E, self.R, F, dtype=torch.bool, device=self.dev)
            self._timed = torch.zeros_like(self._delivered)
        if self.backend == "eager":
            region()
            return
        key = (which, self.log_stats)
        gr = self._graphs.get(key)
        if gr is None:
            gr = self._capture(region)
            self._graphs[key] = gr
        gr.replay()

    def _state(self):
        return self.FIELDS + self.MAC

    def _capture(self, region):
        """Warm up (compiles for the compile backend) and capture region into a CUDA graph.
        State is snapshotted and restored so capture has no side effect on the simulation."""
        snap = {n: getattr(self, n).clone() for n in self._state()}
        snap_io = [x.clone() for x in (self._newest, self._det_env, self._fin, self._overflow, self._seed)]
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
        for dst, src in zip((self._newest, self._det_env, self._fin, self._overflow, self._seed), snap_io):
            dst.copy_(src)
        torch.set_rng_state(cpu_rng)
        return g


def make_net_fast(rung, E, R, device, sizes, params=None, backend="compile"):
    if rung == "L2":
        return NetSlotFast(E, R, device, sizes, backend=backend)
    return _ns.make_net(rung, E, R, device, sizes, params)
