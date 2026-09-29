"""Batched uplink network models at several fidelity levels (L0, L0DR, L05, L05Q, L1, L2).

All state is fixed-shape tensors [E, R, ...]. Frames live in a compacted FIFO
buffer of F slots per robot (index 0 = head of line).

Engine API (every level, every backend):
    net.reset(env_ids=None)                  # partial reset: env_ids = None (all), index tensor/list or bool mask [E]
    net.submit(t, requests, snr_db=None)     # enqueue new messages; returns accepted mask [E,R]
    out = net.step(t, poses_or_snr)          # advance [t, t+1); returns a dict (see NetBase.step)
Time is a per-env clock: t is None (use the engine clock net.clock [E], which reset() zeroes and step()
advances), an int (same step for every env) or a long tensor [E]. The legacy calls add_frames(t, send, det,
hid, snr_db) and step(t, snr_db, cur_hid) -> (newest, det_env) remain as thin wrappers.

Randomness: resets draw from the engine's own generator net.gen (seeded by the `seed` argument), so a
partial reset never shifts the random stream that the other envs consume while stepping.
"""
import math
from dataclasses import dataclass
from typing import Optional

import torch

UL_PER_STEP = 40            # 100 ms control step, TDD DDDSU at 30 kHz SCS: one UL slot per 2.5 ms
S = 5                       # subbands of 10 PRBs (20 MHz carrier)
BYTES_PER_SE = 10 * 12 * 12 / 8   # bytes per subband per UL slot per bit/s/Hz
P_TX_DBM = 23.0
NI_DBM = -90.0              # noise plus inter-cell interference per subband
SE_MIN, SE_MAX = 0.2, 5.5
F = 16                      # frame buffer depth per robot
TIMEOUT = 20                # application deadline in control steps (2 s)
SR_DELAY = 2                # UL slots from SR to first grant
HARQ_RTT = 4                # UL slots
HARQ_MAX = 4
RLC_EXTRA = 10              # extra UL slots after HARQ exhaustion
RHO = 0.93                  # fading correlation per UL slot, J0(2*pi*35 Hz*2.5 ms) at 3 m/s, 3.5 GHz
PF_T = 100.0
PF_AVG_MIN = 1.0            # floor on the PF average (bytes/slot) so long-idle robots never reach avg = 0
PHR_MIN_DB = 3.0            # power headroom: minimum per-subband SNR when adding subbands
NACT_EDGES = [2, 5, 9]      # L05 bins over backlogged robots per env
SNR_EDGES = [0.0, 10.0, 20.0, 30.0]
OWNQ_EDGES = [0, 1, 3]      # L05Q bins over the robot's own queued frames before this frame
RUNGS = ["L0", "L0DR", "L05", "L05Q", "L1", "L2"]


def se_from_snr_db(snr_db):
    return (0.75 * torch.log2(1 + 10 ** (snr_db / 10))).clamp(SE_MIN, SE_MAX)


def req_db(se):
    return 10 * torch.log10(2 ** (se / 0.75) - 1)


def serve_fifo(rem, b):
    """Serve b bytes [E,R] from FIFO frames rem [E,R,F]; return new rem and frames finished now."""
    cum = rem.cumsum(-1)
    newcum = (cum - b[..., None]).clamp(min=0)
    new = torch.diff(newcum, dim=-1, prepend=torch.zeros_like(newcum[..., :1]))
    fin = (rem > 0) & (new <= 1e-3)
    return torch.where(fin, torch.zeros_like(new), new), fin


def lookup_edges(device):
    """Bin edges for lookup_key as device tensors (precompute them to keep lookup_key graph-safe)."""
    return (torch.tensor(NACT_EDGES, device=device), torch.tensor(SNR_EDGES, device=device),
            torch.tensor(OWNQ_EDGES, device=device))


def lookup_key(mode, nact, snr, own, cls, edges=None):
    """Bin indices for the L05 / L05Q lookup tables (same function used by the fit)."""
    ne, se, oe = edges if edges is not None else lookup_edges(nact.device)
    nb = torch.bucketize(nact, ne)
    sb = torch.bucketize(snr, se)
    c = cls - 1
    if mode == "L05":
        return nb, sb, c
    ob = torch.bucketize(own, oe)
    return nb, sb, ob, c


def env_index(env_ids, E, device):
    """Normalize env_ids to None (= all envs) or a long index tensor on device.

    Accepts None, slice(None), a python sequence, a long index tensor or a bool mask [E]
    (a bool mask costs one host sync for nonzero; pass indices in the hot path)."""
    if env_ids is None or (isinstance(env_ids, slice) and env_ids == slice(None)):
        return None
    if isinstance(env_ids, torch.Tensor):
        if env_ids.dtype == torch.bool:
            assert env_ids.shape == (E,), env_ids.shape
            return env_ids.to(device).nonzero(as_tuple=True)[0]
        return env_ids.to(device=device, dtype=torch.long).reshape(-1)
    return torch.as_tensor(list(env_ids), dtype=torch.long, device=device)


def fill_rows(x, ids, v):
    """In-place: x[ids] = v (all rows if ids is None). v is a scalar or a [len(ids), ...] tensor."""
    if ids is None:
        if isinstance(v, torch.Tensor):
            x.copy_(v)
        else:
            x.fill_(v)
    else:
        x[ids] = v


@dataclass
class Requests:
    """Messages each robot enqueues in one control step.

    send: [E,R] long, 0 = nothing, c >= 1 = one message of traffic class c (size sizes[c-1]).
    det:  [E,R] bool, the message carries a detection of the env's current hazard (task payload, optional).
    hid:  [E] long, hazard id the detection refers to (optional).
    """
    send: torch.Tensor
    det: Optional[torch.Tensor] = None
    hid: Optional[torch.Tensor] = None


class Radio:
    """Log-distance path loss from a gNB at the origin plus spatially correlated shadowing."""

    def __init__(self, E, device, K=8, sigma=6.0, generator=None):
        self.E, self.K, self.dev, self.gen = E, K, device, generator
        self.k = torch.zeros(E, K, 2, device=device)
        self.phi = torch.zeros(E, K, device=device)
        self.amp = sigma * math.sqrt(2 / K)
        self.reset()

    def reset(self, env_ids=None):
        """Redraw the shadowing field of env_ids (all envs if None), in place."""
        ids = env_index(env_ids, self.E, self.dev)
        n = self.E if ids is None else ids.numel()
        if n == 0:
            return
        kw = dict(device=self.dev, generator=self.gen)
        ang = torch.rand(n, self.K, **kw) * 2 * math.pi
        wl = 20 + 40 * torch.rand(n, self.K, **kw)
        k = torch.stack([torch.cos(ang), torch.sin(ang)], -1) * (2 * math.pi / wl)[..., None]
        phi = torch.rand(n, self.K, **kw) * 2 * math.pi
        fill_rows(self.k, ids, k)
        fill_rows(self.phi, ids, phi)

    def snr_db(self, pos):
        """Single-subband, full-power uplink SNR in dB for positions [E,R,2] (or [E,R,3], z = height)."""
        d = pos.norm(dim=-1).clamp(min=1.0)
        pl = 40 + 35 * torch.log10(d)
        sh = self.amp * torch.cos(torch.einsum("erc,ekc->erk", pos[..., :2], self.k) + self.phi[:, None, :]).sum(-1)
        return P_TX_DBM - pl - sh - NI_DBM


class NetBase:
    FIELDS = ["cap", "cls", "det", "hid", "rem", "dlv", "f_nact", "f_snr", "f_own"]
    FEATS = ["cls", "f_nact", "f_snr", "f_own"]
    # initial value of every per-frame field
    INIT = {"cap": -1, "cls": 0, "det": False, "hid": -1, "rem": 0.0, "dlv": float("inf"),
            "f_nact": 0, "f_snr": 0.0, "f_own": 0}
    DTYPE = {"cap": torch.long, "cls": torch.long, "det": torch.bool, "hid": torch.long, "rem": torch.float32,
             "dlv": torch.float32, "f_nact": torch.long, "f_snr": torch.float32, "f_own": torch.long}

    def __init__(self, E, R, device, sizes, seed=None):
        self.E, self.R, self.dev = E, R, torch.device(device)
        self.sizes = torch.tensor(sizes, device=self.dev, dtype=torch.float32)
        self.log_stats = False
        self.log_cap_max = 10 ** 9      # only log frames captured at or before this step (avoid censoring)
        if seed is None:
            seed = int(torch.randint(0, 2 ** 62, ()).item())
        self.gen = torch.Generator(device=self.dev)
        self.gen.manual_seed(seed)
        self.radio = None
        E, R, d = self.E, self.R, self.dev
        for n in self.FIELDS:
            setattr(self, n, torch.full((E, R, F), self.INIT[n], dtype=self.DTYPE[n], device=d))
        self.clock = torch.zeros(E, dtype=torch.long, device=d)
        self._last_snr = torch.zeros(E, R, device=d)
        self._last_hid = torch.zeros(E, dtype=torch.long, device=d)
        self._alloc_state()
        self.clear_stats()
        self.reset()

    # ------------------------------------------------------------------ reset
    def reset(self, env_ids=None):
        """Re-initialize env_ids (None = all): frame buffers, MAC/level state, clock and radio. In place."""
        ids = env_index(env_ids, self.E, self.dev)
        if ids is not None and ids.numel() == 0:
            return
        for n in self.FIELDS:
            fill_rows(getattr(self, n), ids, self.INIT[n])
        fill_rows(self.clock, ids, 0)
        fill_rows(self._last_snr, ids, 0.0)
        fill_rows(self._last_hid, ids, 0)
        self._reset_state(ids)
        if self.radio is not None:
            self.radio.reset(ids)

    def _nrows(self, ids):
        return self.E if ids is None else ids.numel()

    def clear_stats(self):
        self.stats = {"delay": [], "overflow": 0}
        for f in self.FEATS:
            self.stats["d_" + f] = []
            self.stats["x_" + f] = []

    def _alloc_state(self):
        pass

    def _reset_state(self, ids):
        pass

    def queued(self):
        return (self.cap >= 0).sum(-1)

    # ------------------------------------------------------------------ helpers
    def _tvec(self, t):
        """Per-env clock [E] long from None (engine clock), an int or a tensor."""
        if t is None:
            return self.clock.clone()
        if isinstance(t, torch.Tensor):
            t = t.to(device=self.dev, dtype=torch.long)
            return t.expand(self.E).clone() if t.dim() == 0 else t
        return torch.full((self.E,), int(t), dtype=torch.long, device=self.dev)

    def attach_radio(self, radio):
        """Use an external Radio for step(t, poses); reset(env_ids) then resets its rows too."""
        self.radio = radio

    def _snr_from(self, x):
        """x is SNR [E,R] in dB, or positions [E,R,2|3] that go through the engine's Radio."""
        if x.dim() == 3:
            if self.radio is None:
                self.radio = Radio(self.E, self.dev, generator=self.gen)
            return self.radio.snr_db(x)
        return x

    # ------------------------------------------------------------------ enqueue
    def submit(self, t, requests, snr_db=None):
        """Enqueue new messages at capture time t (None = engine clock).

        requests: Requests or a send tensor [E,R]. snr_db [E,R] is the SNR recorded as a frame feature
        (used by the L05/L05Q lookup); None uses the SNR of the previous step. Returns accepted [E,R] bool
        (False where the robot sent nothing or its FIFO was full)."""
        t = self._tvec(t)
        req = requests if isinstance(requests, Requests) else Requests(send=requests)
        send = req.send
        det = req.det if req.det is not None else torch.zeros_like(send, dtype=torch.bool)
        hid = req.hid if req.hid is not None else torch.zeros(self.E, dtype=torch.long, device=self.dev)
        if snr_db is None:
            snr_db = self._last_snr
        self._last_hid = hid
        return self._enqueue(t, send, det, hid, snr_db)

    def add_frames(self, t, send, det, hid, snr_db):
        """Legacy wrapper: send [E,R] in {0,1,2}; det [E,R] bool; hid [E] current hazard id; snr_db [E,R]."""
        self.submit(t, Requests(send, det, hid), snr_db)

    def _enqueue(self, t, send, det, hid, snr_db):
        count = self.queued()
        new = (send > 0) & (count < F)
        if self.log_stats:
            self.stats["overflow"] += int(((send > 0) & (count >= F)).sum())
        draws = self._arrival_draws()
        e, r = new.nonzero(as_tuple=True)
        if e.numel() == 0:
            return new
        i = count[e, r]
        c = send[e, r]
        self.cap[e, r, i] = t[e]
        self.cls[e, r, i] = c
        self.det[e, r, i] = det[e, r]
        self.hid[e, r, i] = hid[e]
        self.rem[e, r, i] = self.sizes[c - 1]
        nact = (self.cap >= 0).any(-1).sum(-1)
        self.f_nact[e, r, i] = nact[e]
        self.f_snr[e, r, i] = snr_db[e, r]
        self.f_own[e, r, i] = i
        self._on_arrival(t, e, r, i, draws)
        return new

    def _arrival_draws(self):
        return None

    def _on_arrival(self, t, e, r, i, draws):
        pass

    # ------------------------------------------------------------------ step
    def step(self, t, x, cur_hid=None):
        """Advance [t, t+1) (t = None uses the engine clock). x: SNR [E,R] dB or positions [E,R,2|3].

        Legacy form step(t, snr_db, cur_hid) returns (newest delivered capture step [E,R], detection
        delivered [E]). Without cur_hid it returns a dict:
          delivered  [E,R,F] bool   message delivered during this step (message slots as queued before the step)
          timed_out  [E,R,F] bool   message dropped at its deadline this step
          cap, cls   [E,R,F]        capture step and traffic class of those message slots (-1 / 0 = empty)
          delay      [E,R,F] float  delivery time - capture step, in control steps; NaN if not delivered
          newest     [E,R] long     newest delivered capture step this step, -1 if none
          det_env    [E] bool       a detection of the env's current hazard (hid of the last submit) arrived
          queue_len  [E,R] long, queue_bytes [E,R] float: FIFO state after the step
          sinr_db    [E,R]          wideband SNR used for this step
          t          [E] long       the clock value of this step (the engine clock is now t + 1)
        """
        if cur_hid is not None:
            out = self._advance(t, x, cur_hid, full=False)
            return out["newest"], out["det_env"]
        return self._advance(t, x, None, full=True)

    def _advance(self, t, x, cur_hid, full):
        t = self._tvec(t)
        snr_db = self._snr_from(x)
        self._last_snr = snr_db
        if cur_hid is None:
            cur_hid = self._last_hid
        fin = self._transmit(t, snr_db)
        delivered = (self.cap >= 0) & torch.isfinite(fin)
        capd = torch.where(delivered, self.cap, torch.full_like(self.cap, -1))
        newest = capd.max(-1).values
        det_env = (delivered & self.det & (self.hid == cur_hid[:, None, None])).flatten(1).any(-1)
        timed = (self.cap >= 0) & ~delivered & ((t[:, None, None] + 1 - self.cap) >= TIMEOUT)
        out = {"newest": newest, "det_env": det_env}
        if full:
            out.update(delivered=delivered, timed_out=timed, cap=self.cap.clone(), cls=self.cls.clone(),
                       delay=torch.where(delivered, fin - self.cap.float(), torch.full_like(fin, float("nan"))))
        if self.log_stats:
            st = self.stats
            keep = self.cap <= self.log_cap_max
            dk, tk = delivered & keep, timed & keep
            st["delay"].append((fin - self.cap.float())[dk].cpu())
            for f in self.FEATS:
                st["d_" + f].append(getattr(self, f)[dk].cpu())
                st["x_" + f].append(getattr(self, f)[tk].cpu())
        gone = delivered | timed
        self.cap[gone] = -1
        self.rem[gone] = 0.0
        self.dlv[gone] = float("inf")
        self.det[gone] = False
        self._compact()
        self._after_step()
        if full:
            out.update(queue_len=self.queued(), queue_bytes=self.rem.sum(-1), sinr_db=snr_db, t=t.clone())
        self.clock.copy_(t + 1)
        return out

    def _compact(self):
        key = (self.cap < 0).long() * F + torch.arange(F, device=self.dev)
        order = key.argsort(-1)
        for n in self.FIELDS:
            setattr(self, n, getattr(self, n).gather(-1, order))

    def _after_step(self):
        pass

    def _transmit(self, t, snr_db):
        raise NotImplementedError

    @staticmethod
    def _finvals(t, k):
        """Finish time of UL slot k of step t, [E,1,1] float32 (double add, then rounded, as float(t + (k+1)/40))."""
        return (t.double() + (k + 1) / UL_PER_STEP).float()[:, None, None]

    def collect(self):
        st = self.stats
        cat = lambda k: torch.cat(st[k]) if st[k] else torch.zeros(0)
        return {k: cat(k) for k in st if k != "overflow"} | {"overflow": st["overflow"]}


class NetDelay(NetBase):
    """L0 / L0DR / L05: each frame gets a sampled delivery time at arrival, no queue interaction."""

    def __init__(self, E, R, device, sizes, mode, params, seed=None):
        self.mode, self.params = mode, params
        super().__init__(E, R, device, sizes, seed=seed)

    def _alloc_state(self):
        E, d = self.E, self.dev
        if self.mode == "L0DR":
            self.mu = torch.zeros(E, device=d)
            self.sig = torch.zeros(E, device=d)
            self.p = torch.zeros(E, device=d)
        elif self.mode in ("L05", "L05Q"):
            self.q = self.params["q"].to(d)          # [*key bins, 101] delay quantiles in steps
            self.pd = self.params["pdrop"].to(d)     # [*key bins]

    def _reset_state(self, ids):
        if self.mode == "L0DR":     # per-env randomized delay/loss, redrawn at reset
            n, kw = self._nrows(ids), dict(device=self.dev, generator=self.gen)
            fill_rows(self.mu, ids, math.log(0.05) + (math.log(10.0) - math.log(0.05)) * torch.rand(n, **kw))
            fill_rows(self.sig, ids, 0.2 + 1.0 * torch.rand(n, **kw))
            fill_rows(self.p, ids, 0.2 * torch.rand(n, **kw))

    def _arrival_draws(self):
        """Per-robot draws [E,R] for every submit (positional, so one env's arrivals never shift another's):
        z ~ N(0,1) and u1, u2 ~ U(0,1). L0/L0DR use z (delay) and u1 (loss); L05/L05Q use u1 (quantile), u2 (loss)."""
        E, R, d = self.E, self.R, self.dev
        return torch.randn(E, R, device=d), torch.rand(E, R, device=d), torch.rand(E, R, device=d)

    def _on_arrival(self, t, e, r, i, draws):
        z, u1, u2 = (x[e, r] for x in draws)
        if self.mode == "L0":
            mu, sig, p = self.params["mu"], self.params["sig"], self.params["p"]
            delay = torch.exp(mu + sig * z)
            lost = u1 < p
        elif self.mode == "L0DR":
            delay = torch.exp(self.mu[e] + self.sig[e] * z)
            lost = u1 < self.p[e]
        else:
            key = lookup_key(self.mode, self.f_nact[e, r, i], self.f_snr[e, r, i],
                             self.f_own[e, r, i], self.cls[e, r, i])
            qf = self.q[key]                         # [n, 101]
            u = u1 * 100
            lo = u.floor().long().clamp(max=99)
            w = u - lo
            delay = qf.gather(1, lo[:, None]).squeeze(1) * (1 - w) + qf.gather(1, (lo + 1)[:, None]).squeeze(1) * w
            lost = u2 < self.pd[key]
        dlv = t[e] + delay
        dlv[lost] = float("inf")
        self.dlv[e, r, i] = dlv

    def _transmit(self, t, snr_db):
        ok = (self.cap >= 0) & (self.dlv < (t + 1)[:, None, None])
        return torch.where(ok, self.dlv, torch.full_like(self.dlv, float("inf")))


class NetFluid(NetBase):
    """L1: equal share of subbands among backlogged robots, no MAC state."""

    def _transmit(self, t, snr_db):
        fin_t = torch.full_like(self.rem, float("inf"))
        for k in range(UL_PER_STEP):
            q = self.rem.sum(-1)
            back = q > 0
            nb = back.sum(-1, keepdim=True).clamp(min=1).float()
            share = S / nb
            split = share.clamp(min=1.0)
            snr_sb = snr_db - 10 * torch.log10(split)
            se = (0.75 * torch.log2(1 + 10 ** (snr_sb / 10))).clamp(max=SE_MAX) * 0.9
            b = share * se * BYTES_PER_SE * back
            self.rem, fin = serve_fifo(self.rem, b)
            fin_t = torch.where(fin, self._finvals(t, k), fin_t)
        return fin_t


class NetSlot(NetBase):
    """L2: slot-level uplink with SR/BSR, PF over subbands, fading, OLLA, BLER and HARQ."""

    MAC = ["bsr", "sr_t", "avg", "olla", "wait", "hcnt", "h"]
    MAC_INIT = {"bsr": 0.0, "sr_t": -1, "avg": 100.0, "olla": 0.0, "wait": 0, "hcnt": 0.0}

    def _alloc_state(self):
        E, R, d = self.E, self.R, self.dev
        self.bsr = torch.zeros(E, R, device=d)
        self.sr_t = torch.full((E, R), -1, dtype=torch.long, device=d)
        self.avg = torch.full((E, R), 100.0, device=d)
        self.olla = torch.zeros(E, R, device=d)
        self.wait = torch.zeros(E, R, dtype=torch.long, device=d)
        self.hcnt = torch.zeros(E, R, device=d)
        self.h = torch.zeros(E, R, S, 2, device=d)
        self.ar = torch.arange(E, device=d)

    def _reset_state(self, ids):
        for n, v in self.MAC_INIT.items():
            fill_rows(getattr(self, n), ids, v)
        n = self._nrows(ids)
        fill_rows(self.h, ids, torch.randn(n, self.R, S, 2, device=self.dev, generator=self.gen) / math.sqrt(2))

    def _gain_db(self):
        return 10 * torch.log10((self.h ** 2).sum(-1).clamp(min=1e-6))

    def _transmit(self, t, snr_db):
        fin_t = torch.full_like(self.rem, float("inf"))
        ar = self.ar
        for k in range(UL_PER_STEP):
            g = (t * UL_PER_STEP + k)[:, None]          # [E,1] env-clock slot index
            q = self.rem.sum(-1)
            # scheduling request for newly backlogged robots unknown to the gNB
            need_sr = (q > 0) & (self.bsr <= 0) & (self.sr_t < 0)
            self.sr_t = torch.where(need_sr, g, self.sr_t)
            granted = (self.sr_t >= 0) & (g - self.sr_t >= SR_DELAY)
            self.bsr[granted] = torch.clamp(self.bsr[granted], min=1.0)
            self.sr_t[granted] = -1
            # fading: gNB estimate from previous slot, transmission sees the new one
            gain_prev = self._gain_db()
            self.h = RHO * self.h + math.sqrt(1 - RHO ** 2) * torch.randn_like(self.h) / math.sqrt(2)
            gain_now = self._gain_db()
            bonus = 3.0 * self.hcnt          # chase-combining gain; retransmissions reuse the MCS
            est_db = snr_db[..., None] + gain_prev + self.olla[..., None]
            rate_est = se_from_snr_db(est_db) * BYTES_PER_SE
            elig = (self.bsr > 0) & (g >= self.wait) & (q > 0)
            need = torch.where(elig, torch.minimum(self.bsr, q), torch.zeros_like(q))
            # power headroom: stop adding subbands once per-subband SNR would fall below PHR_MIN_DB
            n_max = torch.floor(10 ** ((snr_db - PHR_MIN_DB) / 10)).clamp(1, S)
            cnt = torch.zeros_like(q)
            won = torch.zeros(self.E, self.R, S, dtype=torch.bool, device=self.dev)
            for s in range(S):
                m = torch.where((need > 0) & (cnt < n_max), rate_est[..., s] / self.avg,
                                torch.full_like(need, -1.0))
                best, w = m.max(-1)
                ok = best > 0
                won[ar, w, s] = ok
                need[ar, w] -= rate_est[ar, w, s] * ok
                cnt[ar, w] += ok.float()
            n = won.sum(-1)
            tx = n > 0
            nf = n.clamp(min=1).float()
            split_db = 10 * torch.log10(nf)
            mean_est = (est_db * won).sum(-1) / nf - split_db
            se = se_from_snr_db(mean_est)
            # one transport block per robot per slot, decoded on the mean-dB effective SINR
            act = snr_db[..., None] - split_db[..., None] + gain_now
            act_eff = (act * won).sum(-1) / nf + bonus
            p_ok = torch.sigmoid(1.5 * (act_eff - req_db(se)))
            ok_tb = tx & (torch.rand_like(p_ok) < p_ok)
            fail = tx & ~ok_tb
            served = torch.minimum(n * se * BYTES_PER_SE * ok_tb, q)
            self.rem, fin = serve_fifo(self.rem, served)
            fin_t = torch.where(fin, self._finvals(t, k), fin_t)
            # link adaptation, HARQ, buffer status, PF averages
            self.olla = (self.olla + 0.05 * ok_tb - 0.45 * fail).clamp(-10, 10)
            hc = self.hcnt + 1
            exhausted = fail & (hc >= HARQ_MAX)
            self.hcnt = torch.where(fail, torch.where(exhausted, torch.zeros_like(hc), hc),
                                    torch.where(tx, torch.zeros_like(hc), self.hcnt))
            self.wait = torch.where(fail, g + HARQ_RTT + RLC_EXTRA * exhausted.long(), self.wait)
            self.bsr = torch.where(tx, self.rem.sum(-1), self.bsr)
            self.avg = ((1 - 1 / PF_T) * self.avg + (1 / PF_T) * served).clamp(min=PF_AVG_MIN)
        return fin_t

    def _after_step(self):
        q = self.rem.sum(-1)
        self.bsr = torch.minimum(self.bsr, q)
        self.hcnt = torch.where(q > 0, self.hcnt, torch.zeros_like(self.hcnt))


def make_net(rung, E, R, device, sizes, params=None, seed=None):
    if rung in ("L0", "L0DR", "L05", "L05Q"):
        return NetDelay(E, R, device, sizes, rung, params, seed=seed)
    if rung == "L1":
        return NetFluid(E, R, device, sizes, seed=seed)
    if rung == "L2":
        return NetSlot(E, R, device, sizes, seed=seed)
    raise ValueError(rung)
