"""Batched uplink network models at several fidelity levels (L0, L0DR, L05, L05Q, L1, L2).

All state is fixed-shape tensors [E, R, ...]. Frames live in a compacted FIFO
buffer of F slots per robot (index 0 = head of line).
"""
import math
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


def lookup_key(mode, nact, snr, own, cls):
    """Bin indices for the L05 / L05Q lookup tables (same function used by the fit)."""
    dev = nact.device
    nb = torch.bucketize(nact, torch.tensor(NACT_EDGES, device=dev))
    sb = torch.bucketize(snr, torch.tensor(SNR_EDGES, device=dev))
    c = cls - 1
    if mode == "L05":
        return nb, sb, c
    ob = torch.bucketize(own, torch.tensor(OWNQ_EDGES, device=dev))
    return nb, sb, ob, c


class Radio:
    """Log-distance path loss from a gNB at the origin plus spatially correlated shadowing."""

    def __init__(self, E, device, K=8, sigma=6.0):
        ang = torch.rand(E, K, device=device) * 2 * math.pi
        wl = 20 + 40 * torch.rand(E, K, device=device)
        self.k = torch.stack([torch.cos(ang), torch.sin(ang)], -1) * (2 * math.pi / wl)[..., None]
        self.phi = torch.rand(E, K, device=device) * 2 * math.pi
        self.amp = sigma * math.sqrt(2 / K)

    def snr_db(self, pos):
        """Single-subband, full-power uplink SNR in dB for positions [E,R,2]."""
        d = pos.norm(dim=-1).clamp(min=1.0)
        pl = 40 + 35 * torch.log10(d)
        sh = self.amp * torch.cos(torch.einsum("erc,ekc->erk", pos, self.k) + self.phi[:, None, :]).sum(-1)
        return P_TX_DBM - pl - sh - NI_DBM


class NetBase:
    FIELDS = ["cap", "cls", "det", "hid", "rem", "dlv", "f_nact", "f_snr", "f_own"]
    FEATS = ["cls", "f_nact", "f_snr", "f_own"]

    def __init__(self, E, R, device, sizes):
        self.E, self.R, self.dev = E, R, device
        self.sizes = torch.tensor(sizes, device=device, dtype=torch.float32)
        self.log_stats = False
        self.log_cap_max = 10 ** 9      # only log frames captured at or before this step (avoid censoring)
        self.reset()

    def reset(self):
        E, R, d = self.E, self.R, self.dev
        self.cap = torch.full((E, R, F), -1, dtype=torch.long, device=d)
        self.cls = torch.zeros((E, R, F), dtype=torch.long, device=d)
        self.det = torch.zeros((E, R, F), dtype=torch.bool, device=d)
        self.hid = torch.full((E, R, F), -1, dtype=torch.long, device=d)
        self.rem = torch.zeros((E, R, F), device=d)
        self.dlv = torch.full((E, R, F), float("inf"), device=d)
        self.f_nact = torch.zeros((E, R, F), dtype=torch.long, device=d)
        self.f_snr = torch.zeros((E, R, F), device=d)
        self.f_own = torch.zeros((E, R, F), dtype=torch.long, device=d)
        if not hasattr(self, "stats"):
            self.clear_stats()
        self._reset_state()

    def clear_stats(self):
        self.stats = {"delay": [], "overflow": 0}
        for f in self.FEATS:
            self.stats["d_" + f] = []
            self.stats["x_" + f] = []

    def _reset_state(self):
        pass

    def queued(self):
        return (self.cap >= 0).sum(-1)

    def add_frames(self, t, send, det, hid, snr_db):
        """send [E,R] in {0,1,2}; det [E,R] bool; hid [E] current hazard id; snr_db [E,R]."""
        count = self.queued()
        new = (send > 0) & (count < F)
        if self.log_stats:
            self.stats["overflow"] += int(((send > 0) & (count >= F)).sum())
        e, r = new.nonzero(as_tuple=True)
        if e.numel() == 0:
            return
        i = count[e, r]
        c = send[e, r]
        self.cap[e, r, i] = t
        self.cls[e, r, i] = c
        self.det[e, r, i] = det[e, r]
        self.hid[e, r, i] = hid[e]
        self.rem[e, r, i] = self.sizes[c - 1]
        nact = (self.cap >= 0).any(-1).sum(-1)
        self.f_nact[e, r, i] = nact[e]
        self.f_snr[e, r, i] = snr_db[e, r]
        self.f_own[e, r, i] = i
        self._on_arrival(t, e, r, i)

    def _on_arrival(self, t, e, r, i):
        pass

    def step(self, t, snr_db, cur_hid):
        """Advance [t, t+1). Returns (newest delivered capture step [E,R], detection delivered [E])."""
        fin = self._transmit(t, snr_db)
        delivered = (self.cap >= 0) & torch.isfinite(fin)
        capd = torch.where(delivered, self.cap, torch.full_like(self.cap, -1))
        newest = capd.max(-1).values
        det_env = (delivered & self.det & (self.hid == cur_hid[:, None, None])).flatten(1).any(-1)
        timed = (self.cap >= 0) & ~delivered & ((t + 1 - self.cap) >= TIMEOUT)
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
        return newest, det_env

    def _compact(self):
        key = (self.cap < 0).long() * F + torch.arange(F, device=self.dev)
        order = key.argsort(-1)
        for n in self.FIELDS:
            setattr(self, n, getattr(self, n).gather(-1, order))

    def _after_step(self):
        pass

    def _transmit(self, t, snr_db):
        raise NotImplementedError

    def collect(self):
        st = self.stats
        cat = lambda k: torch.cat(st[k]) if st[k] else torch.zeros(0)
        return {k: cat(k) for k in st if k != "overflow"} | {"overflow": st["overflow"]}


class NetDelay(NetBase):
    """L0 / L0DR / L05: each frame gets a sampled delivery time at arrival, no queue interaction."""

    def __init__(self, E, R, device, sizes, mode, params):
        self.mode, self.params = mode, params
        super().__init__(E, R, device, sizes)

    def _reset_state(self):
        E, d = self.E, self.dev
        if self.mode == "L0DR":
            self.mu = math.log(0.05) + (math.log(10.0) - math.log(0.05)) * torch.rand(E, device=d)
            self.sig = 0.2 + 1.0 * torch.rand(E, device=d)
            self.p = 0.2 * torch.rand(E, device=d)
        elif self.mode in ("L05", "L05Q"):
            self.q = self.params["q"].to(d)          # [*key bins, 101] delay quantiles in steps
            self.pd = self.params["pdrop"].to(d)     # [*key bins]

    def _on_arrival(self, t, e, r, i):
        n = e.numel()
        d = self.dev
        if self.mode == "L0":
            mu, sig, p = self.params["mu"], self.params["sig"], self.params["p"]
            delay = torch.exp(mu + sig * torch.randn(n, device=d))
            lost = torch.rand(n, device=d) < p
        elif self.mode == "L0DR":
            delay = torch.exp(self.mu[e] + self.sig[e] * torch.randn(n, device=d))
            lost = torch.rand(n, device=d) < self.p[e]
        else:
            key = lookup_key(self.mode, self.f_nact[e, r, i], self.f_snr[e, r, i],
                             self.f_own[e, r, i], self.cls[e, r, i])
            qf = self.q[key]                         # [n, 101]
            u = torch.rand(n, device=d) * 100
            lo = u.floor().long().clamp(max=99)
            w = u - lo
            delay = qf.gather(1, lo[:, None]).squeeze(1) * (1 - w) + qf.gather(1, (lo + 1)[:, None]).squeeze(1) * w
            lost = torch.rand(n, device=d) < self.pd[key]
        dlv = t + delay
        dlv[lost] = float("inf")
        self.dlv[e, r, i] = dlv

    def _transmit(self, t, snr_db):
        ok = (self.cap >= 0) & (self.dlv < t + 1)
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
            fin_t[fin] = t + (k + 1) / UL_PER_STEP
        return fin_t


class NetSlot(NetBase):
    """L2: slot-level uplink with SR/BSR, PF over subbands, fading, OLLA, BLER and HARQ."""

    def _reset_state(self):
        E, R, d = self.E, self.R, self.dev
        self.bsr = torch.zeros(E, R, device=d)
        self.sr_t = torch.full((E, R), -1, dtype=torch.long, device=d)
        self.avg = torch.full((E, R), 100.0, device=d)
        self.olla = torch.zeros(E, R, device=d)
        self.wait = torch.zeros(E, R, dtype=torch.long, device=d)
        self.hcnt = torch.zeros(E, R, device=d)
        self.h = torch.randn(E, R, S, 2, device=d) / math.sqrt(2)
        self.ar = torch.arange(E, device=d)

    def _gain_db(self):
        return 10 * torch.log10((self.h ** 2).sum(-1).clamp(min=1e-6))

    def _transmit(self, t, snr_db):
        fin_t = torch.full_like(self.rem, float("inf"))
        ar = self.ar
        for k in range(UL_PER_STEP):
            g = t * UL_PER_STEP + k
            q = self.rem.sum(-1)
            # scheduling request for newly backlogged robots unknown to the gNB
            need_sr = (q > 0) & (self.bsr <= 0) & (self.sr_t < 0)
            self.sr_t[need_sr] = g
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
            fin_t[fin] = t + (k + 1) / UL_PER_STEP
            # link adaptation, HARQ, buffer status, PF averages
            self.olla = (self.olla + 0.05 * ok_tb - 0.45 * fail).clamp(-10, 10)
            hc = self.hcnt + 1
            exhausted = fail & (hc >= HARQ_MAX)
            self.hcnt = torch.where(fail, torch.where(exhausted, torch.zeros_like(hc), hc),
                                    torch.where(tx, torch.zeros_like(hc), self.hcnt))
            self.wait = torch.where(fail, g + HARQ_RTT + RLC_EXTRA * exhausted.long(), self.wait)
            self.bsr = torch.where(tx, self.rem.sum(-1), self.bsr)
            self.avg = (1 - 1 / PF_T) * self.avg + (1 / PF_T) * served
        return fin_t

    def _after_step(self):
        q = self.rem.sum(-1)
        self.bsr = torch.minimum(self.bsr, q)
        self.hcnt = torch.where(q > 0, self.hcnt, torch.zeros_like(self.hcnt))


def make_net(rung, E, R, device, sizes, params=None):
    if rung in ("L0", "L0DR", "L05", "L05Q"):
        return NetDelay(E, R, device, sizes, rung, params)
    if rung == "L1":
        return NetFluid(E, R, device, sizes)
    if rung == "L2":
        return NetSlot(E, R, device, sizes)
    raise ValueError(rung)
