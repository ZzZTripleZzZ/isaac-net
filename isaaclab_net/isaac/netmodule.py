"""Backend-agnostic, GPU-batched 5G uplink module for parallel robot simulators.

Registry engine of the Isaac layer. Snapshot of isaac/demo's core/ref_engine.py (2026-09-29), which is the
bug-fixed successor of the prototype isaac/netmodule.py skeleton: MessageHistory cap-0 bugfix; one-hot
(sync-free) enqueue; optional per-message `tag` carried through the FIFO and reported as
NetOutput.delivered_tag. Package change: the L2 PF average is floored at PF_AVG_MIN = 1 byte/slot as in
core/proto/netsim.py, so its L2 matches the current NetSlot (tests/test_isaac_layer.py).
Pure torch, no simulator imports, eager only. For speed use the fast engine (L2 backend graph/triton) through
isaaclab_net.isaac.NetModule.
The only contract with a physics backend is tensors:

    net = NetModule(NetConfig(num_envs=E, num_robots=R, ...))
    net.reset(env_ids)                                  # partial reset, any subset of envs
    out = net.step(poses_local[E,R,3], TrafficRequest(send[E,R]), blocked=None)
    out.delivered, out.newest_cap, out.delay_s, out.aoi_s, out.queue_bytes, out.sinr_db

Every per-env tensor is created through ``_register(name, init_fn)``, and ``reset(env_ids)``
re-initialises exactly those rows with ``init_fn(len(env_ids))``. Per-env parameters
(domain randomisation) live in a separate registry so that a reset never overwrites values an
EventTerm just sampled. Time is a per-env clock (control steps since that env's reset), so
envs that reset at different moments never compare capture times across episodes.

Rungs: "L0" i.i.d. lognormal delay + Bernoulli loss (the DelayBuffer-style baseline),
"L1" fluid equal-share uplink, "L2" slot-level SR/BSR + PF over subbands + fading + OLLA +
BLER + HARQ (ported from ../netsim.py NetSlot as of 2026-09-29: one TB per robot per slot,
power-headroom cap, rho = 0.93; per-env slot clocks). Keep in sync with netsim.py.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence, Union

import torch

EnvIds = Optional[Union[torch.Tensor, Sequence[int], slice]]
PF_AVG_MIN = 1.0      # floor on the L2 PF average (bytes/slot), as core/proto/netsim.PF_AVG_MIN


# --------------------------------------------------------------------------------------
# Configuration, inputs and outputs
# --------------------------------------------------------------------------------------
@dataclass
class NetConfig:
    num_envs: int
    num_robots: int
    device: str = "cuda"
    step_dt: float = 0.1                 # control (policy) step in seconds = sim.dt * decimation
    slot_dt: float = 0.0025              # one UL opportunity per 2.5 ms (TDD DDDSU, 30 kHz SCS)
    pose_chunks: int = 4                 # SNR re-evaluated this many times per control step
    rung: str = "L1"                     # "L0" | "L1" | "L2"
    num_subbands: int = 5                # 5 x 10 PRB on a 20 MHz carrier
    frame_depth: int = 16                # F: per-robot message FIFO depth
    msg_sizes: Sequence[float] = (1500.0, 12000.0)   # bytes per traffic class (class 1, 2, ...)
    timeout_steps: int = 20              # application deadline in control steps
    gnb_pos: Sequence[Sequence[float]] = ((0.0, 0.0, 6.0),)   # env-local gNB positions [G,3]
    shadow_modes: int = 8                # spatially correlated shadowing: sum of K plane waves
    # L2 MAC constants
    sr_delay_slots: int = 2
    harq_rtt_slots: int = 4
    harq_max: int = 4
    rlc_extra_slots: int = 10
    fading_rho: float = 0.93                 # J0(2*pi*35 Hz*2.5 ms): 3 m/s at 3.5 GHz
    phr_min_db: float = 3.0                  # power headroom floor per subband
    pf_window: float = 100.0
    backend: str = "ref"                     # "ref" = this eager engine; L2 also: "eager"/"graph"/"compile"/"triton" (fast engine, isaac/net_module.py)

    @property
    def slots_per_step(self) -> int:
        k = self.step_dt / self.slot_dt
        if abs(k - round(k)) > 1e-6:
            raise ValueError(f"step_dt/slot_dt must be an integer, got {k}")
        return int(round(k))


@dataclass
class ParamRanges:
    """Default values and DR ranges for per-env network parameters (uniform sampling)."""
    p_tx_dbm: tuple = (23.0, 23.0)
    noise_dbm: tuple = (-90.0, -90.0)        # noise plus inter-cell interference per subband
    pl_const_db: tuple = (40.0, 40.0)
    pl_exp: tuple = (3.5, 3.5)               # log-distance exponent
    shadow_sigma_db: tuple = (6.0, 6.0)
    blockage_db: tuple = (20.0, 20.0)        # extra loss when the LOS ray is blocked
    bg_load: tuple = (0.0, 0.0)              # fraction of subbands taken by background UEs
    l0_log_mu: tuple = (math.log(0.05), math.log(0.05))   # L0 only: lognormal delay in steps
    l0_log_sigma: tuple = (0.5, 0.5)
    l0_loss: tuple = (0.0, 0.0)


@dataclass
class TrafficRequest:
    """Messages each robot enqueues this control step.

    send: [E,R] long, 0 = nothing, c >= 1 = one message of traffic class c.
    bytes: optional [E,R] float payload override (else cfg.msg_sizes[c-1]).
    """
    send: torch.Tensor
    bytes: Optional[torch.Tensor] = None
    tag: Optional[torch.Tensor] = None     # [E,R] long, opaque per-message label (e.g. hazard id if detecting, else -1)


@dataclass
class NetOutput:
    delivered: torch.Tensor      # [E,R] bool, >= 1 message delivered during this step
    newest_cap: torch.Tensor     # [E,R] long, env-clock capture step of newest delivered msg, -1 if none
    last_cap: torch.Tensor       # [E,R] long, newest capture step ever delivered this episode
    delay_s: torch.Tensor        # [E,R,F] float seconds per delivered message, NaN elsewhere
    aoi_s: torch.Tensor          # [E,R] age of the freshest delivered info at the end of the step
    dropped: torch.Tensor        # [E,R] long, messages expired this step (deadline) + overflow
    queue_bytes: torch.Tensor    # [E,R]
    queue_len: torch.Tensor      # [E,R] long
    sinr_db: torch.Tensor        # [E,R] wideband SINR averaged over the step
    blocked: torch.Tensor        # [E,R] bool, LOS blocked to the serving gNB (last chunk)
    serving: torch.Tensor        # [E,R] long, serving gNB index
    delivered_tag: torch.Tensor  # [E,R,F] long, tag of each message delivered this step (pre-compaction order), -1 elsewhere


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
BYTES_PER_SE = 10 * 12 * 12 / 8          # bytes per subband per UL slot per bit/s/Hz
SE_MIN, SE_MAX = 0.2, 5.5


def se_from_snr_db(snr_db: torch.Tensor) -> torch.Tensor:
    return (0.75 * torch.log2(1 + 10 ** (snr_db / 10))).clamp(SE_MIN, SE_MAX)


def req_db(se: torch.Tensor) -> torch.Tensor:
    return 10 * torch.log10(2 ** (se / 0.75) - 1)


def serve_fifo(rem: torch.Tensor, b: torch.Tensor):
    """Serve b bytes [E,R] from FIFO frames rem [E,R,F] (index 0 = head). Returns (rem', finished)."""
    cum = rem.cumsum(-1)
    newcum = (cum - b[..., None]).clamp(min=0)
    new = torch.diff(newcum, dim=-1, prepend=torch.zeros_like(newcum[..., :1]))
    fin = (rem > 0) & (new <= 1e-3)
    return torch.where(fin, torch.zeros_like(new), new), fin


def segment_sphere_blocked(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, r: float,
                           ignore: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Torch fallback for LOS blockage by spherical proxies (other robots, people).

    a: [E,R,G,3] ray origins (gNB), b: [E,R,G,3] ray ends (UE), c: [E,M,3] blocker centres.
    ignore: [E,R,M] bool, blockers to skip (e.g. the robot itself). Returns [E,R,G] bool.
    Cost O(E R G M); fine for R, M <= ~100. The Warp kernel below replaces it for meshes.
    """
    d = b - a                                               # [E,R,G,3]
    ac = c[:, None, None, :, :] - a[..., None, :]           # [E,R,G,M,3]
    tt = ((ac * d[..., None, :]).sum(-1) / (d * d).sum(-1, keepdim=True).clamp(min=1e-9)).clamp(0.0, 1.0)
    closest = a[..., None, :] + tt[..., None] * d[..., None, :]
    hit = (closest - c[:, None, None, :, :]).norm(dim=-1) < r    # [E,R,G,M]
    hit = hit & (tt > 1e-3) & (tt < 1 - 1e-3)
    if ignore is not None:
        hit = hit & ~ignore[:, :, None, :]
    return hit.any(-1)


# --------------------------------------------------------------------------------------
# The module
# --------------------------------------------------------------------------------------
class NetModule:
    def __init__(self, cfg: NetConfig, ranges: Optional[ParamRanges] = None):
        self.cfg = cfg
        self.E, self.R, self.F = cfg.num_envs, cfg.num_robots, cfg.frame_depth
        self.dev = torch.device(cfg.device)
        self.K = cfg.slots_per_step
        self.S = cfg.num_subbands
        self.gnb = torch.tensor(cfg.gnb_pos, dtype=torch.float32, device=self.dev)   # [G,3]
        self.G = self.gnb.shape[0]
        self.sizes = torch.tensor(cfg.msg_sizes, dtype=torch.float32, device=self.dev)
        self.ranges = ranges or ParamRanges()
        self._arE = torch.arange(self.E, device=self.dev)
        self._state: dict[str, Callable[[int], torch.Tensor]] = {}
        self._params: list[str] = []
        self._build_params()
        self._build_state()

    # ---------------- registries ----------------
    def _register(self, name: str, init: Callable[[int], torch.Tensor]):
        """Create per-env state tensor self.<name> = init(E) with leading dim E; reset re-inits rows."""
        t = init(self.E)
        assert t.shape[0] == self.E, name
        setattr(self, name, t)
        self._state[name] = init

    def _full(self, shape, val, dtype=torch.float32):
        return lambda n: torch.full((n, *shape), val, dtype=dtype, device=self.dev)

    def _build_params(self):
        for name, (lo, hi) in vars(self.ranges).items():
            setattr(self, name, torch.full((self.E,), (lo + hi) / 2, device=self.dev))
            self._params.append(name)

    def _build_state(self):
        E, R, F, S, d = self.E, self.R, self.F, self.S, self.dev
        L = torch.long
        # per-env clock (control steps since this env's reset)
        self._register("t", self._full((), 0, L))
        # message FIFO, compacted so that slot 0 is head of line
        self._register("cap", self._full((R, F), -1, L))        # capture step, -1 = empty
        self._register("cls", self._full((R, F), 0, L))
        self._register("rem", self._full((R, F), 0.0))         # bytes still to send
        self._register("dlv", self._full((R, F), float("inf")))  # L0 only: sampled delivery time
        self._register("tag", self._full((R, F), -1, L))        # opaque per-message label, carried through the FIFO
        # freshness bookkeeping. last_cap = 0 means "state at reset is known" (documented choice)
        self._register("last_cap", self._full((R,), 0, L))
        # pose history for intra-step interpolation, valid flag avoids interpolating across a reset
        self._register("prev_pos", self._full((R, 3), 0.0))
        self._register("prev_valid", self._full((), False, torch.bool))
        # spatially correlated shadowing: unit-variance field, scaled by shadow_sigma_db at use time
        Kh = self.cfg.shadow_modes

        def _k(n):
            ang = torch.rand(n, Kh, device=d) * 2 * math.pi
            wl = 20 + 40 * torch.rand(n, Kh, device=d)
            return torch.stack([torch.cos(ang), torch.sin(ang)], -1) * (2 * math.pi / wl)[..., None]
        self._register("sh_k", _k)                                                    # [E,Kh,2]
        self._register("sh_phi", lambda n: torch.rand(n, Kh, device=d) * 2 * math.pi)  # [E,Kh]
        if self.cfg.rung == "L2":
            self._register("bsr", self._full((R,), 0.0))
            self._register("sr_t", self._full((R,), -1, L))       # env-clock slot index of the SR
            self._register("avg", self._full((R,), 100.0))
            self._register("olla", self._full((R,), 0.0))
            self._register("wait", self._full((R,), 0, L))        # env-clock slot until eligible
            self._register("hcnt", self._full((R,), 0.0))
            self._register("h", lambda n: torch.randn(n, R, S, 2, device=d) / math.sqrt(2))

    # ---------------- index helpers ----------------
    def _ids(self, env_ids: EnvIds) -> torch.Tensor:
        if env_ids is None or (isinstance(env_ids, slice) and env_ids == slice(None)):
            return self._arE
        if isinstance(env_ids, torch.Tensor):
            if env_ids.dtype == torch.bool:
                return env_ids.nonzero(as_tuple=True)[0].to(self.dev)
            return env_ids.to(self.dev, torch.long)
        return torch.as_tensor(list(env_ids), device=self.dev, dtype=torch.long)

    # ---------------- public API ----------------
    def reset(self, env_ids: EnvIds = None):
        """Masked reset: every registered state tensor gets fresh rows for env_ids only.

        Parameters (DR) are untouched; call set_params / sample_params for those.
        Works with index tensors, bool masks [E], python lists or None (= all).
        """
        ids = self._ids(env_ids)
        if ids.numel() == 0:
            return
        for name, init in self._state.items():
            getattr(self, name)[ids] = init(ids.numel())

    def set_params(self, env_ids: EnvIds = None, **values):
        """Overwrite per-env network parameters for env_ids. Values: scalar or [len(env_ids)] tensor."""
        ids = self._ids(env_ids)
        for k, v in values.items():
            if k not in self._params:
                raise KeyError(k)
            getattr(self, k)[ids] = torch.as_tensor(v, dtype=torch.float32, device=self.dev)

    def sample_params(self, env_ids: EnvIds = None, ranges: Optional[dict] = None):
        """Uniform DR over the given ranges (defaults to self.ranges) for env_ids."""
        ids = self._ids(env_ids)
        rng = ranges if ranges is not None else vars(self.ranges)
        for k, (lo, hi) in rng.items():
            u = torch.rand(ids.numel(), device=self.dev)
            self.set_params(ids, **{k: lo + (hi - lo) * u})

    def queued(self) -> torch.Tensor:
        return (self.cap >= 0).sum(-1)

    def state_dict(self) -> dict:
        return {k: getattr(self, k).clone() for k in list(self._state) + self._params}

    # ---------------- radio ----------------
    def snr_db(self, pos: torch.Tensor, blocked: Optional[torch.Tensor] = None):
        """Full-power single-subband uplink SNR per gNB. pos [E,R,3] env-local. Returns [E,R,G]."""
        d = (pos[:, :, None, :] - self.gnb[None, None]).norm(dim=-1).clamp(min=1.0)   # [E,R,G]
        pl = self.pl_const_db[:, None, None] + 10 * self.pl_exp[:, None, None] * torch.log10(d)
        xy = pos[..., :2]
        ph = torch.einsum("erc,ekc->erk", xy, self.sh_k) + self.sh_phi[:, None, :]
        sh = math.sqrt(2 / self.cfg.shadow_modes) * torch.cos(ph).sum(-1)              # unit variance
        sh = sh * self.shadow_sigma_db[:, None]
        snr = self.p_tx_dbm[:, None, None] - pl - sh[..., None] - self.noise_dbm[:, None, None]
        if blocked is not None:
            snr = snr - self.blockage_db[:, None, None] * blocked.float()
        return snr

    # ---------------- enqueue ----------------
    def _enqueue(self, req: TrafficRequest) -> torch.Tensor:
        # One-hot write into FIFO slot `count` (the buffer is compacted), no nonzero()/host sync.
        # Was: nonzero + fancy-index scatter, which forced a device->host sync every control step.
        count = self.queued()
        want = req.send > 0
        new = want & (count < self.F)
        overflow = (want & ~new).long()
        m = new[..., None] & (torch.arange(self.F, device=self.dev) == count[..., None])     # [E,R,F]
        c = req.send.clamp(min=1)
        size = req.bytes if req.bytes is not None else self.sizes[c - 1]
        self.cap = torch.where(m, self.t[:, None, None].expand_as(self.cap), self.cap)
        self.cls = torch.where(m, c[..., None].expand_as(self.cls), self.cls)
        self.rem = torch.where(m, size[..., None].expand_as(self.rem), self.rem)
        if req.tag is not None:
            self.tag = torch.where(m, req.tag[..., None].expand_as(self.tag), self.tag)
        else:
            self.tag = torch.where(m, torch.full_like(self.tag, -1), self.tag)
        if self.cfg.rung == "L0":
            delay = torch.exp(self.l0_log_mu[:, None] + self.l0_log_sigma[:, None] * torch.randn(self.E, self.R, device=self.dev))
            lost = torch.rand(self.E, self.R, device=self.dev) < self.l0_loss[:, None]
            dl = torch.where(lost, torch.full_like(delay, float("inf")), self.t[:, None].float() + delay)
            self.dlv = torch.where(m, dl[..., None].expand_as(self.dlv), self.dlv)
        return overflow

    # ---------------- MAC rungs; each returns finish time [E,R,F] in env-clock steps ----------------
    def _mac_l0(self, snr_chunks):
        tf = self.t.float()[:, None, None]
        ok = (self.cap >= 0) & (self.dlv < tf + 1)
        return torch.where(ok, self.dlv, torch.full_like(self.dlv, float("inf")))

    def _chunk_of(self, k: int) -> int:
        return min(k * self.cfg.pose_chunks // self.K, self.cfg.pose_chunks - 1)

    def _mac_l1(self, snr_chunks):
        fin_t = torch.full_like(self.rem, float("inf"))
        tf = self.t.float()[:, None, None]
        s_avail = (self.S * (1 - self.bg_load)).clamp(min=0.0)[:, None]         # [E,1]
        for k in range(self.K):
            snr = snr_chunks[self._chunk_of(k)]
            back = self.rem.sum(-1) > 0
            nb = back.sum(-1, keepdim=True).clamp(min=1).float()
            share = s_avail / nb
            split = share.clamp(min=1.0)
            snr_sb = snr - 10 * torch.log10(split)
            se = (0.75 * torch.log2(1 + 10 ** (snr_sb / 10))).clamp(max=SE_MAX) * 0.9
            b = share * se * BYTES_PER_SE * back
            self.rem, fin = serve_fifo(self.rem, b)
            fin_t = torch.where(fin, tf + (k + 1) / self.K, fin_t)
        return fin_t

    def _mac_l2(self, snr_chunks):
        cfg, E, R, S = self.cfg, self.E, self.R, self.S
        fin_t = torch.full_like(self.rem, float("inf"))
        tf = self.t.float()[:, None, None]
        rho = cfg.fading_rho
        # background load removes whole subbands per env (simple, maskable)
        sb_free = torch.arange(S, device=self.dev)[None, :] < (S * (1 - self.bg_load))[:, None].round()
        for k in range(self.K):
            snr = snr_chunks[self._chunk_of(k)]
            g = (self.t * self.K + k)[:, None]                                   # [E,1] env-clock slot
            q = self.rem.sum(-1)
            need_sr = (q > 0) & (self.bsr <= 0) & (self.sr_t < 0)
            self.sr_t = torch.where(need_sr, g.expand_as(self.sr_t), self.sr_t)
            granted = (self.sr_t >= 0) & (g - self.sr_t >= cfg.sr_delay_slots)
            self.bsr = torch.where(granted, self.bsr.clamp(min=1.0), self.bsr)
            self.sr_t = torch.where(granted, torch.full_like(self.sr_t, -1), self.sr_t)
            gain = lambda h: 10 * torch.log10((h ** 2).sum(-1).clamp(min=1e-6))
            gain_prev = gain(self.h)
            self.h = rho * self.h + math.sqrt(1 - rho ** 2) * torch.randn_like(self.h) / math.sqrt(2)
            gain_now = gain(self.h)
            bonus = 3.0 * self.hcnt                      # chase combining; retransmissions reuse the MCS
            est_db = snr[..., None] + gain_prev + self.olla[..., None]
            rate_est = se_from_snr_db(est_db) * BYTES_PER_SE
            elig = (self.bsr > 0) & (g >= self.wait) & (q > 0)
            need = torch.where(elig, torch.minimum(self.bsr, q), torch.zeros_like(q))
            # power headroom: stop adding subbands once per-subband SNR would fall below phr_min_db
            n_max = torch.floor(10 ** ((snr - cfg.phr_min_db) / 10)).clamp(1, S)
            cnt = torch.zeros_like(q)
            won = torch.zeros(E, R, S, dtype=torch.bool, device=self.dev)
            for s in range(S):
                m = torch.where((need > 0) & (cnt < n_max), rate_est[..., s] / self.avg,
                                torch.full_like(need, -1.0))
                best, w = m.max(-1)
                ok = (best > 0) & sb_free[:, s]
                won[self._arE, w, s] = ok
                need[self._arE, w] -= rate_est[self._arE, w, s] * ok
                cnt[self._arE, w] += ok.float()
            n = won.sum(-1)
            tx = n > 0
            nf = n.clamp(min=1).float()
            split_db = 10 * torch.log10(nf)
            se = se_from_snr_db((est_db * won).sum(-1) / nf - split_db)
            # one transport block per robot per slot, decoded on the mean-dB effective SINR
            act = snr[..., None] - split_db[..., None] + gain_now
            act_eff = (act * won).sum(-1) / nf + bonus
            p_ok = torch.sigmoid(1.5 * (act_eff - req_db(se)))
            ok_tb = tx & (torch.rand_like(p_ok) < p_ok)
            fail = tx & ~ok_tb
            served = torch.minimum(n * se * BYTES_PER_SE * ok_tb, q)
            self.rem, fin = serve_fifo(self.rem, served)
            fin_t = torch.where(fin, tf + (k + 1) / self.K, fin_t)
            self.olla = (self.olla + 0.05 * ok_tb - 0.45 * fail).clamp(-10, 10)
            hc = self.hcnt + 1
            exhausted = fail & (hc >= cfg.harq_max)
            self.hcnt = torch.where(fail, torch.where(exhausted, torch.zeros_like(hc), hc),
                                    torch.where(tx, torch.zeros_like(hc), self.hcnt))
            self.wait = torch.where(fail, g + cfg.harq_rtt_slots + cfg.rlc_extra_slots * exhausted.long(), self.wait)
            self.bsr = torch.where(tx, self.rem.sum(-1), self.bsr)
            self.avg = ((1 - 1 / cfg.pf_window) * self.avg + (1 / cfg.pf_window) * served).clamp(min=PF_AVG_MIN)
        return fin_t

    def _after_step_l2(self):
        # as netsim NetSlot._after_step: runs AFTER expired frames are removed
        q = self.rem.sum(-1)
        self.bsr = torch.minimum(self.bsr, q)
        self.hcnt = torch.where(q > 0, self.hcnt, torch.zeros_like(self.hcnt))

    # ---------------- one control step ----------------
    def step(self, pos: torch.Tensor, req: TrafficRequest, blocked: Optional[torch.Tensor] = None,
             blocked_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None) -> NetOutput:
        """Advance every env by one control step [t, t+1).

        pos: [E,R,3] env-local positions at the END of this control step (subtract env origins).
        req: messages captured at the start of this step (capture time = env clock t).
        blocked: [E,R,G] bool LOS blockage, or blocked_fn(pos_chunk)->[E,R,G] evaluated per chunk.
        """
        overflow = self._enqueue(req)
        # intra-step pose interpolation between the previous and current control-step poses
        prev = torch.where(self.prev_valid[:, None, None], self.prev_pos, pos)
        C = self.cfg.pose_chunks
        snr_chunks, sinr_acc, blk_last, serving = [], 0.0, None, None
        for c in range(C):
            a = (c + 0.5) / C
            p = prev + a * (pos - prev)
            blk = blocked_fn(p) if blocked_fn is not None else blocked
            snr_g = self.snr_db(p, blk)                                   # [E,R,G]
            snr, serving = snr_g.max(-1)                                   # strongest cell, no HO model
            snr_chunks.append(snr)
            sinr_acc = sinr_acc + snr / C
            if blk is not None:
                blk_last = blk.gather(-1, serving[..., None]).squeeze(-1)
        fin = {"L0": self._mac_l0, "L1": self._mac_l1, "L2": self._mac_l2}[self.cfg.rung](snr_chunks)

        tf1 = self.t + 1
        delivered_f = (self.cap >= 0) & torch.isfinite(fin)
        capd = torch.where(delivered_f, self.cap, torch.full_like(self.cap, -1))
        newest = capd.max(-1).values
        self.last_cap = torch.maximum(self.last_cap, newest)
        delay = torch.where(delivered_f, (fin - self.cap.float()) * self.cfg.step_dt,
                            torch.full_like(fin, float("nan")))
        dtag = torch.where(delivered_f, self.tag, torch.full_like(self.tag, -1))
        timed = (self.cap >= 0) & ~delivered_f & ((tf1[:, None, None] - self.cap) >= self.cfg.timeout_steps)
        gone = delivered_f | timed
        self.cap = torch.where(gone, torch.full_like(self.cap, -1), self.cap)
        self.rem = torch.where(gone, torch.zeros_like(self.rem), self.rem)
        self.dlv = torch.where(gone, torch.full_like(self.dlv, float("inf")), self.dlv)
        self.tag = torch.where(gone, torch.full_like(self.tag, -1), self.tag)
        order = self._compact_order()
        for n in ("cap", "cls", "rem", "dlv", "tag"):
            setattr(self, n, getattr(self, n).gather(-1, order))
        # delay stays in pre-compaction FIFO order: entry f is the f-th queued message this step
        if self.cfg.rung == "L2":
            self._after_step_l2()
        self.prev_pos = pos.clone()
        self.prev_valid = torch.ones_like(self.prev_valid)
        self.t = tf1
        return NetOutput(
            delivered=delivered_f.any(-1),
            newest_cap=newest,
            last_cap=self.last_cap.clone(),
            delay_s=delay,
            aoi_s=(self.t[:, None] - self.last_cap).float() * self.cfg.step_dt,
            dropped=timed.sum(-1) + overflow,
            queue_bytes=self.rem.sum(-1),
            queue_len=self.queued(),
            sinr_db=sinr_acc,
            blocked=blk_last if blk_last is not None else torch.zeros_like(newest, dtype=torch.bool),
            serving=serving,
            delivered_tag=dtag,
        )

    def _compact_order(self) -> torch.Tensor:
        key = (self.cap < 0).long() * self.F + torch.arange(self.F, device=self.dev)
        return key.argsort(-1)


# --------------------------------------------------------------------------------------
# What the receiver sees: delayed observations / commands
# --------------------------------------------------------------------------------------
class MessageHistory:
    """Ring buffer of per-robot payloads indexed by env-clock capture step.

    push(t_env, data) stores data[E,R,D] captured at each env's own clock. After net.step,
    update(newest_cap) replaces the receiver's copy for robots whose newest delivered capture is
    fresher than what it holds; others keep the last received payload (hold-last on loss).
    history_len must exceed cfg.timeout_steps so any deliverable capture is still stored.
    """

    def __init__(self, num_envs: int, num_robots: int, dim: int, history_len: int, device="cuda"):
        self.H, self.dev = history_len, torch.device(device)
        self.hist = torch.zeros(history_len, num_envs, num_robots, dim, device=self.dev)
        self.seen = torch.zeros(num_envs, num_robots, dim, device=self.dev)
        # BUGFIX (2026-09-29): was zeros. With seen_cap = 0 the strict test newest_cap > seen_cap
        # discarded the first delivered capture (env-clock 0) of every episode. -1 = "only the reset state".
        self.seen_cap = torch.full((num_envs, num_robots), -1, dtype=torch.long, device=self.dev)
        self._arE = torch.arange(num_envs, device=self.dev)

    def push(self, t_env: torch.Tensor, data: torch.Tensor):
        self.hist[t_env % self.H, self._arE] = data

    def update(self, newest_cap: torch.Tensor) -> torch.Tensor:
        fresher = newest_cap > self.seen_cap
        idx = newest_cap.clamp(min=0) % self.H                                  # [E,R]
        R = newest_cap.shape[1]
        got = self.hist[idx, self._arE[:, None], torch.arange(R, device=self.dev)[None, :]]
        self.seen = torch.where(fresher[..., None], got, self.seen)
        self.seen_cap = torch.where(fresher, newest_cap, self.seen_cap)
        return self.seen

    def reset(self, env_ids: torch.Tensor, init: torch.Tensor):
        """init [n,R,D]: the receiver is told the true state at reset (matches last_cap = 0)."""
        self.hist[:, env_ids] = init[None]
        self.seen[env_ids] = init
        self.seen_cap[env_ids] = -1       # BUGFIX: was 0, see __init__


def net_features(out: NetOutput, step_dt: float) -> torch.Tensor:
    """Compact per-robot network observation [E,R,5] for the policy."""
    return torch.stack([
        out.aoi_s / (10 * step_dt),
        (out.sinr_db / 30.0).clamp(-1, 2),
        out.queue_len.float() / 16.0,
        out.delivered.float(),
        out.blocked.float(),
    ], -1)


# --------------------------------------------------------------------------------------
# Optional Warp LOS kernel (static meshes). Not imported unless warp is available.
# --------------------------------------------------------------------------------------
try:
    import warp as wp

    @wp.kernel
    def los_blocked_kernel(mesh: wp.uint64, ue: wp.array(dtype=wp.vec3, ndim=2),
                           gnb: wp.array(dtype=wp.vec3), env_origin: wp.array(dtype=wp.vec3),
                           out: wp.array(dtype=wp.int32, ndim=3)):
        """One ray per (env, robot, gNB) against a static world-frame mesh (warehouse shelves).

        ue: [E,R] env-local positions, gnb: [G] env-local, env_origin: [E] world offsets.
        Launch with dim=(E, R, G). Dynamic blockers (other robots, people) use the torch
        sphere test above or a second kernel over capsules.
        """
        e, r, g = wp.tid()
        a = gnb[g] + env_origin[e]
        b = ue[e, r] + env_origin[e]
        d = b - a
        L = wp.length(d)
        q = wp.mesh_query_ray(mesh, a, d / L, L - 0.05)
        out[e, r, g] = wp.where(q.result, 1, 0)
except Exception:          # warp missing on this machine; the torch fallback still works
    wp = None
