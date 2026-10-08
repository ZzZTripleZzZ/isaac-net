"""Adaptive and mixed fidelity: one engine API over a cheap and an expensive level, routed per env.

    from isaac_net.core.adaptive import FidelityConfig, make_adaptive
    fid = FidelityConfig(cheap="L1", expensive="L2-legacy", expensive_backend="triton",
                         mode="load", indicator="backlog", up_threshold=4000.0, active_budget=0.25)
    net = make_adaptive(E, R, "cuda", NRConfig(), fid)            # reset / submit / step / clock like make_engine
    out = net.step(None, snr)                                     # out["fidelity"] [E]: 1 = this step ran on L2

AdaptiveEngine holds one instance of each level. Every env has exactly one authoritative level at a time: its
queued messages live in that level's FIFO and the other level's rows for the env are empty. Each control step runs
both instances, the cheap one on all E envs and the expensive one on its rows, and merges the step dicts per env.

Modes (FidelityConfig.mode)
  "static"     a fixed random subset of round(fraction * E) envs runs on the expensive level for the whole run
               (mixed fidelity; see docs/adaptive-fidelity.md for the training argument)
  "load"       per env, the cheap level runs while an indicator stays below up_threshold; at or above it the env
               moves to the expensive level, and it moves back when the indicator falls below down_threshold.
               Indicators (per robot, after each step): "backlog" = queued bytes, "contention" = share of robots
               with a non-empty queue, "offered" = EWMA of the bytes submitted per step. up_threshold = 0 puts every
               env on the expensive level for the whole run, up_threshold = inf on the cheap level.
  "cheap" / "expensive"   all envs on one level; set_fraction(f) (the curriculum hook) switches to static with f.

Layouts (FidelityConfig.layout / active_budget)
  "mask"       the expensive instance has E rows (row e = env e) and runs every env every step; rows of cheap envs
               hold empty queues and their outputs are discarded. The simple, correct baseline.
  "subbatch"   the expensive instance has M = active_budget rows (slots). An env that moves to the expensive level
               takes a free slot and keeps it until it moves back; M caps how many envs run expensive at once (the
               requests beyond it wait on the cheap level, highest indicator first). Only M rows pay the expensive
               step. Shapes stay fixed: the slot tables are [E] and [M] tensors, and moving an env is a masked
               gather / where, with no host sync.

Handoff (between two control steps, after the step whose indicator triggered it)
  queues      the FIFO rows move exactly (capture step, class, detection tag, remaining bytes, lookup features):
              every level stores frames in the same compacted F-slot FIFO.
  L2-legacy   entered with steady-state MAC values: BSR = queued bytes (the gNB knows the buffer, no SR pending),
              no HARQ process in flight, fading drawn from its stationary distribution CN(0, 1), OLLA offset = the
              robot's own offset when it last left the level in this episode, else the mean of its env's remembered
              offsets, else OLLA_PRIOR, and PF average = the robot's PF-average estimate: its actual PF average when
              it last left the level (the level's reset value after a reset), continued as an EWMA with the PF time
              constant of the bytes the cheap level served (delivered bytes on delay levels), floor PF_AVG_MIN. All of
              it is per env, so envs stay independent. Leaving it, the queue keeps the bytes of undecoded HARQ
              transmissions (NetSlot removes bytes only on success).
  L0 ... L05Q frames entering a delay level get a delivery time drawn from the level's delay distribution conditioned
              on the time already waited (delay > now - capture), and a loss with the matching conditional
              probability p / (p + (1 - p) (1 - F(waited))).
  L1          has no state beyond the FIFO.
  L2 (NR)     the FIFO is laid out on a fresh byte stream (air bytes = remaining payload x air(size) / size), MAC state
              from its reset values with BSR = queued air bytes, CSI = the gain of a stationary fading draw, and OLLA
              and PF average as for L2-legacy (per UL data slot); leaving it, a frame keeps the payload share of its
              air bytes above the RLC in-order pointer (at least 1 byte).
Resets start an env on the level its mode assigns at zero load, with that level's exact reset state.

Randomness. make_adaptive requires rng="engine" (NRConfig.rng default): both instances use the same seed, so an env's
draws on either level are the ones a plain engine of that level draws for it (keyed by env id, episode and call
count). The NR engine (L2) draws its stepping randomness from the global RNG, so it matches a plain L2 engine only
under the same global seed and layout. In the subbatch layout each slot's stream keys are set to its env's, so for
the reference / eager / graph backends the subbatch layout is bitwise equal to the mask layout (on the CPU for
M = E; see docs/adaptive-fidelity.md). The triton kernel of L2-legacy keys its draws by
row index: there a slot run by another env gets an independent stream (hashed episode key) instead.

Graph mode (FidelityConfig.graph, "auto" = CUDA and fast backends on both levels): after two eager calls, submit, step
and reset are each captured in one CUDA graph that runs both levels' regions and the routing, so a step costs a few
graph launches instead of hundreds of small kernels; reset is a masked reset of both levels with the same keyed
draws as their own reset. Bitwise equal to the eager engine.

Step dict: the merged outputs of the two levels (same keys as make_engine), plus
  fidelity [E] long             1 if the env's step ran on the expensive level
  fidelity_indicator [E] float  the indicator after the step (before any switch it triggers)
fidelity_stats() returns cumulative counts: switches up / down, requests denied by the budget, env-steps per level.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from . import cuda_graph
from .checkpoint import StateDictMixin
from .config import NRConfig
from .proto import netsim as _ns
from .proto.rng import RESET, STEP, SUBMIT, CounterRNG, combine, mix32

HANDOFF = 4                                       # counter-RNG channel of the handoff draws (rng.py uses 1 to 3)
FIFO = ("cap", "cls", "det", "hid", "rem", "f_nact", "f_snr", "f_own")
FIELDS = FIFO + ("dlv",)
DELAY_LEVELS = ("L0", "L0DR", "L05", "L05Q")
CHEAP_LEVELS = DELAY_LEVELS + ("L1",)
EXPENSIVE_LEVELS = ("L1", "L2-legacy", "L2")
MODES = ("static", "load", "cheap", "expensive")
INDICATORS = ("backlog", "contention", "offered")
INF = float("inf")
# OLLA offset (dB) of a robot entering the expensive level when neither it nor its env has a remembered offset: the
# steady-state mean measured on L2-legacy (about -3.1 dB, std 1.5-2 dB, nearly independent of load; see
# docs/adaptive-fidelity.md); 0 = the NR engine's reset value
OLLA_PRIOR = {"L2-legacy": -3.0, "L2": 0.0}


@dataclass
class FidelityConfig:
    """Adaptive / mixed fidelity (core/adaptive.py). NRConfig(fidelity=FidelityConfig(...)) + make_adaptive, or
    make_adaptive(E, R, dev, cfg, FidelityConfig(...)).

      cheap, expensive       levels: cheap in L0, L0DR, L05, L05Q, L1; expensive in L1, L2-legacy (single cell) or
                             L2 (the NR engine: one cell, uplink, no traffic models)
      *_backend, *_params    make_engine backend and params of each level
      mode                   "static", "load", "cheap" or "expensive" (see the module docstring)
      fraction               static: share of envs on the expensive level (a fixed random subset)
      indicator              load: "backlog" (queued bytes per robot), "contention" (share of robots with a
                             non-empty queue) or "offered" (EWMA of submitted bytes per robot per step)
      up_threshold           load: move up when indicator >= up_threshold (0 = always expensive, inf = never)
      down_threshold         load: move down when indicator < down_threshold (None = up_threshold / 2)
      min_dwell_steps        load: steps an env stays on a level before it may switch again
      offered_ewma           weight of the old value in the "offered" EWMA (per control step)
      layout                 "mask", "subbatch" or "auto" (static: subbatch sized to the static set; otherwise
                             subbatch if active_budget is set, else mask)
      active_budget          subbatch rows M: an int, or a float in (0, 1] = share of E (rounded up)
      decision_period        evaluate switches every this many steps (1 = every step); the switching work runs only
                             on those steps (graph mode captures a step with and a step without it)
      olla_prior_db          OLLA offset of a robot entering the expensive level without a remembered offset in
                             its env (None = OLLA_PRIOR of the level: -3 dB for L2-legacy, 0 for L2)
      graph                  capture submit, step and reset (both levels and the routing) in one CUDA graph each
                             after two eager calls; "auto" = on when the device is CUDA and both levels use a fast
                             backend
                             (eager / graph / compile / triton). Calls with an explicit t or cur_hid run eagerly.
    """

    cheap: str = "L1"
    expensive: str = "L2-legacy"
    cheap_backend: str = "reference"
    expensive_backend: str = "reference"
    cheap_params: object = None
    expensive_params: object = None
    mode: str = "load"
    fraction: float = 0.1
    indicator: str = "backlog"
    up_threshold: float = 4000.0
    down_threshold: float | None = None
    min_dwell_steps: int = 5
    offered_ewma: float = 0.5
    layout: str = "auto"
    active_budget: int | float | None = None
    decision_period: int = 1
    graph: bool | str = "auto"
    olla_prior_db: float | None = None

    def __post_init__(self):
        if self.cheap not in CHEAP_LEVELS:
            raise ValueError(f"cheap level {self.cheap!r}; one of {CHEAP_LEVELS}")
        if self.expensive not in EXPENSIVE_LEVELS:
            raise ValueError(f"expensive level {self.expensive!r}; one of {EXPENSIVE_LEVELS}")
        if self.cheap == self.expensive:
            raise ValueError("cheap and expensive levels must differ")
        assert self.mode in MODES, f"mode must be one of {MODES}"
        assert self.indicator in INDICATORS, f"indicator must be one of {INDICATORS}"
        assert self.layout in ("auto", "mask", "subbatch") and self.graph in ("auto", True, False)
        assert 0.0 <= self.fraction <= 1.0 and self.min_dwell_steps >= 0 and self.decision_period >= 1
        assert 0.0 <= self.offered_ewma < 1.0
        assert self.up_threshold >= 0.0
        assert self.down_threshold is None or 0.0 <= self.down_threshold <= self.up_threshold

    @property
    def down(self):
        return self.up_threshold / 2 if self.down_threshold is None else self.down_threshold

    def budget_rows(self, E):
        """Rows M of the expensive instance, or None for the mask layout."""
        layout = self.layout
        if layout == "auto":
            layout = "subbatch" if (self.mode == "static" or self.active_budget is not None) else "mask"
        if layout == "mask":
            return None
        b = self.active_budget
        if b is None:
            b = math.ceil(self.fraction * E) if self.mode == "static" else E
        elif isinstance(b, float):
            if not 0.0 < b <= 1.0:
                raise ValueError("active_budget as a float is a share of E in (0, 1]")
            b = math.ceil(b * E)
        return max(1, min(int(b), E))


def make_adaptive(E, R, device="cpu", config: NRConfig | None = None, fidelity: FidelityConfig | None = None, *,
                  seed=None):
    """AdaptiveEngine over make_engine(fidelity.cheap) and make_engine(fidelity.expensive). The fidelity config
    comes from the argument or config.fidelity. The wrappers of the config go around the adaptive engine, not
    around the levels (whose step the adaptive engine bypasses): config.edge wraps it in EdgeLoop and config.energy
    in EnergyLoop (outermost, as make_engine does). Background users (config.background with n_background > 0)
    are refused: their offered load scales the queue of one level in place, which the handoff between the levels
    does not carry."""
    cfg = config if config is not None else NRConfig()
    bg, en = getattr(cfg, "background", None), getattr(cfg, "energy", None)
    if bg is not None and getattr(bg, "n_background", 0) > 0:
        raise ValueError("make_adaptive does not support background users (NRConfig.background): the offered-load "
                         "wrapper scales one level's queue in place and the handoff does not carry it; use "
                         "make_engine with a single level")
    net = AdaptiveEngine(E, R, device, cfg, fidelity, seed=seed)
    if getattr(cfg, "edge", None) is not None:
        from .edge import EdgeLoop
        net = EdgeLoop(net, cfg.edge)
    if en is not None:
        from .energy import EnergyLoop
        net = EnergyLoop(net, en, seed=net.seed, config=cfg)
    return net


def _where_rows(mask, a, b):
    """torch.where(mask [rows] broadcast over the trailing dims of a, a, b)."""
    return torch.where(mask.view(-1, *([1] * (a.dim() - 1))), a, b)


class _Level:
    """Row access to one prototype engine (NetBase reference or NetFast): read, write, clear and load rows."""

    has_rng = True                     # counter-RNG streams keyed by env (rng="engine")

    def __init__(self, eng, level):
        self.eng, self.level = eng, level
        self.delay = level in DELAY_LEVELS
        self.slot = level == "L2-legacy"
        self.triton = getattr(eng, "backend", None) == "triton"
        self.h_shape = (eng.R, _ns.S, 2) if self.slot else None
        self.edges = _ns.lookup_edges(eng.dev) if level in ("L05", "L05Q") else None

    def get(self, name):
        return getattr(self.eng, name)

    def fifo(self):
        """FIFO rows in the common format: cap (env clock), cls, det, hid, rem (payload bytes), f_* [rows,R,F]."""
        return {n: getattr(self.eng, n) for n in FIFO}

    def olla(self):
        """OLLA offsets [rows,R] (dB) of a level with link adaptation, else None."""
        return self.eng.olla if self.slot else None

    def pf_served(self):
        """PF average as bytes per control step [rows,R] (L2-legacy: avg per UL slot x K), else None."""
        return self.eng.avg * self.eng.K if self.slot else None

    def pf_reset_served(self):
        """The level's reset PF average in bytes per control step (0 without a PF scheduler)."""
        return _ns.NetSlot.MAC_INIT["avg"] * self.eng.K if self.slot else 0.0

    def submit(self, t, req, snr):
        return self.eng.submit(t, req, snr)

    def put(self, name, mask, value):
        """eng.<name>[rows where mask] = value (in place: graph backends keep their buffer pointers)."""
        t = getattr(self.eng, name)
        if not torch.is_tensor(value):
            value = torch.full_like(t, value)
        t.copy_(_where_rows(mask, value.to(t.dtype), t))

    def clear(self, mask):
        for n in FIELDS:
            self.put(n, mask, _ns.NetBase.INIT[n])

    def queued(self):
        return (self.eng.cap >= 0).sum(-1)

    def lvl(self):
        """Delay-level parameters as the fast backend's arrival_body expects them."""
        e = self.eng
        if self.level == "L0":
            lvl = getattr(e, "_lvl", None) or {}
            if "q" in lvl:                       # empirical marginal (fast backends)
                return {"q": lvl["q"], "p": float(lvl["p"])}
            if "q" in e.params:                  # empirical marginal (reference backend)
                return {"q": e.q0, "p": float(e.params["p"])}
            return {k: float(e.params[k]) for k in ("mu", "sig", "p")}
        if self.level == "L0DR":
            return {"mu": e.mu[:, None, None], "sig": e.sig[:, None, None], "p": e.p[:, None, None]}
        q = e._lvl["q"] if hasattr(e, "_lvl") else e.q
        pd = e._lvl["pd"] if hasattr(e, "_lvl") else e.pd
        return {"q": q, "pd": pd, "edges": self.edges}

    def cond_dlv(self, src, now, u, v):
        """Delivery time [rows,R,F] of queued frames entering this delay level at env clock now [rows]: delay drawn
        from the level's distribution conditioned on delay > waited = now - cap, loss with the conditional
        probability p / (p + (1 - p)(1 - F(waited))). u, v: U(0,1) draws [rows,R,F]."""
        cap = src["cap"]
        valid = cap >= 0
        nowf = now.double()[:, None, None]
        w = (nowf - cap.double()).clamp(min=0.0).float()
        lv = self.lvl()
        top = 1.0 - 2.0 ** -24
        if self.level == "L0" and "q" in lv:
            # empirical marginal: delay = q[floor(u K)] over the sorted table q [K]; conditioned on delay > w the
            # index is uniform over the entries above w, which starts at F(w) = #(q <= w) / K
            q0, p = lv["q"], lv["p"]
            K = q0.numel()
            Fw = torch.searchsorted(q0, w.contiguous(), right=True).float() / K
            uu = Fw + u * (1 - Fw)
            d = q0[(uu * K).long().clamp(0, K - 1)]
        elif self.level in ("L0", "L0DR"):
            mu, sig, p = lv["mu"], lv["sig"], lv["p"]
            lw = torch.log(w.clamp(min=1e-30))
            Fw = torch.where(w > 0, 0.5 * (1 + torch.erf((lw - mu) / (sig * math.sqrt(2)))), torch.zeros_like(w))
            uu = (Fw + u * (1 - Fw)).clamp(max=top)
            d = torch.exp(mu + sig * math.sqrt(2) * torch.erfinv(2 * uu - 1))
        else:
            c = torch.where(valid, src["cls"], torch.ones_like(src["cls"]))
            key = _ns.lookup_key(self.level, src["f_nact"], src["f_snr"], src["f_own"], c, lv["edges"])
            qf = lv["q"][key]                                             # [rows,R,F,101]
            p = lv["pd"][key]
            idx = torch.searchsorted(qf.contiguous(), w[..., None].contiguous()).squeeze(-1)   # first q >= w
            lo = (idx - 1).clamp(0, 99)
            a = qf.gather(-1, lo[..., None]).squeeze(-1)
            b = qf.gather(-1, (lo + 1)[..., None]).squeeze(-1)
            frac = torch.where(b > a, ((w - a) / (b - a).clamp(min=1e-30)).clamp(0, 1), torch.ones_like(w))
            Fw = torch.where(idx > 100, torch.ones_like(w), (lo + frac) / 100)
            Fw = torch.where(idx <= 0, torch.zeros_like(w), Fw)
            uu = (Fw + u * (1 - Fw)).clamp(max=1.0)
            x = uu * 100
            lo2 = x.floor().long().clamp(max=99)
            ww = x - lo2
            q_lo = qf.gather(-1, lo2[..., None]).squeeze(-1)
            q_hi = qf.gather(-1, (lo2 + 1)[..., None]).squeeze(-1)
            d = q_lo * (1 - ww) + q_hi * ww
        den = p + (1 - p) * (1 - Fw)
        pc = torch.where(den > 0, p / den.clamp(min=1e-30), torch.ones_like(Fw))
        lost = v < pc
        dlv = torch.maximum(cap.float() + d, nowf.float())
        return torch.where(valid & ~lost, dlv, torch.full_like(dlv, INF))

    def load(self, mask, src, now, ctx):
        """Rows where mask take the FIFO src (dict of [rows,R,F]) at env clock now [rows], with this level's
        entry rule (module docstring). ctx: "u", "v" [rows,R,F] (delay levels), "served" [rows,R] (delivered bytes
        per control step, EWMA), "olla" [rows,R] (dB) and "h0" [rows,R,S,2] (L2-legacy)."""
        for n in FIFO:
            self.put(n, mask, src[n])
        if self.delay:
            self.put("dlv", mask, self.cond_dlv(src, now, ctx["u"], ctx["v"]))
        else:
            self.put("dlv", mask, INF)
        if self.slot:
            self.put("bsr", mask, src["rem"].sum(-1))
            self.put("sr_t", mask, -1)
            self.put("avg", mask, (ctx["served"] / self.eng.K).clamp(min=_ns.PF_AVG_MIN))
            self.put("olla", mask, ctx["olla"])
            self.put("wait", mask, 0)
            self.put("hcnt", mask, 0.0)
            self.put("h", mask, ctx["h0"])
        self.put("clock", mask, now)

    def advance(self, t, snr, cur_hid):
        return self.eng._advance(t, snr, cur_hid, True)

    def masked_reset(self, m):
        """Masked, sync-free equivalent of eng.reset(rows where m) (NetBase / NetFast with rng="engine"): new episode
        and zeroed call counters, FIFO, clock and last inputs back to their initial values, and the level's reset
        draws (L2-legacy fading, L0DR delay parameters, the radio's shadowing field) from the same keyed streams."""
        e, rng = self.eng, self.eng.rng
        rng.episode.add_(m.long())
        for c in rng.ctr.values():
            c.copy_(torch.where(m, torch.zeros_like(c), c))
        self.clear(m)
        for n, v in (("clock", 0), ("_last_snr", 0.0), ("_last_hid", 0)):
            self.put(n, m, v)
        z = rng._zero
        rand = lambda st, *tail: rng._draw(rng.env, rng.episode, z, RESET, st, tail, False)   # noqa: E731
        if self.slot:
            for n, v in _ns.NetSlot.MAC_INIT.items():
                self.put(n, m, v)
            self.put("h", m, rng._draw(rng.env, rng.episode, z, RESET, 0, self.h_shape, True) / math.sqrt(2))
        if self.level == "L0DR":
            dr = e._dr if hasattr(e, "_dr") else e.dr
            mu, sig, p = _ns.l0dr_draw(e.E, dr, None, lambda i: rand(1 + i))
            for n, v in (("mu", mu), ("sig", sig), ("p", p)):
                self.put(n, m, v)
        r = e.radio
        if r is not None:                     # proto Radio.reset with the engine's streams (10 + i)
            ang = rand(10, r.K) * 2 * math.pi
            wl = 20 + 40 * rand(11, r.K)
            k = torch.stack([torch.cos(ang), torch.sin(ang)], -1) * (2 * math.pi / wl)[..., None]
            r.k.copy_(_where_rows(m, k, r.k))
            r.phi.copy_(_where_rows(m, rand(12, r.K) * 2 * math.pi, r.phi))

    def reset_rows(self, mask):
        """Masked, sync-free equivalent of eng.reset(rows) for the slot levels (L1, L2-legacy), given that the RNG rows
        already hold the new episode with zeroed call counters: FIFO and MAC back to their initial values, fading
        from the level's own reset draw (stream 0 of the RESET channel), clock 0."""
        self.clear(mask)
        if self.slot:
            for n, v in _ns.NetSlot.MAC_INIT.items():
                self.put(n, mask, v)
            rng = self.eng.rng
            h0 = rng._draw(rng.env, rng.episode, rng._zero, RESET, 0, self.h_shape, True) / math.sqrt(2)
            self.put("h", mask, h0)
        self.put("clock", mask, 0)


class _NRLevel:
    """Row access to the NR engine (level "L2", NREngine over NRNet, one cell, uplink only).

    The NR queue is a per-robot byte stream: frame i occupies stream bytes [start, end) on the air (payload plus
    per-packet overhead, NRNet.air_bytes), and a frame completes when the RLC in-order pointer (MacLink.ack_ptr)
    passes its end. Converting to the common FIFO format: the remaining payload of a frame is its size times the
    share of its air bytes above the in-order pointer (at least 1 byte while it is queued), and the capture step
    moves between the engine's global clock and the env clock through NREngine.epoch. Entering, the frames are laid
    out from stream offset 0 in FIFO order, with air bytes = remaining payload times air(size) / size."""

    has_rng = False
    delay = slot = triton = False

    def __init__(self, eng, level):
        self.eng, self.level = eng, level
        self.net = eng.net
        c = eng.config
        self.h_shape = (eng.R, c.n_subbands, 2)
        self.n_ul = c.ul_slots_per_step

    def _sizes(self, cls):
        size = self.net.sizes[(cls - 1).clamp(min=0)]
        return size, self.net.air_bytes(size)

    def fifo(self):
        q, ul, eng = self.net.ul.q, self.net.ul, self.eng
        valid = q.cap >= 0
        cap = torch.where(valid, q.cap - eng.epoch[:, None, None], torch.full_like(q.cap, -1))
        size, air = self._sizes(q.cls)
        left = (q.end - torch.maximum(q.start, ul.ack_ptr()[..., None])).clamp(min=0)
        rem = torch.where(valid, (size * left.float() / air).clamp(min=1.0), torch.zeros_like(size))
        out = {"cap": cap, "rem": rem}
        for n in FIFO:
            if n not in out:
                out[n] = getattr(q, n)
        return out

    def queued(self):
        return (self.net.ul.q.cap >= 0).sum(-1)

    def olla(self):
        return self.net.ul.olla

    def pf_served(self):
        return self.net.ul.avg * self.n_ul

    def pf_reset_served(self):
        from .mac import MacLink
        return MacLink.STATE["avg"][2] * self.n_ul

    def clear(self, mask):
        self.net.ul.reset(mask)

    def load(self, mask, src, now, ctx):
        """Rows where mask: fresh MAC (reset values, no HARQ in flight) with BSR = queued air bytes, CSI = the gain
        of the new fading state, PF average = delivered bytes per UL data slot (EWMA, floor 1 B), OLLA = ctx["olla"],
        fading = ctx["h0"] (stationary draw); the queue laid out from stream offset 0; env clock now [rows] via
        epoch."""
        net, ul, eng = self.net, self.net.ul, self.eng
        q = ul.q
        ul.reset(mask)
        valid = src["cap"] >= 0
        size, air = self._sizes(src["cls"])
        ab = torch.where(valid, (src["rem"] * air / size).round().long(), torch.zeros_like(src["cap"]))
        end = ab.cumsum(-1)
        start = end - ab
        epoch = eng.T - now
        m3 = mask.view(-1, 1, 1)
        q.cap = torch.where(m3, torch.where(valid, src["cap"] + epoch[:, None, None], src["cap"]), q.cap)
        q.start = torch.where(m3, start, q.start)
        q.end = torch.where(m3, end, q.end)
        q.enq = torch.where(mask[:, None], end[..., -1], q.enq)
        for n in ("cls", "det", "hid", "f_nact", "f_snr", "f_own"):
            setattr(q, n, torch.where(m3, src[n].to(getattr(q, n).dtype), getattr(q, n)))
        net.h = _where_rows(mask, ctx["h0"].to(net.h.dtype), net.h)
        ul.bsr = torch.where(mask[:, None], q.enq, ul.bsr)
        ul.csi = _where_rows(mask, net._gain(), ul.csi)
        ul.avg = torch.where(mask[:, None], (ctx["served"] / self.n_ul).clamp(min=1.0), ul.avg)
        ul.olla = torch.where(mask[:, None], ctx["olla"].to(ul.olla.dtype), ul.olla)
        eng.epoch = torch.where(mask, epoch, eng.epoch)
        eng._uniform_epoch = None

    def submit(self, t, req, snr):
        return self.eng.submit(None, req, snr)

    def advance(self, t, snr, cur_hid):
        return self.eng.step(None, snr)


def _patch_rng_rows(rng):
    """Reset draws of a subbatch instance keyed by the env its slot holds (rng.env[slot]), not by the slot index."""
    def rows(ids):
        if ids is None:
            return rng.env, rng.episode, rng._zero
        return rng.env.index_select(0, ids), rng.episode.index_select(0, ids), torch.zeros_like(ids)
    rng._rows = rows


class AdaptiveEngine(StateDictMixin):
    """Per-env routing of control steps between a cheap and an expensive level (module docstring).

    Every piece of routing state is a persistent buffer updated in place, so graph mode can capture submit and step
    (each with both levels and the switching logic) in one CUDA graph each; see FidelityConfig.graph."""

    level = "adaptive"
    WARM = 2                                   # eager calls before a region is captured (compiles, caches)

    def __init__(self, E, R, device="cpu", config: NRConfig | None = None, fidelity: FidelityConfig | None = None,
                 *, seed=None):
        from .engine import FAST_BACKENDS, make_engine
        cfg = config if config is not None else NRConfig()
        fid = fidelity if fidelity is not None else (getattr(cfg, "fidelity", None) or FidelityConfig())
        # the wrappers (edge, energy, background) apply to the adaptive engine as a whole (make_adaptive), never to
        # the levels: _Level.advance calls the level's own step, which would bypass a wrapper's accounting
        base = cfg.with_(fidelity=None, edge=None, background=None, energy=None)
        if base.rng != "engine":
            raise ValueError("AdaptiveEngine needs NRConfig.rng='engine' (the default): the handoff keys both levels' "
                             "draws by env, episode and call count")
        if fid.expensive == "L2-legacy" and not base.is_legacy_cell():
            raise ValueError("the expensive level L2-legacy must be the single legacy cell (NetSlot); multi-cell "
                             "NetSlotMC has no handoff rule yet")
        if fid.expensive == "L2" and (base.n_cells != 1 or base.dl or base.traffic):
            raise ValueError("the expensive level L2 is supported with one cell, uplink only and no traffic models "
                             "(the handoff moves the uplink FIFO)")
        self.E, self.R, self.dev = E, R, torch.device(device)
        self.fid, self._config = fid, cfg
        self.seed = _ns.resolve_seed(base.seed if seed is None else seed)
        self.cheap = make_engine(fid.cheap, E, R, self.dev, base, fid.cheap_backend, params=fid.cheap_params, strict=None,
                                 seed=self.seed)
        M = fid.budget_rows(E)
        self.subbatch = M is not None
        self.M = M = E if M is None else M
        self.exp = make_engine(fid.expensive, M, R, self.dev, base, fid.expensive_backend, strict=None,
                               params=fid.expensive_params, seed=self.seed)
        self.lc = _Level(self.cheap, fid.cheap)
        self.lx = (_NRLevel if fid.expensive == "L2" else _Level)(self.exp, fid.expensive)
        self.K, self.F = self.cheap.K, self.cheap.F
        self.sizes = self.cheap.sizes
        self._hrng = CounterRNG(self.seed, E, self.dev)            # handoff draws, keyed like the engines' draws
        d = self.dev
        self.arE = torch.arange(E, device=d)
        self.arM = torch.arange(M, device=d)
        if self.subbatch:
            if self.lx.has_rng:
                _patch_rng_rows(self.exp.rng)
            self.slot_env = torch.full((M,), -1, dtype=torch.long, device=d)
            self.env_slot = torch.full((E,), -1, dtype=torch.long, device=d)
        else:
            self.slot_env, self.env_slot = self.arE.clone(), self.arE.clone()
        self.active = torch.zeros(E, dtype=torch.bool, device=d)
        self.dwell = torch.zeros(E, dtype=torch.long, device=d)
        self.ind = torch.zeros(E, device=d)
        self.offered = torch.zeros(E, device=d)
        self._sub_bytes = torch.zeros(E, device=d)
        self.served = torch.zeros(E, R, device=d)                   # PF-average estimate, bytes per step
        self.served.fill_(self.lx.pf_reset_served())
        self._served_a = 1.0 - (1.0 - 1.0 / _ns.PF_T) ** self.K
        self._last_snr = torch.zeros(E, R, device=d)
        self._nsteps = 0                                            # steps taken (host counter: decision period)
        # OLLA on entering the expensive level: the robot's own offset when it last left it in this episode (NaN =
        # none), else the mean of its env's remembered offsets, else the level's prior (env-local: no coupling)
        self.olla_mem = torch.full((E, R), math.nan, device=d)
        self.olla_prior = OLLA_PRIOR.get(fid.expensive, 0.0) if fid.olla_prior_db is None else float(fid.olla_prior_db)
        g = torch.Generator().manual_seed(self.seed)
        rank = torch.empty(E, dtype=torch.long)
        rank[torch.randperm(E, generator=g)] = torch.arange(E)
        self._perm_rank = rank.to(d)                                # static subset = the envs of rank < n
        self.mode = fid.mode
        self.static_mask = self._fraction_mask(fid.fraction if fid.mode == "static" else
                                               (1.0 if fid.mode == "expensive" else 0.0))
        self.stats = {k: torch.zeros((), dtype=torch.long, device=d)
                      for k in ("up", "down", "denied", "steps_cheap", "steps_expensive")}
        fast = (fid.cheap_backend in FAST_BACKENDS and fid.expensive_backend in FAST_BACKENDS
                and fid.expensive != "L2" and d.type == "cuda")
        self.graph = fast if fid.graph == "auto" else bool(fid.graph)
        if self.graph and not fast:
            raise ValueError("graph mode needs a CUDA device and the eager / graph / compile / triton backends of both "
                             "levels (not the NR engine L2)")
        if self.graph:
            for eng in (self.cheap, self.exp):     # regions run inside this engine's graphs, not the level's own
                eng._run = (lambda e: (lambda which: (e._add_region if which == "add" else e._step_region)()))(eng)
        self._graphs, self._calls, self._gin, self._gout, self._pool = {}, {}, {}, {}, None
        self._reset_radio = False
        self._assign_reset(torch.ones(E, dtype=torch.bool, device=d), fresh=True)

    # ------------------------------------------------------------------ properties / passthroughs
    @property
    def config(self):
        return self._config

    @property
    def clock(self):
        return self.cheap.clock

    def output_schema(self):
        """{key: {"shape", "dtype", "unit", "doc", "when"}} of the keys step() returns (core/schema.py)."""
        from .schema import schema
        return schema("base", "fidelity")

    def queued(self):
        qx = self.lx.queued()
        if self.subbatch:
            qx = qx.index_select(0, self.env_slot.clamp(min=0))
        return torch.where(self.active[:, None], qx, self.lc.queued())

    def attach_radio(self, radio):
        """Radio for step(t, poses): the cheap instance turns poses into SNR for both levels."""
        self.cheap.attach_radio(radio)

    def fidelity_stats(self):
        """Cumulative counts: switches up / down, requests denied by the budget, env-steps on each level."""
        return {k: int(v) for k, v in self.stats.items()}

    # ------------------------------------------------------------------ assignment
    def _fraction_mask(self, f):
        n = int(round(f * self.E))
        return self._perm_rank < n

    def _invalidate(self):
        """The captured step and reset bake in the mode (switching, level at reset): capture again after a change."""
        for name in ("step", "step_switch", "reset"):
            self._graphs.pop(name, None)
            self._calls[name] = 0

    def set_fraction(self, f):
        """Curriculum hook: static mode with a share f of the envs on the expensive level (a nested random subset,
        so raising f only adds envs). The switch happens now, between two control steps, through the handoff."""
        if self.mode != "static":
            self._invalidate()
        self.mode = "static"
        self.static_mask.copy_(self._fraction_mask(float(f)))
        self._transition(force=True)

    def set_assignment(self, mask):
        """Static mode with an explicit assignment: envs where mask [E] is True run on the expensive level. The switch
        happens now, through the handoff (a custom mix, or forced switching in experiments)."""
        if self.mode != "static":
            self._invalidate()
        self.mode = "static"
        self.static_mask.copy_(torch.as_tensor(mask, dtype=torch.bool, device=self.dev).reshape(self.E))
        self._transition(force=True)

    def set_mode(self, mode):
        """"cheap" / "expensive" (all envs), "load" (thresholds of the config) or "static" (current fraction)."""
        assert mode in MODES
        if mode in ("cheap", "expensive"):
            self.static_mask.copy_(self._fraction_mask(1.0 if mode == "expensive" else 0.0))
        if (mode == "load") != (self.mode == "load"):
            self._invalidate()
        self.mode = mode
        self._transition(force=True)

    def _want(self, ignore_dwell=False):
        """Target level per env (True = expensive) from the mode and, for "load", indicator, hysteresis, dwell."""
        if self.mode != "load":
            return self.static_mask.clone()
        fid = self.fid
        can = self.dwell >= (0 if ignore_dwell else fid.min_dwell_steps)
        up = (self.ind >= fid.up_threshold) & can
        down = (self.ind < fid.down) & can
        return torch.where(self.active, ~down, up)

    def _want_reset(self):
        if self.mode != "load":
            return self.static_mask.clone()
        return torch.full((self.E,), 0.0 >= self.fid.up_threshold, dtype=torch.bool, device=self.dev)

    def _allocate(self, req, prio):
        """Free slots to requesting envs, highest prio first (ties: lower env id). Returns grant [E], slot [E]."""
        E, M = self.E, self.M
        free = self.slot_env < 0
        nfree = free.sum()
        key = torch.where(req, -prio.double(), torch.full_like(prio, INF, dtype=torch.float64))
        order = torch.argsort(key, stable=True)
        rank = torch.empty_like(order).scatter_(0, order, self.arE)
        grant = req & (rank < nfree)
        # home slot e mod M first (identity placement when M = E): among granted envs whose home is free, the
        # best-ranked one takes it; the other granted envs take the remaining free slots in rank order
        home = self.arE % M
        cand = grant & free.gather(0, home)
        best = torch.full((M,), E, dtype=torch.long, device=self.dev).scatter_reduce(
            0, home, torch.where(cand, rank, torch.full_like(rank, E)), reduce="amin")
        at_home = cand & (rank == best.gather(0, home))
        taken = torch.zeros(M, dtype=torch.long, device=self.dev).scatter_add_(0, home, at_home.long()) > 0
        rest = grant & ~at_home
        free2 = free & ~taken
        rest_by_rank = torch.zeros(E, dtype=torch.long, device=self.dev).scatter_(0, rank, rest.long())
        r2 = rest_by_rank.cumsum(0).gather(0, rank) - 1
        frank = free2.long().cumsum(0) - 1
        kth = torch.full((M + 1,), M - 1, dtype=torch.long, device=self.dev)
        kth.scatter_(0, torch.where(free2, frank, torch.full_like(frank, M)), self.arM)
        slot = torch.where(at_home, home, kth[:M].gather(0, r2.clamp(0, M - 1)))
        return grant, slot

    def _set_slots(self, grant, slot):
        """Record grants in the slot tables; returns the mask [M] of newly filled slots."""
        M = self.M
        idx = torch.where(grant, slot, torch.full_like(slot, M))
        ext = torch.cat([self.slot_env, self.slot_env.new_full((1,), -1)])
        ext.scatter_(0, idx, self.arE)
        filled = torch.zeros(M + 1, dtype=torch.bool, device=self.dev).scatter_(0, idx, grant)[:M]
        self.slot_env.copy_(torch.where(filled, ext[:M], self.slot_env))
        self.env_slot.copy_(torch.where(grant, slot, self.env_slot))
        return filled

    def _free_slots(self, rel):
        """Release the slots of envs rel [E]; returns the mask [M] of freed slots."""
        M = self.M
        idx = torch.where(rel, self.env_slot, torch.full_like(self.env_slot, M))
        freed = torch.zeros(M + 1, dtype=torch.bool, device=self.dev).scatter_(0, idx, rel)[:M]
        self.slot_env.copy_(torch.where(freed, torch.full_like(self.slot_env, -1), self.slot_env))
        self.env_slot.copy_(torch.where(rel, torch.full_like(self.env_slot, -1), self.env_slot))
        return freed

    def _slot_keys(self, filled, env_m, reset):
        """Point the RNG rows of newly filled slots at their env (subbatch): env id, episode, call counters."""
        rng, crng = self.exp.rng, self.cheap.rng
        ep = crng.episode.index_select(0, env_m)
        if self.lx.triton:          # the kernel keys by row: exact at identity placement, hashed episode otherwise
            h = combine(mix32(env_m & 0xFFFFFFFF), ep & 0xFFFFFFFF) | (1 << 31)
            ep = torch.where(env_m == self.arM, ep, h)
        rng.env.copy_(torch.where(filled, env_m, rng.env))
        rng.episode.copy_(torch.where(filled, ep, rng.episode))
        for ch in (SUBMIT, STEP):
            c = torch.zeros_like(env_m) if reset else crng.ctr[ch].index_select(0, env_m)
            rng.ctr[ch].copy_(torch.where(filled, c, rng.ctr[ch]))

    def _assign_reset(self, m, fresh=False):
        """Level of the envs m [E] after their reset. fresh: at construction, where every engine row is freshly reset
        (the NR engine's rows are kept as built; the prototype rows are re-keyed to the env they hold)."""
        want = self._want_reset() & m
        if not self.subbatch:
            self.active.copy_(torch.where(m, want, self.active))
            return
        grant, slot = self._allocate(want, torch.zeros(self.E, device=self.dev))
        filled = self._set_slots(grant, slot)
        env_m = self.slot_env.clamp(min=0)
        if self.lx.has_rng:
            self._slot_keys(filled, env_m, reset=True)
            self.lx.reset_rows(filled)                              # = exp.reset(slots), without a host sync
        elif not fresh:
            ids = filled.nonzero(as_tuple=True)[0]                  # NR engine: its own partial reset
            if ids.numel():
                self.exp.reset(ids)
        self.active.copy_(torch.where(m, grant, self.active))
        self.stats["denied"] += (want & ~grant).sum()

    # ------------------------------------------------------------------ contract API
    def reset(self, env_ids=None):
        """Partial reset of env_ids (None = all) on both levels; each env restarts on the level its mode assigns at
        zero load, with that level's exact reset state. Other envs are bitwise unaffected."""
        ids = _ns.env_index(env_ids, self.E, self.dev)
        if ids is not None and ids.numel() == 0:
            return
        m = torch.ones(self.E, dtype=torch.bool, device=self.dev) if ids is None else \
            torch.zeros(self.E, dtype=torch.bool, device=self.dev).index_fill_(0, ids, True)
        if self.graph:                          # one captured, masked reset of both levels and the routing state
            radio = self.cheap.radio is not None
            if self._reset_radio != radio:
                self._graphs.pop("reset", None)
                self._calls["reset"] = 0
                self._reset_radio = radio
            self._replay("reset", self._reset_body, (m,))
            return
        self.cheap.reset(ids)
        if self.subbatch:
            rel = m & (self.env_slot >= 0)
            freed = self._free_slots(rel)
            self.lx.clear(freed)
        else:
            self.exp.reset(ids)
        self._reset_tables(m)

    def _reset_tables(self, m):
        for name in ("dwell", "ind", "offered", "_sub_bytes", "_last_snr"):
            t = getattr(self, name)
            t.copy_(_where_rows(m, torch.zeros_like(t), t))
        self.served.copy_(_where_rows(m, torch.full_like(self.served, self.lx.pf_reset_served()), self.served))
        self.olla_mem.copy_(_where_rows(m, torch.full_like(self.olla_mem, math.nan), self.olla_mem))
        self.active.copy_(self.active & ~m)
        self._assign_reset(m)

    def _reset_body(self, m):
        """reset() of the envs m [E] with fixed shapes and no host sync (graph mode): the levels' masked resets."""
        self.lc.masked_reset(m)
        if self.subbatch:
            rel = m & (self.env_slot >= 0)
            freed = self._free_slots(rel)
            self.lx.clear(freed)
        else:
            self.lx.masked_reset(m)
        self._reset_tables(m)
        return {}

    def submit(self, t, requests, snr_db=None):
        """Enqueue new messages; each env's go to the level it is on. Returns accepted [E,R]."""
        req = requests if isinstance(requests, _ns.Requests) else _ns.Requests(send=requests)
        send = req.send
        det = req.det if req.det is not None else torch.zeros_like(send, dtype=torch.bool)
        hid = req.hid if req.hid is not None else torch.zeros(self.E, dtype=torch.long, device=self.dev)
        snr = self._last_snr if snr_db is None else snr_db
        if self.graph and t is None:
            return self._replay("submit", self._submit_body, (send, det, hid, snr))["accepted"].clone()
        return self._submit_body(send, det, hid, snr, t)["accepted"]

    def _submit_body(self, send, det, hid, snr, t=None):
        tv = self.cheap._tvec(t)
        act = self.active[:, None]
        acc_c = self.lc.submit(tv, _ns.Requests(torch.where(act, torch.zeros_like(send), send), det, hid), snr)
        if self.subbatch:
            em, ok = self.slot_env.clamp(min=0), (self.slot_env >= 0)[:, None]
            sx = torch.where(ok, send.index_select(0, em), torch.zeros_like(send[: self.M]))
            acc_x = self.lx.submit(tv.index_select(0, em), _ns.Requests(sx, det.index_select(0, em),
                                                                      hid.index_select(0, em)),
                                   snr.index_select(0, em))
            acc_x = acc_x.index_select(0, self.env_slot.clamp(min=0))
        else:
            acc_x = self.lx.submit(tv, _ns.Requests(torch.where(act, send, torch.zeros_like(send)), det, hid), snr)
        nb = torch.where(send > 0, self.sizes[(send - 1).clamp(min=0)], torch.zeros_like(send, dtype=torch.float32))
        self._sub_bytes.add_(nb.sum(-1))
        return {"accepted": torch.where(act, acc_x, acc_c)}

    def add_frames(self, t, send, det, hid, snr_db):
        """Legacy wrapper (NetSlot API)."""
        self.submit(t, _ns.Requests(send, det, hid), snr_db)

    def step(self, t, x, cur_hid=None):
        """Advance [t, t+1) on both levels and merge per env. x: SNR [E,R] dB or poses [E,R,2|3] (through the cheap
        instance's radio, shared by both levels). Legacy form step(t, x, cur_hid) -> (newest, det_env)."""
        snr = self.cheap._snr_from(x)
        self._nsteps += 1
        switch = self.mode == "load" and self._nsteps % self.fid.decision_period == 0
        if cur_hid is not None:
            out = self._step_body(snr, t, cur_hid, switch)
            return out["newest"], out["det_env"]
        if self.graph and t is None:          # two captured steps: with and without the switching logic
            name = "step_switch" if switch else "step"
            out = self._replay(name, lambda s: self._step_body(s, None, None, switch), (snr,))
            return {k: v.clone() for k, v in out.items()}
        return self._step_body(snr, t, None, switch)

    def _step_body(self, snr, t=None, cur_hid=None, switch=False):
        tv = self.cheap._tvec(t)
        ch = self.cheap._last_hid if cur_hid is None else cur_hid
        rem_pre = self.cheap.rem.clone() if (self.lx.h_shape is not None and not self.lc.delay) else None
        oc = self.lc.advance(tv, snr, ch)
        if rem_pre is not None:                 # bytes the cheap level served this step (not those that timed out)
            self._served_step = ((rem_pre * ~oc["timed_out"]).sum(-1) - self.cheap.rem.sum(-1)).clamp(min=0)
        if self.subbatch:
            em = self.slot_env.clamp(min=0)
            ox = self.lx.advance(tv.index_select(0, em), snr.index_select(0, em), ch.index_select(0, em))
            es = self.env_slot.clamp(min=0)
            ox = {k: v.index_select(0, es) for k, v in ox.items()}
        else:
            ox = self.lx.advance(tv, snr, ch)
        act = self.active.clone()
        out = {k: _where_rows(act, ox[k], v) if k in ox else v for k, v in oc.items()}
        for k, v in ox.items():                 # keys only the expensive level reports (NR: dropped, serving_cell)
            if k not in out:
                out[k] = _where_rows(act, v, torch.zeros_like(v))
        self._last_snr.copy_(snr)
        self.stats["steps_expensive"] += act.sum()
        self.stats["steps_cheap"] += (~act).sum()
        self._observe(out)
        out["fidelity"] = act.long()
        out["fidelity_indicator"] = self.ind.clone()
        self.dwell.add_(1)
        if switch:
            self._transition()
        return out

    # ------------------------------------------------------------------ graph mode
    def _replay(self, name, body, inputs):
        """Run body on static copies of inputs: eagerly for the first WARM calls, then captured once and replayed.
        Returns the static outputs (valid until the next call)."""
        bufs = self._gin.get(name)
        if bufs is None or any(b.shape != x.shape for b, x in zip(bufs, inputs)):
            bufs = self._gin[name] = tuple(x.clone() for x in inputs)
            self._graphs.pop(name, None)
            self._calls[name] = 0
        else:
            for b, x in zip(bufs, inputs):
                b.copy_(x)
        g = self._graphs.get(name)
        if g is None:
            n = self._calls.get(name, 0)
            if n < self.WARM:
                self._calls[name] = n + 1
                return body(*bufs)
            g = torch.cuda.CUDAGraph()
            if self._pool is None:
                self._pool = torch.cuda.graph_pool_handle()
            with cuda_graph.graph(g, pool=self._pool):
                self._gout[name] = body(*bufs)
            self._graphs[name] = g
        g.replay()
        return self._gout[name]

    # ------------------------------------------------------------------ indicators and switching
    def _observe(self, out):
        fid, R = self.fid, self.R
        a = fid.offered_ewma
        self.offered.copy_(a * self.offered + (1 - a) * self._sub_bytes / R)
        self._sub_bytes.zero_()
        if self.lx.h_shape is not None:         # the PF average of envs on the cheap level, continued (see load())
            if self.lc.delay:                   # delay levels serve a message at its delivery
                cls = out["cls"]
                db = torch.where(out["delivered"], self.sizes[(cls - 1).clamp(min=0)], torch.zeros_like(out["delay"]))
                sv = db.sum(-1)
            else:
                sv = self._served_step
            self.served.copy_((1 - self._served_a) * self.served + self._served_a * sv)
        if fid.indicator == "backlog":
            self.ind.copy_(out["queue_bytes"].sum(-1) / R)
        elif fid.indicator == "contention":
            self.ind.copy_((out["queue_len"] > 0).float().mean(-1))
        else:
            self.ind.copy_(self.offered)

    def _handoff_uv(self, rows_env):
        ep = self.cheap.rng.episode.index_select(0, rows_env)
        ctr = self.cheap.clock.index_select(0, rows_env)
        tail = (self.R, self.F)
        u = self._hrng._draw(rows_env, ep, ctr, HANDOFF, 1, tail, False)
        v = self._hrng._draw(rows_env, ep, ctr, HANDOFF, 2, tail, False)
        return u, v

    def _transition(self, force=False):
        """Move envs whose target level changed. Releases first (so their slots can be reused), then grants."""
        want = self._want(ignore_dwell=force)               # curriculum / set_mode: no dwell
        rel = self.active & ~want
        req = want & ~self.active
        now = self.cheap.clock
        olla = self.lx.olla()
        # ---- release: expensive rows -> cheap rows
        if self.subbatch:
            es = self.env_slot.clamp(min=0)
            src = {n: v.index_select(0, es) for n, v in self.lx.fifo().items()}
            olla_e = None if olla is None else olla.index_select(0, es)
        else:
            src = self.lx.fifo()
            olla_e = olla
        if olla_e is not None:
            self.olla_mem.copy_(_where_rows(rel, olla_e, self.olla_mem))
        pf = self.lx.pf_served()
        if pf is not None:                      # the PF-average estimate continues from the actual PF average
            pf_e = pf.index_select(0, es) if self.subbatch else pf
            self.served.copy_(_where_rows(rel, pf_e, self.served))
        ctx = {}
        if self.lc.delay:
            ctx["u"], ctx["v"] = self._handoff_uv(self.arE)
        self.lc.load(rel, src, now, ctx)
        freed = self._free_slots(rel) if self.subbatch else rel
        self.lx.clear(freed)
        # ---- grant: cheap rows -> expensive rows
        if self.subbatch:
            grant, slot = self._allocate(req, self.ind)
            filled = self._set_slots(grant, slot)
            env_m = self.slot_env.clamp(min=0)
            src = {n: v.index_select(0, env_m) for n, v in self.lc.fifo().items()}
        else:
            grant = filled = req
            env_m = self.arE
            src = self.lc.fifo()
        now_m = now.index_select(0, env_m)
        ctx = {}
        if self.lx.h_shape is not None:
            ep = self.cheap.rng.episode.index_select(0, env_m)
            ctx["h0"] = self._hrng._draw(env_m, ep, now_m, HANDOFF, 0, self.lx.h_shape, True) / math.sqrt(2)
            ctx["served"] = self.served.index_select(0, env_m)
            mem = self.olla_mem.index_select(0, env_m)
            known = ~torch.isnan(mem)
            env_mean = torch.where(known, mem, torch.zeros_like(mem)).sum(-1) / known.sum(-1).clamp(min=1)
            env_mean = torch.where(known.any(-1), env_mean, torch.full_like(env_mean, self.olla_prior))
            ctx["olla"] = torch.where(known, mem, env_mean[:, None].expand_as(mem))
        if self.lx.delay:
            ctx["u"], ctx["v"] = self._handoff_uv(env_m)
        self.lx.load(filled, src, now_m, ctx)
        if self.subbatch and self.lx.has_rng:
            self._slot_keys(filled, env_m, reset=False)
        self.lc.clear(grant)
        self.active.copy_((self.active & ~rel) | grant)
        self.dwell.copy_(torch.where(rel | grant, torch.zeros_like(self.dwell), self.dwell))
        self.stats["up"] += grant.sum()
        self.stats["down"] += rel.sum()
        self.stats["denied"] += (req & ~grant).sum()


class FidelityCurriculum:
    """A schedule over training iterations, as a callback: call it (or on_iteration) once per iteration.

        cur = FidelityCurriculum(net, [(0, 0.0), (k, 1.0)])    # cheap level for iterations < k, then expensive
        for it in range(n_iter):
            cur(it)                                            # switches between two control steps (handoff)
            ...collect a rollout, update the policy...

    schedule: list of (first iteration, share of envs on the expensive level), or a callable it -> share. The share
    changes only when the schedule value changes; each change is one set_fraction(f) on the engine (nested random
    subsets, so a rising share only adds envs). With the subbatch layout the share is capped by the budget."""

    def __init__(self, engine, schedule):
        self.engine = engine
        self.schedule = schedule
        self.current = None

    @classmethod
    def step_at(cls, engine, k):
        """Cheap level for iterations 0 .. k-1, the expensive level from iteration k on."""
        return cls(engine, [(0, 0.0), (int(k), 1.0)])

    def fraction(self, it):
        if callable(self.schedule):
            return float(self.schedule(it))
        f = 0.0
        for start, v in sorted(self.schedule):
            if it >= start:
                f = float(v)
        return f

    def on_iteration(self, it):
        f = self.fraction(it)
        if f != self.current:
            eng = getattr(self.engine, "engine", self.engine)      # EdgeLoop(AdaptiveEngine) too
            eng.set_fraction(f)
            self.current = f
        return f

    __call__ = on_iteration


__all__ = ["FidelityConfig", "AdaptiveEngine", "FidelityCurriculum", "make_adaptive"]
