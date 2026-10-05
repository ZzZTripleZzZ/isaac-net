"""Traffic: what robots hand to the network each control step.

Two sources feed the uplink queues, and they can be mixed freely:

1. The policy. Requests(send, det=None, hid=None): send [E,R] long, 0 = nothing, c >= 1 = one message of traffic
   class c (size NRConfig.msg_sizes[c-1] bytes); det [E,R] bool marks a message that carries the env's current
   task event (for example a hazard detection), hid [E] long is that event's id. Every engine's submit() takes it.
   These messages arrive at the start of the control step (arrival offset 0).

2. Traffic models, chosen in the config and run inside the engine step (level "L2"):

       NRConfig(traffic=[TrafficModel.periodic(200, period_ms=10).on(robots=[0, 1]),
                         TrafficModel.video(fps=30, mean_frame_bytes=None, gop=(40_000, 8_000, 15)).on(robots=[2]),
                         TrafficModel.bursty(1400, rate_hz=50, burst_size=4, on_off=(0.5, 2.0)),
                         TrafficModel.event(4000, trigger="alarm", det=True),
                         TrafficModel.policy()])

   Each model applies to the robots given with .on(robots=...) (robot indices inside an env, default all), so a
   robot class or a single robot can have its own generator, and several models can feed the same robot. A model
   can emit several messages per control step: every message carries an arrival offset in slots inside the step,
   the MAC cannot send its bytes before that slot, and its delay is measured from that slot.

Direction. A model feeds the uplink by default; direction="dl" (or model.downlink()) makes it a downlink source: its
messages go into the gNB-side DL queue of the robot (the NR engine's DL path, config dl=True), with the same arrival
offsets, sizes, tags and deadlines, and the DL scheduler serves them. UL and DL models can be mixed in one list.
DL models draw from their own generator (TrafficGen(direction="dl")), so adding a DL model never changes the arrivals
of the UL models of the same config.

Generated messages are tensors with fixed shapes [E, R, M] (M = the sum of the models' max_msgs_per_step), so the
generator has no data-dependent shapes and no host syncs. Draws come from a generator that the engine owns and
seeds from its own seed, so policy sampling does not shift the traffic and the traffic does not shift the
network's draws. Step draws and reset draws use two generators, so a partial reset of some envs leaves the draws
of every other env unchanged.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Callable, Optional, Union

import torch

from .proto.netsim import Requests  # noqa: F401

__all__ = ["Requests", "TrafficModel", "TrafficGen", "Arrivals", "normalize_traffic"]

KINDS = ("periodic", "bursty", "video", "event", "policy")
DIRECTIONS = ("ul", "dl")


@dataclass(frozen=True)
class TrafficModel:
    """One traffic generator. Build it with the constructors below, never directly.

    Common optional extras (every constructor): tag (int, carried by the queue and reported per message; default
    = 1 + the model's position in NRConfig.traffic, 0 is the policy's), priority (int, carried and reported; the
    MAC is FIFO per robot and does not use it), deadline_ms (reported as `deadline_miss`; inf = none),
    max_msgs_per_step (the fixed number of arrivals per robot per step the model reserves; more arrivals than
    that are deferred to the next step, never dropped, and counted in TrafficGen.deferred), direction ("ul", the
    default, or "dl": the messages go to the robot's downlink queue; see downlink()).
    """

    kind: str
    size_bytes: float = 0.0
    period_ms: float = 0.0
    jitter_ms: float = 0.0
    phase: str = "random"
    rate_hz: float = 0.0
    burst_size: int = 1
    on_off: tuple = (1.0, 0.0)
    gop: Optional[tuple] = None
    trigger: Union[str, Callable, None] = None
    det: bool = False
    robots: Optional[tuple] = None
    tag: Optional[int] = None
    priority: int = 0
    deadline_ms: float = math.inf
    max_msgs_per_step: Optional[int] = None
    direction: str = "ul"

    # ------------------------------------------------------------------ constructors
    @staticmethod
    def periodic(size_bytes, period_ms, jitter_ms=0.0, phase="random", **kw):
        """A message of size_bytes every period_ms (telemetry, control loops); the period may be shorter than the
        control step. phase "random": each robot's first message at a uniform time in [0, period) after a reset;
        "aligned": at 0 for every robot. jitter_ms: each message is late by a uniform [0, jitter_ms) draw around
        its nominal time (no drift; jitter_ms < period_ms keeps the order)."""
        assert period_ms > 0 and 0 <= jitter_ms < period_ms and size_bytes > 0, "need size > 0, 0 <= jitter < period"
        assert phase in ("random", "aligned")
        return TrafficModel("periodic", size_bytes=float(size_bytes), period_ms=float(period_ms),
                            jitter_ms=float(jitter_ms), phase=phase, **kw)

    @staticmethod
    def bursty(size_bytes, rate_hz, burst_size, on_off=(1.0, 1.0), **kw):
        """Markov on/off source: ON and OFF periods are exponential with means on_off = (mean_on_s, mean_off_s);
        while ON, bursts arrive as a Poisson process of rate_hz, and each burst is burst_size messages of
        size_bytes arriving together. mean_off_s = 0 gives an always-on Poisson burst source."""
        assert size_bytes > 0 and rate_hz > 0 and burst_size >= 1 and on_off[0] > 0 and on_off[1] >= 0
        return TrafficModel("bursty", size_bytes=float(size_bytes), rate_hz=float(rate_hz),
                            burst_size=int(burst_size), on_off=(float(on_off[0]), float(on_off[1])), **kw)

    @staticmethod
    def video(fps, mean_frame_bytes=None, gop=None, phase="random", jitter_ms=0.0, **kw):
        """One frame every 1000 / fps ms. gop = (I_bytes, P_bytes, gop_len): frame k of a robot (k counted from
        its reset) is an I frame of I_bytes when k % gop_len == 0, else a P frame of P_bytes. Without gop every
        frame is mean_frame_bytes. With both, the I and P sizes are scaled so that their mean is mean_frame_bytes
        (gop then only fixes the I:P ratio)."""
        assert fps > 0 and (mean_frame_bytes is not None or gop is not None)
        if gop is not None:
            i_b, p_b, n = float(gop[0]), float(gop[1]), int(gop[2])
            assert i_b > 0 and p_b > 0 and n >= 1
            if mean_frame_bytes is not None:
                s = float(mean_frame_bytes) / ((i_b + (n - 1) * p_b) / n)
                i_b, p_b = i_b * s, p_b * s
            gop = (i_b, p_b, n)
            size = (i_b + (n - 1) * p_b) / n
        else:
            size = float(mean_frame_bytes)
        assert size > 0
        period = 1000.0 / fps
        assert 0 <= jitter_ms < period and phase in ("random", "aligned")
        return TrafficModel("video", size_bytes=size, period_ms=period, jitter_ms=float(jitter_ms), phase=phase,
                            gop=gop, **kw)

    @staticmethod
    def event(size_bytes, trigger="event", det=False, **kw):
        """One message of size_bytes at the start of every step in which the robot's trigger is set. trigger: a
        name looked up in engine.step(..., triggers={name: mask}) (a mask [E,R] or [E] bool, or a tensor passed
        as triggers= directly, which then feeds every event model), or a callable f(clock [E]) -> mask. det=True
        marks the message as carrying the env's current task event (det / hid, like Requests.det)."""
        assert size_bytes > 0
        return TrafficModel("event", size_bytes=float(size_bytes), trigger=trigger, det=bool(det), **kw)

    @staticmethod
    def policy():
        """The policy's own messages through submit() (always on; listing it only documents the mix)."""
        return TrafficModel("policy")

    # ------------------------------------------------------------------ helpers
    def on(self, robots):
        """The same model restricted to these robot indices (int, sequence, or None = all robots)."""
        if robots is None:
            return replace(self, robots=None)
        if isinstance(robots, int):
            robots = (robots,)
        return replace(self, robots=tuple(int(r) for r in robots))

    def with_(self, **kw):
        return replace(self, **kw)

    def downlink(self):
        """The same model as a downlink source (direction="dl"): the gNB sends its messages to the robot."""
        return replace(self, direction="dl")

    @property
    def generates(self):
        return self.kind != "policy"

    def max_per_step(self, step_ms):
        """Arrivals per robot per control step the model reserves (fixed tensor width)."""
        if self.max_msgs_per_step is not None:
            return int(self.max_msgs_per_step)
        if self.kind in ("periodic", "video"):
            return int(math.floor((step_ms + self.jitter_ms) / self.period_ms)) + 1
        if self.kind == "bursty":
            return self.burst_size * self._bursts_per_step(step_ms)
        if self.kind == "event":
            return 1
        return 0

    def _bursts_per_step(self, step_ms):
        if self.max_msgs_per_step is not None:
            return max(1, int(self.max_msgs_per_step) // self.burst_size)
        lam = self.rate_hz * step_ms / 1000.0
        return int(math.ceil(lam + 4 * math.sqrt(lam))) + 1     # Poisson mean + 4 sigma: deferral is rare


def normalize_traffic(traffic):
    """None, a TrafficModel or a sequence of them -> None or a tuple of TrafficModel (validated)."""
    if traffic is None:
        return None
    if isinstance(traffic, TrafficModel):
        traffic = (traffic,)
    traffic = tuple(traffic)
    for m in traffic:
        if not isinstance(m, TrafficModel) or m.kind not in KINDS:
            raise TypeError(f"NRConfig.traffic takes TrafficModel objects (TrafficModel.periodic(...), ...), got {m!r}")
        if m.direction not in DIRECTIONS:
            raise ValueError(f"traffic model {m.kind}: direction must be 'ul' or 'dl', not {m.direction!r}")
        if m.direction == "dl" and m.kind == "policy":
            raise ValueError("policy() is the policy's uplink submit(); downlink messages from the policy go through "
                             "add_dl_frames(), not a DL policy() model")
        if m.direction == "dl" and m.det:
            raise ValueError(f"traffic model {m.kind}: det=True marks an uplink task event; a DL model cannot carry it")
    return traffic or None


def generates(traffic, direction=None):
    """True if the traffic tuple has a model that generates messages (anything but policy()), in `direction` ("ul" or
    "dl"; None = either)."""
    return traffic is not None and any(m.generates and (direction is None or m.direction == direction)
                                       for m in traffic)


@dataclass
class Arrivals:
    """Messages generated in one control step, [E, R, M], sorted by arrival slot per robot (invalid last)."""
    valid: torch.Tensor     # bool
    nbytes: torch.Tensor    # float32 application bytes
    slot: torch.Tensor      # long, arrival slot inside the step, 0 .. slots_per_step - 1
    tag: torch.Tensor       # long
    prio: torch.Tensor      # long
    dline: torch.Tensor     # float64 deadline in ms (inf = none)
    det: torch.Tensor       # bool


class _Stream:
    """Runtime state of one model: per-robot tensors [E, R], active only on the model's robots."""

    def __init__(self, model: TrafficModel, idx, E, R, device, step_ms, slot_ms, gen, rgen):
        self.m, self.E, self.R, self.dev = model, E, R, device
        self.step_ms, self.slot_ms, self.gen, self.rgen = step_ms, slot_ms, gen, rgen
        self.M = (model.burst_size * model._bursts_per_step(step_ms) if model.kind == "bursty"
                  else model.max_per_step(step_ms))
        self.tag = model.tag if model.tag is not None else idx + 1
        rob = torch.zeros(R, dtype=torch.bool, device=device)
        if model.robots is None:
            rob[:] = True
        else:
            bad = [r for r in model.robots if not 0 <= r < R]
            if bad:
                raise ValueError(f"traffic model {model.kind}: robot indices {bad} are outside 0..{R - 1}")
            rob[list(model.robots)] = True
        self.rob = rob[None, :].expand(E, R)
        z = lambda dt, v: torch.full((E, R), v, dtype=dt, device=device)
        if model.kind in ("periodic", "video"):
            self.nom, self.nxt, self.cnt = z(torch.float64, 0.0), z(torch.float64, 0.0), z(torch.long, 0)
            self.state = ("nom", "nxt", "cnt")
        elif model.kind == "bursty":
            self.on, self.t_tog, self.t_bur = z(torch.bool, False), z(torch.float64, 0.0), z(torch.float64, 0.0)
            self.state = ("on", "t_tog", "t_bur")
        else:
            self.state = ()
        self.deferred = torch.zeros((), dtype=torch.long, device=device)

    def _hold(self):
        return {n: getattr(self, n) for n in self.state}

    def _commit(self, held):
        """Write the new state into the original buffers (in place), so a captured CUDA graph that replays
        step() reads and writes the same memory every time."""
        for n, buf in held.items():
            buf.copy_(getattr(self, n))
            setattr(self, n, buf)

    def _u(self):
        return torch.rand(self.E, self.R, generator=self.gen, device=self.dev, dtype=torch.float64)

    def _exp(self, mean_ms):
        return -torch.log1p(-self._u()) * mean_ms

    def reset(self, m_e):
        """Redraw the state of the envs where m_e [E] is True (fixed shape)."""
        held, g = self._hold(), self.gen
        self.gen = self.rgen                    # reset draws: own stream, so a partial reset of some envs
        try:                                    # leaves the step draws of every other env unchanged
            self._reset(m_e)
        finally:
            self.gen = g
        self._commit(held)

    def _reset(self, m_e):
        md = self.m
        m = m_e[:, None]
        if md.kind in ("periodic", "video"):
            nom = (self._u() * md.period_ms) if md.phase == "random" else torch.zeros_like(self.nom)
            nxt = nom + self._u() * md.jitter_ms
            self.nom = torch.where(m, nom, self.nom)
            self.nxt = torch.where(m, nxt, self.nxt)
            self.cnt = torch.where(m, torch.zeros_like(self.cnt), self.cnt)
        elif md.kind == "bursty":
            on_ms, off_ms = md.on_off[0] * 1000.0, md.on_off[1] * 1000.0
            p_on = on_ms / (on_ms + off_ms)
            on = self._u() < p_on                                          # stationary start
            t_tog = torch.where(on, self._exp(on_ms), self._exp(off_ms))   # memoryless residual times
            t_bur = self._exp(1000.0 / md.rate_hz)
            if off_ms == 0:
                t_tog = torch.full_like(t_tog, math.inf)
            self.on = torch.where(m, on, self.on)
            self.t_tog = torch.where(m, t_tog, self.t_tog)
            self.t_bur = torch.where(m, t_bur, self.t_bur)

    def step(self, clock, trig):
        """-> (valid [E,R,M], nbytes, time_ms from step start), times < step_ms."""
        held = self._hold()
        out = self._step(clock, trig)
        self._commit(held)
        return out

    def _step(self, clock, trig):
        md, E, R, M, S = self.m, self.E, self.R, self.M, self.step_ms
        if md.kind in ("periodic", "video"):
            vs, bs, ts = [], [], []
            for _ in range(M):
                hit = (self.nxt < S) & self.rob
                if md.kind == "video" and md.gop is not None:
                    i_b, p_b, n = md.gop
                    size = torch.where(self.cnt % n == 0, torch.full_like(self.nxt, i_b), torch.full_like(self.nxt, p_b))
                else:
                    size = torch.full_like(self.nxt, md.size_bytes)
                vs.append(hit), bs.append(size), ts.append(self.nxt)
                self.cnt = self.cnt + hit.long()
                self.nom = torch.where(hit, self.nom + md.period_ms, self.nom)
                self.nxt = torch.where(hit, self.nom + self._u() * md.jitter_ms, self.nxt)
            self.deferred += ((self.nxt < S) & self.rob).sum()
            self.nom = self.nom - S
            self.nxt = self.nxt - S
            return torch.stack(vs, -1), torch.stack(bs, -1), torch.stack(ts, -1)
        if md.kind == "bursty":
            Kb = M // md.burst_size
            on_ms, off_ms = md.on_off[0] * 1000.0, md.on_off[1] * 1000.0
            iat = 1000.0 / md.rate_hz
            L = Kb + 2 * (int(math.ceil(S / max(min(on_ms, off_ms if off_ms > 0 else on_ms), 1e-9))) + 1)
            L = min(L, Kb + 64)
            cnt = torch.zeros(E, R, dtype=torch.long, device=self.dev)
            tb = torch.full((E, R, Kb), math.inf, dtype=torch.float64, device=self.dev)
            for _ in range(L):
                t_b = torch.where(self.on, self.t_bur, torch.full_like(self.t_bur, math.inf))
                burst = self.rob & self.on & (t_b <= self.t_tog) & (t_b < S) & (cnt < Kb)
                blocked = self.rob & self.on & (t_b <= self.t_tog) & (t_b < S) & (cnt >= Kb)
                tog = self.rob & ~burst & ~blocked & (self.t_tog < S) & (self.t_tog < t_b)
                tb = torch.where(onehot_(cnt, Kb) & burst[..., None], t_b[..., None], tb)
                cnt = cnt + burst.long()
                self.t_bur = torch.where(burst, self.t_bur + self._exp(iat), self.t_bur)
                now_on = ~self.on
                new_tog = self.t_tog + torch.where(now_on, self._exp(on_ms), self._exp(off_ms) if off_ms > 0
                                                   else torch.full_like(self.t_tog, math.inf))
                self.t_bur = torch.where(tog & now_on, self.t_tog + self._exp(iat), self.t_bur)
                self.t_tog = torch.where(tog, new_tog, self.t_tog)
                self.on = torch.where(tog, now_on, self.on)
            t_b = torch.where(self.on, self.t_bur, torch.full_like(self.t_bur, math.inf))
            self.deferred += (self.rob & (torch.minimum(t_b, self.t_tog) < S)).sum()
            self.t_bur = self.t_bur - S
            self.t_tog = self.t_tog - S
            valid = torch.isfinite(tb).repeat_interleave(md.burst_size, -1)
            t = tb.repeat_interleave(md.burst_size, -1)
            return valid, torch.full(valid.shape, md.size_bytes, dtype=torch.float32, device=self.dev), t
        if md.kind == "event":
            if callable(md.trigger):
                mask = md.trigger(clock)
            elif torch.is_tensor(trig):
                mask = trig
            elif isinstance(trig, dict) and md.trigger in trig:
                mask = trig[md.trigger]
            else:
                mask = None
            if mask is None:
                valid = torch.zeros(E, R, dtype=torch.bool, device=self.dev)
            else:
                mask = mask.to(self.dev, torch.bool)
                valid = (mask[:, None].expand(E, R) if mask.dim() == 1 else mask) & self.rob
            z = torch.zeros(E, R, 1, dtype=torch.float64, device=self.dev)
            return valid[..., None], torch.full((E, R, 1), md.size_bytes, dtype=torch.float32, device=self.dev), z
        raise AssertionError(md.kind)


def onehot_(idx, n):
    return idx[..., None] == torch.arange(n, device=idx.device)


class TrafficGen:
    """All generating models of one direction of a config for E envs x R robots. step() -> Arrivals [E, R, M] per
    control step. direction "ul" (default) or "dl": only the models of that direction get a stream; a model's default
    tag is still 1 + its position in the whole list."""

    def __init__(self, traffic, E, R, device, step_ms, slots_per_step, generator=None, seed=None, direction="ul"):
        self.E, self.R, self.dev = E, R, torch.device(device)
        self.N = int(slots_per_step)
        self.step_ms, self.slot_ms = float(step_ms), float(step_ms) / self.N
        seed = 0 if seed is None else int(seed)
        if generator is None:
            generator = torch.Generator(device=self.dev)
            generator.manual_seed(seed)
        self.gen = generator                       # step draws (register it when capturing step() in a graph)
        self.rgen = torch.Generator(device=self.dev)
        self.rgen.manual_seed((seed + 0x9E3779B97F4A7C15) % 2 ** 63)   # reset draws
        models = normalize_traffic(traffic) or ()
        self.streams = [_Stream(m, i, E, R, self.dev, self.step_ms, self.slot_ms, self.gen, self.rgen)
                        for i, m in enumerate(models) if m.generates and m.direction == direction]
        self.M = sum(s.M for s in self.streams)
        self.reset(None)

    @property
    def deferred(self):
        """Arrivals pushed to a later step because a model's max_msgs_per_step was full (a tensor)."""
        return sum((s.deferred for s in self.streams), torch.zeros((), dtype=torch.long, device=self.dev))

    def reset(self, env_mask_e=None):
        """env_mask_e: bool [E] (None = all envs)."""
        m = torch.ones(self.E, dtype=torch.bool, device=self.dev) if env_mask_e is None else env_mask_e
        for s in self.streams:
            s.reset(m)
        if env_mask_e is None:
            for s in self.streams:
                s.deferred.zero_()

    def step(self, clock=None, triggers=None) -> Arrivals:
        E, R, d = self.E, self.R, self.dev
        vs, bs, ts, tags, prios, dls, dets = [], [], [], [], [], [], []
        for s in self.streams:
            v, b, t = s.step(clock, triggers)
            k = v.shape[-1]
            vs.append(v), bs.append(b.float()), ts.append(t)
            tags.append(torch.full((E, R, k), s.tag, dtype=torch.long, device=d))
            prios.append(torch.full((E, R, k), s.m.priority, dtype=torch.long, device=d))
            dls.append(torch.full((E, R, k), s.m.deadline_ms, dtype=torch.float64, device=d))
            dets.append(torch.full((E, R, k), s.m.det, dtype=torch.bool, device=d))
        if not vs:
            z = torch.zeros(E, R, 0, device=d)
            return Arrivals(z.bool(), z.float(), z.long(), z.long(), z.long(), z.double(), z.bool())
        valid, nbytes, t = torch.cat(vs, -1), torch.cat(bs, -1), torch.cat(ts, -1)
        t = torch.where(valid, t, torch.zeros_like(t))          # empty entries: slot 0 (never inf -> long)
        slot = torch.floor(t.clamp(min=0) / self.slot_ms).long().clamp(0, self.N - 1)
        M = valid.shape[-1]
        key = torch.where(valid, slot, torch.full_like(slot, self.N)) * M + torch.arange(M, device=d)
        order = key.argsort(-1)
        g = lambda x: x.gather(-1, order)
        return Arrivals(g(valid), g(nbytes), g(slot), g(torch.cat(tags, -1)), g(torch.cat(prios, -1)),
                        g(torch.cat(dls, -1)), g(torch.cat(dets, -1)))
