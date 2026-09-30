"""Thin adapter between the tests and the engine API.

Every call into the engine's public API (construction, add_frames/step, stats) goes through the
functions in this file, so that an API change (for example the submit/step API planned on
feat/engine-api) only needs edits here. The probes at the bottom reach into engine internals
(`_transmit`, the module-level `serve_fifo`) to observe per-slot state without changing the engine.
"""
from __future__ import annotations

import math
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch

from isaaclab_net.core.proto import netsim as ns

SIZES = (4000.0, 30000.0)
F, TIMEOUT, S, UL_PER_STEP = ns.F, ns.TIMEOUT, ns.S, ns.UL_PER_STEP
DELAY_LEVELS = ("L0", "L0DR", "L05", "L05Q")
SLOT_LEVELS = ("L1", "L2")
LEVELS = tuple(ns.RUNGS)


# ----------------------------------------------------------------------------------------------
# Construction
# ----------------------------------------------------------------------------------------------
def l0_params(mu=math.log(0.5), sig=0.5, p=0.05):
    return {"mu": mu, "sig": sig, "p": p}


def lookup_shape(mode):
    """Table shape [*key bins] for L05 / L05Q (one more bin than edges, two traffic classes)."""
    dims = [len(ns.NACT_EDGES) + 1, len(ns.SNR_EDGES) + 1]
    if mode == "L05Q":
        dims.append(len(ns.OWNQ_EDGES) + 1)
    dims.append(len(SIZES))
    return tuple(dims)


def lookup_params(mode, quantiles=None, pdrop=None):
    """L05 / L05Q params. Default: delay quantiles 0.2..3 steps in every bin, 2% drop."""
    shape = lookup_shape(mode)
    if quantiles is None:
        quantiles = torch.linspace(0.2, 3.0, 101).expand(*shape, 101).clone()
    if pdrop is None:
        pdrop = torch.full(shape, 0.02)
    return {"q": quantiles, "pdrop": pdrop}


def default_params(level):
    if level == "L0":
        return l0_params()
    if level in ("L05", "L05Q"):
        return lookup_params(level)
    return None


def make_ref(level, E, R, device, sizes=SIZES, params=None):
    """Reference (eager) engine at fidelity `level`."""
    if params is None:
        params = default_params(level)
    return ns.make_net(level, E, R, device, sizes, params)


def make_fast(E, R, device, backend, sizes=SIZES, inject=False):
    """Fast L2 engine (backend in eager / graph / compile / triton)."""
    from isaaclab_net.core.proto.netsim_fast import NetSlotFast
    return NetSlotFast(E, R, device, sizes, backend=backend, inject=inject)


# ----------------------------------------------------------------------------------------------
# Per-step API
# ----------------------------------------------------------------------------------------------
def queued(net):
    """Frames queued per robot [E,R]."""
    return net.queued()


def submit(net, t, send, det, hid, snr):
    """Enqueue this step's frames. Returns the accepted mask [E,R] (send > 0 and a free FIFO slot)."""
    accepted = (send > 0) & (queued(net) < F)
    net.add_frames(t, send, det, hid, snr)
    return accepted


def advance(net, t, snr, hid):
    """Advance [t, t+1). Returns (newest delivered capture step [E,R], detection delivered [E])."""
    return net.step(t, snr, hid)


def enable_stats(net, on=True):
    net.log_stats = on


def collect(net):
    return net.collect()


def set_noise(net, nz, u):
    net.set_noise(nz, u)


# L2 MAC state names, shared by reference and fast engines
FIFO_FIELDS = ("cap", "cls", "det", "hid", "rem", "dlv", "f_nact", "f_snr", "f_own")
MAC_FIELDS = ("bsr", "sr_t", "avg", "olla", "wait", "hcnt", "h")


def state(net, names=FIFO_FIELDS + MAC_FIELDS):
    return {n: getattr(net, n).clone() for n in names if hasattr(net, n)}


# ----------------------------------------------------------------------------------------------
# Workload (device-agnostic, own generator so it does not disturb the engine's RNG stream)
# ----------------------------------------------------------------------------------------------
class Workload:
    """Regime-switching synthetic traffic: idle, medium, burst (all large), medium-small.

    Same generator as tests/scripts/test_equiv.py. `snr_lo/snr_hi` bound the per-robot SNR walk.
    """

    def __init__(self, E, R, device, seed=0, period=25, snr_lo=-10.0, snr_hi=40.0, p=None, big=None):
        dev = torch.device(device)
        self.g = torch.Generator(device=dev).manual_seed(seed)
        self.E, self.R, self.dev, self.period = E, R, dev, period
        self.snr_lo, self.snr_hi = snr_lo, snr_hi
        self.p = p if p is not None else [0.03, 0.3, 0.9, 0.5]
        self.big = big if big is not None else [0.3, 0.5, 1.0, 0.1]
        self.base = (snr_lo + 5) + (snr_hi - snr_lo - 10) * torch.rand(E, R, device=dev, generator=self.g)
        self.hid = torch.zeros(E, dtype=torch.long, device=dev)

    def inputs(self, t):
        E, R, d, g = self.E, self.R, self.dev, self.g
        phase = (t // self.period) % len(self.p)
        tx = torch.rand(E, R, device=d, generator=g) < self.p[phase]
        lg = torch.rand(E, R, device=d, generator=g) < self.big[phase]
        send = tx.long() * (1 + lg.long())
        det = tx & (torch.rand(E, R, device=d, generator=g) < 0.3)
        self.hid = self.hid + (torch.rand(E, device=d, generator=g) < 0.05).long()
        self.base = (self.base + 0.5 * torch.randn(E, R, device=d, generator=g)).clamp(self.snr_lo, self.snr_hi)
        return send, det, self.hid.clone(), self.base.clone()

    def noise(self):
        E, R, d, g = self.E, self.R, self.dev, self.g
        return (torch.randn(UL_PER_STEP, E, R, S, 2, device=d, generator=g),
                torch.rand(UL_PER_STEP, E, R, device=d, generator=g))


# ----------------------------------------------------------------------------------------------
# Probes (engine internals; update here if the internals move)
# ----------------------------------------------------------------------------------------------
class InjectNoise:
    """Feed the reference L2 engine pre-generated per-slot draws.

    NetSlot draws randn_like(h) [E,R,S,2] (fading) and rand_like(p_ok) [E,R] (BLER) once per UL slot.
    Inside this context both are replaced by readers of nz [40,E,R,S,2] and u [40,E,R].
    """

    def __init__(self, nz, u):
        self.nz, self.u, self.i, self.j = nz, u, 0, 0

    def __enter__(self):
        self._rn, self._ru = torch.randn_like, torch.rand_like

        def rn(x, *a, **k):
            assert x.shape == self.nz.shape[1:], (x.shape, self.nz.shape)
            v = self.nz[self.i]
            self.i += 1
            return v

        def ru(x, *a, **k):
            assert x.shape == self.u.shape[1:], (x.shape, self.u.shape)
            v = self.u[self.j]
            self.j += 1
            return v

        torch.randn_like, torch.rand_like = rn, ru
        return self

    def __exit__(self, *exc):
        torch.randn_like, torch.rand_like = self._rn, self._ru
        if exc[0] is None:
            assert self.i == UL_PER_STEP and self.j == UL_PER_STEP, (self.i, self.j)


@contextmanager
def slot_probe(keys, module=ns):
    """Record, once per UL slot, the caller's locals named in `keys` at the moment `serve_fifo` runs.

    Works for the reference L1/L2 `_transmit` loops (module=netsim) and for netsim_fast.slot_body
    (module=netsim_fast). Each record also holds 'rem_in', 'b', 'rem_out', 'fin' of serve_fifo and,
    when the caller is a method, 'self.<attr>' for any key written as 'self.<attr>' (pre-update values).
    """
    orig = module.serve_fifo
    records = []

    def probe(rem, b):
        caller = sys._getframe(1).f_locals
        new, fin = orig(rem, b)
        rec = {"rem_in": rem.clone(), "b": b.clone(), "rem_out": new.clone(), "fin": fin.clone()}
        for k in keys:
            if k.startswith("self."):
                obj = caller.get("self")
                v = getattr(obj, k[5:]) if obj is not None else None
            else:
                v = caller.get(k)
            rec[k] = v.clone() if torch.is_tensor(v) else v
        records.append(rec)
        return new, fin

    module.serve_fifo = probe
    try:
        yield records
    finally:
        module.serve_fifo = orig


@dataclass
class StepRecord:
    t: int
    cap: torch.Tensor          # [E,R,F] FIFO capture steps during the step (before removal)
    cls: torch.Tensor
    rem_after_tx: torch.Tensor  # bytes left per frame after this step's service, before removal
    fin: torch.Tensor          # finish time per frame, inf if not finished this step
    delivered: torch.Tensor    # [E,R,F] bool
    timed: torch.Tensor        # [E,R,F] bool
    newest: torch.Tensor = None
    accepted: torch.Tensor = None
    extra: dict = field(default_factory=dict)


class StepProbe:
    """Wrap a reference engine's `_transmit` to observe finish times and residual bytes per step."""

    def __init__(self, net):
        self.net, self.last = net, None
        self._orig = net._transmit

        def wrapped(t, snr_db):
            # the engine passes the per-env clock t [E]; the tests drive all envs with one global step
            fin = self._orig(t, snr_db)
            n = self.net
            cap = n.cap.clone()
            tt = t[:, None, None] if torch.is_tensor(t) else t
            delivered = (cap >= 0) & torch.isfinite(fin)
            timed = (cap >= 0) & ~delivered & ((tt + 1 - cap) >= TIMEOUT)
            t_int = int(t[0]) if torch.is_tensor(t) else t
            self.last = StepRecord(t_int, cap, n.cls.clone(), n.rem.clone(), fin.clone(), delivered, timed)
            return fin

        net._transmit = wrapped

    def detach(self):
        self.net._transmit = self._orig


def fast_finish_fields(net, fin, t):
    """netsim_fast.finish_body on a reference engine's FIFO state at global step t; returns the
    9 compacted FIFO fields in FIFO_FIELDS order."""
    from isaaclab_net.core.proto.netsim_fast import finish_body
    E = net.E
    tv = torch.full((E,), int(t), dtype=torch.long)
    fields, _ = finish_body(net.cap, net.cls, net.det, net.hid, net.rem, net.dlv, net.f_nact, net.f_snr,
                            net.f_own, fin, tv, torch.zeros(E, dtype=torch.long))
    return fields


def run(net, wl, steps, t0=0, probe=False, on_step=None):
    """Drive `net` with workload `wl` for `steps` steps; returns list of StepRecord if probe=True."""
    sp = StepProbe(net) if probe else None
    out = []
    try:
        for t in range(t0, t0 + steps):
            send, det, hid, snr = wl.inputs(t)
            acc = submit(net, t, send, det, hid, snr)
            newest, det_env = advance(net, t, snr, hid)
            if sp is not None:
                rec = sp.last
                rec.newest, rec.accepted = newest, acc
                rec.extra["send"] = send
                out.append(rec)
            if on_step is not None:
                on_step(t, net, newest, det_env)
    finally:
        if sp is not None:
            sp.detach()
    return out


def make_blackhole(level, E, R, device):
    """An engine that never delivers: returns (net, snr_db [E,R]).

    Delay levels drop every frame (loss probability 1); slot levels see -60 dB SNR, where the L1
    rate is ~1e-4 bytes per slot and the L2 transport block fails with probability ~1.
    """
    dev = torch.device(device)
    params = None
    if level == "L0":
        params = l0_params(p=1.0)
    elif level in ("L05", "L05Q"):
        params = lookup_params(level, pdrop=torch.ones(lookup_shape(level)))
    net = make_ref(level, E, R, dev, params=params)
    if level == "L0DR":
        net.p.fill_(1.0)          # per-env loss probability sampled at reset
    snr = torch.full((E, R), -60.0 if level in SLOT_LEVELS else 10.0, device=dev)
    return net, snr
