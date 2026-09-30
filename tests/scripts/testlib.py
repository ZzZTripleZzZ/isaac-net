"""Shared helpers for the equivalence, partial-reset and regression tests.

Inject / InjectArrival replace the reference engine's random draws by pre-generated tensors, so the
reference and a fast backend (which gets the same tensors through set_noise) see identical randomness.
"""
import math

import os
import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root
_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from isaaclab_net.core.proto import netsim
from isaaclab_net.core.proto.netsim import UL_PER_STEP, S

SIZES = (4000.0, 30000.0)
L0_PARAMS = {"mu": math.log(0.3), "sig": 0.8, "p": 0.05}


def synthetic_params(rung, device="cpu", seed=0):
    """L0 params, or monotone synthetic L05/L05Q quantile tables with the bin layout of netsim.lookup_key."""
    if rung == "L0":
        return dict(L0_PARAMS)
    if rung not in ("L05", "L05Q"):
        return None
    g = torch.Generator().manual_seed(seed)
    bins = [len(netsim.NACT_EDGES) + 1, len(netsim.SNR_EDGES) + 1]
    if rung == "L05Q":
        bins.append(len(netsim.OWNQ_EDGES) + 1)
    bins.append(2)
    scale = 0.05 + 3.0 * torch.rand(*bins, 1, generator=g)
    q = scale * torch.linspace(0.0, 1.0, 101) ** 2 * 4 + 0.02         # quantiles in control steps
    pd = 0.1 * torch.rand(*bins, generator=g)
    return {"q": q.to(device), "pdrop": pd.to(device)}


class Inject:
    """Replace torch.randn_like / torch.rand_like in NetSlot._transmit by per-slot pre-drawn tensors."""

    def __init__(self, nz, u):
        self.nz, self.u, self.i, self.j = nz, u, 0, 0

    def __enter__(self):
        self._rn, self._ru = torch.randn_like, torch.rand_like
        def rn(x, *a, **k):
            assert x.shape == self.nz.shape[1:]
            v = self.nz[self.i]; self.i += 1; return v
        def ru(x, *a, **k):
            assert x.shape == self.u.shape[1:]
            v = self.u[self.j]; self.j += 1; return v
        torch.randn_like, torch.rand_like = rn, ru
        return self

    def __exit__(self, *exc):
        torch.randn_like, torch.rand_like = self._rn, self._ru
        if exc[0] is None:
            assert self.i == UL_PER_STEP and self.j == UL_PER_STEP, (self.i, self.j)


class InjectArrival:
    """Replace torch.randn / torch.rand in NetDelay._arrival_draws by pre-drawn z, u1, u2 [E,R]."""

    def __init__(self, z, u1, u2):
        self.vals_n = [z]
        self.vals_u = [u1, u2]

    def __enter__(self):
        self._rn, self._ru = torch.randn, torch.rand
        def rn(*a, **k):
            assert "generator" not in k and tuple(a) == tuple(self.vals_n[0].shape), a
            return self.vals_n.pop(0)
        def ru(*a, **k):
            assert "generator" not in k and tuple(a) == tuple(self.vals_u[0].shape), a
            return self.vals_u.pop(0)
        torch.randn, torch.rand = rn, ru
        return self

    def __exit__(self, *exc):
        torch.randn, torch.rand = self._rn, self._ru
        if exc[0] is None:
            assert not self.vals_n and not self.vals_u


class Workload:
    """Synthetic regime-switching traffic (idle, medium, burst, medium-small) that exercises SR, HARQ,
    RLC wait, overflow and timeouts; plus all random draws the engines need, from its own generator."""

    def __init__(self, E, R, dev, seed):
        self.g = torch.Generator(device=dev).manual_seed(seed)
        self.E, self.R, self.dev = E, R, dev
        self.base = -5 + 40 * torch.rand(E, R, device=dev, generator=self.g)
        self.hid = torch.zeros(E, dtype=torch.long, device=dev)

    def inputs(self, t):
        E, R, d, g = self.E, self.R, self.dev, self.g
        phase = (t // 25) % 4
        p = [0.03, 0.3, 0.9, 0.5][phase]
        big = [0.3, 0.5, 1.0, 0.1][phase]
        tx = torch.rand(E, R, device=d, generator=g) < p
        lg = torch.rand(E, R, device=d, generator=g) < big
        send = tx.long() * (1 + lg.long())
        det = tx & (torch.rand(E, R, device=d, generator=g) < 0.3)
        self.hid = self.hid + (torch.rand(E, device=d, generator=g) < 0.05).long()
        self.base = (self.base + 0.5 * torch.randn(E, R, device=d, generator=g)).clamp(-10, 40)
        return send, det, self.hid.clone(), self.base.clone()

    def noise(self):
        E, R, d, g = self.E, self.R, self.dev, self.g
        return (torch.randn(UL_PER_STEP, E, R, S, 2, device=d, generator=g),
                torch.rand(UL_PER_STEP, E, R, device=d, generator=g))

    def arrival_noise(self):
        E, R, d, g = self.E, self.R, self.dev, self.g
        return (torch.randn(E, R, device=d, generator=g), torch.rand(E, R, device=d, generator=g),
                torch.rand(E, R, device=d, generator=g))

    def reset_ids(self, frac):
        """Random subset of envs (never empty, never all when E > 1)."""
        m = torch.rand(self.E, device=self.dev, generator=self.g) < frac
        m[0] = True
        if self.E > 1:
            m[-1] = False
        return m.nonzero(as_tuple=True)[0]


def is_ref(net):
    return isinstance(net, netsim.NetBase)


def drive(net, rung, t, req, snr, wl_noise, arr_noise, api="new", hid=None):
    """One submit + step with injected draws on either engine kind. Returns the step dict (new API) or the
    legacy (newest, det_env) tuple."""
    if is_ref(net):
        if rung in ("L0", "L0DR", "L05", "L05Q"):
            with InjectArrival(*arr_noise):
                _submit(net, t, req, snr, api)
        else:
            _submit(net, t, req, snr, api)
        if rung == "L2":
            with Inject(*wl_noise):
                return _step(net, t, snr, api, hid)
        return _step(net, t, snr, api, hid)
    if rung in ("L0", "L0DR", "L05", "L05Q"):
        net.set_noise(*arr_noise)
    _submit(net, t, req, snr, api)
    if rung == "L2":
        net.set_noise(*wl_noise)
    return _step(net, t, snr, api, hid)


def _submit(net, t, req, snr, api):
    if api == "new":
        net.submit(t, req, snr)
    else:
        net.add_frames(t, req.send, req.det, req.hid, snr)


def _step(net, t, snr, api, hid):
    if api == "new":
        return net.step(t, snr)
    return net.step(t, snr, hid)


def state_names(rung):
    names = list(netsim.NetBase.FIELDS) + ["clock"]
    if rung == "L2":
        names += netsim.NetSlot.MAC
    if rung == "L0DR":
        names += ["mu", "sig", "p"]
    return names


def same(x, y):
    """Bitwise tensor equality, NaN == NaN."""
    if x.dtype.is_floating_point:
        nx, ny = torch.isnan(x), torch.isnan(y)
        return torch.equal(nx, ny) and torch.equal(torch.where(nx, 0.0, x), torch.where(ny, 0.0, y))
    return torch.equal(x, y)


def maxdiff(x, y):
    d = (x.double() - y.double()).abs()
    d = d[~torch.isnan(d)]
    return float(d.max()) if d.numel() else 0.0
