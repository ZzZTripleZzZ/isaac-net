"""Levels beyond the simulators: fitted surrogates (TR, GE, QA, NN) and value-of-information bounds (ORACLE,
NOCOMM). Build them with `isaac_net.core.make_engine(level, ...)`; fit the surrogates with
`python -m isaac_net.tools.fit_levels`.

    base.py        LevelNet: graph-safe engine base (reference and graph backends, per-env clocks, partial resets)
    surrogates.py  NetTR, NetGE, NetQA, NetNN (+ DelayNet, nn_features)
    bounds.py      NetOracle, NetNoComm
"""
from __future__ import annotations

import os

import torch

from .base import LevelNet
from .bounds import NetNoComm, NetOracle
from .surrogates import DelayNet, NetGE, NetNN, NetQA, NetTR, nn_features

SURROGATE_LEVELS = ("TR", "GE", "QA", "NN")
BOUND_LEVELS = ("ORACLE", "NOCOMM")
CLASSES = {"TR": NetTR, "GE": NetGE, "QA": NetQA, "NN": NetNN, "ORACLE": NetOracle, "NOCOMM": NetNoComm}
NEEDS_FIT = ("TR", "GE", "NN")
QA_DEFAULT = {"eta": 0.9, "pf": True}
# Application timing a fit was made with, as recorded in its meta; a fit file without these keys predates them and
# was fitted with the prototype constants below.
FIT_APP_KEYS = {"frame_buffer": 16, "timeout_steps": 20, "control_step_ms": 100.0, "ul_slots_per_step": 40}


def fit_app(cfg):
    """The FIT_APP_KEYS values of an NRConfig (what fit_levels records and load_level_params checks)."""
    return {"frame_buffer": int(cfg.frame_buffer), "timeout_steps": int(cfg.timeout_steps),
            "control_step_ms": float(cfg.control_step_ms), "ul_slots_per_step": int(cfg.proto_slots_per_step)}


def load_level_params(level, params, sizes=None, app=None):
    """Parameters of one level from `params`, which is one of
      None                   QA falls back to QA_DEFAULT (uncalibrated); ORACLE / NOCOMM need nothing;
                             TR, GE and NN raise, because they only exist as fits
      a path (str / PathLike) to a fit file written by isaac_net.tools.fit_levels (or the legacy
                             baseline fitter), loaded with torch.load(weights_only=True)
      a fit-file dict        {"TR": ..., "GE": ..., "QA": ..., "NN": ..., "meta": {...}}: the entry of `level`
      the level's own dict   used as is
    A fit file records the message sizes it was fitted with (meta["sizes"]); they must equal `sizes`. It also records
    the frame buffer, timeout, control step and UL slots per step (FIT_APP_KEYS); they must equal `app` (a dict as
    returned by fit_app), because the fitted delays are in control steps of that configuration."""
    if level in BOUND_LEVELS:
        return None
    if isinstance(params, (str, os.PathLike)):
        params = torch.load(os.path.expanduser(os.fspath(params)), map_location="cpu", weights_only=True)
    if isinstance(params, dict) and level in params:
        meta = params.get("meta") or {}
        if sizes is not None and "sizes" in meta and tuple(float(s) for s in meta["sizes"]) != tuple(sizes):
            raise ValueError(f"the fit file was fitted with msg_sizes={tuple(meta['sizes'])}, the engine uses "
                             f"{tuple(sizes)}; refit for these sizes")
        if app is not None:
            bad = {k: (meta.get(k, v0), app[k]) for k, v0 in FIT_APP_KEYS.items()
                   if k in app and type(v0)(meta.get(k, v0)) != app[k]}
            if bad:
                raise ValueError("the fit file was fitted with " + ", ".join(f"{k}={a}" for k, (a, _) in bad.items())
                                 + "; the config asks for " + ", ".join(f"{k}={b}" for k, (_, b) in bad.items())
                                 + ". Refit with `python -m isaac_net.tools.fit_levels` under this config")
        params = params[level]
    if params is None:
        if level == "QA":
            return dict(QA_DEFAULT)
        raise ValueError(f"level {level} needs fitted parameters: run `python -m isaac_net.tools.fit_levels` "
                         "and pass make_engine(..., params=<fit file path or dict>)")
    return params


def make_level(level, E, R, device, sizes, params=None, backend="reference", inject=False, seed=None, app=None,
               **kw):
    """Build a surrogate or bound level (make_engine calls this after its config checks). app: fit_app(cfg), checked
    against the fit file; kw: fb, timeout, ul_per_step, rng for the engine."""
    p = load_level_params(level, params, sizes, app)
    return CLASSES[level](E, R, device, sizes, params=p, backend=backend, inject=inject, seed=seed, **kw)


__all__ = ["LevelNet", "NetTR", "NetGE", "NetQA", "NetNN", "NetOracle", "NetNoComm", "DelayNet", "nn_features",
           "SURROGATE_LEVELS", "BOUND_LEVELS", "CLASSES", "FIT_APP_KEYS", "fit_app", "load_level_params", "make_level"]
