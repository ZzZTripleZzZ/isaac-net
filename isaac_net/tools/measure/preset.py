"""Preset files: the output of calibrate.py that turns a measurement afternoon into an NRConfig.

A preset is JSON:

  {"name": "lab_srsran_2026-10-02",
   "base": "srsran_like",              # a preset function of isaac_net.core.config, or null for NRConfig()
   "nrconfig": {"sr_period_slots": 40, "proc_offset_ms": 2.1, ...},    # NRConfig field overrides
   "calibration_knobs": {...},         # fitted quantities with no NRConfig field (capacity scale, SINR offsets)
   "provenance": {"proc_offset_ms": "engine replay fit, experiment a, W1 0.8 ms", ...},
   "metrics": {...}}

  from isaac_net.tools.measure.preset import load_preset
  cfg = load_preset("lab_srsran.json")            # an NRConfig
  net = make_engine("L2", E, R, "cuda", cfg)
"""
from __future__ import annotations

import json
from dataclasses import fields

from isaac_net.core import config as C

BASES = {"srsran_like": C.srsran_like, "oai_like": C.oai_like, "lena_like": C.lena_like,
         "netslot_compat": C.netslot_compat, None: C.NRConfig, "": C.NRConfig}
TUPLE_FIELDS = {f.name for f in fields(C.NRConfig) if isinstance(f.default, tuple)}


def to_nrconfig(preset, **overrides):
    base = preset.get("base")
    if base not in BASES:
        raise ValueError(f"unknown base preset {base!r}; one of {sorted(k for k in BASES if k)}")
    kw = dict(preset.get("nrconfig", {}))
    kw.update(overrides)
    known = {f.name for f in fields(C.NRConfig)}
    bad = set(kw) - known
    if bad:
        raise ValueError(f"not NRConfig fields: {sorted(bad)}")
    for k in TUPLE_FIELDS & set(kw):
        v = kw[k]
        kw[k] = tuple(tuple(x) if isinstance(x, list) else x for x in v) if isinstance(v, list) else v
    return BASES[base](**kw)


def load_preset(path, **overrides):
    """NRConfig from a preset file written by calibrate.py; keyword overrides win over the file."""
    with open(path) as f:
        return to_nrconfig(json.load(f), **overrides)


def write_preset(path, name, base, nrconfig, knobs=None, provenance=None, metrics=None):
    p = {"name": name, "base": base, "nrconfig": nrconfig, "calibration_knobs": knobs or {},
         "provenance": provenance or {}, "metrics": metrics or {}}
    to_nrconfig(p)                                            # validate before writing
    with open(path, "w") as f:
        json.dump(p, f, indent=1, default=float)
    return p
