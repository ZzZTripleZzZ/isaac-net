"""Compatibility module. The registry engine that lived here (isaac/demo's ref_engine.py: per-env-clock L0 / L1 /
L2 port of the prototype levels) was retired when the Isaac layer moved onto core.make_engine, whose levels now
have per-env clocks and exact partial resets themselves. Its public names map onto the new layer:

    NetConfig, NetModule, TrafficRequest, MessageHistory, net_features   -> isaac/net_module.py
    ParamRanges, segment_sphere_blocked, los_blocked_kernel               -> isaac/radio.py

NetModule(NetConfig(..., rung="L2")) now runs level "L2-legacy" (the same NetSlot model) through make_engine and
returns a dict from step(t, poses, cur_tag); the registry's NetOutput dataclass and its per-env MAC parameters
(bg_load, L0 lognormal parameters) are gone: use level "L0DR" for randomized delay.
"""
from .net_module import MessageHistory, NetConfig, NetModule, TrafficRequest, net_features  # noqa: F401
from .radio import ParamRanges, segment_sphere_blocked  # noqa: F401
from . import radio as _radio

los_blocked_kernel = getattr(_radio, "los_blocked_kernel", None)
