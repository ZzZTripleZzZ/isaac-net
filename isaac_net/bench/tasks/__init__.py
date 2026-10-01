"""Benchmark tasks and their variants.

    TASKS      {name: NetTask subclass}
    VARIANTS   {name: (description, TaskConfig overrides)}; "background" only once the background-UE feature exists
    make_task(cfg, device) -> NetTask (or the Isaac Lab adapter with cfg.sim == "isaac")
"""
from __future__ import annotations

from ..spec import TaskConfig, background_available
from .coop_map import CoopMap
from .coverage_nav import CoverageNav
from .edge_control import EdgeControl
from .fleet_alert import FleetAlert

TASKS = {c.NAME: c for c in (FleetAlert, CoopMap, CoverageNav, EdgeControl)}

_VARIANTS = {
    "default": ("the standard task", {}),
    "light": ("negative control: every message size x the task's LIGHT_SCALE, everything else identical, so even the "
              "heaviest send choice at every step offers about a tenth of the default cell's uplink capacity and the "
              "load the team creates no longer decides delivery", {}),
    "background": ("the standard task plus background UEs that share the cell (needs NRConfig.background)", {}),
}


def variants() -> dict:
    """The variants available in this installation (background only when the feature is present)."""
    return {k: v for k, v in _VARIANTS.items() if k != "background" or background_available()}


def resolve(cfg: TaskConfig) -> TaskConfig:
    """cfg with its variant's overrides applied (size_scale etc.); refuses unknown or unavailable variants."""
    avail = variants()
    if cfg.variant not in avail:
        why = " (the background-UE feature is not in this installation)" if cfg.variant == "background" else ""
        raise ValueError(f"unknown variant {cfg.variant!r}{why}; one of {tuple(avail)}")
    if cfg.task not in TASKS:
        raise ValueError(f"unknown task {cfg.task!r}; one of {tuple(TASKS)}")
    over = dict(avail[cfg.variant][1])
    if cfg.variant == "light":
        over["size_scale"] = TASKS[cfg.task].LIGHT_SCALE
    return cfg.with_(**over) if over else cfg


def make_task(cfg: TaskConfig, device="cpu"):
    cfg = resolve(cfg)
    if cfg.sim == "isaac":
        from ..isaac_adapter import make_isaac_task
        return make_isaac_task(cfg, device)
    return TASKS[cfg.task](cfg, device)


__all__ = ["TASKS", "variants", "resolve", "make_task", "FleetAlert", "CoopMap", "CoverageNav", "EdgeControl"]
