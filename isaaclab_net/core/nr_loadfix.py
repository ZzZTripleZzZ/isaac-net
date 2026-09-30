"""Compatibility shim for the load-gap prototype (docs/fidelity-load-gap.md).

The 5G-LENA MAC behaviors under load that the prototype added as subclasses are now NRConfig switches of the NR
engine itself (mac.py, mac_ul.py): pf_update, pf_avg_idle, ul_retx_sched, ul_amc_alloc, ul_grant_model and the BSR
pipeline parameters; lena_match_v2() / lena_validation_v2() turn them all on. This module only maps the prototype's
LoadFixConfig names onto those fields, so the load-gap benchmark scripts and their arm names keep working:

    pf_intra_slot   -> pf_update="rbg"            pf_active_only  -> pf_avg_idle="freeze"
    retx_tdma       -> ul_retx_sched="tdma"       amc_prev_alloc  -> ul_amc_alloc="previous"
    grant_pipeline  -> ul_grant_model="bsr"       tb_overhead_bytes, sr_boot_slots, bsr_delay_slots, bsr_hdr_bytes,
    boot_bytes      -> sr_boot_bytes              rlc_tail_bytes: same names
    est_hdr_bytes   -> bsr_est_hdr_bytes          rlc_tail_timer_slots -> rlc_tail_timer_ms (x slot_ms)

LoadFixNet(E, R, device, sizes, cfg, lf=...) is NRNet(E, R, device, sizes, apply_loadfix(cfg, lf)).
"""
from __future__ import annotations

from dataclasses import dataclass

from .config import NRConfig
from .mac_ul import BSR_LEVELS, BSR_STATE, UlMac
from .nr_engine import NRNet

__all__ = ["BSR_LEVELS", "LoadFixConfig", "LoadFixNet", "LoadFixUlMac", "ARMS", "make_arm", "apply_loadfix"]

LoadFixUlMac = UlMac          # the switches live in UlMac / MacLink now


@dataclass
class LoadFixConfig:
    """The prototype's switch set (all off = NRConfig unchanged); see apply_loadfix for the NRConfig fields."""
    pf_intra_slot: bool = False
    pf_active_only: bool = False
    retx_tdma: bool = False
    amc_prev_alloc: bool = False
    grant_pipeline: bool = False
    tb_overhead_bytes: int | None = None
    sr_boot_slots: int = 6
    boot_bytes: int = 17
    bsr_delay_slots: int = 10
    bsr_hdr_bytes: int = 8
    est_hdr_bytes: int = 5
    rlc_tail_bytes: int = 16
    rlc_tail_timer_slots: int = 20

    def any(self):
        return (self.pf_intra_slot or self.pf_active_only or self.retx_tdma or self.amc_prev_alloc
                or self.grant_pipeline or self.tb_overhead_bytes is not None)


def apply_loadfix(cfg: NRConfig, lf: LoadFixConfig | None) -> NRConfig:
    """cfg with the NRConfig fields of the prototype switch set lf (lf None or all off: cfg itself)."""
    if lf is None or not lf.any():
        return cfg
    kw = {}
    if lf.pf_intra_slot:
        kw["pf_update"] = "rbg"
    if lf.pf_active_only:
        kw["pf_avg_idle"] = "freeze"
    if lf.retx_tdma:
        kw["ul_retx_sched"] = "tdma"
    if lf.amc_prev_alloc:
        kw["ul_amc_alloc"] = "previous"
    if lf.tb_overhead_bytes is not None:
        kw["tb_overhead_bytes"] = lf.tb_overhead_bytes
    if lf.grant_pipeline:
        kw.update(ul_grant_model="bsr", sr_boot_slots=lf.sr_boot_slots, sr_boot_bytes=lf.boot_bytes,
                  bsr_delay_slots=lf.bsr_delay_slots, bsr_hdr_bytes=lf.bsr_hdr_bytes,
                  bsr_est_hdr_bytes=lf.est_hdr_bytes, rlc_tail_bytes=lf.rlc_tail_bytes,
                  rlc_tail_timer_ms=lf.rlc_tail_timer_slots * cfg.slot_ms)
    return cfg.with_(**kw)


# named switch sets of the load-gap benchmark (benchmarks/fidelity/loadfix/)
ARMS = {
    "base": {},
    "pf": dict(pf_intra_slot=True, pf_active_only=True),
    "pf_intra": dict(pf_intra_slot=True),
    "pf_active": dict(pf_active_only=True),
    "retx": dict(retx_tdma=True),
    "amc": dict(amc_prev_alloc=True),
    "oh8": dict(tb_overhead_bytes=8),
    "pipe": dict(grant_pipeline=True, tb_overhead_bytes=8),
    "pf_pipe": dict(pf_intra_slot=True, pf_active_only=True, grant_pipeline=True, tb_overhead_bytes=8),
    "all": dict(pf_intra_slot=True, pf_active_only=True, retx_tdma=True, amc_prev_alloc=True,
                grant_pipeline=True, tb_overhead_bytes=8),
}


def make_arm(name, **extra):
    """LoadFixConfig of a named arm (ARMS) with overrides."""
    kw = dict(ARMS[name])
    kw.update(extra)
    return LoadFixConfig(**kw)


class LoadFixNet(NRNet):
    """NRNet on apply_loadfix(cfg, lf). Kept for the prototype's scripts; new code sets the NRConfig fields."""

    X_STATE = tuple(BSR_STATE)

    def __init__(self, E, R, device, sizes, cfg: NRConfig | None = None, generator=None,
                 lf: LoadFixConfig | None = None, seed=None):
        self.lf = lf or LoadFixConfig()
        super().__init__(E, R, device, sizes, apply_loadfix(cfg or NRConfig(), self.lf), generator, seed)
