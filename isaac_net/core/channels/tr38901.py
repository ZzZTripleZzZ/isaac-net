"""3GPP TR 38.901 (V17.0.0) large-scale models: path loss, LOS probability, shadow fading and O2I penetration.

Sources (TR 38.901 V17.0.0, 2022-04):
  Table 7.4.1-1   LOS / NLOS path loss and shadow-fading sigma of RMa, UMa, UMi-Street Canyon, InH-Office, InF
  Table 7.4.2-1   LOS probability (InH: the mixed-office curve)
  Table 7.4.3-1/2 O2I building penetration (low-loss and high-loss models, d_2D-in = min of two U(0, 25 m) draws,
                  U(0, 10 m) for RMa; RMa low-loss only)
  Table 7.2-4, Table 7.8-7  InF clutter parameters (r = 20 % / 60 %, d_clutter = 10 m / 2 m, h_c = 2 m / 6 m) and
                  heights (BS 1.5 m for SL / DL, 8 m for SH / DH, UT 1.5 m)
  Table 7.5-6     shadow-fading correlation distances; Table 7.6.3.1-2 LOS-state correlation distances

Simplifications (documented in docs/channels.md):
  * UMa uses h_E = 1 m (exact for h_UT < 13 m, which covers ground robots); the stochastic h_E is not drawn.
  * Distances below the applicability range (10 m outdoor, 1 m indoor) extrapolate the formulas; d_2D >= 1 m.
  * The optional (single-slope) NLOS formulas and the < 6 GHz backwards-compatible O2I model (Table 7.4.3-3) are
    not implemented.
Formulas take d_2D and d_3D in metres and fc in GHz, and work on tensors or Python floats (tests hand-check them).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

C_LIGHT = 3.0e8        # TR 38.901 Note 1 / Note 5 use c = 3.0e8 m/s


@dataclass(frozen=True)
class Scenario:
    name: str
    h_bs: float                 # default BS height (m)
    los_corr_m: float           # LOS-state correlation distance (Table 7.6.3.1-2)
    sf_corr_los_m: float        # shadow-fading correlation distance, LOS / NLOS (Table 7.5-6)
    sf_corr_nlos_m: float
    o2i: tuple = ()             # O2I models this scenario allows ("low", "high")
    d_in_max_m: float = 25.0    # d_2D-in = min(U(0, max), U(0, max))


# InF clutter defaults: (density r, clutter size d_clutter m, clutter height h_c m), Table 7.8-7.
INF_CLUTTER = {"InF-SL": (0.2, 10.0, 2.0), "InF-DL": (0.6, 2.0, 6.0),
               "InF-SH": (0.2, 10.0, 2.0), "InF-DH": (0.6, 2.0, 6.0)}

SCENARIOS = {
    "RMa": Scenario("RMa", 35.0, 60.0, 37.0, 120.0, ("low",), 10.0),
    "UMa": Scenario("UMa", 25.0, 50.0, 37.0, 50.0, ("low", "high")),
    "UMi": Scenario("UMi", 10.0, 50.0, 10.0, 13.0, ("low", "high")),
    "InH": Scenario("InH", 3.0, 10.0, 10.0, 6.0),
    # InF LOS-state correlation = d_clutter / 2 (filled in by inf_params); SF correlation 10 m
    "InF-SL": Scenario("InF-SL", 1.5, 5.0, 10.0, 10.0),
    "InF-DL": Scenario("InF-DL", 1.5, 1.0, 10.0, 10.0),
    "InF-SH": Scenario("InF-SH", 8.0, 5.0, 10.0, 10.0),
    "InF-DH": Scenario("InF-DH", 8.0, 1.0, 10.0, 10.0),
}
ALIASES = {"rma": "RMa", "uma": "UMa", "umi": "UMi", "inh": "InH", "inh_office": "InH", "inf_sl": "InF-SL",
           "inf_dl": "InF-DL", "inf_sh": "InF-SH", "inf_dh": "InF-DH"}


def scenario_name(s):
    if s in SCENARIOS:
        return s
    key = s.lower().replace("-", "_")
    if key in ALIASES:
        return ALIASES[key]
    raise ValueError(f"unknown TR 38.901 scenario {s!r}; one of {tuple(SCENARIOS)}")


def _lg(x):
    return torch.log10(x) if isinstance(x, torch.Tensor) else math.log10(x)


def _max2(a, b):
    ta, tb = isinstance(a, torch.Tensor), isinstance(b, torch.Tensor)
    if ta and tb:
        return torch.maximum(a, b)
    if ta:
        return a.clamp(min=b)
    if tb:
        return b.clamp(min=a)
    return max(a, b)


def _max(*xs):
    out = xs[0]
    for x in xs[1:]:
        out = _max2(out, x)
    return out


def _where(c, a, b):
    """torch.where that accepts Python scalars (filled on the device with full_like: no host copy, graph-safe)."""
    if isinstance(c, torch.Tensor):
        ref = a if isinstance(a, torch.Tensor) else (b if isinstance(b, torch.Tensor) else c.float())
        a = a if isinstance(a, torch.Tensor) else torch.full_like(ref, a, dtype=torch.float32)
        b = b if isinstance(b, torch.Tensor) else torch.full_like(ref, b, dtype=torch.float32)
        return torch.where(c, a, b)
    return a if c else b


def _exp(x):
    return torch.exp(x) if isinstance(x, torch.Tensor) else math.exp(x)


def breakpoint_m(scn, fc_ghz, h_bs, h_ut):
    """d_BP (RMa, Note 5) or d'_BP with h_E = 1 m (UMa, UMi, Note 1); inf for the indoor scenarios."""
    fc = fc_ghz * 1e9
    if scn == "RMa":
        return 2 * math.pi * h_bs * h_ut * fc / C_LIGHT
    if scn in ("UMa", "UMi"):
        return 4 * (h_bs - 1.0) * (h_ut - 1.0) * fc / C_LIGHT
    return math.inf


def _rma_pl1(d3, fc, h=5.0):
    return (20 * _lg(40 * math.pi * d3 * fc / 3) + min(0.03 * h ** 1.72, 10) * _lg(d3)
            - min(0.044 * h ** 1.72, 14.77) + 0.002 * math.log10(h) * d3)


def pl_los(scn, d2, d3, fc, h_bs, h_ut):
    """LOS path loss (dB)."""
    if scn == "RMa":
        dbp = breakpoint_m(scn, fc, h_bs, h_ut)
        pl2 = _rma_pl1(dbp, fc) + 40 * _lg(d3 / dbp)          # PL1(d_BP) + 40 log10(d_3D / d_BP), as ns-3
        return _where(d2 <= dbp, _rma_pl1(d3, fc), pl2)
    if scn in ("UMa", "UMi"):
        dbp = breakpoint_m(scn, fc, h_bs, h_ut)
        a, n1, b = (28.0, 22.0, 9.0) if scn == "UMa" else (32.4, 21.0, 9.5)
        pl1 = a + n1 * _lg(d3) + 20 * math.log10(fc)
        pl2 = a + 40 * _lg(d3) + 20 * math.log10(fc) - b * math.log10(dbp ** 2 + (h_bs - h_ut) ** 2)
        return _where(d2 <= dbp, pl1, pl2)
    if scn == "InH":
        return 32.4 + 17.3 * _lg(d3) + 20 * math.log10(fc)
    return 31.84 + 21.50 * _lg(d3) + 19.00 * math.log10(fc)          # InF, all sub-scenarios


def pl_nlos(scn, d2, d3, fc, h_bs, h_ut, W=20.0, h=5.0):
    """NLOS path loss (dB), including the max(., PL_LOS) floors of Table 7.4.1-1."""
    los = pl_los(scn, d2, d3, fc, h_bs, h_ut)
    lf = 20 * math.log10(fc)
    if scn == "RMa":
        p = (161.04 - 7.1 * math.log10(W) + 7.5 * math.log10(h) - (24.37 - 3.7 * (h / h_bs) ** 2) * math.log10(h_bs)
             + (43.42 - 3.1 * math.log10(h_bs)) * (_lg(d3) - 3) + lf
             - (3.2 * math.log10(11.75 * h_ut) ** 2 - 4.97))
        return _max(los, p)
    if scn == "UMa":
        return _max(los, 13.54 + 39.08 * _lg(d3) + lf - 0.6 * (h_ut - 1.5))
    if scn == "UMi":
        return _max(los, 35.3 * _lg(d3) + 22.4 + 21.3 * math.log10(fc) - 0.3 * (h_ut - 1.5))
    if scn == "InH":
        return _max(los, 38.3 * _lg(d3) + 17.30 + 24.9 * math.log10(fc))
    sl = 33.0 + 25.5 * _lg(d3) + lf
    if scn == "InF-SL":
        return _max(sl, los)
    if scn == "InF-DL":
        return _max(18.6 + 35.7 * _lg(d3) + lf, los, sl)
    if scn == "InF-SH":
        return _max(32.4 + 23.0 * _lg(d3) + lf, los)
    if scn == "InF-DH":
        return _max(33.63 + 21.9 * _lg(d3) + lf, los)
    raise ValueError(scn)


def sigma_sf(scn, d2=None, fc=None, h_bs=None, h_ut=None):
    """(sigma_LOS, sigma_NLOS) in dB. RMa LOS is 4 dB up to d_BP and 6 dB beyond (needs d2 and the heights)."""
    if scn == "RMa":
        los = 4.0 if d2 is None else _where(d2 <= breakpoint_m(scn, fc, h_bs, h_ut), 4.0, 6.0)
        return los, 8.0
    return {"UMa": (4.0, 6.0), "UMi": (4.0, 7.82), "InH": (3.0, 8.03), "InF-SL": (4.3, 5.7),
            "InF-DL": (4.3, 7.2), "InF-SH": (4.3, 5.9), "InF-DH": (4.3, 4.0)}[scn]


def inf_k_subsce(scn, h_bs, h_ut, r=None, d_clutter=None, h_c=None):
    """k_subsce (m) of the InF LOS probability exp(-d_2D / k_subsce), Table 7.4.2-1."""
    r0, dc0, hc0 = INF_CLUTTER[scn]
    r = r0 if r is None else r
    dc = dc0 if d_clutter is None else d_clutter
    hc = hc0 if h_c is None else h_c
    k = -dc / math.log(1 - r)
    if scn in ("InF-SH", "InF-DH"):
        k = k * (h_bs - h_ut) / (hc - h_ut)
    return k


def p_los(scn, d2, h_ut=1.5, k_subsce=None):
    """LOS probability (Table 7.4.2-1); d2 = d_2D-out (outdoor scenarios) or d_2D (indoor)."""
    if scn == "RMa":
        return _where(d2 <= 10.0, 1.0, _exp(-(d2 - 10.0) / 1000.0))
    if scn in ("UMi", "UMa"):
        d = _max(d2, 18.0)
        dd = 36.0 if scn == "UMi" else 63.0
        p = 18.0 / d + _exp(-d / dd) * (1 - 18.0 / d)
        if scn == "UMa" and h_ut > 13.0:
            c = ((h_ut - 13.0) / 10.0) ** 1.5
            p = p * (1 + c * 5 / 4 * (d / 100.0) ** 3 * _exp(-d / 150.0))
        return _where(d2 <= 18.0, 1.0, p)
    if scn == "InH":
        return _where(d2 <= 1.2, 1.0, _where(d2 < 6.5, _exp(-(d2 - 1.2) / 4.7), _exp(-(d2 - 6.5) / 32.6) * 0.32))
    return _exp(-d2 / k_subsce)


def o2i_wall_db(model, fc):
    """PL_tw (dB) of the low- or high-loss O2I model (Table 7.4.3-2); sigma_P is 4.4 / 6.5 dB."""
    l_concrete = 5 + 4 * fc
    if model == "low":
        l_glass = 2 + 0.2 * fc
        return 5 - 10 * math.log10(0.3 * 10 ** (-l_glass / 10) + 0.7 * 10 ** (-l_concrete / 10))
    l_irr = 23 + 0.3 * fc
    return 5 - 10 * math.log10(0.7 * 10 ** (-l_irr / 10) + 0.3 * 10 ** (-l_concrete / 10))


O2I_SIGMA_DB = {"low": 4.4, "high": 6.5}
