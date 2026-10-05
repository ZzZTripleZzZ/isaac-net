"""gNB antenna element pattern of TR 38.901 Table 7.3-1 (Sec. 7.3), for radio.RadioMC (NRConfig.gnb_antenna).

    A_V(theta) = -min(12 ((theta - 90 - tilt) / 65)^2, SLA_V)     vertical cut, zenith angle theta (90 = horizon)
    A_H(phi)   = -min(12 (phi / 65)^2, A_max)                      horizontal cut, azimuth phi from boresight
    A(theta, phi) = -min(-(A_V + A_H), A_max)                      3-D pattern
    G(theta, phi) = G_E,max + A(theta, phi)                        dBi

with HPBW 65 deg in both planes, SLA_V = 30 dB, A_max = 30 dB and G_E,max = 8 dBi (NRConfig.gnb_antenna_gain_dbi).
Downtilt (NRConfig.cell_tilt_deg, deg below the horizon) moves the vertical boresight to theta = 90 + tilt; this is
the separable form of the mechanical tilt (exact for links in the boresight's vertical plane, the usual system-level
simplification instead of the full 38.901 Sec. 7.1 rotation). One element per cell, no array gain, UE isotropic.
"""
from __future__ import annotations


import torch

HPBW_DEG = 65.0       # theta_3dB = phi_3dB
SLA_V_DB = 30.0       # vertical side-lobe attenuation
A_MAX_DB = 30.0       # front-back ratio
_CONST = {}


def _const(values, device):
    """Device copy of a per-cell constant, made once per device (no host-to-device copy inside a captured step)."""
    key = (tuple(values), str(device))
    if key not in _CONST:
        _CONST[key] = torch.tensor(values, dtype=torch.float32, device=device)
    return _CONST[key]


def element_gain_db(theta_deg, phi_deg, g_max_dbi=8.0, tilt_deg=0.0):
    """TR 38.901 Table 7.3-1 gain (dBi) at zenith angle theta_deg (90 = horizon) and azimuth phi_deg from the
    boresight (any real value; wrapped to [-180, 180)). Broadcasts; tilt_deg may be a tensor."""
    phi = torch.remainder(phi_deg + 180.0, 360.0) - 180.0
    a_v = -torch.clamp(12 * ((theta_deg - 90.0 - tilt_deg) / HPBW_DEG) ** 2, max=SLA_V_DB)
    a_h = -torch.clamp(12 * (phi / HPBW_DEG) ** 2, max=A_MAX_DB)
    return g_max_dbi - torch.clamp(-(a_v + a_h), max=A_MAX_DB)


def link_angles_deg(pos, gnb3, h_ut):
    """Azimuth (deg, from +x toward +y) and zenith angle (deg, 90 = horizon) of every gNB -> robot link.
    pos [E,R,2], gnb3 [C,3] (x, y, height), h_ut robot antenna height -> ([E,R,C], [E,R,C])."""
    d = pos[:, :, None, :] - gnb3[:, :2]                                        # [E,R,C,2]
    az = torch.rad2deg(torch.atan2(d[..., 1], d[..., 0]))
    d2 = d.norm(dim=-1)
    zen = torch.rad2deg(torch.atan2(d2, h_ut - gnb3[:, 2]))                     # 90 deg when the heights match
    return az, zen


def gnb_antenna_gain_db(cfg, pos, gnb3, h_ut):
    """Gain (dBi) of each cell's gNB antenna toward every robot [E,R,C] (0 for gnb_antenna="isotropic").
    pos [E,R,2]; gnb3 [C,3]; boresights cfg.cell_azimuths(), tilts cfg.cell_tilts()."""
    if cfg.gnb_antenna == "isotropic":
        return torch.zeros(*pos.shape[:2], gnb3.shape[0], device=pos.device)
    az, zen = link_angles_deg(pos, gnb3, h_ut)
    bore, tilt = _const(cfg.cell_azimuths(), pos.device), _const(cfg.cell_tilts(), pos.device)
    return element_gain_db(zen, az - bore, float(cfg.gnb_antenna_gain_dbi), tilt)
