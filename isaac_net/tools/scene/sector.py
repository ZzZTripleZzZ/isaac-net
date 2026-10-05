"""gNB sector antennas at bake time: the TR 38.901 Table 7.3-1 element pattern (core/channels/antenna.py) applied to
the isotropic path-gain maps of the bake (bake.py --gnb-antenna sector) or of the synthetic map tool.

The bake traces isotropic gNB antennas. This adds, per cell c and grid point (x_j, y_i), the element gain toward the
direction of that point seen from the gNB antenna (azimuth from the cell's boresight, zenith angle from the antenna
height gnb_z[c] to the robot antenna height), the same angles and gain the engine computes at run time for
NRConfig(gnb_antenna="sector") (radio.RadioMC), evaluated once per grid point instead of once per robot and step.

The pattern multiplies the whole path gain of a cell: exact for links whose power arrives along the direct path
(line of sight, or a wall loss on that path), an approximation for links dominated by reflections that leave the gNB
in other directions (tracing each ray with the pattern needs Sionna RT's own antenna pattern, not done here).

The map metadata records the pattern (gnb_antenna = "sector", cell_azimuth_deg, cell_tilt_deg,
gnb_antenna_gain_dbi), and channels.models.RadioMapChannel refuses NRConfig(gnb_antenna="sector") with such a map, so
the pattern is never applied twice: use gnb_antenna="isotropic" (the default) with a sector-baked map.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

SECTOR_META = ("gnb_antenna", "cell_azimuth_deg", "cell_tilt_deg", "gnb_antenna_gain_dbi")


def _per_cell(v, C, name):
    a = np.atleast_1d(np.asarray(v, np.float64))
    if a.size == 1:
        a = np.full(C, float(a[0]))
    if a.shape != (C,):
        raise ValueError(f"{name}: one value or one per cell ({C}), got {a.size}")
    return a


def sector_gain_db(shape, bounds, gnb_xy, azimuths, tilts=0.0, g_max_dbi=8.0, gnb_z=None,
                   ue_h=1.5) -> np.ndarray:
    """Element gain (dBi) [C, H, W] of each cell's sector antenna toward every grid point of a map of shape
    (C, H, W) and bounds (x0, y0, x1, y1) (the first and last grid points, as channels.RadioMap). gnb_xy [C, 2];
    gnb_z [C] antenna heights (None: ue_h, the 2-D geometry the engine uses without gnb_height_m); azimuths and
    tilts (deg) one value or one per cell, as NRConfig.cell_azimuth_deg / cell_tilt_deg."""
    import torch

    from isaac_net.core.channels.antenna import element_gain_db, link_angles_deg

    C, H, W = (int(s) for s in shape)
    xy = np.asarray(gnb_xy, np.float64).reshape(-1, 2)
    if xy.shape[0] != C:
        raise ValueError(f"gnb_xy has {xy.shape[0]} cells, the map {C}")
    z = np.full(C, float(ue_h)) if gnb_z is None else _per_cell(gnb_z, C, "gnb_z")
    az, tl = _per_cell(azimuths, C, "cell azimuths"), _per_cell(tilts, C, "cell tilts")
    x0, y0, x1, y1 = (float(b) for b in bounds)
    X, Y = np.meshgrid(np.linspace(x0, x1, W), np.linspace(y0, y1, H))            # [H, W], row = y
    pos = torch.from_numpy(np.stack([X, Y], -1))                                   # [H, W, 2]
    gnb3 = torch.from_numpy(np.concatenate([xy, z[:, None]], 1))                   # [C, 3]
    a, zen = link_angles_deg(pos, gnb3, float(ue_h))                               # [H, W, C]
    g = element_gain_db(zen, a - torch.from_numpy(az), float(g_max_dbi), torch.from_numpy(tl))
    return g.permute(2, 0, 1).numpy()


def apply_sector_pattern(m: dict, azimuths: Sequence[float], tilts=0.0, g_max_dbi: float = 8.0,
                         gnb_z: Optional[Sequence[float]] = None, ue_h: Optional[float] = None) -> dict:
    """A copy of the map dict m (bake.py: gain_db [C,H,W], bounds, gnb_xy, gnb_z, ue_height_m) with the sector
    pattern added to gain_db and recorded in the metadata. Refuses a map that already carries a pattern."""
    if str(m.get("gnb_antenna", "isotropic")) == "sector":
        raise ValueError("this map already has the sector pattern (metadata gnb_antenna='sector')")
    if "gnb_xy" not in m:
        raise ValueError("the sector pattern needs the gNB positions (gnb_xy) in the map")
    g = np.asarray(m["gain_db"])
    C = g.shape[0]
    z = gnb_z if gnb_z is not None else m.get("gnb_z")
    h = float(ue_h if ue_h is not None else m.get("ue_height_m", 1.5))
    az, tl = _per_cell(azimuths, C, "cell azimuths"), _per_cell(tilts, C, "cell tilts")
    out = dict(m)
    out["gain_db"] = (g + sector_gain_db(g.shape, m["bounds"], m["gnb_xy"], az, tl, g_max_dbi, z, h)).astype(g.dtype)
    if z is not None:
        out["gnb_z"] = np.asarray(_per_cell(z, C, "gnb_z"))
    out["ue_height_m"] = h
    out.update(gnb_antenna="sector", cell_azimuth_deg=az, cell_tilt_deg=tl, gnb_antenna_gain_dbi=float(g_max_dbi))
    return out


def sector_radio_map(rmap, azimuths, tilts=0.0, g_max_dbi=8.0, gnb_z=None, ue_h=None):
    """apply_sector_pattern on a channels.RadioMap (e.g. make_synthetic_map's); returns a new RadioMap."""
    from isaac_net.core.channels.radio_map import RadioMap

    m = dict(rmap.meta)
    m["gain_db"] = rmap.gain.view(rmap.C, rmap.H, rmap.W).cpu().numpy()
    m["bounds"] = np.asarray(rmap.bounds, np.float64)
    out = apply_sector_pattern(m, azimuths, tilts, g_max_dbi, gnb_z, ue_h)
    meta = {k: v for k, v in out.items() if k not in ("gain_db", "bounds")}
    grid = lambda x, *s: None if x is None else x.view(*s).cpu().numpy()    # noqa: E731
    return RadioMap(out["gain_db"], out["bounds"], meta=meta, los_prob=grid(rmap.los_prob, rmap.C, rmap.H, rmap.W),
                    obstacle_z=grid(rmap.obstacle_z, rmap.H, rmap.W))


def add_sector_args(ap):
    """--gnb-antenna / --cell-azimuth / --cell-tilt / --gnb-antenna-gain on an argparse parser."""
    g = ap.add_argument_group("gNB antenna (TR 38.901 sector pattern applied at bake time)")
    g.add_argument("--gnb-antenna", choices=("isotropic", "sector"), default="isotropic",
                   help="sector: add the TR 38.901 Table 7.3-1 pattern to the map (then run the engine with "
                        "gnb_antenna='isotropic')")
    g.add_argument("--cell-azimuth", type=float, nargs="+", metavar="DEG",
                   help="boresight azimuth per cell (deg, from +x toward +y); default 30, 150, 270 cycled "
                        "(NRConfig.cell_azimuths)")
    g.add_argument("--cell-tilt", type=float, nargs="+", default=[0.0], metavar="DEG",
                   help="downtilt (deg below the horizon), one value or one per cell")
    g.add_argument("--gnb-antenna-gain", type=float, default=8.0, help="maximum element gain G_E,max (dBi)")
    return g


def sector_from_args(a, n_cells):
    """(azimuths, tilts, g_max) from add_sector_args' options, or None for --gnb-antenna isotropic."""
    if a.gnb_antenna != "sector":
        if a.cell_azimuth is not None:
            raise SystemExit("--cell-azimuth needs --gnb-antenna sector")
        return None
    az = a.cell_azimuth if a.cell_azimuth is not None else [(30.0 + 120.0 * c) % 360.0 for c in range(n_cells)]
    if len(az) != n_cells:
        raise SystemExit(f"--cell-azimuth needs one azimuth per cell ({n_cells})")
    return list(az), list(a.cell_tilt), float(a.gnb_antenna_gain)
