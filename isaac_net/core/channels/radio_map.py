"""Precomputed radio maps: path gain per cell on a regular grid, sampled bilinearly at robot positions.

File format (.npz, or .pt holding the same keys):
  gain_db  float [C, H, W]  path gain in dB (negative: received power minus transmit power) between cell c and a
                            UT at grid point (x_j, y_i), antenna gains included if the source had them. Reciprocal,
                            so it serves UL and DL. Row i runs along y, column j along x.
  bounds   float [4]        (x0, y0, x1, y1) env-local metres: x_j = x0 + j (x1 - x0) / (W - 1), y_i likewise, so the
                            grid points include the arena corners.
  los_prob    float [C, H, W]  optional: share of LOS points per grid cell (tools/scene/bake.py --los-map); kept as
                               RadioMap.los_prob, read by NRConfig.los_source="map" (channels/los.py)
  obstacle_z  float [H, W]     optional: obstacle top height above the floor (m, 0 = free floor) per grid cell, the
                               max triangle z of the static scene over the cell (bake.py --obstacle-z); kept as
                               RadioMap.obstacle_z, read by los_source="raycast"
  optional metadata: fc_ghz, ue_height_m, gnb_xy, gnb_z, diffraction (bool), source (str).
Outside the bounds the map is clamped to its border. Sampling is in dB, bilinear, fixed-shape and graph-safe; los_prob
and obstacle_z are sampled the same way. Files without los_prob / obstacle_z load as before (both None).

Sources of maps: tools/bake_radio_map_sionna.py (Sionna RT RadioMapSolver on a scene), measurements on a grid, or
make_synthetic_map() below (the tiny map shipped in core/data/radio_map_synthetic.npz).
"""
from __future__ import annotations

import os

import numpy as np
import torch

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
SYNTHETIC_MAP = os.path.join(DATA_DIR, "radio_map_synthetic.npz")


GRIDS = ("los_prob", "obstacle_z")        # optional grids kept as attributes (not metadata)


class RadioMap:
    def __init__(self, gain_db, bounds, device="cpu", meta=None, los_prob=None, obstacle_z=None):
        g = torch.as_tensor(gain_db, dtype=torch.float32)
        if g.dim() == 2:
            g = g[None]
        assert g.dim() == 3 and g.shape[1] >= 2 and g.shape[2] >= 2, "gain_db must be [C, H, W] with H, W >= 2"
        self.C, self.H, self.W = g.shape
        x0, y0, x1, y1 = (float(b) for b in bounds)
        assert x1 > x0 and y1 > y0, "bounds = (x0, y0, x1, y1) with x1 > x0, y1 > y0"
        self.bounds = (x0, y0, x1, y1)
        self.meta = dict(meta or {})
        self.gain = g.to(device).reshape(self.C, self.H * self.W).contiguous()
        self.los_prob = self.obstacle_z = None
        if los_prob is not None:
            lp = torch.as_tensor(los_prob, dtype=torch.float32)
            lp = lp[None] if lp.dim() == 2 else lp
            assert lp.shape == g.shape, f"los_prob must be [C, H, W] = {tuple(g.shape)}, got {tuple(lp.shape)}"
            self.los_prob = lp.to(device).reshape(self.C, self.H * self.W).contiguous()
        if obstacle_z is not None:
            oz = torch.as_tensor(obstacle_z, dtype=torch.float32)
            assert tuple(oz.shape) == (self.H, self.W), f"obstacle_z must be [H, W] = {(self.H, self.W)}"
            self.obstacle_z = oz.to(device).reshape(1, self.H * self.W).contiguous()

    def _grid(self, x):
        return None if x is None else x.view(-1, self.H, self.W)

    def to(self, device):
        oz = None if self.obstacle_z is None else self.obstacle_z.view(self.H, self.W)
        return RadioMap(self.gain.view(self.C, self.H, self.W), self.bounds, device, self.meta,
                        los_prob=self._grid(self.los_prob), obstacle_z=oz)

    @classmethod
    def load(cls, path, device="cpu"):
        if path.endswith(".pt"):
            d = torch.load(path, map_location="cpu")
            d = {k: (v.numpy() if isinstance(v, torch.Tensor) else v) for k, v in d.items()}
        else:
            with np.load(path, allow_pickle=False) as z:
                d = {k: z[k] for k in z.files}
        meta = {k: (v.item() if hasattr(v, "item") and np.ndim(v) == 0 else v) for k, v in d.items()
                if k not in ("gain_db", "bounds") + GRIDS}
        grids = {k: np.asarray(d[k], dtype=np.float32) for k in GRIDS if k in d}
        return cls(d["gain_db"], np.asarray(d["bounds"]).reshape(4), device, meta, **grids)

    def save(self, path):
        meta = {k: np.asarray(v) for k, v in self.meta.items()}
        grids = {}
        if self.los_prob is not None:
            grids["los_prob"] = self.los_prob.view(self.C, self.H, self.W).cpu().numpy()
        if self.obstacle_z is not None:
            grids["obstacle_z"] = self.obstacle_z.view(self.H, self.W).cpu().numpy()
        np.savez_compressed(path, gain_db=self.gain.view(self.C, self.H, self.W).cpu().numpy(),
                            bounds=np.asarray(self.bounds, dtype=np.float64), **meta, **grids)

    def sample(self, pos):
        """pos [E,R,2|3] (z ignored) -> path gain [E,R,C] dB, bilinear in dB, clamped at the border."""
        return self._bilinear(self.gain, pos)

    def sample_los(self, pos):
        """pos [E,R,2|3] -> baked LOS share [E,R,C] (bilinear; needs los_prob)."""
        return self._bilinear(self.los_prob, pos)

    def sample_height(self, xy):
        """xy [..., 2] -> obstacle top height [...] in m (bilinear; needs obstacle_z)."""
        return self._bilinear(self.obstacle_z, xy)[..., 0]

    def _bilinear(self, g, pos):
        """g [K, H*W] -> [*pos.shape[:-1], K], bilinear, clamped at the border (the op order of sample())."""
        x0, y0, x1, y1 = self.bounds
        W, H = self.W, self.H
        fx = ((pos[..., 0] - x0) * ((W - 1) / (x1 - x0))).clamp(0, W - 1)
        fy = ((pos[..., 1] - y0) * ((H - 1) / (y1 - y0))).clamp(0, H - 1)
        ix = fx.floor().clamp(max=W - 2)
        iy = fy.floor().clamp(max=H - 2)
        tx, ty = fx - ix, fy - iy
        i00 = (iy.long() * W + ix.long()).reshape(-1)
        v00, v01 = g[:, i00], g[:, i00 + 1]
        v10, v11 = g[:, i00 + W], g[:, i00 + W + 1]
        tx, ty = tx.reshape(-1), ty.reshape(-1)
        out = (v00 * (1 - tx) + v01 * tx) * (1 - ty) + (v10 * (1 - tx) + v11 * tx) * ty      # [C, E*R]
        return out.t().reshape(*pos.shape[:-1], g.shape[0])


def make_synthetic_map(gnb_xy, bounds=(0.0, 0.0, 150.0, 150.0), H=16, W=16, pl_const_db=40.0, pl_exp=3.5,
                       wall_x=75.0, wall_loss_db=10.0):
    """Tiny deterministic map for tests and examples: log-distance path loss from each gNB, plus a wall at
    x = wall_x that costs wall_loss_db on links that cross it."""
    x0, y0, x1, y1 = bounds
    xs = np.linspace(x0, x1, W)
    ys = np.linspace(y0, y1, H)
    X, Y = np.meshgrid(xs, ys)                     # [H, W], row = y
    maps = []
    for gx, gy in gnb_xy:
        d = np.maximum(np.hypot(X - gx, Y - gy), 1.0)
        pg = -(pl_const_db + 10 * pl_exp * np.log10(d))
        cross = (X - wall_x) * (gx - wall_x) < 0
        maps.append(pg - wall_loss_db * cross)
    meta = {"gnb_xy": np.asarray(gnb_xy, dtype=np.float64), "source": "synthetic: log-distance %.1f + %.1f log10(d), %.1f dB wall at x = %.1f m"
            % (pl_const_db, 10 * pl_exp, wall_loss_db, wall_x)}
    return RadioMap(np.stack(maps).astype(np.float32), bounds, meta=meta)


def synthetic_gnb_xy():
    """The two gNBs of the shipped synthetic map (one on each side of its wall)."""
    return [(25.0, 75.0), (125.0, 75.0)]

