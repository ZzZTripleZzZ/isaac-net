"""Precomputed radio maps: path gain per cell on a regular grid, sampled bilinearly at robot positions.

File format (.npz, or .pt holding the same keys):
  gain_db  float [C, H, W]  path gain in dB (negative: received power minus transmit power) between cell c and a
                            UT at grid point (x_j, y_i), antenna gains included if the source had them. Reciprocal,
                            so it serves UL and DL. Row i runs along y, column j along x.
  bounds   float [4]        (x0, y0, x1, y1) env-local metres: x_j = x0 + j (x1 - x0) / (W - 1), y_i likewise, so the
                            grid points include the arena corners.
  optional metadata: fc_ghz, ue_height_m, source (str).
Outside the bounds the map is clamped to its border. Sampling is in dB, bilinear, fixed-shape and graph-safe.

Sources of maps: tools/bake_radio_map_sionna.py (Sionna RT RadioMapSolver on a scene), measurements on a grid, or
make_synthetic_map() below (the tiny map shipped in core/data/radio_map_synthetic.npz).
"""
from __future__ import annotations

import os

import numpy as np
import torch

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
SYNTHETIC_MAP = os.path.join(DATA_DIR, "radio_map_synthetic.npz")


class RadioMap:
    def __init__(self, gain_db, bounds, device="cpu", meta=None):
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

    def to(self, device):
        return RadioMap(self.gain.view(self.C, self.H, self.W), self.bounds, device, self.meta)

    @classmethod
    def load(cls, path, device="cpu"):
        if path.endswith(".pt"):
            d = torch.load(path, map_location="cpu")
            d = {k: (v.numpy() if isinstance(v, torch.Tensor) else v) for k, v in d.items()}
        else:
            with np.load(path, allow_pickle=False) as z:
                d = {k: z[k] for k in z.files}
        meta = {k: (v.item() if hasattr(v, "item") and np.ndim(v) == 0 else v) for k, v in d.items()
                if k not in ("gain_db", "bounds")}
        return cls(d["gain_db"], np.asarray(d["bounds"]).reshape(4), device, meta)

    def save(self, path):
        meta = {k: np.asarray(v) for k, v in self.meta.items()}
        np.savez_compressed(path, gain_db=self.gain.view(self.C, self.H, self.W).cpu().numpy(),
                            bounds=np.asarray(self.bounds, dtype=np.float64), **meta)

    def sample(self, pos):
        """pos [E,R,2|3] (z ignored) -> path gain [E,R,C] dB, bilinear in dB, clamped at the border."""
        x0, y0, x1, y1 = self.bounds
        W, H = self.W, self.H
        fx = ((pos[..., 0] - x0) * ((W - 1) / (x1 - x0))).clamp(0, W - 1)
        fy = ((pos[..., 1] - y0) * ((H - 1) / (y1 - y0))).clamp(0, H - 1)
        ix = fx.floor().clamp(max=W - 2)
        iy = fy.floor().clamp(max=H - 2)
        tx, ty = fx - ix, fy - iy
        i00 = (iy.long() * W + ix.long()).reshape(-1)
        g = self.gain
        v00, v01 = g[:, i00], g[:, i00 + 1]
        v10, v11 = g[:, i00 + W], g[:, i00 + W + 1]
        tx, ty = tx.reshape(-1), ty.reshape(-1)
        out = (v00 * (1 - tx) + v01 * tx) * (1 - ty) + (v10 * (1 - tx) + v11 * tx) * ty      # [C, E*R]
        return out.t().reshape(*pos.shape[:-1], self.C)


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

