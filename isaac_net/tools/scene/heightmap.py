"""Obstacle height map obstacle_z [H, W] from scene triangles (numpy only), for NRConfig.los_source="raycast".

obstacle_z[i, j] is the largest z of any triangle point inside the grid cell of point (x_j, y_i), the square of one
grid spacing centred on it, clamped below at 0 (the floor). The grid is the radio map's: bounds (x0, y0, x1, y1) are
the first and last grid points. Each triangle is sampled on a barycentric lattice whose spacing is at most
spacing / sub along every edge (3-D lengths, so vertical walls get their top edge), and every sample raises the cell
it falls in. Robots must not be in the triangles (export with --exclude, as for the radio map) and a ceiling must be
dropped (--z-max), or every ray is blocked. Overhangs (a mezzanine above an aisle) count as solid down to the floor:
the 2.5-D limitation of the ray march (docs/obstacles.md).
"""
from __future__ import annotations

import glob
import os
from typing import Iterable, Optional, Sequence

import numpy as np


def height_map(vertices: np.ndarray, faces: np.ndarray, bounds: Sequence[float], H: int, W: int, sub: int = 2,
               z_max: Optional[float] = None, max_points: int = 2_000_000) -> np.ndarray:
    """vertices [V,3], faces [F,3] (metres, Z up, the map frame) -> obstacle_z [H, W] float32."""
    x0, y0, x1, y1 = (float(b) for b in bounds)
    dx, dy = (x1 - x0) / (W - 1), (y1 - y0) / (H - 1)
    step = min(dx, dy) / max(int(sub), 1)
    out = np.zeros(H * W, np.float64)
    v = np.asarray(vertices, np.float64).reshape(-1, 3)
    f = np.asarray(faces, np.int64).reshape(-1, 3)
    if len(f) == 0:
        return out.reshape(H, W).astype(np.float32)
    tri = v[f]                                                           # [F,3,3]
    if z_max is not None:
        tri = tri[~(tri[..., 2] > z_max).all(1)]
    # drop triangles entirely outside the grid footprint
    lo, hi = tri[..., :2].min(1), tri[..., :2].max(1)
    keep = (hi[:, 0] >= x0 - dx / 2) & (lo[:, 0] <= x1 + dx / 2) & (hi[:, 1] >= y0 - dy / 2) & (lo[:, 1] <= y1 + dy / 2)
    tri = tri[keep]
    L = np.stack([np.linalg.norm(tri[:, i] - tri[:, (i + 1) % 3], axis=1) for i in range(3)], 1).max(1)
    m = np.maximum(np.ceil(L / step).astype(np.int64), 1)
    for mm in np.unique(m):
        group = tri[m == mm]
        i, j = np.meshgrid(np.arange(mm + 1), np.arange(mm + 1), indexing="ij")
        sel = (i + j) <= mm
        a, b = i[sel] / mm, j[sel] / mm                                  # barycentric lattice [P]
        w = np.stack([1 - a - b, a, b], 1)                               # [P,3]
        n = max(1, max_points // len(w))
        for s in range(0, len(group), n):
            pts = np.einsum("pk,tkx->tpx", w, group[s:s + n]).reshape(-1, 3)
            ix = np.rint((pts[:, 0] - x0) / dx).astype(np.int64)
            iy = np.rint((pts[:, 1] - y0) / dy).astype(np.int64)
            ok = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
            np.maximum.at(out, iy[ok] * W + ix[ok], pts[ok, 2])
    return np.maximum(out, 0.0).reshape(H, W).astype(np.float32)


def box_triangles(boxes: Iterable[Sequence[float]]) -> tuple:
    """Axis-aligned boxes (x0, y0, z0, x1, y1, z1) -> (vertices [8B,3], faces [12B,3]), for synthetic scenes."""
    vs, fs = [], []
    quads = ((0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3))
    for k, (ax, ay, az, bx, by, bz) in enumerate(boxes):
        c = np.array([[x, y, z] for x in (ax, bx) for y in (ay, by) for z in (az, bz)], np.float64)
        vs.append(c)
        for q in quads:
            fs += [(8 * k + q[0], 8 * k + q[1], 8 * k + q[2]), (8 * k + q[0], 8 * k + q[2], 8 * k + q[3])]
    return np.concatenate(vs), np.asarray(fs, np.int64)


def scene_triangles(scene_dir: str) -> tuple:
    """(vertices, faces) of every mesh of an exported scene directory (usd_export / mitsuba_writer layout)."""
    from .mitsuba_writer import read_ply

    vs, fs, off = [], [], 0
    for p in sorted(glob.glob(os.path.join(scene_dir, "meshes", "*.ply"))):
        v, f = read_ply(p)
        vs.append(v)
        fs.append(f.astype(np.int64) + off)
        off += len(v)
    if not vs:
        raise FileNotFoundError(f"no meshes/*.ply under {scene_dir}")
    return np.concatenate(vs), np.concatenate(fs)
