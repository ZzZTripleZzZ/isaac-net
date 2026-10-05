"""Synthetic radio maps for tests and examples (numpy only).

Default: regenerate the tiny map shipped in core/data/radio_map_synthetic.npz: two gNBs at (25, 75) and (125, 75) m
in a 150 m arena, log-distance 40 + 35 log10(d), a 10 dB wall at x = 75 m, 16 x 16 grid. Use it with
NRConfig(channel="radio_map", radio_map_path="synthetic", n_cells=2, cell_positions_m=((25, 75), (125, 75))).

--obstacles: a warehouse hall with racks (make_obstacle_map) that also carries los_prob [C,H,W] (exact segment-box
tests on k x k points per cell, as bake.py --los-map does with Mitsuba) and obstacle_z [H,W] (the racks' triangles
through tools/scene/heightmap.py, as bake.py --obstacle-z), for NRConfig.los_source="map" / "raycast" without USD
or Sionna:

    python -m isaac_net.tools.make_synthetic_radio_map [out.npz]
    python -m isaac_net.tools.make_synthetic_radio_map --obstacles out.npz
"""
import sys

import numpy as np

from isaac_net.core.channels.radio_map import SYNTHETIC_MAP, RadioMap, make_synthetic_map, synthetic_gnb_xy
from isaac_net.tools.scene.heightmap import box_triangles, height_map

# hall 40 x 24 m: four rows of racks (1 m deep, 2.5 m tall) with aisles, gNBs on the short walls at 6 m
OBSTACLE_GNBS = ((2.0, 12.0, 6.0), (38.0, 12.0, 6.0))
OBSTACLE_RACKS = tuple((x0, y0, 0.0, x0 + 24.0, y0 + 1.0, 2.5) for x0 in (8.0,) for y0 in (4.0, 9.0, 14.0, 19.0))


def segment_box_blocked(p0, p1, boxes):
    """Exact test: does the open segment p0 -> p1 ([N,3] each) cross any box (x0, y0, z0, x1, y1, z1)? -> [N]."""
    p0, p1 = np.asarray(p0, np.float64), np.asarray(p1, np.float64)
    d = p1 - p0
    hit = np.zeros(len(p0), bool)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / d
    for b in boxes:
        lo, hi = np.asarray(b[:3], np.float64), np.asarray(b[3:], np.float64)
        t1, t2 = (lo - p0) * inv, (hi - p0) * inv
        par = d == 0                                           # parallel to a slab: inside it or never
        inside = (p0 >= lo) & (p0 <= hi)
        tmin = np.where(par, np.where(inside, -np.inf, np.inf), np.minimum(t1, t2)).max(1)
        tmax = np.where(par, np.where(inside, np.inf, -np.inf), np.maximum(t1, t2)).min(1)
        hit |= (tmax > np.maximum(tmin, 1e-9)) & (tmin < 1 - 1e-9)
    return hit


def make_obstacle_map(gnbs=OBSTACLE_GNBS, racks=OBSTACLE_RACKS, bounds=(0.0, 0.0, 40.0, 24.0), cell=0.25,
                      ue_h=1.5, sub=3, pl_const_db=40.0, pl_exp=3.0, nlos_loss_db=15.0):
    """RadioMap with gain_db, los_prob and obstacle_z of a hall with box racks. gain = -(pl_const + 10 n log10 d)
    - nlos_loss_db (1 - los_prob): a stand-in for a ray-traced map whose NLOS loss is already in the map."""
    x0, y0, x1, y1 = bounds
    W = int(round((x1 - x0) / cell)) + 1
    H = int(round((y1 - y0) / cell)) + 1
    xs, ys = np.linspace(x0, x1, W), np.linspace(y0, y1, H)
    X, Y = np.meshgrid(xs, ys)
    off = ((np.arange(sub) + 0.5) / sub - 0.5) * cell
    ox, oy = np.meshgrid(off, off)
    px = (X[..., None] + ox.reshape(-1)).reshape(-1)
    py = (Y[..., None] + oy.reshape(-1)).reshape(-1)
    pts = np.stack([px, py, np.full_like(px, ue_h)], 1)
    gains, losp = [], []
    for g in gnbs:
        blk = segment_box_blocked(pts, np.broadcast_to(np.asarray(g, np.float64), pts.shape), racks)
        lp = 1.0 - blk.reshape(H, W, sub * sub).mean(-1)
        d = np.maximum(np.sqrt((X - g[0]) ** 2 + (Y - g[1]) ** 2 + (g[2] - ue_h) ** 2), 1.0)
        gains.append(-(pl_const_db + 10 * pl_exp * np.log10(d)) - nlos_loss_db * (1 - lp))
        losp.append(lp)
    v, f = box_triangles(racks)
    oz = height_map(v, f, bounds, H, W)
    meta = {"gnb_xy": np.asarray([g[:2] for g in gnbs], np.float64), "gnb_z": np.asarray([g[2] for g in gnbs]),
            "ue_height_m": float(ue_h), "source": "synthetic hall with %d box racks" % len(racks)}
    return RadioMap(np.stack(gains).astype(np.float32), bounds, meta=meta, los_prob=np.stack(losp).astype(np.float32),
                    obstacle_z=oz)


def main(*argv):
    args = list(argv)
    obstacles = "--obstacles" in args
    args = [a for a in args if a != "--obstacles"]
    if obstacles:
        if not args:
            raise SystemExit("--obstacles needs an output path")
        m = make_obstacle_map()
        out = args[0]
    else:
        m = make_synthetic_map(synthetic_gnb_xy())
        out = args[0] if args else SYNTHETIC_MAP
    m.save(out)
    print(f"wrote {out}: {m.C} cells, {m.H} x {m.W}, bounds {m.bounds}")


if __name__ == "__main__":
    main(*sys.argv[1:])
