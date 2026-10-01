"""Check the USD -> radio-map pipeline against closed-form channels (needs pxr and sionna-rt).

    python -m isaac_net.tools.scene.validate [--variant llvm] [--samples 4e6] [--out dir]

Three synthetic scenes (tools/scene/synthetic.py, 40 m x 20 m, gNB at (10, 10, 4), robot antenna at 1.5 m, 3.5 GHz,
0.5 m cells) go through export and bake, and the map is compared at fixed points with free-space loss and ITU-R
P.2040 slab transmission (tests/test_scene_bake.py asserts the same comparisons):

    plane   free space plus one concrete wall at x = 20 modelled as a two-sided quad: one slab (0.1 m)
    box     the same wall as a closed 0.2 m box: two faces, two slabs
    arena   floor and four walls around the arena plus the box wall ending at y = 14: reflections add power

Behind the wall the values are power averages over 3 x 3 cells (few rays reach one cell there).
"""
from __future__ import annotations

import argparse
import os
import tempfile

import numpy as np

FC, TX, UE_H = 3.5, (10.0, 10.0, 4.0), 1.5
LOS_PTS = [(4.75, 9.75), (14.75, 9.75), (15.25, 12.25), (6.25, 4.25)]
NLOS_PTS = [(26.25, 9.75), (30.25, 10.25), (34.75, 9.75), (29.75, 12.25)]
SCENES = {"plane": dict(floor=False, arena_walls=False, wall_kind="plane"),
          "box": dict(floor=False, arena_walls=False, wall_kind="box", wall_t=0.2),
          "arena": dict(wall_kind="box", wall_y1=14.0)}


def sample(m, pts, k=1):
    """[(map dB, free-space dB, los_prob)] at the cells nearest to pts, power-averaged over k x k cells."""
    from .materials import free_space_gain_db

    x0, y0, x1, y1 = m["bounds"]
    g = m["gain_db"][0]
    H, W = g.shape
    xs, ys = np.linspace(x0, x1, W), np.linspace(y0, y1, H)
    r = k // 2
    out = []
    for px, py in pts:
        j, i = int(np.argmin(abs(xs - px))), int(np.argmin(abs(ys - py)))
        J, I = np.meshgrid(xs[j - r:j + r + 1], ys[i - r:i + r + 1])
        fs = free_space_gain_db(np.sqrt((J - TX[0]) ** 2 + (I - TX[1]) ** 2 + (UE_H - TX[2]) ** 2), FC)
        v = g[i - r:i + r + 1, j - r:j + r + 1]
        lp = float(m["los_prob"][0, i, j]) if "los_prob" in m else float("nan")
        out.append((10 * np.log10(np.mean(10 ** (v / 10))), 10 * np.log10(np.mean(10 ** (fs / 10))), lp))
    return out


def run(out_dir, variant="llvm", samples=4_000_000):
    from .bake import bake_scene
    from .materials import slab_transmission_db
    from .synthetic import box_arena_usd
    from .usd_export import export_usd

    slab = slab_transmission_db("concrete", 0.1, FC)
    rows = []
    for name, kw in SCENES.items():
        usd = os.path.join(out_dir, f"{name}.usda")
        box_arena_usd(usd, L=40, W=20, H=6, **kw)
        r = export_usd(usd, os.path.join(out_dir, name))
        m = bake_scene(r.xml_path, [TX], FC, (0, 0, 40, 20), cell=0.5, ue_h=UE_H, samples=samples, depth=4,
                       los_map=True, variant=variant, fill=False)
        n_slab = {"plane": 1, "box": 2, "arena": 2}[name]
        for kind, pts, k, loss in (("LOS", LOS_PTS, 1, 0.0), ("behind wall", NLOS_PTS, 3, n_slab * slab)):
            if name == "arena" and kind != "LOS":
                pts, loss = [(30.25, 19.25)], 0.0
                kind = "LOS past wall end"
            for (px, py), (g, fs, lp) in zip(pts, sample(m, pts, k)):
                rows.append((name, kind, px, py, g, fs - loss, g - (fs - loss), lp))
        print(f"{name}: bake {m['bake_s']:.2f} s ({m['radio_map_s']:.2f} s radio map, {m['los_s']:.2f} s LOS map)",
              flush=True)
    print(f"\nslab loss, concrete 0.1 m at {FC} GHz, normal incidence: {slab:.2f} dB\n")
    print("| scene | point | x, y (m) | map (dB) | expected (dB) | map - expected (dB) | LOS prob |")
    print("|:---|:---|:---|---:|---:|---:|---:|")
    for name, kind, px, py, g, ex, d, lp in rows:
        print(f"| {name} | {kind} | {px:.2f}, {py:.2f} | {g:.2f} | {ex:.2f} | {d:+.2f} | {lp:.2f} |")
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", default="llvm")
    ap.add_argument("--samples", type=float, default=4e6)
    ap.add_argument("--out", default=None, help="keep the scenes here (default: a temp dir)")
    a = ap.parse_args(argv)
    if a.variant:
        from .bake import _set_variant
        _set_variant(a.variant)
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        run(a.out, a.variant, int(a.samples))
    else:
        with tempfile.TemporaryDirectory() as d:
            run(d, a.variant, int(a.samples))


if __name__ == "__main__":
    main()
