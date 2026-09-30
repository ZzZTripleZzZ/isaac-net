"""Bake a radio map (channel="radio_map") with Sionna RT's RadioMapSolver.

Needs sionna-rt (>= 1.0; tested with sionna-rt 2.2.0, Mitsuba 3.9.1, Dr.Jit 1.5.0 in a side venv:
`pip install sionna-rt`). It is not a dependency of isaaclab_net and imports only numpy and sionna-rt, so run it
as a script (the path below) in an environment without torch. The output is an .npz for channels.RadioMap:
gain_db [C, H, W] (path gain per gNB at UE height, dB), bounds (x0, y0, x1, y1) = the outermost cell centres, gnb_xy.

Scene: --scene PATH loads a Mitsuba XML scene (e.g. one exported from Blender after a USD import; see
docs/channels.md). Without it the tool writes a simple warehouse: a concrete floor and four concrete walls around
the arena and a row of metal shelves (boxes) across the middle, no ceiling.

    python isaaclab_net/tools/bake_radio_map_sionna.py --out warehouse.npz --arena 60 40 \\
        --gnb 10 20 6 --gnb 50 20 6 --fc 3.5 --cell 1.0 --samples 2000000 --depth 4

Use: NRConfig(channel="radio_map", radio_map_path="warehouse.npz", n_cells=2,
cell_positions_m=((10, 20), (50, 20))). The env-local frame of the robots must be the scene frame.
"""
from __future__ import annotations

import argparse
import os
import tempfile

import numpy as np

WAREHOUSE_XML = """<scene version="2.1.0">
  <bsdf type="itu-radio-material" id="concrete"><string name="type" value="concrete"/>
    <float name="thickness" value="0.2"/></bsdf>
  <bsdf type="itu-radio-material" id="metal"><string name="type" value="metal"/>
    <float name="thickness" value="0.01"/></bsdf>
{shapes}
</scene>
"""


def _cube(name, mat, cx, cy, cz, sx, sy, sz):
    """Axis-aligned box centred at (cx, cy, cz) with full sizes (sx, sy, sz) (Mitsuba cube = [-1, 1]^3)."""
    return (f'  <shape type="cube" id="{name}"><transform name="to_world">'
            f'<scale x="{sx / 2}" y="{sy / 2}" z="{sz / 2}"/><translate x="{cx}" y="{cy}" z="{cz}"/></transform>'
            f'<ref id="{mat}" name="bsdf"/></shape>')


def warehouse_xml(L, W, wall_h=8.0, shelf_h=4.0, n_shelves=3, t=0.2):
    s = [_cube("floor", "concrete", L / 2, W / 2, -t / 2, L + 2 * t, W + 2 * t, t),
         _cube("wall_s", "concrete", L / 2, -t / 2, wall_h / 2, L + 2 * t, t, wall_h),
         _cube("wall_n", "concrete", L / 2, W + t / 2, wall_h / 2, L + 2 * t, t, wall_h),
         _cube("wall_w", "concrete", -t / 2, W / 2, wall_h / 2, t, W, wall_h),
         _cube("wall_e", "concrete", L + t / 2, W / 2, wall_h / 2, t, W, wall_h)]
    # shelves: a row across the middle (x = L/2), 1 m deep, with gaps between them
    seg = W / (2 * n_shelves + 1)
    for i in range(n_shelves):
        cy = seg * (2 * i + 1.5)
        s.append(_cube(f"shelf{i}", "metal", L / 2, cy, shelf_h / 2, 1.0, seg, shelf_h))
    return WAREHOUSE_XML.format(shapes="\n".join(s))


def bake(scene_xml, gnbs, fc_ghz, bounds, cell, ue_h, samples, depth, floor_db=-200.0, variant=None):
    if variant:
        import mitsuba as mi
        mi.set_variant({"cuda": "cuda_ad_mono_polarized", "llvm": "llvm_ad_mono_polarized"}.get(variant, variant))
    import sionna.rt as rt

    scene = rt.load_scene(scene_xml)
    scene.frequency = fc_ghz * 1e9
    arr = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
    scene.tx_array, scene.rx_array = arr, arr
    for i, (x, y, z) in enumerate(gnbs):
        scene.add(rt.Transmitter(name=f"gnb{i}", position=[x, y, z]))
    x0, y0, x1, y1 = bounds
    rm = rt.RadioMapSolver()(scene, center=[(x0 + x1) / 2, (y0 + y1) / 2, ue_h], orientation=[0, 0, 0],
                             size=[x1 - x0, y1 - y0], cell_size=[cell, cell], samples_per_tx=samples,
                             max_depth=depth)
    pg = np.asarray(rm.path_gain.numpy() if hasattr(rm.path_gain, "numpy") else rm.path_gain)   # [C, ny, nx]
    cc = np.asarray(rm.cell_centers.numpy() if hasattr(rm.cell_centers, "numpy") else rm.cell_centers)
    # orient rows along +y and columns along +x
    if cc[0, -1, 0] < cc[0, 0, 0]:
        pg, cc = pg[:, :, ::-1], cc[:, ::-1]
    if cc[-1, 0, 1] < cc[0, 0, 1]:
        pg, cc = pg[:, ::-1, :], cc[::-1]
    with np.errstate(divide="ignore"):
        gain = np.maximum(10 * np.log10(pg), floor_db).astype(np.float32)
    b = (float(cc[0, 0, 0]), float(cc[0, 0, 1]), float(cc[-1, -1, 0]), float(cc[-1, -1, 1]))
    return gain, b


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scene", default=None, help="Mitsuba XML scene (default: generated warehouse)")
    ap.add_argument("--arena", type=float, nargs=2, default=(60.0, 40.0), metavar=("L", "W"))
    ap.add_argument("--gnb", type=float, nargs=3, action="append", metavar=("X", "Y", "Z"))
    ap.add_argument("--fc", type=float, default=3.5, help="carrier (GHz)")
    ap.add_argument("--cell", type=float, default=1.0, help="map cell size (m)")
    ap.add_argument("--ue-height", type=float, default=1.5)
    ap.add_argument("--samples", type=int, default=1_000_000)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--variant", default=None, help="Mitsuba variant: cuda, llvm (CPU) or a full name; default: "
                    "Sionna's choice (CUDA needs OptiX, which WSL 2 does not expose by default: use llvm there)")
    a = ap.parse_args(argv)
    gnbs = a.gnb or [(a.arena[0] / 4, a.arena[1] / 2, 6.0), (3 * a.arena[0] / 4, a.arena[1] / 2, 6.0)]
    L, W = a.arena
    scene = a.scene
    tmp = None
    if scene is None:
        tmp = tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False)
        tmp.write(warehouse_xml(L, W))
        tmp.close()
        scene = tmp.name
    try:
        gain, bounds = bake(scene, gnbs, a.fc, (0.0, 0.0, L, W), a.cell, a.ue_height, a.samples, a.depth,
                             variant=a.variant)
    finally:
        if tmp is not None:
            os.unlink(tmp.name)
    np.savez_compressed(a.out, gain_db=gain, bounds=np.asarray(bounds), gnb_xy=np.asarray([g[:2] for g in gnbs]),
                        fc_ghz=np.asarray(a.fc), ue_height_m=np.asarray(a.ue_height),
                        source=np.asarray("sionna-rt RadioMapSolver, " + ("generated warehouse" if a.scene is None
                                                                          else os.path.basename(a.scene))))
    print(f"wrote {a.out}: gain_db {gain.shape}, bounds {bounds}, range {gain.min():.1f} .. {gain.max():.1f} dB")


if __name__ == "__main__":
    main()
