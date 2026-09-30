"""Bake a radio map (NRConfig.channel="radio_map") from a USD stage or a Sionna RT scene with Sionna RT.

    python -m isaaclab_net.tools.scene.bake --usd scene.usd --tx 10 20 6 --tx 50 20 6 --fc 3.5 --cell 1.0 \\
        --out map.pt [--los-map] [--variant llvm]

Steps: export the stage (usd_export: meshes, world transforms, ITU materials, map frame), run Sionna RT's
RadioMapSolver for every transmitter on a horizontal grid at the robot antenna height, and write the file that
channels.RadioMap loads:

    gain_db   [C, H, W]  path gain in dB between gNB c and a robot antenna at grid point (x_j, y_i), isotropic
                         antennas (0 dBi), vertical polarization; row i along y, column j along x
    bounds    [4]        (x0, y0, x1, y1): the first and last grid points (cell centres), map frame, metres
    gnb_xy    [C, 2], gnb_z [C]
    los_prob  [C, H, W]  optional (--los-map): share of the cell (k x k points at the antenna height, --los-sub k)
                         with an unobstructed straight segment to the gNB antenna
    valid     [H, W]     cells that received ray energy from at least one gNB. Cells a gNB's rays never reached
                         (inside solid obstacles, or beyond max_depth interactions) are filled per gNB from their
                         neighbours (--no-fill keeps the floor value, -250 dB)
    metadata  fc_ghz, ue_height_m, cell_m, samples, max_depth, scene_hash, bake_key, materials (JSON), bake_s, source

A .pt output holds torch tensors (torch is then needed, and RadioMap.load reads it with weights_only); any other
suffix writes an .npz, which needs numpy only.

Units and frame: metres, Z up, the map frame of usd_export (--frame, --offset). Transmitter positions are in the same
frame. The solver averages the path gain over each cell; cells next to a transmitter therefore read below the
free-space gain at the cell centre.

Sionna RT (pip install sionna-rt; tested with 2.2.0, Mitsuba 3.9.1, Dr.Jit 1.5.0) runs on CUDA with OptiX or on the
CPU (--variant llvm). WSL 2 does not expose OptiX by default: use --variant llvm there.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from typing import Optional, Sequence

import numpy as np

BAKE_VERSION = 2          # bump when the bake's output for identical inputs changes


def bake_key(scene_hash: str, tx, fc_ghz, bounds, cell, ue_h, samples, depth, refraction=True, diffraction=False,
             los_map=False, los_sub=3, seed=42) -> str:
    """Cache key of a bake: the scene hash and every parameter that changes the map."""
    d = dict(v=BAKE_VERSION, scene=scene_hash, tx=[[round(float(x), 4) for x in t] for t in tx],
             fc=round(float(fc_ghz), 6), bounds=None if bounds is None else [round(float(b), 4) for b in bounds],
             cell=round(float(cell), 4), ue_h=round(float(ue_h), 4), samples=int(samples), depth=int(depth),
             refraction=bool(refraction), diffraction=bool(diffraction), los=bool(los_map), los_sub=int(los_sub),
             seed=int(seed))
    return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()


def fill_holes(gain_db: np.ndarray, valid: np.ndarray, max_iter: Optional[int] = None) -> np.ndarray:
    """Fill invalid cells of [C, H, W] with the mean of their valid 4-neighbours, growing inward until none is left."""
    g = gain_db.copy()
    v = valid.copy()
    H, W = v.shape
    for _ in range(max_iter or (H + W)):
        if v.all():
            break
        acc = np.zeros_like(g)
        cnt = np.zeros((H, W))
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            sv = np.zeros((H, W), bool)
            sg = np.zeros_like(g)
            ys = slice(max(dy, 0), H + min(dy, 0))
            yd = slice(max(-dy, 0), H + min(-dy, 0))
            xs = slice(max(dx, 0), W + min(dx, 0))
            xd = slice(max(-dx, 0), W + min(-dx, 0))
            sv[yd, xd] = v[ys, xs]
            sg[:, yd, xd] = g[:, ys, xs]
            acc += sg * sv
            cnt += sv
        new = ~v & (cnt > 0)
        g[:, new] = acc[:, new] / cnt[new]
        v = v | new
    return g


def _set_variant(variant: Optional[str]):
    """Select the Mitsuba variant. Must run before sionna.rt is imported: Sionna registers its plugins (the ITU radio
    material) for the variant active at import."""
    if variant:
        import sys

        import mitsuba as mi
        v = {"cuda": "cuda_ad_mono_polarized", "llvm": "llvm_ad_mono_polarized"}.get(variant, variant)
        if mi.variant() == v:
            return
        if "sionna.rt" in sys.modules:
            raise RuntimeError(f"sionna.rt was imported with Mitsuba variant {mi.variant()}; select {v} before "
                               "importing sionna.rt (mitsuba.set_variant) or pass variant=None")
        mi.set_variant(v)


def los_probability(scene, tx: Sequence[float], cell_centers: np.ndarray, cell: float, sub: int = 3) -> np.ndarray:
    """[H, W] share of k x k points per cell (at the cell centres' height) with a clear segment to tx."""
    import mitsuba as mi

    H, W = cell_centers.shape[:2]
    off = ((np.arange(sub) + 0.5) / sub - 0.5) * cell
    ox, oy = np.meshgrid(off, off)
    pts = cell_centers[:, :, None, :].repeat(sub * sub, 2).copy()           # [H, W, k^2, 3]
    pts[..., 0] += ox.reshape(-1)
    pts[..., 1] += oy.reshape(-1)
    pts = pts.reshape(-1, 3).astype(np.float64)
    o = np.asarray(tx, np.float64)[None].repeat(len(pts), 0)
    d = pts - o
    dist = np.linalg.norm(d, axis=1)
    d = d / np.maximum(dist, 1e-9)[:, None]
    f32 = lambda a: mi.Float(np.ascontiguousarray(a, dtype=np.float32))   # noqa: E731
    ray = mi.Ray3f(mi.Point3f(f32(o[:, 0]), f32(o[:, 1]), f32(o[:, 2])),
                   mi.Vector3f(f32(d[:, 0]), f32(d[:, 1]), f32(d[:, 2])))
    ray.maxt = f32(np.maximum(dist * (1 - 1e-4) - 1e-3, 0.0))
    blocked = np.array(scene.mi_scene.ray_test(ray), dtype=bool)
    return 1.0 - blocked.reshape(H, W, sub * sub).mean(-1)


def bake_scene(scene_xml: str, tx: Sequence[Sequence[float]], fc_ghz: float, bounds: Sequence[float],
               cell: float = 1.0, ue_h: float = 1.5, samples: int = 1_000_000, depth: int = 4,
               refraction: bool = True, diffraction: bool = False, los_map: bool = False, los_sub: int = 3,
               seed: int = 42, variant: Optional[str] = None, fill: bool = True, floor_db: float = -250.0) -> dict:
    """Run Sionna RT's RadioMapSolver on a Mitsuba scene. bounds (x0, y0, x1, y1) is the area to cover; the grid
    starts at (x0, y0) with cells of `cell` metres and is rounded up to whole cells. Returns the map dict (module
    docstring) with numpy arrays."""
    _set_variant(variant)
    import sionna.rt as rt

    t0 = time.perf_counter()
    scene = rt.load_scene(scene_xml)
    scene.frequency = fc_ghz * 1e9
    arr = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
    scene.tx_array, scene.rx_array = arr, arr
    for i, p in enumerate(tx):
        scene.add(rt.Transmitter(name=f"gnb{i}", position=[float(x) for x in p]))
    x0, y0, x1, y1 = (float(b) for b in bounds)
    nx = max(int(np.ceil((x1 - x0) / cell - 1e-6)), 2)
    ny = max(int(np.ceil((y1 - y0) / cell - 1e-6)), 2)
    size = [nx * cell, ny * cell]
    center = [x0 + size[0] / 2, y0 + size[1] / 2, ue_h]
    rm = rt.RadioMapSolver()(scene, center=center, orientation=[0, 0, 0], size=size, cell_size=[cell, cell],
                             samples_per_tx=int(samples), max_depth=int(depth), refraction=refraction,
                             diffraction=diffraction, seed=seed)
    npy = lambda a: np.asarray(a.numpy() if hasattr(a, "numpy") else a)   # noqa: E731
    pg = npy(rm.path_gain).astype(np.float64)                   # [C, ny, nx]
    cc = npy(rm.cell_centers).astype(np.float64)               # [ny, nx, 3]
    if cc[0, -1, 0] < cc[0, 0, 0]:                              # rows along +y, columns along +x
        pg, cc = pg[:, :, ::-1], cc[:, ::-1]
    if cc[-1, 0, 1] < cc[0, 0, 1]:
        pg, cc = pg[:, ::-1, :], cc[::-1]
    t_rm = time.perf_counter() - t0
    valid = (pg > 0).any(0)
    with np.errstate(divide="ignore"):
        gain = np.where(pg > 0, 10 * np.log10(np.maximum(pg, 1e-300)), floor_db)
    if fill:
        # cells no ray of a gNB reached (inside solids, or shadowed beyond max_depth) take their neighbours' values
        # for that gNB, so bilinear sampling next to an obstacle stays sane
        for c in range(gain.shape[0]):
            if (pg[c] > 0).any():
                gain[c:c + 1] = fill_holes(gain[c:c + 1], pg[c] > 0)
    out = dict(gain_db=gain.astype(np.float32),
               bounds=np.array([cc[0, 0, 0], cc[0, 0, 1], cc[-1, -1, 0], cc[-1, -1, 1]], np.float64),
               gnb_xy=np.array([[p[0], p[1]] for p in tx], np.float64), gnb_z=np.array([p[2] for p in tx], np.float64),
               valid=valid, fc_ghz=float(fc_ghz), ue_height_m=float(ue_h), cell_m=float(cell), samples=int(samples),
               max_depth=int(depth), radio_map_s=float(t_rm))
    if los_map:
        t1 = time.perf_counter()
        out["los_prob"] = np.stack([los_probability(scene, p, cc, cell, los_sub) for p in tx]).astype(np.float32)
        out["los_s"] = float(time.perf_counter() - t1)
    out["bake_s"] = float(time.perf_counter() - t0)
    try:
        from importlib.metadata import version
        out["sionna_rt"] = version("sionna-rt")
    except Exception:
        pass
    return out


def write_map(path: str, m: dict) -> str:
    """Write a map dict: .pt (torch tensors, loadable with weights_only) or .npz."""
    if path.endswith(".pt"):
        import torch
        d = {}
        for k, v in m.items():
            if isinstance(v, np.ndarray):
                d[k] = torch.from_numpy(np.ascontiguousarray(v))
            elif isinstance(v, (bool, np.bool_)):
                d[k] = bool(v)
            elif isinstance(v, (int, float, str)):
                d[k] = v
            else:
                d[k] = str(v)
        torch.save(d, path)
    else:
        np.savez_compressed(path, **{k: np.asarray(v) for k, v in m.items()})
    return path


def read_map(path: str) -> dict:
    if path.endswith(".pt"):
        import torch
        d = torch.load(path, map_location="cpu", weights_only=True)
        return {k: (v.numpy() if hasattr(v, "numpy") else v) for k, v in d.items()}
    with np.load(path, allow_pickle=False) as z:
        return {k: (z[k].item() if z[k].ndim == 0 else z[k]) for k in z.files}


def _kv(items, conv=str) -> dict:
    out = {}
    for s in items or []:
        k, _, v = s.rpartition("=")
        if not k:
            raise SystemExit(f"expected KEY=VALUE, got {s!r}")
        out[k] = conv(v)
    return out


def build_parser():
    ap = argparse.ArgumentParser(prog="python -m isaaclab_net.tools.scene.bake", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--usd", help="USD stage to export (needs pxr / usd-core)")
    src.add_argument("--scene-xml", help="an exported (or hand-made) Mitsuba XML scene instead of --usd")
    ap.add_argument("--out", help="output map (.pt or .npz)")
    ap.add_argument("--tx", type=float, nargs=3, action="append", metavar=("X", "Y", "Z"),
                    help="gNB / AP antenna position in the map frame (repeat per cell)")
    ap.add_argument("--fc", type=float, default=3.5, help="carrier (GHz)")
    ap.add_argument("--cell", type=float, default=1.0, help="grid cell size (m)")
    ap.add_argument("--ue-height", type=float, default=1.5, help="robot antenna height (m)")
    ap.add_argument("--bounds", type=float, nargs=4, metavar=("X0", "Y0", "X1", "Y1"),
                    help="area to cover (default: the exported geometry's footprint)")
    ap.add_argument("--samples", type=float, default=1e6, help="rays per transmitter")
    ap.add_argument("--depth", type=int, default=4, help="maximum number of interactions per path")
    ap.add_argument("--no-refraction", action="store_true", help="no transmission through walls")
    ap.add_argument("--diffraction", action="store_true", help="enable wedge diffraction")
    ap.add_argument("--los-map", action="store_true", help="also write los_prob [C,H,W]")
    ap.add_argument("--los-sub", type=int, default=3, help="k x k points per cell for los_prob")
    ap.add_argument("--no-fill", action="store_true", help="leave cells no ray reached at the floor value")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--variant", default=None, help="Mitsuba variant: cuda, llvm (CPU) or a full name")
    g = ap.add_argument_group("export (with --usd)")
    g.add_argument("--root", default="/", help="subtree to export")
    g.add_argument("--frame", default=None, help="prim whose frame is the map frame (default: stage world)")
    g.add_argument("--offset", type=float, nargs=3, default=(0.0, 0.0, 0.0), metavar=("DX", "DY", "DZ"),
                   help="added after the frame change (Isaac: IsaacNetCfg.pose_offset_m)")
    g.add_argument("--map", action="append", metavar="PATTERN=MATERIAL",
                   help="user material rule: fnmatch pattern on prim path / semantic label / USD material name")
    g.add_argument("--map-json", help="JSON file {pattern: material}, applied before --map")
    g.add_argument("--default-material", default="concrete")
    g.add_argument("--no-keywords", action="store_true", help="no keyword rules (only --map and the default)")
    g.add_argument("--thickness", action="append", metavar="MATERIAL=M", help="per-face slab thickness (m)")
    g.add_argument("--include", action="append", default=[], help="fnmatch pattern of prim paths to export")
    g.add_argument("--exclude", action="append", default=[], help="fnmatch pattern of prim paths to skip")
    g.add_argument("--crop", type=float, nargs=4, metavar=("X0", "Y0", "X1", "Y1"), help="drop triangles outside")
    g.add_argument("--z-max", type=float, default=None, help="drop triangles entirely above this height")
    g.add_argument("--meters-per-unit", type=float, default=None, help="override the stage's metersPerUnit")
    g.add_argument("--up-axis", default=None, choices=("Y", "Z"), help="override the stage's up axis")
    g.add_argument("--scene-dir", default=None, help="keep the exported scene here (default: a temp dir)")
    g.add_argument("--export-only", action="store_true", help="export the scene and stop (no Sionna needed)")
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    manifest = None
    tmp = None
    if a.usd:
        from .usd_export import export_usd
        mapping = {}
        if a.map_json:
            with open(a.map_json) as fh:
                mapping.update(json.load(fh))
        mapping.update(_kv(a.map))
        scene_dir = a.scene_dir
        if scene_dir is None:
            tmp = tempfile.TemporaryDirectory(prefix="scene_")
            scene_dir = tmp.name
        t0 = time.perf_counter()
        res = export_usd(a.usd, scene_dir, root=a.root, frame=a.frame, offset_m=a.offset, material_map=mapping,
                         default_material=a.default_material, keywords=not a.no_keywords,
                         thickness=_kv(a.thickness, float), include=a.include, exclude=a.exclude, crop=a.crop,
                         z_max=a.z_max, meters_per_unit=a.meters_per_unit, up_axis=a.up_axis)
        print(f"exported {a.usd} in {time.perf_counter() - t0:.1f} s -> {res.xml_path}: {res.summary()}", flush=True)
        xml, scene_hash = res.xml_path, res.scene_hash
        manifest = dict(materials=res.materials, sources=res.sources, coverage=res.coverage())
        footprint = (res.bounds_min[0], res.bounds_min[1], res.bounds_max[0], res.bounds_max[1])
    else:
        xml = a.scene_xml
        with open(xml, "rb") as fh:
            scene_hash = hashlib.sha256(fh.read()).hexdigest()
        footprint = None
        mf = os.path.join(os.path.dirname(os.path.abspath(xml)), "manifest.json")
        if os.path.exists(mf):
            with open(mf) as fh:
                man = json.load(fh)
            scene_hash = man.get("scene_hash", scene_hash)
            footprint = (man["bounds_min"][0], man["bounds_min"][1], man["bounds_max"][0], man["bounds_max"][1])
            manifest = dict(materials=man["materials"], sources=man["sources"])
    if a.export_only:
        return
    if not a.out or not a.tx:
        raise SystemExit("--out and at least one --tx are needed to bake")
    bounds = a.bounds or footprint
    if bounds is None:
        raise SystemExit("--bounds is needed with --scene-xml without a manifest.json")
    m = bake_scene(xml, a.tx, a.fc, bounds, a.cell, a.ue_height, int(a.samples), a.depth,
                   refraction=not a.no_refraction, diffraction=a.diffraction, los_map=a.los_map, los_sub=a.los_sub,
                   seed=a.seed, variant=a.variant, fill=not a.no_fill)
    m["scene_hash"] = scene_hash
    m["bake_key"] = bake_key(scene_hash, a.tx, a.fc, bounds, a.cell, a.ue_height, int(a.samples), a.depth,
                             not a.no_refraction, a.diffraction, a.los_map, a.los_sub, a.seed)
    m["source"] = "sionna-rt RadioMapSolver, " + os.path.basename(a.usd or a.scene_xml)
    if manifest is not None:
        m["materials"] = json.dumps(manifest)
    write_map(a.out, m)
    g = m["gain_db"]
    print(f"wrote {a.out}: gain_db {tuple(g.shape)}, bounds {tuple(np.round(m['bounds'], 3))}, "
          f"range {g.min():.1f} .. {g.max():.1f} dB, valid {100 * m['valid'].mean():.1f}%, "
          f"bake {m['bake_s']:.1f} s" + (f", LOS map {m['los_s']:.1f} s" if "los_s" in m else ""), flush=True)
    if tmp is not None:
        tmp.cleanup()


if __name__ == "__main__":
    main()
