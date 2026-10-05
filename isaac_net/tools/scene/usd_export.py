"""USD stage -> Sionna RT scene (Mitsuba XML + PLY meshes with ITU radio materials).

Needs `pxr`: inside Isaac Sim / Isaac Lab it is already importable, elsewhere `pip install usd-core`.

What is exported. Every imageable prim under `root` (default the pseudo-root), instance proxies included, that is
visible, has purpose default or render (guide and proxy geometry are skipped) and is not excluded:
UsdGeom.Mesh (polygons fan-triangulated, holes ignored, subdivision surfaces as their control mesh), the implicit
gprims Cube, Sphere, Cylinder, Cone, Capsule and Plane (tessellated), and PointInstancer instances of those. Each
prim gets one ITU material from materials.MaterialRules (user mapping > semantic label > bound material name >
prim name > concrete). Triangles are merged into one PLY per material.

Map frame (the coordinate convention). A point p of a prim is written as

    p_map = C( p_local M_prim M_frame^-1 ) + offset_m

where M_prim is the prim's local-to-world matrix at `time` (USD row-vector convention), M_frame the local-to-world
matrix of the `frame` prim (identity when frame is None, i.e. the stage's world frame), and C converts stage units to
metres (UsdGeom metersPerUnit) and a Y-up stage to Z-up ((x, y, z) -> (x, -z, y)). The result is metres, Z up,
which is Sionna's convention. For an Isaac Lab env the map frame must be the frame of the robot poses the network
sees: the env-local frame (frame = "/World/envs/env_0") shifted by IsaacNetCfg.pose_offset_m (offset_m).
"""
from __future__ import annotations

import fnmatch
import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Mapping, Optional, Sequence

import numpy as np

from .materials import DEFAULT_THICKNESS, MaterialRules
from .mitsuba_writer import write_scene

SOURCES = ("user", "semantic", "material", "name", "default")


@dataclass
class ExportResult:
    xml_path: str
    out_dir: str
    scene_hash: str                      # sha256 of the exported geometry, materials and thicknesses
    n_prims: int
    n_triangles: int
    materials: dict                      # {material: {"prims", "triangles", "thickness"}}
    sources: dict                        # {source: {"prims", "triangles"}} (materials.SOURCES order)
    bounds_min: list                     # map frame, metres
    bounds_max: list
    meters_per_unit: float
    up_axis: str
    skipped: dict                        # {reason: prim count}
    prims: list = field(default_factory=list)   # [(path, type, material, source, triangles)]

    def obstacle_z(self, bounds, H: int, W: int) -> "np.ndarray":
        """Obstacle height map [H, W] of the exported triangles on a radio-map grid (heightmap.height_map; the
        export's exclude / z_max options decide which prims count, so exclude the robots and drop the ceiling)."""
        from .heightmap import height_map, scene_triangles
        v, f = scene_triangles(self.out_dir)
        return height_map(v, f, bounds, H, W)

    def coverage(self) -> float:
        """Share of triangles whose material came from a rule other than the default."""
        n = max(self.n_triangles, 1)
        return 1.0 - self.sources.get("default", {}).get("triangles", 0) / n

    def summary(self) -> str:
        mats = ", ".join(f"{m} {d['prims']} prims / {d['triangles']} tris" for m, d in self.materials.items())
        src = ", ".join(f"{s} {d['prims']}" for s, d in self.sources.items() if d["prims"])
        lo = ", ".join(f"{x:.2f}" for x in self.bounds_min)
        hi = ", ".join(f"{x:.2f}" for x in self.bounds_max)
        return (f"{self.n_prims} prims, {self.n_triangles} triangles; materials: {mats}; assigned by: {src}; "
                f"coverage {100 * self.coverage():.1f}%; bounds ({lo}) .. ({hi}) m; skipped {self.skipped}")


# ------------------------------------------------------------------------------------------------ geometry helpers
def _fan(counts: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Fan triangulation of polygons (faceVertexCounts, faceVertexIndices) -> [T,3] vertex indices."""
    counts = np.asarray(counts, dtype=np.int64)
    idx = np.asarray(idx, dtype=np.int64)
    if counts.size == 0:
        return np.zeros((0, 3), np.int64)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    ntri = np.clip(counts - 2, 0, None)
    poly = np.repeat(np.arange(len(counts)), ntri)
    k = np.arange(ntri.sum()) - np.repeat(np.cumsum(ntri) - ntri, ntri) + 1
    s = starts[poly]
    return np.stack([idx[s], idx[s + k], idx[s + k + 1]], -1)


def _box(half):
    hx, hy, hz = half
    v = np.array([[x, y, z] for x in (-hx, hx) for y in (-hy, hy) for z in (-hz, hz)], float)
    quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    f = [(a, b, c) for a, b, c, d in quads] + [(a, c, d) for a, b, c, d in quads]
    return v, np.array(f, np.int64)


def _axis_rot(axis: str) -> np.ndarray:
    """Rows map a Z-aligned shape onto `axis` (row-vector convention)."""
    if axis == "X":
        return np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]], float)
    if axis == "Y":
        return np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)
    return np.eye(3)


def _revolve(r_bottom, r_top, height, axis="Z", seg=24):
    """Cylinder / cone around `axis`, centred at the origin, with caps."""
    a = np.linspace(0, 2 * np.pi, seg, endpoint=False)
    ring = np.stack([np.cos(a), np.sin(a)], -1)
    bot = np.c_[ring * r_bottom, np.full(seg, -height / 2)]
    top = np.c_[ring * r_top, np.full(seg, height / 2)]
    v = np.vstack([bot, top, [[0, 0, -height / 2], [0, 0, height / 2]]])
    i = np.arange(seg)
    j = (i + 1) % seg
    f = np.vstack([np.stack([i, j, seg + j], -1), np.stack([i, seg + j, seg + i], -1),
                   np.stack([np.full(seg, 2 * seg), j, i], -1), np.stack([np.full(seg, 2 * seg + 1), seg + i, seg + j], -1)])
    return v @ _axis_rot(axis), f


def _sphere(r, seg=16, rings=8):
    th = np.linspace(0, np.pi, rings + 1)[1:-1]
    ph = np.linspace(0, 2 * np.pi, seg, endpoint=False)
    v = [[0, 0, r]] + [[r * np.sin(t) * np.cos(p), r * np.sin(t) * np.sin(p), r * np.cos(t)] for t in th for p in ph]
    v.append([0, 0, -r])
    f = []
    for p in range(seg):
        q = (p + 1) % seg
        f.append((0, 1 + p, 1 + q))
        for k in range(rings - 2):
            a, b = 1 + k * seg + p, 1 + k * seg + q
            f += [(a, a + seg, b + seg), (a, b + seg, b)]
        last = 1 + (rings - 2) * seg
        f.append((len(v) - 1, last + q, last + p))
    return np.array(v, float), np.array(f, np.int64)


def gprim_triangles(prim, tc):
    """(vertices [N,3] in prim-local units, faces [T,3]) of a supported gprim, or None."""
    from pxr import UsdGeom

    def get(attr, default):
        v = attr.Get(tc) if attr and attr.HasAuthoredValue() else None
        return default if v is None else v

    if prim.IsA(UsdGeom.Mesh):
        m = UsdGeom.Mesh(prim)
        pts = m.GetPointsAttr().Get(tc)
        counts = m.GetFaceVertexCountsAttr().Get(tc)
        idx = m.GetFaceVertexIndicesAttr().Get(tc)
        if pts is None or counts is None or idx is None or len(pts) == 0:
            return None
        f = _fan(np.asarray(counts), np.asarray(idx))
        if m.GetOrientationAttr().Get(tc) == UsdGeom.Tokens.leftHanded:
            f = f[:, [0, 2, 1]]
        return np.asarray(pts, float), f
    if prim.IsA(UsdGeom.Cube):
        s = float(get(UsdGeom.Cube(prim).GetSizeAttr(), 2.0))
        return _box((s / 2, s / 2, s / 2))
    if prim.IsA(UsdGeom.Sphere):
        return _sphere(float(get(UsdGeom.Sphere(prim).GetRadiusAttr(), 1.0)))
    if prim.IsA(UsdGeom.Cylinder):
        g = UsdGeom.Cylinder(prim)
        r = float(get(g.GetRadiusAttr(), 1.0))
        return _revolve(r, r, float(get(g.GetHeightAttr(), 2.0)), str(get(g.GetAxisAttr(), "Z")))
    if prim.IsA(UsdGeom.Cone):
        g = UsdGeom.Cone(prim)
        return _revolve(float(get(g.GetRadiusAttr(), 1.0)), 1e-6, float(get(g.GetHeightAttr(), 2.0)),
                        str(get(g.GetAxisAttr(), "Z")))
    if prim.IsA(UsdGeom.Capsule):
        g = UsdGeom.Capsule(prim)
        r = float(get(g.GetRadiusAttr(), 0.5))
        return _revolve(r, r, float(get(g.GetHeightAttr(), 1.0)) + 2 * r, str(get(g.GetAxisAttr(), "Z")))
    if hasattr(UsdGeom, "Plane") and prim.IsA(UsdGeom.Plane):
        g = UsdGeom.Plane(prim)
        w, ln = float(get(g.GetWidthAttr(), 2.0)), float(get(g.GetLengthAttr(), 2.0))
        v = np.array([[-w / 2, -ln / 2, 0], [w / 2, -ln / 2, 0], [w / 2, ln / 2, 0], [-w / 2, ln / 2, 0]], float)
        return v @ _axis_rot(str(get(g.GetAxisAttr(), "Z"))), np.array([[0, 1, 2], [0, 2, 3]], np.int64)
    return None


def _mat(m) -> np.ndarray:
    return np.array(m, dtype=float).reshape(4, 4)


def _apply(v: np.ndarray, M: np.ndarray) -> np.ndarray:
    return v @ M[:3, :3] + M[3, :3]


# ------------------------------------------------------------------------------------------------ exporter
def open_stage(stage):
    """A pxr Usd.Stage from a stage object or a file path."""
    if isinstance(stage, (str, os.PathLike)):
        from pxr import Usd
        s = Usd.Stage.Open(str(stage))
        if s is None:
            raise FileNotFoundError(f"cannot open USD stage {stage}")
        return s
    return stage


def export_usd(stage, out_dir: str, root: str = "/", frame: Optional[str] = None,
               offset_m: Sequence[float] = (0.0, 0.0, 0.0), material_map: Optional[Mapping[str, str]] = None,
               default_material: str = "concrete", keywords: bool = True,
               thickness: Optional[Mapping[str, float]] = None, include: Sequence[str] = (),
               exclude: Sequence[str] = (), time: Optional[float] = None,
               crop: Optional[Sequence[float]] = None, z_max: Optional[float] = None,
               meters_per_unit: Optional[float] = None, up_axis: Optional[str] = None,
               purposes: Sequence[str] = ("default", "render"), keep_prim_list: int = 20000) -> ExportResult:
    """Export the meshes of a USD stage (object or path) under `root` into out_dir (see the module docstring).

    frame / offset_m: the map frame. material_map {fnmatch pattern: ITU material}, default_material, keywords:
    materials.MaterialRules. thickness {material: m}: per-face slab thickness. include / exclude: fnmatch patterns on
    prim paths (a prim is exported when it or an ancestor matches an include pattern, if any, and neither it nor an
    ancestor matches an exclude pattern). crop (x0, y0, x1, y1) and z_max, in the map frame: drop triangles entirely
    outside. meters_per_unit / up_axis: override the stage metadata."""
    from pxr import Usd, UsdGeom, UsdShade

    stage = open_stage(stage)
    tc = Usd.TimeCode.Default() if time is None else Usd.TimeCode(time)
    mpu = float(meters_per_unit if meters_per_unit is not None else UsdGeom.GetStageMetersPerUnit(stage))
    up = str(up_axis if up_axis is not None else UsdGeom.GetStageUpAxis(stage)).upper()
    xfc = UsdGeom.XformCache(tc)
    # map-frame conversion: world -> frame -> metres, Z up -> + offset
    Minv = np.eye(4)
    if frame:
        fp = stage.GetPrimAtPath(frame)
        if not fp or not fp.IsValid():
            raise ValueError(f"frame prim {frame} not found")
        Minv = np.linalg.inv(_mat(xfc.GetLocalToWorldTransform(fp)))
    C = np.eye(4) * mpu
    C[3, 3] = 1.0
    if up == "Y":
        C = C @ np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], float)
    elif up != "Z":
        raise ValueError(f"up axis {up!r}")
    T = Minv @ C
    T[3, :3] += np.asarray(offset_m, float)

    rules = MaterialRules(material_map, default_material, keywords)
    th = dict(DEFAULT_THICKNESS)
    th.update(thickness or {})
    root_prim = stage.GetPrimAtPath(root) if root != "/" else stage.GetPseudoRoot()
    if not root_prim or not root_prim.IsValid():
        raise ValueError(f"root prim {root} not found")
    inc = [p.lower() for p in include]
    exc = [p.lower() for p in exclude]

    label_cache: dict = {}

    def own_labels(p):
        key = str(p.GetPath())
        if key not in label_cache:
            out = []
            for a in p.GetAttributes():
                n = a.GetName()
                if n.startswith("semantics:labels:") or (n.startswith("semantic:") and n.endswith("semanticData")):
                    v = a.Get()
                    if v is None:
                        continue
                    out += [str(x) for x in v] if not isinstance(v, str) else [v]
            label_cache[key] = out
        return label_cache[key]

    def labels_of(p):
        out = []
        while p and p.IsValid() and not p.IsPseudoRoot():
            out += own_labels(p)
            p = p.GetParent()
        return out

    def material_name(p):
        try:
            m, _ = UsdShade.MaterialBindingAPI(p).ComputeBoundMaterial()
            return m.GetPrim().GetName() if m else None
        except Exception:
            return None

    def matches(path, pats):
        low = path.lower()
        parts = low.split("/")
        anc = ["/".join(parts[:i]) for i in range(2, len(parts) + 1)]
        return any(fnmatch.fnmatchcase(a, p) for a in anc for p in pats)

    groups: dict = {}
    stats_m = {}
    stats_s = {s: {"prims": 0, "triangles": 0} for s in SOURCES}
    skipped: dict = {}
    prims = []

    def add(prim, v_local, f, M_world, ptype):
        if f.size == 0:
            skipped["empty"] = skipped.get("empty", 0) + 1
            return
        v = _apply(_apply(np.asarray(v_local, float), M_world), T)
        a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
        keep = np.linalg.norm(np.cross(b - a, c - a), axis=1) > 1e-10
        if crop is not None:
            x0, y0, x1, y1 = crop
            xs, ys = v[:, 0][f], v[:, 1][f]
            keep &= ~((xs < x0).all(1) | (xs > x1).all(1) | (ys < y0).all(1) | (ys > y1).all(1))
        if z_max is not None:
            keep &= ~(v[:, 2][f] > z_max).all(1)
        f = f[keep]
        if len(f) == 0:
            skipped["cropped_or_degenerate"] = skipped.get("cropped_or_degenerate", 0) + 1
            return
        used, inv = np.unique(f, return_inverse=True)
        v, f = v[used], inv.reshape(-1, 3)
        path = str(prim.GetPath())
        mat, src = rules.assign(path, labels_of(prim), material_name(prim))
        groups.setdefault(mat, []).append((v.astype(np.float32), f))
        d = stats_m.setdefault(mat, {"prims": 0, "triangles": 0, "thickness": th[mat]})
        d["prims"] += 1
        d["triangles"] += len(f)
        stats_s[src]["prims"] += 1
        stats_s[src]["triangles"] += len(f)
        if len(prims) < keep_prim_list:
            prims.append((path, ptype, mat, src, int(len(f))))

    it = iter(Usd.PrimRange(root_prim, Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate)))
    for prim in it:
        if prim.IsPseudoRoot():
            continue
        path = str(prim.GetPath())
        if exc and matches(path, exc):
            it.PruneChildren()
            continue
        img = UsdGeom.Imageable(prim)
        if img:
            if img.ComputeVisibility(tc) == UsdGeom.Tokens.invisible:
                skipped["invisible"] = skipped.get("invisible", 0) + 1
                it.PruneChildren()
                continue
            if str(img.ComputePurpose()) not in purposes:
                skipped["purpose"] = skipped.get("purpose", 0) + 1
                it.PruneChildren()
                continue
        if inc and not matches(path, inc):
            continue
        if prim.IsA(UsdGeom.PointInstancer):
            _add_instancer(prim, tc, xfc, add, skipped)
            it.PruneChildren()
            continue
        if not prim.IsA(UsdGeom.Gprim):
            continue
        tri = gprim_triangles(prim, tc)
        if tri is None:
            skipped[prim.GetTypeName() or "untyped"] = skipped.get(prim.GetTypeName() or "untyped", 0) + 1
            continue
        add(prim, tri[0], tri[1], _mat(xfc.GetLocalToWorldTransform(prim)), prim.GetTypeName())

    merged = {}
    for mat, parts in groups.items():
        vs, fs, off = [], [], 0
        for v, f in parts:
            vs.append(v)
            fs.append(f + off)
            off += len(v)
        merged[mat] = (np.concatenate(vs), np.concatenate(fs).astype(np.int32))
    if not merged:
        raise ValueError(f"no exportable geometry under {root} (skipped: {skipped})")
    allv = np.concatenate([v for v, _ in merged.values()])
    h = hashlib.sha256()
    for mat in sorted(merged):
        v, f = merged[mat]
        h.update(f"{mat}:{th[mat]:.6g}:{len(v)}:{len(f)}".encode())
        h.update(np.round(v.astype(np.float64), 4).astype(np.float32).tobytes())
        h.update(f.astype(np.int32).tobytes())
    scene_hash = h.hexdigest()
    os.makedirs(out_dir, exist_ok=True)
    xml = write_scene(out_dir, merged, th, comment=f"isaac_net.tools.scene.usd_export scene_hash={scene_hash}")
    res = ExportResult(xml_path=xml, out_dir=out_dir, scene_hash=scene_hash, n_prims=sum(d["prims"] for d in stats_m.values()),
                       n_triangles=int(sum(len(f) for _, f in merged.values())), materials=dict(sorted(stats_m.items())),
                       sources=stats_s, bounds_min=allv.min(0).tolist(), bounds_max=allv.max(0).tolist(),
                       meters_per_unit=mpu, up_axis=up, skipped=skipped, prims=prims)
    man = asdict(res)
    man.update(root=root, frame=frame, offset_m=list(map(float, offset_m)), material_map=dict(material_map or {}),
               default_material=default_material, include=list(include), exclude=list(exclude))
    with open(os.path.join(out_dir, "manifest.json"), "w") as fh:
        json.dump(man, fh, indent=1)
    return res


def _add_instancer(prim, tc, xfc, add, skipped):
    """Instances of a UsdGeom.PointInstancer: every gprim under each prototype, placed by the instance transforms."""
    from pxr import Usd, UsdGeom

    pi = UsdGeom.PointInstancer(prim)
    ids = pi.GetProtoIndicesAttr().Get(tc)
    targets = pi.GetPrototypesRel().GetTargets()
    if ids is None or not targets:
        skipped["empty_instancer"] = skipped.get("empty_instancer", 0) + 1
        return
    xfs = pi.ComputeInstanceTransformsAtTime(tc, tc)      # prototype-root xform included, mask applied
    stage = prim.GetStage()
    M_inst = _mat(xfc.GetLocalToWorldTransform(prim))
    ids = np.asarray(ids)
    for j, tgt in enumerate(targets):
        proot = stage.GetPrimAtPath(tgt)
        sel = [k for k in range(len(xfs)) if ids[k] == j]
        if not proot or not sel:
            continue
        M_root_inv = np.linalg.inv(_mat(xfc.GetLocalToWorldTransform(proot)))
        for p in Usd.PrimRange(proot, Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate)):
            if not p.IsA(UsdGeom.Gprim):
                continue
            tri = gprim_triangles(p, tc)
            if tri is None:
                continue
            M_rel = _mat(xfc.GetLocalToWorldTransform(p)) @ M_root_inv
            for k in sel:
                add(p, tri[0], tri[1], M_rel @ _mat(xfs[k]) @ M_inst, "PointInstancer:" + p.GetTypeName())
