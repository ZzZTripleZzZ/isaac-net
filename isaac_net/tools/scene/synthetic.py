"""Small USD scenes with known geometry, for validating the exporter and the bake (needs pxr / usd-core).

box_arena_usd writes an arena of L x W metres with its corner at the origin: optionally a floor (z = 0) and four
walls of height H around it, and optionally one interior wall across the arena at x = wall_x, either as a closed
box of thickness wall_t ("box") or as a single two-sided quad ("plane"). The interior wall spans y in
[wall_y0, wall_y1] (default: the full width) and carries the semantic label `wall_label` (default "wall" ->
concrete), the Isaac-style `semantic:Semantics:params:semanticData` attribute. meters_per_unit and up_axis write the
same geometry in other stage conventions (for example centimetres and Y up), so the exporter's conversion back to
metres, Z up can be checked.
"""
from __future__ import annotations

from typing import Optional


def _label(prim, label: str):
    from pxr import Sdf
    prim.CreateAttribute("semantic:Semantics:params:semanticType", Sdf.ValueTypeNames.String).Set("class")
    prim.CreateAttribute("semantic:Semantics:params:semanticData", Sdf.ValueTypeNames.String).Set(label)


def box_arena_usd(path: str, L: float = 40.0, W: float = 20.0, H: float = 6.0, floor: bool = True,
                  arena_walls: bool = True, wall_x: Optional[float] = 20.0, wall_t: float = 0.2,
                  wall_h: Optional[float] = None, wall_kind: str = "box", wall_y0: float = 0.0,
                  wall_y1: Optional[float] = None, wall_label: str = "wall", meters_per_unit: float = 1.0,
                  up_axis: str = "Z", extra: Optional[dict] = None):
    """Write the arena to `path` (see the module docstring) and return the stage. extra {name: (cx, cy, cz, sx, sy,
    sz, label)} adds labelled boxes (centre and full sizes in metres, Z up)."""
    from pxr import Gf, Usd, UsdGeom

    stage = Usd.Stage.CreateNew(path)
    UsdGeom.SetStageMetersPerUnit(stage, meters_per_unit)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y if up_axis.upper() == "Y" else UsdGeom.Tokens.z)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    s = 1.0 / meters_per_unit

    def conv(x, y, z):
        """metres, Z up -> stage units and axes"""
        return (x * s, z * s, -y * s) if up_axis.upper() == "Y" else (x * s, y * s, z * s)

    def box(name, cx, cy, cz, sx, sy, sz, label):
        c = UsdGeom.Cube.Define(stage, f"/World/{name}")
        c.GetSizeAttr().Set(1.0)
        c.AddTranslateOp().Set(Gf.Vec3d(*conv(cx, cy, cz)))
        dims = (sx * s, sz * s, sy * s) if up_axis.upper() == "Y" else (sx * s, sy * s, sz * s)
        c.AddScaleOp().Set(Gf.Vec3f(*dims))
        _label(c.GetPrim(), label)
        return c

    def quad(name, pts, label):
        m = UsdGeom.Mesh.Define(stage, f"/World/{name}")
        m.GetPointsAttr().Set([Gf.Vec3f(*conv(*p)) for p in pts])
        m.GetFaceVertexCountsAttr().Set([4])
        m.GetFaceVertexIndicesAttr().Set([0, 1, 2, 3])
        m.GetDoubleSidedAttr().Set(True)
        _label(m.GetPrim(), label)
        return m

    t = 0.2
    if floor:
        quad("Floor", [(0, 0, 0), (L, 0, 0), (L, W, 0), (0, W, 0)], "floor")
    if arena_walls:
        box("WallSouth", L / 2, -t / 2, H / 2, L + 2 * t, t, H, "wall")
        box("WallNorth", L / 2, W + t / 2, H / 2, L + 2 * t, t, H, "wall")
        box("WallWest", -t / 2, W / 2, H / 2, t, W, H, "wall")
        box("WallEast", L + t / 2, W / 2, H / 2, t, W, H, "wall")
    if wall_x is not None:
        y1 = W if wall_y1 is None else wall_y1
        h = H if wall_h is None else wall_h
        if wall_kind == "box":
            box("InnerWall", wall_x, (wall_y0 + y1) / 2, h / 2, wall_t, y1 - wall_y0, h, wall_label)
        elif wall_kind == "plane":
            quad("InnerWall", [(wall_x, wall_y0, 0), (wall_x, y1, 0), (wall_x, y1, h), (wall_x, wall_y0, h)],
                 wall_label)
        else:
            raise ValueError(wall_kind)
    for name, (cx, cy, cz, sx, sy, sz, label) in (extra or {}).items():
        box(name, cx, cy, cz, sx, sy, sz, label)
    stage.GetRootLayer().Save()
    return stage
