"""USD scene -> Sionna RT scene -> radio map (isaac_net/tools/scene, isaac_net/isaac/scene_map.py).

Pure-numpy parts (material rules, slab formula, PLY/XML writer, hole filling, map files, the Isaac config hook) run
everywhere. The exporter tests need `pxr` (usd-core) and are skipped without it; the bake validation needs sionna-rt
and is in test_scene_bake.py.
"""
import json
import os

import numpy as np
import pytest
import torch

from isaac_net import NRConfig
from isaac_net.tools.scene.bake import bake_key, fill_holes, read_map, write_map
from isaac_net.tools.scene.materials import (MaterialRules, free_space_gain_db, keyword_material,
                                                slab_transmission_db, tokens)
from isaac_net.tools.scene.mitsuba_writer import read_ply, write_ply, write_scene


# ------------------------------------------------------------------------------------------------ materials
def test_tokens_and_keywords():
    assert tokens("SM_RackShelf_01") == ["sm", "rack", "shelf", "01"]
    assert tokens("KLTBin2") == ["klt", "bin", "2"]
    cases = {"SM_RackShelf_01": "metal", "WoodenRack": "wood", "PalletRack": "metal", "SM_PaletteA_01": "wood",
             "GlassWall": "glass", "SM_WallA_6M": "concrete", "KLT_Bins": "wood", "Drywall_Partition": "plasterboard",
             "SM_BracketSlot": "metal", "forklift": "metal", "SM_CardBoxA_02": "wood", "SM_FuseBox_01": "metal",
             "SM_floor27": "concrete", "SM_Window": "glass", "S_Barcode": None, "Xform": None}
    for name, mat in cases.items():
        assert keyword_material([name]) == mat, name


def test_rule_priority():
    r = MaterialRules({"*/special*": "glass", "rack": "wood"})
    assert r.assign("/World/special_rack") == ("glass", "user")                      # path pattern
    assert r.assign("/World/Thing", labels=["rack"]) == ("wood", "user")            # label pattern
    assert r.assign("/World/Thing", labels=["shelf"]) == ("metal", "semantic")
    assert r.assign("/World/Thing", material_name="M_Steel_Galvanized") == ("metal", "material")
    assert r.assign("/World/Shelf/Thing", material_name="M_Blue") == ("metal", "name")
    assert r.assign("/World/Thing", material_name="M_Blue") == ("concrete", "default")
    assert MaterialRules(default="wood", keywords=False).assign("/World/Shelf") == ("wood", "default")
    with pytest.raises(ValueError):
        MaterialRules({"x": "unobtainium"})


def test_slab_and_free_space():
    assert abs(free_space_gain_db(1.0, 3.5) + 43.33) < 0.01                  # 20 log10(4 pi / 0.0857)
    assert abs(free_space_gain_db(10.0, 3.5) - free_space_gain_db(1.0, 3.5) + 20.0) < 1e-9
    c1, c2 = slab_transmission_db("concrete", 0.1, 3.5), slab_transmission_db("concrete", 0.2, 3.5)
    assert 9.0 < c1 < 12.0 and c2 > c1
    assert slab_transmission_db("metal", 0.002, 3.5) > 100.0
    assert 1.0 < slab_transmission_db("glass", 0.006, 3.5) < 4.0            # high eps_r: interface loss
    assert slab_transmission_db("plasterboard", 0.0125, 3.5) < slab_transmission_db("wood", 0.1, 3.5)


# ------------------------------------------------------------------------------------------------ writer / files
def test_ply_roundtrip_and_xml(tmp_path):
    v = np.random.rand(5, 3).astype(np.float32)
    f = np.array([[0, 1, 2], [2, 3, 4]])
    write_ply(str(tmp_path / "a.ply"), v, f)
    v2, f2 = read_ply(str(tmp_path / "a.ply"))
    assert np.array_equal(v, v2) and np.array_equal(f, f2)
    xml = write_scene(str(tmp_path / "s"), {"metal": (v, f), "concrete": (v, f), "wood": (v, f[:0])},
                      thickness={"metal": 0.005})
    txt = open(xml).read()
    assert 'id="metal"' in txt and 'value="0.005"' in txt and "meshes/concrete.ply" in txt and "wood" not in txt
    assert os.path.exists(tmp_path / "s" / "meshes" / "metal.ply")


def test_fill_holes():
    g = np.full((1, 4, 5), -250.0)
    valid = np.zeros((4, 5), bool)
    g[0, 0, 0], valid[0, 0] = -60.0, True
    g[0, 3, 4], valid[3, 4] = -80.0, True
    out = fill_holes(g, valid)
    assert out.max() <= -60.0 and out.min() >= -80.0 and out[0, 0, 0] == -60.0 and out[0, 3, 4] == -80.0
    assert out[0, 0, 1] == -60.0 and out[0, 3, 3] == -80.0


def test_bake_key():
    k = bake_key("abc", [(1, 2, 3)], 3.5, (0, 0, 10, 10), 1.0, 1.5, 1000, 4)
    assert k == bake_key("abc", [(1.0, 2.0, 3.0)], 3.5, [0, 0, 10, 10], 1, 1.5, 1000, 4)
    for kw in (dict(scene_hash="abd"), dict(tx=[(1, 2, 4)]), dict(fc_ghz=28.0), dict(cell=0.5), dict(depth=3)):
        a = dict(scene_hash="abc", tx=[(1, 2, 3)], fc_ghz=3.5, bounds=(0, 0, 10, 10), cell=1.0, ue_h=1.5,
                 samples=1000, depth=4)
        a.update(kw)
        assert bake_key(**a) != k


def _fake_map(C=2, H=6, W=8):
    x = np.linspace(0, 14, W)
    y = np.linspace(0, 10, H)
    X, Y = np.meshgrid(x, y)
    gxy = np.array([[2.0, 5.0], [12.0, 5.0]])[:C]
    g = np.stack([-40 - 20 * np.log10(np.maximum(np.hypot(X - a, Y - b), 1.0)) for a, b in gxy]).astype(np.float32)
    return dict(gain_db=g, bounds=np.array([0.0, 0.0, 14.0, 10.0]), gnb_xy=gxy, gnb_z=np.array([6.0] * C),
                valid=np.ones((H, W), bool), los_prob=np.ones((C, H, W), np.float32), fc_ghz=3.5, ue_height_m=1.5,
                scene_hash="h", bake_key="k", source="test", materials=json.dumps({"concrete": 1}))


@pytest.mark.parametrize("suffix", [".pt", ".npz"])
def test_map_file_roundtrip(tmp_path, suffix):
    from isaac_net.core.channels.radio_map import RadioMap
    m = _fake_map()
    p = write_map(str(tmp_path / f"m{suffix}"), m)
    d = read_map(p)
    assert np.allclose(d["gain_db"], m["gain_db"]) and d["source"] == "test" and float(d["fc_ghz"]) == 3.5
    rm = RadioMap.load(p)
    assert (rm.C, rm.H, rm.W) == (2, 6, 8) and rm.bounds == (0.0, 0.0, 14.0, 10.0)
    assert np.allclose(np.asarray(rm.meta["gnb_xy"]), m["gnb_xy"])
    pos = torch.tensor([[[0.0, 0.0], [14.0, 10.0]]])
    assert torch.allclose(rm.sample(pos)[0, 0], torch.tensor(m["gain_db"][:, 0, 0]))


# ------------------------------------------------------------------------------------------------ Isaac hook
def test_apply_scene_radio_map_and_engine(tmp_path):
    from isaac_net import make_engine
    from isaac_net.isaac import IsaacNetCfg
    from isaac_net.isaac.scene_map import SceneRadioMapCfg, apply_scene_radio_map

    p = write_map(str(tmp_path / "m.pt"), _fake_map())
    isaac = IsaacNetCfg(gnb_pos=((2.0, 5.0, 6.0), (12.0, 5.0, 6.0)), scene_map=SceneRadioMapCfg())
    nr, isc = apply_scene_radio_map(NRConfig(), isaac, p)
    assert nr.channel == "radio_map" and nr.radio_map_path == p and nr.n_cells == 2
    assert nr.cell_layout == "custom" and nr.gnb_xy() == [(2.0, 5.0), (12.0, 5.0)]
    assert isc.radio == "engine" and isc.scene_map is None and isc.gnb_pos == isaac.gnb_pos
    eng = make_engine("L2", 2, 3, "cpu", nr)
    out = eng.step(None, torch.tensor([[[2.0, 5.0], [7.0, 5.0], [12.0, 5.0]]] * 2))
    assert out["serving_cell"].shape == (2, 3) and torch.isfinite(out["sinr_db"]).all()


def test_mixin_hook_uses_cached_map(tmp_path, monkeypatch):
    """net_setup with IsaacNetCfg.scene_map: resolve_scene_map runs before the network is built."""
    import isaac_net.isaac.scene_map as sm
    from isaac_net.isaac import IsaacNetCfg
    from isaac_net.isaac.mixins import NetEnvMixin

    p = write_map(str(tmp_path / "m.pt"), _fake_map())
    calls = []

    def fake_bake(stage, scfg, tx, fc, ue_h, offset, verbose=True):
        calls.append((stage, tx, fc, ue_h, offset))
        return p, dict(path=p, cached=True)

    monkeypatch.setattr(sm, "bake_stage_map", fake_bake)
    monkeypatch.setattr(sm, "current_stage", lambda: "STAGE")

    class Env(NetEnvMixin):
        num_envs, device = 2, "cpu"
        episode_length_buf = torch.ones(2, dtype=torch.long)

    env = Env()
    isaac = IsaacNetCfg(gnb_pos=((2.0, 5.0, 6.0), (12.0, 5.0, 6.0)), pose_offset_m=(1.0, 0.0, 0.0),
                        scene_map=sm.SceneRadioMapCfg(), pose_chunks=1)
    env.net_setup("L2", 3, NRConfig(carrier_ghz=28.0), "reference", isaac=isaac)
    assert calls == [("STAGE", [(2.0, 5.0, 6.0), (12.0, 5.0, 6.0)], 28.0, 1.5, (1.0, 0.0, 0.0))]
    assert env.net.config.channel == "radio_map" and env.net.isaac.radio == "engine" and env.net.config.n_cells == 2
    out = env.net_step(torch.rand(2, 3, 3) * 10, torch.ones(2, 3, dtype=torch.long))
    assert out["sinr_db"].shape == (2, 3)


# ------------------------------------------------------------------------------------------------ exporter (pxr)
def _arena(tmp_path, name="a.usda", **kw):
    pytest.importorskip("pxr")
    from isaac_net.tools.scene.synthetic import box_arena_usd
    path = str(tmp_path / name)
    box_arena_usd(path, **kw)
    return path


def _corners(scene_dir, mat):
    """Unique vertices, sorted."""
    v, _ = read_ply(os.path.join(scene_dir, "meshes", f"{mat}.ply"))
    return np.unique(np.round(v.astype(np.float64), 3), axis=0)


def test_export_synthetic_arena(tmp_path):
    from isaac_net.tools.scene.usd_export import export_usd
    usd = _arena(tmp_path, L=40, W=20, H=6, wall_x=20.0, wall_y1=14.0)
    r = export_usd(usd, str(tmp_path / "s"))
    assert r.n_prims == 6 and r.n_triangles == 2 + 5 * 12
    assert r.materials == {"concrete": {"prims": 6, "triangles": 62, "thickness": 0.1}}
    assert r.sources["semantic"]["prims"] == 6 and r.coverage() == 1.0
    assert np.allclose(r.bounds_min, [-0.2, -0.2, 0.0]) and np.allclose(r.bounds_max, [40.2, 20.2, 6.0])
    v, f = read_ply(os.path.join(r.out_dir, "meshes", "concrete.ply"))
    # the inner wall: a 0.2 m box at x = 20 from y = 0 to 14
    inner = v[(np.abs(v[:, 0] - 20.0) <= 0.1 + 1e-5) & (v[:, 1] <= 14.0 + 1e-5)]
    assert len(inner) >= 8 and np.isclose(inner[:, 0].min(), 19.9) and np.isclose(inner[:, 1].max(), 14.0)
    man = json.load(open(os.path.join(r.out_dir, "manifest.json")))
    assert man["scene_hash"] == r.scene_hash and len(man["prims"]) == 6


def test_export_units_up_axis_frame_offset(tmp_path):
    from isaac_net.tools.scene.usd_export import export_usd
    a = export_usd(_arena(tmp_path, "z.usda"), str(tmp_path / "z"))
    b = export_usd(_arena(tmp_path, "y.usda", meters_per_unit=0.01, up_axis="Y"), str(tmp_path / "y"))
    assert b.meters_per_unit == 0.01 and b.up_axis == "Y"
    # the same surfaces (quads may be split along the other diagonal, so compare corners, counts and bounds)
    assert np.allclose(_corners(a.out_dir, "concrete"), _corners(b.out_dir, "concrete"), atol=2e-4)
    assert a.n_triangles == b.n_triangles and np.allclose(a.bounds_min, b.bounds_min, atol=1e-4)
    assert np.allclose(a.bounds_max, b.bounds_max, atol=1e-4)
    c = export_usd(_arena(tmp_path, "o.usda"), str(tmp_path / "o"), offset_m=(5.0, -2.0, 0.0))
    assert np.allclose(np.array(c.bounds_min) - a.bounds_min, [5.0, -2.0, 0.0], atol=1e-5)
    assert c.scene_hash != a.scene_hash
    # frame: move the whole arena under an Xform at (100, 50) and export in that Xform's frame
    from pxr import Gf, Sdf, Usd, UsdGeom
    st = Usd.Stage.Open(_arena(tmp_path, "f.usda"))
    env = UsdGeom.Xform.Define(st, "/Env")
    env.AddTranslateOp().Set(Gf.Vec3d(100.0, 50.0, 0.0))
    Sdf.CopySpec(st.GetRootLayer(), "/World", st.GetRootLayer(), "/Env/Arena")
    st.RemovePrim("/World")
    d = export_usd(st, str(tmp_path / "f"), root="/Env", frame="/Env")
    assert np.allclose(d.bounds_min, a.bounds_min, atol=1e-4) and d.scene_hash == a.scene_hash
    w = export_usd(st, str(tmp_path / "w"), root="/Env")
    assert np.allclose(np.array(w.bounds_min) - a.bounds_min, [100.0, 50.0, 0.0], atol=1e-4)


def test_export_filters_and_mapping(tmp_path):
    pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom

    from isaac_net.tools.scene.usd_export import export_usd
    usd = _arena(tmp_path, extra={"Shelf_01": (10, 5, 1, 1, 4, 2, "rack"), "Box_02": (30, 5, 0.5, 1, 1, 1, "")})
    r = export_usd(usd, str(tmp_path / "a"))
    assert r.materials["metal"]["prims"] == 1 and r.materials["wood"]["prims"] == 1
    assert r.sources["semantic"]["prims"] == 7 and r.sources["name"]["prims"] == 1
    r = export_usd(usd, str(tmp_path / "b"), material_map={"/world/box*": "glass", "rack": "wood"})
    assert r.materials["glass"]["prims"] == 1 and r.materials["wood"]["prims"] == 1 and r.sources["user"]["prims"] == 2
    r = export_usd(usd, str(tmp_path / "c"), exclude=["*/wall*"])
    assert r.n_prims == 4 and "Floor" in " ".join(p[0] for p in r.prims)
    r = export_usd(usd, str(tmp_path / "d"), include=["/world/shelf*"])
    assert r.n_prims == 1
    r = export_usd(usd, str(tmp_path / "e"), crop=(25, 0, 45, 20), z_max=3.0)
    assert r.n_prims < 8
    st = Usd.Stage.Open(usd)
    UsdGeom.Imageable(st.GetPrimAtPath("/World/Shelf_01")).MakeInvisible()
    r = export_usd(st, str(tmp_path / "f"))
    assert "metal" not in r.materials and r.skipped.get("invisible") == 1
    base = export_usd(usd, str(tmp_path / "g")).scene_hash
    st = Usd.Stage.Open(usd)
    UsdGeom.Xformable(st.GetPrimAtPath("/World/Box_02")).GetOrderedXformOps()[0].Set((30.0, 6.0, 0.5))
    assert export_usd(st, str(tmp_path / "h")).scene_hash != base


def test_export_point_instancer_and_gprims(tmp_path):
    pytest.importorskip("pxr")
    from pxr import Gf, Usd, UsdGeom, Vt

    from isaac_net.tools.scene.usd_export import export_usd
    st = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(st, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(st, 1.0)
    pi = UsdGeom.PointInstancer.Define(st, "/World/Racks")
    proto = UsdGeom.Cube.Define(st, "/World/Racks/Protos/RackCube")
    proto.GetSizeAttr().Set(2.0)
    pi.GetPrototypesRel().SetTargets([proto.GetPath()])
    pi.GetProtoIndicesAttr().Set(Vt.IntArray([0, 0]))
    pi.GetPositionsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(10, 0, 1), Gf.Vec3f(-10, 0, 1)]))
    UsdGeom.Cylinder.Define(st, "/World/Pillar").GetHeightAttr().Set(4.0)
    UsdGeom.Sphere.Define(st, "/World/Ball")
    r = export_usd(st, str(tmp_path / "s"))
    assert r.materials["metal"]["triangles"] == 24 and r.materials["metal"]["prims"] == 2
    v, f = read_ply(os.path.join(r.out_dir, "meshes", "metal.ply"))
    assert np.allclose(sorted(set(np.round(v[:, 0], 4))), [-11, -9, 9, 11])
    assert r.materials["concrete"]["prims"] == 2                      # pillar (name) + ball (default)
    assert np.isclose(r.bounds_max[2], 2.0) and np.isclose(r.bounds_min[2], -2.0)


def test_cli_export_only(tmp_path, capsys):
    from isaac_net.tools.scene.bake import main
    usd = _arena(tmp_path)
    main(["--usd", usd, "--scene-dir", str(tmp_path / "s"), "--export-only", "--map", "*/innerwall=glass",
          "--thickness", "glass=0.01"])
    txt = open(tmp_path / "s" / "scene.xml").read()
    assert 'id="glass"' in txt and 'value="0.01"' in txt
    assert "exported" in capsys.readouterr().out
