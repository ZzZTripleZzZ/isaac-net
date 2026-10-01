"""Bake validation against closed-form channels (needs pxr / usd-core and sionna-rt; skipped without them).

Scenes with known geometry go through the whole pipeline (USD -> export -> Sionna RT RadioMapSolver -> map file) and
the map is compared at points with free-space loss and ITU-R P.2040 single-slab transmission:

  * free space plus one wall (a two-sided quad of concrete, slab 0.1 m, at x = 20): in line of sight the map equals
    free space, behind the wall free space plus the slab loss;
  * the same wall as a closed 0.2 m box: a ray crosses two faces, so the loss is two slabs;
  * a box arena with a floor and walls: reflections add power, so the map lies at or above free space;
  * the LOS map is 1 where the straight segment to the gNB is clear and 0 behind the wall.

Tolerances: in line of sight single cells (0.5 m, 4e6 rays) are within 1 dB of free space; behind the wall the map
is power-averaged over 3 x 3 cells against the same average of free space, because few rays reach one cell there.
The incidence is within 15 degrees of normal, where the slab loss changes by less than 0.5 dB.
"""
import os

import numpy as np
import pytest

VARIANT = os.environ.get("ISAAC_NET_SIONNA_VARIANT", "llvm")   # llvm = CPU; cuda needs OptiX
pytest.importorskip("pxr")
pytest.importorskip("mitsuba").set_variant(f"{VARIANT}_ad_mono_polarized")   # before Sionna registers its plugins
pytest.importorskip("sionna.rt")

from isaac_net.tools.scene.bake import bake_scene  # noqa: E402
from isaac_net.tools.scene.materials import free_space_gain_db, slab_transmission_db  # noqa: E402
from isaac_net.tools.scene.synthetic import box_arena_usd  # noqa: E402
from isaac_net.tools.scene.usd_export import export_usd  # noqa: E402

pytestmark = pytest.mark.slow

FC = 3.5
TX = (10.0, 10.0, 4.0)
UE_H = 1.5
LOS_PTS = [(4.75, 9.75), (14.75, 9.75), (15.25, 12.25), (6.25, 4.25)]
NLOS_PTS = [(26.25, 9.75), (30.25, 10.25), (34.75, 9.75), (29.75, 12.25)]


def _bake(tmp_path, name, **kw):
    usd = str(tmp_path / f"{name}.usda")
    box_arena_usd(usd, L=40, W=20, H=6, **kw)
    r = export_usd(usd, str(tmp_path / name))
    m = bake_scene(r.xml_path, [TX], FC, (0, 0, 40, 20), cell=0.5, ue_h=UE_H, samples=4_000_000, depth=4,
                   los_map=True, variant=VARIANT, fill=False)
    return m


def _at(m, pts, key="gain_db", k=1):
    """(map, free space) at the cells nearest to pts; with k > 1 both are power-averaged over k x k cells, which
    averages out the solver's Monte-Carlo noise (about 20 rays reach a 0.5 m cell 20 m from the gNB)."""
    x0, y0, x1, y1 = m["bounds"]
    g = m[key][0]
    H, W = g.shape
    xs, ys = np.linspace(x0, x1, W), np.linspace(y0, y1, H)
    out = []
    r = k // 2
    for px, py in pts:
        j, i = int(np.argmin(abs(xs - px))), int(np.argmin(abs(ys - py)))
        J, I = np.meshgrid(xs[j - r:j + r + 1], ys[i - r:i + r + 1])
        d = np.sqrt((J - TX[0]) ** 2 + (I - TX[1]) ** 2 + (UE_H - TX[2]) ** 2)
        fs = free_space_gain_db(d, FC)
        v = g[i - r:i + r + 1, j - r:j + r + 1]
        if key == "gain_db":
            out.append((10 * np.log10(np.mean(10 ** (v / 10))), 10 * np.log10(np.mean(10 ** (fs / 10)))))
        else:
            out.append((float(v.mean()), 0.0))
    return np.array(out)


def test_single_wall_plane(tmp_path):
    m = _bake(tmp_path, "plane", floor=False, arena_walls=False, wall_kind="plane")
    slab = slab_transmission_db("concrete", 0.1, FC)
    los = _at(m, LOS_PTS)
    assert np.all(np.abs(los[:, 0] - los[:, 1]) < 1.0), los
    nlos = _at(m, NLOS_PTS, k=3)
    assert np.all(np.abs(nlos[:, 0] - (nlos[:, 1] - slab)) < 1.0), (nlos, slab)
    assert np.all(_at(m, LOS_PTS, "los_prob")[:, 0] == 1.0) and np.all(_at(m, NLOS_PTS, "los_prob")[:, 0] == 0.0)


def test_single_wall_box_two_faces(tmp_path):
    m = _bake(tmp_path, "box", floor=False, arena_walls=False, wall_kind="box", wall_t=0.2)
    slab = slab_transmission_db("concrete", 0.1, FC)
    nlos = _at(m, NLOS_PTS, k=3)
    assert np.all(np.abs(nlos[:, 0] - (nlos[:, 1] - 2 * slab)) < 1.5), (nlos, 2 * slab)


def test_arena_reflections_add_power(tmp_path):
    m = _bake(tmp_path, "arena", wall_kind="box", wall_y1=14.0)
    los = _at(m, LOS_PTS)
    assert np.all(los[:, 0] > los[:, 1] - 0.5) and np.all(los[:, 0] < los[:, 1] + 4.0), los
    assert m["valid"].all()
    # around the end of the inner wall (y > 14) the path is clear
    assert _at(m, [(30.25, 19.25)], "los_prob")[0, 0] == 1.0
