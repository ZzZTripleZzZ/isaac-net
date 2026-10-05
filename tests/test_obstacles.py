"""Obstacle stack (docs/obstacles.md): geometric LOS state (map / raycast / callback), knife-edge diffraction,
TR 38.901 blockage models A and B, soft LOS, the RadioMC interface (los_state / blocked_state), the engine step keys,
the config gating and the Isaac layer. CPU only; the synthetic hall (tools/make_synthetic_radio_map.py) stands in for
a baked scene, so nothing needs USD or Sionna.

Validation items of the design: (a) raycast LOS == baked los_prob at its 0 / 1 points, (b) LOS fraction vs distance
in random InF clutter decays as exp(-d / k_subsce), (c) J(v) shape, (d) human screen loss, (e) model A statistics,
(f) soft LOS continuity.
"""
import math

import numpy as np
import pytest
import torch

from isaac_net.core import NRConfig, Requests, make_engine
from isaac_net.core.channels import tr38901 as tr
from isaac_net.core.channels.blockage import BlockageA, screen_loss_db
from isaac_net.core.channels.los import knife_edge_db, raycast
from isaac_net.core.channels.models import soft_los
from isaac_net.core.channels.radio_map import RadioMap
from isaac_net.core.config import multicell
from isaac_net.core.proto.rng import CounterRNG
from isaac_net.core.radio import RadioMC
from isaac_net.tools.make_synthetic_radio_map import OBSTACLE_GNBS, make_obstacle_map
from isaac_net.tools.scene.heightmap import box_triangles, height_map

LAM35 = 3e8 / 3.5e9
GXY = tuple((g[0], g[1]) for g in OBSTACLE_GNBS)


@pytest.fixture(scope="module")
def hall(tmp_path_factory):
    m = make_obstacle_map()
    p = str(tmp_path_factory.mktemp("maps") / "hall.npz")
    m.save(p)
    return m, p


def _cfg(path, **kw):
    base = dict(radio_map_path=path, n_cells=2, cell_layout="custom", cell_positions_m=GXY)
    base.update(kw)
    return NRConfig(**base)


def _grid_points(m):
    x0, y0, x1, y1 = m.bounds
    Y, X = torch.meshgrid(torch.linspace(y0, y1, m.H), torch.linspace(x0, x1, m.W), indexing="ij")
    return X.reshape(-1), Y.reshape(-1)


# ------------------------------------------------------------------------------------------------ map file
def test_radio_map_grids_roundtrip_and_backward_compat(hall, tmp_path):
    m, p = hall
    r = RadioMap.load(p)
    assert r.los_prob.shape == (2, m.H * m.W) and r.obstacle_z.shape == (1, m.H * m.W)
    assert "los_prob" not in r.meta and "obstacle_z" not in r.meta
    assert torch.equal(r.los_prob, m.los_prob) and torch.equal(r.obstacle_z, m.obstacle_z)
    old = RadioMap(m.gain.view(2, m.H, m.W), m.bounds, meta={"source": "old"})      # a file without the grids
    q = str(tmp_path / "old.npz")
    old.save(q)
    r2 = RadioMap.load(q)
    assert r2.los_prob is None and r2.obstacle_z is None
    pos = torch.rand(3, 4, 2) * 20
    assert torch.equal(r2.sample(pos), m.sample(pos))


def test_height_map_rasterizer():
    v, f = box_triangles([(2.0, 2.0, 0.0, 4.0, 3.0, 2.5)])
    z = height_map(v, f, (0, 0, 10, 5), 21, 41)                  # 0.25 m grid
    assert z.max() == pytest.approx(2.5) and z[10, 12] == pytest.approx(2.5) and z[0, 0] == 0.0
    assert z[5, 30] == 0.0
    # a vertical two-sided wall (zero footprint) is still seen through its top edge
    wv = np.array([[5.0, 0.0, 0.0], [5.0, 5.0, 0.0], [5.0, 5.0, 3.0], [5.0, 0.0, 3.0]])
    wf = np.array([[0, 1, 2], [0, 2, 3]])
    zw = height_map(wv, wf, (0, 0, 10, 5), 21, 41)
    assert zw[:, 20].min() == pytest.approx(3.0) and zw[:, 18].max() == 0.0


# ------------------------------------------------------------------------------------------------ (a) raycast
def test_raycast_equals_baked_los_prob(hall):
    """(a) At grid points where the exact bake says all-LOS (1.0) or all-NLOS (0.0) the 2.5-D ray march agrees,
    with samples finer than the racks' depth (64 samples on links up to 36 m: 0.56 m < 1 m racks)."""
    m, _ = hall
    X, Y = _grid_points(m)
    p0 = torch.stack([X, Y, torch.full_like(X, 1.5)], -1)[:, None]
    g = torch.tensor(OBSTACLE_GNBS)[None]
    blocked, clear, _ = raycast(p0, g, m.sample_height, 64)
    lp = m.los_prob.t()
    sure = (lp == 0) | (lp == 1)
    assert sure.float().mean() > 0.9
    assert torch.equal(~blocked[sure], lp[sure] == 1)
    assert (clear[blocked] < 0).all()


def test_raycast_source_in_radio(hall):
    """los_source="raycast" through RadioMC on every channel: the state follows the ray march; radio_map adds no
    path loss (the map holds the NLOS loss), log_distance adds nlos_extra_loss_db, tr38901 selects PL_LOS / NLOS."""
    m, p = hall
    pos = torch.tensor([[[20.0, 6.5], [20.0, 12.0], [20.0, 2.0], [6.0, 12.0]]])         # aisle centre, behind racks
    base = RadioMC(_cfg(p, channel="radio_map"), 1, "cpu", R=4)
    rad = RadioMC(_cfg(p, channel="radio_map", los_source="raycast", los_raycast_samples=64), 1, "cpu", R=4)
    assert torch.equal(rad.pathgain_db(pos), base.pathgain_db(pos))                     # no double counting
    los = rad.los_state()
    want = ~raycast(torch.cat([pos, torch.full_like(pos[..., :1], 1.5)], -1)[:, :, None],
                    torch.tensor(OBSTACLE_GNBS)[None, None], m.sample_height, 64)[0]
    assert los.shape == (1, 4, 2) and torch.equal(los, want) and los.any() and (~los).any()
    ld0 = RadioMC(_cfg(p, shadow_sigma_db=0.0), 1, "cpu", R=4)
    ld = RadioMC(_cfg(p, shadow_sigma_db=0.0, los_source="raycast", los_raycast_samples=64, nlos_extra_loss_db=12.0),
                 1, "cpu", R=4)
    assert torch.allclose(ld0.pathgain_db(pos) - ld.pathgain_db(pos), 12.0 * (~los).float())
    cfg = _cfg(p, channel="tr38901_inf_sl", gnb_height_m=6.0, los_source="raycast", los_raycast_samples=64,
               shadow_sigma_db=0.0)
    t = RadioMC(cfg, 1, "cpu", R=4)
    pg = t.pathgain_db(pos)
    assert torch.equal(t.los_state(), los)          # tr38901 path: the same geometric state
    d2 = (pos[:, :, None] - torch.tensor(GXY)).norm(dim=-1)
    d3 = torch.sqrt(d2 ** 2 + 4.5 ** 2)
    a = ("InF-SL", d2, d3, 3.5, 6.0, 1.5)
    s_l, s_n = tr.sigma_sf("InF-SL")
    sf = torch.where(los, s_l * t.ch.sf_los(pos), s_n * t.ch.sf_nlos(pos))
    assert torch.allclose(-pg, torch.where(los, tr.pl_los(*a), tr.pl_nlos(*a)) + sf, atol=1e-4)


# ------------------------------------------------------------------------------------------------ map source
def test_map_source_statistics_and_consistency(hall):
    m, p = hall
    lp = m.los_prob[0].view(m.H, m.W)
    i, j = torch.nonzero((lp > 0.3) & (lp < 0.7))[0].tolist()
    x0, y0, x1, y1 = m.bounds
    pt = torch.tensor([x0 + j * (x1 - x0) / (m.W - 1), y0 + i * (y1 - y0) / (m.H - 1)])
    E = 4000
    cfg = _cfg(p, channel="radio_map", los_source="map")
    rad = RadioMC(cfg, E, "cpu", R=1, rng=CounterRNG(3, E, "cpu"))
    pos = pt.expand(E, 1, 2).clone()
    rad.pathgain_db(pos)
    frac = rad.los_state()[:, 0, 0].float().mean().item()
    assert abs(frac - float(lp[i, j])) < 0.03
    first = rad.los_state().clone()
    rad.pathgain_db(pos)
    assert torch.equal(rad.los_state(), first) and int(rad.los_st.n_trans.sum()) == 0     # still robot: same state
    rad6 = RadioMC(cfg, 6, "cpu", R=1, rng=CounterRNG(3, 6, "cpu"))           # keyed by env id: E-invariant
    rad6.pathgain_db(pos[:6])
    assert torch.equal(rad6.los_state(), first[:6])
    # points with los_prob 0 / 1 are always NLOS / LOS
    X, Y = _grid_points(m)
    r2 = RadioMC(cfg, 1, "cpu", rng=CounterRNG(1, 1, "cpu"))
    r2.pathgain_db(torch.stack([X, Y], -1)[None])
    st, lpt = r2.los_state()[0], m.los_prob.t()
    assert torch.equal(st[lpt == 1], torch.ones_like(st[lpt == 1])) and not st[lpt == 0].any()


# ------------------------------------------------------------------------------------------------ (b) InF clutter
@pytest.mark.parametrize("r,dc", [(0.4, 2.0), (0.2, 3.0)])
def test_raycast_inf_clutter_follows_k_subsce(r, dc):
    """(b) Random InF clutter (Boolean model: squares of side d_clutter, Poisson centres, area density r, height
    2 m above the 1.5 m antennas). Along a grid axis, a segment between two clutter-free endpoints is clear with
    probability exp(-lambda dc (d - dc)) = exp(-(d - dc) / k_subsce), k_subsce = -dc / ln(1 - r): the decay length is
    exactly TR 38.901's (the offset dc is the conditioning on clear endpoints)."""
    g = np.random.default_rng(0)
    A, cell = 160.0, 0.1
    n = int(A / cell) + 1
    lam = -math.log(1 - r) / dc ** 2
    cx, cy = g.uniform(-dc, A + dc, (2, g.poisson(lam * (A + 2 * dc) ** 2)))
    z = np.zeros((n, n), np.float32)
    for x, y in zip(cx, cy):
        i0, i1 = max(int(math.ceil((y - dc / 2) / cell)), 0), min(int(math.floor((y + dc / 2) / cell)), n - 1)
        j0, j1 = max(int(math.ceil((x - dc / 2) / cell)), 0), min(int(math.floor((x + dc / 2) / cell)), n - 1)
        if i0 <= i1 and j0 <= j1:
            z[i0:i1 + 1, j0:j1 + 1] = 2.0
    assert abs((z > 0).mean() - r) < 0.03
    m = RadioMap(np.zeros((1, n, n), np.float32), (0, 0, A, A), obstacle_z=z)
    k = tr.inf_k_subsce("InF-SL", 1.5, 1.5, r=r, d_clutter=dc, h_c=2.0)
    assert k == pytest.approx(-dc / math.log(1 - r))
    ds = np.arange(dc + 1.0, dc + 2.5 * k, max(k / 3, 0.5))
    frac = []
    for d in ds:
        p0 = torch.tensor(g.uniform(5, A - 5 - d, (20000, 2)), dtype=torch.float32)
        p1 = p0 + torch.tensor([d, 0.0])
        ok = (m.sample_height(p0) == 0) & (m.sample_height(p1) == 0)
        h = torch.full_like(p0[:, :1], 1.5)
        blk, _, _ = raycast(torch.cat([p0, h], -1), torch.cat([p1, h], -1), m.sample_height, 128)
        frac.append((~blk[ok]).float().mean().item())
    frac = np.asarray(frac)
    slope = np.polyfit(ds, np.log(frac), 1)[0]
    assert -1 / slope == pytest.approx(k, rel=0.15)
    assert np.allclose(frac, np.exp(-(ds - dc) / k), atol=0.05)


# ------------------------------------------------------------------------------------------------ (c) knife edge
def test_knife_edge_shape():
    v = torch.linspace(-3, 5, 801)
    j = knife_edge_db(v)
    assert knife_edge_db(torch.tensor(0.0)).item() == pytest.approx(6.03, abs=0.01)
    assert knife_edge_db(torch.tensor(-1.0)).item() == 0.0 and knife_edge_db(torch.tensor(-3.0)).item() == 0.0
    assert (j[1:] >= j[:-1]).all() and (j >= 0).all()
    assert knife_edge_db(torch.tensor(-0.78 + 1e-4)).item() < 0.02                  # continuous at the cut-off
    assert knife_edge_db(torch.tensor(2.4)).item() == pytest.approx(20.5, abs=0.5)   # P.526 Fig. 9 reading


def test_diffraction_ramp_at_aisle_end():
    """A robot sliding past the end of a 6 m tall rack (vertical edge) toward its shadow: with los_diffraction
    the loss rises smoothly from 0 through about 6 dB at the geometric boundary to the NLOS cap over about one
    Fresnel radius, instead of a step. (Under a 3 m rack the over-the-top path caps the loss near 23 dB.)"""
    W, H = 161, 81
    z = np.zeros((H, W), np.float32)
    xs = np.linspace(0, 40, W)
    ys = np.linspace(0, 20, H)
    z[np.ix_((ys >= 9.0) & (ys <= 10.0), (xs >= 0) & (xs <= 20.0))] = 6.0     # rack ends at x = 20
    gain = np.zeros((1, H, W), np.float32)
    m = RadioMap(gain, (0, 0, 40, 20), obstacle_z=z)
    cfg = NRConfig(channel="log_distance", shadow_sigma_db=0.0, cell_positions_m=((20.0, 19.0),),
                   gnb_height_m=1.5, los_source="raycast", los_raycast_samples=64, los_diffraction=True,
                   nlos_extra_loss_db=30.0)
    ref = RadioMC(cfg.with_(los_source="stochastic", los_diffraction=False), 1, "cpu", R=1)
    rad = RadioMC(cfg, 1, "cpu", R=1, radio_map=m)
    xr = torch.linspace(25.0, 10.0, 301)                     # robot at y = 1: link to (20, 19) crosses y = 9.5
    loss = []
    for x in xr:
        p = torch.tensor([[[float(x), 1.0]]])
        loss.append((ref.pathgain_db(p) - rad.pathgain_db(p)).item())
    loss = torch.tensor(loss)
    assert loss[0].item() == pytest.approx(0.0, abs=0.05) and loss[-1].item() == pytest.approx(30.0, abs=0.5)
    assert (loss[1:] - loss[:-1] >= -0.05).all()             # monotone
    assert (loss[1:] - loss[:-1]).max() < 2.0                # no step: at most 2 dB per 5 cm
    mid = (loss > 3) & (loss < 9)
    assert mid.any()                                         # passes through the 6 dB grazing value
    ramp = xr[(loss > 0.5) & (loss < 25)]
    assert 0.2 < (ramp.max() - ramp.min()).item() < 5.0      # about a Fresnel radius scale (0.9 m at mid-link)


# ------------------------------------------------------------------------------------------------ (d) model B
def test_screen_loss_human_and_vehicle():
    """(d) TR 38.901 model B, eq. 7.6-29/30: a 0.3 x 1.7 m human 1 m from the robot on a 20 m horizontal link at
    1.5 m. Hand value at 3.5 GHz: F_w1 = F_w2 = 0.255, F_h2 (top, 0.2 m above) = 0.300, F_h1 (floor) = 0.464 ->
    L = -20 log10(1 - 0.764 * 0.510) = 4.3 dB. A 0.3 m body is about one Fresnel radius wide here (0.29 m), so
    the 10-25 dB of mmWave body loss is not reached at 3.5 GHz; at 28 GHz the same geometry gives about 11 dB."""
    a = torch.tensor([[[0.0, 0.0, 1.5]]])
    g = torch.tensor([[20.0, 0.0, 1.5]])
    act = torch.ones(1, 1, dtype=torch.bool)
    hum = torch.tensor([[[0.3, 1.7]]])
    out = []
    for q in (0.0, 0.15, 0.3, 0.6, 1.2):
        loss, hit = screen_loss_db(a, g, torch.tensor([[[1.0, q]]]), hum, act, LAM35, 40.0)
        out.append(loss.item())
        assert hit.item() == (q < 0.15)
    assert out[0] == pytest.approx(4.29, abs=0.05)
    assert all(x > y for x, y in zip(out, out[1:])) and out[-1] < 0.1           # falls away from the path
    l28, _ = screen_loss_db(a, g, torch.tensor([[[1.0, 0.0]]]), hum, act, 3e8 / 28e9, 40.0)
    assert 10.0 < l28.item() < 25.0
    # a wide screen grazed at one side edge: 6 dB times the finite-height factor (F_h1 + F_h2 = 0.89 here, the
    # floor and the top 3.5 m above the path diffract too): -20 log10(1 - 0.5 * 0.89) = 5.1 dB
    wide = torch.tensor([[[20.0, 5.0]]])
    lg, _ = screen_loss_db(a, g, torch.tensor([[[10.0, 10.0]]]), wide, act, LAM35, 40.0)
    assert lg.item() == pytest.approx(5.1, abs=0.1)
    tall = torch.tensor([[[20.0, 500.0]]])                     # top edge gone: close to the knife-edge 6 dB
    lt, _ = screen_loss_db(a, g, torch.tensor([[[10.0, 10.0]]]), tall, act, LAM35, 40.0)
    assert lg.item() < lt.item() < knife_edge_db(torch.tensor(0.0)).item()     # only the floor edge differs
    lb, _ = screen_loss_db(a, g, torch.tensor([[[10.0, 0.0]]]), wide, act, LAM35, 40.0)
    assert lb.item() > 15.0                                   # floor and top edges leak: about 18 dB
    lc, _ = screen_loss_db(a, g, torch.tensor([[[10.0, 0.0]]]), wide, act, LAM35, 10.0)
    assert lc.item() == 10.0
    # behind the robot or past the gNB: no loss
    l0, _ = screen_loss_db(a, g, torch.tensor([[[-1.0, 0.0]]]), hum, act, LAM35, 40.0)
    assert l0.item() == 0.0


def test_screens_in_radio_and_engine_step():
    cfg = NRConfig(blockage=True, blockage_model="screen", shadow_sigma_db=0.0, cell_positions_m=((20.0, 0.0),),
                   gnb_height_m=1.5, blocker_size_m=((0.6, 1.8), (0.3, 1.7), (4.8, 1.4)))
    rad = RadioMC(cfg, 1, "cpu", R=2)
    off = RadioMC(cfg.with_(blockage=False), 1, "cpu", R=2)
    pos = torch.tensor([[[0.0, 0.0], [5.0, 0.0]]])                      # robot 1 stands on robot 0's link
    d = off.pathgain_db(pos) - rad.pathgain_db(pos)
    assert d[0, 0, 0] > 2.0 and d[0, 1, 0] == 0.0
    assert rad.blocked_state()[0, :, 0].tolist() == [True, False]
    blk = torch.tensor([[[2.0, 0.0, 1.0], [0.0, 0.0, -1.0]]])           # a human on robot 1's... robot 0's link, a gap
    d2 = off.pathgain_db(pos) - rad.pathgain_db(pos, blk)
    assert d2[0, 0, 0] > d[0, 0, 0] and d2[0, 1, 0] == 0.0
    with pytest.raises(ValueError, match="blockage_model='screen'"):
        off.pathgain_db(pos, blk)
    E, R = 2, 2
    eng = make_engine("L2", E, R, "cpu", cfg, seed=1)
    eng.submit(None, Requests(torch.ones(E, R, dtype=torch.long)))
    o = eng.step(None, pos.expand(E, R, 2), blockers=blk.expand(E, 2, 3))
    assert o["blocked"].shape == (E, R) and o["blocked"][:, 0].all() and "los" not in o
    with pytest.raises(ValueError, match="poses"):
        eng.step(None, torch.zeros(E, R), blockers=blk.expand(E, 2, 3))


# ------------------------------------------------------------------------------------------------ (e) model A
def test_model_a_statistics():
    """(e) Model A, indoor regions (x_k ~ U[15, 45] deg, y_k ~ U[5, 15] deg, r = 2 m), horizontal links: the share
    of time a link sits inside a region is near K E[x_k] / 360 = 1/3 (minus overlaps), and episodes last about the
    field's decorrelation time (d_corr / v = 5 m / 0.83 m/s = 6 s). The attenuation inside a region is small at
    3.5 GHz: for a 30 x 10 deg region at 2 m, F_A1 + F_A2 = 0.76 and F_Z1 + F_Z2 = 0.44, so about 3.5 dB (model A
    targets mmWave; at 28 GHz the same region gives about 10 dB)."""
    E, steps, dt = 400, 300, 0.1
    cfg = NRConfig(blockage=True, blockage_model="stochastic", cell_positions_m=((10.0, 0.0),), gnb_height_m=1.5)
    b = BlockageA(cfg, E, "cpu", rng=CounterRNG(5, E, "cpu"))
    a3 = torch.tensor([0.0, 0.0, 1.5]).expand(E, 1, 3)
    g3 = torch.tensor([[10.0, 0.0, 1.5]])
    hits, losses = [], []
    for _ in range(steps):
        b.advance(dt)
        loss, hit = b.loss_db(a3, g3, 40.0)
        hits.append(hit[:, 0, 0])
        losses.append(loss[:, 0, 0])
    hits, losses = torch.stack(hits), torch.stack(losses)
    share = hits.float().mean().item()
    assert 0.15 < share < 0.4
    assert 1.5 < losses[hits].mean() < 8.0 and losses[~hits].mean() < losses[hits].mean()
    assert (losses >= 0).all() and losses.mean() > share * losses[hits].mean() - 1e-6   # never a gain
    b28 = BlockageA(cfg.with_(carrier_ghz=28.0), E, "cpu", rng=CounterRNG(5, E, "cpu"))
    b28.advance(dt)
    l28, h28 = b28.loss_db(a3, g3, 40.0)
    b.t.zero_()
    b.advance(dt)
    l35, h35 = b.loss_db(a3, g3, 40.0)
    assert torch.equal(h28, h35) and l28[h28].mean() > l35[h35].mean() + 4.0
    starts = (hits[1:] & ~hits[:-1]).sum().item()
    dur = hits.sum().item() * dt / max(starts, 1)
    assert 0.5 < dur < 20.0
    # zenith outside the elevation spans (gNB well above): little loss
    l_hi, h_hi = b.loss_db(a3, torch.tensor([[3.0, 0.0, 8.0]]), 40.0)
    assert not h_hi.any() and l_hi.mean() < 3.0


def test_model_a_window():
    """TR 38.901 V17.0.0 text below eq. 7.6-22: a region attenuates only when |phi_AOA - phi_k| < x_k and
    |theta_ZOA - theta_k| < y_k, otherwise 0 dB."""
    from isaac_net.core.channels.fields import uniform_from_field
    E = 200
    cfg = NRConfig(blockage=True, blockage_model="stochastic", cell_positions_m=((10.0, 0.0),), gnb_height_m=1.5)
    b = BlockageA(cfg, E, "cpu", rng=CounterRNG(3, E, "cpu"))
    b.advance(0.5)
    a3 = torch.tensor([0.0, 0.0, 1.5]).expand(E, 1, 3)
    for g in ([[10.0, 0.0, 1.5]], [[0.0, 10.0, 1.5]], [[3.0, 0.0, 1.9]]):
        g3 = torch.tensor(g)
        loss, _ = b.loss_db(a3, g3, 40.0)
        d = g3[0] - a3[:, 0]
        az = torch.rad2deg(torch.atan2(d[:, 1], d[:, 0]))[:, None]
        zen = 90.0 - torch.rad2deg(torch.atan2(d[:, 2], d[:, :2].norm(dim=-1)))[:, None]
        phik = 360.0 * uniform_from_field(b.field(a3[..., :2] + (b.vel * b.t[:, None])[:, None, :]), b.table)[:, 0]
        da = torch.remainder(az - phik + 180.0, 360.0) - 180.0
        win = ((da.abs() < b.xk) & ((zen - 90.0).abs() < b.yk)).any(-1)
        assert (loss[:, 0, 0][~win] == 0).all() and (loss[:, 0, 0][win] > 0).all()
        assert win.any() and (~win).any()


def test_model_a_in_radio_reset_keyed():
    cfg = NRConfig(blockage=True, blockage_model="stochastic", gnb_height_m=1.5)
    r3 = RadioMC(cfg, 3, "cpu", R=2, rng=CounterRNG(9, 3, "cpu"))
    r6 = RadioMC(cfg, 6, "cpu", R=2, rng=CounterRNG(9, 6, "cpu"))
    pos = torch.rand(6, 2, 2) * 50
    assert torch.equal(r3.pathgain_db(pos[:3]), r6.pathgain_db(pos)[:3])
    assert r3.blocked_state().shape == (3, 2, 1)


# ------------------------------------------------------------------------------------------------ (f) soft LOS
def test_soft_los_continuous():
    p = torch.linspace(0.001, 0.999, 2001)
    for u in (0.1, 0.5, 0.9):
        s = soft_los(torch.full_like(p, u), p, LAM35)
        assert (s[1:] >= s[:-1]).all() and (s[1:] - s[:-1]).abs().max() < 0.05
        assert ((s > 0.5) == (torch.full_like(p, u) < p)).all()        # same state as the hard threshold
    hard = soft_los(torch.full_like(p, 0.5), p, 1e-9)
    assert torch.equal(hard > 0.5, p > 0.5) and ((hard < 0.01) | (hard > 0.99))[(p - 0.5).abs() > 0.01].all()


def test_soft_los_in_tr38901_blends():
    cfg = NRConfig(channel="tr38901_inf_dl", shadow_sigma_db=0.0, los_soft=True)
    hard = RadioMC(cfg.with_(los_soft=False), 4, "cpu", R=1, rng=CounterRNG(2, 4, "cpu"))
    soft = RadioMC(cfg, 4, "cpu", R=1, rng=CounterRNG(2, 4, "cpu"))
    xs = torch.linspace(1, 20, 400)
    pos = torch.stack([xs, torch.zeros_like(xs)], -1)[None].expand(4, -1, -1).contiguous()
    pg_h, pg_s = hard.pathgain_db(pos)[..., 0], soft.pathgain_db(pos)[..., 0]
    assert torch.equal(hard.los_state(), soft.los_state())
    w = soft.ch.soft_w[..., 0]
    assert ((w > 0) & (w < 1)).all()
    agree = (w > 0.99) | (w < 0.01)
    assert (pg_h - pg_s)[agree].abs().max() < 1.0
    assert (pg_s[:, 1:] - pg_s[:, :-1]).abs().max() <= (pg_h[:, 1:] - pg_h[:, :-1]).abs().max() + 1e-4


# ------------------------------------------------------------------------------------------------ callback
def test_callback_source_and_engine_keys():
    """los_source="callback": the blocked_fn of the Isaac layer drives the engine radio of a multi-cell config."""
    cfg = multicell(3, los_source="callback", nlos_extra_loss_db=20.0)
    E, R = 2, 3

    def blocked_fn(p):                                       # cell 0 blocked for robots left of x = 75
        b = torch.zeros(p.shape[0], p.shape[1], 3, dtype=torch.bool)
        b[..., 0] = p[..., 0] < 75
        return b

    ref = make_engine("L2", E, R, "cpu", multicell(3), seed=4)
    eng = make_engine("L2", E, R, "cpu", cfg, seed=4)
    with pytest.raises(RuntimeError, match="callback"):
        eng.step(None, torch.full((E, R, 2), 50.0))
    eng = make_engine("L2", E, R, "cpu", cfg, seed=4)
    eng.set_los_callback(blocked_fn)
    pos = torch.tensor([[[10.0, 50.0], [100.0, 50.0], [60.0, 80.0]]]).expand(E, R, 2)
    o_ref, o = ref.step(None, pos), eng.step(None, pos)
    los = eng.radio.los_state()
    assert torch.equal(los[..., 0], ~(pos[..., 0] < 75)) and los[..., 1:].all()
    pg_ref = ref.radio.pathgain_db(pos)
    assert torch.allclose(pg_ref - eng.radio.pathgain_db(pos), 20.0 * (~los).float())
    assert "los" in o and "los" not in o_ref and "blocked" not in o
    assert torch.equal(o["los"], torch.gather(los, 2, o["serving_cell"][..., None])[..., 0])
    with pytest.raises(ValueError, match="callback"):
        ref.set_los_callback(blocked_fn)


def test_step_keys_unchanged_when_off():
    E, R = 2, 2
    eng = make_engine("L2", E, R, "cpu", NRConfig(channel="tr38901_inf_sh"), seed=1)
    o = eng.step(None, torch.rand(E, R, 2) * 50)
    assert "los" not in o and "blocked" not in o
    assert eng.radio.los_state().shape == (E, R, 1) and eng.radio.blocked_state() is None
    assert RadioMC(NRConfig(), E, "cpu").los_state() is None


# ------------------------------------------------------------------------------------------------ config
def test_obstacle_fields_gated():
    assert NRConfig(los_source="raycast", radio_map_path="m.npz", los_diffraction=True,
                    los_raycast_samples=16).unused_fields("L2") == []
    assert NRConfig(los_raycast_samples=16).unused_fields("L2") == ["los_raycast_samples"]
    assert NRConfig(nlos_extra_loss_db=5.0).unused_fields("L2") == ["nlos_extra_loss_db"]
    assert NRConfig(los_source="map", radio_map_path="m.npz", nlos_extra_loss_db=5.0).unused_fields("L2") == []
    assert NRConfig(channel="tr38901", los_soft=True).unused_fields("L2") == []
    assert NRConfig(channel="tr38901", los_source="map", radio_map_path="m", tr38901_los="los").unused_fields(
        "L2") == ["tr38901_los"]
    assert NRConfig(blockage_model="screen").unused_fields("L2") == ["blockage_model"]
    assert NRConfig(blockage=True, blockage_model="screen", blocker_size_m=((1.0, 1.0),), blockage_max_db=30.0,
                    ).unused_fields("L2") == []
    assert NRConfig(blockage=True, blockage_model="stochastic", blockage_loss_db=3.0,
                    blocker_size_m=((1.0, 1.0),)).unused_fields("L2") == ["blockage_loss_db", "blocker_size_m"]
    assert NRConfig(blockage=True, blockage_max_db=30.0).unused_fields("L2") == ["blockage_max_db"]
    with pytest.raises(AssertionError):
        NRConfig(los_diffraction=True)
    with pytest.raises(AssertionError):
        NRConfig(los_soft=True)
    with pytest.raises(AssertionError):
        NRConfig(los_source="lidar")
    with pytest.raises(ValueError, match="obstacle_z"):
        RadioMC(NRConfig(los_source="raycast", radio_map_path="synthetic"), 1, "cpu")


# ------------------------------------------------------------------------------------------------ Isaac layer
def test_isaac_los_feature_and_blockers(hall):
    from isaac_net.isaac import IsaacNetCfg, NetModule
    from isaac_net.isaac.config import obs_dim
    _, p = hall
    E, R = 2, 2
    cfg = _cfg(p, channel="radio_map", los_source="raycast", los_raycast_samples=64, blockage=True,
               blockage_model="screen")
    isc = IsaacNetCfg(radio="engine", obs_features=("sinr", "los", "blocked"))
    net = NetModule("L2", E, R, "cpu", cfg, isaac=isc, seed=3)
    assert net.obs_dim == obs_dim(("sinr", "los", "blocked"), cfg) == 3
    poses = torch.tensor([[[20.0, 6.5, 0.0], [20.0, 2.0, 0.0]]]).expand(E, R, 3)
    o = net.step(None, poses, blockers=torch.tensor([[[3.0, 12.0, 2.0]]]).expand(E, 1, 3))
    assert o["los"].dtype == torch.bool and o["los"].shape == (E, R)
    x = net.obs()
    assert torch.equal(x[..., 1], o["los"].float()) and torch.equal(x[..., 2], o["blocked"].float())
    inet = NetModule("L2", E, R, "cpu", NRConfig(), isaac=IsaacNetCfg(obs_features=("los",)), seed=3)
    with pytest.raises(ValueError, match="radio='engine'"):
        inet.step(None, poses, blockers=torch.zeros(E, 1, 3))
    o2 = inet.step(None, poses)
    assert torch.equal(o2["los"], ~o2["blocked"])


def test_isaac_engine_blocked_fn_callback():
    from isaac_net.isaac import IsaacNetCfg, NetModule
    E, R = 2, 2
    calls = []

    def blocked_fn(p):
        calls.append(p.shape)
        return torch.ones(p.shape[0], p.shape[1], 3, dtype=torch.bool)

    net = NetModule("L2", E, R, "cpu", multicell(3, los_source="callback", nlos_extra_loss_db=10.0),
                    isaac=IsaacNetCfg(radio="engine"), seed=1)
    o = net.step(None, torch.rand(E, R, 3) * 100, blocked_fn=blocked_fn)
    assert calls and not o["los"].any()
    plain = NetModule("L2", E, R, "cpu", multicell(3), isaac=IsaacNetCfg(radio="engine"), seed=1)
    with pytest.warns(UserWarning, match="callback"):
        plain.step(None, torch.rand(E, R, 3) * 100, blocked_fn=blocked_fn)
