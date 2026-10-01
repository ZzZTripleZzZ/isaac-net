"""Channel models (NRConfig.channel, core/channels/): bitwise default, shadowing correlation, TR 38.901 path loss and
LOS probability against hand-computed values, per-robot Doppler, radio-map sampling, blockage geometry, partial
resets and CUDA-graph capture."""
import math

import pytest
import torch

from isaac_net.core.channels import (RadioMap, blocked_links, install_per_robot_fading, make_synthetic_map,
                                        rho_per_ms_from_speed, synthetic_gnb_xy)
from isaac_net.core.channels import tr38901 as tr
from isaac_net.core.channels.fields import PlaneWaveField
from isaac_net.core.config import NRConfig, fading_rho_from_speed
from isaac_net.core.engine import make_engine
from isaac_net.core.radio import RadioMC

MAP_CFG = dict(channel="radio_map", radio_map_path="synthetic", n_cells=2,
               cell_positions_m=tuple(synthetic_gnb_xy()))


# ---------------------------------------------------------------- default: bitwise the pre-channel RadioMC
def _legacy_draw(n, C, K, g):
    ang = torch.rand(n, C, K, generator=g) * 2 * math.pi
    wl = 20 + 40 * torch.rand(n, C, K, generator=g)
    k = torch.stack([torch.cos(ang), torch.sin(ang)], -1) * (2 * math.pi / wl)[..., None]
    phi = torch.rand(n, C, K, generator=g) * 2 * math.pi
    return k.permute(1, 0, 2, 3).contiguous(), phi.permute(1, 0, 2).contiguous()


def _legacy_rx(cfg, gnb, k, phi, pos):
    amp = cfg.shadow_sigma_db * math.sqrt(2 / cfg.shadow_modes)
    pos = pos[..., :2]
    d = (pos[:, :, None, :] - gnb).norm(dim=-1).clamp(min=1.0)
    pl = cfg.pl_const_db + (10 * cfg.pathloss_exp) * torch.log10(d)
    arg = torch.einsum("erx,cekx->cerk", pos, k) + phi[:, :, None, :]
    sh = (amp * torch.cos(arg).sum(-1)).permute(1, 2, 0)
    return cfg.ue_tx_dbm - pl - sh


@pytest.mark.parametrize("cfg", [NRConfig(), NRConfig(n_cells=3, cell_layout="hex"),
                                 NRConfig(shadow_dcorr_m=10.3, fading_doppler="per_robot")])
def test_default_is_bitwise_legacy(cfg):
    E, R = 6, 5
    radio = RadioMC(cfg, E, "cpu", generator=torch.Generator().manual_seed(3))
    g = torch.Generator().manual_seed(3)
    k, phi = _legacy_draw(E, radio.C, cfg.shadow_modes, g)
    gnb = torch.tensor(cfg.gnb_xy())
    pos = torch.rand(E, R, 3, generator=torch.Generator().manual_seed(9)) * 150
    assert torch.equal(radio.k, k) and torch.equal(radio.phi, phi)
    assert torch.equal(radio.rx_dbm(pos), _legacy_rx(cfg, gnb, k, phi, pos))
    assert torch.equal(radio.pathgain_db(pos), _legacy_rx(cfg, gnb, k, phi, pos) - cfg.ue_tx_dbm)
    # partial reset: the legacy rule (draw all rows, keep the masked ones)
    radio.reset(torch.tensor([1, 4]))
    k2, phi2 = _legacy_draw(E, radio.C, cfg.shadow_modes, g)
    m = torch.zeros(E, dtype=torch.bool)
    m[[1, 4]] = True
    k = torch.where(m.view(1, -1, 1, 1), k2, k)
    phi = torch.where(m.view(1, -1, 1), phi2, phi)
    assert torch.equal(radio.rx_dbm(pos), _legacy_rx(cfg, gnb, k, phi, pos))


def test_default_config_is_legacy_cell():
    assert NRConfig().is_legacy_cell()
    for kw in (dict(channel="tr38901"), dict(shadow_white_frac=0.5), dict(shadow_dcorr_m=30.0),
               dict(shadow_acf="exp"), dict(blockage=True)):
        assert not NRConfig(**kw).is_legacy_cell()


def test_config_aliases_and_unused_fields():
    c = NRConfig(channel="tr38901_inf_dl")
    assert c.channel == "tr38901" and c.tr38901_scenario == "InF-DL"
    assert NRConfig(channel="tr38901", tr38901_scenario="umi").tr38901_scenario == "UMi"
    with pytest.raises(AssertionError):
        NRConfig(channel="tr38901_umi", o2i_indoor_frac=0.5, o2i_model="high").with_(tr38901_scenario="RMa")
    with pytest.raises(AssertionError):
        NRConfig(channel="nope")
    assert "channel" in NRConfig(channel="tr38901").unused_fields("L1")
    assert NRConfig(channel="tr38901", blockage=True).unused_fields("L2") == []
    assert NRConfig(fading_doppler="per_robot").unused_fields("L2") == []
    assert "fading_doppler" in NRConfig(fading_doppler="per_robot").unused_fields("L2-legacy")


# ---------------------------------------------------------------- log-distance shadowing: dcorr, ACF, white part
def _corr_at(field_fn, r, E=6000):
    """Ensemble correlation of a [E,1,1] field between (0, 0) and (r, 0) (and a rotated pair)."""
    p = torch.tensor([[[10.0, 20.0]], [[10.0 + r, 20.0]]]).expand(2, E, 2).permute(1, 0, 2).contiguous()   # [E,2,2]
    f = field_fn(p)[..., 0]
    a, b = f[:, 0], f[:, 1]
    return float((a * b).mean() / (a.pow(2).mean() * b.pow(2).mean()).sqrt()), float(a.var())


@pytest.mark.parametrize("acf,dcorr", [("exp", 10.0), ("exp", 30.0), ("sos", 30.0)])
def test_shadowing_decorrelation_distance(acf, dcorr):
    cfg = NRConfig(shadow_acf=acf, shadow_dcorr_m=dcorr)
    radio = RadioMC(cfg, 6000, "cpu", generator=torch.Generator().manual_seed(0))
    fn = lambda p: -(radio.pathgain_db(p) + cfg.pl_const_db + 10 * cfg.pathloss_exp * torch.log10(  # noqa: E731
        (p[:, :, None, :] - radio.gnb).norm(dim=-1).clamp(min=1.0)))
    rho, var = _corr_at(fn, dcorr)
    assert abs(rho - math.exp(-1)) < 0.06, rho
    assert abs(var - 36.0) < 3.0
    if acf == "exp":
        assert _corr_at(fn, 3 * dcorr)[0] > 0            # no negative lobe (legacy: -0.18 near 2.1 dcorr)


def test_white_shadowing_component():
    cfg = NRConfig(shadow_acf="exp", shadow_dcorr_m=30.0, shadow_white_frac=0.5, shadow_white_dcorr_m=0.5)
    radio = RadioMC(cfg, 6000, "cpu", generator=torch.Generator().manual_seed(1))
    fn = lambda p: -(radio.pathgain_db(p) + cfg.pl_const_db + 10 * cfg.pathloss_exp * torch.log10(  # noqa: E731
        (p[:, :, None, :] - radio.gnb).norm(dim=-1).clamp(min=1.0)))
    rho, var = _corr_at(fn, 5.0)                         # white part gone at 5 m: rho = 0.5 exp(-5/30)
    assert abs(var - 36.0) < 3.0
    assert abs(rho - 0.5 * math.exp(-5 / 30)) < 0.06, rho


def test_sos_dcorr_scales_wavelengths():
    a = RadioMC(NRConfig(), 4, "cpu", generator=torch.Generator().manual_seed(2))
    b = RadioMC(NRConfig(shadow_dcorr_m=20.6), 4, "cpu", generator=torch.Generator().manual_seed(2))
    assert torch.allclose(b.k * 2, a.k, rtol=1e-5) and torch.equal(a.phi, b.phi)


# ---------------------------------------------------------------- TR 38.901: hand-computed values (fc = 3.5 GHz)
PL_CASES = [   # (scenario, los?, d_2D, h_BS, expected dB), computed from the Table 7.4.1-1 formulas by hand
    ("UMi", True, 50.0, 10.0, 79.08964923389806), ("UMi", True, 300.0, 10.0, 98.24426066376931),
    ("UMi", False, 100.0, 10.0, 104.64383201166098),
    ("UMa", True, 100.0, 25.0, 83.13815667685238), ("UMa", True, 1000.0, 25.0, 109.41189477469499),
    ("UMa", False, 200.0, 25.0, 114.46197311942146),
    ("RMa", True, 100.0, 35.0, 84.19841472769242), ("RMa", True, 5000.0, 35.0, 125.96902515472235),
    ("RMa", False, 500.0, 35.0, 118.82266225632624),
    ("InH", True, 10.0, 3.0, 60.66494857628214), ("InH", False, 20.0, 3.0, 80.7233937148878),
    ("InF-SL", True, 30.0, 1.5, 73.93539981912798), ("InF-SL", False, 30.0, 1.5, 81.5479528823569),
    ("InF-DL", False, 30.0, 1.5, 82.21458968049747), ("InF-SH", True, 30.0, 8.0, 74.1495789134629),
    ("InF-SH", False, 30.0, 8.0, 77.4842715674044), ("InF-DH", False, 30.0, 8.0, 77.0784801870375),
]


@pytest.mark.parametrize("scn,los,d2,hbs,want", PL_CASES)
def test_tr38901_pathloss_hand_values(scn, los, d2, hbs, want):
    d3 = math.sqrt(d2 ** 2 + (hbs - 1.5) ** 2)
    f = tr.pl_los if los else tr.pl_nlos
    assert abs(f(scn, d2, d3, 3.5, hbs, 1.5) - want) < 1e-9
    t = f(scn, torch.tensor([d2]), torch.tensor([d3]), 3.5, hbs, 1.5)       # tensor path
    assert abs(float(t) - want) < 1e-3
    assert tr.SCENARIOS[scn].h_bs == hbs


def test_tr38901_pathloss_through_radio():
    """RadioMC with a forced LOS / NLOS state = hand value + sigma * the model's own shadowing field."""
    for scn, los, d2, hbs, want in PL_CASES:
        cfg = NRConfig(channel="tr38901", tr38901_scenario=scn, tr38901_los="los" if los else "nlos",
                       cell_positions_m=((0.0, 0.0),))
        radio = RadioMC(cfg, 2, "cpu", generator=torch.Generator().manual_seed(0))
        pos = torch.tensor([[[d2, 0.0]], [[0.0, d2]]])
        s_los, s_nlos = tr.sigma_sf(tr.scenario_name(scn), torch.full((2, 1, 1), d2), 3.5, hbs, 1.5)
        sf = s_los * radio.ch.sf_los(pos) if los else s_nlos * radio.ch.sf_nlos(pos)
        assert torch.allclose(-radio.pathgain_db(pos) - sf, torch.full((2, 1, 1), want), atol=5e-3)


def test_tr38901_los_probability_hand_values():
    assert abs(tr.p_los("UMi", 50.0) - 0.5195854136174696) < 1e-12
    assert abs(tr.p_los("UMa", 100.0) - 0.3476708368442312) < 1e-12
    assert abs(tr.p_los("RMa", 100.0) - 0.9139311852712282) < 1e-12
    assert abs(tr.p_los("InH", 3.0) - 0.6818274060977494) < 1e-12
    assert abs(tr.p_los("InH", 10.0) - 0.2874241595601194) < 1e-12
    assert tr.p_los("UMi", 10.0) == 1.0 and tr.p_los("RMa", 5.0) == 1.0
    ks = {s: tr.inf_k_subsce(s, tr.SCENARIOS[s].h_bs, 1.5) for s in tr.INF_CLUTTER}
    for s, want in {"InF-SL": 44.81420117724551, "InF-DL": 2.182713335874583, "InF-SH": 582.5846153041916,
                    "InF-DH": 3.1528081518188418}.items():
        assert abs(ks[s] - want) < 1e-9
    assert abs(tr.p_los("InF-SH", 30.0, k_subsce=ks["InF-SH"]) - 0.9498087165248207) < 1e-12
    assert abs(tr.p_los("InF-DL", 3.0, k_subsce=ks["InF-DL"]) - 0.25298221281347033) < 1e-12
    assert abs(tr.o2i_wall_db("low", 3.5) - 12.69750345954387) < 1e-9
    assert abs(tr.o2i_wall_db("high", 3.5) - 26.849786400945717) < 1e-9


@pytest.mark.parametrize("scn", ["UMi", "InF-DL", "InF-SL", "InH"])
def test_los_probability_statistics(scn):
    """Over many envs, the LOS fraction at each distance equals Pr_LOS (binomial 4 sigma)."""
    E = 8000
    cfg = NRConfig(channel="tr38901", tr38901_scenario=scn, cell_positions_m=((0.0, 0.0),))
    radio = RadioMC(cfg, E, "cpu", generator=torch.Generator().manual_seed(5))
    d = torch.tensor([2.0, 5.0, 12.0, 25.0, 40.0, 70.0])
    ang = torch.linspace(0, 1.5, len(d))
    pos = torch.stack([d * torch.cos(ang), d * torch.sin(ang)], -1)[None].expand(E, -1, -1).contiguous()
    radio.pathgain_db(pos)
    frac = radio.ch.los[..., 0].float().mean(0)
    p = torch.tensor([float(tr.p_los(radio.ch.scn, float(x), 1.5, radio.ch.k_subsce)) for x in d])
    tol = 4 * torch.sqrt(p * (1 - p) / E) + 2e-3
    assert torch.all((frac - p).abs() <= tol), (frac, p)


def test_los_state_is_spatially_consistent():
    E = 8000
    cfg = NRConfig(channel="tr38901", tr38901_scenario="UMi", cell_positions_m=((0.0, 0.0),))
    radio = RadioMC(cfg, E, "cpu", generator=torch.Generator().manual_seed(6))
    p0 = torch.tensor([60.0, 0.0])
    pts = [p0, p0 + torch.tensor([0.0, 0.1]), p0 + torch.tensor([0.0, 500.0])]
    states = []
    for q in pts:
        radio.pathgain_db(q.expand(E, 1, 2).contiguous())
        states.append(radio.ch.los[:, 0, 0].clone())
    same_near = (states[0] == states[1]).float().mean()
    # 0.1 m << 50 m correlation distance; the exponential ACF is rough, so a Gaussian threshold near the median
    # flips with probability arccos(exp(-0.1 / 50)) / pi = 0.02
    assert same_near > 0.96
    # 500 m apart: independent states, P(both LOS) = p(60) p(|q|)
    p_a = tr.p_los("UMi", 60.0)
    p_b = tr.p_los("UMi", float(pts[2].norm()))
    both = (states[0] & states[2]).float().mean()
    assert abs(float(both) - p_a * p_b) < 4 * math.sqrt(p_a * p_b / E) + 0.01


def test_tr38901_o2i():
    E, R = 2000, 4
    base = dict(channel="tr38901", tr38901_scenario="UMa", tr38901_los="nlos", cell_positions_m=((0.0, 0.0),))
    out_ = RadioMC(NRConfig(**base), E, "cpu", generator=torch.Generator().manual_seed(7))
    ind = RadioMC(NRConfig(**base, o2i_indoor_frac=1.0, o2i_model="high"), E, "cpu",
                  generator=torch.Generator().manual_seed(7))
    pos = torch.full((E, R, 2), 150.0)
    extra = out_.pathgain_db(pos) - ind.pathgain_db(pos)          # same fields: difference = O2I loss
    want = tr.o2i_wall_db("high", 3.5) + 0.5 * 25 / 3             # E[min(U, U)] over [0, 25] = 25 / 3
    assert abs(float(extra.mean()) - want) < 0.5
    assert abs(float(extra.std()) - math.sqrt(6.5 ** 2 + 0.25 * 25 ** 2 / 18)) < 0.5
    half = RadioMC(NRConfig(**base, o2i_indoor_frac=0.5), E, "cpu", generator=torch.Generator().manual_seed(8))
    half.pathgain_db(pos)
    assert abs(float(half.ch.indoor.float().mean()) - 0.5) < 0.03


@pytest.mark.parametrize("channel", ["tr38901_umi", "tr38901_inf_dh", "log_distance"])
def test_partial_reset_isolation(channel):
    kw = dict(o2i_indoor_frac=0.5) if channel == "tr38901_umi" else {}
    if channel == "log_distance":
        kw = dict(shadow_white_frac=0.4, blockage=True)
    E, R = 5, 6
    radio = RadioMC(NRConfig(channel=channel, n_cells=3, cell_layout="hex", **kw), E, "cpu",
                    generator=torch.Generator().manual_seed(1))
    pos = torch.rand(E, R, 2, generator=torch.Generator().manual_seed(2)) * 150
    before = radio.pathgain_db(pos)
    radio.reset(torch.tensor([0, 3]))
    after = radio.pathgain_db(pos)
    keep = torch.tensor([1, 2, 4])
    assert torch.equal(before[keep], after[keep])
    assert not torch.equal(before[0], after[0]) and not torch.equal(before[3], after[3])


# ---------------------------------------------------------------- per-robot Doppler
def test_rho_per_robot_matches_config_rule():
    v = torch.tensor([0.0, 0.5, 1.0, 3.0, 10.0], dtype=torch.float64)
    got = rho_per_ms_from_speed(v, 3.5)
    for vi, gi in zip(v.tolist(), got.tolist()):
        assert abs(gi - fading_rho_from_speed(vi, 3.5)) < 1e-9
    assert torch.all(got[1:] < got[:-1])


def test_per_robot_speed_from_poses_and_velocity():
    E, R = 3, 4
    cfg = NRConfig(fading_doppler="per_robot", doppler_min_speed_mps=0.2)
    eng = make_engine("L2", E, R, "cpu", cfg, seed=0)
    speeds = torch.tensor([0.0, 1.0, 3.0, 8.0])                     # m/s per robot
    x0 = torch.rand(E, R, 2, generator=torch.Generator().manual_seed(1)) * 50
    step_m = speeds * cfg.control_step_ms * 1e-3
    for t in range(3):
        eng.step(None, x0 + torch.stack([step_m * t, torch.zeros(R)], -1))
    want = rho_per_ms_from_speed(speeds.clamp(min=0.2), cfg.carrier_ghz)
    assert torch.allclose(eng.net.fading_rho_ms, want.expand(E, R), atol=1e-5)
    assert torch.all(eng.net.fading_rho_ms[:, 1:] < eng.net.fading_rho_ms[:, :-1])
    # explicit velocities override the pose difference
    vel = torch.zeros(E, R, 2)
    vel[..., 1] = 5.0
    eng.step(None, x0, vel=vel)
    assert torch.allclose(eng.net.fading_rho_ms, rho_per_ms_from_speed(torch.full((E, R), 5.0), 3.5), atol=1e-6)
    # a reset env starts at the speed floor (no previous pose)
    eng.reset(torch.tensor([1]))
    eng.step(None, x0 + 100.0)
    floor = float(rho_per_ms_from_speed(torch.tensor(0.2), 3.5))
    assert torch.allclose(eng.net.fading_rho_ms[1], torch.full((R,), floor), atol=1e-6)
    assert not torch.allclose(eng.net.fading_rho_ms[0], torch.full((R,), floor), atol=1e-3)


def test_per_robot_fading_equals_global_at_equal_speed():
    """Every robot at the global speed: same draws, same fading state as the global model (up to rho rounding)."""
    E, R = 4, 3
    glob = NRConfig(ue_speed_mps=3.0)
    per = NRConfig(ue_speed_mps=3.0, fading_doppler="per_robot")
    a = make_engine("L2", E, R, "cpu", glob, seed=11)
    b = make_engine("L2", E, R, "cpu", per, seed=11)
    b.net.fading_rho_ms = torch.full((E, R), glob.fading_rho_per_ms)
    snr = torch.full((E, R), 15.0)
    torch.manual_seed(0)
    for _ in range(4):
        a.step(None, snr)
    torch.manual_seed(0)
    for _ in range(4):
        b.step(None, snr)
    assert torch.allclose(a.net.h, b.net.h, atol=1e-5)


def test_per_robot_fading_decorrelates_with_speed():
    """Per-robot AR(1): the lag-one-step correlation of h falls with the robot's speed."""
    E, R = 64, 3
    eng = make_engine("L2", E, R, "cpu", NRConfig(fading_doppler="per_robot"), seed=1)
    eng.net.fading_rho_ms = rho_per_ms_from_speed(torch.tensor([0.0, 0.5, 3.0]), 3.5).expand(E, R).contiguous()
    h0 = eng.net.h.clone()
    torch.manual_seed(1)
    eng.step(None, torch.full((E, R), 15.0))
    c = (h0 * eng.net.h).sum(-1).mean(dim=(0, 2)) / (h0 * h0).sum(-1).mean(dim=(0, 2))
    assert abs(float(c[0]) - 1.0) < 1e-6 and c[0] > c[1] > c[2]


def test_install_requires_nr_fading_state():
    with pytest.raises(RuntimeError):
        install_per_robot_fading(object())


# ---------------------------------------------------------------- radio map
def test_radio_map_bilinear_sampling():
    g = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4) * -1.0      # [C=2, H=3, W=4]
    m = RadioMap(g, (0.0, 10.0, 30.0, 30.0))                                       # x step 10, y step 10
    xs, ys = torch.tensor([0.0, 10.0, 20.0, 30.0]), torch.tensor([10.0, 20.0, 30.0])
    grid = torch.stack(torch.meshgrid(xs, ys, indexing="xy"), -1).reshape(1, -1, 2)  # [1, 12, 2], row-major y
    got = m.sample(grid)[0]                                                          # [12, C]
    assert torch.equal(got.t().reshape(2, 3, 4), g)                                  # exact at grid points
    mid = m.sample(torch.tensor([[[5.0, 15.0], [25.0, 25.0], [-50.0, 99.0], [15.0, 10.0]]]))[0]
    assert torch.allclose(mid[0], (g[:, 0, 0] + g[:, 0, 1] + g[:, 1, 0] + g[:, 1, 1]) / 4)
    assert torch.allclose(mid[1], (g[:, 1, 2] + g[:, 1, 3] + g[:, 2, 2] + g[:, 2, 3]) / 4)
    assert torch.equal(mid[2], g[:, 2, 0])                                           # clamped to the border
    assert torch.allclose(mid[3], (g[:, 0, 1] + g[:, 0, 2]) / 2)


def test_radio_map_file_roundtrip_and_synthetic(tmp_path):
    m = make_synthetic_map(synthetic_gnb_xy())
    shipped = RadioMap.load(__import__("isaac_net.core.channels", fromlist=["x"]).SYNTHETIC_MAP)
    assert torch.equal(m.gain, shipped.gain) and m.bounds == shipped.bounds
    p = str(tmp_path / "m.npz")
    m.save(p)
    assert torch.equal(RadioMap.load(p).gain, m.gain)
    torch.save({"gain_db": m.gain.view(m.C, m.H, m.W), "bounds": torch.tensor(m.bounds)}, str(tmp_path / "m.pt"))
    assert torch.equal(RadioMap.load(str(tmp_path / "m.pt")).gain, m.gain)
    # wall: the far side of x = 75 m from cell 0 costs 10 dB more than log-distance
    radio = RadioMC(NRConfig(**MAP_CFG), 1, "cpu")
    pos = torch.tensor([[[20.0, 70.0], [130.0, 70.0]]])
    pg = radio.pathgain_db(pos)[0]
    d = torch.tensor([math.hypot(5, 5), math.hypot(105, 5)])
    ld = -(40 + 35 * torch.log10(d))
    assert (pg[0, 0] - ld[0]).abs() < 3 and (pg[1, 0] - (ld[1] - 10)).abs() < 3


def test_radio_map_checks_cells():
    with pytest.raises(ValueError):
        RadioMC(NRConfig(channel="radio_map", radio_map_path="synthetic"), 1, "cpu")            # 1 cell vs 2
    with pytest.raises(ValueError):
        RadioMC(NRConfig(**{**MAP_CFG, "cell_positions_m": ((0.0, 0.0), (1.0, 1.0))}), 1, "cpu")
    with pytest.raises(ValueError):
        RadioMC(NRConfig(channel="radio_map"), 1, "cpu")


def test_radio_map_engine_step():
    E, R = 3, 4
    eng = make_engine("L2", E, R, "cpu", NRConfig(**MAP_CFG, noise_model="thermal"), seed=0)
    pos = torch.rand(E, R, 2, generator=torch.Generator().manual_seed(0)) * 150
    for _ in range(2):
        out = eng.step(None, pos)
    left = pos[..., 0] < 75
    assert torch.equal(out["serving_cell"] == 0, left)


# ---------------------------------------------------------------- blockage
def test_blockage_geometry():
    gnb = torch.tensor([[10.0, 0.0, 0.0]])
    robot = [0.0, 0.0, 0.0]
    cases = {(5.0, 0.1): True, (5.0, 0.5): False, (-1.0, 0.0): False, (11.0, 0.0): False, (9.0, -0.25): True}
    for (x, y), want in cases.items():
        pos3 = torch.tensor([[robot, [x, y, 0.0]]])
        b = blocked_links(pos3, gnb, 0.3)
        assert bool(b[0, 0, 0]) == want, (x, y)
        assert b.shape == (1, 2, 1)
    # heights: the segment to a high gNB clears a blocker 5 m away at the robot's height
    gnb_hi = torch.tensor([[10.0, 0.0, 8.0]])
    assert not bool(blocked_links(torch.tensor([[robot, [5.0, 0.0, 0.0]]]), gnb_hi, 0.3)[0, 0, 0])
    assert bool(blocked_links(torch.tensor([[robot, [0.2, 0.0, 0.0]]]), gnb_hi, 0.3)[0, 0, 0])
    # envs are independent; a robot never blocks itself
    pos3 = torch.tensor([[robot, [5.0, 0.0, 0.0]], [robot, [5.0, 3.0, 0.0]]])
    b = blocked_links(pos3, gnb, 0.3)
    assert b[0].tolist() == [[True], [False]] and b[1].tolist() == [[False], [False]]


@pytest.mark.parametrize("channel", ["log_distance", "tr38901_inf_sl"])
def test_blockage_adds_loss(channel):
    base = dict(channel=channel, n_cells=2, cell_layout="custom", cell_positions_m=((10.0, 0.0), (0.0, 10.0)))
    a = RadioMC(NRConfig(**base), 1, "cpu", generator=torch.Generator().manual_seed(0))
    b = RadioMC(NRConfig(**base, blockage=True, blockage_loss_db=17.0), 1, "cpu",
                generator=torch.Generator().manual_seed(0))
    pos = torch.tensor([[[0.0, 0.0], [5.0, 0.05], [30.0, 30.0]]])
    diff = a.pathgain_db(pos) - b.pathgain_db(pos)
    want = torch.zeros(1, 3, 2)
    want[0, 0, 0] = 17.0                                            # robot 0 -> cell 0 passes robot 1
    assert torch.allclose(diff, want, atol=1e-4)


# ---------------------------------------------------------------- graph safety
@pytest.mark.gpu
@pytest.mark.parametrize("kw", [dict(channel="tr38901_umi", o2i_indoor_frac=0.3, blockage=True),
                                dict(MAP_CFG, blockage=True),
                                dict(shadow_acf="exp", shadow_white_frac=0.5, shadow_dcorr_m=30.0)])
def test_channel_cuda_graph_capture(kw):
    dev = "cuda"
    E, R = 64, 8
    radio = RadioMC(NRConfig(**kw), E, dev, generator=torch.Generator(device=dev).manual_seed(0), R=R)
    pos = torch.rand(E, R, 2, device=dev) * 150
    radio.pathgain_db(pos)                                          # warm-up (allocations)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            radio.pathgain_db(pos)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = radio.pathgain_db(pos)
    pos.copy_(torch.rand(E, R, 2, device=dev) * 150)
    g.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, radio.pathgain_db(pos))


def test_plane_wave_field_unit_variance():
    f = PlaneWaveField(20000, 1, 8, "cpu", torch.Generator().manual_seed(0), "exp", 10.0)
    v = f(torch.tensor([[[3.0, 4.0]]]).expand(20000, 1, 2).contiguous())
    assert abs(float(v.var()) - 1.0) < 0.05
