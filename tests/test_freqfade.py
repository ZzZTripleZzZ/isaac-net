"""Frequency-correlated fast fading across the subbands (NRConfig.fading_freq_corr; docs/channels.md,
"Frequency-selective fading").

CPU
  (a) fading_freq_corr=False is bitwise the pre-feature engine: the live NRNet against the same class with the frozen
      pre-feature fading code (tests/nr_frozen/base/nr_engine.py: reset, _evolve, _gain), one and three cells, global
      RNG, with a partial reset; and the off engine carries no frequency-correlation state
  (b) fixed delay spread: the empirical cross-subband correlation of h over many links and slots matches the
      magnitude-consistent model 1 / sqrt(1 + (2 pi df tau)^2) for tau in {10, 50, 100, 300} ns, and the power
      correlation matches |rho|^2; tau = 1 ns is almost fully correlated and tau = 3000 ns almost independent across
      the 1.44 MHz subbands of the default 20 MHz carrier; the initial state (reset) has the same correlation
  (c) E|h|^2 = 1 per subband (within 1 %) and the temporal AR(1) correlation per subband is unchanged
  (d) pf_metric="subband": the subband PF gain over wideband PF shrinks monotonically as the subbands become
      correlated (qualitative)
  (e) E-independence, partial-reset isolation and reset-order independence (fixed delay spread, per-link delay spread
      from the LOS state, three cells); the per-link draw statistics and the LOS / NLOS switch of the grid index
  (f) config gating (unused_fields), validation; the nr_equiv configs run on the reference and their state is in
      nr_fast.state_dict (teacher forcing copies it)
GPU: the nr_equiv configs FCORR_CFGS are in the G1 (graph bitwise) and G2 (triton teacher-forced) lists of
test_nr_fast.py, ul_fcorr also in G7
"""
import math

import pytest
import torch

import nr_equiv
from isaac_net.core import NRConfig, Requests, make_engine
from isaac_net.core.config import multicell
from isaac_net.core.nr_engine import (NRNet, corr_sqrt, lg_ds_params, subband_centers_hz, subband_corr)
from isaac_net.core.nr_fast import state_dict
from nr_frozen.base import nr_engine as frozen

SIZES = (4000.0, 30000.0)
FAST = dict(control_step_ms=10.0)


class OldFading(NRNet):
    """The live engine with the pre-feature fading code (verbatim, from the frozen base)."""
    reset = frozen.NRNet.reset
    _evolve = frozen.NRNet._evolve
    _gain = frozen.NRNet._gain


def _drive_net(net, steps=10, reset_at=4, seed=0):
    E, R, C = net.E, net.R, net.C
    g = torch.Generator().manual_seed(seed)
    z = torch.zeros(E, dtype=torch.long)
    outs = []
    for t in range(steps):
        if t == reset_at:
            net.reset(torch.tensor([1]))
        send = torch.randint(0, 3, (E, R), generator=g)
        net.add_frames(t, send, torch.zeros(E, R, dtype=torch.bool), z, 25 * torch.rand(E, R, generator=g))
        if net.dl is not None:
            net.add_dl_frames(t, send.float() * 2000)
        if C == 1:
            outs.append(net.step(t, 25 * torch.rand(E, R, generator=g), z, full=True))
        else:
            outs.append(net.step_cells(t, -60 - 40 * torch.rand(E, R, C, generator=g), z, full=True))
    return outs


def _same_outs(a, b):
    for x, y in zip(a, b):
        assert x.keys() == y.keys()
        for k in x:
            assert torch.equal(x[k].nan_to_num(-7.0), y[k].nan_to_num(-7.0)), k


# ---------------------------------------------------------------------------------------------------------- (a)
@pytest.mark.parametrize("cfg", [NRConfig(dl=True, **FAST), multicell(3, dl=True, **FAST), NRConfig(rng="global", **FAST),
                                 NRConfig(fading_doppler="per_robot", **FAST)],
                         ids=["c1_dl", "c3_dl", "c1_global", "c1_doppler_field"])
def test_a_off_is_bitwise_the_pre_feature_engine(cfg):
    assert not cfg.fading_freq_corr and cfg.freq_corr_mode is None
    torch.manual_seed(0)
    new = NRNet(3, 4, "cpu", SIZES, cfg, generator=torch.Generator().manual_seed(4), seed=8)
    a = _drive_net(new)
    torch.manual_seed(0)
    old = OldFading(3, 4, "cpu", SIZES, cfg, generator=torch.Generator().manual_seed(4), seed=8)
    b = _drive_net(old)
    _same_outs(a, b)
    assert torch.equal(new.h, old.h)
    assert new.fc_mode is None and not any(n.startswith("fc_") and torch.is_tensor(v) for n, v in vars(new).items())
    eng = make_engine("L2", 2, 3, "cpu", cfg, seed=1)
    assert not any(".fc_" in k for k in state_dict(eng))


def test_a_rician_off_fcorr_keeps_the_rician_engine():
    """Rician on, frequency correlation off: the same engine as before the feature (the Rician test file checks the
    Rician path itself); here only that no frequency-correlation state appears and set_los still sets K."""
    net = NRNet(2, 3, "cpu", SIZES, NRConfig(fading_rician=True, channel="tr38901_inf_sh", **FAST), seed=1)
    assert net.fc_mode is None and not any(n.startswith("fc_") and torch.is_tensor(v) for n, v in vars(net).items())
    net.set_los(torch.ones(2, 3, 1, dtype=torch.bool))
    assert torch.equal(net.k_lin, net.k_los)


# ---------------------------------------------------------------------------------------------------------- (b)
def _samples(tau_ns, E=64, R=16, n=40, seed=3, cfg=None):
    """h [n, E*R, S, 2] at slots 50 ms apart (rho = 0.97^50 = 0.23 between samples), fresh innovations each."""
    cfg = cfg or NRConfig(fading_freq_corr=True, fading_delay_spread_ns=tau_ns)
    net = NRNet(E, R, "cpu", SIZES, cfg, seed=seed)
    out = []
    for i in range(n):
        g = 100 * (i + 1)
        net._evolve(g, 0)
        net.rng.tick_step()
        out.append(net.h.reshape(E * R, net.S, 2).clone())
    return net, torch.stack(out)


def _corr(h):
    """Empirical E[h_i h_j^*] real part [S, S] (= the model covariance; its imaginary part is 0 for a real L)."""
    x = h.reshape(-1, h.shape[-2], 2).double()
    return (torch.einsum("nic,njc->ij", x, x)) / x.shape[0]


def _model(tau_ns, cfg=None):
    return subband_corr(tau_ns * 1e-9, subband_centers_hz(cfg or NRConfig()))


@pytest.mark.parametrize("tau", [10.0, 50.0, 100.0, 300.0])
def test_b_cross_subband_correlation_matches_the_formula(tau):
    net, h = _samples(tau)
    emp, mod = _corr(h), _model(tau)
    assert float((emp - mod).abs().max()) < 0.03, (tau, float((emp - mod).abs().max()))
    # power correlation of circular Gaussian subbands = |rho|^2 (the exact complex model's value)
    p = (h.double() ** 2).sum(-1).reshape(-1, net.S)
    pc = torch.corrcoef(p.T)
    assert float((pc - mod ** 2).abs().max()) < 0.04, float((pc - mod ** 2).abs().max())
    # the closed form at the adjacent-subband spacing of the default carrier (1.44 MHz)
    df = 1.44e6
    want = 1 / math.sqrt(1 + (2 * math.pi * df * tau * 1e-9) ** 2)
    assert abs(float(mod[0, 1]) - want) < 1e-12


def test_b_limits_full_and_no_correlation():
    _, h1 = _samples(1.0, n=10)
    c1 = _corr(h1)
    assert float(c1.min()) > 0.99                         # 1 ns: coherence bandwidth >> carrier
    _, h3 = _samples(3000.0, n=20)
    c3 = _corr(h3)
    off = c3 - torch.diag(torch.diag(c3))
    assert float(off.abs().max()) < 0.07                  # 3 us: 1.44 MHz spacing -> |rho| = 0.037
    assert float(_model(3000.0)[0, 1]) < 0.04


def test_b_initial_state_is_correlated():
    net = NRNet(400, 16, "cpu", SIZES, NRConfig(fading_freq_corr=True, fading_delay_spread_ns=100.0), seed=2)
    emp = _corr(net.h.reshape(-1, net.S, 2))
    assert float((emp - _model(100.0)).abs().max()) < 0.03


def test_b_square_root_is_exact_and_unit_diagonal():
    f = subband_centers_hz(NRConfig())
    for tau in (0.1, 1.0, 30.0, 1000.0):
        c = subband_corr(tau * 1e-9, f)
        L = corr_sqrt(c)
        assert torch.allclose((L ** 2).sum(-1), torch.ones(len(f), dtype=torch.float64), atol=1e-12)
        assert float((L @ L.T - c).abs().max()) < (1e-5 if tau < 1 else 1e-9)
    assert float(torch.linalg.eigvalsh(subband_corr(30e-9, f)).min()) > 0      # positive definite (Bochner)


# ---------------------------------------------------------------------------------------------------------- (c)
def test_c_unit_power_per_subband_and_temporal_correlation():
    net, h = _samples(50.0)
    pw = (h.double() ** 2).sum(-1).mean((0, 1))             # [S]
    assert float((pw - 1).abs().max()) < 0.01, pw
    # temporal correlation over 4 slots (2 ms at mu = 1): rho_t = fading_rho_per_ms ** 2 on every subband
    cfg = NRConfig(fading_freq_corr=True, fading_delay_spread_ns=50.0)
    net = NRNet(64, 16, "cpu", SIZES, cfg, seed=4)
    a, b = [], []
    for i in range(30):
        g = 200 * (i + 1)
        net._evolve(g, 0)
        h0 = net.h.clone()
        net._evolve(g + 4, 1)
        a.append(h0)
        b.append(net.h.clone())
        net.rng.tick_step()
    x, y = torch.stack(a).double(), torch.stack(b).double()
    rt = ((x * y).sum(-1)).mean((0, 1, 2))                  # [S]: E Re(h_t h_{t+4}^*)
    want = cfg.fading_rho_per_ms ** (4 * cfg.slot_ms)
    assert float((rt - want).abs().max()) < 0.015, (rt, want)


# ---------------------------------------------------------------------------------------------------------- (d)
def _thr(tau, metric, E=8, R=6, steps=30, seed=2):
    cfg = NRConfig(control_step_ms=10.0, pf_metric=metric, fading_freq_corr=True, fading_delay_spread_ns=tau,
                   msg_sizes=(30000.0, 30000.0), frame_buffer=64)
    eng = make_engine("L2", E, R, "cpu", cfg, seed=seed)
    snr = 5 + 15 * torch.rand(E, R, generator=torch.Generator().manual_seed(1))
    for _ in range(steps):
        eng.submit(None, Requests(torch.ones(E, R, dtype=torch.long), None, torch.zeros(E, dtype=torch.long)))
        eng.step(None, snr)
    return float(eng.counters()["ul"]["bytes_ok"])


def test_d_subband_pf_gain_shrinks_with_correlation():
    gains = [_thr(tau, "subband") / _thr(tau, "wideband") for tau in (3000.0, 100.0, 10.0, 1.0)]
    assert all(a > b for a, b in zip(gains, gains[1:])), gains
    assert gains[0] > 1.5 and gains[-1] < 1.1, gains


# ---------------------------------------------------------------------------------------------------------- (e)
R, STEPS = 3, 7
E_CASES = {"fixed": NRConfig(fading_freq_corr=True, fading_delay_spread_ns=80.0, dl=True, **FAST),
           "los": NRConfig(fading_freq_corr=True, channel="tr38901_inf_sh", **FAST),
           "los_rician": NRConfig(fading_freq_corr=True, fading_rician=True, channel="tr38901_umi", **FAST),
           "los_c3": multicell(3, fading_freq_corr=True, channel="tr38901_inf_sh", **FAST)}


def _inputs(seed=0, E=6):
    g = torch.Generator().manual_seed(seed)
    return [(torch.randint(0, 3, (E, R), generator=g), torch.rand(E, R, 2, generator=g) * 150)
            for _ in range(STEPS)]


def _drive(eng, E, resets=()):
    rs = dict(resets)
    outs = []
    for k, (send, pos) in enumerate(_inputs()):
        eng.submit(None, Requests(send[:E], None, torch.full((E,), k, dtype=torch.long)))
        if eng.config.dl:
            eng.add_dl_frames(None, send[:E].float() * 3000)
        o = eng.step(None, pos[:E])
        outs.append({n: v.clone() for n, v in o.items() if torch.is_tensor(v) and v.dim() >= 1 and v.shape[0] == E})
        outs[-1]["h"] = eng.net.h.clone()
        if eng.net.fc_mode == "los":
            outs[-1]["fc_idx"] = eng.net.fc_idx.clone()
        if k in rs:
            eng.reset(torch.tensor(rs[k]))
    return outs


def _rows_equal(a, b, rows):
    """Exact for integer / boolean outputs; float outputs to float32 rounding on the CPU, whose batched matmul and
    vectorized kernels may round differently for a different tensor shape."""
    for x, y in zip(a, b):
        for name in x:
            u, v = x[name][rows].nan_to_num(-7.0), y[name][rows].nan_to_num(-7.0)
            if u.is_floating_point():
                assert torch.allclose(u, v, rtol=1e-5, atol=1e-5), name
            else:
                assert torch.equal(u, v), name


@pytest.mark.parametrize("case", list(E_CASES))
def test_e_env_independence_and_partial_reset(case):
    cfg = E_CASES[case]
    mk = lambda E: make_engine("L2", E, R, "cpu", cfg, seed=5)    # noqa: E731
    a = _drive(mk(3), 3, {3: [1]})
    b = _drive(mk(6), 6, {3: [1]})
    _rows_equal(a, b, slice(0, 3))
    c = _drive(mk(4), 4)
    d = _drive(mk(4), 4, {2: [1]})
    _rows_equal(c, d, [0, 2, 3])
    assert not torch.equal(c[4]["h"][1], d[4]["h"][1])
    e = _drive(mk(4), 4, {2: [0]})
    f = _drive(mk(4), 4, {0: [1], 1: [1, 2], 2: [0]})
    _rows_equal(e, f, 0)


def test_e_los_state_switches_the_delay_spread():
    net = NRNet(3, 4, "cpu", SIZES, NRConfig(fading_freq_corr=True, channel="tr38901_inf_sh", **FAST), seed=1)
    assert torch.equal(net.fc_idx, net.fc_idx_nlos)                # no LOS state yet: NLOS
    los = torch.rand(3, 4, 1, generator=torch.Generator().manual_seed(0)) < 0.5
    blocked = torch.zeros_like(los)
    blocked[0, 0] = True
    net.set_los(los, blocked)
    on = (los & ~blocked)[..., 0]
    assert torch.equal(net.fc_idx, torch.where(on, net.fc_idx_los, net.fc_idx_nlos))
    net.reset(torch.tensor([2]))
    assert torch.equal(net.fc_idx[2], net.fc_idx_nlos[2]) and torch.equal(net.fc_idx[:2],
                                                                        torch.where(on, net.fc_idx_los,
                                                                                    net.fc_idx_nlos)[:2])


@pytest.mark.parametrize("scenario", ["InF-SH", "UMi", "RMa"])
def test_e_delay_spread_draw_statistics(scenario):
    cfg = NRConfig(fading_freq_corr=True, tr38901_scenario=scenario)
    net = NRNet(500, 16, "cpu", SIZES, cfg, seed=3)
    lg = torch.log10(net.fc_grid_ns * 1e-9)
    for idx, los in ((net.fc_idx_nlos, False), (net.fc_idx_los, True)):
        mu, sigma = lg_ds_params(cfg, los)
        x = lg[idx]
        assert abs(float(x.mean()) - mu) < 0.02, (scenario, los, float(x.mean()), mu)
        assert abs(float(x.std()) - math.sqrt(sigma ** 2 + 0.2 ** 2 / 12)) < 0.03    # + grid rounding variance


def test_e_ds_table_values():
    """TR 38.901 V17.0.0 Table 7.5-6 lgDS at 3.5 GHz (frequency floors: UMa / InH 6 GHz, UMi 2 GHz)."""
    c = lambda sc, **kw: NRConfig(tr38901_scenario=sc, carrier_ghz=3.5, **kw)     # noqa: E731
    assert lg_ds_params(c("RMa"), True) == (-7.49, 0.55) and lg_ds_params(c("RMa"), False) == (-7.43, 0.48)
    assert lg_ds_params(c("UMa"), True) == pytest.approx((-6.955 - 0.0963 * math.log10(6), 0.66))
    assert lg_ds_params(c("UMa"), False) == pytest.approx((-6.28 - 0.204 * math.log10(6), 0.39))
    assert lg_ds_params(c("UMi"), True) == pytest.approx((-0.24 * math.log10(4.5) - 7.14, 0.38))
    assert lg_ds_params(c("UMi"), False) == pytest.approx((-0.24 * math.log10(4.5) - 6.83,
                                                           0.16 * math.log10(4.5) + 0.28))
    assert lg_ds_params(c("InH"), True) == pytest.approx((-0.01 * math.log10(7) - 7.692, 0.18))
    # InF: log10(26 V/S + 14) - 9.35 (LOS), log10(30 V/S + 32) - 9.44 (NLOS); 300 x 150 x 10 m hall for SH
    vs = 450000 / 99000
    assert lg_ds_params(c("InF-SH"), True) == pytest.approx((math.log10(26 * vs + 14) - 9.35, 0.15))
    assert lg_ds_params(c("InF-SL"), False) == pytest.approx((math.log10(30 * 4.0 + 32) - 9.44, 0.19))
    assert lg_ds_params(c("InF-DH", inf_hall_volume_m3=8000.0, inf_hall_surface_m2=2400.0), True) == \
        pytest.approx((math.log10(26 * 8000 / 2400 + 14) - 9.35, 0.15))
    assert lg_ds_params(c("InF-DL", inf_lg_ds=-7.5), False) == (-7.5, 0.19)
    # the default InF-SH hall gives tens of ns (coherence bandwidth of a few MHz)
    assert 50e-9 < 10 ** lg_ds_params(c("InF-SH"), True)[0] < 70e-9


# ---------------------------------------------------------------------------------------------------------- (f)
def test_f_config_gating():
    assert NRConfig(fading_freq_corr=True).unused_fields("L2") == []
    assert NRConfig(fading_freq_corr=True, fading_delay_spread_ns=50.0).unused_fields("L2") == []
    assert NRConfig(fading_delay_spread_ns=50.0, fading_ds_grid=(1.0, 100.0, 8)).unused_fields("L2") == \
        ["fading_delay_spread_ns", "fading_ds_grid"]
    assert NRConfig(fading=False, fading_freq_corr=True).unused_fields("L2") == ["fading_freq_corr"]
    assert NRConfig(fading_freq_corr=True, fading_delay_spread_ns=50.0, fading_ds_grid=(1.0, 100.0, 8),
                    inf_lg_ds=-7.0).unused_fields("L2") == ["fading_ds_grid", "inf_lg_ds"]
    assert NRConfig(fading_freq_corr=True, tr38901_scenario="UMi").unused_fields("L2") == []
    assert NRConfig(fading_freq_corr=True, tr38901_scenario="UMi", inf_lg_ds=-7.0).unused_fields("L2") == \
        ["inf_lg_ds"]
    assert NRConfig(fading_freq_corr=True, inf_lg_ds=-7.0, inf_hall_volume_m3=1e4,
                    inf_hall_surface_m2=3e3).unused_fields("L2") == ["inf_hall_surface_m2", "inf_hall_volume_m3"]
    assert NRConfig(fading_freq_corr=True).unused_fields("L2-legacy") == ["fading_freq_corr"]
    assert NRConfig(fading_freq_corr=True, fading_delay_spread_ns=5.0).freq_corr_mode == "fixed"
    assert NRConfig(fading_freq_corr=True).freq_corr_mode == "los"
    assert NRConfig(fading=False, fading_freq_corr=True).freq_corr_mode is None
    for bad in (dict(fading_delay_spread_ns=0.0), dict(fading_pdp="uniform"), dict(fading_ds_grid=(10.0, 1.0, 4)),
                dict(inf_hall_volume_m3=1e4), dict(fading_freq_corr=True, fading_ds_from_los=False)):
        with pytest.raises(AssertionError):
            NRConfig(**bad)


@pytest.mark.parametrize("name", nr_equiv.FCORR_CFGS)
def test_f_equiv_configs_run_on_the_reference(name):
    cfg = nr_equiv.CFGS[name]().with_(msg_sizes=SIZES, **FAST)
    assert cfg.fading_freq_corr
    eng = make_engine("L2", 4, 3, "cpu", cfg, "reference", seed=3)
    for t, d in nr_equiv.Workload(cfg, 4, 3, 8, seed=4, p_reset=0.3, device="cpu", phase_offset=25):
        nr_equiv.drive(eng, d)
    sd = state_dict(eng)
    keys = {"net.fc_L"} if cfg.freq_corr_mode == "fixed" else {"net.fc_tab", "net.fc_idx", "net.fc_idx_los",
                                                               "net.fc_idx_nlos"}
    assert keys <= set(sd)
    other = make_engine("L2", 4, 3, "cpu", cfg, "reference", seed=99)
    nr_equiv.copy_state(eng, other)
    for k in keys:
        assert torch.equal(state_dict(other)[k], sd[k])
    if cfg.freq_corr_mode == "los":                 # the radio's LOS state reached the index: some links LOS
        assert bool((eng.net.fc_idx != eng.net.fc_idx_nlos).any())
