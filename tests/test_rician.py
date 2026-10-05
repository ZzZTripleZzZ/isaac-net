"""Rician fast fading of the NR engine (NRConfig.fading_rician; docs/channels.md, "Rician fading").

CPU
  (a) fading_rician=False is bitwise the pre-feature engine: the live NRNet against the same class with the frozen
      pre-feature fading code (tests/nr_frozen/base/nr_engine.py: reset, _evolve, _gain), one and three cells, with a
      partial reset; and the off engine carries no Rician state (the graph backend's state registry is unchanged)
  (b) fixed K: the empirical CDF of |x|^2 per subband over many links and slots matches the Rician power CDF
      (2 (K + 1) |x|^2 ~ noncentral chi^2 with 2 degrees of freedom and noncentrality 2K) for K in {0, 3, 7, 10} dB
      (Kolmogorov-Smirnov distance), and E|x|^2 = 1
  (c) K = 0 (rician_k_db = -inf) reproduces the Rayleigh engine bitwise (same fades, same outputs)
  (d) a LOS -> NLOS flip ramps K linearly over rician_k_ramp_slots slots from the first slot of the control step,
      a flip back mid-ramp continues from the current K, and an env's first LOS state after a reset applies at once
  (e) E-independence, partial-reset isolation and reset-order independence with Rician on (fixed K, and K from a
      LOS state supplied by the radio's los_state())
  (f) the nr_equiv Rician configs run on the reference engine and their Rician state is in nr_fast.state_dict
      (teacher forcing copies it); config gating of the new fields (unused_fields)
GPU: the Rician configs of nr_equiv (RICIAN_CFGS) are in the G1 (graph bitwise), G2 (triton teacher-forced) and G7
lists of test_nr_fast.py
"""
import math

import pytest
import torch

import nr_equiv
from isaac_net.core import NRConfig, Requests, make_engine
from isaac_net.core.config import multicell
from isaac_net.core.nr_engine import RICIAN_K_DB, NRNet, rician_k_params
from isaac_net.core.nr_fast import state_dict
from isaac_net.core.radio import RadioMC
from nr_frozen.base import nr_engine as frozen

SIZES = (4000.0, 30000.0)
FAST = dict(control_step_ms=10.0)      # 20 slots per control step (the default 100 ms step has 200): CPU time


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
@pytest.mark.parametrize("cfg", [NRConfig(dl=True, **FAST), multicell(3, dl=True, **FAST), NRConfig(rng="global", **FAST)],
                         ids=["c1_dl", "c3_dl", "c1_global"])
def test_a_off_is_bitwise_the_pre_feature_engine(cfg):
    assert not cfg.fading_rician and cfg.rician_mode is None
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(4)
    new = NRNet(3, 4, "cpu", SIZES, cfg, generator=gen, seed=8)
    a = _drive_net(new)
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(4)
    old = OldFading(3, 4, "cpu", SIZES, cfg, generator=gen, seed=8)
    b = _drive_net(old)
    _same_outs(a, b)
    assert torch.equal(new.h, old.h)
    assert not any((n.startswith("k_") or n == "spec") and torch.is_tensor(v) for n, v in vars(new).items())
    eng = make_engine("L2", 2, 3, "cpu", cfg, seed=1)
    assert not any(".k_" in k or k.endswith(".spec") for k in state_dict(eng))


def test_a_frozen_fading_code_is_the_live_off_path():
    """The frozen functions used as the reference above are the live ones minus the Rician branch."""
    import inspect
    live = inspect.getsource(NRNet._evolve)
    assert live == inspect.getsource(frozen.NRNet._evolve)
    assert "return 10 * torch.log10((self.h ** 2).sum(-1).clamp(min=1e-6))" in inspect.getsource(NRNet._gain)


# ---------------------------------------------------------------------------------------------------------- (b)
def rician_power_cdf(x, k):
    """P(|x|^2 <= x) for unit-mean Rician power with K-factor k (linear): 2 (k + 1) |x|^2 ~ ncx2(2, 2k)."""
    y = (2 * (k + 1) * x.double())[..., None]
    lam = 2.0 * k
    J = 80 if k > 0 else 1
    j = torch.arange(J, dtype=torch.float64)
    pois = torch.exp(-lam / 2 + j * math.log(lam / 2 if k > 0 else 1.0) - torch.lgamma(j + 1)) if k > 0 else \
        torch.tensor([1.0], dtype=torch.float64)
    # P(chi^2_{2m} <= y) = 1 - exp(-y/2) sum_{i<m} (y/2)^i / i!, m = 1 + j
    i = torch.arange(J, dtype=torch.float64)
    terms = torch.exp(i * torch.log((y / 2).clamp(min=1e-300)) - torch.lgamma(i + 1) - y / 2)     # [..., J]
    tail = torch.cumsum(terms, -1)                                                               # sum_{i<=j}
    return ((1 - tail) * pois).sum(-1)


def _power_samples(k_db, E=48, R=16, n=40, seed=3):
    cfg = NRConfig(fading_rician=True, rician_k_db=k_db)
    net = NRNet(E, R, "cpu", SIZES, cfg, seed=seed)
    out = []
    for i in range(n):
        g = 100 * (i + 1)                      # 50 ms apart: rho = 0.97^50 = 0.23, power correlation 0.05
        net._evolve(g, 0)
        net.rng.tick_step()                    # a new control step: fresh innovations
        out.append(10 ** (net._gain(g) / 10))
    return torch.stack(out).flatten()


@pytest.mark.parametrize("k_db", [float("-inf"), 3.0, 7.0, 10.0])
def test_b_envelope_matches_rician_distribution(k_db):
    p = _power_samples(k_db)
    k = 0.0 if k_db == float("-inf") else 10 ** (k_db / 10)
    assert abs(float(p.mean()) - 1.0) < 0.02, float(p.mean())
    xs, _ = torch.sort(p.double())
    n = xs.numel()
    F = rician_power_cdf(xs, k)
    emp_hi = torch.arange(1, n + 1, dtype=torch.float64) / n
    ks = float(torch.maximum((emp_hi - F).abs(), (emp_hi - 1 / n - F).abs()).max())
    assert ks < 0.012, (k_db, ks, n)
    # deep fades: Rayleigh puts 1% of the power below -20 dB, K = 7 dB near -9.8 dB
    q01 = float(torch.quantile(p[:200_000], 0.01))
    if k_db == 7.0:
        assert 10 * math.log10(q01) > -11.0


def test_b_cdf_helper_is_exponential_at_k0():
    x = torch.tensor([0.01, 0.1, 1.0, 3.0])
    assert torch.allclose(rician_power_cdf(x, 0.0), 1 - torch.exp(-x.double()), atol=1e-12)
    # large K concentrates around 1
    assert float(rician_power_cdf(torch.tensor([0.5]), 100.0)) < 1e-3


# ---------------------------------------------------------------------------------------------------------- (c)
@pytest.mark.parametrize("cfg", [NRConfig(dl=True, **FAST), multicell(3, dl=True, **FAST)], ids=["c1", "c3"])
def test_c_k0_is_the_rayleigh_engine(cfg):
    ray = NRNet(3, 4, "cpu", SIZES, cfg, seed=5)
    ric = NRNet(3, 4, "cpu", SIZES, cfg.with_(fading_rician=True, rician_k_db=float("-inf")), seed=5)
    assert ric.rician == "fixed" and float(ric.k_lin.abs().max()) == 0.0
    _same_outs(_drive_net(ray), _drive_net(ric))
    assert torch.equal(ray.h, ric.h)
    assert torch.equal(ray._gain(), ric._gain(0))


# ---------------------------------------------------------------------------------------------------------- (d)
def _los_net(ramp=4, E=2, R=3):
    cfg = NRConfig(fading_rician=True, rician_k_ramp_slots=ramp, channel="tr38901_inf_sh", **FAST)
    return NRNet(E, R, "cpu", SIZES, cfg, seed=2)


def _step(net, t):
    z = torch.zeros(net.E, dtype=torch.long)
    net.step(t, torch.full((net.E, net.R), 20.0), z)


def test_d_los_nlos_flip_ramps_k():
    net = _los_net(ramp=4)
    N = net.cfg.slots_per_step
    E, R = net.E, net.R
    assert float(net.k_lin.abs().max()) == 0.0                    # no LOS state yet: K = 0
    net.set_los(torch.ones(E, R, 1, dtype=torch.bool))             # first state after the reset: at once
    k = net.k_los.clone()
    assert torch.equal(net.k_lin, k) and torch.equal(net._k_at(0), k)
    _step(net, 0)
    _step(net, 1)
    assert torch.equal(net._k_at(2 * N - 1), k)
    los = torch.ones(E, R, 1, dtype=torch.bool)
    los[0, 1] = False                                              # one link goes NLOS
    blocked = torch.zeros_like(los)
    blocked[1, 2] = True                                           # one LOS link gets blocked
    net.set_los(los, blocked)
    _step(net, 2)
    g0 = 2 * N
    ks = torch.stack([net._k_at(g0 - 1 + i) for i in range(8)])    # slots g0-1 .. g0+6
    for e, r in ((0, 1), (1, 2)):
        traj = ks[:, e, r]
        k0 = float(k[e, r])
        want = [k0, 0.75 * k0, 0.5 * k0, 0.25 * k0, 0.0, 0.0, 0.0, 0.0]
        assert torch.allclose(traj, torch.tensor(want), rtol=1e-6, atol=1e-6), (traj, want)
        assert float((traj[:-1] - traj[1:]).abs().max()) <= k0 / 4 * (1 + 1e-5)
    keep = torch.ones(E, R, dtype=torch.bool)
    keep[0, 1] = keep[1, 2] = False
    assert torch.equal(ks[:, keep], k[keep].expand(8, -1))        # other links untouched


def test_d_flip_back_mid_ramp_continues_from_current_k():
    net = _los_net(ramp=50)                                        # longer than a control step (20 slots)
    N = net.cfg.slots_per_step
    E, R = net.E, net.R
    on = torch.ones(E, R, 1, dtype=torch.bool)
    net.set_los(on)
    _step(net, 0)
    net.set_los(~on)                                               # all NLOS: ramp down over 500 slots
    _step(net, 1)
    k = net.k_los
    mid = net._k_at(2 * N - 1)
    assert torch.allclose(mid, k * (1 - N / 50), rtol=1e-5)
    net.set_los(on)                                                # back to LOS mid-ramp
    _step(net, 2)
    a, b = net._k_at(2 * N - 1), net._k_at(2 * N)
    assert torch.allclose(a, mid) and bool((b > a).all())
    assert float((b - a).abs().max()) <= float(((k - mid) / 50).max()) * (1 + 1e-4) + 1e-7
    assert torch.allclose(net._k_at(2 * N + 49), k)


def test_d_gain_uses_the_ramped_k():
    net = _los_net(ramp=4)
    net.set_los(torch.ones(net.E, net.R, 1, dtype=torch.bool))
    _step(net, 0)
    net.set_los(torch.zeros(net.E, net.R, 1, dtype=torch.bool))
    _step(net, 1)
    g = net.cfg.slots_per_step + 1                                 # second slot of the ramp: K = k_los / 2
    k = (net.k_los / 2)[..., None, None]
    x = torch.sqrt(k / (k + 1)) * net.spec[..., None, :] + torch.sqrt(1 / (k + 1)) * net.h
    assert torch.allclose(net._gain(g), 10 * torch.log10((x ** 2).sum(-1)), atol=1e-4)


def test_d_reset_snaps_first_los_state_and_ramp_zero_is_immediate():
    net = _los_net(ramp=4)
    on = torch.ones(net.E, net.R, 1, dtype=torch.bool)
    net.set_los(on)
    _step(net, 0)
    net.reset(torch.tensor([1]))
    assert float(net.k_lin[1].abs().max()) == 0.0 and bool(net.k_fresh[1]) and not bool(net.k_fresh[0])
    net.set_los(on)
    assert torch.equal(net.k_lin[1], net.k_los[1]) and torch.equal(net.k_from[1], net.k_los[1])
    z = _los_net(ramp=0)
    z.set_los(on)
    _step(z, 0)
    z.set_los(~on)
    _step(z, 1)
    assert float(z._k_at(z.cfg.slots_per_step).abs().max()) == 0.0


def test_d_k_draw_statistics():
    """K per LOS link is log-normal with the scenario's mu_K / sigma_K (InF: 7 / 8 dB)."""
    net = NRNet(400, 16, "cpu", SIZES, NRConfig(fading_rician=True, channel="tr38901_inf_sh"), seed=1)
    kdb = 10 * torch.log10(net.k_los)
    assert abs(float(kdb.mean()) - 7.0) < 0.2 and abs(float(kdb.std()) - 8.0) < 0.2
    phi = torch.atan2(net.spec[..., 1], net.spec[..., 0])
    assert abs(float(torch.cos(phi).mean())) < 0.03 and torch.allclose(net.spec.norm(dim=-1), torch.ones(1))
    assert rician_k_params("umi") == (9.0, 5.0) and set(RICIAN_K_DB) >= {"UMa", "RMa", "InH", "InF-DH"}


# ---------------------------------------------------------------------------------------------------------- (e)
@pytest.fixture
def fake_los(monkeypatch):
    """RadioMC.los_state() as the LOS-state feature will provide it: here LOS iff x < 75 m (deterministic in the
    poses, so it keeps every RNG property of the radio), no blockage state."""
    orig = RadioMC.pathgain_db

    def pathgain_db(self, pos):
        self._fake_pos = pos
        return orig(self, pos)

    def los_state(self):
        return (self._fake_pos[..., 0] < 75.0)[..., None].expand(-1, -1, self.C)

    monkeypatch.setattr(RadioMC, "pathgain_db", pathgain_db)
    monkeypatch.setattr(RadioMC, "los_state", los_state, raising=False)
    monkeypatch.setattr(RadioMC, "blocked_state", lambda self: None, raising=False)


R, EMAX, STEPS = 3, 6, 7
E_CASES = {"fixed": NRConfig(fading_rician=True, rician_k_db=7.0, dl=True, **FAST),
           "los": NRConfig(fading_rician=True, channel="tr38901_inf_sh", rician_k_ramp_slots=4, **FAST),
           "los_c3": multicell(3, fading_rician=True, rician_k_ramp_slots=30, **FAST)}


def _inputs(seed=0):
    g = torch.Generator().manual_seed(seed)
    return [(torch.randint(0, 3, (EMAX, R), generator=g), torch.rand(EMAX, R, 2, generator=g) * 150)
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
        outs[-1]["k"] = eng.net.k_lin.clone()
        if k in rs:
            eng.reset(torch.tensor(rs[k]))
    return outs


def _rows_equal(a, b, rows):
    """Row-wise equality across engines of different E: exact for integer / boolean outputs and on CUDA; the float
    outputs (K from the log-normal transform, SINR) to float32 rounding on the CPU, whose vectorized kernels may round
    differently for a different tensor shape (Linux x86 CI showed sub-print-precision differences in `k`)."""
    for x, y in zip(a, b):
        for name in x:
            u, v = x[name][rows].nan_to_num(-7.0), y[name][rows].nan_to_num(-7.0)
            if u.is_floating_point() and u.device.type == "cpu":
                assert torch.allclose(u, v, rtol=1e-5, atol=1e-5), name
            else:
                assert torch.equal(u, v), name


@pytest.mark.parametrize("case", list(E_CASES))
def test_e_env_independence_and_partial_reset(case, fake_los):
    cfg = E_CASES[case]
    mk = lambda E: make_engine("L2", E, R, "cpu", cfg, seed=5)    # noqa: E731
    a = _drive(mk(3), 3, {3: [1]})
    b = _drive(mk(6), 6, {3: [1]})
    _rows_equal(a, b, slice(0, 3))
    if case != "fixed":
        assert float(a[-1]["k"].max()) > 0 and float(a[-1]["k"].min()) == 0.0     # the LOS state reached K
    c = _drive(mk(4), 4)
    d = _drive(mk(4), 4, {2: [1]})
    _rows_equal(c, d, [0, 2, 3])
    assert not torch.equal(c[4]["sinr_db"][1], d[4]["sinr_db"][1]) or not torch.equal(c[4]["k"][1], d[4]["k"][1])
    e = _drive(mk(4), 4, {2: [0]})
    f = _drive(mk(4), 4, {0: [1], 1: [1, 2], 2: [0]})
    _rows_equal(e, f, 0)


def test_e_two_shards_by_global_env_id_equal_one_engine(fake_los):
    from isaac_net.core.sharded import set_env_offset
    cfg = E_CASES["los"]
    un = _drive(make_engine("L2", 6, R, "cpu", cfg, seed=5), 6)
    parts, off = [], 0
    for e in (2, 4):
        eng = make_engine("L2", e, R, "cpu", cfg, seed=5)
        set_env_offset(eng, off)
        outs = []
        for k, (send, pos) in enumerate(_inputs()):
            eng.submit(None, Requests(send[off:off + e], None, torch.full((e,), k, dtype=torch.long)))
            o = eng.step(None, pos[off:off + e])
            outs.append({n: v.clone() for n, v in o.items() if torch.is_tensor(v) and v.dim() >= 1 and v.shape[0] == e})
            outs[-1]["k"] = eng.net.k_lin.clone()
        parts.append(outs)
        off += e
    cat = [{n: torch.cat([p[k][n] for p in parts]) for n in parts[0][k]} for k in range(STEPS)]
    _rows_equal(un, cat, slice(None))


def test_e_no_los_state_means_k0(monkeypatch):
    """Without a radio LOS state (before the LOS-state feature, or SNR input) K stays 0: plain Rayleigh."""
    monkeypatch.setattr(RadioMC, "los_state", lambda self: None, raising=False)
    eng = make_engine("L2", 2, 3, "cpu", NRConfig(fading_rician=True, channel="tr38901_inf_sh", **FAST), seed=1)
    for _ in range(3):
        eng.submit(None, torch.ones(2, 3, dtype=torch.long))
        eng.step(None, torch.rand(2, 3, 2) * 50)
    assert float(eng.net.k_lin.abs().max()) == 0.0


# ---------------------------------------------------------------------------------------------------------- (f)
@pytest.mark.parametrize("name", nr_equiv.RICIAN_CFGS)
def test_f_equiv_configs_run_on_the_reference(name, fake_los):
    cfg = nr_equiv.CFGS[name]().with_(msg_sizes=SIZES, **FAST)
    assert cfg.fading_rician
    eng = make_engine("L2", 4, 3, "cpu", cfg, "reference", seed=3)
    for t, d in nr_equiv.Workload(cfg, 4, 3, 8, seed=4, p_reset=0.3, device="cpu", phase_offset=25):
        nr_equiv.drive(eng, d)
    sd = state_dict(eng)
    assert {"net.k_lin", "net.k_from", "net.k_g0", "net.spec"} <= set(sd)
    other = make_engine("L2", 4, 3, "cpu", cfg, "reference", seed=99)
    nr_equiv.copy_state(eng, other)
    assert torch.equal(other.net.spec, eng.net.spec) and torch.equal(other.net.k_lin, eng.net.k_lin)


def test_f_config_gating():
    assert NRConfig(fading_rician=True).unused_fields("L2") == []
    assert NRConfig(rician_k_db=5.0, rician_k_ramp_slots=2).unused_fields("L2") == ["rician_k_db", "rician_k_ramp_slots"]
    assert NRConfig(fading=False, fading_rician=True).unused_fields("L2") == ["fading_rician"]
    assert NRConfig(fading_rician=True, rician_k_db=5.0, rician_k_ramp_slots=2).unused_fields("L2") == \
        ["rician_k_ramp_slots"]
    assert NRConfig(fading_rician=True, rician_k_from_los=False, rician_k_ramp_slots=2).unused_fields("L2") == \
        ["rician_k_ramp_slots"]
    assert NRConfig(fading_rician=True, tr38901_scenario="UMi").unused_fields("L2") == []   # mu_K / sigma_K
    assert NRConfig(tr38901_scenario="UMi").unused_fields("L2") == ["tr38901_scenario"]
    assert NRConfig(fading_rician=True).unused_fields("L2-legacy") == ["fading_rician"]
    assert NRConfig(fading_rician=True, rician_k_from_los=False).rician_mode is None
    with pytest.raises(AssertionError):
        NRConfig(rician_k_ramp_slots=-1)
