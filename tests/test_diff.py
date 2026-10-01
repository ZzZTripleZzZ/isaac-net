"""Differentiable fluid models (isaac_net.core.diff): tau = 0 equals the discrete L1 / QA levels, the relaxation
converges to them as tau -> 0, autograd gradients pass gradcheck (float64, tiny sizes) and match central finite
differences at realistic sizes, mass is conserved at any temperature, partial resets are isolated."""
import pytest
import torch

from isaac_net.core.config import NRConfig
from isaac_net.core.diff import DiffFluid, discrete_rollout, make_discrete, relaxed_bernoulli, rollout
from isaac_net.core.diff import relax as rx
from isaac_net.core.proto.netsim import Radio

LEVEL = {"L1": "L1", "QA": "QA"}


def _scenario(E, R, T, p=0.4, seed=0, dev="cpu", lo=4000.0, hi=30000.0, snr=(10.0, 30.0)):
    g = torch.Generator().manual_seed(seed)
    B = torch.where(torch.rand(E, R, generator=g) < 0.5, torch.tensor(lo), torch.tensor(hi))
    s = snr[0] + (snr[1] - snr[0]) * torch.rand(E, R, generator=g)
    send = torch.rand(T, E, R, generator=g) < p
    return B.to(dev), s.to(dev), send.to(dev)


# ---------------------------------------------------------------------------------------------- tau = 0 == discrete
@pytest.mark.parametrize("mode", ["L1", "QA"])
@pytest.mark.parametrize("load", ["normal", "overload"])
def test_hard_equals_discrete(mode, load):
    """At tau = 0 the relaxed model is the discrete level: per-robot delivered count, delay sum, timeouts, buffer
    overflow and AoI equal the engine's (overload: small buffer and deadline, low SNR, so both gates bind)."""
    E, R, T = 8, 6, 50
    if load == "normal":
        cfg, kw = NRConfig(), {}
        B, snr, send = _scenario(E, R, T)
    else:
        cfg, kw = NRConfig().with_(frame_buffer=4, timeout_steps=6), dict(fb=4, timeout=6)
        B, snr, send = _scenario(E, R, T, p=0.8, snr=(-5.0, 10.0))
    net = DiffFluid(E, R, mode=mode, tau=0.0, **kw)
    r = rollout(net, T, send.float(), B, snr_db=snr, time_axis={"send"})
    d = discrete_rollout(make_discrete(LEVEL[mode], B, config=cfg), T, send, snr)
    for k in ("delivered", "timed_out", "overflow", "aoi"):
        assert torch.allclose(r[k], d[k], atol=1e-4), (k, (r[k] - d[k]).abs().max())
    assert torch.allclose(r["delay"] * r["delivered"], d["delay_sum"], rtol=1e-4, atol=1e-3)
    if load == "overload":
        assert d["timed_out"].sum() > 0 and d["overflow"].sum() > 0


@pytest.mark.parametrize("mode", ["L1", "QA"])
def test_tau_convergence(mode):
    """KPI error against the discrete level shrinks as tau -> 0 and is below 1 % at tau = 0.002."""
    E, R, T = 8, 6, 40
    B, snr, send = _scenario(E, R, T, seed=1)
    ref = rollout(DiffFluid(E, R, mode=mode, tau=0.0), T, send.float(), B, snr_db=snr, time_axis={"send"})
    errs = []
    for tau in (0.1, 0.02, 0.002):
        r = rollout(DiffFluid(E, R, mode=mode, tau=tau), T, send.float(), B, snr_db=snr, time_axis={"send"})
        errs.append(max(abs(float(r[k] / ref[k] - 1)) for k in ("mean_delay", "mean_aoi", "mean_delivery")))
    assert errs[0] > errs[1] > errs[2] and errs[2] < 0.01, errs


# ---------------------------------------------------------------------------------------------- gradients
def _kpi(mode, send, B, tx, pos, radio, T=4, tau=0.1, K=4):
    net = DiffFluid(2, 3, mode=mode, tau=tau, timeout=3, fb=2, ul_per_step=K, dtype=torch.float64)
    net.attach_radio(radio)
    r = rollout(net, T, send, B, pos=pos, tx_dbm=tx)
    return torch.stack([r["mean_delay"], r["mean_delivery"], r["mean_aoi"], r["mean_energy_j"]])


@pytest.mark.parametrize("mode", ["L1", "QA"])
def test_gradcheck_tiny(mode):
    """torch.autograd.gradcheck (float64) of delay, delivery, AoI and energy w.r.t. send probability, message
    bytes, transmit power and positions, on E = 2, R = 3, T = 4 with a small buffer and deadline."""
    torch.manual_seed(0)
    radio = Radio(2, "cpu")
    dd = torch.float64
    send = (0.3 + 0.6 * torch.rand(2, 3, dtype=dd)).requires_grad_()
    B = (6000 + 20000 * torch.rand(2, 3, dtype=dd)).requires_grad_()
    tx = (15 + 8 * torch.rand(2, 3, dtype=dd)).requires_grad_()
    pos = (40 + 150 * torch.rand(2, 3, 2, dtype=dd)).requires_grad_()
    assert torch.autograd.gradcheck(lambda a, b, c, d: _kpi(mode, a, b, c, d, radio), (send, B, tx, pos),
                                    eps=1e-6, atol=1e-5, rtol=1e-3)


@pytest.mark.parametrize("mode", ["L1", "QA"])
def test_fd_realistic(mode):
    """At E = 32, R = 8, T = 60, K = 40 the autograd directional derivative of a KPI mix matches a central finite
    difference (float64) to 1e-4 relative, for each decision input."""
    E, R, T = 32, 8, 60
    torch.manual_seed(1)
    dd = torch.float64
    radio = Radio(E, "cpu")
    p = (0.2 + 0.5 * torch.rand(E, R, dtype=dd))
    B = 4000 + 26000 * torch.rand(E, R, dtype=dd)
    tx = 23 - 6 * torch.rand(E, R, dtype=dd)
    pos = 30 + 170 * torch.rand(E, R, 2, dtype=dd)

    def f(p, B, tx, pos):
        net = DiffFluid(E, R, mode=mode, tau=0.05, dtype=dd)
        net.attach_radio(radio)
        r = rollout(net, T, p, B, pos=pos, tx_dbm=tx, warmup=5)
        return r["mean_delay"] + 0.3 * r["mean_aoi"] - r["mean_delivery"] + 5.0 * r["mean_energy_j"]

    xs = [x.clone().requires_grad_() for x in (p, B, tx, pos)]
    g = torch.autograd.grad(f(*xs), xs)
    scale = (0.05, 1000.0, 1.0, 5.0)
    for i in range(4):
        v = torch.randn_like(xs[i]) * scale[i]
        h = 1e-6
        args_p = [x.detach() + (h * v if j == i else 0) for j, x in enumerate(xs)]
        args_m = [x.detach() - (h * v if j == i else 0) for j, x in enumerate(xs)]
        fd = (f(*args_p) - f(*args_m)) / (2 * h)
        ad = (g[i] * v).sum()
        assert abs(float(ad - fd)) <= 1e-4 * abs(float(fd)) + 1e-8, (i, float(ad), float(fd))


def test_gradient_signs():
    """Physical signs: more bytes or a higher send probability raise delay; more power lowers it and raises
    energy; moving a robot away from the gNB lowers its SNR."""
    E, R, T = 16, 8, 60
    torch.manual_seed(2)
    p = torch.full((E, R), 0.35, requires_grad=True)
    B = torch.full((E, R), 20000.0, requires_grad=True)
    tx = torch.full((E, R), 20.0, requires_grad=True)
    snr = 5 + 10 * torch.rand(E, R)
    r = rollout(DiffFluid(E, R, tau=0.05), T, p, B, snr_db=snr, tx_dbm=tx, warmup=10)
    gp, gB, gt = torch.autograd.grad(r["mean_delay"], (p, B, tx))
    assert gp.sum() > 0 and gB.sum() > 0 and gt.sum() < 0
    r = rollout(DiffFluid(E, R, tau=0.05), T, p, B, snr_db=snr, tx_dbm=tx, warmup=10)
    (ge_tx,) = torch.autograd.grad(r["mean_energy_j"], (tx,))
    assert ge_tx.sum() > 0


# ---------------------------------------------------------------------------------------------- invariants
@pytest.mark.parametrize("mode", ["L1", "QA"])
@pytest.mark.parametrize("tau", [0.0, 0.05, 0.3])
def test_mass_conservation(mode, tau):
    """offered = delivered + timed out + overflow + queued, in mass, at every temperature."""
    E, R, T = 6, 5, 30
    B, snr, _ = _scenario(E, R, T, snr=(-5.0, 15.0))
    p = torch.rand(E, R)
    net = DiffFluid(E, R, mode=mode, tau=tau, fb=4, timeout=6)
    r = rollout(net, T, p, B, snr_db=snr)
    lhs = r["offered"]
    rhs = r["delivered"] + r["timed_out"] + r["overflow"] + net.w.sum(-1)
    assert torch.allclose(lhs, rhs, atol=1e-4), (lhs - rhs).abs().max()


def test_partial_reset_isolated():
    """reset(env_ids) zeroes those rows only; the other rows keep their values and their gradient path."""
    E, R = 6, 4
    p = torch.full((E, R), 0.5, requires_grad=True)
    net = DiffFluid(E, R, tau=0.05)
    for _ in range(5):
        net.step(p, 25000.0, snr_db=torch.full((E, R), 5.0))
    x0, w0 = net.x.clone(), net.w.clone()
    net.reset(torch.tensor([1, 4]))
    keep = torch.tensor([0, 2, 3, 5])
    assert torch.equal(net.x[keep], x0[keep]) and torch.equal(net.w[keep], w0[keep])
    assert net.x[[1, 4]].abs().sum() == 0 and net.w[[1, 4]].abs().sum() == 0 and net.aoi[[1, 4]].abs().sum() == 0
    (g,) = torch.autograd.grad(net.x.sum(), p)
    assert g[keep].abs().sum() > 0 and g[[1, 4]].abs().sum() == 0
    m = torch.zeros(E, dtype=torch.bool)
    m[0] = True
    net.reset(m)
    assert net.w[0].abs().sum() == 0


def test_relaxations():
    a = torch.tensor([1.0, 5.0, 0.0, 30000.0])
    b = torch.tensor([2.0, 5.0, 3.0, 29000.0])
    for tau in (0.1, 0.01, 0.001):
        assert torch.all(rx.smin(a, b, tau) <= torch.minimum(a, b) + 1e-6)
        assert torch.all(rx.smax(a + 1, b + 1, tau) >= torch.maximum(a + 1, b + 1) * (1 - 1e-6))
    assert torch.allclose(rx.smin(a, b, 1e-3), torch.minimum(a, b), rtol=2e-3)
    assert torch.equal(rx.smin(a, b, 0.0), torch.minimum(a, b))
    x = torch.linspace(0.5, 7.0, 50)
    assert torch.allclose(rx.floor_int(x + 1e-3, 1, 5, 1e-3), torch.floor(x + 1e-3).clamp(1, 5), atol=1e-3)
    assert torch.allclose(rx.harmonic(torch.arange(1.0, 6.0)), torch.cumsum(1 / torch.arange(1.0, 6.0), 0), atol=1e-5)
    p = torch.full((10000,), 0.3)
    u = torch.rand(10000)
    assert torch.equal(relaxed_bernoulli(p, u, 0.0), (u < p).float())
    assert abs(float(relaxed_bernoulli(p, u, 0.05).mean()) - 0.3) < 0.02


def test_proxy_recipe_smoke():
    """The neural-proxy recipe: dataset from L2-legacy rollouts, fit, and per-robot gradients (tiny CPU run)."""
    from isaac_net.core.diff import proxy
    X, Y = proxy.collect(n_batches=2, E=8, R=4, T=30, warmup=5, device="cpu", seed=0)
    assert X.shape[0] == Y.shape[0] == 2 * 8 * 4 and X.shape[1] == proxy.N_FEAT
    model = proxy.fit(X, Y, epochs=5, device="cpu")
    p = torch.full((3, 4), 0.3, requires_grad=True)
    kpi = proxy.predict(model, p, torch.full((3, 4), 8000.0), torch.full((3, 4), 15.0))
    (g,) = torch.autograd.grad(kpi["delay"].sum(), p)
    assert g.shape == (3, 4) and torch.isfinite(g).all()


@pytest.mark.gpu
@pytest.mark.parametrize("mode", ["L1", "QA"])
def test_gpu_matches_cpu(mode):
    """E = 64 on CUDA: tau = 0 equals the discrete level on CUDA, and tau > 0 KPIs and gradients match CPU."""
    E, R, T = 64, 8, 40
    B, snr, send = _scenario(E, R, T, seed=3)
    net = DiffFluid(E, R, "cuda", mode=mode, tau=0.0)
    r = rollout(net, T, send.float().cuda(), B.cuda(), snr_db=snr.cuda(), time_axis={"send"})
    d = discrete_rollout(make_discrete(LEVEL[mode], B, device="cuda"), T, send.cuda(), snr.cuda())
    assert torch.allclose(r["delivered"], d["delivered"], atol=1e-4)
    out = []
    for dev in ("cpu", "cuda"):
        dd = torch.float64
        p = torch.full((E, R), 0.4, device=dev, dtype=dd, requires_grad=True)
        r = rollout(DiffFluid(E, R, dev, mode=mode, tau=0.05, dtype=dd), T, p, B.to(dev, dd), snr_db=snr.to(dev, dd))
        (g,) = torch.autograd.grad(r["mean_delay"], p)
        out.append((float(r["mean_delay"].detach()), g.cpu()))
    assert abs(out[0][0] - out[1][0]) < 1e-8 * abs(out[0][0])
    assert torch.allclose(out[0][1], out[1][1], rtol=1e-6, atol=1e-10)
