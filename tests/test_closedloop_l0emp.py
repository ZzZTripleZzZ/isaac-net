"""L0's empirical mode (params {"q", "p"}, proto/netsim.py l0_quantile_delay) and the L0-emp arm of the closed-loop
benchmark (benchmarks/closedloop/run_closedloop.py fit_l0_emp, docs/closed-loop.md): delays are resampled i.i.d.
from the given sample whatever the SNR, loss stays the i.i.d. draw of the lognormal mode, the fast backends equal
the reference, and the lognormal default is unchanged."""
import importlib.util
import os

import numpy as np
import pytest
import torch

from isaac_net.core import NRConfig, make_engine

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CFG = NRConfig(msg_sizes=(4000.0, 30000.0), timeout_steps=20, control_step_ms=100.0, frame_buffer=16)


def _runner():
    spec = importlib.util.spec_from_file_location(
        "run_closedloop", os.path.join(ROOT, "benchmarks", "closedloop", "run_closedloop.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _sample(seed=0):
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.lognormal(np.log(250), 0.2, 4000), rng.uniform(600, 1500, 400)])   # ms, heavy tail


def _draw(params, E=16, R=8, T=60, snr=25.0, seed=3, backend="reference", device="cpu", every=4):
    """Every robot submits one class-2 frame every `every` steps; returns delivered delays (steps) and lost count."""
    eng = make_engine("L0", E, R, device, CFG, backend, params=params, seed=seed)
    eng.reset()
    s = torch.full((E, R), snr, device=device)
    d, lost = [], 0
    for t in range(T):
        send = torch.full((E, R), 2 if t % every == 0 else 0, dtype=torch.long, device=device)
        eng.submit(None, send)
        out = eng.step(None, s)
        d.append(out["delay"][out["delivered"]].float().cpu())
        lost += int(out["timed_out"].sum())
    return torch.cat(d).numpy(), lost


def test_fit_keeps_the_whole_sample_and_the_loss_rule():
    m = _runner()
    x = _sample()
    params, rows = m.fit_l0_emp(x, delivered=990, resolved=1000)
    assert params["q"].shape == (len(x),) and np.allclose(params["q"].numpy(), np.sort(x) / 100, rtol=1e-6)
    assert params["p"] == pytest.approx(0.01)                           # whole sample within the 20-step deadline
    assert rows[95]["delay_steps"] == pytest.approx(np.quantile(x, 0.95, method="inverted_cdf") / 100, rel=1e-6)
    y = np.append(x, [2500.0] * 50)                                    # 50 delays past the 2 s deadline
    params, _ = m.fit_l0_emp(y, delivered=1, resolved=1)
    assert params["p"] == 0.0                                          # as fit_l0: no extra loss when L2 lost none


def test_draws_resample_the_marginal_whatever_the_snr():
    m = _runner()
    x = _sample(1)
    params, _ = m.fit_l0_emp(x, delivered=1, resolved=1)
    for snr in (25.0, 2.0):
        d, lost = _draw(params, snr=snr)
        assert lost == 0
        assert m.ks(d, x / 100) < 0.05
        srt = np.sort(x) / 100                                          # every draw is a sample value (float32)
        i = np.searchsorted(srt, d).clip(1, len(srt) - 1)
        assert np.minimum(abs(srt[i] - d), abs(srt[i - 1] - d)).max() < 1e-4
        assert np.quantile(d, 0.99) == pytest.approx(np.quantile(x, 0.99) / 100, rel=0.1)
        assert np.mean(d > 6.0) == pytest.approx(np.mean(x > 600.0), abs=0.02)
    # the lognormal with the same median and log sigma has a much shorter p95 than this marginal
    ls = np.log(x)
    assert np.exp(np.median(ls) + 1.645 * ls.std()) < 0.9 * np.quantile(x, 0.95)


def test_loss_is_iid_at_the_fitted_rate_and_late_draws_time_out():
    params = {"q": torch.full((100,), 2.0), "p": 0.2}
    d, lost = _draw(params, T=400)
    assert np.allclose(d, 2.0, atol=1e-5)
    assert lost / (lost + len(d)) == pytest.approx(0.2, abs=0.03)
    q = torch.cat([torch.full((90,), 2.0), torch.full((10,), 25.0)])     # 10% beyond the 20-step deadline
    d, lost = _draw({"q": q, "p": 0.0}, T=400, every=25)
    assert np.allclose(d, 2.0, atol=1e-5)
    assert lost / (lost + len(d)) == pytest.approx(0.1, abs=0.03)


def test_lognormal_default_unchanged():
    """Without q, L0 draws exp(mu + sig z) as before; without params it takes the config defaults."""
    p = {"mu": float(np.log(2.5)), "sig": 0.3, "p": 0.0}
    d, _ = _draw(p, T=40)
    a, _ = _draw(p, T=40)
    assert np.array_equal(d, a)
    assert np.median(d) == pytest.approx(2.5, rel=0.05)
    assert np.std(np.log(d)) == pytest.approx(0.3, rel=0.1)
    cfgd, _ = _draw(None, T=40)                                         # config default (0.05 steps, 0.5)
    assert np.median(cfgd) == pytest.approx(0.05, rel=0.1)


def test_eager_equals_reference():
    params = {"q": torch.tensor(np.sort(_sample(2)) / 100, dtype=torch.float32), "p": 0.05}
    a, la = _draw(params, backend="reference")
    b, lb = _draw(params, backend="eager")
    assert la == lb and np.array_equal(a, b)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_graph_equals_reference_on_gpu():
    params = {"q": torch.tensor(np.sort(_sample(2)) / 100, dtype=torch.float32), "p": 0.05}
    a, la = _draw(params, backend="reference", device="cuda")
    b, lb = _draw(params, backend="graph", device="cuda")
    assert la == lb and np.array_equal(a, b)
