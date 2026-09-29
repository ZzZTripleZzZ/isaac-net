"""A run seeded with torch.manual_seed is reproducible bit for bit, at every level, on CPU."""
import pytest
import torch

from engine_api import LEVELS, Workload, collect, enable_stats, make_fast, make_ref, run, state


def _trace(level, seed, fast_backend=None, E=3, R=5, steps=25):
    torch.manual_seed(seed)
    net = make_ref(level, E, R, "cpu") if fast_backend is None else make_fast(E, R, "cpu", fast_backend)
    enable_stats(net)
    wl = Workload(E, R, "cpu", seed=41, period=8)
    outs = []
    run(net, wl, steps, on_step=lambda t, n, newest, det: outs.append((newest.clone(), det.clone())))
    return outs, state(net), collect(net)


def _same(a, b):
    (oa, sa, ca), (ob, sb, cb) = a, b
    if len(oa) != len(ob) or any(not (torch.equal(x[0], y[0]) and torch.equal(x[1], y[1])) for x, y in zip(oa, ob)):
        return False
    if sa.keys() != sb.keys() or any(not torch.equal(sa[k], sb[k]) for k in sa):
        return False
    return all((ca[k] == cb[k]) if k == "overflow" else torch.equal(ca[k], cb[k]) for k in ca)


@pytest.mark.parametrize("level", LEVELS)
def test_seeded_run_is_deterministic(level):
    assert _same(_trace(level, 7), _trace(level, 7))


@pytest.mark.parametrize("level", ["L0", "L0DR", "L05", "L2"])
def test_different_seed_changes_random_levels(level):
    assert not _same(_trace(level, 7), _trace(level, 8))


def test_fast_eager_backend_is_deterministic():
    assert _same(_trace("L2", 7, "eager"), _trace("L2", 7, "eager"))
