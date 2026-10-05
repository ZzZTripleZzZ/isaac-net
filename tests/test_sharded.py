"""ShardedEngine (core/sharded.py): two shards on one device are bitwise equal to the unsharded engine.

Engine-owned randomness is keyed by global env id in every shard (CounterRNG.set_env_offset, and the Triton kernels'
env offset), so an env draws the same numbers whether it lives in shard 0 or 1. Checked with partial resets that
cross the shard boundary, on the prototype, surrogate-free bound and L2-legacy levels, with background load and the
energy model on top (CPU reference), and (gpu) on the L2-legacy graph and triton backends and L1 triton. The NR
engine L2 with traffic models is not shard-invariant (one TrafficGen generator per engine) and says so; L2 without
them, multi-cell L2-legacy and WIFI are covered in test_followup_fixes.py.
"""
import pytest
import torch

from isaac_net.core import NRConfig, Requests, make_engine
from isaac_net.core.background import BackgroundConfig
from isaac_net.core.energy import EnergyConfig
from isaac_net.core.proto.rng import CounterRNG
from isaac_net.core.sharded import ShardedEngine

E, R = 7, 3


def _drive(net, dev, steps=10, seed=0, pos=True, resets=((3, (1, 4, 5)), (6, (0, 6)))):
    g = torch.Generator().manual_seed(seed)
    outs = []
    rs = dict(resets)
    for k in range(steps):
        send = torch.randint(0, 3, (E, R), generator=g).to(dev)
        det = (torch.rand(E, R, generator=g) < 0.3).to(dev)
        x = (torch.rand(E, R, 2, generator=g) * 100 if pos else 25 * torch.rand(E, R, generator=g) - 5).to(dev)
        acc = net.submit(None, Requests(send, det, torch.full((E,), k, dtype=torch.long, device=dev)))
        o = net.step(None, x)
        o["accepted"] = acc
        outs.append(o)
        if k in rs:
            net.reset(torch.tensor(rs[k], device=dev))
    return outs


def _same(a, b):
    for x, y in zip(a, b):
        assert x.keys() == y.keys()
        for k in y:
            u, v = x[k], y[k]
            if not torch.is_tensor(v):
                continue
            if v.is_floating_point():
                u, v = u.nan_to_num(-7.0), v.nan_to_num(-7.0)
            assert torch.equal(u.cpu(), v.cpu()), k
    return True


def test_counter_rng_offset_is_shard_invariant():
    a = CounterRNG(3, 8, "cpu")
    a.reset(None)
    b = CounterRNG(3, 5, "cpu")
    b.set_env_offset(3)
    b.reset(None)
    a.reset(torch.tensor([4, 5]))           # global envs 4, 5 = local 1, 2 of the shard at offset 3
    b.reset(torch.tensor([1, 2]))
    for r in (a, b):
        r.tick(2)
    assert torch.equal(a.normal(2, 4, 6)[3:], b.normal(2, 4, 6))
    assert torch.equal(a.reset_uniform(torch.tensor([4, 6]), 1, 3), b.reset_uniform(torch.tensor([1, 3]), 1, 3))


@pytest.mark.parametrize("level", ["L0", "L0DR", "L1", "L2-legacy", "ORACLE", "NOCOMM"])
@pytest.mark.parametrize("split", [None, [2, 5]])
def test_two_shards_bitwise_cpu(level, split):
    cfg = NRConfig()
    un = make_engine(level, E, R, "cpu", cfg, seed=11)
    sh = ShardedEngine(level, E, R, ["cpu", "cpu"], cfg, seed=11, split=split)
    assert sh.shard_invariant
    _same(_drive(sh, "cpu", pos=level != "L1"), _drive(un, "cpu", pos=level != "L1"))
    assert torch.equal(sh.clock, un.clock) and torch.equal(sh.queued(), un.queued())


@pytest.mark.parametrize("level", ["L1", "L2-legacy"])
def test_two_shards_with_background_and_energy(level):
    cfg = NRConfig(background=BackgroundConfig(n_background=4, mobility="random_waypoint", speed_mps=5.0),
                   energy=EnergyConfig(initial_soc=(0.3, 1.0)))
    un = make_engine(level, E, R, "cpu", cfg, seed=2)
    sh = ShardedEngine(level, E, R, ["cpu", "cpu"], cfg, seed=2, split=[4, 3])
    _same(_drive(sh, "cpu", pos=True), _drive(un, "cpu", pos=True))
    assert torch.equal(sh.energy_obs(), un.energy_obs())


def test_locate_and_mapping():
    sh = ShardedEngine("L0", E, R, ["cpu", "cpu", "cpu"], NRConfig(), seed=0, split=[2, 2, 3])
    assert sh.offsets == [0, 2, 4]
    assert [sh.locate(e) for e in (0, 1, 2, 5, 6)] == [(0, 0), (0, 1), (1, 0), (2, 1), (2, 2)]
    assert sh.shard_of.tolist() == [0, 0, 1, 1, 2, 2, 2] and sh.local_of.tolist() == [0, 1, 0, 1, 0, 1, 2]
    with pytest.raises(IndexError):
        sh.locate(7)


def test_nr_engine_shards_run_but_are_not_invariant():
    """L2 with traffic models (TrafficGen: one sequential generator per engine) is not shard-invariant and says so;
    without them L2 is (tests/test_followup_fixes.py)."""
    from isaac_net.core.traffic import TrafficModel
    sh = ShardedEngine("L2", E, R, ["cpu", "cpu"], NRConfig(traffic=(TrafficModel.periodic(500, 50.0),)), seed=1)
    assert not sh.shard_invariant
    o = _drive(sh, "cpu", steps=3)[-1]
    assert o["delivered"].shape == (E, R, 16) and o["t"].shape == (E,)
    newest, det = sh.step(None, torch.full((E, R), 10.0), torch.zeros(E, dtype=torch.long))
    assert newest.shape == (E, R) and det.shape == (E,)


@pytest.mark.gpu
@pytest.mark.parametrize("level,backend", [("L2-legacy", "graph"), ("L2-legacy", "triton"), ("L1", "triton"),
                                           ("L0DR", "graph"), ("L2-legacy", "reference")])
def test_two_shards_one_gpu_bitwise(cuda, level, backend):
    cfg = NRConfig()
    un = make_engine(level, E, R, cuda, cfg, backend=backend, seed=21)
    sh = ShardedEngine(level, E, R, [cuda, cuda], cfg, backend=backend, seed=21, split=[3, 4])
    _same(_drive(sh, cuda, steps=12), _drive(un, cuda, steps=12))
