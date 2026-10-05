"""Follow-up fixes to the engine-RNG radio:

  1. ShardedEngine is shard-invariant on L2 (one cell and multicell(3)) and on the multi-cell L2-legacy (NetSlotMC)
     under rng="engine" without traffic models: the radio draws from the engine counter RNG, keyed by global env id
     (sharded.set_env_offset). Checked through ShardedEngine with splits (2, 4) and (3, 3) and partial resets that
     cross the shard boundary, against one engine of E = 6. rng="global", NRConfig.traffic and L2 background UEs
     (ghost traffic models) keep distinct per-shard seeds.
  2. Level WIFI: RadioMC over the APs draws from the WifiNet's counter RNG, so with poses (a) env 0 is the same at
     E = 3 and E = 6, (b) other envs' resets do not shift env 0, (c) a partial reset leaves the other envs untouched;
     rng="global" keeps the generator path. WIFI is shard-invariant too.

Every comparison is bitwise (on this CPU the new cases show no shape-dependent rounding). _close(bitwise=False) is
the fallback to use if a machine's CPU kernels round differently with the batch shape.
"""
import pytest
import torch

from isaac_net.core import NRConfig, Requests, make_engine
from isaac_net.core.background import BackgroundConfig
from isaac_net.core.config import multicell
from isaac_net.core.sharded import ShardedEngine, shard_invariant
from isaac_net.core.traffic import TrafficModel
from isaac_net.core.wifi import WifiConfig

R, EMAX, STEPS, SEED = 3, 6, 7, 5
APS = ((0.0, 0.0), (40.0, 0.0), (20.0, 30.0))

CASES = {
    "L2": ("L2", NRConfig()),
    "L2-mc3": ("L2", multicell(3)),
    "L2-legacy-mc3": ("L2-legacy", multicell(3)),
    "WIFI": ("WIFI", NRConfig(wifi=WifiConfig(ap_positions_m=APS))),
}
WIFI_CASES = {
    "log_distance": NRConfig(wifi=WifiConfig(ap_positions_m=APS)),
    "white": NRConfig(shadow_white_frac=0.5, wifi=WifiConfig(ap_positions_m=APS)),
    "tr38901-o2i": NRConfig(channel="tr38901", tr38901_scenario="UMi", o2i_indoor_frac=0.5, shadow_white_frac=0.0,
                            wifi=WifiConfig(ap_positions_m=APS)),
}


def _inputs(steps=STEPS, seed=0, arena=150.0):
    g = torch.Generator().manual_seed(seed)
    return [(torch.randint(0, 3, (EMAX, R), generator=g), torch.rand(EMAX, R, 2, generator=g) * arena)
            for _ in range(steps)]


def _drive(net, E, resets=(), steps=STEPS, arena=150.0):
    rs = dict(resets)
    outs = []
    for k, (send, pos) in enumerate(_inputs(steps, arena=arena)):
        net.submit(None, Requests(send[:E], None, torch.full((E,), k, dtype=torch.long)))
        o = net.step(None, pos[:E])
        outs.append({n: v.clone() for n, v in o.items() if torch.is_tensor(v) and v.dim() >= 1 and v.shape[0] == E})
        if k in rs:
            net.reset(torch.tensor(rs[k]))
    return outs


def _close(u, v, bitwise):
    """Bitwise, or (bitwise=False, floating point) to 1e-6: CPU kernels may round differently with the batch shape
    (one ulp, as test_sharded.py::test_two_shards_bitwise_cpu[split1-L2-legacy] shows on some machines)."""
    if u.is_floating_point():
        u, v = u.nan_to_num(-7.0), v.nan_to_num(-7.0)
        if not bitwise:
            return torch.allclose(u, v, rtol=1e-6, atol=1e-6)
    return torch.equal(u, v)


def _rows_equal(a, b, rows, bitwise=True):
    n = 0
    for k, (x, y) in enumerate(zip(a, b)):
        assert x.keys() == y.keys()
        for name in x:
            assert _close(x[name][rows], y[name][rows], bitwise), (k, name)
            n += 1
    assert n > 0


# ---------------------------------------------------------------------------------------------- 1. sharding
@pytest.mark.parametrize("case", list(CASES))
@pytest.mark.parametrize("split", [(2, 4), (3, 3)])
def test_sharded_engine_equals_one_engine(case, split):
    level, cfg = CASES[case]
    E = EMAX
    resets = {2: [1, 2, 3, 4], 4: [0, 5], 5: [2, 3]}        # both rounds cross the boundary of either split
    arena = 60.0 if level == "WIFI" else 150.0
    un = _drive(make_engine(level, E, R, "cpu", cfg, seed=SEED), E, resets, arena=arena)
    sh = ShardedEngine(level, E, R, ["cpu", "cpu"], cfg, seed=SEED, split=list(split))
    assert sh.shard_invariant
    _rows_equal(un, _drive(sh, E, resets, arena=arena), slice(None))


@pytest.mark.parametrize("case", list(CASES))
def test_sharded_engine_differs_without_offsets(case):
    """Sanity: the comparison above is sensitive. A shard at offset 0 for envs 3..5 (no set_env_offset) differs."""
    level, cfg = CASES[case]
    arena = 60.0 if level == "WIFI" else 150.0
    un = _drive(make_engine(level, EMAX, R, "cpu", cfg, seed=SEED), EMAX, arena=arena)
    lone = make_engine(level, 3, R, "cpu", cfg, seed=SEED)
    outs = []
    for k, (send, pos) in enumerate(_inputs(arena=arena)):
        lone.submit(None, Requests(send[3:], None, torch.full((3,), k, dtype=torch.long)))
        outs.append(lone.step(None, pos[3:]))
    key = "wifi_rate_mbps" if level == "WIFI" else "sinr_db"
    assert any(not torch.equal(un[k][key][3:].nan_to_num(-7.0), outs[k][key].nan_to_num(-7.0))
               for k in range(STEPS))


def test_shard_invariant_rule():
    assert shard_invariant("L2", NRConfig()) and shard_invariant("L2", multicell(3))
    assert shard_invariant("L2-legacy", multicell(3)) and shard_invariant("WIFI", NRConfig())
    assert not shard_invariant("L2", NRConfig(rng="global"))
    assert not shard_invariant("L2-legacy", multicell(3).with_(rng="global"))
    assert not shard_invariant("L2", NRConfig(traffic=(TrafficModel.periodic(500, 50.0),)))
    assert not shard_invariant("L2", NRConfig(background=BackgroundConfig(n_background=2)))
    assert shard_invariant("L2-legacy", NRConfig(background=BackgroundConfig(n_background=2)))
    assert shard_invariant("L2", NRConfig(traffic=(TrafficModel.policy(),)))   # policy() generates nothing


@pytest.mark.parametrize("cfg", [NRConfig(rng="global"), NRConfig(traffic=(TrafficModel.periodic(500, 50.0),))],
                         ids=["global", "traffic"])
def test_non_invariant_l2_gets_derived_seeds(cfg):
    sh = ShardedEngine("L2", EMAX, R, ["cpu", "cpu"], cfg, seed=1)
    assert not sh.shard_invariant
    assert sh.shards[0].seed != sh.shards[1].seed


# ---------------------------------------------------------------------------------------------- 2. WIFI radio
def _wifi(case, E, cfg=None):
    return make_engine("WIFI", E, R, "cpu", WIFI_CASES[case] if cfg is None else cfg, seed=SEED)


def _probe(net, E):
    """Received power [E,R,A] of the engine's radio at fixed poses: the shadowing state of every env."""
    pos = torch.rand(EMAX, R, 2, generator=torch.Generator().manual_seed(3))[:E] * 60
    return net.radio.rx_dbm(pos)


@pytest.mark.parametrize("case", list(WIFI_CASES))
def test_wifi_env0_independent_of_E(case):
    resets = {3: [1]}
    na, nb = _wifi(case, 3), _wifi(case, 6)
    a, b = _drive(na, 3, resets, arena=60.0), _drive(nb, 6, resets, arena=60.0)
    _rows_equal(a, b, slice(0, 3))
    assert torch.equal(_probe(na, 3), _probe(nb, 6)[:3])
    assert na.radio.rng is na.rng


@pytest.mark.parametrize("case", list(WIFI_CASES))
def test_wifi_other_resets_do_not_shift_env0(case):
    na, nb = _wifi(case, 4), _wifi(case, 4)
    a = _drive(na, 4, {2: [0]}, arena=60.0)
    b = _drive(nb, 4, {0: [1], 1: [1, 2], 2: [0]}, arena=60.0)
    _rows_equal(a, b, 0)
    _rows_equal(a, b, 3)
    pa, pb = _probe(na, 4), _probe(nb, 4)
    assert torch.equal(pa[[0, 3]], pb[[0, 3]])


@pytest.mark.parametrize("case", list(WIFI_CASES))
def test_wifi_partial_reset_leaves_others_untouched(case):
    na, nb = _wifi(case, 4), _wifi(case, 4)
    a = _drive(na, 4, arena=60.0)
    b = _drive(nb, 4, {2: [1]}, arena=60.0)
    _rows_equal(a, b, [0, 2, 3])
    pa, pb = _probe(na, 4), _probe(nb, 4)
    assert torch.equal(pa[[0, 2, 3]], pb[[0, 2, 3]])
    assert not torch.equal(pa[1], pb[1])                  # env 1 is in a new episode: a new shadowing draw


def test_wifi_global_rng_keeps_the_generator_path():
    net = _wifi("log_distance", 3, NRConfig(rng="global", wifi=WifiConfig(ap_positions_m=APS)))
    _drive(net, 3, steps=1, arena=60.0)
    assert net.radio.rng is None and net.radio.gen is net.gen
