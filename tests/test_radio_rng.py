"""Radio draws under the engine RNG (NRConfig.rng = "engine"): with poses driving the channel, an env's shadowing,
LOS and O2I draws are keyed by (seed, env id, episode) through the engine's CounterRNG (core/radio.py), so

  (a) env 0 is bitwise the same at E = 3 and E = 6,
  (b) resetting env 1 earlier does not change env 0 after env 0's own reset,
  (c) a partial reset leaves the other envs bitwise unaffected,
  (d) two shards keyed by global env id (sharded.set_env_offset) equal one engine bitwise,

on L2 with one cell, L2 with multicell(3) and L2-legacy with multicell(3) (NetSlotMC). Plus (e) NetSlotMC counts
the A3 time-to-trigger and the handover interruption in its own UL slots (proto_slots_per_step), and the C = 1
log-distance field equals the prototype Radio's under the same engine RNG.
"""
import pytest
import torch

from isaac_net.core import NRConfig, Requests, make_engine
from isaac_net.core.config import multicell
from isaac_net.core.proto.netsim import Radio
from isaac_net.core.proto.rng import CounterRNG
from isaac_net.core.radio import RadioMC
from isaac_net.core.sharded import set_env_offset

R, EMAX, STEPS, SEED = 3, 6, 7, 5

CASES = {
    "L2": ("L2", NRConfig()),
    "L2-tr38901-o2i": ("L2", NRConfig(channel="tr38901", tr38901_scenario="UMi", o2i_indoor_frac=0.5,
                                      shadow_white_frac=0.0)),
    "L2-mc3": ("L2", multicell(3)),
    "L2-legacy-mc3": ("L2-legacy", multicell(3)),
}


def _inputs(steps=STEPS, seed=0):
    """Per-step (send [EMAX,R], poses [EMAX,R,2]); an engine with E envs uses the first E rows."""
    g = torch.Generator().manual_seed(seed)
    return [(torch.randint(0, 3, (EMAX, R), generator=g), torch.rand(EMAX, R, 2, generator=g) * 150)
            for _ in range(steps)]


def _drive(net, E, resets=(), steps=STEPS):
    """Per-step outputs (every tensor with leading dim E, cloned); resets = {step: env ids} after that step."""
    rs = dict(resets)
    outs = []
    for k, (send, pos) in enumerate(_inputs(steps)):
        net.submit(None, Requests(send[:E], None, torch.full((E,), k, dtype=torch.long)))
        o = net.step(None, pos[:E])
        outs.append({n: v.clone() for n, v in o.items() if torch.is_tensor(v) and v.dim() >= 1 and v.shape[0] == E})
        if k in rs:
            net.reset(torch.tensor(rs[k]))
    return outs


def _rows_equal(a, b, rows_a, rows_b=None, steps=None, float_tol=False):
    """Every output equal row by row. Integer and boolean outputs (draws, decisions) are always compared exactly;
    with float_tol the float outputs are compared to float32 rounding, which the levels with a link model
    (L2-legacy: log2 / exp in the rate and BLER) need on the CPU, where a kernel may round differently for a
    different tensor shape (as tests/test_sharded.py documents; seen on Linux x86 CI, not on macOS arm64)."""
    rows_b = rows_a if rows_b is None else rows_b
    n = 0
    for k, (x, y) in enumerate(zip(a, b)):
        if steps is not None and k not in steps:
            continue
        assert x.keys() == y.keys()
        for name in x:
            u, v = x[name][rows_a], y[name][rows_b]
            if u.is_floating_point():
                u, v = u.nan_to_num(-7.0), v.nan_to_num(-7.0)
                if float_tol and u.device.type == "cpu":
                    assert torch.allclose(u, v, rtol=1e-5, atol=1e-4), (k, name)
                    n += 1
                    continue
            assert torch.equal(u, v), (k, name)
            n += 1
    assert n > 0


def _tol(case):
    return CASES[case][0] == "L2-legacy"


def _make(case, E):
    level, cfg = CASES[case]
    return make_engine(level, E, R, "cpu", cfg, seed=SEED)


@pytest.mark.parametrize("case", list(CASES))
def test_env0_independent_of_E(case):
    resets = {3: [1]}
    a = _drive(_make(case, 3), 3, resets)
    b = _drive(_make(case, 6), 6, resets)
    _rows_equal(a, b, slice(0, 3), float_tol=_tol(case))
    assert a[0]["sinr_db"].abs().sum() > 0


@pytest.mark.parametrize("case", list(CASES))
def test_other_resets_do_not_shift_env0(case):
    a = _drive(_make(case, 4), 4, {2: [0]})
    b = _drive(_make(case, 4), 4, {0: [1], 1: [1, 2], 2: [0]})
    _rows_equal(a, b, 0, float_tol=_tol(case))            # every step: env 0's draws never see env 1 / 2's resets
    _rows_equal(a, b, 3, float_tol=_tol(case))


@pytest.mark.parametrize("case", list(CASES))
def test_partial_reset_leaves_others_untouched(case):
    a = _drive(_make(case, 4), 4)
    b = _drive(_make(case, 4), 4, {2: [1]})
    _rows_equal(a, b, [0, 2, 3], float_tol=_tol(case))
    # env 1 after its reset is a new episode: its channel differs from the run without the reset
    assert not torch.equal(a[4]["sinr_db"][1], b[4]["sinr_db"][1])


@pytest.mark.parametrize("case", list(CASES))
def test_two_shards_by_global_env_id_equal_one_engine(case):
    """What ShardedEngine does per shard (make_engine with the shared seed, then set_env_offset), checked directly:
    ShardedEngine itself still derives per-shard seeds for L2 and multi-cell L2-legacy (sharded.INVARIANT_LEVELS)."""
    E, split = 6, (2, 4)
    resets = {2: [1, 2, 4], 4: [0, 5]}
    un = _drive(_make(case, E), E, resets)
    parts, off = [], 0
    for e in split:
        eng = _make(case, e)
        set_env_offset(eng, off)
        loc = {k: [i - off for i in v if off <= i < off + e] for k, v in resets.items()}
        # each shard sees only its rows of the shared inputs
        rs = {k: v for k, v in loc.items() if v}
        outs = []
        for k, (send, pos) in enumerate(_inputs()):
            eng.submit(None, Requests(send[off:off + e], None, torch.full((e,), k, dtype=torch.long)))
            o = eng.step(None, pos[off:off + e])
            outs.append({n: v.clone() for n, v in o.items() if torch.is_tensor(v) and v.dim() >= 1 and v.shape[0] == e})
            if k in rs:
                eng.reset(torch.tensor(rs[k]))
        parts.append(outs)
        off += e
    cat = [{n: torch.cat([p[k][n] for p in parts]) for n in parts[0][k]} for k in range(STEPS)]
    _rows_equal(un, cat, slice(None))


def test_global_rng_keeps_the_generator_path():
    """rng="global": the radio draws from the engine generator (no counter RNG), as before."""
    net = make_engine("L2", 3, R, "cpu", multicell(3).with_(rng="global"), seed=SEED)
    _drive(net, 3, steps=1)
    assert net.radio.rng is None and net.radio.gen is net.gen


def test_c1_field_equals_prototype_radio_under_engine_rng():
    rng = CounterRNG(7, 5, "cpu")
    rng.set_env_offset(3)
    rng.reset(None)
    radio = Radio(5, "cpu", rng=rng)
    rmc = RadioMC(NRConfig(), 5, "cpu", rng=rng)
    pos = torch.rand(5, R, 2, generator=torch.Generator().manual_seed(1)) * 150
    assert torch.equal(rmc.rx_dbm(pos)[..., 0] - NRConfig().ni_fixed_dbm, radio.snr_db(pos))


def test_netslotmc_handover_times_in_its_own_slots():
    cfg = multicell(3, proto_ul_slots_per_step=20)          # 100 ms step, 5 ms legacy UL slots
    net = make_engine("L2-legacy", 2, R, "cpu", cfg, seed=0)
    assert net.K == 20
    assert (net.assoc.ttt, net.assoc.ho_int) == (60, 8)     # 300 ms TTT, 40 ms interruption
    assert (cfg.ttt_slots, cfg.ho_int_slots) == (60, 8)
    d = multicell(3)
    assert (d.ttt_slots, d.ho_int_slots) == (120, 16)       # default 2.5 ms slots: unchanged
    nr = make_engine("L2", 2, R, "cpu", cfg, seed=0)        # the NR engine counts in its own slot_ms
    assert nr.net.assoc.ttt == round(300 / cfg.slot_ms) and nr.net.assoc.ho_int == round(40 / cfg.slot_ms)


def test_reference_nr_engine_copies_caller_buffers():
    """NREngine keeps copies of the step's SNR and the submit's hid (as the fast backends do), so a caller that
    reuses its buffers gets the same frame features as one that passes fresh tensors."""
    E = 3
    runs = []
    for reuse in (False, True):
        net = make_engine("L2", E, R, "cpu", NRConfig(), seed=SEED)
        snr, hid = torch.zeros(E, R), torch.zeros(E, dtype=torch.long)
        g = torch.Generator().manual_seed(0)
        for k in range(4):
            new_snr, new_hid = 30 * torch.rand(E, R, generator=g), torch.full((E,), k, dtype=torch.long)
            if reuse:
                snr.copy_(new_snr)
                hid.copy_(new_hid)
            else:
                snr, hid = new_snr, new_hid
            net.submit(None, Requests(torch.ones(E, R, dtype=torch.long), None, hid))
            net.step(None, snr)
            if reuse:
                snr.fill_(-99.0)
                hid.fill_(-5)
        q = net.net.ul.q
        runs.append((q.f_snr.clone(), q.hid.clone()))
    assert torch.equal(runs[0][0], runs[1][0]) and torch.equal(runs[0][1], runs[1][1])
