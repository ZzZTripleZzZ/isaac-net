"""make_engine: one API for every fidelity level (dict step outputs, per-env clocks, partial resets, legacy calls)."""
import math

import pytest
import torch

from engine_api import default_params
from isaaclab_net.core import SIM_LEVELS, Requests, make_engine, netslot_compat
from isaaclab_net.core.proto import netsim

E, R = 4, 5
KEYS = {"delivered", "timed_out", "cap", "cls", "delay", "newest", "det_env", "queue_len", "queue_bytes", "sinr_db", "t"}


def _params(level):
    return default_params("L2" if level == "L2-legacy" else level)


def _engine(level, seed=0, config=None):
    cfg = config if config is not None else (netslot_compat() if level == "L2" else None)
    return make_engine(level, E, R, "cpu", cfg, params=_params(level), seed=seed)


def _drive(net, steps, gen, reset_at=None, ids=None, pos=False):
    outs = []
    for k in range(steps):
        if k == reset_at:
            net.reset(ids)
        send = (torch.rand(E, R, generator=gen) < 0.5).long() * torch.randint(1, 3, (E, R), generator=gen)
        x = torch.rand(E, R, 2, generator=gen) * 150 if pos else 5 + 20 * torch.rand(E, R, generator=gen)
        net.submit(None, Requests(send))
        outs.append(net.step(None, x))
    return outs


@pytest.mark.parametrize("level", SIM_LEVELS)          # the surrogate and bound levels: test_levels.py
def test_dict_outputs_and_clock(level, seeded):
    net = _engine(level)
    F = net.config.frame_buffer
    outs = _drive(net, 8, torch.Generator().manual_seed(1))
    o = outs[-1]
    assert KEYS <= set(o), KEYS - set(o)
    for k in ("delivered", "timed_out", "cap", "cls", "delay"):
        assert o[k].shape == (E, R, F), (k, o[k].shape)
    assert o["newest"].shape == (E, R) and o["det_env"].shape == (E,)
    assert (o["t"] == 7).all() and (net.clock == 8).all()
    assert all((x["newest"] <= x["t"][:, None]).all() for x in outs)
    d = o["delay"][o["delivered"]]
    assert torch.isfinite(d).all() and (d > 0).all()
    assert sum(int(x["delivered"].sum()) for x in outs) > 0


@pytest.mark.parametrize("level", ["L0", "L1", "L2", "L2-legacy"])
def test_partial_reset_zeroes_clock_and_queues(level, seeded):
    net = _engine(level)
    _drive(net, 6, torch.Generator().manual_seed(2))
    net.reset(torch.tensor([1, 3]))
    assert net.clock.tolist() == [6, 0, 6, 0]
    assert (net.queued()[[1, 3]] == 0).all()
    net.submit(None, Requests(torch.ones(E, R, dtype=torch.long)))
    o = net.step(None, torch.full((E, R), 20.0))
    assert o["t"].tolist() == [6, 0, 6, 0]
    assert (o["cap"][[1, 3]][o["cap"][[1, 3]] >= 0] == 0).all()     # reset envs capture at their clock 0


@pytest.mark.parametrize("level", ["L2", "L2-legacy"])
def test_partial_reset_leaves_other_envs_bitwise_unaffected(level):
    """Two engines, same seed and inputs; one resets envs [0, 2] mid-run. Envs 1 and 3 must stay identical."""
    runs = []
    for do_reset in (False, True):
        torch.manual_seed(5)
        net = _engine(level, seed=3)
        outs = _drive(net, 12, torch.Generator().manual_seed(4), reset_at=6 if do_reset else None,
                      ids=torch.tensor([0, 2]))
        runs.append(outs)
    keep = [1, 3]
    for a, b in zip(*runs):
        for k in ("delivered", "cap", "newest", "queue_len", "queue_bytes", "t"):
            assert torch.equal(a[k][keep], b[k][keep]), k
        assert torch.equal(a["delay"][keep].nan_to_num(-1), b["delay"][keep].nan_to_num(-1))


def test_l2_legacy_equals_prototype_netslot(seeded):
    """L2-legacy is the frozen prototype NetSlot: bitwise the same as netsim.make_net('L2')."""
    torch.manual_seed(7)
    a = make_engine("L2-legacy", E, R, "cpu", seed=11)
    torch.manual_seed(7)
    b = netsim.make_net("L2", E, R, "cpu", (4000.0, 30000.0), seed=11)
    assert type(a) is netsim.NetSlot
    g1, g2 = torch.Generator().manual_seed(1), torch.Generator().manual_seed(1)
    torch.manual_seed(8)
    oa = _drive(a, 10, g1)
    torch.manual_seed(8)
    ob = _drive(b, 10, g2)
    for x, y in zip(oa, ob):
        assert torch.equal(x["newest"], y["newest"]) and torch.equal(x["queue_bytes"], y["queue_bytes"])


def test_nr_legacy_calls_and_time_checks(seeded):
    net = _engine("L2")
    snr = torch.full((E, R), 15.0)
    z = torch.zeros(E, dtype=torch.long)
    for t in range(5):                                     # NetSlot-style loop with an explicit int clock
        net.add_frames(t, torch.ones(E, R, dtype=torch.long), torch.zeros(E, R, dtype=torch.bool), z, snr)
        newest, det = net.step(t, snr, z)
        assert newest.shape == (E, R) and det.shape == (E,)
    with pytest.raises(ValueError):
        net.step(9, snr)                                   # the NR engine cannot jump in time
    net.reset([0])
    with pytest.raises(ValueError):
        net.step(5, snr)                                   # clocks differ after a partial reset
    net.step(None, snr)
    net.step(net.clock.clone(), snr)                       # an [E] tensor equal to the clock is fine


def test_nr_poses_match_snr_input(seeded):
    """With the default single-cell config, poses go through RadioMC (gNB at the origin, fixed noise) and give
    the same SNR as the prototype Radio's snr_db on the same shadowing field."""
    from isaaclab_net.core.radio import Radio, RadioMC
    net = _engine("L2")
    pos = torch.rand(E, R, 2) * 150
    radio = Radio(E, "cpu", generator=torch.Generator().manual_seed(9))
    net.attach_radio(RadioMC.from_radio(radio, net.config, "cpu"))
    out = net.step(None, pos)
    assert torch.allclose(out["sinr_db"], radio.snr_db(pos), atol=1e-4)


def test_nr_downlink_outputs(seeded):
    net = make_engine("L2", E, R, "cpu", netslot_compat(dl=True), seed=0)
    for _ in range(6):
        net.submit(None, torch.ones(E, R, dtype=torch.long))
        net.add_dl_frames(None, torch.full((E, R), 3000.0))
        out = net.step(None, torch.full((E, R), 20.0))
    assert out["dl_newest"].shape == (E, R) and (out["dl_newest"] >= 0).any()
    assert out["serving_cell"].shape == (E, R) and (out["serving_cell"] == 0).all()


def test_nr_sinr_hook_is_called(seeded):
    net = _engine("L2")
    calls = []

    def hook(g, direction, won, n_prb, act):
        calls.append((direction, won.shape))
        return act - 3.0
    net.set_sinr_hook(hook)
    net.submit(None, torch.ones(E, R, dtype=torch.long))
    net.step(None, torch.full((E, R), 20.0))
    assert calls and calls[0] == ("ul", (E, R, net.config.n_subbands))


def test_seed_makes_resets_reproducible():
    a, b = _engine("L2", seed=21), _engine("L2", seed=21)
    a.reset([1]); b.reset([1])
    assert torch.equal(a.net.h, b.net.h)
    assert not math.isclose(float(_engine("L2", seed=22).net.h.sum()), float(a.net.h.sum()))
