"""Radio energy model (core/energy.py).

Deterministic accounting (L2: per-slot transmit power from the MAC tap, which leaves the engine bitwise unchanged
and counts every transport block; other levels: the documented airtime approximation), battery and low-battery
flag, partial reset, power control lowering the transmit energy, DL receive slots, background + energy together, and
(gpu) graph safety around the L2-legacy graph backend: EnergyLoop(graph=True) is bitwise equal to the eager energy
step on the reference engine, without host syncs.
"""
import math

import pytest
import torch

from isaac_net.core import NRConfig, Requests, make_engine, multicell
from isaac_net.core.background import BackgroundConfig
from isaac_net.core.energy import EnergyConfig, EnergyLoop, legacy_airtime_slots

E, R = 4, 3
P23 = 10 ** ((23.0 - 30.0) / 10.0)


def _drive(net, steps=10, seed=3, reset_at=None, reset_ids=(1,), dev="cpu", pos=False):
    g = torch.Generator().manual_seed(seed)
    outs = []
    for k in range(steps):
        send = torch.randint(0, 3, (E, R), generator=g).to(dev)
        x = (torch.rand(E, R, 2, generator=g) * 100 if pos else 25 * torch.rand(E, R, generator=g) - 5).to(dev)
        net.submit(None, Requests(send))
        outs.append(net.step(None, x))
        if reset_at is not None and k == reset_at:
            net.reset(torch.tensor(reset_ids, device=dev))
    return outs


def _same(a, b, keys):
    return all(torch.equal(x[k].nan_to_num(-1.0) if x[k].is_floating_point() else x[k],
                           y[k].nan_to_num(-1.0) if y[k].is_floating_point() else y[k])
               for x, y in zip(a, b) for k in keys)


def test_l2_tap_leaves_the_engine_bitwise_unchanged():
    torch.manual_seed(0)
    a = _drive(make_engine("L2", E, R, "cpu", NRConfig(energy=EnergyConfig()), seed=1))
    torch.manual_seed(0)
    b = _drive(make_engine("L2", E, R, "cpu", NRConfig(), seed=1))
    assert _same(a, b, b[0].keys())
    assert "energy_j" in a[0] and "energy_j" not in b[0]


def test_l2_deterministic_accounting():
    ecfg = EnergyConfig(idle_power_w=0.05, msg_energy_j=0.002, pa_efficiency=0.5, tx_circuit_w=0.3)
    net = make_engine("L2", E, R, "cpu", NRConfig(energy=ecfg), seed=2)
    torch.manual_seed(1)
    g = torch.Generator().manual_seed(4)
    slot_s = NRConfig().slot_ms * 1e-3
    cum = torch.zeros(E, R)
    for k in range(12):
        send = torch.randint(0, 3, (E, R), generator=g)
        tb0 = float(net.ul.ctr["tb_new"] + net.ul.ctr["tb_retx"])
        acc = net.submit(None, Requests(send))
        o = net.step(None, 25 * torch.rand(E, R, generator=g) - 5)
        tb1 = float(net.ul.ctr["tb_new"] + net.ul.ctr["tb_retx"])
        assert float(o["tx_slots"].sum()) == tb1 - tb0             # every transport block counted once
        tx_s = o["tx_slots"] * slot_s * 12 / 14                    # PUSCH time: 12 data symbols per U slot
        rad = tx_s * P23                                            # allocated power, no power control: 23 dBm
        e_tx = rad / 0.5 + 0.3 * tx_s
        assert torch.allclose(o["energy_tx_j"], e_tx, rtol=1e-5, atol=1e-12)
        e = e_tx + 0.05 * 0.1 + 0.002 * acc.float()
        assert torch.allclose(o["energy_j"], e, rtol=1e-5)
        cum += o["energy_j"]
        assert torch.allclose(o["energy_cum_j"], cum, rtol=1e-5)
        assert torch.allclose(o["battery_j"], 3600.0 - cum, rtol=1e-6)
    assert float(o["tx_slots"].sum()) > 0


def test_l2_fixed_tx_power_and_power_control():
    net = make_engine("L2", E, R, "cpu", NRConfig(energy=EnergyConfig(tx_power_dbm=10.0)), seed=2)
    torch.manual_seed(0)
    o = _drive(net, steps=3)[-1]
    assert torch.allclose(o["energy_tx_j"], o["tx_slots"] * 0.01 * 5e-4 * 12 / 14, rtol=1e-5)   # 12 PUSCH symbols
    pc = make_engine("L2", E, R, "cpu", multicell(3, energy=EnergyConfig()), seed=2)
    torch.manual_seed(0)
    slot_s = pc.config.slot_ms * 1e-3
    tot_tx = tot_max = 0.0
    for o in _drive(pc, steps=6, pos=True):
        tot_tx += float(o["energy_tx_j"].sum())
        tot_max += float(o["tx_slots"].sum()) * P23 * slot_s
    assert 0 < tot_tx < tot_max            # fractional power control transmits below 23 dBm


def test_l2_downlink_receive_slots():
    cfg = NRConfig(dl=True, energy=EnergyConfig(rx_power_w=0.2, idle_power_w=0.0, msg_energy_j=0.0))
    net = make_engine("L2", E, R, "cpu", cfg, seed=0)
    torch.manual_seed(0)
    for _ in range(3):
        net.add_dl_frames(None, torch.full((E, R), 5000.0))
        o = net.step(None, torch.full((E, R), 15.0))
    assert (o["rx_slots"] > 0).all()
    assert torch.allclose(o["energy_j"], 0.2 * o["rx_slots"] * cfg.slot_ms * 1e-3, rtol=1e-5)


@pytest.mark.parametrize("level", ["L0", "L1", "L2-legacy", "ORACLE"])
def test_airtime_approximation_accounting(level):
    net = make_engine(level, E, R, "cpu", NRConfig(energy=EnergyConfig(msg_energy_j=0.0, idle_power_w=0.0)), seed=1)
    torch.manual_seed(0)
    sizes = torch.tensor([0.0, 4000.0, 30000.0])
    for o in _drive(net, steps=8):
        nb = (sizes[o["cls"]] * o["delivered"]).sum(-1)
        slots = legacy_airtime_slots(nb, o["sinr_db"], 0.1)
        assert torch.allclose(o["tx_slots"], slots)
        assert torch.allclose(o["energy_j"], slots * P23 * 5e-4, rtol=1e-6)


def test_battery_flag_and_empty():
    cfg = NRConfig(energy=EnergyConfig(battery_j=0.05, idle_power_w=0.02, msg_energy_j=0.0, low_battery_frac=0.5))
    net = make_engine("L0", E, R, "cpu", cfg, seed=0)
    frac, low, empty = [], [], []
    for _ in range(30):
        o = net.step(None, torch.zeros(E, R))
        frac.append(float(o["battery_frac"][0, 0]))
        low.append(bool(o["low_battery"][0, 0]))
        empty.append(bool(o["battery_empty"][0, 0]))
    assert frac[0] == pytest.approx(1 - 0.002 / 0.05) and frac[-1] == 0.0
    assert not low[0] and low[15] and empty[-1] and not empty[0]
    obs = net.energy_obs()
    assert obs.shape == (E, R, 2) and float(obs[0, 0, 0]) == 0.0 and float(obs[0, 0, 1]) == 1.0


def test_partial_reset():
    cfg = NRConfig(energy=EnergyConfig(battery_j=10.0, initial_soc=(0.4, 0.9)))
    a_net = make_engine("L1", E, R, "cpu", cfg, seed=9)
    b_net = make_engine("L1", E, R, "cpu", cfg, seed=9)
    b0 = a_net.energy_obs()[..., 0] * 10.0
    assert (b0 >= 4.0).all() and (b0 <= 9.0).all() and b0.unique().numel() == E * R
    a = _drive(a_net, steps=8, reset_at=3, reset_ids=(2,))
    b = _drive(b_net, steps=8)
    keep = torch.tensor([0, 1, 3])
    keys = ["energy_j", "energy_cum_j", "battery_j", "tx_slots"]
    assert all(torch.equal(x[k][keep], y[k][keep]) for x, y in zip(a, b) for k in keys)
    # env 2 restarted at step 4: its cumulative energy covers only steps 4..7 and its battery was redrawn
    cum = sum(o["energy_j"][2] for o in a[4:])
    assert torch.allclose(a[-1]["energy_cum_j"][2], cum)
    assert not torch.equal(a[-1]["battery_j"][2], b[-1]["battery_j"][2])


def test_deterministic_rerun():
    cfg = NRConfig(energy=EnergyConfig(initial_soc=(0.2, 1.0)))
    runs = []
    for _ in range(2):
        torch.manual_seed(0)
        runs.append(_drive(make_engine("L2", E, R, "cpu", cfg, seed=4), steps=5))
    assert _same(runs[0], runs[1], ["energy_j", "battery_j", "tx_slots"])


def test_background_and_energy_together():
    cfg = NRConfig(background=BackgroundConfig(n_background=4), energy=EnergyConfig())
    net = make_engine("L2", E, R, "cpu", cfg, seed=0)
    assert isinstance(net, EnergyLoop) and net.config.background is not None
    torch.manual_seed(0)
    o = _drive(net, steps=4, pos=True)[-1]
    assert o["energy_j"].shape == (E, R) and o["bg_util"].shape == (E, 1)
    assert math.isfinite(float(o["energy_cum_j"].sum()))


@pytest.mark.gpu
def test_graph_safety_l2_legacy(cuda):
    ecfg = EnergyConfig(msg_energy_j=0.003, initial_soc=(0.5, 1.0))
    ref = EnergyLoop(make_engine("L2-legacy", E, R, cuda, NRConfig(), backend="reference", seed=5), ecfg, seed=1)
    gr_eng = make_engine("L2-legacy", E, R, cuda, NRConfig(), backend="graph", seed=5)
    gr = EnergyLoop(gr_eng, ecfg, graph=True, seed=1)
    eg = EnergyLoop(make_engine("L2-legacy", E, R, cuda, NRConfig(), backend="graph", seed=5), ecfg, seed=1)
    a = _drive(ref, steps=12, reset_at=5, reset_ids=(0, 2), dev=cuda, pos=True)
    b = _drive(gr, steps=12, reset_at=5, reset_ids=(0, 2), dev=cuda, pos=True)
    keys = list(EnergyLoop.OUT_KEYS) + ["delivered", "delay"]
    assert _same(a, b, keys)
    # the eager energy step on the graph backend runs without host syncs
    g = torch.Generator().manual_seed(0)
    for k in range(4):
        eg.submit(None, torch.randint(0, 3, (E, R), generator=g).to(cuda))
        out = eg.engine.step(None, (torch.rand(E, R, 2, generator=g) * 100).to(cuda))
        torch.cuda.synchronize()
        torch.cuda.set_sync_debug_mode("error")
        try:
            eg.process(out)
        finally:
            torch.cuda.set_sync_debug_mode("default")
