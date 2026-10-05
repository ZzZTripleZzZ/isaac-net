"""Closed-loop UL TPC (ul_tpc), the 38.214 CQI table (cqi_table="38214") and the gNB sector antenna
(gnb_antenna="sector"), CPU.

  T1 all three off: the single-cell engine is bitwise the frozen engine of tests/nr_frozen/ with the features' other
     fields set (they are not read), and at three cells with poses the off switches equal the defaults bitwise
  T2 TPC accumulate: a 10 dB path-loss step is corrected in four commands (3, 3, 3, 1 dB) back to the target; the
     offset clamps at ul_tpc_range_db; a UE at full power accumulates no positive step; absolute mode jumps to the
     set value at once; the triton backend accepts ul_tpc and cqi_table="38214" (tests/test_triton_tpc_cqi.py; its
     equivalence is test_nr_fast.py G2 on the GPU)
  T3 38.214 CQI: 15 thresholds, non-decreasing, 16 levels over a sweep, CQI 0 below the first threshold, the CQI -> MCS
     mapping never exceeds the CQI's spectral efficiency and takes the highest such MCS (dl_mcs_max respected); the
     tables' efficiencies equal the printed 38.214 values; a DL run reports from the coarser grid
  T4 sector antenna: 8 dBi on the boresight, -3 dB at +-32.5 deg, the 30 dB floor, vertical tilt; three sectors at
     30 / 150 / 270 deg cover 360 deg (worst point -2.2 dBi, equal gains at the borders, 65 deg within 3 dB each);
     at a co-sited 3-sector site the SIR 20 deg from a sector border rises from 0 dB to above 10 dB; two neighbouring
     cells whose sectors point away from each other: the engine's UL and DL interference over thermal drop by more
     than 3 dB with the association unchanged
  T5 unused_fields / fields_read_by gate the new fields by their switches
  T6 with all three on at three cells: env 0 independent of E, other envs' resets and a partial reset (the
     test_radio_rng.py patterns); reset envs' TPC state returns to its initial value
"""
import math

import pytest
import torch

from isaac_net.core import NRConfig, Requests, make_engine, multicell
from isaac_net.core.channels.antenna import element_gain_db, gnb_antenna_gain_db
from isaac_net.core.config import fields_read_by
from isaac_net.core.mac_ul import TPC_STATE
from isaac_net.core.nr_engine import NRNet
from isaac_net.core.nr_fast import NRTritonEngine, TritonUnsupported
from isaac_net.core.phy import CQI_T1, CQI_T2, PHY, cqi_tables
from isaac_net.core.radio import RadioMC
from nr_frozen.nr_engine import NRNet as FrozenNRNet

SIZES = (4000.0, 30000.0)
OFF_FIELDS = dict(ul_tpc=False, ul_tpc_mode="absolute", ul_tpc_target_db=3.0, ul_tpc_steps_db=(-2.0, 2.0),
                  ul_tpc_delay_slots=7, ul_tpc_range_db=4.0, cqi_table="mcs", gnb_antenna="isotropic",
                  cell_tilt_deg=6.0, gnb_antenna_gain_dbi=5.0)


# ---------------------------------------------------------------- T1 off = before
def _run_single(cls, cfg, E=3, R=4, steps=8, reset_at=4):
    torch.manual_seed(11)
    net = cls(E, R, "cpu", SIZES, cfg, generator=torch.Generator().manual_seed(3))
    g = torch.Generator().manual_seed(5)
    outs = []
    for t in range(steps):
        if t == reset_at:
            net.reset(torch.tensor([1]))
        send = torch.randint(0, 3, (E, R), generator=g)
        snr = -5 + 30 * torch.rand(E, R, generator=g)
        hid = torch.randint(0, 3, (E,), generator=g)
        net.add_frames(t, send, torch.rand(E, R, generator=g) < 0.3, hid, snr)
        if cfg.dl:
            net.add_dl_frames(t, torch.where(send > 0, net.sizes[(send - 1).clamp(min=0)], torch.zeros_like(snr)))
        o = net.step(t, snr, hid, full=True) if t % 2 == 0 else net.step_rx(t, snr - 113.0, hid, full=True)
        outs.append({k: v.clone() for k, v in o.items()})
        for lk in (net.ul, net.dl):
            if lk is not None:
                outs[-1].update({f"{lk.dir}.{n}": getattr(lk, n).clone() for n in ("olla", "avg", "csi", "sent")})
    return outs


def _equal(a, b):
    for t, (x, y) in enumerate(zip(a, b)):
        assert x.keys() == y.keys()
        for k in x:
            assert torch.equal(x[k].nan_to_num(-7.0), y[k].nan_to_num(-7.0)), (t, k)


def test_t1_single_cell_off_equals_frozen_engine():
    cfg = NRConfig(dl=True, rng="global", control_step_ms=20.0, **OFF_FIELDS)    # the frozen engine: global RNG
    _equal(_run_single(FrozenNRNet, cfg), _run_single(NRNet, cfg))
    # single-cell open-loop PC postdates the frozen engine: the off fields against the live defaults
    pc = NRConfig(dl=True, ul_pc=True, control_step_ms=20.0)
    _equal(_run_single(NRNet, pc), _run_single(NRNet, pc.with_(**OFF_FIELDS)))


def _drive_poses(cfg, E=3, R=4, steps=5, resets=None, seed=4, emax=6):
    """Inputs drawn for emax envs (an engine with E envs uses the first E rows), resets = {step: env ids}."""
    eng = make_engine("L2", E, R, "cpu", cfg, seed=seed)
    g = torch.Generator().manual_seed(1)
    pos = torch.rand(emax, R, 2, generator=g) * 150
    outs = []
    for k in range(steps):
        send = torch.randint(0, 3, (emax, R), generator=g)
        eng.submit(None, Requests(send[:E], None, torch.full((E,), k)))
        pos = (pos + 3 * (2 * torch.rand(emax, R, 2, generator=g) - 1)).clamp(0, 150)
        o = eng.step(None, pos[:E])
        outs.append({n: v.clone() for n, v in o.items() if torch.is_tensor(v) and v.dim() >= 1 and v.shape[0] == E})
        if resets and k in resets:
            eng.reset(torch.tensor(resets[k]))
    return outs, eng


def test_t1_three_cells_off_equals_defaults():
    base = multicell(3, dl=True, control_step_ms=20.0)
    a, _ = _drive_poses(base)
    b, _ = _drive_poses(base.with_(cell_azimuth_deg=(0.0, 10.0, 20.0), **OFF_FIELDS))
    _equal(a, b)


# ---------------------------------------------------------------- T2 TPC
def _tpc_net(**kw):
    """One robot, one cell, no fading, open-loop PC with alpha = 0 (the transmit PSD ignores path loss, so a
    path-loss step shows up 1:1 in the measured SINR), 1 MB frames so every UL slot carries a PUSCH."""
    cfg = NRConfig(fading=False, ul_pc=True, ul_pc_alpha=0.0, ul_pc_p0_dbm=3.0, ul_tpc=True, ul_tpc_target_db=10.0,
                   msg_sizes=(1e6,), control_step_ms=20.0).with_(**kw)
    net = NRNet(1, 1, "cpu", cfg.msg_sizes, cfg, seed=0)
    net.t_next = 0
    trace = []
    issue = net.ul._tpc_issue

    def rec(g, tx):
        meas, lim = net.ul._tpc_meas
        if bool(tx[0, 0]):
            trace.append((float(meas[0, 0]), bool(lim[0, 0]), float(net.ul.tpc_f[0, 0])))
        issue(g, tx)
    net.ul._tpc_issue = rec
    return net, trace


def _run_tpc(net, snrs):
    """One step per input SNR (full UE power over snr_ref_prbs PRBs), a fresh 1 MB frame each step."""
    one = torch.ones(1, 1, dtype=torch.long)
    for s in snrs:
        t = net.t_next
        snr = torch.full((1, 1), float(s))
        net.add_frames(t, one, torch.zeros(1, 1, dtype=torch.bool), torch.zeros(1, dtype=torch.long), snr)
        net.step(t, snr)
        net.t_next = t + 1


def test_t2_accumulate_corrects_a_10db_step():
    # pc backoff = ue_tx - P0 = 20 dB above the full-power PSD over 10 PRBs; input 30 dB -> PUSCH SINR 10 dB = target
    net, tr = _tpc_net()
    _run_tpc(net, [30.0, 30.0])
    assert all(abs(m - 10.0) < 1e-4 for m, _, _ in tr) and float(net.ul.tpc_f) == 0.0
    n0 = len(tr)
    _run_tpc(net, [20.0])                         # path loss up by 10 dB
    meas = [m for m, _, _ in tr[n0:]]
    f = [x for _, _, x in tr[n0:]]
    # commands 3, 3, 3, 1 dB (nearest of -1, 0, 1, 3 to the error), one per PUSCH with k2 = 2 < the UL spacing
    assert f[:5] == [0.0, 3.0, 6.0, 9.0, 10.0]
    assert [round(x, 3) for x in meas[:5]] == [0.0, 3.0, 6.0, 9.0, 10.0]
    assert all(abs(m - 10.0) < 1e-4 for m in meas[4:]) and float(net.ul.tpc_f) == 10.0
    assert not any(lim for _, lim, _ in tr)


def test_t2_offset_clamped_and_no_windup_at_full_power():
    net, tr = _tpc_net(ul_tpc_range_db=4.0)
    _run_tpc(net, [20.0])
    assert max(x for _, _, x in tr) == 4.0 and float(net.ul.tpc_f) == 4.0      # 3, then 3 clamped to 4
    assert abs(tr[-1][0] - 4.0) < 1e-4
    # P0 = ue_tx: the PC PSD is the full-power PSD over 10 PRBs, so any grant above 10 PRBs is power-limited
    net, tr = _tpc_net(ul_pc_p0_dbm=23.0, ul_tpc_target_db=30.0)
    _run_tpc(net, [10.0])
    assert sum(lim for _, lim, _ in tr) >= 3
    for (_, lim, f0), (_, _, f1) in zip(tr, tr[1:]):
        assert not (lim and f1 > f0)              # the command after a full-power PUSCH never raises f


def test_t2_absolute_jumps_to_the_set_value():
    net, tr = _tpc_net(ul_tpc_mode="absolute")
    assert net.cfg.ul_tpc_set == (-4.0, -1.0, 1.0, 4.0)
    _run_tpc(net, [30.0, 20.0])
    f = [x for _, _, x in tr]
    # on target the set has no 0 dB entry: f + 0 is nearest to -1 and +1, and the tie goes to the first (-1)
    k = next(i for i, (m, _, _) in enumerate(tr) if m < 5.0)       # first PUSCH after the 10 dB step
    assert f[k] == -1.0 and abs(tr[k][0] + 1.0) < 1e-4
    assert f[k + 1] == 4.0 and all(x == 4.0 for x in f[k + 1:])    # one command straight to +4, no ramp
    net, tr = _tpc_net(ul_tpc_mode="absolute")
    _run_tpc(net, [33.0])                         # 3 dB above target: f = the step nearest -3, i.e. -4, at once
    assert [x for _, _, x in tr][:3] == [0.0, -4.0, -4.0] and float(net.ul.tpc_f) == -4.0


def test_t2_triton_accepts():
    for kw in (dict(ul_pc=True, ul_tpc=True), dict(dl=True, cqi_table="38214")):
        assert NRTritonEngine.refusals(NRConfig(**kw)) == []
        with pytest.raises(ValueError, match="CUDA") as ei:       # past the refusals: the CPU device stops it
            make_engine("L2", 2, 2, "cpu", NRConfig(**kw), "triton")
        assert not isinstance(ei.value, TritonUnsupported)
    with pytest.raises(AssertionError, match="ul_pc"):
        NRConfig(ul_tpc=True)


# ---------------------------------------------------------------- T3 CQI table
@pytest.mark.parametrize("table", [1, 2])
def test_t3_cqi_table_levels_and_mapping(table):
    phy = PHY("dl", table, "cpu")
    thr, cqi_mcs = cqi_tables(phy, table)
    assert thr.shape == (15,) and cqi_mcs.shape == (16,)
    assert bool((thr[1:] >= thr[:-1]).all())
    se_c = torch.tensor([0.0] + [q * r / 1024 for q, r in (CQI_T1 if table == 1 else CQI_T2)])
    for k in range(1, 16):
        m = int(cqi_mcs[k])
        ok = [j for j in range(phy.M) if float(phy.se[j]) <= float(se_c[k]) + 1e-6]
        assert m == (max(ok) if ok else 0)                     # the highest MCS not above the CQI's efficiency
        assert float(phy.se[m]) <= float(se_c[k]) + 1e-6 or k == 1
    assert int(cqi_mcs[0]) == 0
    sweep = torch.linspace(-15, 35, 2001)
    cqi = (sweep[:, None] >= thr).sum(-1)
    assert bool((cqi[1:] >= cqi[:-1]).all()) and set(cqi.tolist()) == set(range(16))
    assert int(cqi[sweep < thr[0]].max()) == 0 and int(cqi[sweep >= thr[-1]].min()) == 15
    capped = PHY("dl", table, "cpu")
    capped.mcs_max = 10
    assert int(cqi_tables(capped, table)[1].max()) == 10


def test_t3_cqi_efficiencies_match_38214_print():
    t1 = [0.1523, 0.2344, 0.3770, 0.6016, 0.8770, 1.1758, 1.4766, 1.9141, 2.4063, 2.7305, 3.3223, 3.9023, 4.5234,
          5.1152, 5.5547]
    t2 = [0.1523, 0.3770, 0.8770, 1.4766, 1.9141, 2.4063, 2.7305, 3.3223, 3.9023, 4.5234, 5.1152, 5.5547, 6.2266,
          6.9141, 7.4063]
    for tab, ref in ((CQI_T1, t1), (CQI_T2, t2)):
        assert [round(q * r / 1024 + 1e-9, 4) for q, r in tab] == ref


def test_t3_dl_report_on_the_cqi_grid():
    E, R = 2, 3
    cfg = NRConfig(dl=True, control_step_ms=20.0)
    a, b = NRNet(E, R, "cpu", SIZES, cfg, seed=1), NRNet(E, R, "cpu", SIZES, cfg.with_(cqi_table="38214"), seed=1)
    sinr = torch.linspace(-12, 30, E * R * cfg.n_subbands).view(E, R, -1)
    gain = torch.zeros_like(sinr)
    a.dl.cqi_report(sinr, gain)
    b.dl.cqi_report(sinr, gain)
    ea, eb = sinr + a.dl.csi, sinr + b.dl.csi                      # the gNB's SINR estimates after the report
    assert bool((eb <= ea + 1e-5).all()) and bool((eb < ea - 1e-3).any())     # coarser grid, never above
    allowed = b.dl.phy.thr_ref[b.dl._cqi[1]]
    assert bool(((eb[..., None] - allowed).abs() < 1e-4).any(-1).all())       # thresholds of CQI-mapped MCSs
    for net in (a, b):                                            # and the engine runs with it
        for t in range(3):
            net.add_dl_frames(t, torch.full((E, R), 3000.0))
            net.step(t, torch.full((E, R), 15.0))
    assert float(b.dl.ctr["tb_ok"]) > 0


# ---------------------------------------------------------------- T4 antenna
def test_t4_element_pattern():
    h = torch.tensor(90.0)
    gain = lambda phi, theta=h, tilt=0.0: float(element_gain_db(theta, torch.tensor(float(phi)), 8.0, tilt))
    assert gain(0.0) == 8.0
    assert abs(gain(32.5) - 5.0) < 1e-5 and abs(gain(-32.5) - 5.0) < 1e-5
    assert gain(180.0) == -22.0 and gain(120.0) == -22.0 and gain(-150.0) == -22.0 and gain(540.0) == -22.0
    assert gain(65.0) == 8.0 - 12.0
    assert abs(float(element_gain_db(torch.tensor(0.0), torch.tensor(0.0))) - (8.0 - 12 * (90 / 65) ** 2)) < 1e-4
    assert float(element_gain_db(torch.tensor(0.0), torch.tensor(90.0))) == -22.0         # A_max floor in 3-D
    assert abs(float(element_gain_db(torch.tensor(122.5), torch.tensor(0.0))) - 5.0) < 1e-5  # 32.5 deg below
    assert float(element_gain_db(torch.tensor(100.0), torch.tensor(0.0), 8.0, 10.0)) == 8.0   # 10 deg downtilt


def test_t4_three_sectors_cover_360():
    cfg = NRConfig(n_cells=3, cell_layout="custom", cell_positions_m=((0.0, 0.0),) * 3, gnb_antenna="sector")
    assert cfg.cell_azimuths() == [30.0, 150.0, 270.0]
    az = torch.arange(0.0, 360.0, 0.25)
    pos = torch.stack([torch.cos(torch.deg2rad(az)), torch.sin(torch.deg2rad(az))], -1)[None] * 50.0   # [1,N,2]
    gnb3 = torch.zeros(3, 3)
    g = gnb_antenna_gain_db(cfg, pos, gnb3, 0.0)[0]                                   # [N,3]
    best = g.max(-1).values
    worst = 8.0 - 12.0 * (60.0 / 65.0) ** 2
    assert abs(float(best.min()) - worst) < 1e-3                                       # at the three borders
    for border in (90.0, 210.0, 330.0):
        i = int((az - border).abs().argmin())
        top2 = g[i].topk(2).values
        assert abs(float(top2[0] - top2[1])) < 1e-4                                    # equal gains: the overlap
    for c, b in enumerate(cfg.cell_azimuths()):
        in3db = (g[:, c] >= 5.0 - 1e-5)
        assert abs(float(in3db.sum()) * 0.25 - 65.0) <= 0.5                            # the 65 deg HPBW per sector
    off_border = (torch.remainder(az - 30.0, 120.0) - 60.0).abs() > 0.5
    near = (torch.remainder(az - 30.0 + 60.0, 360.0) // 120.0).long()
    assert bool((g.argmax(-1) == near)[off_border].all())                             # nearest boresight serves


def _site(antenna):
    return multicell(3, cell_layout="custom", cell_positions_m=((75.0, 75.0),) * 3, shadow_sigma_db=0.0, dl=True,
                     gnb_antenna=antenna, control_step_ms=20.0)


def test_t4_sectors_cut_interference():
    pos = torch.tensor([[[75.0 + 40 * math.cos(math.radians(70.0)), 75.0 + 40 * math.sin(math.radians(70.0))]]])
    for antenna, lo, hi in (("isotropic", -1e-4, 1e-4), ("sector", 10.0, 20.0)):
        rx = RadioMC(_site(antenna), 1, "cpu", R=1).rx_dbm(pos)[0, 0]
        sir = float(rx.max() - rx.topk(2).values[1])
        assert lo <= sir <= hi, (antenna, sir)
    # engine: two neighbouring cells 30 m apart whose sectors point away from each other, three robots in front of
    # each (the same association with either antenna); each gNB sees the other cell's robots from behind
    iot = {}
    for antenna in ("isotropic", "sector"):
        cfg = multicell(2, cell_layout="custom", cell_positions_m=((60.0, 75.0), (90.0, 75.0)), shadow_sigma_db=0.0,
                        dl=True, gnb_antenna=antenna, cell_azimuth_deg=(180.0, 0.0), control_step_ms=20.0)
        E, R = 2, 6
        eng = make_engine("L2", E, R, "cpu", cfg, seed=3)
        g = torch.Generator().manual_seed(2)
        x = torch.cat([20 + 30 * torch.rand(E, 3, generator=g), 100 + 30 * torch.rand(E, 3, generator=g)], -1)
        pos = torch.stack([x, 65 + 20 * torch.rand(E, R, generator=g)], -1)
        for k in range(4):
            eng.submit(None, Requests(torch.full((E, R), 2, dtype=torch.long), None, torch.full((E,), k)))
            eng.add_dl_frames(None, torch.full((E, R), 20000.0))
            eng.step(None, pos)
        iot[antenna] = (float(eng.net.iot_db("ul").mean()), float(eng.net.iot_db("dl").mean()))
        assert torch.equal(eng.net.assoc.serv, torch.tensor([[0, 0, 0, 1, 1, 1]] * E))
    assert iot["sector"][0] < iot["isotropic"][0] - 3.0 and iot["sector"][1] < iot["isotropic"][1] - 3.0, iot


# ---------------------------------------------------------------- T5 field gating
def test_t5_unused_fields():
    off = NRConfig(ul_tpc_range_db=5.0, ul_tpc_mode="absolute", cqi_table="38214", cell_tilt_deg=3.0,
                   gnb_antenna_gain_dbi=6.0)
    assert {"ul_tpc_range_db", "ul_tpc_mode", "cqi_table", "cell_tilt_deg", "gnb_antenna_gain_dbi"} <= set(
        off.unused_fields("L2"))
    on = off.with_(ul_pc=True, ul_tpc=True, dl=True, gnb_antenna="sector")
    assert not {"ul_tpc", "ul_tpc_range_db", "ul_tpc_mode", "cqi_table", "cell_tilt_deg", "gnb_antenna",
                "gnb_antenna_gain_dbi"} & set(on.unused_fields("L2"))
    mc = multicell(3, ul_tpc=True, gnb_antenna="sector", cell_tilt_deg=2.0)   # NetSlotMC: radio yes, TPC no
    assert "ul_tpc" in mc.unused_fields("L2-legacy") and "cell_tilt_deg" not in mc.unused_fields("L2-legacy")
    every = fields_read_by("L2")
    assert {"ul_tpc", "ul_tpc_target_db", "cqi_table", "gnb_antenna", "cell_azimuth_deg"} <= every
    assert NRConfig().unused_fields("L2") == []


# ---------------------------------------------------------------- T6 E-independence and partial reset
ALL_ON = multicell(3, dl=True, ul_tpc=True, cqi_table="38214", gnb_antenna="sector", cell_tilt_deg=4.0,
                   control_step_ms=20.0)


def _rows(a, b, rows_a, rows_b=None):
    rows_b = rows_a if rows_b is None else rows_b
    for x, y in zip(a, b):
        for k in x:
            assert torch.equal(x[k][rows_a].nan_to_num(-7.0), y[k][rows_b].nan_to_num(-7.0)), k


def test_t6_env_independence_and_partial_reset():
    a, _ = _drive_poses(ALL_ON, E=3, resets={2: [1]})
    b, _ = _drive_poses(ALL_ON, E=4, resets={2: [1]})
    c, eng = _drive_poses(ALL_ON, E=3, resets={1: [2], 2: [1]})
    d, _ = _drive_poses(ALL_ON, E=3)
    _rows(a, b, slice(0, 3))                      # env rows do not depend on E
    _rows(a, c, 0)                                # other envs' resets leave env 0 alone
    _rows(a[:2], d[:2], slice(None))
    _rows(a, d, [0, 2])                           # a partial reset leaves the other envs bitwise unaffected
    ul = eng.net.ul
    assert float(ul.tpc_f.abs().sum()) > 0
    eng.reset(torch.tensor([0]))
    for n, (_, _, v) in TPC_STATE.items():
        x = getattr(ul, n)[0]
        assert torch.equal(x.nan_to_num(-7.0), torch.full_like(x, v).nan_to_num(-7.0)), n


@pytest.mark.gpu
def test_t6_graph_backend_bitwise_gpu():
    outs = []
    for backend in ("reference", "graph"):
        eng = make_engine("L2", 3, 4, "cuda", ALL_ON, backend, seed=4)
        g = torch.Generator().manual_seed(1)
        pos = (torch.rand(3, 4, 2, generator=g) * 150).cuda()
        res = []
        for k in range(5):
            eng.submit(None, Requests(torch.randint(0, 3, (3, 4), generator=g).cuda(), None,
                                      torch.full((3,), k, device="cuda")))
            o = eng.step(None, pos)
            res.append({n: v.clone() for n, v in o.items() if torch.is_tensor(v)})
            if k == 2:
                eng.reset(torch.tensor([1], device="cuda"))
        outs.append(res)
    _equal(*outs)
