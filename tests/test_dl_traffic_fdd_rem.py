"""DL traffic models, DL background load, FDD and the REM export tool (CPU).

(a) with the new fields at their defaults the engine is bitwise the pre-feature engine (golden digests recorded on the
    commit before the features, fixtures/dl_off_golden.json), and adding a DL model leaves the UL models' arrivals and
    the UL outputs unchanged; (b) a DL periodic model delivers its frames at the expected rate and sizes with
    consistent counters, never before their arrival slot, and stays apart from an edge loop's DL commands;
(c) DL background (ghost DL traffic, dl_load_frac) lowers a robot's DL throughput, with DL byte conservation;
(d) FDD: all-U UL / all-D DL carriers give every slot to both directions, DL CQI / HARQ-ACK timings follow, a DL + UL
    run conserves frames and bytes, the paired DL carrier's PRBs reach the DL MAC; (e) REM: grid shape, serving cell =
    argmax RSRP, determinism, npz round trip; (f) unused_fields / strict honesty for the new fields.
"""
import json
import math
import os

import numpy as np
import pytest
import torch

import dl_off_scenarios as golden
from isaac_net.core import EdgeConfig, NRConfig, Requests, TrafficModel as TM, make_engine, multicell
from isaac_net.core.background import BackgroundConfig
from isaac_net.tools.rem import compute_rem, load_rem, save_rem

E, R = 2, 3
STEP = 20.0                      # ms: 40 slots at mu = 1 keeps the DL runs short


def _snr(v=15.0):
    return torch.full((E, R), v)


# ---------------------------------------------------------------------------------------------- (a) bitwise off
def test_defaults_are_bitwise_the_pre_feature_engine():
    with open(golden.GOLDEN) as fh:
        g = json.load(fh)
    if g["env"] != golden.env_key():
        pytest.skip(f"golden digests recorded on {g['env']}, this is {golden.env_key()} (bitwise is per platform)")
    got = golden.compute()
    assert got == g["digests"], {k: got[k] == v for k, v in g["digests"].items()}


def test_explicit_defaults_equal_omitted_fields():
    a, b = NRConfig(), NRConfig(duplex="tdd", dl_n_prb=None, dl_bandwidth_mhz=None)
    assert a == b and a.unused_fields("L2") == b.unused_fields("L2") == []
    assert BackgroundConfig(n_background=2) == BackgroundConfig(n_background=2, dl_traffic=(), dl_load_frac=0.0)
    assert TM.periodic(100, 10) == TM.periodic(100, 10, direction="ul")


def _ul_run(cfg, steps=8):
    net = make_engine("L2", E, R, "cpu", cfg, seed=3)
    gen = torch.Generator().manual_seed(1)
    outs = []
    for _ in range(steps):
        net.submit(None, Requests((torch.rand(E, R, generator=gen) < 0.4).long()))
        outs.append(net.step(None, 5 + 15 * torch.rand(E, R, generator=gen)))
    return outs


def test_dl_model_does_not_shift_ul_models():
    ul = [TM.periodic(700, 4, jitter_ms=1).on([0, 1]), TM.bursty(1400, 60, 2, (0.3, 0.3)).on(2)]
    a = _ul_run(NRConfig(dl=True, control_step_ms=STEP, traffic=ul))
    b = _ul_run(NRConfig(dl=True, control_step_ms=STEP, traffic=ul + [TM.periodic(2500, 5).downlink()]))
    for oa, ob in zip(a, b):
        for k in oa:                                   # every UL key; the DL keys are new in b
            if k.startswith("dl_") and k not in ("dl_newest", "dl_queue_len"):
                continue
            if k in ("dl_newest", "dl_queue_len"):
                continue                               # the DL queue now carries the generated frames
            assert torch.equal(oa[k].nan_to_num(-1), ob[k].nan_to_num(-1)), k


# ---------------------------------------------------------------------------------------------- (b) DL generators
def test_dl_periodic_rate_sizes_and_counters():
    cfg = NRConfig(dl=True, control_step_ms=STEP, frame_buffer=32,
                   traffic=[TM.periodic(1000, 5, phase="aligned").downlink().on([0, 1]), TM.periodic(500, 10).on(2)])
    net = make_engine("L2", E, R, "cpu", cfg, seed=1)
    n_gen = n_dlv = n_lost = 0
    b_dlv = 0
    delays = []
    for k in range(12):
        o = net.step(None, _snr())
        assert torch.equal(o["gen_dl_accepted"], torch.tensor([[4, 4, 0]] * E))      # 20 ms / 5 ms on robots 0, 1
        assert torch.equal(o["gen_dl_bytes"], 1000 * o["gen_dl_accepted"])
        gen = o["dl_generated"]
        assert torch.equal(o["dl_bytes"][gen], torch.full_like(o["dl_bytes"][gen], 1000))
        assert (o["dl_tag"][gen] == 1).all() and not gen[:, 2].any()
        n_gen += int(o["gen_dl_accepted"].sum())
        n_dlv += int((o["dl_delivered"] & gen).sum())
        n_lost += int((o["dl_lost"] & gen).sum())
        b_dlv += int((o["dl_bytes"] * (o["dl_delivered"] & gen)).sum())
        delays.append(o["dl_delay"][o["dl_delivered"] & gen])
    q = net.net.dl.q
    queued = int((q.cap >= 0).sum())
    st = net.traffic_stats_dl
    assert n_gen == 12 * E * 2 * 4 == int(st["generated"]) == int(st["accepted"]) and int(st["refused"]) == 0
    assert int(st["accepted_bytes"]) == 1000 * n_gen
    assert n_gen == n_dlv + n_lost + queued and b_dlv == 1000 * n_dlv
    d = torch.cat(delays) * STEP
    assert d.numel() > 0 and float(d.min()) >= cfg.slot_ms - 1e-6          # never before its arrival slot
    # UL model on robot 2 untouched: 2 messages per step of 500 B
    assert int(o["gen_bytes"][:, 2].sum()) == E * 2 * 500


def test_dl_models_need_downlink_and_not_triton():
    dl_model = [TM.periodic(1000, 5).downlink()]
    with pytest.raises(ValueError, match="dl=True"):
        make_engine("L2", E, R, "cpu", NRConfig(traffic=dl_model), seed=0)
    with pytest.raises(ValueError, match="DL traffic models"):      # graph runs them (tests/test_limits_closed.py)
        make_engine("L2", E, R, "cpu", NRConfig(dl=True, traffic=dl_model), backend="triton", seed=0)
    with pytest.raises(ValueError, match="ignores NRConfig.traffic"):
        make_engine("L1", E, R, "cpu", NRConfig(traffic=dl_model), seed=0)
    with pytest.raises(ValueError):
        NRConfig(traffic=[TM.policy().downlink()])
    with pytest.raises(ValueError):
        NRConfig(traffic=[TM.event(100, det=True).downlink()])


def test_generated_dl_frames_are_not_edge_commands():
    """EdgeLoop's nr_dl return path shares the DL queue: commands carry cls = cap + 1 >= 1, generated frames cls < 0."""
    cfg = NRConfig(dl=True, control_step_ms=STEP, frame_buffer=32, edge=EdgeConfig(return_path="nr_dl"),
                   traffic=[TM.periodic(800, 5).downlink()])
    net = make_engine("L2", E, R, "cpu", cfg, seed=2)
    ref = make_engine("L2", E, R, "cpu", cfg.with_(traffic=None), seed=2)
    acts = acts_ref = 0
    for _ in range(12):
        for n in (net, ref):
            n.submit(None, Requests(torch.ones(E, R, dtype=torch.long)))
        o, r = net.step(None, _snr()), ref.step(None, _snr())
        acts += int(o["act_new"].sum())
        acts_ref += int(r["act_new"].sum())
        q = net.engine.net.dl.q
        live = q.cap >= 0
        assert ((q.cls[live] < 0) | (q.cls[live] >= 1)).all()
        assert int(o["cmd_dropped"].sum()) == 0
    assert acts > 0 and acts_ref > 0 and abs(acts - acts_ref) <= acts_ref // 2


# ---------------------------------------------------------------------------------------------- (c) DL background
def _robot_dl(bg, steps=10, dl=True):
    cfg = NRConfig(dl=dl, control_step_ms=STEP, frame_buffer=64, background=bg,
                   traffic=[TM.periodic(6000, 2, phase="aligned").downlink()])
    net = make_engine("L2", E, R, "cpu", cfg, seed=4)
    g = torch.Generator().manual_seed(3)
    b = 0
    outs = []
    for _ in range(steps):
        o = net.step(None, torch.rand(E, R, 2, generator=g) * 60)
        b += int((o["dl_bytes"] * o["dl_delivered"]).sum())
        outs.append(o)
    return b, outs, net


def test_dl_background_reduces_robot_dl_throughput():
    ghost_bg = BackgroundConfig(n_background=4, dl_traffic=(TM.periodic(8000, 2),))
    base, _, _ = _robot_dl(BackgroundConfig(n_background=4))
    loaded, outs, net = _robot_dl(ghost_bg)
    frac, _, _ = _robot_dl(BackgroundConfig(n_background=0, dl_load_frac=0.6))
    assert loaded < 0.8 * base and frac < 0.8 * base, (base, loaded, frac)
    # ghost DL byte conservation: offered = delivered + lost + queued, and the ghosts used DL PRBs
    off = sum(float(o["bg_dl_offered_bytes"].sum()) for o in outs)
    dlv = sum(float(o["bg_dl_delivered_bytes"].sum()) for o in outs)
    lost = sum(float(o["bg_dl_lost_bytes"].sum()) for o in outs)
    assert off > 0 and off == dlv + lost + float(outs[-1]["bg_dl_queue_bytes"].sum())
    util = torch.stack([o["bg_dl_util"] for o in outs])
    assert (util > 0).any() and (util <= 1 + 1e-6).all()
    assert outs[-1]["dl_delivered"].shape[1] == R                                 # ghost rows sliced off


def test_dl_background_validation():
    with pytest.raises(ValueError, match="no downlink"):
        make_engine("L1", E, R, "cpu", NRConfig(background=BackgroundConfig(n_background=2, dl_load_frac=0.3)))
    with pytest.raises(ValueError, match="dl=True"):
        make_engine("L2", E, R, "cpu", NRConfig(background=BackgroundConfig(n_background=2, dl_load_frac=0.3)))
    with pytest.raises(ValueError):
        BackgroundConfig(n_background=2, dl_load_frac=1.0)
    with pytest.raises(ValueError):
        BackgroundConfig(n_background=2, dl_traffic=(TM.event(100),))
    assert BackgroundConfig(dl_traffic=(TM.periodic(100, 10),)).dl_traffic[0].direction == "dl"


# ---------------------------------------------------------------------------------------------- (d) FDD
def test_fdd_pattern_helpers():
    c = NRConfig(duplex="fdd", dl=True, control_step_ms=STEP)
    N = c.slots_per_step
    assert c.ul_slots_per_step == N and c.dl_slots_per_step == N
    assert all(c.slot_symbols(p) == (14 - c.dl_ctrl_symbols, c.ul_data_symbols) for p in range(10))
    assert all(c.next_ul_capable(g) == g for g in range(10))
    net = make_engine("L2", E, R, "cpu", c, seed=0)
    sched = net.net._schedule(0)
    assert [s[0] for s in sched] == list(range(N))                     # every slot is active
    assert all(ack == rel + c.k1 for rel, _, _, _, _, ack in sched)     # DL HARQ-ACK exactly K1 later
    cqi = [rel for rel, _, _, _, q, _ in sched if q]
    sr = [rel for rel, _, _, s, _, _ in sched if s]
    assert cqi == list(range(0, N, c.cqi_period_slots)) and sr == list(range(0, N, c.sr_period_slots))
    t = NRConfig(dl=True, control_step_ms=STEP)
    assert t.ul_slots_per_step == N // 5 and t.slot_symbols(4) == (0, t.ul_data_symbols)


def test_fdd_dl_ul_run_conserves():
    cfg = NRConfig(duplex="fdd", dl=True, dl_bandwidth_mhz=40, control_step_ms=STEP, frame_buffer=32,
                   traffic=[TM.periodic(3000, 2).downlink(), TM.periodic(2000, 2)])
    assert cfg.dl_nprb == 106 and sum(cfg.dl_subband_prbs) == 106 and len(cfg.dl_subband_prbs) == cfg.n_subbands
    net = make_engine("L2", E, R, "cpu", cfg, seed=5)
    assert net.net.dl.sb_prb.tolist() == [float(x) for x in cfg.dl_subband_prbs]
    assert net.net.ul.sb_prb.tolist() == [float(x) for x in cfg.subband_prbs]
    g = torch.Generator().manual_seed(0)
    ul_out = dl_out = 0
    for _ in range(10):
        o = net.step(None, torch.rand(E, R, 2, generator=g) * 60)
        ul_out += int((o["bytes"] * (o["delivered"] | o["timed_out"] | o["dropped"])).sum())
        dl_out += int((o["dl_bytes"] * (o["dl_delivered"] | o["dl_lost"])).sum())
    for q, st, out in ((net.net.ul.q, net.traffic_stats, ul_out), (net.net.dl.q, net.traffic_stats_dl, dl_out)):
        queued = int(((q.end - q.start) * (q.cap >= 0)).sum())
        assert int(st["accepted_bytes"]) == out + queued and out > 0
    c = net.counters()
    assert c["dl"]["prb_avail"] == pytest.approx(106 * E * 10 * cfg.slots_per_step)


def test_fdd_wider_dl_carrier_carries_more():
    def run(**kw):
        cfg = NRConfig(duplex="fdd", dl=True, control_step_ms=STEP, frame_buffer=64,
                       traffic=[TM.periodic(9000, 1, phase="aligned").downlink()], **kw)
        net = make_engine("L2", E, R, "cpu", cfg, seed=6)
        b = 0
        for _ in range(8):
            o = net.step(None, _snr(25.0))
            b += int((o["dl_bytes"] * o["dl_delivered"]).sum())
        return b
    assert run(dl_bandwidth_mhz=50) > 1.3 * run()


def test_fdd_validation_and_backends():
    with pytest.raises(ValueError, match="duplex"):
        NRConfig(duplex="hd")
    with pytest.raises(ValueError, match="fdd"):
        NRConfig(dl_n_prb=100)
    with pytest.raises(ValueError, match="per_period"):
        NRConfig(duplex="fdd", proactive_grant="per_period")
    with pytest.raises(ValueError, match="triton"):
        make_engine("L2", E, R, "cpu", NRConfig(duplex="fdd"), backend="triton", seed=0)
    assert "FDD" in NRConfig(duplex="fdd", dl=True).summary()


def test_fdd_multicell_runs():
    cfg = multicell(3, duplex="fdd", dl=True, dl_n_prb=100, control_step_ms=10.0)
    net = make_engine("L2", E, R, "cpu", cfg, seed=7)
    assert net.net.dl_psd_db == pytest.approx(cfg.gnb_tx_dbm - 10 * math.log10(100))
    g = torch.Generator().manual_seed(0)
    for _ in range(3):
        net.add_dl_frames(None, torch.full((E, R), 2000.0))
        o = net.step(None, torch.rand(E, R, 2, generator=g) * 150)
    assert torch.isfinite(o["sinr_db"]).all()


# ---------------------------------------------------------------------------------------------- (e) REM
@pytest.mark.parametrize("cfg", [multicell(3), multicell(2, channel="tr38901_umi")], ids=["logd", "tr38901"])
def test_rem_shapes_serving_and_determinism(cfg, tmp_path):
    a = compute_rem(cfg, resolution_m=10.0, seed=3)
    C, H, W = a["rsrp_dbm"].shape
    assert (C, H, W) == (cfg.n_cells, 15, 15) and a["sinr_db"].shape == (H, W) and a["x"].shape == (W,)
    assert np.array_equal(a["serving"], a["rsrp_dbm"].argmax(0))
    assert np.isfinite(a["sinr_db"]).all()
    b = compute_rem(cfg, resolution_m=10.0, seed=3)
    assert all(np.array_equal(a[k], b[k]) for k in a)
    c = compute_rem(cfg, resolution_m=10.0, seed=4)
    assert not np.array_equal(a["pathgain_db"], c["pathgain_db"])
    if cfg.channel == "tr38901":
        assert a["los"].shape == (C, H, W) and a["los"].dtype == bool
    p = save_rem(a, os.path.join(tmp_path, "rem.npz"))
    r = load_rem(p)
    assert set(r) == set(a) and all(np.array_equal(r[k], a[k]) for k in a if k != "meta")
    assert json.loads(r["meta"])["seed"] == 3


def test_rem_shape_bounds_and_cli(tmp_path):
    rem = compute_rem(NRConfig(), bounds=(-20, -10, 20, 10), shape=(4, 8))
    assert rem["rsrp_dbm"].shape == (1, 4, 8) and (rem["serving"] == 0).all()
    assert rem["x"][0] == pytest.approx(-17.5) and rem["y"][-1] == pytest.approx(7.5)
    from isaac_net.tools.rem import main
    out = os.path.join(tmp_path, "cli.npz")
    assert main(["--out", out, "--res", "25", "--set", "n_cells=2", "cell_layout=grid"]) == 0
    assert load_rem(out)["rsrp_dbm"].shape == (2, 6, 6)


# ---------------------------------------------------------------------------------------------- (f) unused_fields
def test_unused_fields_honesty():
    fdd = NRConfig(duplex="fdd", dl=True, dl_n_prb=80)
    assert fdd.unused_fields("L2") == []
    assert set(fdd.unused_fields("L1")) == {"duplex", "dl_n_prb", "dl"}
    assert "duplex" in fdd.unused_fields("L2-legacy")
    assert NRConfig(duplex="fdd", tdd_pattern="DSUUU", dl=True).unused_fields("L2") == ["tdd_pattern"]
    assert NRConfig(duplex="fdd", dl_bandwidth_mhz=40).unused_fields("L2") == ["dl_bandwidth_mhz"]   # dl=False
    with pytest.raises(ValueError, match="dl_bandwidth_mhz"):
        make_engine("L2", E, R, "cpu", NRConfig(duplex="fdd", dl_bandwidth_mhz=40), strict=True, seed=0)
    dl_tr = NRConfig(dl=True, traffic=[TM.periodic(100, 10).downlink()])
    assert dl_tr.unused_fields("L2") == [] and "traffic" in dl_tr.unused_fields("L1")
