"""The known-limits follow-ups (docs/STATUS.md items 18, 19, 20, 22 and the triton refusals). CPU unless marked gpu.

(a) configurations that do not use the new paths are bitwise the engine before the changes (golden digests recorded
    on that commit, fixtures/limits_off_golden.json): RLF without RACH (with and without the access stage), RACH
    without RLF, the EdgeLoop "delay" return path on one carrier, DL traffic models on the reference backend, the
    run-time sector pattern on a radio map
(b) DL traffic models on the graph backend: the graph engine's code path (on CPU each replay re-runs the captured
    region with the capture's host inputs, host state and gate tensors, so the static gate buffers are exercised) is
    bitwise equal to the reference, with UL models, one and three cells, RACH, and partial resets; gpu: CUDA graphs
(c) triton refusals: every feature the fused kernel does not implement is refused in NRTritonEngine.__init__ with one
    message, and docs/configurability.md lists the same features; RACH / DRX, FDD and DL traffic models are accepted
(d) EdgeLoop "delay" return path under FDD: the command rate uses the DL carrier's dl_nprb
(e) RLF re-establishment through RACH: the robot enters RACH toward the selected cell, the outage lasts until
    contention resolution, collisions retry; graph code path equal to the reference
(f) sector pattern at bake time: equals the run-time pattern at the grid points, the engine refuses applying it
    twice, bake keys and metadata, the synthetic map tool's CLI
"""
import json
import os
import re

import numpy as np
import pytest
import torch

import limits_off_scenarios as golden
from isaac_net.core import EdgeConfig, EdgeLoop, NRConfig, Requests, TrafficModel as TM, make_engine, multicell
from isaac_net.core.nr_fast import TRITON_NOT_IMPLEMENTED, NRGraphEngine, NRTritonEngine, TritonUnsupported, state_dict

HERE = os.path.dirname(os.path.abspath(__file__))
QUIET = dict(ul_interference=False, dl_interference=False, control_step_ms=20.0)


class CPUGraph(NRGraphEngine):
    """The graph engine on CPU: captures and replays through nr_fast._EagerReplay (see the module docstring)."""
    _require_cuda = False


def _equal(a, b):
    if a.is_floating_point():
        return torch.equal(a.nan_to_num(-7.0), b.nan_to_num(-7.0))
    return torch.equal(a, b)


def _same_outputs(oa, ob, t):
    assert set(oa) == set(ob), (t, set(oa) ^ set(ob))
    for k in oa:
        if torch.is_tensor(oa[k]):
            assert _equal(oa[k], ob[k]), (t, k)


# ---------------------------------------------------------------------------------------------- (a) bitwise off
def test_off_paths_are_bitwise_the_pre_change_engine():
    with open(golden.GOLDEN) as fh:
        g = json.load(fh)
    if g["env"] != golden.env_key():
        pytest.skip(f"golden digests recorded on {g['env']}, this is {golden.env_key()} (bitwise is per platform)")
    got = golden.compute()
    assert got == g["digests"], {k: got[k] == v for k, v in g["digests"].items()}


# ---------------------------------------------------------------------------------------------- (b) DL traffic on graph
DL_MODELS = [TM.periodic(1200, 5, jitter_ms=2).downlink().on([0, 1]),
             TM.bursty(900, 30, 2, (0.4, 0.4)).downlink().on([2, 3]),
             TM.video(10, 4000, gop=(8000, 2000, 4)).downlink().on(1)]
UL_MODELS = [TM.periodic(700, 10).on([0, 2]), TM.event(500, trigger="alarm")]


def _drive(eng, E, R, steps, pos, reset_at, dl_frames, seed=0):
    g = torch.Generator().manual_seed(seed)
    dev = getattr(eng, "dev", "cpu")                        # inputs are drawn on the CPU, moved to the engine device
    p = torch.rand(E, R, 2, generator=g) * 150
    outs = []
    for t in range(steps):
        if t == reset_at:
            eng.reset(torch.tensor([1], device=dev))
        send = (torch.rand(E, R, generator=g) < 0.3).long()
        trig = {"alarm": (torch.rand(E, R, generator=g) < 0.3).to(dev)}
        eng.submit(None, Requests(send.to(dev)))
        if dl_frames:
            eng.add_dl_frames(None, torch.where(torch.rand(E, R, generator=g) < 0.3, 2500.0, 0.0).to(dev))
        if pos:
            p = (p + 5.0 * (2 * torch.rand(E, R, 2, generator=g) - 1)).clamp(0, 150)
            o = eng.step(None, p.to(dev), triggers=trig)
        else:
            o = eng.step(None, (25 * torch.rand(E, R, generator=g) - 5).to(dev), triggers=trig)
        outs.append({k: v.clone() for k, v in o.items() if torch.is_tensor(v)})
    return outs


def _compare_graph(cfg, E=3, R=4, steps=12, pos=False, reset_at=7, dl_frames=True, device="cpu", seed=3):
    ref = make_engine("L2", E, R, device, cfg, seed=seed)
    gr = (make_engine("L2", E, R, device, cfg, "graph", seed=seed) if device != "cpu"
          else CPUGraph(E, R, device, cfg, seed=seed))
    a = _drive(ref, E, R, steps, pos, reset_at, dl_frames)
    b = _drive(gr, E, R, steps, pos, reset_at, dl_frames)
    for t, (oa, ob) in enumerate(zip(a, b)):
        _same_outputs(oa, ob, t)
    sa, sb = state_dict(ref), state_dict(gr)
    for k in sa:
        assert _equal(sa[k], sb[k]), k
    assert ref.counters() == gr.counters()
    assert {k: int(v) for k, v in ref.traffic_stats_dl.items()} == {k: int(v) for k, v in gr.traffic_stats_dl.items()}
    assert gr.n_replays == steps
    return a


@pytest.mark.parametrize("case", ["dl_only", "ul_and_dl", "three_cells", "rach"])
def test_dl_traffic_graph_code_path_equals_reference(case):
    if case == "dl_only":
        cfg, pos = NRConfig(dl=True, control_step_ms=20.0, frame_buffer=32, traffic=DL_MODELS), False
    elif case == "ul_and_dl":
        cfg, pos = NRConfig(dl=True, control_step_ms=20.0, frame_buffer=32, traffic=DL_MODELS + UL_MODELS), False
    elif case == "three_cells":
        cfg, pos = multicell(3, dl=True, control_step_ms=20.0, frame_buffer=32, ho_rlc="flush", a3_ttt_ms=40.0,
                             cell_isd_m=60.0, traffic=DL_MODELS + UL_MODELS), True
    else:
        cfg, pos = NRConfig(dl=True, control_step_ms=20.0, frame_buffer=32, rach=True, rach_initial="idle",
                            rach_release_after_ms=60.0, traffic=DL_MODELS), False
    outs = _compare_graph(cfg, pos=pos)
    assert sum(int(o["gen_dl_accepted"].sum()) for o in outs) > 0
    assert sum(int((o["dl_delivered"] & o["dl_generated"]).sum()) for o in outs) > 0


def test_graph_cpu_stand_in_catches_a_missing_static_gate(monkeypatch):
    """Negative control: without copying the DL gate into static buffers, the replays read stale arrivals."""
    def no_static(self):
        g = getattr(self, "_dl_gate", None)
        if g is None:
            return None
        self._static_g0()
        return tuple(tuple(x.shape) for x in g)
    monkeypatch.setattr(NRGraphEngine, "_static_dl_gate", no_static)
    with pytest.raises(AssertionError):
        _compare_graph(NRConfig(dl=True, control_step_ms=20.0, frame_buffer=32, traffic=DL_MODELS))


def test_graph_backend_accepts_dl_traffic_models():
    cfg = NRConfig(dl=True, traffic=[TM.periodic(1000, 5).downlink()])
    with pytest.raises(ValueError, match="CUDA"):            # past the former DL-traffic refusal: the device check
        make_engine("L2", 2, 2, "cpu", cfg, backend="graph", seed=0)


@pytest.mark.gpu
@pytest.mark.parametrize("case", ["ul_and_dl", "three_cells"])
def test_dl_traffic_graph_cuda_bitwise(case):
    if case == "ul_and_dl":
        cfg, pos = NRConfig(dl=True, control_step_ms=20.0, frame_buffer=32, traffic=DL_MODELS + UL_MODELS), False
    else:
        cfg, pos = multicell(3, dl=True, control_step_ms=20.0, frame_buffer=32, a3_ttt_ms=40.0, cell_isd_m=60.0,
                             traffic=DL_MODELS + UL_MODELS), True
    _compare_graph(cfg, pos=pos, steps=24, reset_at=11, device="cuda")


# ---------------------------------------------------------------------------------------------- (c) triton refusals
TRITON_CASES = {
    "n_cells": (multicell(3), "several cells"),
    "rlf": (multicell(3, rlf=True), "radio link failure"),
    "bsr": (NRConfig(ul_grant_model="bsr"), "grant pipeline"),
    "ul_tpc": (NRConfig(ul_pc=True, ul_tpc=True), "ul_tpc"),
    "cqi": (NRConfig(dl=True, cqi_table="38214"), "38.214 CQI"),
}
# implemented by the kernel since feat/tritonB (tests/test_triton_access_fdd_dl.py): accepted, not refused
TRITON_ACCEPTS = {
    "rach": NRConfig(rach=True),
    "drx": NRConfig(drx=True),
    "fdd": NRConfig(duplex="fdd", dl=True, dl_bandwidth_mhz=40),
    "dl_traffic": NRConfig(dl=True, traffic=[TM.periodic(1000, 5).downlink()]),
}


@pytest.mark.parametrize("name", list(TRITON_CASES))
def test_triton_refuses_in_one_place(name):
    cfg, what = TRITON_CASES[name]
    assert any(what in f for f, _ in NRTritonEngine.refusals(cfg))
    with pytest.raises(TritonUnsupported, match=re.escape(what)) as ei:
        make_engine("L2", 2, 2, "cpu", cfg, "triton", seed=0)
    assert isinstance(ei.value, ValueError) and isinstance(ei.value, NotImplementedError)
    assert "backend='graph'" in str(ei.value) and TRITON_NOT_IMPLEMENTED in str(ei.value)


@pytest.mark.parametrize("name", list(TRITON_ACCEPTS))
def test_triton_accepts_access_fdd_and_dl_traffic(name):
    cfg = TRITON_ACCEPTS[name]
    assert NRTritonEngine.refusals(cfg) == []
    with pytest.raises(ValueError, match="CUDA"):           # past the former refusal: the device check
        make_engine("L2", 2, 2, "cpu", cfg, "triton", seed=0)


def test_triton_refusals_list_several_features_and_pass_defaults():
    assert NRTritonEngine.refusals(NRConfig()) == [] and NRTritonEngine.refusals(NRConfig(dl=True)) == []
    cfg = multicell(3, rach=True, drx=True, dl=True, ul_grant_model="bsr", traffic=[TM.periodic(1000, 5).downlink()])
    assert len(NRTritonEngine.refusals(cfg)) == 2           # several cells, the BSR pipeline


def test_docs_backend_table_lists_the_triton_refusals():
    with open(os.path.join(HERE, "..", "docs", "configurability.md")) as fh:
        doc = fh.read()
    sec = doc[doc.index("## NR engine backends"):]
    sec = sec[:sec.index("\n## ", 3)]
    rows = [ln for ln in sec.splitlines() if ln.startswith("| ") and "refused" in ln]
    for key in ("n_cells", "rlf", "ul_grant_model", "ul_tpc", "cqi_table", "SINR hook"):
        assert any(key in r for r in rows), key
    runs = [ln for ln in sec.splitlines() if ln.startswith("| ") and ln.rstrip().endswith("| yes |")]
    for key in ("rach", "drx", "duplex", "direction=\"dl\""):                # run on triton since feat/tritonB
        assert not any(key in r for r in rows) and any(key in r for r in runs), key


# ---------------------------------------------------------------------------------------------- (d) EdgeLoop under FDD
class _Scripted:
    """One message per robot delivered at step 0 (capture 0, arrival 0.1), SINR 20 dB."""

    def __init__(self, cfg, E=1, R=2):
        self.E, self.R, self.dev, self.config, self.F = E, R, torch.device("cpu"), cfg, cfg.frame_buffer
        self.clock = torch.zeros(E, dtype=torch.long)

    def reset(self, env_ids=None):
        self.clock.zero_()

    def submit(self, t, requests, snr_db=None):
        return requests.send > 0

    def step(self, t=None, x=None):
        E, R, F = self.E, self.R, self.F
        t = self.clock.clone()
        first = int(t[0]) == 0
        dlv = torch.zeros(E, R, F, dtype=torch.bool)
        dlv[..., 0] = first
        cap = torch.where(dlv, 0, -1)
        delay = torch.where(dlv, 0.1, float("nan"))
        self.clock += 1
        return {"delivered": dlv, "cap": cap, "cls": torch.zeros(E, R, F, dtype=torch.long), "delay": delay, "t": t,
                "sinr_db": torch.full((E, R), 20.0)}


@pytest.mark.parametrize("dl_n_prb", [None, 106])
def test_edge_delay_return_path_uses_the_dl_carrier(dl_n_prb):
    import math
    ecfg = EdgeConfig(service_ms=20.0, return_path="delay", ret_fixed_ms=5.0, cmd_bytes=20000, ret_share=0.5)
    tdd = NRConfig(frame_buffer=4)
    fdd = NRConfig(frame_buffer=4, duplex="fdd", dl=True, dl_n_prb=dl_n_prb)
    assert fdd.dl_nprb == (fdd.nprb if dl_n_prb is None else dl_n_prb) and fdd.nprb == tdd.nprb
    loop = {c: EdgeLoop(_Scripted(cfg), ecfg) for c, cfg in (("tdd", tdd), ("fdd", fdd))}
    assert loop["fdd"].ret_rate_hz == pytest.approx(loop["tdd"].ret_rate_hz * fdd.dl_nprb / tdd.nprb, rel=1e-12)
    o = loop["fdd"].step(None)
    snr = 10 ** ((20.0 + ecfg.ret_snr_offset_db) / 10)
    rate = ecfg.ret_rate_eta * 0.5 * fdd.dl_nprb * 12 * fdd.scs_khz * 1e3 * math.log2(1 + snr)
    ret = (5.0 + 8 * 20000 / rate * 1e3) / fdd.control_step_ms
    assert float(o["act_ret_delay"][0, 0]) == pytest.approx(ret, rel=1e-5)
    if dl_n_prb is not None:              # a wider DL carrier returns the command sooner
        assert float(o["act_ret_delay"][0, 0]) < float(loop["tdd"].step(None)["act_ret_delay"][0, 0])


# ---------------------------------------------------------------------------------------------- (e) RLF through RACH
def _rlf_run(rach, steps=24, E=2, R=3, preambles=64, fail_robots=(0,), backoff_ms=20.0, reset=None, eng_cls=None):
    """Robots fail_robots: 20 dB on cell 0 for 5 steps, then -15 dB on cell 0 and 5 dB on cell 1; the others stay at
    20 dB on cell 0. A3 is off, so only RLF moves them."""
    cfg = multicell(3, rlf=True, a3_ttt_ms=1e6, t310_ms=200.0, reest_delay_ms=50.0, frame_buffer=64,
                    timeout_steps=200, rlf_rlc="carry", rach=rach, rach_preambles=preambles,
                    rach_backoff_ms=backoff_ms, **QUIET)
    eng = (eng_cls or (lambda *a, **k: make_engine("L2", *a, **k)))(E, R, "cpu", cfg, seed=0)
    eng.net.ul.trace = [] if eng_cls is None else None
    rec = []
    for t in range(steps):
        if reset is not None and t == reset[0]:
            eng.reset(torch.tensor(reset[1]))
        rows = [[-15.0, 5.0, 0.0] if (r in fail_robots and t >= 5) else [20.0, 10.0 if r in fail_robots else 0.0, 0.0]
                for r in range(R)]
        pg = (torch.tensor(rows) - cfg.ue_tx_dbm + cfg.subband_noise_dbm)[None].expand(E, R, 3).clone()
        eng.submit(None, torch.ones(E, R, dtype=torch.long))
        o = eng.step(None, pathgain_db=pg)
        rec.append({k: v.clone() for k, v in o.items() if torch.is_tensor(v)})
    return eng, cfg, rec


def _outage(rec, e=0, r=0):
    return [bool(o["rlf"][e, r]) for o in rec]


def test_rlf_reestablishes_through_rach():
    eng, cfg, rec = _rlf_run(True)
    N, asc, acc = cfg.slots_per_step, eng.net.assoc, eng.access
    g_rlf = 5 * N + asc.t310                                  # T310 from the first out-of-sync indication
    t_rlf = g_rlf // N
    ro = acc._next_ro((t_rlf + 1) * N)                       # the first RO the stage processes after the selection
    g_back = ro + acc.cres                                   # contention resolution: served from here
    assert _outage(rec) == [g_rlf <= t * N + N - 1 < g_back for t in range(len(rec))], _outage(rec)
    assert [int(o["serving_cell"][0, 0]) for o in rec] == [0 if t * N + N - 1 < g_back else 1 for t in range(len(rec))]
    att = [int(o["rach_attempts"][0, 0]) for o in rec]
    assert att[ro // N] == 1 and sum(att) == 1                # one preamble, at that RO
    c = eng.counters()
    assert c["access"]["rach_attempts"] == 2.0 and c["access"]["rach_successes"] == 2.0      # robot 0 of 2 envs
    assert c["rlf"]["rlf"] == 2.0 and c["rlf"]["reest"] == 2.0
    tx = sorted({round(frac * N) - 1 for frac, txm, *_ in eng.net.ul.trace if txm[0, 0]})
    assert not any(g_rlf <= g < g_back for g in tx)            # never scheduled during the outage incl. access
    ul = [g for g in range(len(rec) * N) if cfg.slot_symbols(g)[1] > 0]
    assert next(g for g in ul if g >= g_back) in tx
    assert not asc.rlf_reest_request.any() and not acc.reest.any()
    assert int(rec[-1]["access_state"][0, 0]) == 2           # connected


def test_rlf_without_rach_keeps_the_fixed_delay():
    eng, cfg, rec = _rlf_run(False)
    N, asc = cfg.slots_per_step, eng.net.assoc
    g_rlf = 5 * N + asc.t310
    assert _outage(rec) == [g_rlf <= t * N + N - 1 < g_rlf + asc.reest for t in range(len(rec))]
    assert eng.access is None and not asc.reest_rach and not hasattr(asc, "rlf_reest_request")


def test_rlf_rach_collisions_lengthen_the_outage():
    """Two robots of an env fail at the same slot toward the same cell with one preamble: they collide and back off,
    so the outage lasts longer than with 64 preambles and the collision counter rises."""
    kw = dict(fail_robots=(0, 1), backoff_ms=40.0, steps=30)
    eng1, cfg, rec1 = _rlf_run(True, preambles=1, **kw)
    eng64, _, rec64 = _rlf_run(True, preambles=64, **kw)
    c1, c64 = eng1.counters()["access"], eng64.counters()["access"]
    assert c1["rach_collisions"] >= 4 and c64["rach_collisions"] == 0
    assert c1["rach_attempts"] > c64["rach_attempts"] == 4
    assert sum(_outage(rec1)) > sum(_outage(rec64))
    assert not rec1[-1]["rlf"].any()                           # all re-established in the end


def test_rlf_rach_partial_reset_clears_the_request():
    eng, cfg, rec = _rlf_run(True, reset=(16, [1]))
    asc, acc = eng.net.assoc, eng.access
    assert int(asc.reest_end[1, 0]) == -1 and not bool(asc.rlf_active[1, 0]) and not bool(acc.reest[1, 0])
    assert eng.counters()["rlf"]["rlf"] == 2.0


def test_rlf_rach_graph_code_path_equals_reference():
    _, _, a = _rlf_run(True, reset=(12, [1]))
    _, _, b = _rlf_run(True, reset=(12, [1]), eng_cls=CPUGraph)
    for t, (oa, ob) in enumerate(zip(a, b)):
        _same_outputs(oa, ob, t)


# ---------------------------------------------------------------------------------------------- (f) sector at bake
from isaac_net.core.channels.antenna import gnb_antenna_gain_db  # noqa: E402
from isaac_net.core.channels.radio_map import make_synthetic_map, synthetic_gnb_xy  # noqa: E402
from isaac_net.core.radio import RadioMC  # noqa: E402
from isaac_net.tools.scene.bake import bake_key, build_parser  # noqa: E402
from isaac_net.tools.scene.sector import apply_sector_pattern, sector_gain_db, sector_radio_map  # noqa: E402

AZ = (20.0, 200.0)
GNB = synthetic_gnb_xy()


def _grid(m, E=None):
    x0, y0, x1, y1 = m.bounds
    X, Y = np.meshgrid(np.linspace(x0, x1, m.W), np.linspace(y0, y1, m.H))
    return torch.tensor(np.stack([X, Y], -1), dtype=torch.float32)        # [H, W, 2]


def test_bake_pattern_equals_runtime_pattern_at_grid_points():
    iso = make_synthetic_map(GNB)
    cfg = NRConfig(channel="radio_map", n_cells=2, cell_positions_m=tuple(GNB), gnb_antenna="sector",
                   cell_azimuth_deg=AZ, cell_tilt_deg=(4.0, 8.0), gnb_height_m=6.0)
    bake = sector_gain_db((2, iso.H, iso.W), iso.bounds, GNB, AZ, (4.0, 8.0), 8.0, gnb_z=(6.0, 6.0),
                          ue_h=cfg.ue_height_m)
    gnb3 = torch.tensor([[x, y, 6.0] for x, y in GNB])
    run = gnb_antenna_gain_db(cfg, _grid(iso), gnb3, cfg.ue_height_m).permute(2, 0, 1).numpy()
    assert np.allclose(bake, run, atol=1e-4)
    assert bake.max() <= 8.0 + 1e-9 and bake.min() >= 8.0 - 30.0 - 1e-9


def test_sector_baked_map_with_isotropic_engine_equals_runtime_sector():
    iso = make_synthetic_map(GNB)
    sec = sector_radio_map(iso, AZ, 0.0, 8.0)
    base = dict(channel="radio_map", n_cells=2, cell_positions_m=tuple(GNB), cell_azimuth_deg=AZ)
    pos = _grid(iso)[None]                                                 # every grid point, E = 1, R = H * W
    pos = pos.reshape(1, -1, 2)
    a = RadioMC(NRConfig(**base, gnb_antenna="sector"), 1, "cpu", radio_map=iso).pathgain_db(pos)
    b = RadioMC(NRConfig(**base), 1, "cpu", radio_map=sec).pathgain_db(pos)
    assert torch.allclose(a, b, atol=1e-3) and not torch.allclose(a, RadioMC(NRConfig(**base), 1, "cpu",
                                                                            radio_map=iso).pathgain_db(pos))
    assert sec.meta["gnb_antenna"] == "sector" and list(sec.meta["cell_azimuth_deg"]) == list(AZ)


def test_engine_refuses_the_pattern_twice_and_bake_refuses_a_second_pass(tmp_path):
    sec = sector_radio_map(make_synthetic_map(GNB), AZ)
    p = os.path.join(tmp_path, "sec.npz")
    sec.save(p)
    cfg = NRConfig(channel="radio_map", radio_map_path=p, n_cells=2, cell_positions_m=tuple(GNB), dl=True,
                   control_step_ms=20.0)
    with pytest.raises(ValueError, match="twice"):
        make_engine("L2", 1, 2, "cpu", cfg.with_(gnb_antenna="sector"), seed=0).step(None, torch.full((1, 2, 2), 50.0))
    o = make_engine("L2", 1, 2, "cpu", cfg, seed=0).step(None, torch.full((1, 2, 2), 50.0))
    assert torch.isfinite(o["sinr_db"]).all()
    with pytest.raises(ValueError, match="already"):
        sector_radio_map(sec, AZ)
    with pytest.raises(ValueError, match="one value or one per cell"):
        apply_sector_pattern({"gain_db": np.zeros((2, 3, 3)), "bounds": (0, 0, 1, 1), "gnb_xy": GNB}, (0.0, 1.0, 2.0))


def test_bake_key_and_cli_options():
    args = ("h", [[0, 0, 6]], 3.5, (0, 0, 10, 10), 1.0, 1.5, 1000, 4)
    assert bake_key(*args) == bake_key(*args, sector=None)
    k = bake_key(*args, sector=([30.0], [0.0], 8.0))
    assert k != bake_key(*args) and k != bake_key(*args, sector=([40.0], [0.0], 8.0))
    a = build_parser().parse_args(["--scene-xml", "s.xml", "--out", "m.npz", "--tx", "0", "0", "6", "--tx", "9", "0",
                                   "6", "--gnb-antenna", "sector", "--cell-azimuth", "0", "180", "--cell-tilt", "6"])
    from isaac_net.tools.scene.sector import sector_from_args
    assert sector_from_args(a, 2) == ([0.0, 180.0], [6.0], 8.0)
    assert build_parser().parse_args(["--scene-xml", "s.xml"]).gnb_antenna == "isotropic"


def test_synthetic_tool_writes_a_sector_map(tmp_path):
    from isaac_net.core.channels.radio_map import RadioMap
    from isaac_net.tools.make_synthetic_radio_map import main
    from isaac_net.tools.scene.bake import read_map, write_map
    p = os.path.join(tmp_path, "sec.npz")
    main(p, "--gnb-antenna", "sector", "--cell-azimuth", *map(str, AZ))
    m = RadioMap.load(p)
    want = sector_radio_map(make_synthetic_map(GNB), AZ)
    assert m.meta["gnb_antenna"] == "sector" and torch.equal(m.gain, want.gain)
    q = os.path.join(tmp_path, "iso.npz")
    main(q)
    assert "gnb_antenna" not in RadioMap.load(q).meta
    # the bake's own writer keeps the metadata in .pt and .npz files
    d = apply_sector_pattern({"gain_db": np.zeros((2, 4, 5), np.float32), "bounds": np.array([0.0, 0.0, 4.0, 3.0]),
                              "gnb_xy": np.array(GNB) / 40, "gnb_z": np.array([3.0, 3.0])}, AZ)
    for suffix in (".pt", ".npz"):
        r = read_map(write_map(os.path.join(tmp_path, "b" + suffix), d))
        assert str(r["gnb_antenna"]) == "sector" and np.allclose(r["gain_db"], d["gain_db"])
        assert str(RadioMap.load(os.path.join(tmp_path, "b" + suffix)).meta["gnb_antenna"]) == "sector"
