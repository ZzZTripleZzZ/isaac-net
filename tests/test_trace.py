"""Slot-level trace (core/trace.py) and its plots (viz/trace.py), CPU.

The trace is bitwise invisible to the engine (UL and UL + DL, one and three cells, partial resets, traffic models with
RACH / DRX / TPC); every frame the step dict reports delivered has an arrival, at least one transport block and a
delivered event whose delay equals the step's `delay`; with every robot traced the TB counts equal the MAC counters;
the graph and triton backends are refused; the tables round-trip through CSV (and Parquet with pyarrow); the plots
render with the Agg backend and the slot map has one row per traced robot.
"""
import math

import pytest
import torch

from isaac_net.core import NRConfig, Requests, TrafficModel, make_engine, multicell
from isaac_net.core.trace import EVENT_COLS, SlotTrace, TraceTable

E, R = 3, 3


def _drive(net, steps=10, seed=3, reset_at=4, reset_ids=(1,), pos=False, dl=False, sizes=True):
    g = torch.Generator().manual_seed(seed)
    outs = []
    for k in range(steps):
        send = torch.randint(0, 3, (E, R), generator=g)
        x = torch.rand(E, R, 2, generator=g) * 100 if pos else 25 * torch.rand(E, R, generator=g) - 5
        if sizes:
            net.submit(None, Requests(send))
        if dl:
            net.add_dl_frames(None, torch.where(send > 0, 1500.0, 0.0))
        T = int(net.T)
        o = net.step(None, x)
        outs.append((T, {k_: v.clone() for k_, v in o.items() if torch.is_tensor(v)}))
        if reset_at is not None and k == reset_at:
            net.reset(torch.tensor(reset_ids))
    return outs


def _same(a, b):
    assert [t for t, _ in a] == [t for t, _ in b]
    for (_, x), (_, y) in zip(a, b):
        assert set(x) == set(y)
        for k in x:
            u, v = x[k], y[k]
            if u.is_floating_point():
                u, v = u.nan_to_num(-7.0), v.nan_to_num(-7.0)
            assert torch.equal(u, v), k


CASES = {
    "ul_1cell": (NRConfig(), False, False),
    "uldl_1cell": (NRConfig(dl=True), False, True),
    "ul_3cells": (multicell(3), True, False),
    "uldl_3cells": (multicell(3, dl=True), True, True),
}


@pytest.mark.parametrize("case", list(CASES))
def test_trace_is_bitwise_invisible(case):
    cfg, pos, dl = CASES[case]
    a = make_engine("L2", E, R, "cpu", cfg, seed=5)
    b = make_engine("L2", E, R, "cpu", cfg, seed=5)
    tr = SlotTrace.attach(a, env=[0, 1], robots=[0, 2], flush_steps=3)
    torch.manual_seed(0)
    oa = _drive(a, pos=pos, dl=dl)
    torch.manual_seed(0)
    ob = _drive(b, pos=pos, dl=dl)
    _same(oa, ob)
    assert a.counters() == b.counters()
    ev = tr.events()
    kinds = {e["event"] for e in ev}
    assert {"arrival", "grant", "tb_new", "ack", "delivered", "reset"} <= kinds
    if dl:
        assert any(e["dir"] == "dl" and e["event"] == "tb_new" for e in ev)
        assert any(e["event"] == "cqi" for e in ev)
    assert {(e["env"], e["robot"]) for e in ev} <= {(0, 0), (0, 2), (1, 0), (1, 2)}
    assert any(e["episode"] == 1 for e in ev if e["env"] == 1) and all(e["episode"] == 0 for e in ev if e["env"] == 0)
    assert len(tr.samples()) > 0


def test_trace_with_traffic_access_tpc_is_bitwise_invisible():
    cfg = NRConfig(rach=True, rach_initial="idle", drx=True, drx_inactivity_ms=10.0, drx_cycle_ms=40.0,
                   ul_pc=True, ul_tpc=True, control_step_ms=20.0,
                   traffic=(TrafficModel.periodic(900, 10.0),))
    a = make_engine("L2", E, R, "cpu", cfg, seed=2)
    b = make_engine("L2", E, R, "cpu", cfg, seed=2)
    tr = SlotTrace.attach(a, env=0)
    oa = _drive(a, steps=12, sizes=False)
    ob = _drive(b, steps=12, sizes=False)
    _same(oa, ob)
    assert a.counters() == b.counters()
    kinds = {e["event"] for e in tr.events()}
    assert {"access", "rach_preamble", "tpc", "delivered"} <= kinds
    _check_delivered(tr, oa, a.config, pairs=[(0, r) for r in range(R)])


def _check_delivered(tr, outs, cfg, pairs):
    ev = tr.events()
    fr = {(f["env"], f["robot"], f["frame"]): f for f in tr.frames("ul")}
    by = {}
    for e in ev:
        if e["dir"] == "ul" and e["event"] == "delivered":
            by[(e["step"], e["env"], e["robot"], e["fidx"])] = e
    n = 0
    for T, o in outs:
        for e_, r_ in pairs:
            for f in torch.nonzero(o["delivered"][e_, r_]).flatten().tolist():
                d = by[(T, e_, r_, f)]
                frame = fr[(e_, r_, d["frame"])]
                assert frame["arrival_ms"] <= d["t_ms"]
                assert len(frame["tbs"]) >= 1
                assert abs(d["delay_ms"] - float(o["delay"][e_, r_, f]) * cfg.control_step_ms) <= cfg.slot_ms
                n += 1
    assert n > 0
    return n


@pytest.mark.parametrize("traffic", [False, True])
def test_events_match_step_outputs_and_counters(traffic):
    cfg = NRConfig(traffic=(TrafficModel.periodic(1200, 25.0),)) if traffic else NRConfig()
    net = make_engine("L2", E, R, "cpu", cfg, seed=4)
    pairs = [(e, r) for e in range(E) for r in range(R)]
    tr = SlotTrace.attach(net, pairs=pairs)
    outs = _drive(net, steps=12, reset_at=5, reset_ids=(0, 2), sizes=not traffic)
    _check_delivered(tr, outs, cfg, pairs)
    ev = tr.events()
    c = net.counters()["ul"]
    assert sum(e["event"] == "tb_retx" and e["dir"] == "ul" for e in ev) == c["tb_retx"]
    assert sum(e["event"] == "tb_new" and e["dir"] == "ul" for e in ev) == c["tb_new"]
    assert sum(e["event"] == "ack" for e in ev) == c["tb_ok"]
    assert sum(e["event"] == "nack" for e in ev) == c["tb_fail"]
    # every TB event of a traced robot is preceded by its grant in the same slot
    grants = {(e["env"], e["robot"], e["g"], e["pid"]) for e in ev if e["event"] == "grant"}
    assert all((e["env"], e["robot"], e["g"], e["pid"]) in grants for e in ev if e["event"] in ("tb_new", "tb_retx"))
    s = tr.summary()
    rows = s.to_dict("records") if hasattr(s, "to_dict") else s
    assert len(rows) == len(pairs)
    assert sum(r["tb_retx"] for r in rows) == c["tb_retx"]


def test_refuses_fast_backends_and_other_levels():
    from isaac_net.core.nr_fast import NRGraphEngine, NRTritonEngine
    with pytest.raises(ValueError, match="triton"):
        SlotTrace.attach(NRTritonEngine.__new__(NRTritonEngine))
    with pytest.raises(ValueError, match="reference"):
        SlotTrace.attach(NRGraphEngine.__new__(NRGraphEngine))
    with pytest.raises(ValueError, match="L2"):
        SlotTrace.attach(make_engine("L1", 2, 2, "cpu", NRConfig()))
    with pytest.raises(IndexError):
        SlotTrace.attach(make_engine("L2", 2, 2, "cpu", NRConfig()), env=5)


def test_max_events_and_detach():
    net = make_engine("L2", E, R, "cpu", NRConfig(), seed=1)
    tr = SlotTrace.attach(net, env=0, max_events=200, flush_steps=1)
    _drive(net, steps=6, reset_at=None)
    assert tr.truncated and len(tr.events()) + len(tr.samples()) <= 200 + 400
    tr2 = SlotTrace.attach(net, env=1)
    tr2.detach()
    n = len(tr2.events())
    _drive(net, steps=2, reset_at=None)
    assert len(tr2.events()) == n


def test_frame_and_csv_round_trip(tmp_path):
    pd = pytest.importorskip("pandas")
    net = make_engine("L2", E, R, "cpu", NRConfig(dl=True), seed=3)
    tr = SlotTrace.attach(net, env=0, robots=[1])
    _drive(net, steps=6, dl=True)
    df = tr.to_frame()
    assert list(df.columns) == list(EVENT_COLS) and len(df) > 0
    files = tr.save(tmp_path / "run.csv")
    assert len(files) == 3
    back = TraceTable.load(tmp_path / "run.csv")
    pd.testing.assert_frame_equal(back.to_frame(), df, check_dtype=False)
    pd.testing.assert_frame_equal(back.to_frame("samples"), tr.to_frame("samples"), check_dtype=False)
    assert back.meta["slot_ms"] == net.config.slot_ms
    s1, s2 = tr.summary(), back.summary()
    assert math.isclose(float(s1["delay_mean_ms"].iloc[0]), float(s2["delay_mean_ms"].iloc[0]), rel_tol=1e-12)
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        return
    tr.save(tmp_path / "run.parquet")
    pq = TraceTable.load(tmp_path / "run.parquet")
    pd.testing.assert_frame_equal(pq.to_frame(), df, check_dtype=False)


def test_plots_render(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from isaac_net.viz.trace import plot_slot_heatmap, plot_timeline, slot_occupancy
    net = make_engine("L2", E, R, "cpu", NRConfig(dl=True, control_step_ms=20.0), seed=2)
    tr = SlotTrace.attach(net, env=0)
    _drive(net, steps=6, dl=True)
    fig = plot_timeline(tr, robot=1, path=tmp_path / "t.png")
    assert (tmp_path / "t.png").stat().st_size > 0 and len(fig.axes) >= 2
    fig = plot_timeline(tr, robot=0, t0_ms=0.0, t1_ms=40.0, path=tmp_path / "t.pdf")
    assert (tmp_path / "t.pdf").stat().st_size > 0
    grid, slots, labels = slot_occupancy(tr, t0_ms=0.0, t1_ms=50.0)
    assert len(grid) == R == len(labels) and all(len(r) == len(slots) == 101 for r in grid)
    fig = plot_slot_heatmap(tr, t0_ms=0.0, t1_ms=50.0, path=tmp_path / "h.png")
    assert fig.slot_grid.shape == (R, 101)
    import matplotlib.pyplot as plt
    plt.close("all")
