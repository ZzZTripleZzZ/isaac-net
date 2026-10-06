"""RecorderLoop (core/record.py): bitwise invisible to the engine, no host sync per step, table schema and round trip."""
import json
import math
import os

import pytest
import torch

from isaac_net import NRConfig, RecordConfig, RecorderLoop, make_engine, record
from isaac_net.core.background import BackgroundConfig
from isaac_net.core.config import multicell
from isaac_net.core.energy import EnergyConfig
from isaac_net.core.record import (CELL_COLUMNS, HIST_COLUMNS, ROBOT_COLUMNS, SCHEMA, STEP_COLUMNS, read_meta,
                                   read_records)

pd = pytest.importorskip("pandas")

E, R, T = 3, 4, 14


def _cfg_l2_wrapped():
    return multicell(3).with_(energy=EnergyConfig(battery_j=50.0), background=BackgroundConfig(n_background=2),
                              rach=True, drx=True)


def _rollout(net, steps=T, seed=5):
    g = torch.Generator().manual_seed(seed)
    net.reset()
    pos = torch.rand(E, R, 2, generator=g) * 60
    outs = []
    for s in range(steps):
        send = (torch.rand(E, R, generator=g) < 0.6).long() * torch.randint(1, 3, (E, R), generator=g)
        acc = net.submit(None, send)
        out = net.step(None, pos)
        outs.append({**{k: v.clone() for k, v in out.items() if torch.is_tensor(v)}, "_acc": acc.clone()})
        pos = (pos + torch.randn(E, R, 2, generator=g)).clamp(0, 60)
        if s in (4, 9):
            net.reset(torch.tensor([s % E]))
    return outs


def _same(a, b):
    if a.dtype.is_floating_point:
        return torch.equal(torch.isnan(a), torch.isnan(b)) and torch.equal(torch.nan_to_num(a), torch.nan_to_num(b))
    return torch.equal(a, b)


@pytest.mark.parametrize("level,cfg", [("L2", None), ("L2", "wrapped"), ("L2-legacy", None), ("L0", None)])
def test_recorder_bitwise_invisible(level, cfg, tmp_path):
    c = _cfg_l2_wrapped() if cfg == "wrapped" else NRConfig()
    torch.manual_seed(0)
    ref = _rollout(make_engine(level, E, R, "cpu", c, seed=1))
    torch.manual_seed(0)
    net = record(make_engine(level, E, R, "cpu", c, seed=1), str(tmp_path), every=3, flush_every=2,
                 raw_envs=(0, 2))
    got = _rollout(net)
    net.close()
    assert len(ref) == len(got)
    for a, b in zip(ref, got):
        assert a.keys() == b.keys()
        for k in a:
            assert _same(a[k], b[k]), (level, cfg, k)
    # the global torch RNG is untouched too
    assert len(read_records(str(tmp_path))) == E * math.ceil(T / 3)


def test_recorder_no_host_sync_in_step(tmp_path, monkeypatch):
    """Between flushes the recorder's own work runs no host-syncing tensor op (item / tolist / cpu / numpy /
    bool / float / int)."""
    net = record(make_engine("L2", E, R, "cpu", _cfg_l2_wrapped(), seed=1), str(tmp_path), every=2,
                 flush_every=100, raw_envs=(1,))
    net.reset()
    obs = net._observe
    banned = ("item", "tolist", "cpu", "numpy", "__bool__", "__float__", "__int__", "__index__")

    def guarded(out, x):
        saved = {n: getattr(torch.Tensor, n) for n in banned}

        def boom(*a, **k):
            raise AssertionError("host sync in RecorderLoop step")
        try:
            for n in banned:
                setattr(torch.Tensor, n, boom)
            return obs(out, x)
        finally:
            for n, f in saved.items():
                setattr(torch.Tensor, n, f)
    monkeypatch.setattr(net, "_observe", guarded)
    pos = torch.rand(E, R, 2) * 60
    for _ in range(9):
        net.submit(None, torch.ones(E, R, dtype=torch.long))
        net.step(None, pos)
    monkeypatch.undo()
    net.close()
    assert len(read_records(str(tmp_path))) == E * 5


@pytest.mark.gpu
def test_recorder_no_sync_cuda(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    net = record(make_engine("L2-legacy", 8, 4, "cuda", backend="graph", seed=0), str(tmp_path), flush_every=50)
    net.reset()
    pos = torch.rand(8, 4, 2, device="cuda") * 60
    for i in range(20):
        net.submit(None, torch.ones(8, 4, dtype=torch.long, device="cuda"))
        torch.cuda.set_sync_debug_mode("error")
        try:
            net.step(None, pos)
        finally:
            torch.cuda.set_sync_debug_mode("default")
    net.close()


@pytest.mark.parametrize("fmt", ["parquet", "csv"])
def test_schema_and_round_trip(fmt, tmp_path):
    if fmt == "parquet":
        pytest.importorskip("pyarrow")
    out = str(tmp_path / fmt)
    net = RecorderLoop(make_engine("L2", E, R, "cpu", _cfg_l2_wrapped(), seed=1),
                       RecordConfig(out_dir=out, every=4, flush_every=2, format=fmt, raw_envs=(1,), label="mc"))
    outs = _rollout(net, steps=10)
    net.close()
    meta = read_meta(out)
    assert meta["schema"] == SCHEMA and meta["format"] == fmt and (meta["E"], meta["R"], meta["C"]) == (E, R, 3)
    st = read_records(out)
    assert list(st.columns) == list(STEP_COLUMNS) + ["label", "level"]
    assert (st.label == "mc").all() and (st.level == "L2").all()
    assert sorted(st.step.unique()) == [4, 8, 10] and list(st.groupby("step").steps.first()) == [4, 4, 2]
    # sums over the windows equal the engine's own outputs
    dl = sum(o["delivered"][:, :R].sum((-1, -2)) for o in outs)
    assert st.groupby("env").delivered.sum().tolist() == dl.double().tolist()
    sent = sum(o["_acc"].sum(-1) for o in outs)
    assert st.groupby("env").sent.sum().tolist() == sent.double().tolist()
    assert st.prb_util.between(0, 1).all() and st.harq_bler.dropna().between(0, 1).all()
    shares = st[["access_idle", "access_rach", "access_connected", "access_dormant"]].sum(1)
    assert ((shares - 1).abs() < 1e-9).all()
    assert st.energy_j.notna().all() and st.bg_util.notna().all() and st.dl_prb_util.isna().all()
    assert st.episode.max() >= 1 and st.episode.min() == 0
    ce = read_records(out, "cells")
    assert list(ce.columns) == list(CELL_COLUMNS) + ["label", "level"] and len(ce) == 3 * E * 3
    assert (ce.groupby(["step", "env"]).robots.sum() == R).all()
    ro = read_records(out, "robots")
    assert list(ro.columns) == list(ROBOT_COLUMNS) + ["label", "level"]
    assert set(ro.env) == {1} and len(ro) == 3 * R and ro.x.notna().all()
    for h in ("delay_hist", "aoi_hist"):
        hh = read_records(out, h)
        assert list(hh.columns) == list(HIST_COLUMNS) + ["label", "level"] and (hh["count"] > 0).all()
    assert read_records(out, "delay_hist")["count"].sum() == dl.sum().item()
    assert read_records(out, "aoi_hist")["count"].sum() == 10 * E * R
    with open(os.path.join(out, "meta.json")) as f:
        assert json.load(f)["tables"]["steps"] == list(STEP_COLUMNS)


def test_minimal_level_columns_nan(tmp_path):
    net = record(make_engine("L0", E, R, "cpu", seed=0), str(tmp_path), per_cell=False, hist=False)
    _rollout(net, steps=3)
    net.close()
    st = read_records(str(tmp_path))
    assert len(st) == 3 * E and st.prb_util.isna().all() and st.energy_j.isna().all()
    assert st.delivered.sum() >= 0 and not os.path.exists(tmp_path / "cells")


def test_record_config_validation():
    with pytest.raises(ValueError):
        RecordConfig(every=0)
    with pytest.raises(ValueError):
        RecordConfig(format="xlsx")
