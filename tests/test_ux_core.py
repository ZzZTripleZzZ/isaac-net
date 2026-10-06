"""Usability layer of the core: scenario presets, NRConfig.describe / diff, backend="auto", output_schema(), and the
unused-field warning of make_engine.

(a) the four scenario presets build on CPU at level L2 (reference), take overrides, read every field they set, and
    the triton refusal list agrees with what their docstrings claim; (b) describe() names every field set away from
    its default and the backend verdict, diff() lists differing fields; (c) backend="auto" picks reference on CPU and
    logs it, and the selection logic for CUDA (monkeypatched) picks triton / graph as documented; (d) output_schema()
    lists exactly the keys step() returns, for the NR engine under eight configs, the wrappers, the legacy levels,
    NetSlotMC, WIFI and the adaptive engine, and the docs table is generated from the registry; (e) make_engine warns
    once per config about ignored fields, strict=True raises, strict=None is silent; (f) defaults are unchanged.
"""
import inspect
import logging
import os
import warnings

import pytest
import torch

from engine_api import default_params
from isaac_net.core import (EdgeConfig, NRConfig, Requests, TrafficModel, UnusedFieldsWarning, lena_validation_v2,
                            make_engine, multicell)
from isaac_net.core import engine as engine_mod
from isaac_net.core.background import BackgroundConfig
from isaac_net.core.energy import EnergyConfig
from isaac_net.core.nr_fast import NRTritonEngine
from isaac_net.core.scenarios import (SCENARIOS, factory_inf, outdoor_campus, urllc_control,
                                      warehouse_private_5g)
from isaac_net.core.schema import GROUPS, STEP_KEYS, markdown_table

E, R = 2, 3
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# what each preset's docstring says triton refuses (substrings of NRTritonEngine.refusals features)
TRITON_CLAIMS = {"warehouse_private_5g": ["ul_grant_model"], "factory_inf": ["several cells", "ul_grant_model"],
                 "outdoor_campus": ["several cells", "ul_grant_model"], "urllc_control": ["mini-slot"]}


def _drive(net, steps=3, pos=True, seed=0):
    g = torch.Generator().manual_seed(seed)
    out = None
    for _ in range(steps):
        net.submit(None, Requests(torch.ones(E, R, dtype=torch.long)))
        x = torch.rand(E, R, 2, generator=g) * 60 + 20 if pos else torch.full((E, R), 12.0)
        out = net.step(None, x)
    return out


# ---------------------------------------------------------------------------------------------- (a) presets
@pytest.mark.parametrize("name", list(SCENARIOS))
def test_scenario_builds_and_runs_on_cpu(name):
    cfg = SCENARIOS[name]()
    assert isinstance(cfg, NRConfig)
    assert cfg.unused_fields("L2") == [], cfg.unused_fields("L2")
    with warnings.catch_warnings():
        warnings.simplefilter("error", UnusedFieldsWarning)
        net = make_engine("L2", E, R, "cpu", cfg, "reference", seed=0)
    out = _drive(net)
    assert out["delivered"].shape == (E, R, cfg.frame_buffer)
    assert cfg.bler_source == "pdsch" and cfg.tbs_mode == "38214" and cfg.fading      # no local 5G-LENA tables


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_scenario_triton_refusals_match_the_docstring(name):
    f = SCENARIOS[name]
    feats = [feat for feat, _ in NRTritonEngine.refusals(f())]
    for claim in TRITON_CLAIMS[name]:
        assert any(claim in x for x in feats), (name, claim, feats)
    assert len(feats) == len(TRITON_CLAIMS[name]), feats
    assert "Triton refuses it" in f.__doc__ or "triton" in f.__doc__.lower()
    assert "Representative, not calibrated" in f.__doc__


def test_warehouse_lumped_runs_on_triton_and_overrides_apply():
    assert NRTritonEngine.refusals(warehouse_private_5g(ul_grant_model="lumped")) == []
    cfg = warehouse_private_5g(frame_buffer=32, ul_tpc=False)
    assert cfg.frame_buffer == 32 and not cfg.ul_tpc and cfg.ul_pc
    assert warehouse_private_5g().los_source == "stochastic"
    assert warehouse_private_5g(radio_map_path="hall.npz").los_source == "raycast"
    assert warehouse_private_5g(radio_map_path="hall.npz", los_source="map").los_source == "map"
    c = warehouse_private_5g()
    assert (c.channel, c.tr38901_scenario, c.blockage_model, c.rician_mode, c.freq_corr_mode) == \
        ("tr38901", "InF-SH", "screen", "los", "los")
    assert c.n_cells == 1 and c.gnb_antenna == "isotropic" and c.scheduler == "pf" and not (c.rach or c.drx)
    assert c.carrier_ghz == 3.5 and c.bandwidth_mhz == 20 and c.duplex == "tdd"


def test_multicell_presets():
    f = factory_inf()
    assert (f.tr38901_scenario, f.n_cells, f.gnb_antenna, f.rlf, f.ul_pc_on) == ("InF-DH", 3, "sector", True, True)
    assert f.cell_azimuth_deg is not None and len(f.cell_azimuth_deg) == 3
    assert not factory_inf(n_cells=1).rlf                    # rlf needs the multi-cell association
    make_engine("L2", E, R, "cpu", factory_inf(n_cells=1), seed=0)
    assert factory_inf(cell_azimuth_deg=(0.0, 0.0, 0.0)).cell_azimuth_deg == (0.0, 0.0, 0.0)
    c = outdoor_campus()
    assert (c.tr38901_scenario, c.blockage, c.blockage_model, c.rlf) == ("UMi", True, "stochastic", True)


def test_urllc_composes_on_a_base():
    u = urllc_control()
    assert (u.control_step_ms, u.ul_mini_slot_symbols, u.mini_slot_dl, u.scheduler, u.qos_classes) == \
        (10.0, 2, True, "qos", 2)
    w = urllc_control(warehouse_private_5g(), qos_pdb_ms=(5.0, 150.0))
    assert w.tr38901_scenario == "InF-SH" and w.scheduler == "qos" and w.qos_pdb_ms == (5.0, 150.0)
    net = make_engine("L2", E, R, "cpu", w, seed=0)
    net.submit(None, Requests(torch.ones(E, R, dtype=torch.long)), priority=1)
    _drive(net, 2)


def test_scenarios_exported():
    import isaac_net
    import isaac_net.core as core
    for name in SCENARIOS:
        assert getattr(isaac_net, name) is SCENARIOS[name] is getattr(core, name)


# ---------------------------------------------------------------------------------------------- (b) describe / diff
DESCRIBE_CONFIGS = [NRConfig(), lena_validation_v2(), multicell(3, rlf=True, dl=True),
                    NRConfig(rach=True, drx=True, traffic=[TrafficModel.periodic(200, period_ms=10)]),
                    NRConfig(edge=EdgeConfig(service_ms=20.0), n_layers_max=2, dl=True)] + \
                   [f() for f in SCENARIOS.values()]


@pytest.mark.parametrize("i", range(len(DESCRIBE_CONFIGS)))
def test_describe_names_every_non_default_field(i):
    cfg = DESCRIBE_CONFIGS[i]
    text = cfg.describe()
    for k in cfg.diff(NRConfig()):
        assert f"\n  {k} " in text, k
    assert "Backends" in text and "triton:" in text and "\x1b" not in text
    assert ("triton: refused" in text) == bool(NRTritonEngine.refusals(cfg))


def test_describe_warns_about_unused_fields_and_reports_the_backend():
    text = NRConfig(pathloss_exp=3.0).describe(level="L0", backend="graph")
    assert "WARNING: level L0 ignores 1 field(s) set here: pathloss_exp" in text
    assert "requested backend='graph'" in text
    assert "Unused fields: none" in warehouse_private_5g().describe()
    assert "triton: yes" in NRConfig().describe()


def test_diff():
    a, b = NRConfig(mu=0, dl=True), NRConfig()
    assert a.diff(b) == {"mu": (0, 1), "dl": (True, False)}
    assert b.diff(a) == {"mu": (1, 0), "dl": (False, True)}
    assert a.diff(a) == {}
    with pytest.raises(TypeError):
        a.diff({"mu": 0})


# ---------------------------------------------------------------------------------------------- (c) backend="auto"
def test_auto_backend_on_cpu_is_reference_and_logged(caplog):
    with caplog.at_level(logging.INFO, logger="isaac_net"):
        net = make_engine("L2", E, R, "cpu", NRConfig(), backend="auto", seed=0)
    assert type(net).__name__ == "NREngine"
    assert any("backend=auto -> reference: device cpu is not CUDA" in r.getMessage() for r in caplog.records)
    ref = make_engine("L2", E, R, "cpu", NRConfig(), backend="reference", seed=0)
    a, b = _drive(net, pos=False), _drive(ref, pos=False)
    assert all(torch.equal(a[k].nan_to_num(-7), b[k].nan_to_num(-7)) for k in a)
    assert inspect.signature(make_engine).parameters["backend"].default == "reference"


def test_auto_backend_wrappers_and_sharded(caplog):
    cfg = NRConfig(energy=EnergyConfig(), edge=EdgeConfig())
    with caplog.at_level(logging.INFO, logger="isaac_net"):
        net = make_engine("L2", E, R, "cpu", cfg, backend="auto", seed=0)
    assert type(net).__name__ == "EnergyLoop"
    assert sum("backend=auto" in r.getMessage() for r in caplog.records) == 1
    from isaac_net.core import ShardedEngine
    sh = ShardedEngine("L2", 4, R, ["cpu", "cpu"], NRConfig(), backend="auto", seed=0)
    assert all(type(s).__name__ == "NREngine" for s in sh.shards)


def test_auto_backend_selection_on_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(engine_mod.importlib.util, "find_spec", lambda name: object())
    rb = engine_mod.resolve_backend
    assert rb("L2", NRConfig(), "cuda") == ("triton", "triton accepts this config")
    b, why = rb("L2", multicell(3), "cuda")
    assert b == "graph" and why.startswith("triton refuses several cells")
    assert rb("L2", warehouse_private_5g(), "cuda")[0] == "graph"
    assert rb("L2", NRConfig(rng="global"), "cuda")[0] == "reference"
    assert rb("L2", NRConfig(dl=True, edge=EdgeConfig(return_path="nr_dl")), "cuda")[0] == "reference"
    assert rb("L2", NRConfig(dl=True, background=BackgroundConfig(dl_load_frac=0.2)), "cuda")[0] == "graph"
    assert rb("L2-legacy", multicell(3), "cuda")[0] == "reference"
    assert rb("L1", NRConfig(), "cuda")[0] == "graph"
    assert rb("L2", NRConfig(), "cpu")[0] == "reference"
    monkeypatch.setattr(engine_mod.importlib.util, "find_spec", lambda name: None)
    assert rb("L2", NRConfig(), "cuda") == ("graph", "the triton package is not installed")


# ---------------------------------------------------------------------------------------------- (d) output_schema
BG = BackgroundConfig(n_background=2, traffic=(TrafficModel.periodic(500, period_ms=20),))
SCHEMA_CONFIGS = {
    "defaults": (NRConfig(), False),
    "dl": (NRConfig(dl=True), False),
    "cells_rlf": (multicell(3, rlf=True, dl=True), True),
    "rach_drx": (NRConfig(rach=True, drx=True, rach_initial="idle"), False),
    "dl_traffic": (NRConfig(dl=True, traffic=[TrafficModel.periodic(200, period_ms=10).downlink(),
                                              TrafficModel.periodic(300, period_ms=20)]), False),
    "mimo": (NRConfig(dl=True, n_layers_max=2, ul_mimo=True), False),
    "minislot": (NRConfig(ul_mini_slot_symbols=2, dl=True, mini_slot_dl=True), False),
    "wrappers": (NRConfig(edge=EdgeConfig(), energy=EnergyConfig(), background=BG), False),
    "obstacles": (NRConfig(channel="tr38901_inf_sh", blockage=True, blockage_model="screen"), True),
    "bg_dl": (NRConfig(dl=True, background=BackgroundConfig(
        n_background=1, traffic=(TrafficModel.periodic(300, period_ms=20),),
        dl_traffic=(TrafficModel.periodic(500, period_ms=20).downlink(),))), False),
    "warehouse": (warehouse_private_5g(), True),
}


def _check_schema(net, out):
    sch = net.output_schema()
    assert list(sch) == list(dict.fromkeys(sch)), "duplicate keys"
    assert set(out) == set(sch), (sorted(set(out) - set(sch)), sorted(set(sch) - set(out)))
    for k, v in out.items():
        e = sch[k]
        assert set(e) == {"shape", "dtype", "unit", "doc", "when"} and e["doc"].endswith(".")
        assert len(e["shape"].strip("[]").split(",")) == v.dim(), (k, e["shape"], tuple(v.shape))
        assert str(v.dtype).replace("torch.", "") == e["dtype"], (k, v.dtype, e["dtype"])


@pytest.mark.parametrize("name", list(SCHEMA_CONFIGS))
def test_output_schema_matches_step_keys_l2(name):
    cfg, pos = SCHEMA_CONFIGS[name]
    net = make_engine("L2", E, R, "cpu", cfg, seed=0)
    net.reset()
    _check_schema(net, _drive(net, 3, pos=pos))


def test_output_schema_after_submit_extras():
    net = make_engine("L2", E, R, "cpu", NRConfig(), seed=0)
    assert "tag" not in net.output_schema()
    net.submit(None, Requests(torch.ones(E, R, dtype=torch.long)), tag=3)
    out = net.step(None, torch.full((E, R), 12.0))
    _check_schema(net, out)
    assert net.output_schema()["tag"]["when"].startswith("after submit")


@pytest.mark.parametrize("level", ["L0", "L0DR", "L05", "L05Q", "L1", "L2-legacy", "ORACLE", "NOCOMM"])
def test_output_schema_legacy_levels(level):
    p = default_params("L2" if level == "L2-legacy" else level)
    net = make_engine(level, E, R, "cpu", None, params=p, seed=0)
    _check_schema(net, _drive(net, 3, pos=False))


def test_output_schema_netslotmc_wifi_bgload_adaptive():
    net = make_engine("L2-legacy", E, R, "cpu", multicell(3), seed=0)
    _check_schema(net, _drive(net, 3, pos=True))
    net = make_engine("WIFI", E, R, "cpu", NRConfig(), seed=0)
    _check_schema(net, _drive(net, 3, pos=True))
    net = make_engine("L1", E, R, "cpu", NRConfig(background=BG), seed=0)
    _check_schema(net, _drive(net, 3, pos=False))
    from isaac_net.core.adaptive import FidelityConfig, make_adaptive
    net = make_adaptive(E, R, "cpu", NRConfig(), FidelityConfig())
    _check_schema(net, _drive(net, 3, pos=False))


def test_schema_registry_and_docs_table():
    assert {k for g in GROUPS.values() for k in g} == set(STEP_KEYS)
    path = os.path.join(ROOT, "docs", "configurability.md")
    if not os.path.exists(path):
        pytest.skip("docs/ is not shipped with the package (source tree only)")
    assert markdown_table() in open(path, encoding="utf-8").read(), \
        "regenerate the Output schema table: python -c 'from isaac_net.core.schema import markdown_table; " \
        "print(markdown_table())'"


# ------------------------------------------------------------------------------------------ (e) unused-field warning
@pytest.fixture
def fresh_warnings(monkeypatch):
    monkeypatch.setattr(engine_mod, "_WARNED_UNUSED", set())


def test_unused_fields_warn_once_per_config(fresh_warnings):
    cfg = NRConfig(olla_up_db=0.1)
    with pytest.warns(UnusedFieldsWarning, match="level L2-legacy ignores .*olla_up_db"):
        make_engine("L2-legacy", E, R, "cpu", cfg, seed=0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        make_engine("L2-legacy", E, R, "cpu", NRConfig(olla_up_db=0.1), seed=0)      # same config: no repeat
        make_engine("L2-legacy", E, R, "cpu", NRConfig(olla_up_db=0.2), strict=None, seed=0)
        make_engine("L2", E, R, "cpu", cfg, seed=0)                                    # L2 reads it
    with pytest.warns(UnusedFieldsWarning):
        make_engine("L2-legacy", E, R, "cpu", NRConfig(olla_up_db=0.3), seed=0)      # another value: warns
    with pytest.raises(ValueError, match="ignores these config fields"):
        make_engine("L2-legacy", E, R, "cpu", cfg, strict=True, seed=0)


def test_unused_fields_warn_wifi(fresh_warnings):
    with pytest.warns(UnusedFieldsWarning, match="level WIFI ignores .*olla_up_db"):
        make_engine("WIFI", E, R, "cpu", NRConfig(olla_up_db=0.1), seed=0)


# ---------------------------------------------------------------------------------------------- (f) defaults unchanged
def test_existing_presets_unchanged():
    assert set(lena_validation_v2().diff(NRConfig())) == {
        "n_prb", "rbg_size", "ul_data_symbols", "dmrs_re_per_prb", "harq_fail", "discard", "bler_source", "tbs_mode",
        "harq_combining", "olla", "pf_metric", "pf_update", "pf_avg_idle", "ul_retx_sched", "ul_amc_alloc",
        "ul_grant_model", "phr_cap", "ul_power", "fading", "noise_model", "gnb_nf_db", "tb_overhead_bytes",
        "pkt_overhead_bytes", "frame_buffer"}
    assert NRConfig().diff(NRConfig()) == {}


def test_default_engine_bitwise_through_the_new_paths(fresh_warnings):
    from isaac_net.core.engine import NREngine
    a = make_engine("L2", E, R, "cpu", NRConfig(), seed=5)
    b = NREngine(E, R, "cpu", NRConfig(), seed=5)
    c = make_engine("L2", E, R, "cpu", NRConfig(), strict=None, seed=5)
    oa, ob, oc = (_drive(n, 4, pos=False, seed=3) for n in (a, b, c))
    for k in oa:
        assert torch.equal(oa[k].nan_to_num(-7), ob[k].nan_to_num(-7)) and torch.equal(oa[k].nan_to_num(-7),
                                                                                       oc[k].nan_to_num(-7)), k
