"""Config files, presets by name, Hydra overrides of the network fields, and the net/ KPIs in extras["log"].

1. NRConfig.to_dict / from_dict / to_json / from_json / to_yaml / from_yaml round-trip every preset and a config with
   every nested wrapper and traffic model exactly (dataclass equality), and __post_init__ leaves a loaded config as
   it is (with_() == itself). Unknown keys and preset names raise with "did you mean"; lists become tuples and the
   "inf" strings floats where the field annotation says so.
2. The generated YAML copies of the presets (isaac_net/core/presets) match scripts/export_presets.py output, load
   from importlib.resources, and ship as package data.
3. IsaacNetCfg.nr: the override dict Isaac Lab's env. CLI prefix fills (its _setattr semantics) and that survives
   an OmegaConf round trip, materialised with NRConfig.from_dict.
4. KPIs: the Direct mixin and the manager runtime put the step's net/ KPIs into extras["log"] as 0-dim tensors,
   equal to a direct computation from the step dict (and the NR MAC counters at level L2).
"""
import ast
import importlib.util
import json
import math
import os
from types import SimpleNamespace

import pytest
import torch

from isaac_net import NRConfig
from isaac_net.core import BackgroundConfig, EdgeConfig, EnergyConfig, TrafficModel
from isaac_net.core.adaptive import FidelityConfig
from isaac_net.core.config import UnknownFieldError
from isaac_net.core.presets import FILE_PRESETS, PRESETS, preset_function, preset_path, preset_yaml
from isaac_net.core.wifi.config import WifiConfig
from isaac_net.isaac import IsaacNetCfg
from isaac_net.isaac.kpis import KPI_KEYS, MAC_KPI_KEYS, step_kpis
from isaac_net.isaac.mixins import NetEnvMixin

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _everything():
    tm = (TrafficModel.periodic(200, period_ms=10).on(robots=[0, 1]),
          TrafficModel.video(fps=30, gop=(40_000, 8_000, 15), priority=1, deadline_ms=50).on(robots=[2]),
          TrafficModel.bursty(1400, rate_hz=50, burst_size=4, on_off=(0.5, 2.0)),
          TrafficModel.event(4000, trigger="alarm", det=True),
          TrafficModel.periodic(300, period_ms=20).downlink(), TrafficModel.policy())
    return NRConfig(traffic=tm, dl=True, edge=EdgeConfig(service_ms=(5.0, 20.0), deadline_ms=80.0),
                    background=BackgroundConfig(n_background=3, arena_m=(0, 0, 100, 100),
                                                dl_traffic=(TrafficModel.periodic(500, period_ms=10),)),
                    energy=EnergyConfig(initial_soc=(0.5, 1.0)),
                    wifi=WifiConfig(ap_positions_m=((0, 0), (50, 50)), class_ac=("VO", "BE")),
                    fidelity=FidelityConfig(mode="static", active_budget=0.5), ue_speed_mps=3.0, scheduler="qos",
                    qos_pdb_ms=(10.0, math.inf), blocker_size_m=((0.6, 1.5),), cell_tilt_deg=6.0,
                    channel="tr38901_inf_dh", seed=7)


def _round_trips(c):
    for only in (False, True):
        d = c.to_dict(only)
        json.dumps(d, allow_nan=False)                                  # strict JSON
        assert NRConfig.from_dict(d) == c
        assert NRConfig.from_json(c.to_json(only_changed=only)) == c
        assert NRConfig.from_yaml(c.to_yaml(only_changed=only)) == c
    assert c.with_() == c                                               # __post_init__ is idempotent


# ------------------------------------------------------------------------------------------------ 1. serialization
@pytest.mark.parametrize("name", sorted(PRESETS))
def test_every_preset_round_trips(name):
    pytest.importorskip("yaml")
    c = preset_function(name)()
    _round_trips(c)
    assert NRConfig.from_dict(c.to_dict()).describe() == c.describe()


def test_every_wrapper_and_traffic_model_round_trips(tmp_path):
    pytest.importorskip("yaml")
    c = _everything()
    _round_trips(c)
    d = c.to_dict()
    assert d["qos_pdb_ms"] == [10.0, "inf"] and d["blocker_size_m"] == [[0.6, 1.5]]
    assert d["edge"]["__type__"] == "EdgeConfig" and d["traffic"][1]["kind"] == "video"
    assert d["traffic"][1]["gop"] == [40000.0, 8000.0, 15] and "jitter_ms" not in d["traffic"][1]
    c.to_yaml(tmp_path / "c.yaml")
    c.to_json(tmp_path / "c.json")
    assert NRConfig.from_yaml(tmp_path / "c.yaml") == c == NRConfig.from_json(str(tmp_path / "c.json"))
    assert "fading_rho_per_ms" in c.describe()


def test_coercion_and_hand_written_files():
    c = NRConfig.from_dict({"msg_sizes": [100, 200], "qos_pdb_ms": ["inf", 5], "cell_positions_m": [[1, 2]],
                            "background": {"n_background": 2, "traffic": [{"kind": "periodic", "size_bytes": 10.0,
                                                                          "period_ms": 5.0}]},
                            "traffic": {"kind": "periodic", "size_bytes": 50.0, "period_ms": 20.0}})
    assert c.msg_sizes == (100, 200) and c.qos_pdb_ms == (math.inf, 5.0) and c.cell_positions_m == ((1, 2),)
    assert c.background == BackgroundConfig(n_background=2, traffic=(TrafficModel.periodic(10, period_ms=5),))
    assert c.traffic == (TrafficModel.periodic(50, period_ms=20),)
    assert NRConfig.from_yaml("ul_tpc: true\nul_pc: true\nmsg_sizes: [1000, 2000]\n" if _has_yaml() else
                              '{"ul_tpc": true, "ul_pc": true, "msg_sizes": [1000, 2000]}') == \
        NRConfig(ul_tpc=True, ul_pc=True, msg_sizes=(1000, 2000))


def _has_yaml():
    return importlib.util.find_spec("yaml") is not None


def test_unknown_keys_suggest_the_field():
    with pytest.raises(UnknownFieldError, match=r"no field 'ul_tcp' \(did you mean 'ul_tpc'"):
        NRConfig.from_dict({"ul_tcp": True})
    with pytest.raises(UnknownFieldError, match=r"background: BackgroundConfig has no field 'n_backgrund' \(did you "
                                                r"mean 'n_background'"):
        NRConfig.from_dict({"background": {"n_backgrund": 2}})
    with pytest.raises(UnknownFieldError, match="did you mean 'warehouse_private_5g'"):
        NRConfig.from_preset("warehouse_private5g")
    with pytest.raises(UnknownFieldError, match="did you mean 'EdgeConfig'"):
        NRConfig.from_dict({"edge": {"__type__": "EdgConfig"}})
    with pytest.raises(UnknownFieldError, match="did you mean 'size_bytes'"):
        NRConfig.from_dict({"traffic": [{"kind": "periodic", "size_byte": 1.0}]})
    with pytest.raises(ValueError, match="expected __type__ 'EdgeConfig', got 'EnergyConfig'"):
        NRConfig.from_dict({"edge": {"__type__": "EnergyConfig"}})


def test_unserializable_values_raise():
    with pytest.raises(TypeError, match="callable trigger"):
        NRConfig(traffic=TrafficModel.event(100, trigger=lambda clock: clock > 0)).to_dict()
    with pytest.raises(TypeError, match="cannot serialize"):
        NRConfig(fidelity=FidelityConfig(cheap_params=object())).to_dict()


def test_from_preset_overrides_and_preset_key():
    assert NRConfig.from_preset("srsran_like", mu=0) == preset_function("srsran_like")(mu=0)   # derived timers follow
    assert NRConfig.from_preset("factory_inf", n_cells=2, msg_sizes=[10, 20]) == \
        preset_function("factory_inf")(n_cells=2, msg_sizes=(10, 20))
    assert NRConfig.from_preset("lena_match") == preset_function("lena_like")()
    base = NRConfig(frame_buffer=8)
    assert NRConfig.from_dict({"ul_tpc": True, "ul_pc": True}, base=base) == base.with_(ul_tpc=True, ul_pc=True)
    assert NRConfig.from_dict({"preset": "warehouse_private_5g", "frame_buffer": 16}, base=base) == \
        preset_function("warehouse_private_5g")(frame_buffer=16)


# ------------------------------------------------------------------------------------------------ 2. preset files
@pytest.mark.parametrize("name", FILE_PRESETS)
def test_preset_files_are_generated_and_load(name):
    pytest.importorskip("yaml")
    text = preset_path(name).read_text(encoding="utf-8")
    assert text == preset_yaml(name), f"{name}.yaml is stale: run python scripts/export_presets.py"
    assert text.startswith("# Generated by scripts/export_presets.py from isaac_net.core.")
    assert NRConfig.from_yaml(preset_path(name)) == preset_function(name)()


def test_export_script_check_and_package_data():
    pytest.importorskip("yaml")
    spec = importlib.util.spec_from_file_location("_export_presets", os.path.join(ROOT, "scripts", "export_presets.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.main(["--check"]) == 0
    pyproject = os.path.join(ROOT, "pyproject.toml")
    if not os.path.exists(pyproject):
        pytest.skip("no pyproject.toml next to the tests")
    with open(pyproject, encoding="utf-8") as fh:
        assert '"isaac_net.core.presets" = ["*.yaml"]' in fh.read()


# ------------------------------------------------------------------------------------------------ 3. Hydra
def _isaaclab_setattr(cfg, arg):
    """isaaclab_tasks.utils.hydra (Isaac Lab 3.0): an env.-prefixed override is applied to the cfg object itself with
    _setattr(cfgs["env"], path, _parse_val(val)), not through Hydra's struct config (copied here: no Isaac Lab)."""
    key, val = arg.split("=", 1)
    assert key.startswith("env.")
    lit = {"true": True, "false": False, "none": None, "null": None}
    if val.lower() in lit:
        val = lit[val.lower()]
    else:
        try:
            val = ast.literal_eval(val)
        except (ValueError, SyntaxError):
            pass
    *parts, leaf = key.split(".", 1)[1].split(".")
    obj = cfg
    for p in parts:
        obj = obj[p] if isinstance(obj, dict) else getattr(obj, p)
    if isinstance(obj, dict):
        obj[leaf] = val
    else:
        setattr(obj, leaf, val)


def test_isaac_cfg_nr_overrides():
    base = NRConfig(msg_sizes=(4000.0, 30000.0), frame_buffer=16)
    assert IsaacNetCfg().resolve_nr(base) is base and IsaacNetCfg().resolve_nr(None) is None
    assert IsaacNetCfg(nr={"ul_tpc": True, "ul_pc": True}).resolve_nr(base) == base.with_(ul_tpc=True, ul_pc=True)
    assert IsaacNetCfg(nr={"preset": "lena_like"}).resolve_nr(base) == preset_function("lena_like")()
    assert IsaacNetCfg(nr=NRConfig(mu=0)).resolve_nr(base) == NRConfig(mu=0)
    with pytest.raises(UnknownFieldError, match="did you mean 'frame_buffer'"):
        IsaacNetCfg(nr={"frame_bufer": 4})
    with pytest.raises(TypeError):
        IsaacNetCfg(nr=[("mu", 0)])

    env = SimpleNamespace(net_isaac=IsaacNetCfg(pose_asset="robots"))
    for arg in ("env.net_isaac.nr.ul_tpc=true", "env.net_isaac.nr.ul_pc=true", "env.net_isaac.nr.msg_sizes=[100,200]",
                "env.net_isaac.nr.ue_speed_mps=1.5", "env.net_isaac.log_every=4"):
        _isaaclab_setattr(env, arg)
    want = base.with_(ul_tpc=True, ul_pc=True, msg_sizes=(100, 200), ue_speed_mps=1.5)
    assert env.net_isaac.resolve_nr(base) == want and env.net_isaac.log_every == 4
    _isaaclab_setattr(env, "env.net_isaac.nr.preset=warehouse_private_5g")
    assert env.net_isaac.resolve_nr(base) == preset_function("warehouse_private_5g")(
        ul_tpc=True, ul_pc=True, msg_sizes=(100, 200), ue_speed_mps=1.5)
    _isaaclab_setattr(env, "env.net_isaac.nr.ul_tcp=true")          # set after construction: raises when resolved
    with pytest.raises(UnknownFieldError, match="did you mean 'ul_tpc'"):
        env.net_isaac.resolve_nr(base)


def test_isaac_cfg_omegaconf_round_trip():
    """The fleet cfg's network part through OmegaConf (what Isaac Lab's Hydra path does with the env cfg dict) with an
    override applied, back into an IsaacNetCfg whose nr dict materialises to the overridden NRConfig."""
    omegaconf = pytest.importorskip("omegaconf")
    from dataclasses import asdict
    isc = IsaacNetCfg(pose_asset="robots", gnb_pos=((0.0, 0.0, 6.0),), nr={"preset": "factory_inf", "n_cells": 2})
    conf = omegaconf.OmegaConf.create({"env": {"net_isaac": asdict(isc)}})
    conf = omegaconf.OmegaConf.merge(conf, omegaconf.OmegaConf.from_dotlist(["env.net_isaac.nr.ul_tpc=true",
                                                                            "env.net_isaac.nr.qos_pdb_ms=[10,inf]"]))
    back = IsaacNetCfg(**omegaconf.OmegaConf.to_container(conf, resolve=True)["env"]["net_isaac"])
    assert back.resolve_nr(NRConfig()) == preset_function("factory_inf")(n_cells=2, ul_tpc=True,
                                                                         qos_pdb_ms=(10.0, math.inf))


# ------------------------------------------------------------------------------------------------ 4. KPIs
class _Asset:
    def __init__(self, p):
        self.data = SimpleNamespace(body_link_pos_w=p)


class _Scene:
    def __init__(self, n, p):
        self.num_envs, self.env_origins, self.ents = n, torch.zeros(n, 3), {"robots": _Asset(p)}

    def __getitem__(self, k):
        return self.ents[k]


def _env(n=3, r=4, step_dt=0.1):
    class Env(NetEnvMixin):
        num_envs, device = n, "cpu"
    env = Env()
    env.step_dt = step_dt
    env.scene = _Scene(n, torch.rand(n, r, 3) * 60)
    return env


def _expected(out):
    dlv, lost = out["msg_delivered"], out["timed_out"]
    if "dropped" in out:
        lost = lost | out["dropped"]
    nd, nl = float(dlv.sum()), float(lost.sum())
    aoi = out["aoi_s"].double().flatten()
    d = out["delay_s"].double()[dlv]
    return {"net/aoi_mean_s": float(aoi.mean()), "net/aoi_p95_s": float(torch.quantile(aoi, 0.95)),
            "net/delay_mean_ms": float(d.sum() * 1000 / nd) if nd else 0.0,
            "net/delivered_frac": nd / (nd + nl) if nd + nl else 0.0,
            "net/dropped_frac": nl / (nd + nl) if nd + nl else 0.0,
            "net/queue_bytes_mean": float(out["queue_bytes"].double().mean()),
            "net/sinr_mean_db": float(out["sinr_db"].double().mean())}


def test_direct_mixin_logs_kpis(seeded):
    env = _env()
    env.net_setup("L2-legacy", 4, NRConfig(timeout_steps=2, frame_buffer=4), "reference",
                  isaac=IsaacNetCfg(pose_asset="robots"))
    seen, resolved = [], 0
    for t in range(12):
        out = env.net_step(send=torch.randint(0, 3, (3, 4)))
        log = env.extras["log"]
        assert set(KPI_KEYS) <= set(log) and not set(MAC_KPI_KEYS) & set(log)
        for k, v in _expected(out).items():
            assert log[k].dim() == 0 and log[k].dtype == torch.float32
            assert float(log[k]) == pytest.approx(v, rel=1e-5, abs=1e-6), (t, k)
        seen.append(log)
        resolved += int(out["msg_delivered"].sum() + out["timed_out"].sum())
    assert len({id(x) for x in seen}) == len(seen)              # a new dict per step: rsl_rl keeps the references
    assert resolved > 0 and any(float(x["net/delivered_frac"]) > 0 for x in seen)
    assert any(float(x["net/dropped_frac"]) > 0 for x in seen)


def test_kpi_switches(seeded):
    env = _env()
    env.net_setup("L1", 4, NRConfig(), "reference", isaac=IsaacNetCfg(pose_asset="robots", log_kpis=False))
    env.net_step(send=torch.ones(3, 4, dtype=torch.long))
    assert "log" not in getattr(env, "extras", {})
    env = _env()
    env.net_setup("L1", 4, NRConfig(), "reference", isaac=IsaacNetCfg(pose_asset="robots", log_every=3))
    vals = []
    for _ in range(6):
        out = env.net_step(send=torch.ones(3, 4, dtype=torch.long))
        vals.append((float(env.extras["log"]["net/aoi_mean_s"]), float(out["aoi_s"].mean())))
    assert vals[1][0] == vals[0][0] == pytest.approx(vals[0][1]) and vals[3][0] == pytest.approx(vals[3][1])
    # net_decimation: the KPIs follow network steps, not the held output of the skipped env steps
    env = _env(step_dt=0.05)
    env.net_setup("L1", 4, NRConfig(), "reference", isaac=IsaacNetCfg(pose_asset="robots", net_decimation=2))
    out = env.net_step(send=torch.ones(3, 4, dtype=torch.long))
    first = env.extras["log"]
    env.net_step(send=torch.ones(3, 4, dtype=torch.long))
    assert env.extras["log"] is first and float(first["net/aoi_mean_s"]) == pytest.approx(float(out["aoi_s"].mean()))


def test_nr_engine_mac_kpis(seeded):
    env = _env(n=2, r=2)
    cfg = NRConfig(drx=True, drx_cycle_ms=40.0, drx_on_ms=10.0, drx_inactivity_ms=10.0, control_step_ms=20.0)
    env.step_dt = 0.02
    env.net_setup("L2", 2, cfg, "reference", isaac=IsaacNetCfg(pose_asset="robots"))
    from isaac_net.core.record import mac_counters, mac_links
    links = {"ul": mac_links(env.net.eng)["ul"]}
    for t in range(4):
        before = mac_counters(links)
        out = env.net_step(send=torch.full((2, 2), 2, dtype=torch.long))
        after = mac_counters(links)
        log = env.extras["log"]
        assert set(KPI_KEYS) | set(MAC_KPI_KEYS) | {"net/access_sleep_frac"} <= set(log)
        d = {k: after[k] - before[k] for k in after}
        assert float(log["net/prb_util"]) == pytest.approx(float(d["ul_prb"].sum() / d["ul_avail"].sum()), rel=1e-5)
        tx = float(d["ul_rvtx"][:, 1].sum())
        assert float(log["net/harq_bler"]) == pytest.approx(float(d["ul_rvfail"][:, 1].sum()) / tx if tx else 0.0)
        assert float(log["net/access_sleep_frac"]) == pytest.approx(float(out["access_sleep_frac"].mean()), rel=1e-5)
    assert float(log["net/prb_util"]) > 0
    env.net_reset(None)                                         # a full reset zeroes the counters: no negative diff
    env.net_step(send=torch.ones(2, 2, dtype=torch.long))
    assert 0.0 <= float(env.extras["log"]["net/prb_util"]) <= 1.0


def test_step_kpis_dropped_and_empty():
    E, R, F = 2, 3, 4
    z = torch.zeros(E, R, F, dtype=torch.bool)
    out = dict(aoi_s=torch.arange(6.0).view(E, R), msg_delivered=z, timed_out=z, dropped=z.clone(),
               delay_s=torch.full((E, R, F), float("nan")), queue_bytes=torch.ones(E, R), sinr_db=torch.zeros(E, R),
               rlf=torch.tensor([[True, False, False], [False, False, False]]))
    k = step_kpis(out)
    assert float(k["net/delay_mean_ms"]) == 0.0 and float(k["net/delivered_frac"]) == 0.0
    assert float(k["net/rlf_frac"]) == pytest.approx(1 / 6)
    out["dropped"][0, 0, 0] = True
    out["msg_delivered"] = z.clone()
    out["msg_delivered"][1, 1, :2] = True
    out["delay_s"][1, 1, :2] = torch.tensor([0.01, 0.03])
    k = step_kpis(out)
    assert float(k["net/dropped_frac"]) == pytest.approx(1 / 3) and float(k["net/delay_mean_ms"]) == pytest.approx(20)
    assert float(k["net/aoi_p95_s"]) == pytest.approx(float(torch.quantile(torch.arange(6.0), 0.95)))


def test_manager_runtime_kpis_survive_reset(seeded):
    from isaac_net.isaac.manager_cfg import NetManagerCfg
    from isaac_net.isaac.runtime import get_runtime

    class Env:
        """ManagerBasedRLEnv's extras handling: _reset_idx replaces extras["log"] with a new dict."""
        num_envs, device, step_dt, common_step_counter = 3, "cpu", 0.1, 0

        def __init__(self):
            self.scene = _Scene(3, torch.rand(3, 4, 3) * 40)
            self.cfg = SimpleNamespace(isaac_net=NetManagerCfg(level="L1", backend="reference", seed=1,
                                                               nr=NRConfig(control_step_ms=100.0),
                                                               isaac=IsaacNetCfg(pose_asset="robots")))
            self.extras = {}

        def _reset_idx(self, env_ids):
            self.extras["log"] = {"Episode_Reward/x": torch.tensor(1.0)}

    env = Env()
    rt = get_runtime(env)
    env.common_step_counter = 1
    out = rt.step()
    assert float(env.extras["log"]["net/aoi_mean_s"]) == pytest.approx(float(out["aoi_s"].mean()))
    env._reset_idx(torch.tensor([0]))
    assert "Episode_Reward/x" in env.extras["log"] and set(KPI_KEYS) <= set(env.extras["log"])
