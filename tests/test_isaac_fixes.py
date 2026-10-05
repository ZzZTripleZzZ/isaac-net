"""Regression tests for the Isaac layer / MJX / bench fixes of the 2026-10-04 code review (section D)."""
import json

import pytest
import torch

from isaac_net import NRConfig
from isaac_net.isaac import IsaacNetCfg
from isaac_net.isaac.mixins import NetEnvMixin


def _env(step_dt, n=2):
    class Env(NetEnvMixin):
        num_envs, device = n, "cpu"
    env = Env()
    env.step_dt = step_dt
    return env


def test_decimation_keeps_tag_on_equal_class(seeded):
    # network step 100 ms, env step 50 ms (k = 2): the network steps on env ticks 1, 3, 5, ...
    env = _env(0.05)
    env.net_setup("ORACLE", 1, NRConfig(), "reference", isaac=IsaacNetCfg(net_decimation=2))
    pos = torch.zeros(2, 1, 3)
    one = torch.ones(2, 1, dtype=torch.long)
    cur = torch.tensor([5, 5])
    env.net_step(pos, one, cur_tag=cur)                                  # tick 1: network step
    env.net_step(pos, one, torch.full_like(one, -1), cur_tag=cur)        # tick 2: untagged class-1 frame
    out = env.net_step(pos, one, torch.full_like(one, 5), cur_tag=cur)   # tick 3: tagged class-1 frame, steps
    assert bool(out["delivered"].all())
    assert bool(out["tag_delivered"].all())                              # was False: the tag was dropped
    # a larger class still wins over a tag, and a tagged pending message is not replaced by an untagged one
    env.net_step(pos, one, torch.full_like(one, 5), cur_tag=cur)         # tick 4: tagged
    out = env.net_step(pos, one, torch.full_like(one, -1), cur_tag=cur)  # tick 5: untagged, same class
    assert bool(out["tag_delivered"].all())


def test_decimation_held_output_ages_and_resets(seeded):
    k = 4
    env = _env(0.1 / k)
    env.net_setup("ORACLE", 1, NRConfig(), "reference", isaac=IsaacNetCfg(net_decimation=k))
    pos = torch.zeros(2, 1, 3)
    zero = torch.zeros(2, 1, dtype=torch.long)
    out = env.net_step(pos, zero)                                        # tick 1: network step
    a0 = out["aoi_s"].clone()
    for j in range(1, k):                                                # ticks 2..k: held, one env step older
        out = env.net_step(pos, zero)
        assert torch.allclose(out["aoi_s"], a0 + j * 0.1 / k)
        assert not bool(out["delivered"].any())
    env.net_step(pos, zero)                                              # tick k + 1: network step
    out = env.net_step(pos, zero)                                        # tick k + 2: held
    env.net_reset(torch.tensor([0]))
    held = env.net_out
    assert float(held["aoi_s"][0, 0]) == 0.0 and int(held["last_cap"][0, 0]) == 0
    assert int(held["queue_len"][0].sum()) == 0
    assert torch.equal(held["aoi_s"][1], out["aoi_s"][1])                # other envs untouched
    out = env.net_step(pos, zero)                                        # tick k + 3: held, reset env ages
    assert torch.allclose(out["aoi_s"][0], torch.tensor([0.1 / k]))


# ---------------------------------------------------------------------------------------------- bench (items 23, 25)
def _res(seed=0, label="", **cfg_kw):
    from isaac_net.bench import TaskConfig
    cfg = TaskConfig(seed=seed, **cfg_kw)
    return {"schema": "isaac-net-bench/1", "label": label, "task": cfg.task, "variant": cfg.variant,
            "level": cfg.level, "backend": cfg.backend, "sim": cfg.sim, "preset": cfg.preset, "traffic": cfg.traffic,
            "baseline": "random", "seed": seed, "config": cfg.describe(),
            "task_spec": {"metric": {"key": "hazard_exposure", "higher_is_better": False}},
            "train": None, "eval": {"envs": cfg.num_envs, "episodes": 1, "metrics": {"hazard_exposure": 0.1 + seed}},
            "timing": {"train_sec": 0.0, "eval_sec_per_step": 0.01}}


def test_result_name_separates_runs():
    from isaac_net.bench.runner import result_name
    a = _res(preset="default", traffic="policy", sim="torch")
    b = _res(preset="multicell3", traffic="policy+telemetry", sim="isaac")
    assert result_name(a) != result_name(b)
    assert "__torch__default__policy__" in result_name(a)
    assert "__isaac__multicell3__policy+telemetry__" in result_name(b)
    # settings outside the name go through the hash; the name is stable for the same settings
    assert result_name(_res(num_envs=64)) != result_name(_res(num_envs=128))
    assert result_name(_res(episode_steps=50)) != result_name(_res(episode_steps=None))
    assert result_name(_res(net_obs=("aoi",))) != result_name(_res())
    assert result_name(_res()) == result_name(_res())
    assert result_name(_res(label="x")).endswith("__x.json")


def test_report_groups_by_sim_and_reads_old_files(tmp_path):
    from isaac_net.bench.report import aggregate, load_results, markdown
    files = [_res(0, sim="torch"), _res(1, sim="torch"), _res(0, sim="isaac")]
    old = _res(2, sim="torch")
    del old["sim"]                                           # a file from before sim was a group key
    for i, r in enumerate(files + [old]):
        (tmp_path / f"r{i}.json").write_text(json.dumps(r))
    rows = aggregate(load_results([str(tmp_path)]))
    by_sim = {r["sim"]: r for r in rows}
    assert set(by_sim) == {"torch", "isaac"}
    assert by_sim["torch"]["seeds"] == [0, 1, 2] and by_sim["isaac"]["seeds"] == [0]
    md = markdown(rows)
    assert "| sim |" in md.splitlines()[0]                  # the column appears because sim differs between rows


def test_cli_progress_line_formats_missing_metric():
    from isaac_net.bench.cli import _num
    assert _num(None, ".4g") == "n/a" and _num(0.123456, ".3g") == "0.123"


def test_isaac_adapter_reads_env_inside_indicator():
    # the adapter must report the env's own per-step indicator, not recompute it from post-reset state
    from isaac_net.bench.isaac_adapter import IsaacFleetAlert
    from isaac_net.bench.metrics import EpisodeMetrics
    from isaac_net.bench.tasks.fleet_alert import FleetAlert
    E, R = 2, 4

    class FakeEnv:
        net_out = None

        def step(self, act):
            self.last_inside = torch.tensor([[True, True, False, False], [False] * 4])
            # post-step state that the old code mixed in: a reset env with zeroed hazard and moved robots
            self.h_on, self.h_start = torch.tensor([False, True]), torch.zeros(E, dtype=torch.long)
            z = torch.zeros(E, R * 3)
            return {"policy": z}, torch.zeros(E), torch.zeros(E, dtype=torch.bool), torch.tensor([True, False]), {}

        def _pos_radio(self):
            raise AssertionError("the adapter must not recompute exposure from post-step poses")

    task = IsaacFleetAlert.__new__(IsaacFleetAlert)
    task.E, task.R, task.T, task.env, task.dev = E, R, 300, FakeEnv(), torch.device("cpu")
    task.metrics = EpisodeMetrics(E, R, 3, "cpu", 0.1, FleetAlert.MSG_SIZES)
    task.metrics.reset(torch.ones(E, dtype=torch.bool))
    task.t = torch.zeros(E, dtype=torch.long)
    task._send_val = torch.tensor([-1.0, 0.0, 1.0])
    _, _, done, info = task.step(torch.zeros(E, R, 2), torch.zeros(E, R, dtype=torch.long))
    assert bool(done[0]) and len(info["episodes"]) == 1
    assert info["episodes"][0][FleetAlert.METRIC.key] == pytest.approx(0.5)


# ---------------------------------------------------------------------------------------------- MJX (item 25)
def _import_mjx_module_with_stub_jax(monkeypatch):
    """isaac_net.mjx.net_module imports JAX at module level; stub the few names it touches at import time so the
    pure-torch build_module can be tested without JAX."""
    import importlib
    import sys
    import types
    if importlib.util.find_spec("jax") is None:
        jax = types.ModuleType("jax")
        jnp = types.ModuleType("jax.numpy")
        jnp.float32, jnp.bool_, jnp.int32 = "float32", "bool", "int32"
        exp = types.ModuleType("jax.experimental")
        bc = types.ModuleType("jax.experimental.buffer_callback")
        bc.buffer_callback = lambda *a, **k: None
        jax.numpy, jax.experimental, exp.buffer_callback = jnp, exp, bc
        for name, mod in (("jax", jax), ("jax.numpy", jnp), ("jax.experimental", exp),
                          ("jax.experimental.buffer_callback", bc)):
            monkeypatch.setitem(sys.modules, name, mod)
        for name in ("isaac_net.mjx", "isaac_net.mjx.net_module"):
            monkeypatch.delitem(sys.modules, name, raising=False)
    return importlib.import_module("isaac_net.mjx.net_module")


def test_mjx_build_module_leaves_global_rng_alone(monkeypatch):
    import sys
    try:
        nm = _import_mjx_module_with_stub_jax(monkeypatch)
        torch.manual_seed(123)
        before = torch.get_rng_state()
        cuda_before = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        a = nm.build_module("L0", 2, 3, "cpu", NRConfig(), "reference", seed=7)
        assert torch.equal(torch.get_rng_state(), before)
        if cuda_before is not None:
            assert all(torch.equal(x, y) for x, y in zip(torch.cuda.get_rng_state_all(), cuda_before))
        b = nm.build_module("L0", 2, 3, "cpu", NRConfig(), "reference", seed=7)
        assert a.seed == b.seed == 7
    finally:
        for name in ("isaac_net.mjx", "isaac_net.mjx.net_module"):
            sys.modules.pop(name, None)
