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
