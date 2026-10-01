"""Benchmark suite (isaac_net.bench): task API, specs, metrics, variants, baselines, runner, report (CPU, small)."""
import json
import math

import pytest
import torch

from isaac_net.bench import (TASKS, HeuristicPolicy, RandomPolicy, TaskConfig, background_available, evaluate,
                                load_results, make_task, mean_ci95, run, train_ppo, variants, write_result)
from isaac_net.bench.cli import main as cli_main
from isaac_net.bench.metrics import NB, bin_edges_ms, delay_bins, hist_quantile
from isaac_net.bench.report import aggregate, markdown
from isaac_net.bench.runner import calibrate

E, R, T = 3, 4, 12
ROW_KEYS = ("return", "sent", "refused", "deliveries", "drops", "delivery_ratio", "delay_p50_ms", "delay_p95_ms",
            "aoi_mean_s", "offered_mbps", "delivered_mbps")


def cfg(task, level="L0", **kw):
    base = dict(task=task, level=level, backend="reference", num_envs=E, num_robots=R, episode_steps=T, seed=0)
    base.update(kw)
    return TaskConfig(**base)


def rollout(task, policy, steps):
    policy.reset(task)
    obs = task.reset()
    rows, done, trace = [], None, []
    for _ in range(steps):
        cont, send = policy.act(task, obs, done)
        obs, rew, done, info = task.step(cont, send)
        rows += info["episodes"]
        trace.append((obs.clone(), rew.clone()))
    return rows, trace


@pytest.mark.parametrize("name", list(TASKS))
def test_task_api_shapes_and_rows(name):
    task = make_task(cfg(name), "cpu")
    obs = task.reset()
    assert obs.shape == (E, R, task.obs_spec.dim)
    assert task.obs_spec.dim == sum(w for _, w in task.obs_spec.blocks)
    assert [b for b, _ in task.obs_spec.blocks][-4:] == ["net:aoi", "net:sinr", "net:queue_len",
                                                          "net:last_delivered"]
    assert task.action_spec.n_send == len(task.SEND_CHOICES)
    rows, trace = rollout(task, RandomPolicy(0), T)
    assert len(rows) == E                                  # every env finished exactly one episode
    key = task.METRIC.key
    for r in rows:
        for k in ROW_KEYS + (key,):
            assert k in r
        assert r["sent"] > 0
        assert all(f"sent_{c}" in r for c in task.SEND_CHOICES)
        assert math.isfinite(r[key])
    for o, rew in trace:
        assert torch.isfinite(o).all() and torch.isfinite(rew).all()
        assert rew.shape == (E, R)
    d = task.describe()
    assert d["obs"]["dim"] == task.obs_spec.dim and d["metric"]["key"] == key


@pytest.mark.parametrize("name", list(TASKS))
def test_fixed_seeds_reproducible(name):
    a = rollout(make_task(cfg(name), "cpu"), RandomPolicy(3), T)
    b = rollout(make_task(cfg(name), "cpu"), RandomPolicy(3), T)
    for (oa, ra), (ob, rb) in zip(a[1], b[1]):
        assert torch.equal(oa, ob) and torch.equal(ra, rb)
    assert json.dumps(a[0]) == json.dumps(b[0])
    c = rollout(make_task(cfg(name, seed=1), "cpu"), RandomPolicy(3), T)
    assert not all(torch.equal(x[0], y[0]) for x, y in zip(a[1], c[1]))


@pytest.mark.parametrize("name", list(TASKS))
def test_partial_reset_leaves_other_envs(name):
    task = make_task(cfg(name), "cpu")
    pol = RandomPolicy(0)
    pol.reset(task)
    obs = task.reset()
    for _ in range(3):
        obs, _, _, _ = task.step(*pol.act(task, obs))
    pos, t = task.pos.clone(), task.t.clone()
    mask = torch.tensor([False, True, False])
    task.reset(mask)
    assert torch.equal(task.pos[~mask], pos[~mask]) and torch.equal(task.t[~mask], t[~mask])
    assert task.t[1] == 0 and torch.equal(task.net.clock[~mask], t[~mask] * task.NET_SUBSTEPS)
    assert task.net.clock[1] == 0


@pytest.mark.parametrize("name", list(TASKS))
def test_bounds_oracle_vs_nocomm(name):
    """ORACLE delivers every message at capture (zero delay), NOCOMM none."""
    ro, _ = rollout(make_task(cfg(name, level="ORACLE"), "cpu"), RandomPolicy(0), T)
    rn, _ = rollout(make_task(cfg(name, level="NOCOMM"), "cpu"), RandomPolicy(0), T)
    assert all(r["deliveries"] > 0 and r["delay_p95_ms"] < 0.1 for r in ro)
    assert all(r["deliveries"] == 0 and math.isnan(r["delay_p50_ms"]) for r in rn)


@pytest.mark.parametrize("name", list(TASKS))
def test_heuristic_and_queue_rule(name):
    task = make_task(cfg(name, level="L2-legacy"), "cpu")
    rows, _ = rollout(task, HeuristicPolicy(), T)
    assert len(rows) == E
    # the send rule: the preferred choice only when the robot's queue was empty
    pol = HeuristicPolicy()
    pol.reset(task)
    obs = task.reset()
    for _ in range(4):
        q = task.queue_len.clone()
        _, ready, busy = task.heuristic()
        cont, send = pol.act(task, obs)
        assert torch.equal(send[q > 0], busy[q > 0])
        obs, _, _, _ = task.step(cont, send)


def test_light_variant_scales_sizes_and_background_gate():
    t = make_task(cfg("coop_map", variant="light"), "cpu")
    assert t.sizes == tuple(s * TASKS["coop_map"].LIGHT_SCALE for s in TASKS["coop_map"].MSG_SIZES)
    assert t.cfg.size_scale == 0.025
    assert t.nr.msg_sizes == t.sizes
    assert set(variants()) >= {"default", "light"}
    if not background_available():
        assert "background" not in variants()
        with pytest.raises(ValueError, match="background"):
            make_task(cfg("coop_map", variant="background"), "cpu")


def test_edge_control_runs_through_edge_loop():
    task = make_task(cfg("edge_control", level="L1"), "cpu")
    assert type(task.net.eng._eng).__name__ == "EdgeLoop"
    rows, _ = rollout(task, HeuristicPolicy(), T)
    r = rows[0]
    assert "stale_frac" in r and "command_age_ms" in r and r["tracking_error_m"] > 0
    assert task.net.clock[0] == 0                          # reset after the episode
    # 5 network steps of 20 ms per task step: deliveries per robot bounded by 50 Hz x episode
    assert all(x["sent"] <= T * 5 + 1e-9 for x in rows)


def test_coverage_nav_holes_cost_snr():
    task = make_task(cfg("coverage_nav", level="L2-legacy"), "cpu")
    task.reset()
    p = task.holes[:, :1, :].clone()                                  # a hole centre per env
    inside = task._snr(p)
    far = task._snr(p + torch.tensor([task.HOLE_R + 1.0, 0.0]))
    task.holes = task.holes + 1000.0
    assert torch.allclose(inside + 25.0, task._snr(p), atol=1e-4)
    assert far.shape == inside.shape


def test_histogram_quantiles_close_to_exact():
    g = torch.Generator().manual_seed(0)
    d = torch.exp(torch.randn(1, 5000, generator=g) * 1.2 + 3.0)              # ms, lognormal
    h = torch.zeros(1, NB, dtype=torch.long).scatter_add_(1, delay_bins(d), torch.ones(1, 5000, dtype=torch.long))
    for q in (0.5, 0.95):
        est, ex = float(hist_quantile(h, q)), float(torch.quantile(d, q))
        assert abs(est / ex - 1) < 0.06
    assert torch.isnan(hist_quantile(torch.zeros(1, NB, dtype=torch.long), 0.5)).all()
    assert bin_edges_ms().shape == (NB + 1,)


def test_mean_ci95():
    m, ci, n = mean_ci95([1.0, 2.0, 3.0])
    assert m == 2.0 and n == 3 and abs(ci - 4.303 * 1.0 / math.sqrt(3)) < 1e-9
    assert math.isnan(mean_ci95([5.0])[1]) and mean_ci95([])[2] == 0


@pytest.mark.parametrize("arch", ["mlp", "gru"])
def test_ppo_smoke(arch):
    task = make_task(cfg("fleet_alert", episode_steps=6), "cpu")
    model, info = train_ppo(task, arch, iters=2, horizon=4, seed=0)
    assert info["samples"] == 2 * 4 * E * R and info["curve"]
    assert all(torch.isfinite(p).all() for p in model.parameters())


def test_run_write_report_and_cli(tmp_path):
    outs = []
    for s in (0, 1):
        res = run(cfg("fleet_alert", seed=s), "random", "cpu", label="t")
        assert res["schema"] == "isaac-net-bench/1" and res["eval"]["seed"] == s + 10_000
        outs.append(write_result(res, str(tmp_path)))
    json.loads(open(outs[0]).read())
    rows = aggregate(load_results([str(tmp_path)]))
    assert len(rows) == 1 and rows[0]["seeds"] == [0, 1]
    assert rows[0]["stats"]["task"][2] == 2
    md = markdown(rows)
    assert "hazard_exposure" in md and "±" in md
    rc = cli_main(["run", "--task", "coop_map", "--level", "L0", "--envs", "2", "--robots", "3", "--episode_steps", "5",
                   "--baselines", "heuristic,ppo_mlp", "--seeds", "0", "--ppo_iters", "1", "--ppo_horizon", "3",
                   "--device", "cpu", "--out", str(tmp_path / "cli"), "--quiet"])
    assert rc == 0 and len(load_results([str(tmp_path / "cli")])) == 2
    assert cli_main(["report", str(tmp_path / "cli")]) == 0


def test_calibrate_min_max_load():
    c = calibrate(cfg("fleet_alert", level="L2-legacy"), "cpu")
    assert c["max"]["offered_mbps"] > c["min"]["offered_mbps"] > 0
    assert c["max"]["choice"] == "large"


def test_evaluate_counts():
    task = make_task(cfg("fleet_alert", level="ORACLE"), "cpu")
    ev = evaluate(task, RandomPolicy(1), episodes=2)
    assert len(ev["rows"]) == 2 * E and ev["task_steps"] == 2 * T


@pytest.mark.gpu
@pytest.mark.parametrize("name", ["coop_map", "edge_control"])
def test_gpu_graph_backend_equals_reference(name):
    """On CUDA the graph backend (and the edge stage's CUDA graph) reproduce the reference engine's episodes."""
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    traces = []
    for backend in ("reference", "graph"):
        torch.manual_seed(0)
        task = make_task(cfg(name, level="L2-legacy", backend=backend), "cuda")
        traces.append(rollout(task, HeuristicPolicy(), T))
    (ra, ta), (rb, tb) = traces
    for (oa, wa), (ob, wb) in zip(ta, tb):
        assert torch.equal(oa, ob) and torch.equal(wa, wb)
    assert json.dumps(ra) == json.dumps(rb)          # NaN-aware
