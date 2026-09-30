"""Isaac Lab tests (marker `isaac`; skipped when Isaac Lab is not installed). Each case runs in its own process,
because one Isaac Sim app hosts one DirectRLEnv at a time.

1. tests/scripts/isaac_fleet_check.py: the fleet env's network on a fast backend (eager) equals a NetModule on
   the reference engine replayed with the recorded PhysX poses, sends, tags and RNG stream, bitwise, through
   partial resets that DirectRLEnv makes itself (episode timeouts of a subset of envs).
2. benchmarks/isaac/bench.py smoke: every level and backend the mixin offers steps the fleet env with finite
   observations.
3. IsaacNetCfg inside Isaac Lab: the replay check with every observation feature and network domain randomization
   (the selected features and the per-env radio parameters match too), and bench smoke runs of the two multi-rate
   modes with a feature selection, sized by IsaacNetCfg.obs_dim.
Run on a machine with Isaac Lab 3.0 (headless is its default): python -m pytest -m isaac tests/test_isaac_env.py
"""
import json
import os
import subprocess
import sys

import pytest

pytestmark = pytest.mark.isaac

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TIMEOUT_S = 900


def _run(script, *argv):
    p = subprocess.run([sys.executable, os.path.join(ROOT, script), *argv], cwd=ROOT,
                       capture_output=True, text=True, timeout=TIMEOUT_S)
    return p.returncode, p.stdout + p.stderr


def _line(text, prefix):
    rows = [ln[len(prefix):] for ln in text.splitlines() if ln.startswith(prefix)]
    assert rows, text[-3000:]
    return json.loads(rows[-1])


@pytest.mark.parametrize("level", ["L2-legacy", "L1", "L0DR"])
def test_fleet_env_network_equals_reference_replay(level):
    rc, out = _run("tests/scripts/isaac_fleet_check.py", "--level", level, "--backend", "eager")
    res = _line(out, "CHECK ")
    assert rc == 0 and res["ok"], res
    assert res["partial_resets"] >= 2 and res["delivered_frac"] > 0


@pytest.mark.parametrize("level,backend", [("off", "graph"), ("L0", "graph"), ("L1", "triton"),
                                           ("L2-legacy", "graph"), ("L2-legacy", "triton"), ("L2", "reference")])
def test_fleet_env_levels_smoke(level, backend, tmp_path):
    if backend == "triton":
        pytest.importorskip("triton")
    out_file = str(tmp_path / "bench.jsonl")
    rc, out = _run("benchmarks/isaac/bench.py", "--num_envs", "16", "--num_robots", "4", "--level", level,
                   "--backend", backend, "--steps", "20", "--warmup", "3", "--max_time", "60", "--out", out_file)
    res = _line(out, "RESULT ")
    assert rc == 0, out[-3000:]
    assert res["obs_finite"] and res["timed_steps"] == 20 and res["obs_shape"] == [16, 4 * 12]
    if level != "off":
        assert res["net_last_step"]["mean_snr_db"] == res["net_last_step"]["mean_snr_db"]   # not NaN


def test_fleet_env_obs_selection_and_dr_equal_reference_replay():
    rc, out = _run("tests/scripts/isaac_fleet_check.py", "--level", "L2-legacy", "--backend", "eager",
                   "--obs", "all", "--dr")
    res = _line(out, "CHECK ")
    assert rc == 0 and res["ok"], res
    assert res["obs_dim"] == 44 and res["worst"]["obs"] == 0.0 and res["dr_initial_equal"]
    assert res["dr"] == ["gnb_offset_m", "noise_dbm", "pl_exp", "shadow_sigma_db"]


@pytest.mark.parametrize("rate", [("1", "5", "1"), ("10", "1", "2")])
def test_fleet_env_multi_rate_smoke(rate, tmp_path):
    env_dec, net_dec, net_sub = rate
    out_file = str(tmp_path / "bench.jsonl")
    rc, out = _run("benchmarks/isaac/bench.py", "--num_envs", "16", "--num_robots", "4", "--level", "L2-legacy",
                   "--backend", "eager", "--steps", "20", "--warmup", "3", "--max_time", "60", "--out", out_file,
                   "--obs", "aoi,delay_history,serving_cell,last_delivered", "--env_decimation", env_dec,
                   "--net_decimation", net_dec, "--net_substeps", net_sub)
    res = _line(out, "RESULT ")
    assert rc == 0, out[-3000:]
    assert res["obs_finite"] and res["obs_shape"] == [16, 4 * (9 + 1 + 4 + 1 + 1)]
