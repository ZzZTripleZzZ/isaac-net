"""Benchmark suite for network-aware multi-robot learning: standard tasks, metrics and baselines.

    from isaaclab_net.bench import TaskConfig, make_task
    task = make_task(TaskConfig(task="coop_map", level="L2-legacy", backend="graph", num_envs=64, seed=0), "cuda")

    python -m isaaclab_net.bench list
    python -m isaaclab_net.bench run --task coop_map --level L0 --backend graph --baselines random,heuristic --seeds 0,1
    python -m isaaclab_net.bench report results/

Modules: spec (TaskConfig, ObsSpec, ActionSpec, presets), base (NetTask), tasks (fleet_alert, coop_map,
coverage_nav, edge_control and their variants), metrics (per-episode metrics), baselines (random, heuristic, PPO
MLP / GRU), runner (run, evaluate, calibrate, result files), report (mean ± 95% CI tables), cli.
Nothing here imports Isaac Lab; TaskConfig(sim="isaac") loads the Isaac adapter lazily. docs/benchmark-suite.md
describes the tasks, the metrics and the results format.
"""
from .base import NetTask  # noqa: F401
from .baselines import BASELINES, HeuristicPolicy, PPOPolicy, RandomPolicy, train_ppo  # noqa: F401
from .metrics import EpisodeMetrics  # noqa: F401
from .report import aggregate, load_results, markdown  # noqa: F401
from .runner import calibrate, evaluate, run, write_result  # noqa: F401
from .spec import (NR_PRESETS, RESULT_SCHEMA, TRAFFIC_PRESETS, ActionSpec, MetricSpec, ObsSpec,  # noqa: F401
                   TaskConfig, background_available, mean_ci95, register_background)
from .tasks import TASKS, make_task, variants  # noqa: F401
