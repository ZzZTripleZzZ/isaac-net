"""Command line of the benchmark suite: python -m isaac_net.bench <command> ...

    list                                      tasks, variants, levels, presets, baselines
    run --task T --level L --backend B --baselines random,heuristic,ppo_mlp --seeds 0,1 --out results/
    report results/ [--metrics task,delay_p95_ms] [--json out.json]
    calibrate --task T --level L --backend B  offered vs delivered load for the lightest and heaviest send choice
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import torch

from ..core.engine import BACKENDS, LEVELS
from .baselines import BASELINES
from .report import DEFAULT_METRICS, aggregate, load_results, markdown, to_json
from .runner import calibrate, run, write_result
from .spec import NR_PRESETS, TRAFFIC_PRESETS, TaskConfig
from .tasks import TASKS, variants


def _csv(s, cast=str):
    return [cast(x) for x in s.split(",") if x.strip()]


def _num(v, spec: str) -> str:
    """A metric for the progress line: summarize() gives None when no evaluation row had a finite value."""
    return "n/a" if v is None else format(v, spec)


def _cfg(a, task, seed) -> TaskConfig:
    return TaskConfig(task=task, variant=a.variant, level=a.level, backend=a.backend, sim=a.sim, preset=a.preset,
                      traffic=a.traffic, num_envs=a.envs, num_robots=a.robots, episode_steps=a.episode_steps,
                      seed=seed, net_obs=tuple(_csv(a.obs)), level_params=a.params)


def _add_common(p):
    p.add_argument("--task", default="fleet_alert", help=f"comma-separated, or 'all': {', '.join(TASKS)}")
    p.add_argument("--variant", default="default", help=f"one of {', '.join(variants())}")
    p.add_argument("--level", default="L2-legacy", help="fidelity level (make_engine)")
    p.add_argument("--backend", default="reference", help=f"engine backend: {', '.join(BACKENDS)}")
    p.add_argument("--sim", default="torch", choices=["torch", "isaac"])
    p.add_argument("--preset", default="default", help=f"NRConfig preset: {', '.join(NR_PRESETS)}")
    p.add_argument("--traffic", default="policy", help=f"traffic preset: {', '.join(TRAFFIC_PRESETS)}")
    p.add_argument("--envs", type=int, default=64)
    p.add_argument("--robots", type=int, default=16)
    p.add_argument("--episode_steps", type=int, default=None, help="default: the task's standard length")
    p.add_argument("--obs", default="aoi,sinr,queue_len,last_delivered", help="NetModule observation features")
    p.add_argument("--params", default=None, help="fit file for TR / GE / QA / NN / L05 / L05Q")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--threads", type=int, default=1, help="CPU threads for torch")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m isaac_net.bench", description="isaac-net benchmark suite")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="tasks, variants, levels, presets and baselines")
    pr = sub.add_parser("run", help="train / evaluate baselines and write one JSON file per run")
    _add_common(pr)
    pr.add_argument("--baselines", default="random,heuristic", help=f"comma-separated: {', '.join(BASELINES)}")
    pr.add_argument("--seeds", default="0", help="comma-separated seeds")
    pr.add_argument("--eval_envs", type=int, default=None)
    pr.add_argument("--eval_episodes", type=int, default=1)
    pr.add_argument("--ppo_iters", type=int, default=30)
    pr.add_argument("--ppo_horizon", type=int, default=32)
    pr.add_argument("--ppo_envs", type=int, default=None)
    pr.add_argument("--label", default="", help="free-form tag in the file name and the result (e.g. sanity)")
    pr.add_argument("--out", default="results")
    pr.add_argument("--quiet", action="store_true")
    pp = sub.add_parser("report", help="aggregate result files: mean ± 95%% CI over seeds")
    pp.add_argument("paths", nargs="+")
    pp.add_argument("--metrics", default=",".join(DEFAULT_METRICS))
    pp.add_argument("--no_timing", action="store_true")
    pp.add_argument("--json", default=None, help="also write the aggregate as JSON")
    pc = sub.add_parser("calibrate", help="offered vs delivered load at the lightest and heaviest send choice")
    _add_common(pc)
    pc.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    if a.cmd == "list":
        print("tasks:")
        for n, c in TASKS.items():
            print(f"  {n:14s} metric {c.METRIC.key} ({'higher' if c.METRIC.higher_is_better else 'lower'} is better)")
        print("variants:", ", ".join(variants()))
        print("levels:", ", ".join(LEVELS))
        print("backends:", ", ".join(BACKENDS))
        print("presets:", ", ".join(NR_PRESETS))
        print("traffic:", ", ".join(TRAFFIC_PRESETS))
        print("baselines:", ", ".join(BASELINES))
        return 0

    if a.cmd == "report":
        res = load_results(a.paths)
        if not res:
            print("no result files found", file=sys.stderr)
            return 1
        rows = aggregate(res)
        print(markdown(rows, _csv(a.metrics), timing=not a.no_timing))
        if a.json:
            with open(a.json, "w") as f:
                json.dump(to_json(rows), f, indent=1)
        return 0

    torch.set_num_threads(a.threads)
    tasks = list(TASKS) if a.task == "all" else _csv(a.task)
    if a.cmd == "calibrate":
        for t in tasks:
            print(json.dumps(calibrate(_cfg(a, t, a.seed), a.device)))
        return 0

    log = None if a.quiet else (lambda s: print("   ", s, flush=True))
    for t in tasks:
        for b in _csv(a.baselines):
            for s in _csv(a.seeds, int):
                t0 = time.time()
                res = run(_cfg(a, t, s), b, a.device, eval_episodes=a.eval_episodes, eval_envs=a.eval_envs,
                          ppo_iters=a.ppo_iters, ppo_horizon=a.ppo_horizon, ppo_envs=a.ppo_envs, label=a.label,
                          log=log)
                path = write_result(res, a.out)
                m = res["eval"]["metrics"]
                key = res["task_spec"]["metric"]["key"]
                print(f"{t} {a.variant} {a.level}/{a.backend} {b} seed {s}: {key} {_num(m.get(key), '.4g')} "
                      f"return {_num(m.get('return'), '.3g')} delay_p95 {_num(m.get('delay_p95_ms'), '.4g')} ms  "
                      f"[{time.time() - t0:.0f}s] -> {path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
