"""Run one baseline on one task configuration and write a result file (format: docs/benchmark-suite.md).

    res = run(TaskConfig(task="coop_map", level="L0", backend="graph", seed=0), "heuristic", device="cuda")
    write_result(res, "results/")          # results/<task>__<variant>__<sim>__<preset>__<traffic>__<level>__<backend>__
                                           #   <baseline>__s<seed>__<config hash>[__<label>].json

Seeds. A run with seed s trains (PPO) on TaskConfig.seed = s and evaluates on the same configuration with seed
s + EVAL_SEED_OFFSET, for every baseline, so all baselines of one seed see the same evaluation episodes up to the
effect of their own actions. torch.manual_seed is set from the seed too, because the NR engine (level L2) and the
edge loop draw from the global generator.
"""
from __future__ import annotations

import datetime
import json
import os
import platform
import subprocess
import time
from typing import Optional

import torch

from .baselines import make_policy, train_ppo
from .spec import RESULT_SCHEMA, TaskConfig
from .tasks import TASKS, make_task, resolve

EVAL_SEED_OFFSET = 10_000


def _git_commit() -> Optional[str]:
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=here, capture_output=True, text=True,
                              timeout=5).stdout.strip() or None
    except Exception:
        return None


def environment(device) -> dict:
    dev = torch.device(device)
    return {"torch": torch.__version__, "device": str(dev),
            "gpu": torch.cuda.get_device_name(dev) if dev.type == "cuda" else None,
            "host": platform.node(), "python": platform.python_version(), "commit": _git_commit(),
            "time": datetime.datetime.now().isoformat(timespec="seconds")}


def _sync(dev):
    if torch.device(dev).type == "cuda":
        torch.cuda.synchronize(dev)


@torch.no_grad()
def evaluate(task, policy, episodes: int = 1) -> dict:
    """Run `episodes` full episodes on every env of `task`; returns the metric rows and timings."""
    policy.reset(task)
    obs = task.reset()
    rows, done, steps = [], None, 0
    _sync(task.dev)
    t0 = time.time()
    while len(rows) < episodes * task.E:
        cont, send = policy.act(task, obs, done)
        obs, _, done, info = task.step(cont, send)
        rows += info["episodes"]
        steps += 1
    _sync(task.dev)
    sec = time.time() - t0
    return {"rows": rows, "sec": sec, "task_steps": steps, "sec_per_step": sec / max(1, steps),
            "robot_steps_per_s": steps * task.E * task.R / max(sec, 1e-9)}


def summarize(rows: list) -> dict:
    """Mean of every metric over the evaluation rows (NaN entries skipped)."""
    import math
    out = {}
    for k in rows[0]:
        v = [r[k] for r in rows if r[k] is not None and math.isfinite(r[k])]
        out[k] = sum(v) / len(v) if v else None
    return out


def run(cfg: TaskConfig, baseline: str, device="cpu", *, eval_episodes: int = 1, eval_envs: Optional[int] = None,
        ppo_iters: int = 30, ppo_horizon: int = 32, ppo_envs: Optional[int] = None, label: str = "",
        keep_rows: bool = False, log=None) -> dict:
    """Train (PPO) and evaluate one baseline; returns the result dict of the results format."""
    cfg = resolve(cfg)
    torch.manual_seed(int(cfg.seed))
    t_build = time.time()
    train = None
    model = None
    if baseline in ("ppo_mlp", "ppo_gru"):
        ttask = make_task(cfg.with_(num_envs=ppo_envs or cfg.num_envs), device)
        model, train = train_ppo(ttask, "mlp" if baseline == "ppo_mlp" else "gru", iters=ppo_iters,
                                 horizon=ppo_horizon, seed=cfg.seed, log=log)
        del ttask
    ecfg = cfg.with_(seed=cfg.seed + EVAL_SEED_OFFSET, num_envs=eval_envs or cfg.num_envs)
    torch.manual_seed(int(ecfg.seed))
    task = make_task(ecfg, device)
    build_sec = time.time() - t_build - (train["train_sec"] if train else 0.0)
    pol = make_policy(baseline, cfg.seed, model)
    ev = evaluate(task, pol, eval_episodes)
    res = {
        "schema": RESULT_SCHEMA,
        "label": label,
        "task": cfg.task, "variant": cfg.variant, "level": cfg.level, "backend": cfg.backend, "sim": cfg.sim,
        "preset": cfg.preset, "traffic": cfg.traffic, "baseline": baseline, "seed": cfg.seed,
        "config": cfg.describe(),
        "task_spec": task.describe(),
        "train": train,
        "eval": {"seed": ecfg.seed, "envs": task.E, "robots": task.R, "episodes": eval_episodes,
                 "n_rows": len(ev["rows"]), "metrics": summarize(ev["rows"])},
        "timing": {"build_sec": build_sec, "train_sec": train["train_sec"] if train else 0.0,
                   "eval_sec": ev["sec"], "eval_sec_per_step": ev["sec_per_step"],
                   "eval_robot_steps_per_s": ev["robot_steps_per_s"]},
        "environment": environment(device),
    }
    if keep_rows:
        res["eval"]["rows"] = ev["rows"]
    return res


_NAME_KEYS = ("task", "variant", "sim", "preset", "traffic", "level", "backend", "baseline", "seed")
_TRAIN_KEYS = ("arch", "iters", "horizon", "lr", "hidden")


def config_hash(res: dict) -> str:
    """8 hex digits over what defines a run besides the named fields: the rest of the TaskConfig (envs, robots,
    episode length, observation features, fit file, overrides), the evaluation envs and episodes, and the PPO
    budget. Stable across Python sessions (sha1 of sorted JSON)."""
    import hashlib
    rest = {k: v for k, v in (res.get("config") or {}).items() if k not in _NAME_KEYS}
    ev = res.get("eval") or {}
    rest["eval"] = {k: ev.get(k) for k in ("envs", "episodes")}
    tr = res.get("train") or {}
    rest["train"] = {k: tr.get(k) for k in _TRAIN_KEYS} if tr else None
    blob = json.dumps(rest, sort_keys=True, default=repr)
    return hashlib.sha1(blob.encode()).hexdigest()[:8]


def result_name(res: dict) -> str:
    """<task>__<variant>__<sim>__<preset>__<traffic>__<level>__<backend>__<baseline>__s<seed>__<hash>[__<label>].json
    (hash: config_hash). Runs that differ in any setting get different files."""
    parts = [res["task"], res["variant"], res.get("sim", "torch"), res.get("preset", "default"),
             res.get("traffic", "policy"), res["level"], res["backend"], res["baseline"], f"s{res['seed']}",
             config_hash(res)]
    if res.get("label"):
        parts.append(res["label"])
    return "__".join(str(p).replace("/", "-") for p in parts) + ".json"


def write_result(res: dict, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, result_name(res))
    with open(path, "w") as f:
        json.dump(res, f, indent=1, default=float)
    return path


def calibrate(cfg: TaskConfig, device="cpu", episodes: int = 1) -> dict:
    """Load calibration of a task: the heuristic motion with a fixed send choice for every robot at every step,
    once with the lightest sending choice and once with the heaviest. Returns offered and delivered Mbit/s,
    delivery ratio and delay quantiles of both, per env and second."""
    from .baselines import HeuristicPolicy

    class Fixed(HeuristicPolicy):
        def __init__(self, choice):
            super().__init__()
            self.choice = choice

        def act(self, task, obs, done=None):
            cont, _, _ = task.heuristic()
            return cont, torch.full((task.E, task.R), self.choice, dtype=torch.long, device=task.dev)

    cfg = resolve(cfg)
    out = {"task": cfg.task, "variant": cfg.variant, "level": cfg.level, "backend": cfg.backend,
           "envs": cfg.num_envs, "robots": cfg.num_robots}
    choices = TASKS[cfg.task].SEND_CHOICES
    lo = 0 if cfg.task == "edge_control" else 1                  # lightest choice that still sends
    for tag, ch in (("min", lo), ("max", len(choices) - 1)):
        torch.manual_seed(int(cfg.seed))
        task = make_task(cfg, device)
        m = summarize(evaluate(task, Fixed(ch), episodes)["rows"])
        out[tag] = {"choice": choices[ch], **{k: m.get(k) for k in (
            "offered_mbps", "delivered_mbps", "delivery_ratio", "delay_p50_ms", "delay_p95_ms", "refused")}}
    return out
