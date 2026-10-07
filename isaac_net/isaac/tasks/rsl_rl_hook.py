"""Network checkpoints next to rsl_rl policy checkpoints (docs/checkpoint.md, "Training with rsl_rl").

rsl_rl's OnPolicyRunner.save(path) writes model_<iter>.pt and has no callback, so install() wraps the runner class's
save and load once per process:

  save(path)   after the policy file, the env's network (isaac/net_module.find_network: env.net of a Direct env with
               NetEnvMixin, or env.isaac_net of a manager-based env) is written next to it, model_<iter>.pt ->
               isaac_net_<iter>.pt (NetModule.save with the host's multi-rate state);
  load(path)   after the policy, the matching isaac_net_<iter>.pt is restored if it exists (Isaac Lab's train script
               calls runner.load(resume_path) for --resume). A file that does not match the env (other level, E, R
               or config) is not loaded: a warning says why, and training continues on the env's fresh network.
               With distributed training only rank 0 saves, so only rank 0 restores; the other ranks start fresh.

The wrap is a no-op for envs without an isaac_net network. isaac_net.isaac.tasks.register() and
`python -m isaac_net.isaac.tasks.train` install it when rsl_rl is importable; ISAAC_NET_RSL_RL_HOOK=0 turns it off.
Without the hook (another runner, or your own loop) call isaac_net.isaac.net_module.save_env_network(env, path) and
load_env_network(env, path) yourself.
"""
from __future__ import annotations

import functools
import os
import re
import warnings

ENV_SWITCH = "ISAAC_NET_RSL_RL_HOOK"
_MARK = "_isaac_net_ckpt_hook"


def net_path(model_path) -> str:
    """model_<iter>.pt -> isaac_net_<iter>.pt in the same directory (any other name: isaac_net_<name>)."""
    d, base = os.path.split(os.fspath(model_path))
    m = re.fullmatch(r"model_(.+)\.pt", base)
    return os.path.join(d, f"isaac_net_{m.group(1)}.pt" if m else f"isaac_net_{base}")


def _env(runner):
    return getattr(runner, "env", None)


def install(runner_cls=None) -> bool:
    """Wrap runner_cls.save / load (default: rsl_rl.runners.OnPolicyRunner). Idempotent. Returns True if the class
    carries the hook, False if rsl_rl is not importable or ISAAC_NET_RSL_RL_HOOK=0."""
    if os.environ.get(ENV_SWITCH, "1") == "0":
        return False
    if runner_cls is None:
        try:
            from rsl_rl.runners import OnPolicyRunner as runner_cls
        except ImportError:
            return False
    if runner_cls.__dict__.get(_MARK, False):
        return True
    save0, load0 = runner_cls.save, runner_cls.load

    @functools.wraps(save0)
    def save(self, path, *args, **kwargs):
        res = save0(self, path, *args, **kwargs)
        from ..net_module import save_env_network
        try:
            save_env_network(_env(self), net_path(path), extra={"policy_checkpoint": os.path.basename(os.fspath(path))})
        except Exception as e:             # noqa: BLE001  (never break the policy checkpoint)
            warnings.warn(f"isaac_net: the network state was not saved next to {path}: {e!r}", stacklevel=2)
        return res

    @functools.wraps(load0)
    def load(self, path, *args, **kwargs):
        res = load0(self, path, *args, **kwargs)
        p = net_path(path)
        if getattr(self, "is_distributed", False) and getattr(self, "gpu_global_rank", 0) != 0:
            return res                     # the file holds rank 0's envs (only rank 0 saves): other ranks start fresh
        if os.path.exists(p):
            from ..net_module import load_env_network
            try:
                load_env_network(_env(self), p, strict=True)
            except (ValueError, KeyError) as e:
                warnings.warn(f"isaac_net: {p} does not match this env's network, which starts fresh: {e}",
                              stacklevel=2)
        return res

    runner_cls.save, runner_cls.load = save, load
    setattr(runner_cls, _MARK, True)
    return True
