"""EventTerm functions for network domain randomisation (modes 'reset' or 'interval')."""
from __future__ import annotations

import torch


def randomize_network(env, env_ids: torch.Tensor | None, ranges: dict[str, tuple]):
    """Resample per-env network parameters uniformly within `ranges` for env_ids.

    Example: EventTermCfg(func=randomize_network, mode="reset",
                          params={"ranges": {"pl_exp": (2.8, 4.0), "shadow_sigma_db": (3.0, 8.0)}})
    Runs inside DirectRLEnv._reset_idx (super) BEFORE net_reset, and net_reset never touches parameters.
    """
    if getattr(env, "net", None) is not None:
        env.net.sample_params(env_ids, ranges)
