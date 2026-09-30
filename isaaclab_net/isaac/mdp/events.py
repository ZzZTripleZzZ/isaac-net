"""EventTerm functions for network domain randomization (modes 'reset' or 'interval')."""
from __future__ import annotations

import torch


def randomize_network(env, env_ids: torch.Tensor | None, ranges: dict[str, tuple]):
    """Resample per-env radio parameters uniformly within `ranges` for env_ids.

    Parameters (isaac.radio.RADIO_PARAMS): p_tx_dbm, noise_dbm, pl_const_db, pl_exp, shadow_sigma_db, blockage_db.
    Example: EventTermCfg(func=randomize_network, mode="reset",
                          params={"ranges": {"pl_exp": (2.8, 4.0), "shadow_sigma_db": (3.0, 8.0)}})
    In mode "reset" it runs inside DirectRLEnv._reset_idx (super) BEFORE net_reset; net_reset never touches
    parameters, so the drawn values hold for the new episode.
    """
    if getattr(env, "net", None) is not None:
        env.net.sample_params(env_ids, ranges)
