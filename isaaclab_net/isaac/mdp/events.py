"""EventTerm functions for network domain randomization (modes 'reset' or 'interval').

The simplest route needs no EventTerm: IsaacNetCfg.dr_ranges with dr_mode "reset" or "interval" makes the NetModule
redraw the parameters itself (see isaac/config.py and docs/isaac-lab.md). randomize_network is the same draw as an
Isaac Lab EventTerm, for tasks that keep all randomization in their event manager.
"""
from __future__ import annotations

import torch


def randomize_network(env, env_ids: torch.Tensor | None, ranges: dict[str, tuple] | None = None):
    """Resample per-env radio parameters uniformly within `ranges` for env_ids.

    Parameters (isaac.radio.RADIO_PARAMS and gnb_offset_m): p_tx_dbm, noise_dbm, pl_const_db, pl_exp,
    shadow_sigma_db, blockage_db, gnb_offset_m (per-env x/y offset of every gNB). ranges=None takes the radio keys
    of the module's IsaacNetCfg.dr_ranges.
    Example: EventTermCfg(func=randomize_network, mode="reset",
                          params={"ranges": {"pl_exp": (2.8, 4.0), "shadow_sigma_db": (3.0, 8.0)}})
    In mode "reset" it runs inside DirectRLEnv._reset_idx (super) BEFORE net_reset; net_reset never touches
    parameters unless IsaacNetCfg.dr_mode is "reset", so the drawn values hold for the new episode. Delay and loss
    of L0 / L0DR are engine parameters: set them through IsaacNetCfg.dr_ranges (NRConfig dr_* / l0_* fields).
    """
    net = getattr(env, "net", None)
    if net is None:
        return
    if ranges is None:
        net.randomize(env_ids)
    else:
        net.sample_params(env_ids, ranges)
