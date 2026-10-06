"""EventTerm functions: network step and reset for the manager-based workflow, and network domain randomization.

    from isaaclab.managers import EventTermCfg as EventTerm
    net_reset = EventTerm(func=net_mdp.net_reset, mode="reset")
    net_step = EventTerm(func=net_mdp.net_step, mode="interval", interval_range_s=(dt, dt), is_global_time=True)

net_reset forwards the env_ids of the reset to env.isaac_net (isaac/runtime.py). net_step in "interval" mode with
interval_range_s = (step_dt, step_dt) and is_global_time=True runs once per env step, but interval events run after
the rewards and after the resets of finished envs: rewards then read the previous step's network output, and a
reset env takes one network step from its post-reset pose before its first observation. The termination term
net_step_done (terminations.py) avoids both and is what NetManagerCfg uses by default (placement="termination").

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
        rt = getattr(env, "isaac_net", None)
        net = getattr(rt, "net", None)
    if net is None:
        return
    if ranges is None:
        net.randomize(env_ids)
    else:
        net.sample_params(env_ids, ranges)


def net_step(env, env_ids=None):
    """Event term: one network step of env.isaac_net (env_ids is ignored: every env steps)."""
    from ..runtime import get_runtime
    get_runtime(env).step()


def net_reset(env, env_ids=None):
    """Event term (mode "reset"): partial reset of env.isaac_net for env_ids (None or a slice: all envs)."""
    from ..runtime import get_runtime
    if isinstance(env_ids, slice):
        env_ids = None
    get_runtime(env).reset(env_ids)
