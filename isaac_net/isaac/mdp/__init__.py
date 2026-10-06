"""MDP terms for Isaac Lab managers: the network in a manager-based env, and network domain randomization.

    observations   net_aoi, net_delivered, net_delay, net_queue, net_sinr, net_los, net_access_state
                   (func(env, asset_cfg=None) -> [E, R * k])
    events         net_step (interval mode), net_reset (reset mode), randomize_network
    terminations   net_step_done: the network step as the first termination term (the default placement)
    rewards        net_aoi_penalty, net_send_cost
    actions        NetSendActionCfg: the policy chooses each robot's message class (Isaac Lab only)
    net_setup      build env.isaac_net (isaac/runtime.py); NetManagerCfg (isaac/manager_cfg.py) wires all of it

Nothing here imports Isaac Lab except mdp.actions, which guards it.
"""
from ..runtime import NetRuntime, get_runtime, net_setup  # noqa: F401
from .events import net_reset, net_step, randomize_network  # noqa: F401
from .observations import (OBS_TERMS, TERM_WIDTH, net_access_state, net_aoi, net_delay, net_delivered,  # noqa: F401
                           net_los, net_queue, net_sinr)
from .rewards import net_aoi_penalty, net_send_cost  # noqa: F401
from .terminations import net_step_done  # noqa: F401
