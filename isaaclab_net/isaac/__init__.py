"""Isaac Lab layer (see ARCHITECTURE.md). Nothing here imports Isaac Lab at package import time.

    net_module.py   NetModule over core.make_engine(level, E, R, device, NRConfig, backend):
                    reset(env_ids) / submit(t, TrafficRequest) / step(t, poses_end, cur_tag) -> dict;
                    MessageHistory (the receiver's delayed view); NetConfig (compatibility alias of the demo)
    radio.py        IsaacRadio: poses -> SNR with per-env parameters (network DR), several gNBs, LOS blockage
    mixins.py       NetEnvMixin for DirectRLEnv (net_setup / net_step / net_reset / net_obs)
    mdp/            randomize_network EventTerm (network domain randomization)
    netmodule.py    compatibility module: the demo's registry engine was retired, its names now come from here
    isaac_env_skeleton.py, maniskill_adapter_skeleton.py   design skeletons (import Isaac Lab / ManiSkill)
"""
from .net_module import (BACKENDS, FAST_BACKENDS, MessageHistory, NetConfig, NetModule, TrafficRequest,  # noqa: F401
                         net_features)
from .radio import RADIO_PARAMS, IsaacRadio, ParamRanges, segment_sphere_blocked  # noqa: F401
