"""Isaac Lab layer (see ARCHITECTURE.md). Nothing here imports Isaac Lab at package import time.

    config.py       IsaacNetCfg: the Isaac-specific settings next to the NRConfig (pose source, multi-rate,
                    blockage, network domain randomization, observation selection), obs_dim, dr_support
    net_module.py   NetModule(level, E, R, device, NRConfig, backend, isaac=IsaacNetCfg(...)) over
                    core.make_engine: reset(env_ids) / submit(t, TrafficRequest) / step(t, poses_end, cur_tag) ->
                    dict / obs(); MessageHistory (the receiver's delayed view); NetConfig (deprecated alias)
    obs.py          NetObs: the selected, normalized observation features
    radio.py        IsaacRadio: poses -> SNR with per-env parameters (network DR), several gNBs, LOS blockage
    mixins.py       NetEnvMixin for DirectRLEnv (net_setup / net_step / net_reset / net_obs, markers=...)
    runtime.py      NetRuntime: the same mixin bound to a manager-based env (env.isaac_net)
    manager_cfg.py  NetManagerCfg(...).apply(env_cfg): the network terms of a ManagerBasedRLEnvCfg in one call
    mdp/            manager terms: observations (net_aoi, ...), net_step / net_reset events, net_step_done,
                    net_aoi_penalty, NetSendActionCfg, randomize_network
    markers.py      NetMarkers: viewport overlays (links by SINR, gNBs and coverage, AoI bars, access state)
    marker_geometry.py  NetMarkersCfg and the overlay geometry in pure torch
    tasks/          gymnasium registration (Isaac-NetFleet-Direct-v0, -L0-v0, -Warehouse-v0), rsl_rl / skrl cfgs
    netmodule.py    compatibility module: the demo's registry engine was retired, its names now come from here
    maniskill_adapter_skeleton.py   design sketch of the same module in ManiSkill3 (not run)
"""
from .config import (DEFAULT_OBS, DR_KEYS, OBS_FEATURES, IsaacNetCfg, dr_support, dr_table,  # noqa: F401
                     obs_dim)
from .net_module import (BACKENDS, FAST_BACKENDS, MessageHistory, NetConfig, NetModule, TrafficRequest,  # noqa: F401
                         net_features)
from .mixins import NetEnvMixin, rigid_positions_local  # noqa: F401
from .marker_geometry import NetMarkersCfg  # noqa: F401
from .markers import NetMarkers  # noqa: F401
from .runtime import NetRuntime  # noqa: F401
from .manager_cfg import NetManagerCfg  # noqa: F401
from .obs import NetObs  # noqa: F401
from .radio import RADIO_PARAMS, IsaacRadio, ParamRanges, segment_sphere_blocked  # noqa: F401
