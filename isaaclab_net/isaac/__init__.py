"""Isaac Lab layer (see ARCHITECTURE.md). Nothing here imports Isaac Lab at package import time.

    net_module.py   NetModule: reset(env_ids) / submit(t, req) / step(t, poses, cur_tag) -> dict
                    (L2 fast engine through core.proto.netsim_fast, or the registry engine netmodule.py)
    netmodule.py    registry engine L0 / L1 / L2 with per-env clocks, radio with LOS blockage, parameter DR
    mixins.py       NetEnvMixin for DirectRLEnv (net_setup / net_step / net_reset / net_obs)
    mdp/            randomize_network EventTerm (network domain randomisation)
    isaac_env_skeleton.py, maniskill_adapter_skeleton.py   design skeletons (import Isaac Lab / ManiSkill)

Snapshot of the isaac/demo work of 2026-09-29, which is still in development; its NetConfig is the Isaac-side
configuration until it is folded into core.NRConfig and core.make_engine (ARCHITECTURE.md, follow-ups).
"""
from .net_module import FAST_BACKENDS, NetConfig, NetModule, ParamRanges, TrafficRequest, net_features  # noqa: F401
