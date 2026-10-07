"""Env configurations of the registered tasks (isaac_net/isaac/tasks/__init__.py). Each class builds its scene and
spaces in __post_init__, so Isaac Lab can instantiate it without arguments (load_cfg_from_registry); the env re-derives
them from num_robots at construction (finalize_fleet_cfg), so `env.num_robots=32` on the command line works.

Defaults: 1024 envs x 16 robots (the end-to-end PPO check of docs/isaac-lab.md), 0.1 s control step (dt = 1/50 s,
decimation 5), network on the "graph" backend, which needs no Triton. `env.net_backend=triton` selects the fused
kernel where triton(-windows) is installed. The physics backend follows ISAAC_NET_PHYSICS (isaacsim_physx by
default; ovphysx, newton), as make_cfg does; the Isaac Lab `physics=` preset selector does not apply to these tasks.

Network fields from the command line: every registered cfg holds an IsaacNetCfg in net_isaac whose nr dict is applied
to the task's NRConfig when the network is built (IsaacNetCfg.resolve_nr, NRConfig.from_dict: a typo raises with the
closest field names), so Isaac Lab's env. overrides reach any NRConfig field or a preset:

    env.net_isaac.nr.ul_tpc=true env.net_isaac.nr.ul_pc=true            # fields on top of the task's NRConfig
    env.net_isaac.nr.preset=warehouse_private_5g env.net_isaac.nr.frame_buffer=16   # a preset, then fields
    env.net_isaac.log_kpis=false                                         # no net/ KPIs in the training log

Isaac Lab applies env.-prefixed overrides to the cfg object itself (isaaclab_tasks.utils.hydra._setattr), and an
nr dict takes new keys, so no "+" is needed (docs/config-files.md "Hydra overrides").
"""
from __future__ import annotations

import os

from isaaclab.utils import configclass

from isaac_net.examples.isaac_fleet_env import NetFleetEnvCfg, finalize_fleet_cfg, fleet_isaac_cfg, physics_cfg
from isaac_net.examples.isaac_warehouse_env import WarehouseFleetEnvCfg, finalize_warehouse_cfg

NUM_ENVS = 1024
NUM_ROBOTS = 16


def _physics(cfg):
    name = os.environ.get("ISAAC_NET_PHYSICS", "isaacsim_physx")
    cfg.sim.physics = physics_cfg(name)
    if name != "isaacsim_physx":
        cfg.sim.gravity = (0.0, 0.0, 0.0)        # Newton ignores the per-body disable_gravity (see make_cfg)


@configclass
class NetFleetDirectEnvCfg(NetFleetEnvCfg):
    """Isaac-NetFleet-Direct-v0: the fleet task with the slot-level 5G uplink (L2-legacy)."""
    num_robots: int = NUM_ROBOTS
    net_level: str = "L2-legacy"
    net_backend: str = "graph"

    def __post_init__(self):
        _physics(self)
        if self.net_isaac is None:
            self.net_isaac = fleet_isaac_cfg()          # holds the nr dict the env.net_isaac.nr.<field> overrides fill
        finalize_fleet_cfg(self, num_envs=NUM_ENVS)


@configclass
class NetFleetL0EnvCfg(NetFleetDirectEnvCfg):
    """Isaac-NetFleet-Direct-L0-v0: the same task with level L0 (independent lognormal delays, no contention)."""
    net_level: str = "L0"


@configclass
class NetFleetWarehouseEnvCfg(WarehouseFleetEnvCfg):
    """Isaac-NetFleet-Direct-Warehouse-v0: the fleet in Isaac Lab's warehouse (Nucleus asset), two gNBs, the NR
    engine L2 with its log-distance radio; no baked radio map (set env.net_isaac with scene_map for that)."""
    num_robots: int = NUM_ROBOTS
    net_level: str = "L2"
    net_backend: str = "reference"

    def __post_init__(self):
        _physics(self)
        finalize_warehouse_cfg(self, num_envs=64)
