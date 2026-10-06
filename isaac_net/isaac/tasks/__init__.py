"""Gymnasium registration of the Isaac Lab tasks, following isaaclab_tasks (no Isaac imports at import time).

    import isaac_net.isaac.tasks          # registers the ids below (needs gymnasium, which Isaac Lab ships)

    Isaac-NetFleet-Direct-v0             NetFleetEnv: 16 robots per env, 5G uplink at level L2-legacy (graph backend)
    Isaac-NetFleet-Direct-L0-v0          the same task with the L0 network (independent lognormal delays, no
                                         contention): the delay baseline
    Isaac-NetFleet-Direct-Warehouse-v0   WarehouseFleetEnv: the fleet in Isaac Lab's warehouse, two gNBs, NR engine
                                         L2 with its own log-distance radio (no baked radio map, no Sionna)
    Isaac-NetFleet-Manager-v0            manager-based ManagerBasedRLEnv: uplink scheduling for a static fleet, built
                                         with NetManagerCfg (isaac_net/examples/isaac_manager_fleet_env.py)

Every id carries env_cfg_entry_point, rsl_rl_cfg_entry_point (RslRlOnPolicyRunnerCfg), skrl_cfg_entry_point and
default_agent "rsl_rl", as the tasks of isaaclab_tasks do, so Isaac Lab's unified train script runs them:

    isaaclab train --rl_library rsl_rl --task Isaac-NetFleet-Direct-v0 \
        --external_callback isaac_net.isaac.tasks.register
    python -m isaac_net.isaac.tasks.train --rl_library skrl --task Isaac-NetFleet-Direct-v0      # any library

The entry points are strings, so registration imports nothing from Isaac Lab; the env and cfg modules are imported
when the task is resolved. TASKS holds the metadata; register() is idempotent and returns None (the value
Isaac Lab's --external_callback expects when the callback consumes no arguments).
"""
from __future__ import annotations

_PKG = __name__
_AGENTS = f"{_PKG}.agents"
_FLEET_ENV = "isaac_net.examples.isaac_fleet_env:NetFleetEnv"
_WAREHOUSE_ENV = "isaac_net.examples.isaac_warehouse_env:WarehouseFleetEnv"


def _kwargs(cfg: str, agent: str, cfg_module: str = f"{_PKG}.fleet_env_cfg") -> dict:
    return {
        "env_cfg_entry_point": f"{cfg_module}:{cfg}",
        "rsl_rl_cfg_entry_point": f"{_AGENTS}.rsl_rl_ppo_cfg:{agent}PPORunnerCfg",
        "skrl_cfg_entry_point": f"{_AGENTS}.skrl_ppo_cfg:{_snake(agent)}_skrl_cfg",
        "default_agent": "rsl_rl",
    }


def _snake(name: str) -> str:
    return "".join("_" + c.lower() if c.isupper() and i else c.lower() for i, c in enumerate(name))


TASKS = {
    "Isaac-NetFleet-Direct-v0": dict(entry_point=_FLEET_ENV, kwargs=_kwargs("NetFleetDirectEnvCfg", "NetFleet")),
    "Isaac-NetFleet-Direct-L0-v0": dict(entry_point=_FLEET_ENV, kwargs=_kwargs("NetFleetL0EnvCfg", "NetFleetL0")),
    "Isaac-NetFleet-Direct-Warehouse-v0": dict(entry_point=_WAREHOUSE_ENV,
                                               kwargs=_kwargs("NetFleetWarehouseEnvCfg", "NetFleetWarehouse")),
    "Isaac-NetFleet-Manager-v0": dict(entry_point="isaaclab.envs:ManagerBasedRLEnv",
                                      kwargs=_kwargs("NetFleetManagerEnvCfg", "NetFleetManager",
                                                     "isaac_net.examples.isaac_manager_fleet_env")),
}


def register() -> None:
    """Register TASKS with gymnasium (skips ids that are already registered)."""
    import gymnasium as gym

    for task_id, spec in TASKS.items():
        if task_id in gym.registry:
            continue
        gym.register(id=task_id, entry_point=spec["entry_point"], disable_env_checker=True,
                     kwargs=dict(spec["kwargs"]))


try:
    register()
except ImportError:          # gymnasium missing: TASKS stays importable for inspection
    pass
