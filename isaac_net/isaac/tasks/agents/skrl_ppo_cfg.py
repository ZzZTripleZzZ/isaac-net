"""skrl PPO configurations of the registered tasks, as functions returning the dict Isaac Lab's skrl backend reads
from its YAML files (the layout of isaaclab_tasks' skrl_direct_ppo_cfg.yaml). Functions, not package data files, so
they ship with the wheel without package_data and load through the same "module:attr" entry point."""
from __future__ import annotations


def _ppo(directory: str) -> dict:
    net = [{"name": "net", "input": "OBSERVATIONS", "layers": [256, 128], "activations": "elu"}]
    return {
        "seed": 42,
        "models": {
            "separate": False,
            "policy": {"class": "GaussianMixin", "clip_actions": False, "clip_log_std": True, "min_log_std": -20.0,
                       "max_log_std": 2.0, "initial_log_std": 0.0, "network": [dict(n) for n in net],
                       "output": "ACTIONS"},
            "value": {"class": "DeterministicMixin", "clip_actions": False, "network": [dict(n) for n in net],
                      "output": "ONE"},
        },
        "memory": {"class": "RandomMemory", "memory_size": -1},
        "agent": {
            "class": "PPO", "rollouts": 24, "learning_epochs": 4, "mini_batches": 4, "discount_factor": 0.99,
            "lambda": 0.95, "learning_rate": 3.0e-4, "learning_rate_scheduler": "KLAdaptiveLR",
            "learning_rate_scheduler_kwargs": {"kl_threshold": 0.01},
            "state_preprocessor": "RunningStandardScaler", "state_preprocessor_kwargs": None,
            "value_preprocessor": "RunningStandardScaler", "value_preprocessor_kwargs": None,
            "random_timesteps": 0, "learning_starts": 0, "grad_norm_clip": 1.0, "ratio_clip": 0.2,
            "value_clip": 0.2, "clip_predicted_values": True, "entropy_loss_scale": 0.005, "value_loss_scale": 1.0,
            "kl_threshold": 0.0, "rewards_shaper_scale": 1.0, "time_limit_bootstrap": False,
            "experiment": {"directory": directory, "experiment_name": "", "write_interval": "auto",
                           "checkpoint_interval": "auto"},
        },
        "trainer": {"class": "SequentialTrainer", "timesteps": 7200, "environment_info": "log"},
    }


def net_fleet_skrl_cfg() -> dict:
    return _ppo("netfleet_direct")


def net_fleet_l0_skrl_cfg() -> dict:
    return _ppo("netfleet_direct_l0")


def net_fleet_warehouse_skrl_cfg() -> dict:
    return _ppo("netfleet_direct_warehouse")


def net_fleet_manager_skrl_cfg() -> dict:
    return _ppo("netfleet_manager")
