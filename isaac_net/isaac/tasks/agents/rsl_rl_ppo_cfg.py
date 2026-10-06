"""RSL-RL PPO runner configurations of the registered tasks (the settings benchmarks/isaac/train_ppo.py ran with on
the lab box: 256-128 ELU MLPs with observation normalization, 24 steps per env per iteration, adaptive learning rate).
The observation is the centralized [E, 12 R] vector and the action [E, 3 R] (velocity x, y and the send channel)."""
from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class NetFleetPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 300
    save_interval = 50
    experiment_name = "netfleet_direct"
    clip_actions = 1.0
    actor = RslRlMLPModelCfg(
        hidden_dims=[256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.8),
    )
    critic = RslRlMLPModelCfg(hidden_dims=[256, 128], activation="elu", obs_normalization=True)
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=4,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class NetFleetL0PPORunnerCfg(NetFleetPPORunnerCfg):
    experiment_name = "netfleet_direct_l0"


@configclass
class NetFleetWarehousePPORunnerCfg(NetFleetPPORunnerCfg):
    experiment_name = "netfleet_direct_warehouse"


@configclass
class NetFleetManagerPPORunnerCfg(NetFleetPPORunnerCfg):
    experiment_name = "netfleet_manager"
