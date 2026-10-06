"""A manager-based Isaac Lab task with the network in the loop: uplink scheduling for a fleet (Isaac-NetFleet-Manager-v0).

The scene is the fleet arena of isaac_fleet_env.py (16 floating spheres per env on a grid, gNB on a 6 m mast at the
arena corner), and the robots do not move: the policy only decides, per robot and step, whether to send a status
update and how large (NetSendActionCfg: none / 4 KB / 30 KB). The reward trades freshness against airtime:
-1 x mean AoI / time scale (net_aoi_penalty) - 0.2 x fraction of robots sending (net_send_cost). Under contention
the best policy staggers the updates instead of sending every step.

Everything network-related comes from one NetManagerCfg(...).apply(self) in __post_init__ (isaac/manager_cfg.py):
the network step as the first termination term, the network reset event, the observation terms, the AoI reward and
the send action. The env holds the network as env.isaac_net.
"""
from __future__ import annotations

import os

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs import mdp as il_mdp
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from isaac_net.isaac import mdp as net_mdp
from isaac_net.isaac.manager_cfg import NetManagerCfg
from isaac_net.isaac.marker_geometry import NetMarkersCfg

from .isaac_fleet_env import fleet_isaac_cfg, make_scene_cfg, net_config, physics_cfg

NUM_ROBOTS = 16


@configclass
class ActionsCfg:
    """Filled by NetManagerCfg(send_action=True): actions.net_send."""


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    send_cost = RewTerm(func=net_mdp.net_send_cost, weight=-0.2)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=il_mdp.time_out, time_out=True)


@configclass
class NetFleetManagerEnvCfg(ManagerBasedRLEnvCfg):
    scene = make_scene_cfg(NUM_ROBOTS, 1024)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    net_markers: NetMarkersCfg | None = None       # set before apply() runs, e.g. by a subclass

    def __post_init__(self):
        self.decimation = 5
        self.episode_length_s = 30.0
        self.sim = SimulationCfg(dt=1 / 50, render_interval=5,
                                 physics=physics_cfg(os.environ.get("ISAAC_NET_PHYSICS", "isaacsim_physx")))
        NetManagerCfg(level="L2-legacy", backend="graph", num_robots=NUM_ROBOTS, nr=net_config(0.1),
                      isaac=fleet_isaac_cfg(obs_features=("aoi", "queue_len", "sinr")), markers=self.net_markers,
                      obs_terms=("net_aoi", "net_queue", "net_sinr", "net_delivered"), aoi_weight=-1.0,
                      send_action=True, send_classes=2).apply(self)
