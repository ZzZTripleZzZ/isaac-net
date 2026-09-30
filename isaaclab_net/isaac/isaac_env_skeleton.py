"""Isaac Lab 3.0 (DirectRLEnv) skeleton: a fleet of R simple robots per env behind a 5G uplink.

NOT RUN. Written against the release/3.0.0 branch API (checked 2026-09-29):
  - isaaclab.envs.DirectRLEnv / DirectRLEnvCfg, isaaclab.assets.RigidObjectCollection(Cfg)
  - asset data returns ProxyArray; use `.torch` for a zero-copy torch view (wp.to_torch is a
    deprecated shim), `.warp` for kernels
  - RigidObjectCollectionData.body_link_pos_w -> (num_envs, num_bodies) vec3f == torch [E,R,3]
  - write_body_link_velocity_to_sim_index(body_velocities=..., env_ids=...)
  - EventTermCfg(func, mode in {"prestartup","startup","reset","interval"}, params=...)

Control/communication story: an edge server runs the (centralised, parameter-shared-ready)
policy. Robots send state updates uplink through NetModule; the server acts on the freshest
DELIVERED state (MessageHistory), plus per-robot network features (AoI, SINR, queue).
Downlink is ideal in v0 (commands arrive in the same step); a DL NetModule instance can be
added symmetrically with a second MessageHistory on the robot side.

Timing (the multi-rate contract):
  physics dt = sim.dt = 1/200 s, decimation = 20  ->  control step_dt = 0.1 s
  NR UL slot = 2.5 ms                              ->  K = step_dt / slot_dt = 40 slots per step
  pose_chunks = 4                                  ->  radio re-evaluated every 10 slots on
                                                       linearly interpolated poses
Poses are sampled once per control step on purpose: with Newton in 3.0 the physics backend may
own decimation (`_physics_handles_decimation`), in which case _apply_action runs only once per
step and per-substep hooks are not available. Stepping the network per control step with
interpolated poses behaves the same on PhysX and Newton.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg, RigidObjectCollection, RigidObjectCollectionCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from isaaclab_net.isaac.netmodule import (MessageHistory, NetConfig, NetModule, ParamRanges, TrafficRequest,
                       net_features, segment_sphere_blocked)

NUM_ROBOTS = 32
ROBOT_RADIUS = 0.3
ARENA = 20.0            # square arena side, metres; gNB on a 6 m pole at the centre
STATE_DIM = 4           # what each robot reports: x, y, vx, vy
OBS_PER_ROBOT = STATE_DIM + 2 + 5   # delivered state + goal offset + net_features


# ------------------------------------------------------------------------------------
# Event terms (domain randomisation of the network)
# ------------------------------------------------------------------------------------
def randomize_network(env: "NetFleetEnv", env_ids: torch.Tensor | None, ranges: dict[str, tuple]):
    """EventTerm func: resample per-env network parameters for env_ids (mode 'reset' or 'interval')."""
    env.net.sample_params(env_ids, ranges)


def _grid_positions(n: int, spacing: float = 1.5) -> list[tuple[float, float, float]]:
    side = math.ceil(math.sqrt(n))
    off = (side - 1) * spacing / 2
    return [((i % side) * spacing - off, (i // side) * spacing - off, 0.25) for i in range(n)]


# ------------------------------------------------------------------------------------
# Config
# ------------------------------------------------------------------------------------
@configclass
class FleetSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=2000.0))
    robots: RigidObjectCollectionCfg = RigidObjectCollectionCfg(
        rigid_objects={
            f"robot_{i}": RigidObjectCfg(
                prim_path=f"/World/envs/env_.*/Robot_{i}",
                spawn=sim_utils.CylinderCfg(
                    radius=ROBOT_RADIUS, height=0.5,
                    mass_props=sim_utils.MassCfg(mass=20.0),
                    collision_props=sim_utils.UsdPhysicsCollisionCfg(),
                    # rigid_props: backend-specific (PhysxRigidBodyCfg / Newton equivalent),
                    # see examples/multi_asset.py on release/3.0.0
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=p),
            )
            for i, p in enumerate(_grid_positions(NUM_ROBOTS))
        }
    )


@configclass
class NetEventsCfg:
    net_reset = EventTerm(
        func=randomize_network, mode="reset",
        params={"ranges": {"pl_exp": (2.8, 4.0), "shadow_sigma_db": (3.0, 8.0),
                           "blockage_db": (10.0, 30.0), "noise_dbm": (-95.0, -85.0),
                           "bg_load": (0.0, 0.6)}},
    )
    # cell load drifting mid-episode, sampled per env every 2-5 s
    net_load = EventTerm(func=randomize_network, mode="interval", interval_range_s=(2.0, 5.0),
                         params={"ranges": {"bg_load": (0.0, 0.8)}})


@configclass
class NetFleetEnvCfg(DirectRLEnvCfg):
    sim: SimulationCfg = SimulationCfg(dt=1 / 200, render_interval=20)
    decimation = 20
    episode_length_s = 30.0
    scene: FleetSceneCfg = FleetSceneCfg(num_envs=1024, env_spacing=ARENA + 5.0)
    events: NetEventsCfg = NetEventsCfg()
    action_space = NUM_ROBOTS * 2                  # planar velocity command per robot
    observation_space = NUM_ROBOTS * OBS_PER_ROBOT
    state_space = 0
    # network
    net_rung: str = "L2"
    slot_dt: float = 0.0025
    pose_chunks: int = 4
    report_period_steps: int = 1                   # each robot reports every N control steps
    max_speed: float = 1.5


# ------------------------------------------------------------------------------------
# Env
# ------------------------------------------------------------------------------------
class NetFleetEnv(DirectRLEnv):
    cfg: NetFleetEnvCfg

    # Hook map (order inside DirectRLEnv.step on release/3.0.0):
    #   _pre_physics_step(a) -> [decimation x (_apply_action, sim.step)] -> _get_dones
    #   -> _get_rewards -> _reset_idx(done ids) [scene.reset + EventTerm 'reset']
    #   -> EventTerm 'interval' -> _get_observations
    # Network placement:
    #   _pre_physics_step : clip actions (DL ideal in v0; DL NetModule would gate them here)
    #   _apply_action     : write velocity targets (runs once per step if physics owns decimation)
    #   _get_dones        : FIRST post-physics hook -> read poses, push reports, net.step, cache out
    #                       (must run before _reset_idx so pre-reset poses and queues are used)
    #   _get_rewards      : may use self._net_out (e.g. penalise collisions under stale info)
    #   _reset_idx        : net.reset(env_ids) + MessageHistory.reset(env_ids, true state)
    #   _get_observations : server view = MessageHistory.seen + net_features(out)

    def _setup_scene(self):
        # 3.0 Direct envs (e.g. core/cartpole/cartpole_direct_env.py) declare assets in the scene
        # cfg and let InteractiveScene clone them; no manual clone_environments call needed.
        self.robots: RigidObjectCollection = self.scene["robots"]
        E, R, dev = self.scene.num_envs, NUM_ROBOTS, self.sim.device
        step_dt = self.cfg.sim.dt * self.cfg.decimation
        self.net = NetModule(NetConfig(num_envs=E, num_robots=R, device=str(dev), step_dt=step_dt,
                                       slot_dt=self.cfg.slot_dt, pose_chunks=self.cfg.pose_chunks,
                                       rung=self.cfg.net_rung, timeout_steps=20),
                             ParamRanges())
        self.uplink = MessageHistory(E, R, STATE_DIM, history_len=32, device=str(dev))
        self.goals = torch.zeros(E, R, 2, device=dev)
        self._actions = torch.zeros(E, R, 2, device=dev)
        self._net_out = None
        # static blockers (e.g. shelf columns) as spheres in env-local coords; swap for the Warp
        # mesh kernel (netmodule.los_blocked_kernel) once a warehouse USD is loaded
        self.static_blockers = torch.tensor([[4.0, 4.0, 1.0], [-4.0, 3.0, 1.0], [0.0, -5.0, 1.0]], device=dev)

    # -- pose adapter: works for 2.3 (torch) and 3.0 (ProxyArray) --------------------
    def _robot_state_local(self) -> tuple[torch.Tensor, torch.Tensor]:
        pos = self.robots.data.body_link_pos_w
        vel = self.robots.data.body_link_lin_vel_w
        pos = pos.torch if hasattr(pos, "torch") else pos          # [E,R,3] zero-copy view
        vel = vel.torch if hasattr(vel, "torch") else vel
        return pos - self.scene.env_origins[:, None, :], vel      # env-local, as NetModule expects

    def _blocked_fn(self, p_local: torch.Tensor) -> torch.Tensor:
        E, R = p_local.shape[:2]
        G = self.net.G
        a = self.net.gnb[None, None].expand(E, R, G, 3)
        b = p_local[:, :, None, :].expand(E, R, G, 3)
        blockers = torch.cat([p_local, self.static_blockers[None].expand(E, -1, -1)], dim=1)   # [E,R+M,3]
        ignore = torch.zeros(E, R, blockers.shape[1], dtype=torch.bool, device=p_local.device)
        ignore[:, torch.arange(R), torch.arange(R)] = True                                 # not yourself
        return segment_sphere_blocked(a, b, blockers, ROBOT_RADIUS, ignore)

    # -- hooks ------------------------------------------------------------------------
    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.view(self.num_envs, NUM_ROBOTS, 2).clamp(-1, 1) * self.cfg.max_speed

    def _apply_action(self):
        vel = torch.zeros(self.num_envs, NUM_ROBOTS, 6, device=self.device)
        vel[..., :2] = self._actions
        self.robots.write_body_link_velocity_to_sim_index(body_velocities=vel)

    def _get_dones(self):
        # ---- network step (first post-physics hook) ----
        pos, vel = self._robot_state_local()
        report = torch.cat([pos[..., :2], vel[..., :2]], -1)                         # [E,R,4]
        self.uplink.push(self.net.t, report)                                          # captured at env clock t
        send = ((self.net.t % self.cfg.report_period_steps) == 0)[:, None].expand(-1, NUM_ROBOTS).long()
        self._net_out = self.net.step(pos, TrafficRequest(send=send), blocked_fn=self._blocked_fn)
        self.uplink.update(self._net_out.newest_cap)
        # ---- task termination ----
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        out_of_arena = (pos[..., :2].abs() > ARENA / 2).any(-1).any(-1)
        return out_of_arena, time_out

    def _get_rewards(self) -> torch.Tensor:
        pos, _ = self._robot_state_local()
        dist = (pos[..., :2] - self.goals).norm(dim=-1)                               # [E,R]
        d = torch.cdist(pos[..., :2], pos[..., :2]) + torch.eye(NUM_ROBOTS, device=self.device) * 1e3
        collisions = (d < 2 * ROBOT_RADIUS).float().sum((-1, -2)) / 2
        return -dist.mean(-1) - 0.5 * collisions

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        super()._reset_idx(env_ids)        # scene.reset + EventTerm 'reset' (network DR) + counters
        n = env_ids.numel()
        # robots: default grid pose + jitter, zero velocity (write_*_index with env_ids)
        default = self.robots.data.default_body_pose
        default = (default.torch if hasattr(default, "torch") else default)[env_ids].clone()  # [n,R,7]
        default[..., :3] += self.scene.env_origins[env_ids, None, :]
        default[..., :2] += 0.2 * torch.randn(n, NUM_ROBOTS, 2, device=self.device)
        self.robots.write_body_link_pose_to_sim_index(body_poses=default, env_ids=env_ids)
        self.robots.write_body_link_velocity_to_sim_index(
            body_velocities=torch.zeros(n, NUM_ROBOTS, 6, device=self.device), env_ids=env_ids)
        self.goals[env_ids] = (torch.rand(n, NUM_ROBOTS, 2, device=self.device) - 0.5) * (ARENA - 2)
        # network: masked reset of every state tensor, params already resampled by the EventTerm
        self.net.reset(env_ids)
        local = default[..., :3] - self.scene.env_origins[env_ids, None, :]
        init = torch.cat([local[..., :2], torch.zeros(n, NUM_ROBOTS, 2, device=self.device)], -1)
        self.uplink.reset(env_ids, init)

    def _get_observations(self) -> dict:
        seen = self.uplink.seen                                                       # stale server view
        goal_rel = self.goals - seen[..., :2]
        if self._net_out is None:
            feats = torch.zeros(self.num_envs, NUM_ROBOTS, 5, device=self.device)
        else:
            feats = net_features(self._net_out, self.net.cfg.step_dt)
            # envs reset this step: their features describe the old episode, zero them
            fresh = self.episode_length_buf == 0
            feats = torch.where(fresh[:, None, None], torch.zeros_like(feats), feats)
        obs = torch.cat([seen, goal_rel, feats], -1).view(self.num_envs, -1)
        return {"policy": obs}
