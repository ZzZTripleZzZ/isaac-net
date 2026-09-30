"""Isaac Lab 3.0 DirectRLEnv: E envs x R velocity-driven rigid spheres, 5G uplink in the loop.

Mirrors isaaclab_net/examples/fleet_task.py (FleetEnv, task T1) inside Isaac Lab:
  - 150 m x 150 m arena per env, gNB at the arena corner (radio coords (0,0), 6 m mast)
  - action per robot: planar velocity in [-1,1]^2 (x 3 m/s) + send choice (none / small / large),
    the send choice is the third action channel bucketed at -1/3 and +1/3
  - hazards spawn near a random robot, grow, last 10 s; a frame detects with p = 0.6 / 1.0 within
    25 / 50 m; when a detecting frame is DELIVERED the whole fleet learns the hazard location
  - obs per robot: 9 task values (pos, goal offset, known-hazard offset and radius, siren, known flag) followed by
    the network features selected by IsaacNetCfg.obs_features (default queued frames, age of information, SNR:
    12 values per robot); make_cfg sizes the spaces with IsaacNetCfg.obs_dim
Network levels: "off" (ideal: detections known the same step, no network features) or any make_engine level
("L0", "L0DR", "L1", "L2-legacy", "L2"), on any backend the level has ("reference", "eager", "graph", "compile",
"triton" for L1 / L2-legacy; the NR engine "L2" has "reference" only). For scale use L2-legacy on triton.
The network is wired through isaaclab_net.isaac.NetEnvMixin (net_setup / net_step / net_reset / net_obs) and
configured by one NRConfig (message sizes, frame buffer, timeout, control step; net_config()) plus an IsaacNetCfg
(fleet_isaac_cfg(): poses read from the "robots" collection, gNB on a 6 m mast at the arena corner).

Physics: PhysX via Isaac Sim, dt = 1/50 s, decimation 5 -> 0.1 s control step (K = 40 UL slots).
Spheres have gravity disabled and float at z = 0.5 m, so velocity writes are not fought by friction.
Robot-robot contacts stay enabled.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg, RigidObjectCollection, RigidObjectCollectionCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab_physx.physics import PhysxCfg
from isaaclab_physx.sim.schemas import PhysxRigidBodyCfg

from isaaclab_net import NRConfig
from isaaclab_net.isaac import IsaacNetCfg
from isaaclab_net.isaac.mixins import NetEnvMixin, rigid_positions_local

ARENA = 150.0
HALF = ARENA / 2
VMAX = 3.0                    # m/s (0.3 m per 100 ms step, as env.py)
RADIUS = 0.3
F_DEPTH = 16
TIMEOUT = 20
SIZES = (4000.0, 30000.0)     # T1 small / large frame bytes
# hazard constants from env.py
H_R, H_GROW, H_LIFE, H_RATE, SIREN = 15.0, 0.3, 100, 1 / 80, 10
RANGE = (25.0, 50.0)
PDET = (0.6, 1.0)
TASK_OBS = 9                  # task values per robot; the network features follow
FLEET_OBS = ("queue_len", "aoi", "sinr")
OBS_PER_ROBOT = TASK_OBS + len(FLEET_OBS)     # 12 with the default network features


def make_scene_cfg(num_robots: int, num_envs: int):
    side = math.ceil(math.sqrt(num_robots))
    sp = ARENA / (side + 1)

    def p(i):
        return ((i % side + 1) * sp - HALF, (i // side + 1) * sp - HALF, 0.5)

    phys = dict(
        rigid_props=PhysxRigidBodyCfg(disable_gravity=True, solver_position_iteration_count=4,
                                      solver_velocity_iteration_count=0, linear_damping=0.0, angular_damping=0.0),
        mass_props=sim_utils.MassCfg(mass=1.0),
        collision_props=sim_utils.UsdPhysicsCollisionCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.4, 1.0)),
    )

    @configclass
    class FleetSceneCfg(InteractiveSceneCfg):
        ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
        light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=2000.0))
        robots: RigidObjectCollectionCfg = RigidObjectCollectionCfg(
            rigid_objects={
                f"robot_{i}": RigidObjectCfg(
                    prim_path=f"/World/envs/env_.*/Robot_{i}",
                    spawn=sim_utils.SphereCfg(radius=RADIUS, **phys),
                    init_state=RigidObjectCfg.InitialStateCfg(pos=p(i)),
                )
                for i in range(num_robots)
            }
        )

    return FleetSceneCfg(num_envs=num_envs, env_spacing=ARENA + 10.0, replicate_physics=True)


@configclass
class NetFleetEnvCfg(DirectRLEnvCfg):
    decimation = 5
    episode_length_s = 30.0                       # 300 control steps, as env.py
    sim: SimulationCfg = SimulationCfg(dt=1 / 50, render_interval=5, physics=PhysxCfg())
    scene: InteractiveSceneCfg = None             # filled by make_cfg
    num_robots: int = 16
    action_space = 16 * 3
    observation_space = 16 * OBS_PER_ROBOT
    state_space = 0
    net_level: str = "L2-legacy"                  # "off" | "L0" | "L0DR" | "L1" | "L2-legacy" | "L2"
    net_backend: str = "graph"                    # reference | eager | graph | compile | triton
    net_nr: NRConfig | None = None                # None: net_config(step of the network)
    net_isaac: IsaacNetCfg | None = None          # None: fleet_isaac_cfg()
    net_seed: int | None = None                   # engine and radio generators (reset draws); None = random


def fleet_isaac_cfg(**kw) -> IsaacNetCfg:
    """Isaac-side network settings of the task: poses of the "robots" collection shifted into the [0,150]^2 radio
    frame, gNB on a 6 m mast at the arena corner, network features FLEET_OBS. kw overrides IsaacNetCfg fields."""
    base = dict(pose_asset="robots", pose_offset_m=(HALF, HALF, 0.0), gnb_pos=((0.0, 0.0, 6.0),),
                obs_features=FLEET_OBS)
    return IsaacNetCfg(**{**base, **kw})


def make_cfg(num_envs: int, num_robots: int, level: str, device: str = "cuda:0", backend: str = "graph",
             isaac: IsaacNetCfg | None = None, nr: NRConfig | None = None) -> NetFleetEnvCfg:
    cfg = NetFleetEnvCfg()
    cfg.scene = make_scene_cfg(num_robots, num_envs)
    cfg.num_robots = num_robots
    cfg.net_isaac = isaac or fleet_isaac_cfg()
    cfg.net_nr = nr
    per_robot = TASK_OBS + cfg.net_isaac.obs_dim(nr or net_config(0.1))
    cfg.action_space = num_robots * 3
    cfg.observation_space = num_robots * per_robot
    cfg.net_level = level
    cfg.sim.device = device
    cfg.net_backend = backend
    return cfg


def net_config(step_dt: float) -> NRConfig:
    """The network configuration of the task: T1 frame sizes, 16-frame buffer, 2 s timeout, one gNB. step_dt is
    the network control step."""
    return NRConfig(msg_sizes=SIZES, frame_buffer=F_DEPTH, timeout_steps=TIMEOUT, control_step_ms=step_dt * 1000.0)


class NetFleetEnv(NetEnvMixin, DirectRLEnv):
    cfg: NetFleetEnvCfg

    # ---------------------------------------------------------------- setup
    def _setup_scene(self):
        E, R, dev = self.scene.num_envs, self.cfg.num_robots, self.device
        self.R = R
        self.robots: RigidObjectCollection = self.scene["robots"]
        isaac = self.cfg.net_isaac or fleet_isaac_cfg()
        step_dt = self.cfg.sim.dt * self.cfg.decimation * isaac.net_decimation / isaac.net_substeps   # network step
        self.net_setup(self.cfg.net_level, R, self.cfg.net_nr or net_config(step_dt), self.cfg.net_backend,
                       isaac=isaac, seed=self.cfg.net_seed)
        self._rng = torch.tensor(RANGE, device=dev)
        self._pdet = torch.tensor(PDET, device=dev)
        self.tt = torch.zeros(E, dtype=torch.long, device=dev)            # per-env control-step clock
        self.goal = torch.zeros(E, R, 2, device=dev)                      # radio coords [0,150]
        self.h_on = torch.zeros(E, dtype=torch.bool, device=dev)
        self.h_pos = torch.zeros(E, 2, device=dev)
        self.h_id = torch.zeros(E, dtype=torch.long, device=dev)
        self.h_start = torch.zeros(E, dtype=torch.long, device=dev)
        self.known = torch.zeros(E, dtype=torch.bool, device=dev)
        self._vel = torch.zeros(E, R, 6, device=dev)
        self._send = torch.zeros(E, R, dtype=torch.long, device=dev)
        self._pos0 = torch.zeros(E, R, 2, device=dev)
        self._rew = torch.zeros(E, device=dev)
        self.ep_stats = {k: torch.zeros(E, device=dev) for k in ("expo", "goals", "s1", "s2", "dlv")}

    # ---------------------------------------------------------------- pose adapter
    def _pos_radio(self) -> torch.Tensor:
        """[E,R,2] robot positions in radio/task coordinates: env-local shifted so the arena is [0,150]^2."""
        return rigid_positions_local(self.robots, self.scene.env_origins)[..., :2] + HALF

    # ---------------------------------------------------------------- hooks
    def _pre_physics_step(self, actions: torch.Tensor):
        E, R = self.num_envs, self.R
        a = actions.view(E, R, 3).clamp(-1, 1)
        pos = self._pos_radio()
        self._pos0 = pos                                                   # capture pose at start of step
        v = a[..., :2] * VMAX
        # keep robots inside the arena: cancel velocity components pointing out of it
        out_lo = (pos <= 0.5) & (v < 0)
        out_hi = (pos >= ARENA - 0.5) & (v > 0)
        v = torch.where(out_lo | out_hi, torch.zeros_like(v), v)
        self._vel.zero_()
        self._vel[..., :2] = v
        self._send = torch.bucketize(a[..., 2].contiguous(), torch.tensor([-1 / 3, 1 / 3], device=self.device))

    def _apply_action(self):
        self.robots.write_body_link_velocity_to_sim_index(body_velocities=self._vel)

    def _radius(self):
        return ((self.tt - self.h_start + 1).float() * H_GROW).clamp(max=H_R) * self.h_on.float()

    def _get_dones(self):
        E, R, dev, t = self.num_envs, self.R, self.device, self.tt
        # ---- hazard lifecycle at step start (as env.py)
        end = self.h_on & (t - self.h_start >= H_LIFE)
        self.h_on &= ~end
        self.known &= ~end
        spawn = ~self.h_on & (torch.rand(E, device=dev) < H_RATE)
        j = torch.randint(0, R, (E,), device=dev)
        c = self._pos0[torch.arange(E, device=dev), j] + 10 * torch.randn(E, 2, device=dev)
        self.h_pos = torch.where(spawn[:, None], c.clamp(0, ARENA), self.h_pos)
        self.h_id = self.h_id + spawn.long()
        self.h_start = torch.where(spawn, t, self.h_start)
        self.h_on |= spawn
        self.known &= ~spawn
        # ---- frames captured at the start-of-step pose
        send = self._send
        dist = (self._pos0 - self.h_pos[:, None, :]).norm(dim=-1)
        ci = (send - 1).clamp(min=0)
        det = (send > 0) & self.h_on[:, None] & (dist < self._rng[ci]) & (torch.rand(E, R, device=dev) < self._pdet[ci])
        if self.net is None:
            det_env = det.any(-1)                                          # ideal network
        else:
            tag = torch.where(det, self.h_id[:, None].expand(E, R), torch.full_like(send, -1))
            # network step: end-of-step poses read from the "robots" collection (IsaacNetCfg.pose_asset),
            # interpolated internally
            cur = torch.where(self.h_on, self.h_id, torch.full_like(self.h_id, -1))
            out = self.net_step(None, send, tag, cur_tag=cur)
            det_env = out["tag_delivered"]
            self.ep_stats["dlv"] += out["delivered"].float().mean(-1)
        self.known |= det_env & self.h_on
        # ---- reward (env.py): progress, hazard exposure, goals
        pos = self._pos_radio()
        d_old = (self.goal - self._pos0).norm(dim=-1)
        d_new = (self.goal - pos).norm(dim=-1)
        inside = self.h_on[:, None] & ((pos - self.h_pos[:, None, :]).norm(dim=-1) < self._radius()[:, None])
        reached = d_new < 3.0
        self._rew = ((d_old - d_new) - inside.float() + 2.0 * reached.float()).mean(-1)
        self.goal = torch.where(reached[..., None], torch.rand(E, R, 2, device=dev) * ARENA, self.goal)
        st = self.ep_stats
        st["expo"] += inside.float().mean(-1)
        st["goals"] += reached.float().mean(-1)
        st["s1"] += (send == 1).float().mean(-1)
        st["s2"] += (send == 2).float().mean(-1)
        self.tt += 1
        time_out = self.episode_length_buf >= self.max_episode_length
        return torch.zeros_like(time_out), time_out

    def _get_rewards(self) -> torch.Tensor:
        return self._rew

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        n, R, dev = env_ids.numel(), self.R, self.device
        if n == 0:
            return
        done_stats = {k: v[env_ids].mean() for k, v in self.ep_stats.items()}
        self.extras.setdefault("log", {}).update({f"Episode/{k}": v for k, v in done_stats.items()})
        super()._reset_idx(env_ids)
        # uniform random start poses in the arena (as env.py), zero velocity
        pose = self.robots.data.default_body_pose
        pose = (pose.torch if hasattr(pose, "torch") else pose)[env_ids].clone()        # [n,R,7]
        xy = torch.rand(n, R, 2, device=dev) * (ARENA - 2) + 1
        pose[..., :2] = xy - HALF + self.scene.env_origins[env_ids, None, :2]
        pose[..., 2] = 0.5
        self.robots.write_body_link_pose_to_sim_index(body_poses=pose, env_ids=env_ids)
        self.robots.write_body_link_velocity_to_sim_index(body_velocities=torch.zeros(n, R, 6, device=dev),
                                                          env_ids=env_ids)
        self.goal[env_ids] = torch.rand(n, R, 2, device=dev) * ARENA
        for b in (self.h_on, self.known):
            b[env_ids] = False
        for b in (self.h_id, self.h_start, self.tt):
            b[env_ids] = 0
        for v in self.ep_stats.values():
            v[env_ids] = 0
        self.net_reset(env_ids)

    def _get_observations(self) -> dict:
        E, R, L = self.num_envs, self.R, ARENA
        pos = self._pos_radio()
        kf = self.known.float()[:, None].expand(-1, R)
        hrel = (self.h_pos[:, None, :] - pos) / L * kf[..., None]
        hr = self._radius()[:, None] / L * kf
        siren = (self.h_on & (self.tt - self.h_start < SIREN)).float()[:, None].expand(-1, R)
        net = self.net_obs()                                               # [E,R,obs_dim]: IsaacNetCfg.obs_features
        obs = torch.cat([pos / L, (self.goal - pos) / L, hrel] + [x[..., None] for x in (hr, siren, kf)] + [net], -1)
        return {"policy": obs.reshape(E, -1)}
