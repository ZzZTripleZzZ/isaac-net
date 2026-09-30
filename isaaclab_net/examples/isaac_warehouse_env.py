"""The fleet task inside Isaac Lab's warehouse, with the radio map baked from the warehouse's own USD geometry.

WarehouseFleetEnv is NetFleetEnv (isaac_fleet_env.py) moved into an Isaac Lab warehouse asset
(default: Environments/Simple_Warehouse/warehouse_multiple_shelves.usd, 24 m x 38.8 m, 9.3 m high): the warehouse is
spawned in every env at the env origin, the robots are the same velocity-driven spheres, they spawn and pick goals
only in free floor cells (an occupancy grid of the warehouse geometry between 0.05 m and 1.2 m, dilated by the robot
radius), and they collide with shelves and walls. Positions are env-local (map frame = env frame, no offset).

The network: two gNBs on 7 m masts (GNB), NRConfig from net_config() with n_cells = 2 at those positions, radio =
"engine". With IsaacNetCfg.scene_map set (warehouse_isaac_cfg(scene_map=True)), net_setup exports env_0's warehouse,
bakes (or loads the cached) radio map with Sionna RT and switches the channel to "radio_map" (isaac/scene_map.py).
benchmarks/isaac/warehouse_map_demo.py runs the same scripted fleet under the baked map and under the log-distance
default and compares the delivered KPIs.
"""
from __future__ import annotations

import math
import os
from collections.abc import Sequence

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg, RigidObjectCollectionCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab_physx.sim.schemas import PhysxRigidBodyCfg

from isaaclab_net import NRConfig
from isaaclab_net.isaac import IsaacNetCfg
from isaaclab_net.isaac.mixins import rigid_positions_local
from isaaclab_net.isaac.scene_map import SceneRadioMapCfg

from . import isaac_fleet_env as fleet
from .isaac_fleet_env import (FLEET_OBS, H_LIFE, H_R, H_RATE, PDET, RADIUS, RANGE, SIREN, TASK_OBS, VMAX,
                              NetFleetEnv, NetFleetEnvCfg, physics_cfg)

WAREHOUSE_USD = f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/warehouse_multiple_shelves.usd"
AREA = (-11.5, -17.5, 11.5, 20.3)        # env-local floor rectangle inside the walls (x0, y0, x1, y1)
GNB = ((-5.0, -8.0, 7.0), (5.0, 10.0, 7.0))
SPACING = 45.0
OCC_Z = (0.05, 1.2)                      # height band that blocks a robot (sphere of radius 0.3 at z = 0.5)
OCC_RES = 0.25


def warehouse_scene_cfg(num_robots: int, num_envs: int, usd_path: str = WAREHOUSE_USD):
    x0, y0, x1, y1 = AREA
    side = math.ceil(math.sqrt(num_robots))

    def p(i):   # a spread-out initial grid; reset moves every robot to a free cell anyway
        return (x0 + (i % side + 0.5) * (x1 - x0) / side, y0 + (i // side + 0.5) * (y1 - y0) / side, 0.5)

    phys = dict(
        rigid_props=PhysxRigidBodyCfg(disable_gravity=True, solver_position_iteration_count=4,
                                      solver_velocity_iteration_count=0, linear_damping=0.0, angular_damping=0.0),
        mass_props=sim_utils.MassCfg(mass=1.0),
        collision_props=sim_utils.UsdPhysicsCollisionCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.4, 1.0)),
    )

    @configclass
    class WarehouseSceneCfg(InteractiveSceneCfg):
        warehouse = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/Warehouse", spawn=sim_utils.UsdFileCfg(usd_path=usd_path))
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

    return WarehouseSceneCfg(num_envs=num_envs, env_spacing=SPACING, replicate_physics=True)


def warehouse_isaac_cfg(scene_map: bool | SceneRadioMapCfg = True, **kw) -> IsaacNetCfg:
    """Isaac-side settings: poses of "robots" (env-local = map frame), the two gNBs, engine radio, and optionally
    the radio map baked from env_0's warehouse (True: SceneRadioMapCfg defaults for this scene)."""
    if scene_map is True:
        scene_map = SceneRadioMapCfg(root="/World/envs/env_0/Warehouse", frame="/World/envs/env_0",
                                     crop=(AREA[0] - 1, AREA[1] - 1, AREA[2] + 1, AREA[3] + 1), bounds=AREA,
                                     cell_m=0.5, samples=4_000_000, depth=5,
                                     python=os.environ.get("ISAACLAB_NET_SIONNA_PYTHON"),
                                     variant=os.environ.get("ISAACLAB_NET_SIONNA_VARIANT"))
    base = dict(pose_asset="robots", pose_offset_m=(0.0, 0.0, 0.0), gnb_pos=GNB, radio="engine",
                obs_features=FLEET_OBS, scene_map=scene_map or None)
    return IsaacNetCfg(**{**base, **kw})


def warehouse_net_config(step_dt: float, **kw) -> NRConfig:
    """fleet.net_config plus the two cells at GNB (log-distance default channel unless overridden)."""
    return fleet.net_config(step_dt).with_(n_cells=len(GNB), cell_layout="custom",
                                           cell_positions_m=tuple((g[0], g[1]) for g in GNB), **kw)


@configclass
class WarehouseFleetEnvCfg(NetFleetEnvCfg):
    episode_length_s = 30.0
    usd_path: str = WAREHOUSE_USD


def make_warehouse_cfg(num_envs: int, num_robots: int, level: str = "L2", device: str = "cuda:0",
                       backend: str = "reference", isaac: IsaacNetCfg | None = None, nr: NRConfig | None = None,
                       usd_path: str = WAREHOUSE_USD, physics: str = "isaacsim_physx") -> WarehouseFleetEnvCfg:
    cfg = WarehouseFleetEnvCfg()
    cfg.sim.physics = physics_cfg(physics)
    cfg.scene = warehouse_scene_cfg(num_robots, num_envs, usd_path)
    cfg.usd_path = usd_path
    cfg.num_robots = num_robots
    cfg.net_isaac = isaac if isaac is not None else warehouse_isaac_cfg()
    cfg.net_nr = nr if nr is not None else warehouse_net_config(cfg.sim.dt * cfg.decimation)
    per_robot = TASK_OBS + cfg.net_isaac.obs_dim(cfg.net_nr)
    cfg.action_space = num_robots * 3
    cfg.observation_space = num_robots * per_robot
    cfg.net_level = level
    cfg.net_backend = backend
    cfg.sim.device = device
    return cfg


def occupancy_from_scene(scene_dir: str, area=AREA, res=OCC_RES, z_band=OCC_Z, inflate=RADIUS + 0.2) -> np.ndarray:
    """[H, W] bool, True = blocked: cells of `area` covered by the xy bounding box of an exported triangle that
    reaches into z_band, dilated by `inflate` metres. Conservative for slanted triangles."""
    from isaaclab_net.tools.scene.mitsuba_writer import read_ply

    x0, y0, x1, y1 = area
    W, H = int(np.ceil((x1 - x0) / res)), int(np.ceil((y1 - y0) / res))
    occ = np.zeros((H, W), bool)
    mdir = os.path.join(scene_dir, "meshes")
    for f in os.listdir(mdir):
        v, t = read_ply(os.path.join(mdir, f))
        tri = v[t]                                               # [T, 3, 3]
        z = tri[..., 2]
        hit = (z.max(1) >= z_band[0]) & (z.min(1) <= z_band[1])
        lo, hi = tri[hit].min(1), tri[hit].max(1)
        i0 = np.clip(np.floor((lo[:, 1] - y0) / res), 0, H - 1).astype(int)
        i1 = np.clip(np.floor((hi[:, 1] - y0) / res), 0, H - 1).astype(int)
        j0 = np.clip(np.floor((lo[:, 0] - x0) / res), 0, W - 1).astype(int)
        j1 = np.clip(np.floor((hi[:, 0] - x0) / res), 0, W - 1).astype(int)
        inside = (hi[:, 0] >= x0) & (lo[:, 0] <= x1) & (hi[:, 1] >= y0) & (lo[:, 1] <= y1)
        for a, b, c, d in zip(i0[inside], i1[inside], j0[inside], j1[inside]):
            occ[a:b + 1, c:d + 1] = True
    k = int(np.ceil(inflate / res))
    grown = occ.copy()
    for dy in range(-k, k + 1):
        for dx in range(-k, k + 1):
            if dx * dx + dy * dy <= k * k:
                grown |= np.roll(np.roll(occ, dy, 0), dx, 1)
    grown[:k], grown[-k:], grown[:, :k], grown[:, -k:] = True, True, True, True
    return grown


class WarehouseFleetEnv(NetFleetEnv):
    cfg: WarehouseFleetEnvCfg

    def _setup_scene(self):
        # free floor cells from the warehouse geometry (the same exporter the radio map uses)
        import tempfile

        from isaaclab_net.isaac.scene_map import current_stage
        from isaaclab_net.tools.scene.usd_export import export_usd

        d = tempfile.mkdtemp(prefix="occ_")
        x0, y0, x1, y1 = AREA
        export_usd(current_stage(), d, root="/World/envs/env_0/Warehouse", frame="/World/envs/env_0",
                   crop=(x0 - 1, y0 - 1, x1 + 1, y1 + 1), z_max=OCC_Z[1] + 1.0)
        occ = occupancy_from_scene(d)
        free = np.argwhere(~occ)                                  # [N, 2] (i, j)
        self._free_xy = torch.tensor(np.stack([x0 + (free[:, 1] + 0.5) * OCC_RES, y0 + (free[:, 0] + 0.5) * OCC_RES],
                                              -1), dtype=torch.float32, device=self.device)
        self.occ_free_frac = float((~occ).mean())
        super()._setup_scene()

    def _sample_free(self, *shape) -> torch.Tensor:
        idx = torch.randint(0, self._free_xy.shape[0], shape, device=self.device)
        jit = (torch.rand(*shape, 2, device=self.device) - 0.5) * OCC_RES
        return self._free_xy[idx] + jit

    def _pos_radio(self) -> torch.Tensor:
        return rigid_positions_local(self.robots, self.scene.env_origins)[..., :2]

    def _pre_physics_step(self, actions: torch.Tensor):
        E, R = self.num_envs, self.R
        a = actions.view(E, R, 3).clamp(-1, 1)
        pos = self._pos_radio()
        self._pos0 = pos
        v = a[..., :2] * VMAX
        lo = torch.tensor(AREA[:2], device=self.device) + 0.5
        hi = torch.tensor(AREA[2:], device=self.device) - 0.5
        v = torch.where(((pos <= lo) & (v < 0)) | ((pos >= hi) & (v > 0)), torch.zeros_like(v), v)
        self._vel.zero_()
        self._vel[..., :2] = v
        self._send = torch.bucketize(a[..., 2].contiguous(), torch.tensor([-1 / 3, 1 / 3], device=self.device))

    def _get_dones(self):
        E, R, dev, t = self.num_envs, self.R, self.device, self.tt
        lo = torch.tensor(AREA[:2], device=dev)
        hi = torch.tensor(AREA[2:], device=dev)
        end = self.h_on & (t - self.h_start >= H_LIFE)
        self.h_on &= ~end
        self.known &= ~end
        spawn = ~self.h_on & (torch.rand(E, device=dev) < H_RATE)
        j = torch.randint(0, R, (E,), device=dev)
        c = self._pos0[torch.arange(E, device=dev), j] + 3 * torch.randn(E, 2, device=dev)
        self.h_pos = torch.where(spawn[:, None], torch.maximum(torch.minimum(c, hi), lo), self.h_pos)
        self.h_id = self.h_id + spawn.long()
        self.h_start = torch.where(spawn, t, self.h_start)
        self.h_on |= spawn
        self.known &= ~spawn
        send = self._send
        dist = (self._pos0 - self.h_pos[:, None, :]).norm(dim=-1)
        ci = (send - 1).clamp(min=0)
        det = (send > 0) & self.h_on[:, None] & (dist < self._rng[ci]) & (torch.rand(E, R, device=dev) < self._pdet[ci])
        if self.net is None:
            det_env = det.any(-1)
        else:
            tag = torch.where(det, self.h_id[:, None].expand(E, R), torch.full_like(send, -1))
            cur = torch.where(self.h_on, self.h_id, torch.full_like(self.h_id, -1))
            out = self.net_step(None, send, tag, cur_tag=cur)
            det_env = out["tag_delivered"]
            self.ep_stats["dlv"] += out["delivered"].float().mean(-1)
        self.known |= det_env & self.h_on
        pos = self._pos_radio()
        d_old = (self.goal - self._pos0).norm(dim=-1)
        d_new = (self.goal - pos).norm(dim=-1)
        inside = self.h_on[:, None] & ((pos - self.h_pos[:, None, :]).norm(dim=-1) < self._radius()[:, None])
        reached = d_new < 1.5
        self._rew = ((d_old - d_new) - inside.float() + 2.0 * reached.float()).mean(-1)
        self.goal = torch.where(reached[..., None], self._sample_free(E, R), self.goal)
        st = self.ep_stats
        st["expo"] += inside.float().mean(-1)
        st["goals"] += reached.float().mean(-1)
        st["s1"] += (send == 1).float().mean(-1)
        st["s2"] += (send == 2).float().mean(-1)
        self.tt += 1
        time_out = self.episode_length_buf >= self.max_episode_length
        return torch.zeros_like(time_out), time_out

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        n, R, dev = env_ids.numel(), self.R, self.device
        if n == 0:
            return
        done_stats = {k: v[env_ids].mean() for k, v in self.ep_stats.items()}
        self.extras.setdefault("log", {}).update({f"Episode/{k}": v for k, v in done_stats.items()})
        super(NetFleetEnv, self)._reset_idx(env_ids)
        pose = self.robots.data.default_body_pose
        pose = (pose.torch if hasattr(pose, "torch") else pose)[env_ids].clone()
        pose[..., :2] = self._sample_free(n, R) + self.scene.env_origins[env_ids, None, :2]
        pose[..., 2] = 0.5
        self.robots.write_body_link_pose_to_sim_index(body_poses=pose, env_ids=env_ids)
        self.robots.write_body_link_velocity_to_sim_index(body_velocities=torch.zeros(n, R, 6, device=dev),
                                                          env_ids=env_ids)
        self.goal[env_ids] = self._sample_free(n, R)
        for b in (self.h_on, self.known):
            b[env_ids] = False
        for b in (self.h_id, self.h_start, self.tt):
            b[env_ids] = 0
        for v in self.ep_stats.values():
            v[env_ids] = 0
        self.net_reset(env_ids)

    def _get_observations(self) -> dict:
        E, R = self.num_envs, self.R
        x0, y0, x1, y1 = AREA
        L = max(x1 - x0, y1 - y0)
        pos = self._pos_radio()
        kf = self.known.float()[:, None].expand(-1, R)
        hrel = (self.h_pos[:, None, :] - pos) / L * kf[..., None]
        hr = self._radius()[:, None] / L * kf
        siren = (self.h_on & (self.tt - self.h_start < SIREN)).float()[:, None].expand(-1, R)
        net = self.net_obs()
        origin = torch.tensor([x0, y0], device=self.device)
        obs = torch.cat([(pos - origin) / L, (self.goal - pos) / L, hrel] + [x[..., None] for x in (hr, siren, kf)]
                        + [net], -1)
        return {"policy": obs.reshape(E, -1)}


__all__ = ["WarehouseFleetEnv", "WarehouseFleetEnvCfg", "make_warehouse_cfg", "warehouse_isaac_cfg",
           "warehouse_net_config", "occupancy_from_scene", "AREA", "GNB", "WAREHOUSE_USD", "H_R", "RANGE", "PDET"]
