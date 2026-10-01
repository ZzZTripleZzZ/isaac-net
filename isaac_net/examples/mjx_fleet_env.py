"""MuJoCo Playground / MJX env: E envs x R velocity-actuated planar bodies, 5G uplink in the loop.

The MJX counterpart of isaac_net/examples/isaac_fleet_env.py (same task, same network module):
  - 150 m x 150 m arena per env, gNB at the arena corner (radio coords (0, 0), 6 m mast)
  - one MJX model per env with R bodies; each body has two slide joints (x, y) and two velocity actuators
    (kv = 20, mass 1 kg), gravity off, no contacts; dt = 1/50 s, 5 substeps -> 0.1 s control step (40 UL slots)
  - action per robot: planar velocity in [-1,1]^2 (x 3 m/s) + send choice (none / small / large), the third
    channel bucketed at -1/3 and +1/3; velocity components pointing out of the arena are cancelled
  - hazards spawn near a random robot, grow, last 10 s; a frame detects with p = 0.6 / 1.0 within 25 / 50 m;
    when a detecting frame is DELIVERED the fleet learns the hazard location
  - obs per robot (12, the Isaac layout): pos, goal offset, known-hazard offset and radius, siren, queued frames,
    AoI, SNR, known flag
Network: net_level "off" (ideal: detections known the same step, zero network features) or any make_engine level
on any backend it has, through isaac_net.mjx.NetModuleMJX (one torch NetModule for the E envs, stepped from
inside the vmapped, jitted env step by a buffer_callback).

Conventions (Playground): reset(rng) and step(state, action) act on ONE env and are vmapped by the Brax wrappers
(mujoco_playground.wrapper.wrap_for_brax_training). The network is built for num_envs envs, so wrap the env for
that batch size; a Brax eval env is a second MJXFleetEnv with num_envs = num_eval_envs.
Resets: BraxAutoResetWrapper restores the cached first mjx.Data / obs of a done env and keeps `info`. data.time is
0 only in data that come out of reset(), so the step starts by restoring the task state of reset() (info["task0"])
and resetting the env's network when data.time == 0. Frames are captured at the start-of-step pose; the network
steps with the end-of-step pose (the Isaac layer's order).
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jnp
import mujoco
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground._src import mjx_env

from isaac_net import NRConfig

ARENA = 150.0
VMAX = 3.0                    # m/s
RADIUS = 0.3
F_DEPTH = 16
TIMEOUT = 20
SIZES = (4000.0, 30000.0)     # T1 small / large frame bytes
H_R, H_GROW, H_LIFE, H_RATE, SIREN = 15.0, 0.3, 100, 1 / 80, 10
RANGE = (25.0, 50.0)
PDET = (0.6, 1.0)
OBS_PER_ROBOT = 12
TASK_KEYS = ("goal", "h_on", "h_pos", "h_id", "h_start", "known", "tt")


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        num_envs=256,
        num_robots=16,
        net_level="L2-legacy",     # "off" or a make_engine level
        net_backend="triton",      # reference | eager | graph | compile | triton
        net_seed=0,
        net_record=False,          # keep the network inputs for isaac_net.mjx.replay (validation)
        pose_chunks=4,
        device="cuda",
        ctrl_dt=0.1,
        sim_dt=0.02,
        episode_length=300,
        action_repeat=1,
        impl="warp",               # MJX implementation: "warp" (Playground's default) or "jax"
        naconmax=0,
        njmax=8,                   # the warp impl needs njmax > 0 even without constraints (with 0 nothing moves)
        kv=20.0,
    )


def fleet_xml(num_robots: int, kv: float, sim_dt: float) -> str:
    bodies, acts = [], []
    for i in range(num_robots):
        bodies.append(f'<body name="robot_{i}" pos="0 0 0.5">'
                      f'<joint name="x_{i}" type="slide" axis="1 0 0"/><joint name="y_{i}" type="slide" axis="0 1 0"/>'
                      f'<geom type="sphere" size="{RADIUS}" mass="1" contype="0" conaffinity="0"/></body>')
        acts.append(f'<velocity joint="x_{i}" kv="{kv}"/><velocity joint="y_{i}" kv="{kv}"/>')
    return (f'<mujoco model="fleet"><option timestep="{sim_dt}" gravity="0 0 0" integrator="implicitfast"/>'
            f'<worldbody>{"".join(bodies)}</worldbody><actuator>{"".join(acts)}</actuator></mujoco>')


def net_config(step_dt: float) -> NRConfig:
    """The network configuration of the task: T1 frame sizes, 16-frame buffer, 2 s timeout, one gNB."""
    return NRConfig(msg_sizes=SIZES, frame_buffer=F_DEPTH, timeout_steps=TIMEOUT, control_step_ms=step_dt * 1000.0)


class MJXFleetEnv(mjx_env.MjxEnv):
    """Fleet task with the network in the loop (module docstring)."""

    def __init__(self, config: config_dict.ConfigDict = None,
                 config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None):
        super().__init__(config if config is not None else default_config(), config_overrides)
        c = self._config
        self.R = int(c.num_robots)
        self._mj_model = mujoco.MjModel.from_xml_string(fleet_xml(self.R, c.kv, c.sim_dt))
        self._mjx_model = mjx.put_model(self._mj_model, impl=c.impl)
        self.net = None
        if c.net_level not in (None, "off"):
            from isaac_net.mjx import NetModuleMJX
            self.net = NetModuleMJX(c.net_level, c.num_envs, self.R, c.device, net_config(self.dt), c.net_backend,
                                    seed=c.net_seed, record=c.net_record, pose_chunks=c.pose_chunks,
                                    gnb_pos=((0.0, 0.0, 6.0),))

    # ---------------------------------------------------------------- Playground API
    @property
    def xml_path(self) -> str:
        return "<generated>"

    @property
    def action_size(self) -> int:
        return 3 * self.R

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model

    def _make_data(self, qpos):
        c = self._config
        data = mjx_env.make_data(self._mj_model, qpos=qpos, impl=self._mjx_model.impl.value,
                                 naconmax=c.naconmax, njmax=c.njmax)
        return mjx.forward(self._mjx_model, data)

    def reset(self, rng: jax.Array) -> mjx_env.State:
        R = self.R
        rng, k1, k2 = jax.random.split(rng, 3)
        xy = jax.random.uniform(k1, (R, 2), minval=1.0, maxval=ARENA - 1.0)
        data = self._make_data(xy.reshape(-1))
        task = dict(goal=jax.random.uniform(k2, (R, 2), maxval=ARENA), h_on=jnp.bool_(False), h_pos=jnp.zeros(2),
                    h_id=jnp.int32(0), h_start=jnp.int32(0), known=jnp.bool_(False), tt=jnp.int32(0))
        info = dict(rng=rng, task=task, task0=task, net=self._net_zeros())
        obs = self._obs(xy, task, jnp.zeros((R, 4)))
        metrics = {"delivered": jnp.zeros(()), "exposure": jnp.zeros(()), "goals": jnp.zeros(()),
                   "sent": jnp.zeros(())}
        return mjx_env.State(data, obs, jnp.zeros(()), jnp.zeros(()), metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        R = self.R
        fresh = state.data.time == 0.0                     # data straight out of reset() (also after autoreset)
        task = jax.tree.map(lambda a, b: jnp.where(fresh, a, b), state.info["task0"], state.info["task"])
        rng, k_sp, k_j, k_c, k_det, k_goal = jax.random.split(state.info["rng"], 6)
        a = jnp.clip(action.reshape(R, 3), -1.0, 1.0)
        pos0 = state.data.qpos.reshape(R, 2)                # capture pose (start of step)
        v = a[:, :2] * VMAX
        out = ((pos0 <= 0.5) & (v < 0)) | ((pos0 >= ARENA - 0.5) & (v > 0))
        v = jnp.where(out, 0.0, v)
        data = mjx_env.step(self._mjx_model, state.data, v.reshape(-1), self.n_substeps)
        pos = data.qpos.reshape(R, 2)                       # end-of-step pose
        send = (a[:, 2] > -1 / 3).astype(jnp.int32) + (a[:, 2] > 1 / 3).astype(jnp.int32)

        # ---- hazard lifecycle at step start (as the Isaac env)
        t = task["tt"]
        end = task["h_on"] & (t - task["h_start"] >= H_LIFE)
        h_on = task["h_on"] & ~end
        known = task["known"] & ~end
        spawn = ~h_on & (jax.random.uniform(k_sp) < H_RATE)
        j = jax.random.randint(k_j, (), 0, R)
        c = jnp.clip(pos0[j] + 10.0 * jax.random.normal(k_c, (2,)), 0.0, ARENA)
        h_pos = jnp.where(spawn, c, task["h_pos"])
        h_id = task["h_id"] + spawn.astype(jnp.int32)
        h_start = jnp.where(spawn, t, task["h_start"])
        h_on = h_on | spawn
        known = known & ~spawn
        # ---- frames captured at the start-of-step pose
        dist = jnp.linalg.norm(pos0 - h_pos, axis=-1)
        ci = jnp.clip(send - 1, 0, 1)
        det = (send > 0) & h_on & (dist < jnp.asarray(RANGE)[ci]) & \
            (jax.random.uniform(k_det, (R,)) < jnp.asarray(PDET)[ci])
        if self.net is None:
            det_env = det.any()
            net = self._net_zeros()
        else:
            tag = jnp.where(det, h_id, -1)
            cur = jnp.where(h_on, h_id, -1)
            p3 = jnp.concatenate([pos, jnp.full((R, 1), 0.5)], -1)
            net = self.net(p3, send, tag, cur, fresh)
            det_env = net["tag_delivered"]
        known = known | (det_env & h_on)
        # ---- reward (Isaac env / env.py): progress, hazard exposure, goals
        goal = task["goal"]
        d_old = jnp.linalg.norm(goal - pos0, axis=-1)
        d_new = jnp.linalg.norm(goal - pos, axis=-1)
        radius = jnp.clip((t - h_start + 1) * H_GROW, max=H_R) * h_on
        inside = h_on & (jnp.linalg.norm(pos - h_pos, axis=-1) < radius)
        reached = d_new < 3.0
        reward = ((d_old - d_new) - inside + 2.0 * reached).mean()
        goal = jnp.where(reached[:, None], jax.random.uniform(k_goal, (R, 2), maxval=ARENA), goal)
        task = dict(goal=goal, h_on=h_on, h_pos=h_pos, h_id=h_id, h_start=h_start, known=known, tt=t + 1)
        info = dict(state.info, rng=rng, task=task, net=net)
        obs = self._obs(pos, task, net["feats"])
        metrics = dict(state.metrics)                      # keep keys the wrappers added (Brax adds "reward")
        metrics.update(delivered=net["delivered"].mean(dtype=jnp.float32), exposure=inside.mean(dtype=jnp.float32),
                       goals=reached.mean(dtype=jnp.float32), sent=(send > 0).mean(dtype=jnp.float32))
        done = jnp.isnan(data.qpos).any().astype(jnp.float32)
        return mjx_env.State(data, obs, reward, done, metrics, info)

    # ---------------------------------------------------------------- helpers
    def _net_zeros(self) -> dict:
        from isaac_net.mjx.net_module import NET_OUTPUTS
        return {k: jnp.zeros(f(self.R), dt) for k, (f, dt) in NET_OUTPUTS.items()}

    def _obs(self, pos, task, feats) -> jax.Array:
        L, R = ARENA, self.R
        kf = jnp.broadcast_to(task["known"].astype(jnp.float32), (R,))
        hrel = (task["h_pos"] - pos) / L * kf[:, None]
        radius = jnp.clip((task["tt"] - task["h_start"] + 1) * H_GROW, max=H_R) * task["h_on"]
        hr = jnp.broadcast_to(radius / L, (R,)) * kf
        siren = jnp.broadcast_to((task["h_on"] & (task["tt"] - task["h_start"] < SIREN)).astype(jnp.float32), (R,))
        cols = [pos / L, (task["goal"] - pos) / L, hrel] + \
            [x[:, None] for x in (hr, siren, feats[:, 2], feats[:, 0], feats[:, 1], kf)]
        return jnp.concatenate(cols, -1).reshape(R * OBS_PER_ROBOT)


__all__ = ["MJXFleetEnv", "default_config", "net_config", "fleet_xml", "OBS_PER_ROBOT"]
