# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Tutorial 04: Isaac Lab integration
#
# The Isaac Lab layer (`isaac_net.isaac`) puts a network into a `DirectRLEnv` with four hook calls. Most of it
# does not import Isaac Lab: `NetModule`, `MessageHistory` and the `NetEnvMixin` hooks are plain PyTorch, so
# this tutorial runs them on a CPU with a stand-in environment.
#
# > **Part of this tutorial needs Isaac Lab.** The `DirectRLEnv` class in the section "The mixin in a real
# > DirectRLEnv" needs Isaac Lab 3.0 and Isaac Sim 6.1. It is shown as a code listing and is **not executed**
# > here. Every executable cell runs without Isaac.
#
# For readers from networking: a `DirectRLEnv` steps `E` copies of a scene in parallel on one GPU. Each call of
# `step(actions)` runs the policy's actions through a few physics substeps and then calls the task's hooks, in
# this order: `_pre_physics_step`, physics, `_get_dones`, `_get_rewards`, `_reset_idx` for the environments that
# finished, and `_get_observations`.

# %%
import torch

from isaac_net import NRConfig
from isaac_net.isaac import MessageHistory, NetEnvMixin, NetModule, TrafficRequest

torch.manual_seed(0)
E, R, dev = 4, 3, "cpu"

# %% [markdown]
# ## NetModule: the network of one environment batch
#
# `NetModule(level, E, R, device, config, backend)` wraps any level of `make_engine` and adds three things a task
# needs:
#
# 1. **An Isaac-side radio.** Poses become SNR through `IsaacRadio`, with per-environment radio parameters for
#    domain randomization, several gNBs and optional line-of-sight blockage. The SNR of a control step is the
#    average over `pose_chunks` poses interpolated between the previous and the current end-of-step pose.
# 2. **A per-message tag.** `TrafficRequest(send, tag)` carries a label per message, such as the id of the
#    hazard a camera frame captured. `step(..., cur_tag)` returns `tag_delivered [E]`, which is true when a
#    message with the environment's current tag arrived.
# 3. **Freshness outputs.** `last_cap` is the newest capture step delivered this episode, and `aoi_s` is its age
#    in seconds at the end of the step.

# %%
net = NetModule("L2-legacy", E, R, dev, NRConfig(msg_sizes=(4000.0, 30000.0)), backend="reference", seed=0)

pos_end = torch.rand(E, R, 3) * 60.0                   # end-of-step poses from the simulator, env-local metres
cur_tag = torch.tensor([7, -1, -1, -1])                 # env 0 has an active hazard with id 7
send = torch.randint(0, 3, (E, R))                      # 0 nothing, 1 small, 2 large
send[0, 1] = 2                                          # robot 1 of env 0 sends a large frame ...
tag = torch.full((E, R), -1)                            # -1: no tag
tag[0, 1] = 7                                           # ... that captured hazard 7
net.submit(None, TrafficRequest(send=send, tag=tag))
for k in range(5):
    out = net.step(None, pos_end, cur_tag=cur_tag)
    print(f"step {k}: tag delivered per env {out['tag_delivered'].tolist()}, "
          f"frames queued in env 0: {out['queue_len'][0].tolist()}")
    net.submit(None, TrafficRequest(send=torch.zeros(E, R, dtype=torch.long)))
print(sorted(out))
print("AoI (s):\n", out["aoi_s"])

# %% [markdown]
# The tagged frame of robot 1 needed three control steps, and `tag_delivered` is `True` only in the step in which
# it arrived. Every message here was captured at step 0 and nothing newer was sent, so at the end of the fifth
# step the freshest information about every robot is 5 control steps old, which is the `aoi_s` of 0.5 s.
#
# `step` takes the poses at the **end** of the control step, and the messages submitted before it are treated as
# captured at the **start** of the step. This ordering matters. If the post-physics pose were stamped as the
# capture of step `t`, every delay and AoI would come out one control step too optimistic.
#
# ## MessageHistory: what the receiver sees
#
# A task usually needs the *content* that arrived, not only the fact that something arrived. `MessageHistory`
# stores each robot's payload by capture step, and after every network step `update(out["newest_cap"])` gives
# the receiver's current view: robots whose newest delivered capture is fresher get the stored payload, and the
# others keep the last payload they had.

# %%
hist = MessageHistory(E, R, dim=2, history_len=32, device=dev)
hist.reset(torch.arange(E), torch.zeros(E, R, 2))       # the receiver knows the state at reset
xy = torch.rand(E, R, 2) * 60.0
for _ in range(10):
    hist.push(net.clock, xy)                            # payload captured at the start-of-step pose
    net.submit(None, TrafficRequest(send=torch.ones(E, R, dtype=torch.long)))
    xy = xy + torch.randn_like(xy)
    out = net.step(None, torch.cat([xy, torch.zeros(E, R, 1)], -1))
    seen = hist.update(out["newest_cap"])
print("true positions of env 0:\n", xy[0])
print("what the receiver of env 0 knows:\n", seen[0])

# %% [markdown]
# ## NetEnvMixin with a stand-in environment
#
# `NetEnvMixin` adds `net_setup`, `net_step`, `net_reset` and `net_obs` to a `DirectRLEnv`. It needs only
# `num_envs`, `device` and `episode_length_buf` from the environment, so a small stand-in class shows the whole
# lifecycle on a CPU.

# %%
class StandInEnv(NetEnvMixin):
    num_envs, device = E, dev
    episode_length_buf = torch.ones(E, dtype=torch.long)


env = StandInEnv()
env.net_setup("L1", R, NRConfig(), "reference", pose_chunks=2, seed=0)
for _ in range(5):
    out = env.net_step(torch.rand(E, R, 3) * 60.0, torch.ones(E, R, dtype=torch.long))
obs = env.net_obs()
print("net_obs shape:", tuple(obs.shape), "(AoI, SNR, queued, delivered), scaled to about [0, 1]")
env.net_reset(torch.tensor([2]))
print("clock after resetting env 2:", env.net.clock.tolist())

# %% [markdown]
# `net_setup("off")` removes the network: `net_step` returns `None` and `net_obs` returns zeros, which gives the
# ideal-link baseline with the same observation shape.
#
# ## The mixin in a real DirectRLEnv
#
# > **Not executed: needs Isaac Lab 3.0.** The listing below is the pattern of
# > `isaac_net/examples/isaac_fleet_env.py`, the complete demo env. Run it inside an Isaac Lab installation
# > (see `docs/isaac-lab.md`).
#
# ```python
# from isaaclab.envs import DirectRLEnv
# from isaac_net import NRConfig
# from isaac_net.isaac import NetEnvMixin, rigid_positions_local
#
# class MyFleetEnv(NetEnvMixin, DirectRLEnv):
#     def _setup_scene(self):
#         ...                                         # self.robots: a RigidObjectCollection of R robots
#         self.net_setup("L2-legacy", R, NRConfig(msg_sizes=(4000.0, 30000.0)), backend="triton")
#
#     def _pre_physics_step(self, actions):
#         ...                                         # read the START-of-step pose, decide what to send
#         self.send = choose_messages(actions)        # [E,R] long: 0 nothing, 1 small, 2 large
#
#     def _get_dones(self):
#         pos = rigid_positions_local(self.robots, self.scene.env_origins)   # END-of-step poses [E,R,3]
#         out = self.net_step(pos, self.send)         # the first post-physics hook, before _reset_idx
#         ...
#
#     def _reset_idx(self, env_ids):
#         super()._reset_idx(env_ids)
#         ...
#         self.net_reset(env_ids)                     # partial reset of the network state of these envs
#
#     def _get_observations(self):
#         net = self.net_obs()                        # [E,R,4]: AoI, SNR, queued frames, delivered
#         ...
# ```
#
# `net_step` belongs in `_get_dones` because that is the first hook after physics: it sees the end-of-step poses
# and the queues before `_reset_idx` clears the finished environments. `net_obs` returns zeros for environments
# that reset in this step, since their last network output belongs to the previous episode.
#
# ## Network domain randomization
#
# `isaac_net.isaac.mdp.randomize_network` is an Isaac Lab event term that redraws per-environment radio
# parameters (transmit power, noise, path-loss constant and exponent, shadowing, blockage loss). Registered in
# mode `"reset"`, it runs inside `_reset_idx` before `net_reset`, and `net_reset` never touches the parameters,
# so the new values hold for the next episode:
#
# ```python
# from isaaclab.managers import EventTermCfg
# from isaac_net.isaac.mdp import randomize_network
#
# randomize_net = EventTermCfg(func=randomize_network, mode="reset",
#                              params={"ranges": {"pl_exp": (2.8, 4.0), "shadow_sigma_db": (3.0, 8.0)}})
# ```
#
# The same draw works without Isaac through `NetModule.sample_params`:

# %%
net.sample_params(torch.tensor([0, 1]), {"pl_exp": (2.8, 4.0), "shadow_sigma_db": (3.0, 8.0)})
print("per-env path-loss exponent:", [round(x, 2) for x in net.radio.params()["pl_exp"].tolist()])

# %% [markdown]
# ## Choosing a level and backend in Isaac
#
# `net_setup` takes any level of `make_engine` and the same backends. For training at scale, use `L2-legacy` or
# `L1` on the `triton` backend. Use `graph` when you need results that are bitwise identical to the reference
# engine, and `reference` for debugging. The NR engine (`L2`) runs on `reference` only. A multi-cell config needs
# the engine's own radio: pass `radio="engine"` to `net_setup`.
