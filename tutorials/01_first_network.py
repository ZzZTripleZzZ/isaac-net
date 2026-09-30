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
# # Tutorial 01: Your first network
#
# This tutorial builds one network engine for a small batch of environments, sends messages through it, reads
# the delivery outcomes (delay, age of information and SINR) and resets a subset of environments. It runs on a
# CPU in a few seconds.
#
# The engine simulates the 5G uplink of `E` environments with `R` robots each. Every call to `step` advances all
# environments by one control step (100 ms of simulated time by default), which matches one policy step of a
# typical robot-learning task.

# %%
import torch

from isaaclab_net import NRConfig, Requests, make_engine

torch.manual_seed(0)                      # stepping draws come from the global torch RNG
E, R, dev = 4, 3, torch.device("cpu")     # 4 envs, 3 robots each

# %% [markdown]
# ## Build an engine
#
# `make_engine(level, E, R, device, config, backend)` builds every fidelity level behind one API. `L2-legacy` is
# the slot-level uplink model: proportional-fair scheduling, HARQ retransmissions and link adaptation over 40
# uplink slots per control step. `seed` fixes the engine's own generator, which draws the random state of each
# reset (shadowing, fading). The default `NRConfig()` sends two traffic classes of 4,000 and 30,000 bytes.

# %%
cfg = NRConfig()
net = make_engine("L2-legacy", E, R, dev, cfg, backend="reference", seed=0)
print("level:", net.level, "| message sizes (bytes):", cfg.msg_sizes)
print("per-env clock:", net.clock.tolist())

# %% [markdown]
# ## Submit messages and step
#
# Each control step has two calls. `submit(t, Requests(send))` enqueues at most one new message per robot, where
# `send[e, r]` is 0 for nothing or the traffic class `c >= 1`. `step(t, poses)` then advances the network by one
# control step. Passing `t=None` uses each environment's own clock, which is the recommended form.
#
# `step` accepts robot positions `[E, R, 2]` or `[E, R, 3]` in metres, which go through the engine's radio model
# (path loss and shadowing from one gNB at the origin), or an SNR tensor `[E, R]` in dB if your simulator
# computes the radio itself.

# %%
pos = torch.rand(E, R, 2) * 60.0                      # robots within 60 m of the gNB
send = torch.ones(E, R, dtype=torch.long)             # every robot sends one small (class 1) message
accepted = net.submit(None, Requests(send))
out = net.step(None, pos)

print("accepted [E,R]:\n", accepted)
print("output keys:", sorted(out))

# %% [markdown]
# ## Read the outputs
#
# The per-message outputs have shape `[E, R, F]`, where `F` is the frame buffer depth (16 messages per robot).
# Slot `f` refers to the message that sat at position `f` of the robot's queue *before* the step:
#
# - `delivered`, `timed_out`: boolean masks of the messages that completed or hit their deadline this step.
# - `delay`: delivery time minus capture step, in control steps (`NaN` if not delivered).
# - `cap`, `cls`: capture step and traffic class of each slot (`-1` and `0` for an empty slot).
#
# The per-robot outputs have shape `[E, R]`: `newest` (the capture step of the newest message delivered this
# step, `-1` if none), `queue_len`, `queue_bytes` and `sinr_db`. `t` `[E]` is the clock value of this step.

# %%
dt_ms = cfg.control_step_ms
d = out["delay"][out["delivered"]] * dt_ms
print("delivered messages:", int(out["delivered"].sum()), "of", int(accepted.sum()))
print("delays of the delivered messages (ms):", [round(x, 1) for x in d.tolist()])
print("SNR per robot (dB):\n", out["sinr_db"].round(decimals=1))
print("messages still queued per robot:\n", out["queue_len"])

# %% [markdown]
# ## Age of information over an episode
#
# The age of information (AoI) of a robot is how old the freshest delivered information about it is. With the
# capture step of the newest delivered message, `last`, the AoI at the end of step `t` is `t + 1 - last` control
# steps. The loop below runs 50 control steps in which robots move slowly and send with probability 0.5, and it
# tracks the mean AoI and the delivery counts.

# %%
last = torch.zeros(E, R, dtype=torch.long)        # the state at reset counts as known (capture step 0)
n_sent = n_dlv = n_to = 0
for _ in range(50):
    send = (torch.rand(E, R) < 0.5).long() * torch.randint(1, 3, (E, R))   # class 1 or 2, or nothing
    n_sent += int(net.submit(None, Requests(send)).sum())
    out = net.step(None, pos)
    last = torch.maximum(last, out["newest"])
    aoi = out["t"][:, None] + 1 - last            # [E,R], in control steps
    n_dlv += int(out["delivered"].sum())
    n_to += int(out["timed_out"].sum())
    pos = (pos + 0.3 * torch.randn_like(pos)).clamp(0, 60)

print(f"accepted {n_sent}, delivered {n_dlv}, timed out {n_to}, still queued {int(out['queue_len'].sum())}")
print("AoI at the end (ms):\n", (aoi * dt_ms).round())
print("per-env clock:", net.clock.tolist())

# %% [markdown]
# ## Partial reset
#
# Robot-learning frameworks reset environments individually when their episodes end. `reset(env_ids)` takes an
# index tensor, a list or a boolean mask `[E]`, and re-initializes the queues, MAC state, fading, radio and
# clock of those environments only. The other environments are left bit for bit unaffected, and the random
# draws of the reset come from the engine's own generator, so the random stream of the other environments does
# not shift.

# %%
before = {k: getattr(net, k).clone() for k in ("cap", "rem")}
net.reset(torch.tensor([1, 3]))
print("per-env clock after resetting envs 1 and 3:", net.clock.tolist())
print("queued per env:", net.queued().sum(-1).tolist())
same = all(torch.equal(before[k][[0, 2]], getattr(net, k)[[0, 2]]) for k in before)
print("queues of envs 0 and 2 unchanged:", same)
last[[1, 3]] = 0                                  # reset your own per-env bookkeeping too

# %% [markdown]
# After a reset, an environment's clock restarts at 0 and its outputs use that clock, so capture steps and
# `newest` are comparable within one episode. Continue with `t=None` and every environment keeps its own clock.

# %%
out = net.step(None, pos)
print("clock value of this step per env:", out["t"].tolist())

# %% [markdown]
# ## Next steps
#
# - Tutorial 02 runs the same traffic through several fidelity levels.
# - Tutorial 03 configures the NR engine with `NRConfig`.
# - The "Engine" page of the API reference lists every argument and output.
