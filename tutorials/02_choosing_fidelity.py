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
# # Tutorial 02: Choosing fidelity
#
# Every fidelity level has the same API, so a task switches between them by changing the first argument of
# `make_engine`. This tutorial drives four levels with identical traffic and robot positions and reads the same
# network statistics from each, and then fits the surrogate levels from rollouts of a slot-level engine and
# loads them.
#
# The sizes here are tiny so that the tutorial runs on a CPU in about a minute. The numbers it prints show what
# each level reports and how to read it. They are not a calibrated comparison of the levels.

# %%
import tempfile

import torch

from isaac_net import LEVELS, NRConfig, Requests, make_engine

print("levels:", LEVELS)
E, R, T, dev = 8, 4, 60, torch.device("cpu")
cfg = NRConfig()

# %% [markdown]
# ## The same traffic for every level
#
# The traffic is drawn once: every robot sends with probability 0.4 per control step, a small message (4 kB)
# three times out of four and a large one (30 kB) otherwise. The robots stay at fixed positions 10 to 100 m from
# the gNB, so every level sees the same offered load and geometry.

# %%
g = torch.Generator().manual_seed(1)
sends = (torch.rand(T, E, R, generator=g) < 0.4).long()
sends = sends * torch.where(torch.rand(T, E, R, generator=g) < 0.75, 1, 2)
ang = torch.rand(E, R, generator=g) * 1.5
dist = 10 + 90 * torch.rand(E, R, generator=g)
pos = torch.stack([dist * torch.cos(ang), dist * torch.sin(ang)], -1)


def run(level, config=cfg, **kw):
    """Drive one level with the shared traffic; return counts, delay quantiles (ms) and timeouts."""
    torch.manual_seed(0)
    net = make_engine(level, E, R, dev, config, seed=0, **kw)
    acc = dlv = to = 0
    delays = []
    for t in range(T):
        acc += int(net.submit(None, Requests(sends[t])).sum())
        out = net.step(None, pos)
        dlv += int(out["delivered"].sum())
        to += int(out["timed_out"].sum())
        delays.append(out["delay"][out["delivered"]])
    d = torch.cat(delays) * cfg.control_step_ms
    q = d.quantile(torch.tensor([0.5, 0.95])).tolist() if d.numel() else [float("nan")] * 2
    return {"accepted": acc, "delivered": dlv, "timed_out": to, "p50_ms": q[0], "p95_ms": q[1]}


def show(name, s):
    print(f"{name:10s} accepted {s['accepted']:4d}  delivered {s['delivered']:4d}  timed out {s['timed_out']:3d}  "
          f"delay p50 {s['p50_ms']:7.1f} ms  p95 {s['p95_ms']:7.1f} ms")


# %% [markdown]
# ## Four levels
#
# - `L0` gives every message an independent lognormal delay and loss. It has no queue interaction, so the delay
#   of a message does not depend on how many other robots send.
# - `L1` is a fluid slot model. Robots with queued data share the uplink equally in every slot, so contention
#   raises delay, but there is no MAC detail.
# - `L2-legacy` is the prototype slot-level MAC and PHY: scheduling requests, proportional-fair scheduling,
#   HARQ and link adaptation.
# - `L2` is the configurable NR engine: 3GPP MCS and TBS tables, BLER curves, multiple HARQ processes and a
#   configurable frame structure. Its default `NRConfig` differs from the fixed constants of `L2-legacy`
#   (for example 16 HARQ processes instead of one), so the two slot-level engines are not expected to agree.

# %%
for level in ("L0", "L1", "L2-legacy", "L2"):
    show(level, run(level))

# %% [markdown]
# `L0` and `L1` take their parameters from the config (`l0_*` and `l1_eta`), so a task can move the `L0` delay
# distribution without touching code. `L0DR` redraws the `L0` parameters for each environment at every reset,
# which is the domain-randomization baseline.

# %%
cfg_slow = cfg.with_(l0_delay_median_steps=2.0, l0_loss=0.05)    # median 200 ms, 5% loss
show("L0 slow", run("L0", config=cfg_slow))
show("L0DR", run("L0DR"))

# %% [markdown]
# ## Bounds for task design
#
# `ORACLE` delivers every message at its capture step with zero delay, and `NOCOMM` never delivers anything.
# They are not network models: they bracket what any level can give a task. A task in which network fidelity can
# matter must show a large gap between its returns under these two bounds, so run them first when you design a
# task.

# %%
for level in ("ORACLE", "NOCOMM"):
    show(level, run(level))

# %% [markdown]
# ## Fit and use a surrogate level
#
# The surrogate levels `TR` (trace replay), `GE` (Markov-modulated delay and loss), `QA` (analytic queue) and
# `NN` (learned surrogate) are fitted from rollouts of `L2` or `L2-legacy` on the example fleet task. The
# command-line tool writes one parameter file outside the repository:
#
# ```bash
# python -m isaac_net.tools.fit_levels --source L2-legacy --task T1 --device cuda --backend graph
# ```
#
# The same fit is available as a function. The call below uses a toy size (4 envs, 2 training episodes of 30
# steps, 20 training steps for the NN) so that it finishes in seconds on a CPU. A real fit uses the tool's
# defaults (64 envs x 16 robots, 12 episodes of 300 steps) and takes minutes on a GPU.

# %%
from isaac_net.tools import fit_levels as fl  # noqa: E402

fit, info = fl.fit_levels("L2-legacy", E=4, R=4, episodes=2, test_episodes=1, T=30, sizes=cfg.msg_sizes,
                          device="cpu", nn_steps=20, qa_envs=4, qa_etas=(0.7, 1.0), log=lambda *a: None)
print("fitted levels:", sorted(k for k in fit if k != "meta"), "| meta:", fit["meta"])

# %% [markdown]
# `save_fit` writes the file and a JSON summary next to it, and it refuses any path inside the source tree:
# fitted parameters never go into the repository. The default location is
# `~/.cache/isaac_net/levels/<source>_<task>.pt`, or `$ISAAC_NET_LEVELS_DIR`.

# %%
tmp = tempfile.mkdtemp()
path = fl.save_fit(fit, f"{tmp}/L2-legacy_toy.pt", info)
print("saved", path)

# %% [markdown]
# `make_engine` loads the file with `params=`. One file holds all four surrogates, and the engine checks that
# the message sizes it was fitted with equal the config's `msg_sizes`. The fitted levels then run like any other
# level. With a toy fit their statistics mean little, so the loop below only shows that they run.

# %%
for level in ("TR", "GE", "QA", "NN"):
    s = run(level, params=path)
    print(f"{level:3s} runs: accepted {s['accepted']}, delivered {s['delivered']}, timed out {s['timed_out']}")

try:
    make_engine("NN", E, R, dev, cfg.with_(msg_sizes=(500.0, 1500.0)), params=path)
except ValueError as err:
    print("size check:", err)

# %% [markdown]
# ## Backends
#
# Every level has a readable eager `reference` backend. The prototype levels (`L0` to `L1`, `L2-legacy`) and the
# surrogates also have a `graph` backend, which records the same operations as a CUDA graph and is bitwise
# identical to the reference, and `L1` and `L2-legacy` have a fused `triton` kernel for scale. The NR engine
# (`L2`) has only the reference backend so far. The fast backends need a CUDA GPU, so this tutorial stays on
# `reference`; the Concepts page explains the equivalence guarantee.
