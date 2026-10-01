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
# # Tutorial 05: Validating against ns-3
#
# The ns-3 bridges run a task's traffic through ns-3 with the 5G-LENA NR module, a packet-level reference
# simulator, instead of the GPU engine. They are for validation only and are never needed for training: one
# ns-3 process simulates one cell of one environment on one CPU core.
#
# > **Not executable without ns-3.** This tutorial needs a local build of ns-3.48 with 5G-LENA v5.1 and the
# > bridge program compiled against it. Neither is part of this package (both are GPL-2.0). The rendered notebook
# > was not executed. The script checks for the build and exits with a message when it is missing.
#
# The bridge programs in `isaac_net/bridges/ns3/` are our own C++ code written against the ns-3 APIs. They
# run the reference scenario of the 5G-LENA validation (one gNB, 20 MHz with 50 PRBs, TDD `DDDSU` at 30 kHz,
# 3GPP UMi fading, HARQ with an EESM error model, RLC UM with a 2 s PDCP discard) and add a step interface.

# %% [markdown]
# ## 1. Build the bridge program
#
# The build compiles one C++ file against an existing ns-3 build with that build's exact flags, which takes 15 s
# to 1 min and does not reconfigure ns-3. On Linux (or WSL), with the lockstep bridge as the example:
#
# ```bash
# export NS3BRIDGE_ROOT=$HOME/bridge_lockstep     # holds ns-3.48/ (a built copy) and receives bin/
# export NS3_TOOLCHAIN_ENV=$HOME/ns3ref/env        # conda env whose lib/ the ns-3 build links against
# cp isaac_net/bridges/ns3/lockstep/*.sh isaac_net/bridges/ns3/lockstep/netslot-bridge.cc $NS3BRIDGE_ROOT/
# cd $NS3BRIDGE_ROOT && bash build_bridge.sh       # -> bin/netslot-bridge
# ```
#
# `isaac_net/bridges/ns3/lockstep/README.md` documents the directory layout the build scripts expect, the
# optional ns3-ai shared-memory transport, and the wire protocol. The process-pool bridge in
# `isaac_net/bridges/ns3/pool/` builds the same way and reads `BRIDGE_ROOT`.

# %%
import os
import sys

import torch

root = os.environ.get("NS3BRIDGE_ROOT", "")
binary = os.path.join(root, "bin", "netslot-bridge")
HAVE_NS3 = bool(root) and os.path.exists(binary)
if not HAVE_NS3:
    print("ns-3 bridge not found (set NS3BRIDGE_ROOT to a directory with bin/netslot-bridge); "
          "the cells below are for reading only.")
    if __name__ == "__main__" and "ipykernel" not in sys.modules:
        sys.exit(0)

# %% [markdown]
# ## 2. Lockstep co-simulation with Ns3Net
#
# `Ns3Net` implements the engine API on top of ns-3. Every control step it sends the robot positions, a
# per-robot shadowing value and the new messages to ns-3, runs ns-3 for 100 ms of simulated time, and reads back
# which messages completed and when. The queue bookkeeping, timeouts and outputs are the engine's own code, so
# the outputs have the same keys and meaning as those of `make_engine`.
#
# Two details differ from the GPU engine:
#
# - ns-3 cannot rewind part of a simulation, so `Ns3Net` supports **full resets only**: `reset()` rebuilds every
#   environment, and `reset(env_ids)` raises.
# - The positions come from the environment object that `bind_env` attaches, which must have `pos [E, R, 2]`.
#   With `shadow="env"` (the default), the shadowing sent to ns-3 is derived from the SNR you pass to `step`, so
#   ns-3's large-scale SNR equals the engine radio's.
#
# The loop below parks robots 15 to 40 m from the gNB, sends the same traffic through ns-3 and through the
# `L2-legacy` engine, and compares delivery counts and the median delay.

# %%
from isaac_net import Requests, make_engine  # noqa: E402
from isaac_net.bridges.ns3_lockstep.lockstep_net import Ns3Net  # noqa: E402
from isaac_net.core.proto.netsim import Radio  # noqa: E402


class ParkedRobots:
    """The minimum Ns3Net reads from an environment: positions pos [E,R,2] and a radio."""

    def __init__(self, E, R):
        ang = torch.rand(E, R) * 1.5 + 0.03
        d = 15 + 25 * torch.rand(E, R)
        self.pos = torch.stack([d * torch.cos(ang), d * torch.sin(ang)], -1)
        self.radio = Radio(E, "cpu")


torch.manual_seed(0)
E, R, T, sizes = 2, 8, 60, (4000.0, 30000.0)
env = ParkedRobots(E, R)
snr = env.radio.snr_db(env.pos)
sends = (torch.rand(T, E, R) < 0.5).long()

results = {}
for name in ("ns-3", "L2-legacy"):
    if name == "ns-3":
        net = Ns3Net(E, R, "cpu", sizes, mode="procs", transport="tcp").bind_env(env)
    else:
        net = make_engine("L2-legacy", E, R, "cpu", seed=0)
    net.reset()
    dlv, delays = 0, []
    for t in range(T):
        net.submit(None, Requests(sends[t]), snr_db=snr)
        out = net.step(None, snr)
        dlv += int(out["delivered"].sum())
        delays.append(out["delay"][out["delivered"]])
    d = torch.cat(delays) * 100.0
    results[name] = (dlv, float(d.median()) if d.numel() else float("nan"))
    if name == "ns-3":
        net.close()                       # stops the ns-3 processes
for name, (dlv, p50) in results.items():
    print(f"{name:10s} delivered {dlv:4d}, median delay {p50:.1f} ms")

# %% [markdown]
# The two numbers are not expected to match exactly: 5G-LENA and the slot-level engine differ in several
# modelling choices, which `docs/validation-5g-lena.md` lists with the side that was changed. Per-frame
# comparisons between ns-3 and the GPU engine are only meaningful statistically.
#
# `mode` chooses how environments map to ns-3 processes: `"procs"` runs one process per environment in parallel,
# `"single"` runs all environments as independent cells in one process, and `"groups"` with
# `envs_per_proc=k` is in between. `transport` is `"tcp"`, `"unix"` (a Unix socket) or `"shm"` (ns3-ai shared
# memory, which needs the `netslot-bridge-ai` build). Every transport gives identical results; they differ only
# in overhead.
#
# ## 3. The Isaac NetModule API over ns-3
#
# `Ns3NetModule` offers the `NetModule` call pattern (`reset(env_ids)`, `step(poses, TrafficRequest)`), so an
# Isaac task can run one evaluation against ns-3. In mode `"procs"` a partial reset rebuilds only that
# environment's ns-3 process. From Windows, start the servers in WSL with `serve_wsl.sh` and connect with
# `spawn=False`:
#
# ```python
# from isaac_net.bridges.ns3_lockstep.netmodule_ns3 import Ns3NetModule
# from isaac_net.isaac import NetConfig, TrafficRequest
#
# net = Ns3NetModule(NetConfig(num_envs=E, num_robots=R, device="cpu", msg_sizes=(4000.0, 30000.0)),
#                    transport="tcp", spawn=False, endpoints=[f"tcp:{57100 + e}" for e in range(E)])
# out = net.step(poses_end, TrafficRequest(send))          # out.ns3 holds the raw per-UE ns-3 statistics
# ```
#
# ## 4. Offline replay of the 5G-LENA sweep
#
# The NR engine's validation replays a 5G-LENA sweep with identical per-robot link budgets, without running
# ns-3 again. Each run directory of the sweep holds `ues.csv`, `meta.txt`, `summary.json` and `delay_cdf.csv`:
#
# ```bash
# python -m isaac_net.bridges.ns3_offline.lena_replay <sweep_dir> replay.csv 4
# python -m isaac_net.bridges.ns3_offline.lena_replay <sweep_dir> replay_sr20.csv 4 sr_grant_delay_slots=20
# ```
#
# The third argument is the number of replicas per run, and `key=value` pairs override fields of the
# `lena_validation()` preset. The replay needs the 5G-LENA BLER tables, generated locally with
# `python -m isaac_net.tools.extract_lena_tables <your nr checkout>`.
#
# ## Caveats
#
# - **Bit-exact comparisons between ns-3 runs need an identical command line.** 5G-LENA's results depend on the
#   process heap layout, so even the length of a path argument can change a sample path.
# - **Offline trace replay (`ns3_offline.ReplayNet`) is valid only for the policy that recorded the trace.**
#   Another policy sends differently, which changes the queues and the delays that the replay cannot see.
# - Frames that reach the 2 s application timeout stay in 5G-LENA's RLC queue and keep using air time.
#
# `docs/bridges.md` describes the correctness checks, costs and known limits of every bridge.
