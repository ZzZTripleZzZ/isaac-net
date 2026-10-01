# ns-3 bridges

The bridges run a task's network through ns-3.48 with 5G-LENA v5.1 instead of the GPU engine. They are for **validation only** and are never needed for training. They need a local ns-3 + 5G-LENA build, which is not part of this package, and binaries built against ns-3 are GPL-bound (see [Licensing](../licensing.md)). [Tutorial 05](../tutorials/05_validating_against_ns3.ipynb) walks through a run, and [ns-3 bridges in depth](../bridges.md) has the correctness checks, costs and known limits.

| Bridge | Python entry point | C++ program | Use |
|:---|:---|:---|:---|
| lockstep (TCP, Unix socket or ns3-ai shared memory) | `bridges.ns3_lockstep.lockstep_net.Ns3Net`, `bridges.ns3_lockstep.netmodule_ns3.Ns3NetModule` | `bridges/ns3/lockstep/netslot-bridge.cc` | closed-loop co-simulation of a few environments |
| process pool | `bridges.ns3_pool.poolnet.PoolNet` | `bridges/ns3/pool/netslot-bridge.cc` | one ns-3 process per environment, the CPU co-simulation baseline |
| offline replay | `bridges.ns3_offline` (`rollout`, `offline_ns3`, `ReplayNet`) | the pool program in file mode | ns-3 after the fact on a recorded rollout |
| 5G-LENA sweep replay | `python -m isaac_net.bridges.ns3_offline.lena_replay` | none | replays a 5G-LENA sweep in the NR engine with identical link budgets |

## Environment variables

| Variable | Used by | Meaning |
|:---|:---|:---|
| `NS3BRIDGE_ROOT` | lockstep | directory with `bin/netslot-bridge` (and `bin/netslot-bridge-ai` for shared memory) and the ns-3.48 build |
| `NS3_TOOLCHAIN_ENV` | lockstep | the conda environment whose `lib/` the ns-3 build links against |
| `BRIDGE_ROOT` | pool, offline | directory with the pool's `bin/netslot-bridge` |

The defaults are paths on the lab machine. Build instructions and the wire protocol are in `isaac_net/bridges/ns3/lockstep/README.md` and `isaac_net/bridges/ns3/pool/README.md`.

## Usage

`Ns3Net` and `PoolNet` implement the engine contract on top of ns-3, so they replace `make_engine(...)` in a loop that drives the engine. Their positions come from an environment object with a `pos [E, R, 2]` attribute, attached with `bind_env(env)` or `attach_env(env)`:

```python
from isaac_net import Requests
from isaac_net.bridges.ns3_lockstep.lockstep_net import Ns3Net

net = Ns3Net(E, R, "cpu", (4000.0, 30000.0), mode="procs", transport="tcp").bind_env(env)
net.reset()                                   # full resets only
for t in range(T):
    net.submit(None, Requests(send[t]), snr_db=snr)
    out = net.step(None, snr)                 # the same dict as make_engine's engines
net.close()                                   # stops the ns-3 processes
```

The Isaac-style `Ns3NetModule` takes a `NetConfig` and `step(poses_end, TrafficRequest)`. In mode `"procs"` it rebuilds only the reset environment's process, so partial resets work there.

## Limits to keep in mind

- ns-3 cannot rewind part of a simulation. `Ns3Net` and `PoolNet` support full resets only, and a partial reset in single-process lockstep mode is logical only.
- 5G-LENA's results depend on the process heap layout, so bit-exact comparisons between ns-3 runs need an identical command line. Comparisons between ns-3 and the GPU engine are statistical.
- Offline replay is valid only for the policy that recorded the trace, because another policy's own traffic changes the queues.

## Classes

::: isaac_net.bridges.ns3_lockstep.lockstep_net.Ns3Net
    options:
      heading_level: 3
      members: [bind_env, close]

::: isaac_net.bridges.ns3_lockstep.netmodule_ns3.Ns3NetModule
    options:
      heading_level: 3
      members: [reset, step, close]

::: isaac_net.bridges.ns3_lockstep.core.Ns3Lockstep
    options:
      heading_level: 3
      members: [reset, step, close]

::: isaac_net.bridges.ns3_pool.poolnet.PoolNet
    options:
      heading_level: 3
      members: [attach_env, close]

::: isaac_net.bridges.ns3_offline.replaynet.ReplayNet
    options:
      heading_level: 3
      members: false
