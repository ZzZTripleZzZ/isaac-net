# Checkpoints

A long training run stops: a node is preempted, a job hits its time limit, or you want to branch an experiment from iteration 500. The policy has its checkpoint. The network has one too: every engine, wrapper and module of `isaac_net` can write its complete state and continue from it later, bit for bit.

```python
from isaac_net import NRConfig, make_engine
from isaac_net.core import checkpoint

net = make_engine("L2", E, R, "cuda", cfg, backend="graph", seed=0)
...                                                   # steps
checkpoint.save(net, "runs/a/isaac_net_500.pt", extra={"iter": 500})

net2 = make_engine("L2", E, R, "cuda", cfg, backend="graph")   # same level, E, R and config
extra = checkpoint.load(net2, "runs/a/isaac_net_500.pt")        # checks, then restores in place
```

In memory, without a file, the same pair is a method of every engine:

```python
sd = net.state_dict()          # flat {key: tensor | int | float | str | bool | None | list}, a snapshot
net2.load_state_dict(sd)       # strict=True by default
```

## What is saved

`state_dict()` walks the engine's objects and records everything that the next step reads:

- **Tensors**: the message queues and their per-message extras, MAC and HARQ processes, SR and BSR state, OLLA, PF averages, CSI and CQI, the fading state `h`, the Rician and frequency-correlation state, the radio's shadowing and LOS fields, cell association and handover timers, RLF timers, the RACH / DRX state machine, closed-loop power control, the counters behind `counters()`, the per-env clocks (`epoch`), and the counter RNG's episode counters, call counters and env offsets.
- **Generators**: every `torch.Generator` (`get_state()`): the traffic models' generators, the engine generator, the radio's.
- **Host values**: the NR engine's global control step `T`, the last fading slot, the interference-estimate counts, flags such as whether the message extras are on.
- **Wrappers and composites**: the edge server's jobs and commands in flight (`EdgeLoop`), batteries (`EnergyLoop`), background UEs (`BackgroundLoop`), the recorder's window and flush accumulators and the part numbers of its output files (`RecorderLoop`), every shard of a `ShardedEngine`, both levels and the routing state of an `AdaptiveEngine`, and for `NetModule` the Isaac radio with the per-env parameters that domain randomization drew.

Keys are the attribute paths: `net.ul.q.cap`, `net.ul.ctr[tb_ok]`, `traffic.streams[0].nxt`, `engine.state[battery]`, `shards[1].net.h`. A component with its own `state_dict` contributes under its path, so its load hooks run.

Not saved, on purpose (`checkpoint.SKIP` lists each name with its reason): the configuration (it is checked instead), PHY and other constant tables, captured CUDA graphs and their static scratch buffers (they are captured again or keep their memory), caches, the `log_stats` lists behind `collect()` (after a load, `collect()` covers the steps since the load), debug traces, and the recorder's open files and loggers. A test (`test_state_dict_covers_everything_a_step_changes`) snapshots every tensor, generator and host value reachable from the engine before and after steps, and fails if something that changed is neither saved nor on its own justified list.

## Restoring

`load_state_dict(sd, strict=True)` writes every value in place. Tensors are copied with `copy_` into the tensors the engine already has, so nothing is reallocated: the graph backends keep their static buffers and captured graphs keep reading the right memory, which also makes it safe to load into an engine that has already stepped. A captured graph also bakes in host values, above all the counter RNG's seed key: when a load changes one of them (a checkpoint from an engine with another seed, the usual case when resuming into a freshly built engine), the graph backends drop their captured graphs and capture again at the next step, which costs one capture per graph key and changes no result. A load from an engine with the same seed keeps the graphs. Parts that an engine builds lazily are built first: the engine's radio (made at the first step with poses) and the NR engine's per-message queue extras. State that a step creates on first use, for example the uplink power-control backoff, becomes a new attribute.

`strict=True` raises `KeyError` on a key the engine has no place for, on state of the engine that the dict lacks, and on a shape or dtype mismatch. `strict=False` restores what fits and skips the rest.

`checkpoint.load(engine, path, strict=True)` first compares the file's header with the engine: engine class, level, `E`, `R`, backend and the configuration field by field (`NRConfig.to_dict()` when the config has it, else its dataclass fields). The seed is the one field that may differ, because the checkpoint's RNG state wins. A mismatch raises `ValueError` naming the fields, or warns with `strict=False`. An external radio attached with `attach_radio` must be attached before the load.

## Guarantees

| Resume | Result |
|:---|:---|
| same backend, same device | bitwise equal to the uninterrupted run: every step output, `counters()` and the next `state_dict()` |
| NR engine `reference` ↔ `graph` | bitwise: the two share their state and are bitwise equal, so their checkpoints load into each other strictly |
| other backends of one level (e.g. `graph` ↔ `triton`) | refused by `strict=True`; with `strict=False` the state restores, and the continuation is as close as the two backends are (`triton` equals the reference to float rounding) |
| CPU ↔ CUDA | the state restores (a warning says so); the continuation equals the uninterrupted run to float rounding, because the counter RNG's normals and some kernels round differently on the two devices |
| `rng="global"` | the step noise comes from the global torch RNG, which is not engine state; `save` records it and `load(..., global_rng=True)` restores it, after which the resume is exact |
| `ShardedEngine` | per shard, under `shards[i].`; the split must be the same (the per-shard counters and generators cannot be redistributed) |

`tests/test_checkpoint.py` checks the first two rows on CPU for every level (L0 to L1, L2-legacy with one and three cells, L2 with one and three cells, traffic models, edge, energy and background wrappers, RACH and DRX, Rician fading with frequency correlation, QoS, mini-slots, MIMO, power control, FDD, the surrogates and bounds, WIFI), through a partial reset right after the resume, through a file, and into an engine that has already stepped. The CPU stand-in of the graph backend replays with the host values of its capture, seed key included, as a CUDA graph does, so the CPU tests also check that a load into an engine that has already captured drops graphs that would replay the old seed. The `gpu`-marked tests do the same for the CUDA `graph` and `triton` backends of the NR engine and the fast backends of `L2-legacy`, `L1` and `GE`.

## File format and size

`checkpoint.save` writes `torch.save({"isaac_net": version, "format", "level", "backend", "device", "E", "R", "kind", "config", "state", "global_rng", "extra"})` to `path + ".tmp"` and renames it, so a crash never leaves a half-written checkpoint. The state is on the CPU. Everything but your `extra` is plain data and tensors, so `torch.load(path, weights_only=True)` reads it. `checkpoint.read(path)` returns the dict, and `checkpoint.nbytes(sd)` the size of a state dict's tensors.

The size grows with the number of robots, `E × R`, and with the frame buffer:

| Level | Per robot | Per engine and env, at R = 16 |
|:---|---:|---:|
| `L2`, uplink | about 2.6 kB | about 11 kB |
| `L2`, uplink and downlink | about 4.5 kB | about 11 kB |
| `L2-legacy`, `L1`, `L0` | about 0.9 kB | about 5 kB |

Measured with the default config (frame buffer 16, 16 HARQ processes); a frame buffer of 64 roughly triples the queue part. A run with 4,096 envs × 16 robots on `L2` with a downlink writes about 300 MB per checkpoint.

## Isaac Lab and rsl_rl

`NetModule` has the same pair plus `save(path, extra=None, host=None)` and `load(path, strict=True, host=None)`. `host` is the object that runs the network in the env: the env itself for a Direct env with `NetEnvMixin`, or `env.isaac_net` for a manager-based env. With it, the file also holds the multi-rate state that lives on the host: the held outputs between network steps, the messages pending for the next network step, the decimation tick, and the runtime's send buffers. `isaac_net.isaac.net_module.find_network(env)` finds both through any gymnasium or rsl_rl wrapper, and `save_env_network(env, path)` / `load_env_network(env, path)` use it.

rsl_rl's `OnPolicyRunner.save` has no callback, so `isaac_net.isaac.tasks.rsl_rl_hook.install()` wraps the runner's `save` and `load` once per process. Every `model_<iter>.pt` then gets an `isaac_net_<iter>.pt` next to it, and `runner.load(path)`, which Isaac Lab's train script calls for `--resume`, restores the matching file when it exists. A file that does not match the env (another level, number of envs or config) is not loaded: a warning says why and training continues on the fresh network. Both `isaac_net.isaac.tasks.register` (the `--external_callback` of `isaaclab train`) and `python -m isaac_net.isaac.tasks.train` install the hook when rsl_rl is importable, and `ISAAC_NET_RSL_RL_HOOK=0` turns it off. Envs without an `isaac_net` network are not affected. With distributed training only rank 0 saves the policy, so only rank 0 saves and restores the network; the other ranks start fresh.

What the restored network means for training: Isaac Lab does not checkpoint the physics scene. On `--resume` it rebuilds the scene and starts every env from its reset state, while the network continues where it stopped, with its queues, clocks and channel state. What carries over is what matters for a long run: the domain-randomization parameters, the RNG episode counters (so the resumed run draws new channels instead of replaying the first episodes of the original run), the cumulative counters, and a recorder's part numbering. If you prefer every env to start a clean network episode after the resume, call `env.unwrapped.net_reset(None)` for a Direct env (or `env.unwrapped.isaac_net.reset(None)` for a manager-based one) after the load; the reset advances each env's episode, so the streams stay fresh.

For other runners or your own loop, call the functions yourself:

```python
from isaac_net.isaac.net_module import load_env_network, save_env_network

save_env_network(env, f"{log_dir}/isaac_net_{it}.pt", extra={"iter": it})
load_env_network(env, f"{log_dir}/isaac_net_{it}.pt")
```
