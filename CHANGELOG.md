# Changelog

All notable changes to `isaac-net` are listed here, grouped by area. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses [semantic versioning](https://semver.org/) from 0.1.0 on: while the major version is 0, a minor release may change the API, and every such change is listed under **Changed**. [RELEASE.md](RELEASE.md) describes how a release is cut.

## [0.1.0] - 2026-09-30

The first packaged release. It collects everything built since the initial prototype.

### Changed

- The project is renamed from `isaaclab-net` to `isaac-net`: the distribution is `isaac-net`, the package is `isaac_net` (`import isaaclab_net` no longer works), the console scripts are `isaac-net-bench`, `isaac-net-bake` and `isaac-net-measure`, the environment variables are `ISAAC_NET_*` (formerly `ISAACLAB_NET_*`), and the default cache directory is `~/.cache/isaac_net/` (move files from `~/.cache/isaaclab_net/` or point the variables at them). NVIDIA's Isaac Lab and its `isaaclab` packages keep their names.

### Engine and API

- One factory, `make_engine(level, E, R, device, config, backend)`, builds every fidelity level, and every engine honors the same contract: fixed-shape state with leading dims `[E, R]`, per-env clocks, partial `reset(env_ids)` that leaves other envs bitwise unaffected, and `submit` / `step` returning a dict of per-message and per-robot outputs. The earlier `add_frames` / `step(t, snr, hid)` calls still work.
- One `NRConfig` dataclass configures every module, with the presets `netslot_compat`, `lena_like`, `lena_match`, `lena_validation`, `srsran_like`, `oai_like` and `multicell(n)`. `NRConfig.unused_fields(level)` lists the fields a level ignores, and `make_engine(..., strict=True)` refuses them.
- Engine-owned counter-based random streams (`rng="engine"`, the default) for every level, keyed by seed, env, episode and step, so an env's draws do not depend on the batch size or on other envs' resets. `rng="global"` reproduces the earlier behavior.
- The frame buffer depth, application timeout and control step are configurable at every level (`frame_buffer`, `timeout_steps`, `control_step_ms`).

### Fidelity levels

- Prototype levels `L0`, `L0DR`, `L05`, `L05Q`, `L1` and the slot-level uplink, now `L2-legacy` (frozen, with the multi-cell `NetSlotMC`).
- The configurable NR engine as level `L2`: numerology 0 to 2, any FR1 bandwidth and TDD pattern, 3GPP MCS/TBS tables, EESM with Sionna or locally generated 5G-LENA BLER tables, multiple HARQ processes with chase or IR combining, RLC AM retry or UM loss, OLLA, downlink with delayed CQI, and the schedulers `pf`, `pf_wideband`, `maxci` and `rr`.
- Multi-cell NR: up to 7 cells, a PF scheduler and HARQ per cell, same-slot uplink and downlink interference, fractional uplink power control and A3 handover.
- Fitted surrogate levels `TR`, `GE`, `QA`, `NN`, the bounds `ORACLE` and `NOCOMM`, and the fit tool `python -m isaac_net.tools.fit_levels`.
- Level `WIFI`: a mean-field 802.11 DCF / EDCA uplink with 802.11ax / ac / a rates, A-MPDU, RTS/CTS, several APs with RSSI association and optional hidden nodes, plus an exact slot-level CSMA/CA event simulator for validation (`core/wifi`).
- Adaptive and mixed fidelity (`core/adaptive.py`): a cheap and an expensive level behind one engine, per env, with static mixes, load-triggered switching with queue handoff, and curricula.
- Differentiable fluid models `L1D` and `QAD` (`core/diff`), with gradients of delay, delivery, AoI and energy with respect to send probability, message size, transmit power and position, and a neural-proxy recipe. Exploratory; not a `make_engine` level.
- The 5G-LENA load-gap mechanisms as `NRConfig` switches: `pf_update` (`slot` or `rbg`), `pf_avg_idle` (`decay` or `freeze`), `ul_retx_sched` (`ofdma` or `tdma`), `ul_amc_alloc` (`current` or `previous`) and `ul_grant_model` (`lumped` or `bsr`, with `rlc_tail_timer_ms` and the buffer-report parameters). Defaults keep the engine bitwise unchanged; the presets `lena_match_v2` and `lena_validation_v2` turn every switch on with no fitted parameter and bring the median-delay error at moderate and saturated load to within about 1% of 5G-LENA. `graph` captures all switches exactly; `triton` covers all but `ul_grant_model="bsr"`. `core/nr_loadfix.py` is a compatibility shim over these fields.

### Backends

- `graph` (CUDA graph) backends, bitwise equal to the reference, for every prototype level, the surrogates and bounds, the NR engine (one or several cells), the Wi-Fi level and the edge stage.
- `triton` fused kernels for `L1`, `L2-legacy` and the single-cell NR engine (uplink and downlink), equal to the reference to rounding.
- `compile` (torch.compile plus CUDA graph) for the prototype levels.
- Uncontended benchmark campaign on an idle RTX 4090 and a dedicated L40 (`benchmarks/uncontended/`, results in `benchmarks/results/uncontended/`): network step time for every level and backend at four batch shapes, the fleet task, Isaac Lab scale up to 1,048,576 robots with `L2-legacy` and the NR engine on `triton`, and the ns-3 cost comparison recomputed against them; `benchmarks/isaac/bench.py --repeats` for repeated timing windows.
- `ShardedEngine` splits the envs over several GPUs behind one engine API. Two shards are bitwise equal to one engine for the prototype levels and `L2-legacy` on the tested backends. `L2` runs sharded but is not shard-invariant.

### Channels and scenes

- Selectable large-scale channel (`NRConfig.channel`): log-distance with correlated and white shadowing, TR 38.901 RMa / UMa / UMi / InH / InF path loss with a spatially consistent LOS state and O2I, and precomputed radio maps. Optional robot-body blockage and per-robot Doppler.
- Radio maps from USD scenes (`tools/scene`): USD export with ITU-R P.2040 materials, Sionna RT bake, a cache keyed by the scene hash, and the Isaac hook `IsaacNetCfg.scene_map`. A small synthetic map ships in `core/data/`.

### Traffic, edge, background load and energy

- Traffic models inside the `L2` step: periodic (sub-step periods), Markov on/off bursty, video I/P and event-triggered generators, with arrival offsets, tags, priorities and deadlines. The `graph` and `triton` backends run them.
- `EdgeLoop`: edge servers per env (FIFO or processor sharing, deterministic or exponential service, bounded queue, deadlines) and the return path to the robot (instant, delay from SINR, or a real NR downlink message), on top of any level.
- Background users per cell (`BackgroundConfig`): ghost UEs in the NR engine, an offered-load approximation on `L1` and `L2-legacy`.
- Radio energy and battery model (`EnergyConfig`): per-slot accounting on `L2` through a read-only slot tap, an airtime approximation elsewhere.

### Simulator integration

- Isaac Lab 3.0 layer on `make_engine`: `NetModule`, the `NetEnvMixin` for `DirectRLEnv` with four hook calls, `IsaacNetCfg` (pose source, network rate, blockage, domain randomization, observation selection), mdp terms and the fleet and warehouse demo envs.
- Install scripts for Windows (`scripts/windows/`) and a Linux / HPC recipe with kit-less Isaac Lab on Newton or OV PhysX in Apptainer (`scripts/hazel/`, `ISAAC_NET_PHYSICS`).
- MuJoCo Playground / MJX backend (`isaac_net.mjx.NetModuleMJX`) through `jax.experimental.buffer_callback` with zero-copy DLPack views, an MJX fleet env and Brax PPO.

### Validation and measurement tools

- ns-3 bridges (lockstep, process pool, offline replay) and their C++ programs for ns-3.48 + 5G-LENA v5.1, plus the 5G-LENA replay tool for the NR engine. Validation only.
- OAI 5G rfsim bridge (`bridges/oai`): Docker deployment, traffic agents, a virtual clock from T-tracer slot ticks, `OaiNet` as a drop-in network, campaign and analysis scripts in `benchmarks/oai/`.
- Measurement tools for a lab gNB or POWDER (`tools/measure`): srsRAN and OAI log parsers, a timestamped UDP probe, one-way-delay extraction, a unified schema and calibration hooks that write an `NRConfig` preset.
- Formal comparison scripts against 5G-LENA with CSV results (`benchmarks/fidelity/`), and the public-data calibration that produced `srsran_like` and `oai_like`.

### Benchmark suite

- `isaac_net.bench`: four network-aware multi-robot tasks (`fleet_alert`, `coop_map`, `coverage_nav`, `edge_control`) with `default`, `light` and `background` variants, shared metrics, random / heuristic / PPO (MLP and GRU) baselines, a versioned result format, a load-calibration command and a report command with 95% confidence intervals.

### Packaging

- Version 0.1.0, single-sourced from `isaac_net.__version__`. Full project metadata, BSD-3-Clause license expression, and the Sionna table license shipped as a license file.
- Extras: `dev`, `docs`, `mjx`, `isaac`, `ns3`, `oai`, `sionna`, `wifi` and `all`.
- Console scripts: `isaac-net-bench`, `isaac-net-bake` and `isaac-net-measure` (probe, ingest and calibrate subcommands).
- The wheel ships the Sionna BLER tables with their Apache-2.0 license, the synthetic radio map, the BSD-3 ns-3 bridge sources and the OAI compose file. GPL-derived 5G-LENA tables are excluded from both the wheel and the sdist.

### Documentation

- Docs site (mkdocs-material) with Concepts, five tutorials as executed notebooks, an API reference generated from the sources, the benchmark suite, and project notes on every subsystem, its validation and its open items.
- `docs/licensing.md` on shipped third-party data, locally generated GPL-derived tables, GPL-bound bridge binaries, Isaac Sim and the Omniverse EULA, and public datasets.

### Tests and CI

- pytest suite with CPU tests, GPU equivalence tests (`gpu`), and `slow`, `isaac` and `mjx` markers. CI runs ruff, the CPU tests and the strict docs build.
