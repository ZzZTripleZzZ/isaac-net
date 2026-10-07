# Changelog

All notable changes to `isaac-net` are listed here, grouped by area. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses [semantic versioning](https://semver.org/) from 0.1.0 on: while the major version is 0, a minor release may change the API, and every such change is listed under **Changed**. [RELEASE.md](RELEASE.md) describes how a release is cut.

## [Unreleased]

A usability and visualization wave: scenario presets, a config summary, an install doctor, slot traces, a KPI recorder with a plotting package, registered Isaac Lab tasks with viewport overlays, and new onboarding pages, then checkpoints of a running network, config files with generated preset YAML, network KPIs in the Isaac Lab log and an online docs site. Nothing in the engines' step changes: the recorder and the trace are bitwise invisible, and `make_engine` keeps `backend="reference"` as its default.

### Added

#### Configuration and presets

- Scenario presets in `isaac_net/core/scenarios.py`: `warehouse_private_5g()`, `factory_inf(n_cells=3)`, `outdoor_campus(n_cells=3)` and `urllc_control(base=None)`, exported from `isaac_net` and listed in `SCENARIOS` ([docs/configurability.md](docs/configurability.md#scenario-presets)). They are representative, not calibrated: each starts from the `lena_validation_v2()` MAC with `fading=True` and the shipped Sionna PDSCH tables (`SHIPPED_PHY`: `bler_source="pdsch"`, `tbs_mode="38214"`, `harq_combining="cc"`), and every keyword argument overrides the field it names.
- The two multi-cell presets, `factory_inf` and `outdoor_campus`, turn on `olla=True`, which `lena_validation_v2()` runs off: with several cells the scheduler's MCS uses the N+I of the previous slot, and without OLLA many more transport blocks exhaust HARQ (the reason is in their docstrings). `urllc_control` adds a 10 ms control step, 2-symbol mini-slots in both directions, `scheduler="qos"` with two classes and `ul_grant_model="lumped"` with `proactive_grant="every_ul_slot"`, on top of `base` or the preset MAC. No preset runs on `triton` (SR / BSR pipeline, several cells or mini-slots).
- `NRConfig.describe(level="L2", backend=None)` returns a plain-text page of what the level will run, which backends can run the config, every field set away from its default, and the fields the level ignores. `NRConfig.diff(other)` lists the fields two configs disagree on ([docs/configurability.md](docs/configurability.md#describing-a-configuration)).
- `make_engine(..., backend="auto")` picks `triton`, `graph` or `reference` from the level, the config and the device and logs the choice and its reason on the `isaac_net` logger; `isaac_net.core.resolve_backend(level, cfg, device)` returns the same choice without building anything. The default stays `backend="reference"`, and `ShardedEngine(..., backend="auto")` resolves it per shard ([docs/configurability.md](docs/configurability.md#choosing-a-backend-automatically)).
- `output_schema()` on every engine and wrapper lists the keys its `step` returns under its current config, with shape, dtype, unit and meaning, from one registry, `isaac_net/core/schema.py` (`STEP_KEYS`, exported from `isaac_net.core`). The table in [docs/configurability.md](docs/configurability.md#output-schema) is generated from it.

#### Diagnostics

- `isaac-net-doctor` (`isaac_net/tools/doctor.py`, self-test in `selftest.py`, also `python -m isaac_net.tools.doctor`): an environment report (Python, torch, CUDA, triton and the optional stacks, found with importlib metadata rather than imported, except triton), a CPU self-test plus a CUDA one when a GPU is present (`--device auto`), `--quick` for a shorter self-test, `--no-selftest`, `--config SPEC` to check a preset, an expression or `file.py:CFG` against the backends (the self-test then runs only with `--selftest`), and `--json` for one JSON document. Exit code 0 when no check fails, 1 otherwise, 2 for a bad command line or config ([docs/doctor.md](docs/doctor.md)).

#### Traces and recording

- `SlotTrace` (`isaac_net/core/trace.py`): `SlotTrace.attach(net, env=0, robots=[...])` or `pairs=` records slot-level MAC events of selected (env, robot) pairs on `L2` with the reference backend (`graph` and `triton` are refused): frame arrivals, SRs, TB grants with HARQ process, MCS, TBS and RBGs, decode results, DL CQI reports, deliveries and resets. It reads only, draws no random number and gathers on the device with one host copy every `flush_steps=32` steps, so the engine's outputs are bitwise unchanged. `to_frame("events" | "samples")` gives a pandas DataFrame (a list of dicts without pandas), `save()` writes Parquet with pyarrow or CSV otherwise, and `summary()` gives per-robot delay mean / p95 and retransmissions per TB ([docs/trace.md](docs/trace.md)).
- Two read-only observer points used by the trace: `MacLink.slot_hook` (`None` by default, called after decoding in every slot or mini-slot occasion) and `SlotTap.add_observer(fn)` (called with the SINR the MAC decodes with, after the hook chain).
- `RecorderLoop` and `isaac_net.record(engine, out_dir="records", ...)` (`isaac_net/core/record.py`, settings in `RecordConfig`): a per-step KPI recorder that wraps any engine or wrapper with the same API and step dict. It accumulates on the device without a per-step host sync, closes a row every `every=1` steps, keeps `flush_every=256` windows before one host copy, and writes `steps`, `cells` (`per_cell=True`), `robots` (for `raw_envs=()`) and `delay_hist` / `aoi_hist` (`hist=True`) tables as Parquet, or CSV without pyarrow (`format="auto"`), plus `meta.json`. It draws no random number and writes no engine tensor, so the outputs are bitwise those of the bare engine.
- The recorder logs to TensorBoard with `tensorboard="logdir"` (default `None`) and to Weights & Biases with `wandb=True` or a dict of `wandb.init` arguments (default `False`), every `log_every=1` steps at flush time.

#### Plots and reports

- The `isaac_net.viz` package and the `viz` extra (`matplotlib`, `pandas`, `pyarrow`): `style` (paper style), `cdf` (`plot_delay_cdf`, `plot_aoi_cdf` from the recorder histograms), `loads` (`plot_throughput_vs_load`, `plot_cell_utilization`), `maps` (`plot_rem`, `plot_arena`), `compare` (`plot_vs_ns3`, the engine against ns-3 5G-LENA), `report` (`from_records(dirs, "report.html")`, a self-contained HTML page) and `trace` (`plot_timeline`, `plot_slot_heatmap` of a `SlotTrace`). [docs/viz.md](docs/viz.md) has a gallery in `docs/img/viz/`, regenerated by `docs/img/viz/make_gallery.py`.
- `isaac-net-bench report --html FILE` writes an HTML report with figures (needs the `viz` extra), and `--figures DIR` writes them as PNGs. `isaac-net-bench run --keep_rows` (off by default) keeps the per env-episode evaluation rows in the result file, so the report's CDFs run over episodes instead of seeds ([docs/viz.md](docs/viz.md#reports)).
- `isaac-net-rem --panels` chooses the panels of the `--png` figure among `rsrp`, `sinr`, `serving`, `los` and `pathgain` (default `rsrp,sinr,serving,los`), drawn by `viz.maps.plot_rem` ([docs/rem.md](docs/rem.md)).

#### Isaac Lab

- Registered gymnasium tasks `Isaac-NetFleet-Direct-v0`, `Isaac-NetFleet-Direct-L0-v0`, `Isaac-NetFleet-Direct-Warehouse-v0` and `Isaac-NetFleet-Manager-v0`, each with rsl_rl and skrl PPO configs (`default_agent` `"rsl_rl"`). They train with `python -m isaac_net.isaac.tasks.train --task ...` or with Isaac Lab's own script and `--external_callback isaac_net.isaac.tasks.register` ([docs/isaac-lab.md](docs/isaac-lab.md#train-the-registered-task)). Isaac Lab 3.0 runs headless by default and has no `--headless` flag; `--viz kit` opens the viewport.
- A manager-based workflow: `NetManagerCfg(...).apply(env_cfg)` adds the network to a `ManagerBasedRLEnv` config through `NetRuntime`, with the observation terms `net_aoi`, `net_delivered`, `net_delay`, `net_queue`, `net_sinr`, `net_los` and `net_access_state`, the events `net_step` and `net_reset`, the termination `net_step_done`, the rewards `net_aoi_penalty` and `net_send_cost`, and the action term `NetSendActionCfg` ([docs/isaac-lab.md](docs/isaac-lab.md#manager-based-workflow)).
- Viewport overlays `NetMarkers` / `NetMarkersCfg`: links to the serving gNB coloured by SINR (blocked links dimmer and dashed), gNB masts with coverage discs, AoI bars and glyphs for idle, RACH, DRX-dormant and RLF robots. They are inert when headless: nothing is computed and nothing from Isaac Lab is imported ([docs/isaac-lab.md](docs/isaac-lab.md#viewport-overlays)).
- `NetModule.step` passes through `access_state`, `rlf`, `access_sleep_frac` and `rach_attempts` when the engine returns them.

#### Docs and onboarding

- A Colab quick start, `docs/tutorials/00_quickstart_colab.ipynb`, with an "Open in Colab" badge; new pages [choosing.md](docs/choosing.md), [cookbook.md](docs/cookbook.md) (ten recipes that run on a CPU), [faq.md](docs/faq.md), [doctor.md](docs/doctor.md), [trace.md](docs/trace.md) and [viz.md](docs/viz.md); a shorter landing page `docs/index.md`.
- `docker/Dockerfile` and `docker/README.md` for the kit-less Linux path: CUDA 12.6 runtime, torch from the cu126 index, and an optional `isaaclab` stage pinned to the validated Isaac Lab commit. It passes hadolint and has not been built.
- A CI job `docs-examples` executes the Colab notebook and every Python block of the cookbook on a CPU.

#### Docs site

- The documentation is online at [docs.isaacnet.zifanzhang.com](https://docs.isaacnet.zifanzhang.com/) (Cloudflare Pages project `isaacnet-docs`): the latest `main` at the root and each release under `/<version>/` (now [`/0.2.0/`](https://docs.isaacnet.zifanzhang.com/0.2.0/)), with a `versions.json` for the mkdocs-material version selector and a `/latest/` redirect.
- `scripts/build_docs.sh` builds `main` and every version in `DOCS_VERSIONS` (default `"0.2.0"`) with `mkdocs build --strict`, and `scripts/deploy_docs.sh` deploys the result with wrangler. The `Docs` workflow (`.github/workflows/docs.yml`) builds on every push to `main` and on `v*` tags, and deploys only when the repository secrets `CLOUDFLARE_API_TOKEN` and `CLOUDFLARE_ACCOUNT_ID` exist. Step 3 of [RELEASE.md](RELEASE.md) now says to add the new version to `DOCS_VERSIONS`.

#### Checkpoints

- `state_dict()` and `load_state_dict(sd, strict=True)` on every engine, wrapper, `ShardedEngine`, `AdaptiveEngine` and `NetModule`, from one `StateDictMixin` in `isaac_net/core/checkpoint.py` ([docs/checkpoint.md](docs/checkpoint.md)). The state dict is flat and keyed by attribute path; it holds every state tensor (cloned), the state of every `torch.Generator` and the host values such as the NR engine's slot clock. `checkpoint.SKIP` lists, with a reason each, what is left out: configurations, PHY tables, captured CUDA graphs and their static buffers, caches and the `log_stats` statistics. `strict=True` raises `KeyError` on a key that cannot be placed, on missing state and on a shape or dtype mismatch; `strict=False` restores what fits.
- `checkpoint.save(engine, path, extra=None)` and `checkpoint.load(engine, path, strict=True, global_rng=False)`. The file carries a header (package version, format, engine kind, level, backend, device, `E`, `R`, config) that `load` checks field by field before restoring (the seed may differ); a mismatch raises `ValueError` naming the fields, or warns with `strict=False`, and a CPU / CUDA difference only warns. The file is written to `path + ".tmp"` and renamed, holds the state on the CPU and, apart from `extra`, loads with `torch.load(..., weights_only=True)`. `save` also records the global torch RNG, which `load(..., global_rng=True)` restores for `rng="global"`.
- A resume on the same backend and device is bitwise equal to the uninterrupted run: step outputs, `counters()` and the next `state_dict()`. NR engine checkpoints of `reference` and `graph` load into each other strictly; other backend pairs need `strict=False`. Graph backends restore in place with `copy_`, so captured CUDA graphs keep their buffers. `ShardedEngine` stores each shard under `shards[i].` and needs the same split.
- Sizes at the default config: about 2.6 kB per robot on `L2` uplink, 4.5 kB with the downlink, 0.9 kB on `L2-legacy`, `L1` and `L0`.
- Isaac Lab: `NetModule.save(path, extra=None, host=None)` and `load(path, strict=True, host=None, global_rng=False)`, where `host` (the Direct env, or `env.isaac_net` for a manager-based env) adds the multi-rate state that lives on the host; `find_network(env)`, `save_env_network(env, path)` and `load_env_network(env, path)` in `isaac_net.isaac.net_module`.
- An rsl_rl hook, `isaac_net/isaac/tasks/rsl_rl_hook.py`, installed by `isaac_net.isaac.tasks.register()` and `python -m isaac_net.isaac.tasks.train` when rsl_rl is importable (`ISAAC_NET_RSL_RL_HOOK=0` turns it off): every `model_<iter>.pt` gets an `isaac_net_<iter>.pt` next to it, restored on `--resume`. Only rank 0 saves and restores under distributed training. Isaac Lab does not checkpoint the physics, so on a resume the envs restart from their reset state while the network continues ([docs/checkpoint.md](docs/checkpoint.md#isaac-lab-and-rsl_rl)).

#### Config files and presets

- `NRConfig.to_dict(only_changed=False)`, `from_dict(d, base=None)`, `to_json` / `from_json`, `to_yaml` / `from_yaml` and `from_preset(name, **overrides)` ([docs/config-files.md](docs/config-files.md)). Every preset and scenario round-trips exactly: tuples become lists, `None` stays `None`, `inf` / `-inf` / `nan` become strings, a nested config carries a `__type__` key, and a `TrafficModel` is written as its kind and the fields it changed. A dict with a `preset` key builds that preset with the other keys as overrides. YAML needs PyYAML, which is optional: without it `to_yaml` writes JSON text and `from_yaml` reads JSON content only.
- Reading is strict: a key that is not a field raises `UnknownFieldError` (a `ValueError`, in `isaac_net.core.config`) with the closest field names, and so does an unknown preset name.
- Twelve generated preset files in `isaac_net/core/presets/` (the eight validation presets and the four scenario presets, fields away from `NRConfig()` only), shipped as package data; `PRESETS` and `preset_path(name)` in `isaac_net.core.presets`. `scripts/export_presets.py` writes them and `--check` exits 1 when one differs from its Python preset, which `tests/test_config_io.py` also checks.

#### Isaac Lab KPIs and overrides

- `IsaacNetCfg.nr` (default `{}`): a dict of `NRConfig` field overrides, or `{"preset": name, ...}`, applied on top of the env's `NRConfig` by `net_setup`, `finalize_fleet_cfg` and `NetModule`; an `NRConfig` there replaces the env's config. From the command line of a registered task: `env.net_isaac.nr.ul_tpc=true` or `env.net_isaac.nr.preset=warehouse_private_5g`, and `env.isaac_net.isaac.nr.ul_tpc=true` on the manager-based task. A misspelled key raises `UnknownFieldError` before the simulation starts ([docs/config-files.md](docs/config-files.md#hydra-overrides-in-the-registered-tasks)).
- Network KPIs in `extras["log"]`, which rsl_rl writes to TensorBoard: `net/aoi_mean_s`, `net/aoi_p95_s`, `net/delay_mean_ms`, `net/delivered_frac`, `net/dropped_frac`, `net/queue_bytes_mean` and `net/sinr_mean_db` on every level, `net/prb_util` and `net/harq_bler` on `L2`, and `net/rlf_frac` / `net/access_sleep_frac` when the step reports them. They are computed on the device without a host sync by `isaac_net/isaac/kpis.py`, in `NetEnvMixin.net_step` and `NetRuntime.step`, with `IsaacNetCfg.log_kpis=True` and `log_every=1` by default ([docs/isaac-lab.md](docs/isaac-lab.md#network-kpis-in-tensorboard)).
- `isaac_net.core.record.mac_links(engine)` and `mac_counters(links)` expose the NR MAC counter snapshot that the recorder used internally; the KPIs reuse it.

### Changed

- `make_engine(..., strict=False)`, the default, now emits one `UnusedFieldsWarning` (a `UserWarning`) naming the fields a level ignores, once per level and set of ignored values per process; before, they were ignored silently. `strict=None` skips the check and `strict=True` still raises ([docs/configurability.md](docs/configurability.md#ignored-fields-warning-and-strict-mode)).
- `AdaptiveEngine` builds its cheap and expensive levels with `strict=None`, so the one config it passes to both raises no warning.
- `isaac-net-rem` draws its PNG through `isaac_net.viz.maps.plot_rem` and takes `--panels`, and `isaac-net-bench` gains `report --html`, `report --figures` and `run --keep_rows` (see Added).
- Tutorial 03 no longer says that `L2-legacy` refuses `timeout_steps=10`: the prototype levels accept the frame buffer, timeout and control step, and ignored MAC fields give an `UnusedFieldsWarning`, or an error with `strict=True`.
- `CITATION.cff` is at version 0.2.0, released 2026-10-05.
- `NetModule.__init__` applies `IsaacNetCfg.nr` itself (`IsaacNetCfg.resolve_nr`), so a `NetModule` built without `net_setup` gets the same overrides or preset; with an empty `nr` the config is unchanged.
- Isaac Lab envs now put the `net/` KPIs into `extras["log"]` on every network step by default (`IsaacNetCfg.log_kpis=True`); `log_kpis=False` restores the previous log.
- `RecorderLoop` reads the NR MAC counters through `mac_links` / `mac_counters`; its output is unchanged.
- The README's Documentation links and `Documentation` in `pyproject.toml` point at [docs.isaacnet.zifanzhang.com](https://docs.isaacnet.zifanzhang.com/) instead of the `docs/` folder on GitHub.

## [0.2.0] - 2026-10-05

Thirty bug fixes from a full code review, three waves of optional features (all off by default and bitwise unchanged when off), the triton kernel extended to most of them, and a re-run of the 5G-LENA validation that confirmed every published number. The minimum torch version is 2.7.

### Added

Every new field below defaults to off (or to the earlier behaviour), and at the defaults the engine's outputs are bitwise unchanged.

#### Obstacles and NLOS

- Geometric LOS state ([docs/obstacles.md](docs/obstacles.md)): `NRConfig.los_source` (default `"stochastic"`, the TR 38.901 probability) can be `"map"` (a baked `los_prob` grid thresholded against the spatially consistent field), `"raycast"` (a 2.5-D fixed-step ray march over an `obstacle_z` height map, `los_raycast_samples=32`) or `"callback"` (an Isaac `blocked_fn`, also with multi-cell `radio="engine"`). `RadioMC.los_state()` and `blocked_state()` return the state of every link.
- `los_diffraction=False` adds an ITU-R P.526 single knife-edge loss from the ray-march clearance (obstacle tops and vertical rack edges), `los_soft=False` the TR 38.901 §7.6.3.3 soft LOS blend, and `nlos_extra_loss_db=0.0` a fixed extra loss on NLOS links of `log_distance`. With `channel="radio_map"` the LOS state adds no path loss, since the map already holds the NLOS loss, and maps baked with `--diffraction` are refused together with `los_diffraction`.
- TR 38.901 blockage models through `blockage_model` (default `"sphere"`, the earlier robot spheres): `"screen"` is model B (§7.6.4.2), knife-edge screens for the robots and for per-step blockers passed as `step(..., blockers=)` rows (x, y, class), and `"stochastic"` is model A (§7.6.4.1), K = 4 spatially and temporally consistent regions without self-blocking. `blocker_size_m` sets the (w, h) per class (robot, human, vehicle) and `blockage_max_db=40.0` caps the summed loss.
- With the obstacle stack on, the step dict gains `los` and `blocked`, the Isaac layer gains a `los` observation feature, and `NetModule.step` takes `blockers=`.
- `RadioMap.los_prob` and `obstacle_z` grids, `bake.py --obstacle-z`, `make_synthetic_radio_map --obstacles` (a synthetic rack hall) and `isaac.radio.mesh_blocked_fn` (Warp, not yet tested).

#### Fading

- Rician fast fading on the NR engine ([docs/channels.md](docs/channels.md#rician-fading)): `fading_rician=False`, `rician_k_db=None` (a fixed K for every link), `rician_k_from_los=True` (K log-normal per link from TR 38.901 Table 7.5-6 for the scenario, K = 0 when NLOS or blocked) and `rician_k_ramp_slots=4` (linear ramp on LOS changes), with a fixed per-link specular phase from the engine RNG. Runs on the reference, `graph` and `triton` backends. At K = 7 dB the 1% fade is about −9.8 dB, against −20 dB for Rayleigh.
- `nr_rng` has two new reset sites, KFAC (5) and KPHI (6). The `triton` kernel takes the new inputs `k_ptr`, `kf_ptr`, `kg_ptr`, `phi_ptr` and the `RICIAN` constexpr.
- Frequency-correlated fast fading across the RBGs on `L2` ([docs/channels.md](docs/channels.md#frequency-selective-fading)): `fading_freq_corr=False`. An exponential power-delay profile (`fading_pdp="exponential"`) gives the subband correlation 1/sqrt(1 + (2π Δf τ)²), applied through the Cholesky factor of the [S, S] correlation matrix to the AR(1) innovation and the initial state, so each subband keeps unit power. It needs `fading=True`. A set `fading_delay_spread_ns` (default `None`) gives every link one delay spread; otherwise each link draws a TR 38.901 Table 7.5-6 log-normal delay spread by its LOS state (`fading_ds_from_los=True`) on the log grid `fading_ds_grid=(3.0, 3000.0, 16)` of 3 ns to 3 µs, with the InF hall set by `inf_hall_volume_m3` / `inf_hall_surface_m2` or `inf_lg_ds`. Runs on the reference, `graph` and `triton` backends.
- At 20 MHz (13 subbands of 1.44 MHz) τ = 10 ns gives an adjacent-subband correlation of 0.996, and near-independence needs microseconds (0.037 at 3 µs). The UMi and UMa delay-spread rows use the V17.0.0 values (Release 19 changed them).
- `nr_rng` has a new reset site, DSPR (7), for the per-link delay spread. The `triton` kernel takes the new inputs `fcl_ptr`, `fci_ptr` and the `FCORR` constexpr (0, 1 or 2).

#### QoS scheduling

- `scheduler="qos"` ([docs/configurability.md](docs/configurability.md#qos-scheduling)), modelled on 5G-LENA `NrMacSchedulerOfdmaQos` and checked against `nr-mac-scheduler-ue-info-qos.h`. Each message gets a class from its `priority` (clamped to 0..Q−1), a class weighs (100 − P) · D with the 5G-LENA delay-budget factor D = PDB / (PDB − HOL) below the budget and PDB / 0.1 past it, and the metric per RBG is qw · r^γ / max(avg, `AVG_MIN`). D applies to classes with a finite `qos_pdb_ms`, in the uplink as well as the downlink.
- New fields, read only with `scheduler="qos"`: `qos_classes=2`, `qos_priority=(10, 70)` (5QI 5 vs 5QI 7), `qos_pdb_ms=(inf, inf)` and `qos_gamma=1.0`. `pf_update="rbg"` is allowed with it. Runs on the reference, `graph` and `triton` backends (per-robot weights `qw` in the kernel, `SCHED=3`, `QOS_G`).
- Bytes go out in class order through a stable per-step reorder of the messages whose bytes are all unsent (`FrameQueue.reorder`). The robot keeps one queue, so a new class-0 message waits behind at most one partly sent message. With `discard="pdcp_arrival"` the discard checks the head of the byte stream, which after a reorder is the most important class.

#### Multi-cell

- Radio link failure and re-establishment in the multi-cell NR engine ([docs/multicell.md](docs/multicell.md#radio-link-failure)): `rlf=False`, `rlf_qout_db=-8.0`, `rlf_qin_db=-6.0`, `n310=1`, `n311=1`, `t310_ms=1000`, `t311_ms=3000`, `reest_delay_ms=40` and `rlf_rlc=None` (follow `ho_rlc`). The serving-link SINR is compared with Qout and Qin once per control step, a robot in RLF cannot be scheduled, it re-establishes at the strongest suitable cell and goes idle when T311 expires. With `rlf=True` the step dict gains `rlf` and `counters()` gains `"rlf"`. `n_cells > 1`, reference and `graph` backends.
- The monitored quantity is the uplink `sinr_db`, so the −8 / −6 dB defaults are tuning knobs rather than the 3GPP PDCCH BLER points, and the N310, N311, T310 and T311 defaults are network-configured values.
- A3 target admission `a3_min_target_rsrp_dbm=None` (5G-LENA `MinTargetRsrpDbm`) in the NR engine and `NetSlotMC`; it is also the RSRP floor of the RLF cell search.
- With `rach=True` and `rlf=True`, RLF re-establishment goes through the contention-based RACH model instead of the fixed `reest_delay_ms` ([docs/multicell.md](docs/multicell.md#radio-link-failure)). Requests are taken once per control step; contention-free RACH and T301 are not modelled.

#### Power control, CQI and antennas

- Closed-loop uplink power control `ul_tpc=False` (TS 38.213 §7.1.1) on top of `ul_pc`, with `ul_tpc_mode="accumulate"` (or `"absolute"`), `ul_tpc_target_db=None` (the 10% BLER SINR of MCS 14 of `mcs_table`), `ul_tpc_steps_db=None` (38.213 Table 7.1.1-1), `ul_tpc_delay_slots=None` (k2) and `ul_tpc_range_db=20`. The offset enters `UlMac._pc()` wherever `pc_backoff` did (scheduler estimate, power split, inter-cell interference, energy tap), and a handover resets it. `L2`, every backend ([docs/configurability.md](docs/configurability.md#closed-loop-power-control-cqi-table-and-sector-antennas)).
- `cqi_table="38214"` (default `"mcs"`) reports the DL CQI on TS 38.214 Tables 5.2.2.1-2 / -3 with a CQI-to-MCS mapping (`phy.CQI_T1`, `CQI_T2`, `cqi_tables`); it needs `dl=True`.
- `gnb_antenna="sector"` (default `"isotropic"`) applies the TR 38.901 Table 7.3-1 gNB element per cell in `RadioMC.rx_dbm`, with `cell_azimuth_deg=None` (30, 150 and 270 degrees cycled over the cells), `cell_tilt_deg=0.0` and `gnb_antenna_gain_dbi=8.0`. It changes only the path gain, so every backend runs it ([docs/channels.md](docs/channels.md#antenna-patterns)).
- `isaac_net/tools/scene/bake.py --gnb-antenna sector --cell-azimuth ... --cell-tilt ... --gnb-antenna-gain` applies the TR 38.901 sector pattern at bake time, per grid point along the direct direction rather than per ray, and records it in the map metadata ([docs/scene-radio-map.md](docs/scene-radio-map.md#sector-antennas-at-bake-time)). `RadioMapChannel` raises if `gnb_antenna="sector"` is used with a sector-baked map, since the doubled pattern would shift links by up to 16 dB.

#### Access: RACH and DRX

- Contention-based RACH and connection setup on `L2` ([docs/access.md](docs/access.md)): `rach=False`, `rach_occasion_slots=20`, `rach_preambles=64`, `rach_rar_window_slots=10`, `rach_msg3_slots=10`, `rach_backoff_ms=20`, `rach_max_attempts=10`, `rach_initial="connected"` (or `"idle"`) and `rach_release_after_ms=None`. Collisions are counted per occasion and cell with one `scatter_add` over the preambles, and the preamble and backoff draws use counter-RNG sites 16 and 17.
- Connected-mode DRX: `drx=False`, `drx_inactivity_ms=100`, `drx_cycle_ms=160`, `drx_on_ms=10`, `drx_short_cycle_ms=None`, `drx_short_cycles=2`, `drx_start_offset_ms=0` and `drx_ul_wake="sr"` (or `"on_duration"`). 5G-LENA has RACH but no DRX.
- An `AccessStage` inside the engine gates robots through `MacLink.sched_ok`. With RACH or DRX on, the step dict gains `access_state`, `access_sleep_frac` and `rach_attempts`, and `counters()` gains `"access"`. Reference, `graph` and `triton` backends.
- `EnergyConfig.drx_sleep_power_w=None` (= `idle_power_w`) is charged for the dormant or idle share of each step.

#### Downlink, duplexing and tools

- Downlink traffic models on `L2` with `dl=True`: `TrafficModel(..., direction="dl")` or `.downlink()` ([docs/configurability.md](docs/configurability.md#downlink-models)). They draw from their own generator with a separate seed, so the uplink draws are unchanged. The step adds per DL frame `dl_delivered`, `dl_lost`, `dl_delay`, `dl_tag`, `dl_bytes`, `dl_generated` and `dl_deadline_miss`, and per robot `gen_dl_accepted` and `gen_dl_bytes`. Generated DL frames carry `cls < 0`, which keeps them apart from `EdgeLoop` `nr_dl` commands. Reference, `graph` and `triton` backends: on `graph` and `triton` the DL arrival gate reads static buffers refilled before each step, and the `graph` outputs are bitwise equal to the reference.
- Downlink background on `L2`: `BackgroundConfig.dl_traffic=()` (DL traffic models of the background UEs) and `dl_load_frac=0.0` (a fixed share of every DL RBG) ([docs/background-energy-sharding.md](docs/background-energy-sharding.md#downlink-background)).
- FDD: `NRConfig.duplex="fdd"` (default `"tdd"`) with `dl_n_prb=None` / `dl_bandwidth_mhz=None` (default: the UL carrier) gives an all-`U` UL carrier and an all-`D` DL carrier that share the subband grid and the fading state. The per-PRB DL SINR shifts by −10 log10(dl_nprb / nprb) ([docs/configurability.md](docs/configurability.md#duplexing-tdd-and-fdd)).
- Radio environment map export, `isaac_net.tools.rem` and the console script `isaac-net-rem`: per-cell path gain and RSRP, best-cell SINR, serving cell and LOS state as `.npz`, with an optional PNG ([docs/rem.md](docs/rem.md)).
- `SlotTap.dl_prb` counts the PRB-slots of DL transport blocks.

#### MIMO rank and mini-slot grants

- Rank-1/2 SU-MIMO on `L2` ([docs/configurability.md](docs/configurability.md#mimo-rank)): `n_layers_max=1` (2 turns it on), `rank_rule="sinr_los"` (or `"sinr"`, `"los"`), `rank_sinr_min_db=10`, `rank_k_max_db=3`, `rank_layer_penalty_db=3`, `ul_mimo=False` and `dl_mimo=True`. The rank is chosen per new TB from the wideband SINR and the Rician K of the serving link and kept per HARQ process (`h_rank`), the TBS counts the layers (`tbs_38214(layers=)`, also `tbs_lena`), and MCS selection and decoding use the per-layer SINR, SINR − 10 log10(rank) − `rank_layer_penalty_db`, with both layers as one codeword.
- With `n_layers_max=2` the step dict gains `rank` (UL) and `dl_rank`. There is no PMI, Type-I codebook or rank-conditioned CQI, so rank 2 at a 40 dB SNR gives about 1.75 times the rank-1 saturated throughput instead of 2. Reference and `graph` backends; `triton` refuses it.
- Mini-slot (type B) grants on `L2` ([docs/configurability.md](docs/configurability.md#mini-slot-grants)): `ul_mini_slot_symbols=None` (2, 4 or 7) splits every UL data slot into round(nsym / m) scheduling occasions, the last one taking the remaining symbols, and `mini_slot_dl=False` splits the DL data slots too. Fading, CQI, SR, handover and the N+I estimate run once per slot, and grant, PF, MCS, TBS, decode, HARQ and RLC once per occasion. Reference and `graph` backends; `triton` refuses it.
- Timers stay in slots, and feedback from an occasion takes effect at the next slot (OLLA, the lumped BSR and the UL CSI are frozen within the slot). With mini-slots on, `pf_window` counts occasions, `delay` carries the end fraction of the occasion within its slot, and occasion j decodes with the RNG slot key rel + j·N. At light load the mean delay of 10 B commands drops from 5.77 ms to 5.62, 5.51 and 5.42 ms for m = 7, 4 and 2, and the saturated throughput is 0.91, 0.83 and 0.55 of whole slots (DMRS overhead).

#### Triton backend

- Closed-loop UL power control (`ul_tpc`, accumulate and absolute) and the 38.214 CQI table (`cqi_table="38214"`) run in the fused kernel (constexprs `TPC`, `CQI38214`). The per-robot TPC state is carried through the step and stored back.
- The RACH / DRX access gate (`ACCESS`), FDD with a DL carrier of its own width (`FDD`, `NPRB_D`, a per-RBG DL PRB table) and the DL arrival gate of the DL traffic models (`DLGATE`, static buffers) run in the fused kernel. The kernel recomputes the per-slot schedulability from the `AccessStage` state, because DRX depends on activity within the slot, so with RACH or DRX on the UL and DL slots run in one kernel.

### Documentation

- Blockage models A and B, soft LOS, the Table 7.5-6 K-factors, the TS 38.214 CQI tables and the 802.11 VHT exclusions now cite ETSI TR 138 901 V17.0.0 / TS 138 214 V17.1.0, and every `[verify]` marker in the code is resolved ([docs/obstacles.md](docs/obstacles.md)).
- The paper is on arXiv as [arXiv:2610.02370](https://arxiv.org/abs/2610.02370); the README citation, `CITATION.cff`, the project URLs and the status page link to it.

### Changed

- The minimum torch version is 2.7: the `compile` backend sets `torch._dynamo.config.recompile_limit`, which first appears in 2.7 (it was `cache_size_limit` before).
- The sdist now includes `prototype/`.
- The `triton` NR backend refuses every feature it does not implement in one place, `NRTritonEngine.__init__`, with `TritonUnsupported` (both a `NotImplementedError` and a `ValueError`) and a message that points to `graph`. `NRTritonEngine.refusals(cfg)` now lists only several cells (and with them A3 handover and RLF), the SR / BSR grant pipeline (`ul_grant_model="bsr"`), rank-2 MIMO and mini-slot grants; SINR hooks are refused at the first step. A test checks the backend table of docs/configurability.md against the code.
- `core.slot_tap` counts a mini-slot occasion as its share of the slot's data symbols.
- `proactive_grant="per_period"` is refused with FDD, and every level other than `L2` refuses `rach` / `drx` (`ValueError`).
- With `radio="engine"`, the `blocked` output of `NetModule` reports the engine's blockage. The Isaac `blocked_fn` drives the engine radio only with `los_source="callback"` and is otherwise ignored with a warning.

### Fixed

#### L2 MAC and energy

- HARQ processes whose bytes were all purged at the deadline are freed at compaction (`MacLink.compact`). Before, with `discard="purge"` and `harq_fail="rlc_am"`, they retransmitted dead data at retransmission priority indefinitely; partially purged processes are kept.
- Admitted retransmissions now always take priority over new data on each RBG, and the PF average is floored at 1e-9 bytes (`AVG_MIN`), so a long-idle robot's decayed average can no longer outrank a retransmission. Behaviour is unchanged whenever every metric was below 1e9; the `triton` kernel mirrors the change.
- With `retx_priority=False`, RBGs won by a retransmission that falls short of its RBG count go to new data instead of staying empty.
- The wideband PF rate estimate respects `ul_mcs_max` / `dl_mcs_max`.
- `SlotTap.ul_tx_j` counts the slot's PUSCH symbols (12 in a U slot by default) instead of 14, and the new `SlotTap.ul_tx_s` gives the PUSCH time. `EnergyLoop` uses that PUSCH time for the circuit term and for the fixed-power path.

#### Randomness and sharding

- With poses on `L2` (one or several cells) and on multi-cell `L2-legacy`, the radio's shadowing, LOS-state and O2I draws now come from the engine's counter RNG, keyed by seed, global env id and episode. They no longer depend on E or on other envs' resets, which changes `L2` numerics with poses under `rng="engine"`; `rng="global"` is unchanged.
- `WIFI`: the radio's shadowing, LOS and O2I draws come from the engine counter RNG too, so an env's channel no longer depends on E or on other envs' resets.
- `ShardedEngine` is shard-invariant (bitwise equal to one engine) on `L2` with one or several cells, multi-cell `L2-legacy` and `WIFI` under `rng="engine"`. `L2` with traffic models or background users, any level with an edge loop that draws from the global RNG (`service_dist="exponential"`, `ret_jitter_ms > 0`), and `rng="global"` keep derived per-shard seeds (`sharded.shard_invariant`).
- `CounterRNG.reset` with duplicate env ids advances the episode once.
- `NetSlotMC` counts the A3 time-to-trigger and the handover interruption in its own UL slots (`proto_ul_slots_per_step`).

#### Configuration, wrappers and edge

- `NRConfig.unused_fields` / `fields_read_by(level, cfg)` follow the switches: DL fields only with `dl=True`, handover and interference fields only with several cells (interference also needs thermal noise), the noise fields of the active noise model, the fields of the selected channel, blockage and fading fields only when on, and `lena_ref_sc_per_rb` only with `tbs_mode="lena"`. `NetSlotMC` reads only the frame fields of its handover conversion.
- `fading_rho_from_speed` (and the tensor version `channels.doppler.rho_per_ms_from_speed`) is monotone: rho = 0 from the first zero of J0 (about 13.1 m/s at 3.5 GHz).
- `make_adaptive` applies `NRConfig.energy` (and `edge`) around the adaptive engine, so batteries drain. Background users are refused under `make_adaptive`.
- `EnergyConfig.seed` takes precedence over the engine seed. `EnergyLoop` raises on the legacy `step(t, x, cur_hid)` form, and the offered-load `BackgroundLoop` accounts the legacy form like the dict form.
- `EdgeConfig(return_path="delay")` sizes the command rate on the DL carrier (`dl_nprb`) with `duplex="fdd"`; TDD is unchanged.
- `EdgeConfig(return_path="nr_dl")` is refused on the `graph` and `triton` NR backends. `act_age` is NaN before a robot's first action, and the edge example treats it as stale.
- An adaptive cheap `L0` with an empirical `{"q", "p"}` marginal accepts handed-back frames.
- `srsran_like` / `oai_like` compute their slot counts from the given `mu` / `tdd_pattern`, `with_(fading_rho_per_ms=...)` keeps the explicit value, and `isaac_net.LEVELS` includes `"WIFI"`.

#### Channels and Wi-Fi

- TR 38.901 blockage model A (`blockage_model="stochastic"`) applies the eq. 7.6-22 loss only inside |φ_AOA − φ_k| < x_k and |θ_ZOA − θ_k| < y_k, as §7.6.4.1 states. Before, it applied the loss at every angle; the indoor average loss drops by about 0.15 dB, and the defaults are unchanged.
- TR 38.901 UMa/UMi with `ue_height_m` <= 1 m gave a non-positive breakpoint and 13–23 dB optimistic path loss. Heights outside Table 7.4.1-1 (UMa/UMi 1.5–22.5 m, RMa 1–10 m) and InF-SH/DH without h_UT < h_c < h_BS now raise `ValueError` when the channel is built.
- The Wi-Fi Poisson draw capped successful accesses at 8 per robot per sub-step, cutting goodput by 18% (80 MHz / 2 ms) to 66% (160 MHz / 5 ms). The cap now comes from the config with a 6-sigma margin (`poisson_cap`).
- The Wi-Fi event simulator charged frames lost to FER the collision time instead of Ts, unlike the mean-field model. Validation table B adds FER = 0.2 rows.
- With `channel="log_distance"`, Wi-Fi used the 40 dB constant of the 3.5 GHz NR carrier at 5.2 GHz. The default is now the free-space loss at 1 m at `carrier_ghz` (46.8 dB at 5.2 GHz); a user-set `pl_const_db` is kept.
- 802.11ac no longer offers VHT-MCS 6 at 80 MHz / 3 streams or MCS 9 at 160 MHz / 3 streams, and `wifi_mcs` reports MCS numbers.

#### Isaac Lab, MJX and the benchmark suite

- With `net_decimation > 1` a tagged message is no longer dropped by a same-class message in the same window. Between network steps `aoi_s` grows by one env step per tick, and `net_reset` clears the held output of reset envs.
- Benchmark result files are named `<task>__<variant>__<sim>__<preset>__<traffic>__<level>__<backend>__<baseline>__s<seed>__<hash>[__<label>].json`, so runs with different settings no longer overwrite each other. `bench report` groups by `sim` too, and `bench run` prints `n/a` for a missing metric instead of crashing.
- With `sim="isaac"`, `hazard_exposure` reads the fleet env's own per-step indicator, the same definition as torch.
- Building `NetModuleMJX` no longer resets the global CUDA RNG, and `MJXFleetEnv` observations follow the Isaac fleet env order.
- Warehouse example: the default config uses a 7 m gNB height, and the default network step respects multi-rate settings. `los_blocked_kernel` takes radio-frame inputs.

#### Backends

- The prototype reference and the reference NR engine store copies of the caller's step SNR and submit hid, so the prototype reference is bitwise equal to the eager and `graph` backends when a caller reuses its buffers.
- The NR `graph` backend recaptures on a change of input dtype, and `NRGraphEngine.submit` honours `tag`, `priority` and `deadline_ms`.
- The `L2-legacy` `triton` backend reads env ids from `rng.env`, so `set_env_offset` after capture works. `NetFast` capture restores the CUDA RNG: global-mode graph runs after capture are deterministic but no longer reproduce earlier global-mode numbers.

#### Bridges and measurement tools

- `Ns3NetModule` uses the isaac `NetConfig` / `TrafficRequest` and returns the `NetModule` dict. The old `step(poses, req)` form is kept.
- The ns-3 lockstep bridge interpolates poses from the last commanded end pose. This is a C++ change and needs a rebuild of the bridge program.
- OAI `rx_vt.csv` maps `t_tx_ns` to virtual time, and telnet replies are prompt-framed and checked.
- `owd` and `calibrate` skip send-error rows (`t_tx_ns < 0`) and count them as `send_errors`. `unwrap_slots(realtime=False)` and the manifest key `gnb.realtime_slots` are new.
- MAC-NR CCCH sizes are corrected (LCID 0 = 8 B, LCID 52 = 6 B, TS 38.321 Table 6.2.1-2).

### Tests

- NR `graph` / `triton` equivalence now runs under load, with a phase offset into the medium-load and burst phases.
- The frozen NR golden references build on a frozen copy of the shared modules (`tests/nr_frozen/base/`; re-freeze with `tests/scripts/refreeze_nr.py`), and the RNG hash is pinned to fixed values.
- New GPU tests cover the `compile` backend and partial-reset isolation of the free-running `triton` kernel.
- On CPU the shard test compares `L1` / `L2-legacy` float outputs to float32 rounding; on CUDA it stays bitwise.
- `tests/test_qos.py` and `tests/test_freqfade.py` cover the new scheduler and the frequency correlation, and the configs `qos`, `qos_rbg`, `ul_fcorr` and `ul_fcorr_rician` of `tests/nr_equiv.py` join the GPU equivalence lists.
- `tests/test_limits_closed.py` and `tests/limits_off_scenarios.py` check the closed limits, with ten switch-off configurations against `tests/fixtures/limits_off_golden.json` (digests valid on the platform that wrote them).
- `tests/test_mimo.py`, `tests/test_minislot.py`, `tests/test_triton_tpc_cqi.py` and `tests/test_triton_access_fdd_dl.py` cover the wave-3 features. The equivalence configs `ul_tpc`, `ul_tpc_abs`, `ul_dl_cqi38214`, `ul_tpc_cqi`, `ul_access`, `ul_fdd`, `ul_dl_fdd` and `ul_dl_traffic` join G1, G2 and G7, and `ul_dl_mimo2`, `ul_minislot2` and `ul_minislot4` join G1 and G7 (`graph` only).
- GPU and Isaac tests are skipped by marker, not by keyword. CI tests torch 2.7 on Python 3.10 and adds Python 3.12.

### Known limits

- The `triton` backend does not implement several cells, the SR / BSR grant pipeline, SINR hooks, rank-2 MIMO or mini-slot grants.
- No equivalence config yet combines TPC or the CQI table with the access gate, FDD or DL traffic on `triton`. The kernel changes were first compiled in the 2026-10-05 lab-box GPU run, where the teacher-forced G2 test is the decisive check.
- Mini-slot grants shorten only commands that fit the first occasion. With SR access a 200 B frame arrives one TDD period later, because the one-RBG bootstrap grant carries about 30 B in a 2-symbol occasion ([docs/configurability.md](docs/configurability.md#mini-slot-grants)).
- The Sionna RT bake applies the sector pattern per grid point along the direct direction, not per traced ray.
- The 802.11 VHT exclusions are confirmed through FreeBSD net80211 and the N_CBPS / N_ES rule, because the standard itself is paywalled.
- Soft LOS mixes path loss and shadowing linearly in dB, while TR 38.901 eq. 7.6-19 mixes the channel matrices with power weights.

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
