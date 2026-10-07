# Isaac Lab integration

The engine was run inside NVIDIA Isaac Lab 3.0 on Isaac Sim 6.1, installed natively on Windows 11 on the lab box, with the uplink stepped in lockstep with PhysX. This page gives the install recipe with exact versions, the workaround that lets CUDA jobs run while nobody is logged in at the console, the demo environment, the scale results up to 1,048,576 robots on an idle GPU, and the bugs fixed in the original Isaac adapter. The work was done on 2026-09-29, and the Isaac layer has since been rebuilt on `make_engine` and `NRConfig` and validated on the same install (see [Rebuilt layer](#rebuilt-layer-on-make_engine) below). The README section "Isaac Lab quick start" is the short version of this page.

## Versions

| Component | Version |
|:---|:---|
| OS | Windows 11, NVIDIA driver 617.14 (≥ 580.88 is required for the cu130 PyTorch build, 581.42 is recommended) |
| Isaac Sim | 6.1.0.0, installed with pip |
| Isaac Lab | 3.0, branch `release/3.0.0` at `30f8e69` (package `isaaclab` 25.0.0) |
| Python | 3.12.14 in a uv venv |
| PyTorch | 2.12.0+cu130, torchvision 0.27.0 |
| Isaac Lab extras | physx 7.3.0, newton 9.1.0 and 1.6.0 (two packages, as listed by the installer), rsl-rl-lib 5.5.1, skrl 2.1.0, warp-lang 1.17.0 |
| Triton | `triton-windows` 3.8.0.post29 (community build, bundles TinyCC, no MSVC needed) |
| Tools | uv 0.12.20 (standalone zip), MinGit 2.56.0 (portable) |

Isaac Lab 3.0 needs Isaac Sim 6.1 (5.1 and older are not supported). The whole installation lives under `C:\isaac5g` and takes 39.6 GB. The only system-wide change is `LongPathsEnabled = 1`, which the Isaac Lab Windows page requires.

## Install recipe

The scripts are in `scripts/windows/` and follow the official Isaac Lab 3.0 page "Python environment with Isaac Sim" for Windows with uv. They were run from an administrator PowerShell over ssh, copied to the box and invoked with `-File`, because piping multi-line blocks into `-Command -` silently drops them. They still assume the `C:\isaac5g` paths.

| Step | Script | What it does | Time |
|:---|:---|:---|:---|
| 0 | `env.ps1`, dot-sourced by every other script | puts uv, MinGit and the venv on `PATH`, keeps the uv, Python and pip caches under `C:\isaac5g`, sets `OMNI_KIT_ACCEPT_EULA=YES` for these processes only, and redirects `USERPROFILE`, `HOME`, `LOCALAPPDATA`, `APPDATA` and `TEMP` to `C:\isaac5g\home` so the Kit, Omniverse, Triton and uv caches stay inside the install | |
| 1 | `01_bootstrap.ps1` | sets `LongPathsEnabled=1`, unpacks uv and MinGit into `C:\isaac5g\tools` | 1 min |
| 2 | `02_install_isaacsim.ps1` (detached through `launch.ps1`) | `uv venv --python 3.12 --seed`; `uv pip install "isaacsim[all,extscache]==6.1.0.0" --extra-index-url https://pypi.nvidia.com --index-strategy unsafe-best-match --prerelease=allow`; `uv pip install -U torch==2.12.0 torchvision==0.27.0 --index-url https://download.pytorch.org/whl/cu130`; clones Isaac Lab `release/3.0.0` with the git-lfs filter disabled | 8 min |
| 3 | `03_install_isaaclab.ps1` | `isaaclab.bat -i` with all extras | 3 min |
| 4 | `04_triton_windows.ps1` (optional) | `uv pip install triton-windows`, needed for the engine's `triton` backend | 1 min |
| 5 | `05_verify_cartpole.ps1`, **run as SYSTEM** through `systask.ps1` | official headless cartpole training | 3 min |

The verification ran `isaaclab train --rl_library rsl_rl --task Isaac-Cartpole-Direct physics=isaacsim_physx` with 4,096 envs for 30 iterations at 18,421 steps/s on the shared GPU. Headless is the default in Isaac Lab 3.0. The first Isaac launch takes about 45 s and later launches about 10 s. The NVIDIA Omniverse EULA was accepted on the PI's instruction, only inside the install and run processes.

Four problems cost time. Detached `Start-Process` children die with the ssh session because they belong to its job object, so `launch.ps1` uses `Win32_Process.Create` through CIM instead, which is fine for installs that do not need the GPU. The Isaac Lab clone failed on git-lfs, which MinGit lacks, and cloning with the LFS filter disabled fixes it (the LFS files are docs media and test images, and assets come from the cloud). The rsl_rl configuration must pass through `handle_deprecated_rsl_rl_cfg(cfg, check_rsl_rl_version())` as in the official train script, or rsl-rl-lib 5.5.1 rejects `stochastic`. Triton is not available on Windows by default, and `triton-windows` 3.8 works with torch 2.12 cu130 and compiles the fused 40-slot kernel.

## Running CUDA jobs with nobody logged in

**CUDA is unavailable from the ssh session.** `torch.cuda.is_available()` returns False with `cudaErrorNotSupported`, and `nvidia-smi.exe` reports "Access is denied". Both the OpenSSH token (network logon) and an S4U scheduled task (batch logon) fail, and an Interactive-logon task cannot run because nobody is logged in at the console after a reboot. **Running as SYSTEM works.** Every GPU run is therefore a one-shot SYSTEM scheduled task: `scripts/windows/systask.ps1 -Script <ps1> -Log <log> -Name <task>` registers it, and `wait.ps1 -Log <log> -Name <task> -Max 570` waits for it on the box and removes the task when it finishes. All such tasks were removed after the runs.

Three side effects follow. The first bare SYSTEM CUDA probe wrote an NVIDIA compute cache under `C:\Windows\System32\config\systemprofile\AppData\Roaming\NVIDIA`, which is why `env.ps1` redirects the profile directories. WSL cannot run as SYSTEM (`WSL_E_LOCAL_SYSTEM_NOT_SUPPORTED`), so in-process GPU sampling uses `C:\Windows\System32\nvidia-smi.exe`. **Once someone logs in at the console, runs can use `-LogonType Interactive` tasks instead of SYSTEM, and nothing else changes.** This is open item 6 in [STATUS.md](STATUS.md).

## The demo environment

`isaac_net/examples/isaac_fleet_env.py` defines `NetFleetEnv`, a `DirectRLEnv` with E envs of R velocity-driven rigid spheres in a 150 m × 150 m arena, which runs the example fleet task of `fleet_task.py` (goals, hazards and detection frames sent over the uplink) inside Isaac Lab. Physics is PhysX through Isaac Sim with dt = 1/50 s and decimation 5, so one control step is 0.1 s and 40 UL slots. The robots are spheres of radius 0.3 m floating at z = 0.5 with gravity disabled, their velocity is written every substep through `write_body_link_velocity_to_sim_index`, and robot-robot contacts are on. Each robot's action is a velocity (vx, vy) up to 3 m/s plus a send channel bucketed into none, a small 4 KB frame or a large 30 KB frame. Each robot observes 12 values, including its queued frames, the age of information of its reports and its SNR, and the centralized observation and action are `[E, 12R]` and `[E, 3R]`.

Frames are captured at the pose at the start of the step, and the network step runs in `_get_dones` before `_reset_idx`. The network is wired through `isaac_net/isaac/net_module.py` (a `NetModule` with `reset(env_ids)`, `submit(t, req)` and `step(t, poses, cur_tag) -> dict`) and the `NetEnvMixin` in `mixins.py`, whose four hooks are `net_setup`, `net_step`, `net_reset` and `net_obs`. In the demo, the legacy slot-level engine ran through its fast backends with partial resets written in place into the persistent buffers, so the captured CUDA graphs stayed valid. The rebuilt layer gets the same from `make_engine`, whose engines keep per-env clocks and exact partial resets at every level and backend. The radio uses log-distance path loss plus a per-env shadowing field redrawn on reset, with SNR averaged over 4 poses interpolated across the control step. A per-message tag carries an application label, and `step` returns whether a tagged message was delivered. `mdp/events.py` provides a network domain-randomization event term.

In the demo, the fast engine driven through `NetModule` on the eager backend matched the demo's registry reference engine bitwise, including a partial reset of every third env, and on the `graph` and `triton` backends a partial reset left other envs bit-identical, with delivered frames per step of 0.1274 (eager), 0.1274 (`graph`) and 0.1275 (`triton`) on the same workload. The rebuilt layer's tests are described below. An end-to-end PPO run through the rsl_rl wrapper of Isaac Lab 3.0, with the `triton` network in the loop at 1,024 × 16, trained without errors for 30 iterations in 241 s (3.3k env-steps/s, 53k robot-steps/s under contention). It is a pipeline check only.

## Scale results

**NR engine in its validated configuration, 3 processes per cell (2026-09-30).** `benchmarks/isaac/run_repeats.ps1`, run as one-shot SYSTEM tasks (one per size, removed afterwards), ran the network off and the NR uplink on `triton` as 3 separate processes each, interleaved. The NR engine ran `ul_v2l`, `lena_validation_v2()` without the SR / BSR grant pipeline (lumped 40-slot SR-to-grant delay) with the task's frame sizes, 16-frame buffer and 2 s timeout, which replays the 5G-LENA sweep with a median p50 error of −3.5% / −5.7% / −0.8% at light / moderate / saturated load ([fidelity-vs-lena.md](fidelity-vs-lena.md#scale-configurations)). Actions are random, and each robot sends on about two thirds of steps, half of them large frames, so the uplink is saturated, the worst case for the network's cost. Each process runs 10 warm-up steps, then 3 windows of 50 steps, and reports its median window. "Network only" is the median of 3 windows of 20 isolated `submit` + `step` calls at the end of the run. Before every process the GPU was at 0% utilization and no other python or Kit process ran; the WSL load average was 0.00–0.66 (other agents' jobs paused), and Windows Defender used 0–4.1 cores in the samples around each process. Values are the mean over the 3 processes with the smallest and largest in parentheses; "added per step" is the step-time difference of each adjacent off / NR pair, mean and range. Results: `benchmarks/results/uncontended/isaac_scale_repeats_rtx4090.csv` (per process, with the CPU state) and `isaac_scale_repeats_summary_rtx4090.csv`.

| E × R | Robots | Off (control steps/s) | NR `ul_v2l` `triton` | On / off | Added per step | Network only |
|:---|---:|---:|---:|---:|---:|---:|
| 2048 × 128 | 262,144 | 5.38 (5.04–5.61) | 4.11 (3.82–4.34) | 0.76 | 57 ms (52–63) | 50 ms |
| 4096 × 128 | 524,288 | 2.95 (2.84–3.07) | 2.44 (2.37–2.52) | 0.83 | 71 ms (45–86) | 100 ms |
| 8192 × 128 | 1,048,576 | 1.50 (1.37–1.57) | 1.24 (1.23–1.24) | 0.83 | 140 ms (80–171) | 199 ms |

One control step is one Isaac Lab environment step for all E environments, so robot-steps per second is the control-step rate times E × R. The largest configuration, 1,048,576 robots, runs at 1.24 control steps per second with the NR uplink in its validated configuration (1.30 M robot-steps/s), against 1.50 with the network off, within a device peak of 19.5 GiB (1.2 GiB of it held by a desktop session). The Isaac step is bound by host work (device-wide GPU utilization 23–38% during the timed windows). At 262k robots the network adds at least its isolated cost, so nothing overlaps there. From 524k robots up the step grows by 40–86% of the isolated network cost, so part of the network's GPU work, about 30% on average, is hidden behind PhysX and Python host work. The ranges come mostly from the network-off processes, whose rate varies by 8–14% between processes, so a single process per cell cannot resolve on / off differences of a few percent. `graph` was not rerun: it replays about 6,000 kernels per step, so **`triton` is the backend for scale and `graph` is for bitwise-reference runs**. The eager reference is not viable inside Isaac at these sizes.

**Earlier single-process runs (`L2-legacy` and `NRConfig()`).** The first idle-GPU campaign (`benchmarks/isaac/run_uncontended.ps1`, `isaac_scale_rtx4090.csv`) ran one process per configuration, with the NR engine in its default configuration `NRConfig()`, which is not validated against 5G-LENA (median delay 29–76% low in the replay). Every run started at 0% utilization with 36 MiB in use, and no WSL or other Windows GPU job ran.

| E × R | Robots | Network off (control steps/s) | `L2-legacy` `triton` | On / off | Network only | NR `L2` `triton`, `NRConfig()` | On / off | Network only |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2048 × 128 | 262,144 | 5.19 | 4.08 | 0.79 | 11 ms | 4.15 | 0.80 | 37 ms |
| 4096 × 128 | 524,288 | 2.58 | 2.39 | 0.93 | 23 ms | 2.44 | 0.95 | 72 ms |
| 8192 × 128 | 1,048,576 | 1.50 | 1.51 | 1.01 | 45 ms | 1.32 | 0.88 | 146 ms |

At 1M robots `L2-legacy` ran at 1.51 control steps per second (1.59 M robot-steps/s) within a device peak of 13.9 GiB. Its on / off ratios of 0.93–1.01 from 524k robots up, and the NR ratios of this table, are single runs and lie within the process-to-process spread measured above, so they show that the legacy kernel's cost is small next to the Isaac step, not by how much. The windows of a run agree to within 10%, except the first network-off window at 2048 × 128. The earlier contended runs of this table, and a comparison with them, are in [performance.md](performance.md#contended-vs-uncontended): the Isaac throughput was not inflated by the busy GPU, while the isolated network cost was about 2.3× lower on the idle GPU.

**What limits scale.** Scene startup grows about linearly with the number of robots, from the PhysX clone and a `RigidObjectCollection` of R distinct objects: 259–272 s at 262k, 510–548 s at 524k and 1,038–1,084 s at 1M robots in the first campaign, and 322–363, 647–760 and 1,158–1,585 s in the repeats, possibly slowed by Defender activity on the host. The process held 4.3–7.8 GiB of device memory at 262k robots, 6.0–11.7 GiB at 524k and 9.4–19.5 GiB at 1M (network off to NR `ul_v2l`, the desktop session's 1.2 GiB included in the repeats), so the NR engine at 1M robots needs most of a 24 GB GPU. Taking 2 control steps per second as the bar for interactive use (a 24-step PPO rollout in at most 12 s), every configuration up to 524k robots meets it with either network, and 1M robots runs at 1.2–1.5 steps/s.

## Adapter bugs fixed

The original adapter in `isaac/netmodule.py` and `isaac/isaac_env_skeleton.py` had the following problems. All are fixed in the rebuilt layer, and both files are gone (`netmodule.py` only re-exports the new names):

1. **The first delivered capture of every episode was dropped.** The message history started `seen_cap` at 0 and tested strictly (`newest_cap > seen_cap`), so capture 0 never reached the receiver. It now starts at −1 in `__init__` and `reset`.
2. **A host sync every control step in `_enqueue`** from `nonzero` plus fancy-index scatter. It is replaced by a sync-free one-hot write into FIFO slot `count`, and the slot-level engine stays bitwise equal to the reference.
3. **A capture-time off-by-one in the skeleton.** It pushed the post-physics pose under capture step t, which made age of information and delay optimistic by one control step. Frames are now captured in `_pre_physics_step`, as in the demo.
4. **Episodes ended one step early**, because the skeleton's timeout used `>= max_episode_length - 1` where Isaac Lab 3.0 cartpole uses `>=`. Minor.
5. **No per-message label.** An application tag with `delivered_tag` / `tag_delivered` outputs was added, which the example task needs.
6. The fast engine then had only a full `reset()`, so `reset(env_ids)` lived in the Isaac `NetModule`. The engine API work on `main` has since added partial resets to every level and backend.

The port of the slot-level engine itself was correct and bitwise equal to the reference. A naive CUDA-graph capture of the reference engine's slot loop, tried first, was **not** bitwise equal because its RNG stream diverged, and it was dropped in favor of the fast engine.

## Rebuilt layer on make_engine

`isaac_net.isaac.NetModule(level, E, R, device, NRConfig, backend)` builds its engine with `make_engine`, so every level is available inside Isaac Lab: L0, L0DR, L1 and L2-legacy on the reference and fast backends (`triton` for L1 and L2-legacy), the NR engine L2 on the reference backend, the surrogates with a fit file, and the ORACLE / NOCOMM bounds. The module adds only the Isaac radio (`isaac/radio.py`: per-env radio parameters for domain randomization, several gNBs, line-of-sight blockage, SNR averaged over 4 poses interpolated across the step), the per-message tag, and the freshness outputs (`last_cap`, `aoi_s`). Multi-cell configurations use the engine's own radio (`radio="engine"`). The demo's `NetConfig` remains as a deprecated alias without defaults of its own (see [Configuring the network](#configuring-the-network)), in which `rung="L2"` means `L2-legacy`. The demo's registry engine was retired, so its per-env `bg_load` and per-env L0 parameters are gone; `L0DR` covers randomized delay.

Tests, validated on 2026-09-29 on WSL and on the Windows install (GPU jobs as SYSTEM tasks):
- `tests/test_isaac_layer.py`: the module equals the reference engine driven directly, bitwise, at L0, L0DR, L1, L2-legacy (eager) and L2 (NR), through a mid-run partial reset, on CPU and GPU; `graph` is bitwise equal with injected draws; `graph` and `triton` keep other envs untouched on a partial reset.
- `tests/test_isaac_env.py` (marker `isaac`): inside Isaac Lab, the fleet env's network equals a reference-backend replay of the recorded PhysX poses, sends, tags and RNG stream with zero difference, through partial resets that DirectRLEnv makes itself, at L2-legacy, L1 and L0DR; every level and backend steps the fleet env. All 9 cases pass.
- The README quick start, run verbatim as a SYSTEM task, exits 0 on every command. Its benchmark row (256 × 16, L2-legacy `triton`) gave 4.18 control steps/s and 17 ms of network per step under 99% GPU contention, against 5.03 and 13 ms in the demo, within the ±40% run-to-run noise. A 5-iteration PPO smoke run took 38.5 s.

## Configuring the network

**One configuration.** The network in the loop is configured by the same `NRConfig` that `make_engine` takes: message sizes, frame buffer, timeout, control step, radio and cell layout, MAC, PHY, and the L0 and L0DR delay parameters. The level and the backend are the `make_engine` arguments. The Isaac layer adds only a small `IsaacNetCfg` for what an Isaac Lab task needs on top:

```python
from isaac_net import NRConfig
from isaac_net.isaac import IsaacNetCfg, NetEnvMixin

nr = NRConfig(msg_sizes=(4000.0, 30000.0), control_step_ms=100.0)
isaac = IsaacNetCfg(pose_asset="robots", gnb_pos=((0.0, 0.0, 6.0),),
                    obs_features=("aoi", "sinr", "queue_len", "delay_history"), obs_history=4,
                    dr_ranges={"noise_dbm": (-95.0, -85.0), "shadow_sigma_db": (3.0, 9.0),
                               "gnb_offset_m": (-10.0, 10.0)})
observation_space = R * (TASK_OBS + isaac.obs_dim(nr))       # in the env cfg
self.net_setup("L2-legacy", R, nr, "triton", isaac=isaac)     # in _setup_scene
```

| Field group | Fields | What it does |
|:---|:---|:---|
| Pose source | `radio`, `pose_asset`, `pose_body_ids`, `pose_offset_m`, `gnb_pos`, `gnb_height_m`, `pose_chunks` | `radio="isaac"` computes the SNR from poses with the Isaac radio, and `"engine"` passes the poses to the engine's own radio (needed for `n_cells > 1`). With `pose_asset` set, `net_step(None, send, ...)` reads the end-of-step poses from that scene entity (`RigidObjectCollection`, `Articulation` or `RigidObject`). Without `gnb_pos`, the gNBs are at `NRConfig.gnb_xy()` at height `gnb_height_m`. |
| Multi-rate | `net_decimation`, `net_substeps` | `net_decimation = k` steps the network every k env steps and merges the messages of the window into one per robot. `net_substeps = m` runs m network steps per env step. `net_setup` checks that the env step times k / m equals `NRConfig.control_step_ms`. Between network steps (k > 1) `net_step` returns the last output with the per-step flags cleared and `aoi_s` advanced by one env step per tick. The merge keeps the largest class, and on a tie the tagged message. `net_reset` clears the held output of the reset envs, and `net_obs()` keeps the last network-step observation. |
| Blockage | `blockage`, `robot_blockers`, `blocker_radius_m`, `blockage_db` | `blockage=False` switches blockage loss off and ignores `blocked_fn`. `robot_blockers=True` lets the robots of an env block each other's line of sight as spheres. |
| Domain randomization | `dr_ranges`, `dr_mode`, `dr_interval_steps`, `dr_strict` | per-env uniform draws, redrawn at every reset (`"reset"`), every U[lo, hi] network steps (`"interval"`), or never (`"off"`) |
| Observation | `obs_features`, `obs_history`, `obs_time_scale_s` | the network features `net_obs()` returns, sized by `obs_dim(nr)` |
| Network overrides | `nr` | a dict of `NRConfig` fields applied to the `NRConfig` given to `net_setup` (a `preset` key starts from that preset), or a whole `NRConfig` that replaces it; `IsaacNetCfg.resolve_nr(nr)` applies it. This is what the command-line overrides of the registered tasks fill ([Config files and presets](config-files.md#hydra-overrides-in-the-registered-tasks)) |
| Logging | `log_kpis`, `log_every` | the per-step network KPIs in `extras["log"]` (see [Network KPIs in TensorBoard](#network-kpis-in-tensorboard)) |

The Isaac radio takes its nominal parameters from the `NRConfig` (`ue_tx_dbm`, the noise floor, `pl_const_db`, `pathloss_exp`, `shadow_sigma_db`, `shadow_modes`). With the Isaac radio the engine sees only the SNR, so L1, L2-legacy and QA get the legacy cell fields and stay on their fast backends whatever radio fields the `NRConfig` sets. `NetModule(..., strict=True)` raises when the `NRConfig` sets fields the level ignores (`NRConfig.unused_fields`). The keyword arguments `pose_chunks`, `gnb_pos` and `radio` of `NetModule` and `net_setup` still work as shortcuts for the `IsaacNetCfg` fields.

**Deprecated `NetConfig`.** `NetConfig` warns on construction and has no defaults of its own: every field left at `None` comes from the `NRConfig` or from `IsaacNetCfg`. Two defaults therefore changed for code that relied on them. The message sizes are the `NRConfig` default (4,000 and 30,000 bytes) instead of 1,500 and 12,000, and the gNB is at the origin at height 0 instead of on a 6 m mast. The level defaults to `L2-legacy` and the backend to `reference`.

**Observation features.** `obs_features` picks among the features below, in the order given. Every feature uses one normalization: times are divided by the time scale (`obs_time_scale_s`, default 50 control steps, which is 5 s at 100 ms) and clamped to [0, 1], queue length is divided by the frame buffer, queue bytes by the frame buffer times the largest message size, dB quantities by 40 dB, and flags and one-hot codes are 0 or 1. An env that has not stepped since its reset observes zeros. The earlier `net_features` and the fleet env's own scaling are replaced by this one, and `net_features` remains as a deprecated wrapper for the default selection.

| Feature | Width | Value |
|:---|---:|:---|
| `delivered_mask` | F | message slot delivered this step (FIFO slots as queued before the step) |
| `msg_delay` | F | delay of each delivered slot / time scale; 0 where not delivered |
| `aoi` | 1 | age of information / time scale |
| `queue_len` | 1 | queued messages / frame buffer |
| `queue_bytes` | 1 | queued bytes / (frame buffer × largest message size) |
| `sinr` | 1 | SINR in dB / 40 |
| `rsrp` | 1 | (received power − nominal noise floor of the `NRConfig`) in dB / 40; with the Isaac radio this is the SNR plus the env's noise-floor offset |
| `serving_cell` | G | one-hot serving cell, G = number of gNBs |
| `last_delivered` | 1 | at least one message of the robot was delivered this step |
| `delay_history` | k | delays of the last k delivered messages / time scale, newest first (`obs_history = k`) |
| `blocked` | 1 | line of sight to the serving gNB blocked in the last pose chunk |

The default selection is `aoi`, `sinr`, `queue_len` and `last_delivered`, which is the earlier `[E,R,4]` observation. `isaac_net.isaac.obs_dim(features, nr, n_cells, history)` gives the width without building a module.

**Network domain randomization.** `dr_ranges` maps a key to a `(lo, hi)` range. The radio keys `p_tx_dbm`, `noise_dbm`, `pl_const_db`, `pl_exp`, `shadow_sigma_db`, `blockage_db` and `gnb_offset_m` (cell placement: a per-env x and y offset of every gNB) are per-env parameters of the Isaac radio. The delay keys `delay_median_steps`, `delay_log_sigma` and `loss` are written into the `NRConfig`: into `dr_delay_median_steps`, `dr_delay_log_sigma` and `dr_loss` at L0DR, whose engine draws them per env at every reset, and into the `l0_*` fields at L0, which takes fixed values only (lo = hi). `dr_support(level, nr, isaac)` reports per key whether a level honors it. For the delay keys it uses the field groups behind `NRConfig.unused_fields`. A key the level does not honor warns, or raises with `dr_strict=True`:

| Level | Radio keys (Isaac radio) | Delay keys |
|:---|:---|:---|
| L0 | no | yes, fixed values |
| L0DR | no | yes, per env at reset |
| L05, L05Q, L1, L2-legacy, L2, QA, NN | yes | no |
| TR, GE, ORACLE, NOCOMM | no | no |

With `radio="engine"` no radio key is honored, because the engine radio has one parameter set per config. The delay keys are redrawn only at reset, also with `dr_mode="interval"`. `mdp.randomize_network` remains as an EventTerm for tasks that keep all randomization in their event manager. With `ranges=None` it draws the module's `dr_ranges`.

**Command-line options.** `benchmarks/isaac/bench.py`, `train_ppo.py` and `tests/scripts/isaac_fleet_check.py` take `--obs` (a comma-separated feature list or `all`), `--env_decimation`, `--net_decimation`, `--net_substeps` and `--dr` (noise, shadowing, path-loss exponent and gNB placement per env), defined in `isaac_net/examples/fleet_args.py`.

**Validation of the configuration layer (2026-09-29, Windows install, SYSTEM tasks).** `tests/test_isaac_config.py` (obs_dim against every feature, the normalization feature by feature, reset zeroing, delay keys into `NRConfig`, `dr_support` against `NRConfig.unused_fields`, radio and cell-placement draws at reset and at an interval, blockage switches, the mixin's pose source and both multi-rate modes) and `tests/test_isaac_layer.py` pass (30 passed, the long GPU NR case deselected). The Isaac-marked tests are now 12, and all pass: the 9 earlier cases, a replay check inside Isaac with every observation feature and network domain randomization (the 44 feature columns and the per-env radio parameters equal the reference replay with zero difference), and bench smoke runs of `net_decimation = 5` (env step 20 ms) and `net_substeps = 2` (env step 200 ms). A 5-iteration PPO run at 256 × 16 with `--obs aoi,sinr,queue_len,delay_history,serving_cell --dr` finished in 7.4 s. The README quick start, run verbatim afterwards, exited 0 on every command. Its benchmark row (256 × 16, L2-legacy `triton`) gave 22.6 control steps/s and 5.5 ms of network per step with the GPU at 71–78% utilization from other jobs, and the PPO smoke run took 9.2 s.

## Train the registered task

`isaac_net.isaac.tasks` registers the tasks below with gymnasium in the layout of `isaaclab_tasks`: an `entry_point`, `disable_env_checker=True`, and `env_cfg_entry_point`, `rsl_rl_cfg_entry_point`, `skrl_cfg_entry_point` and `default_agent="rsl_rl"` in the kwargs. Importing the package imports nothing from Isaac Lab, because every entry point is a `"module:attr"` string resolved when the task is built.

| Task id | Env | Network |
|:---|:---|:---|
| `Isaac-NetFleet-Direct-v0` | `NetFleetEnv`, 1,024 envs × 16 robots | `L2-legacy` on `graph` |
| `Isaac-NetFleet-Direct-L0-v0` | the same | `L0`: independent lognormal delays without contention, the delay baseline |
| `Isaac-NetFleet-Direct-Warehouse-v0` | `WarehouseFleetEnv`, 64 envs × 16 robots in Isaac Lab's warehouse (Nucleus asset) | NR engine `L2` on `reference` with two cells and the engine's log-distance radio, no baked radio map |
| `Isaac-NetFleet-Manager-v0` | `ManagerBasedRLEnv` built with `NetManagerCfg` (see [Manager-based workflow](#manager-based-workflow)): a static fleet that learns when to send | `L2-legacy` on `graph` |

The rsl_rl configuration (`tasks/agents/rsl_rl_ppo_cfg.py`) is the one `benchmarks/isaac/train_ppo.py` ran with: 256-128 ELU MLPs with observation normalization, 24 steps per env per iteration and an adaptive learning rate. The skrl configuration (`tasks/agents/skrl_ppo_cfg.py`) has the same network and PPO settings, written as functions that return the dict Isaac Lab reads from its skrl YAML files, so no package data is needed.

Isaac Lab 3.0's train script imports only `isaaclab_tasks`, so the registration has to reach it in one of two ways. The rsl_rl backend takes `--external_callback`, a function it calls before it reads the task metadata, and the callback `isaac_net.isaac.tasks.register` imports the package and registers the ids. For every library, including skrl, `python -m isaac_net.isaac.tasks.train` does what `scripts/reinforcement_learning/train.py` does after importing the package, and it takes the same arguments. Run either from the Isaac Lab checkout, where logs go to `logs/<library>/<experiment>`. Windows (PowerShell, after `. C:\isaac5g\env.ps1` and `uv pip install --no-deps -e .` in this repository):

```powershell
cd C:\isaac5g\IsaacLab
isaaclab train --rl_library rsl_rl --task Isaac-NetFleet-Direct-v0 --external_callback isaac_net.isaac.tasks.register --num_envs 1024 --max_iterations 300
python scripts\reinforcement_learning\train.py --rl_library rsl_rl --task Isaac-NetFleet-Direct-v0 --external_callback isaac_net.isaac.tasks.register
python -m isaac_net.isaac.tasks.train --rl_library skrl --task Isaac-NetFleet-Direct-L0-v0 --num_envs 1024 --max_iterations 50
python -m isaac_net.isaac.tasks.train --task Isaac-NetFleet-Manager-v0 --num_envs 1024     # rsl_rl, the default agent
python -m isaac_net.isaac.tasks.train --play --rl_library rsl_rl --task Isaac-NetFleet-Direct-v0 --num_envs 16 --checkpoint latest --viz kit
```

Linux uses the same commands with `/` in the paths. Three details differ from older Isaac Lab tutorials:

- Isaac Lab 3.0 runs headless by default and has no `--headless` flag. An unknown flag is passed on to Hydra, which rejects it, so leave it out. `--viz kit` opens a viewport.
- With `--external_callback`, pass `--rl_library` explicitly. The selector reads the default library before the callback has registered the task. `python -m isaac_net.isaac.tasks.train` registers first, so `default_agent` applies there.
- The registered configs take their physics backend from `ISAAC_NET_PHYSICS` (`isaacsim_physx` by default, `ovphysx`, `newton`), as `make_cfg` does. They do not declare Isaac Lab physics presets, so the `physics=` selector fails on them. Leave it out.

Command-line overrides reach the env config through Isaac Lab's `env.` prefix, for example `env.num_robots=32 env.net_level=L1 env.net_backend=triton`. `NetFleetEnv.__init__` calls `finalize_fleet_cfg`, which rebuilds the scene when `num_robots` changed and sizes the action and observation spaces from the robot count and `IsaacNetCfg.obs_features`.

Every `NRConfig` field is reachable the same way through the `nr` dict of the task's `IsaacNetCfg`, and so is a preset by name:

```bash
python -m isaac_net.isaac.tasks.train --task Isaac-NetFleet-Direct-v0 \
    env.net_isaac.nr.ul_pc=true env.net_isaac.nr.ul_tpc=true env.net_isaac.nr.frame_buffer=32
python -m isaac_net.isaac.tasks.train --task Isaac-NetFleet-Direct-Warehouse-v0 \
    env.net_isaac.nr.preset=warehouse_private_5g env.net_isaac.nr.frame_buffer=16
```

A misspelled field stops the run before the simulation starts, with the closest field names (`NRConfig has no field 'ul_tcp' (did you mean 'ul_tpc' or 'ul_pc'?)`). [Config files and presets](config-files.md#hydra-overrides-in-the-registered-tasks) describes the rules, the presets and the YAML files.

## Network KPIs in TensorBoard

With `IsaacNetCfg.log_kpis=True` (the default), every network step puts its KPIs into `extras["log"]` as 0-dim tensors on the env's device. Isaac Lab's RL wrappers pass `extras` to the runner, and rsl_rl's runner averages every `extras["log"]` entry over the steps of an iteration and writes it to TensorBoard. A key that contains a `/` keeps its name there, so the values appear as their own group, `net/`, next to the episode statistics. The Direct mixin writes them in `net_step` and the manager-based runtime in its `net_step` term. The values cover all envs and robots of the batch:

| Key | Value |
|:---|:---|
| `net/aoi_mean_s` | mean age of information over the robots after the step (s) |
| `net/aoi_p95_s` | 95th percentile of the robots' AoI (`torch.quantile`, linear interpolation) (s) |
| `net/delay_mean_ms` | mean delay of the messages delivered in the step (ms); 0 when none was delivered |
| `net/delivered_frac` | delivered messages / messages resolved in the step (delivered or lost); 0 when none was resolved |
| `net/dropped_frac` | lost messages / messages resolved: timed out at the application deadline, or dropped by the engine (RLC UM loss, handover flush, radio link failure) where the step reports it |
| `net/queue_bytes_mean` | mean queued uplink bytes per robot after the step |
| `net/sinr_mean_db` | mean SINR of the step (dB) |
| `net/prb_util` | uplink PRB-slots granted / available in the step; level `L2` (NR engine) only |
| `net/harq_bler` | failed / sent first transmissions of uplink transport blocks in the step (0 when none was sent); level `L2` only |
| `net/rlf_frac` | share of robots in radio link failure; `L2` with several cells and `rlf=True` |
| `net/access_sleep_frac` | mean share of the step the robots spent DRX-dormant or idle; `L2` with `drx` or `rach` |

`net/prb_util` and `net/harq_bler` come from the cumulative counters of the NR MAC (`core/record.py`, `mac_counters`), differenced between two network steps. A full engine reset zeroes the counters, and the next step then counts from zero. The step-dict KPIs of a step with `net_substeps = m > 1` are those of the returned output: `delivered`-based values cover all m substeps, the per-message ones (`delay`, `delivered_frac`, `dropped_frac`) the last substep. With `net_decimation = k > 1`, the KPIs are refreshed on the env steps that step the network, and the env steps in between repeat them.

**Cost.** About ten reductions over the `[E, R]` and `[E, R, F]` step tensors, one quantile over the E · R AoI values, and at level `L2` a copy of the MAC counters (a few `[E]` and `[E, max_harq_tx + 1]` tensors). Everything stays on the device and nothing is synchronized with the host. The runner calls `.item()` when it logs. `log_every = n` computes them every n network steps and repeats the last values in between, and `log_kpis=False` switches them off:

```bash
python -m isaac_net.isaac.tasks.train --task Isaac-NetFleet-Direct-v0 env.net_isaac.log_every=10
python -m isaac_net.isaac.tasks.train --task Isaac-NetFleet-Direct-v0 env.net_isaac.log_kpis=false
python -m isaac_net.isaac.tasks.train --task Isaac-NetFleet-Manager-v0 env.isaac_net.isaac.log_kpis=false
```

Two details of the plumbing matter for custom envs. Every step assigns a new dict to `extras["log"]` (the earlier entries plus the KPIs), because rsl_rl keeps a reference to each step's dict until it logs, and an update in place would rewrite the values of the earlier steps. `ManagerBasedRLEnv._reset_idx` replaces `extras["log"]` with a new dict on every step that resets an env, after the termination terms have run, so `NetRuntime` wraps the env's `_reset_idx` once to put that step's KPIs back. The last values are also available as `env.net_kpis` (Direct) or `env.isaac_net.net_kpis` (manager-based), and `isaac_net.isaac.kpis.step_kpis(out)` computes the step-dict part from any `NetModule.step` output.

## Manager-based workflow

`NetManagerCfg(...).apply(env_cfg)` puts the network into a `ManagerBasedRLEnvCfg` in one call. Call it at the end of `__post_init__`, after `decimation` and `sim.dt` are set: without an `NRConfig` it sets the network step to the env step, and when the network is built the step is checked against the env step as in the Direct mixin:

```python
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils import configclass
from isaac_net import NRConfig
from isaac_net.isaac import IsaacNetCfg, NetManagerCfg, NetMarkersCfg

@configclass
class MyEnvCfg(ManagerBasedRLEnvCfg):
    ...                                         # scene with a "robots" asset, observations.policy, actions, ...

    def __post_init__(self):
        self.decimation, self.sim.dt = 5, 1 / 50            # env step 0.1 s
        NetManagerCfg(level="L2-legacy", backend="graph",
                      nr=NRConfig(msg_sizes=(4000.0, 30000.0), control_step_ms=100.0),
                      isaac=IsaacNetCfg(pose_asset="robots", gnb_pos=((0.0, 0.0, 6.0),)),
                      obs_terms=("net_aoi", "net_sinr", "net_queue", "net_delivered"),
                      aoi_weight=-1.0, send_action=True, markers=NetMarkersCfg()).apply(self)
```

`apply` stores the `NetManagerCfg` as `env_cfg.isaac_net` and adds these terms next to the existing ones. The env cfg holds it rather than the scene cfg, because `InteractiveScene` treats every scene attribute as an asset.

| Manager | Term | Function |
|:---|:---|:---|
| terminations | `net_step` (placed first, never ends an episode) | `mdp.net_step_done(env)` |
| events | `net_reset`, mode `"reset"` | `mdp.net_reset(env, env_ids)` forwards the reset env ids |
| observations | one term per name in `obs_terms`, in the group `obs_group` | `func(env, asset_cfg=None) -> [E, R·k]` |
| rewards | `net_aoi` with weight `aoi_weight` (skipped with `None`) | `mdp.net_aoi_penalty(env)` |
| actions | `net_send` (with `send_action=True`) | `NetSendActionCfg`: one channel per robot, bucketed into none, class 1, ..., class `send_classes` |

The observation terms use the normalization of the Direct env and observe zeros until an env's first network step after a reset. Per-robot values are robot-major, and `asset_cfg.body_ids` selects robots when it is a list:

| Term | Width per robot | Value |
|:---|---:|:---|
| `net_aoi` | 1 | age of information divided by the time scale, clamped to [0, 1] |
| `net_delivered` | 1 | at least one message of the robot was delivered this step |
| `net_delay` | 1 | delay of the newest message delivered this step divided by the time scale |
| `net_queue` | 1 | queued messages divided by the frame buffer |
| `net_sinr` | 1 | SINR in dB divided by 40 |
| `net_los` | 1 | line of sight to the serving gNB |
| `net_access_state` | 5 | one-hot idle, RACH, connected and DRX-dormant, then a radio-link-failure flag. It needs `NRConfig(rach=True)`, DRX or RLF at level `L2`, and without them every robot reads as connected |

`mdp.net_send_cost` is a second reward term, the fraction of robots that sent in the last step. The network itself is `env.isaac_net`, a `NetRuntime` that binds `NetEnvMixin` to the manager-based env, so both workflows run the same multi-rate, pose-source and overlay code. The terms build it on first use from `env_cfg.isaac_net`. The first use is the observation manager probing the term shapes, which happens after the scene exists and the simulation has been reset. A custom env can also call `isaac_net.isaac.mdp.net_setup(env, level, num_robots, nr, backend, isaac=...)` itself. Without a send action, every robot sends a message of class `traffic_class` every `traffic_period` network steps, so AoI and delay are defined for any task. A task can also write the classes of a step with `env.isaac_net.write_send(send)`. [`isaac_net/examples/isaac_manager_fleet_env.py`](https://github.com/ZzZTripleZzZ/isaac-net/blob/main/isaac_net/examples/isaac_manager_fleet_env.py) is the complete example behind `Isaac-NetFleet-Manager-v0`.

**Where the network step runs.** After physics, `ManagerBasedRLEnv.step` (Isaac Lab 3.0) computes the terminations, then the rewards, then resets the finished envs (the `"reset"` events), then applies the `"interval"` events, and last computes the observations. The network must step after physics, because it needs the end-of-step poses, and before the observations. The termination manager is the first post-physics stage. A network step placed there sees the end-of-step poses and the queues before any reset, the rewards read the fresh output, the resets clear it for finished envs, and the observations see the result. The Direct mixin runs `net_step` in `_get_dones` for the same reason, because `DirectRLEnv.step` calls `_get_dones`, `_get_rewards`, `_reset_idx` and `_get_observations` in that order. `NetManagerCfg(placement="interval")` uses the event term `mdp.net_step` in `"interval"` mode instead, with `interval_range_s = (step_dt, step_dt)` and `is_global_time=True`, so it runs once per env step. Interval events run after the rewards and the resets, however, so the reward then reads the previous step's network output, and a reset env takes one network step from its new pose before its first observation. `NetRuntime.step` runs at most once per env step, so placing it twice does not step the network twice.

## Viewport overlays

`NetMarkers` (`isaac_net/isaac/markers.py`) draws the network state in the viewport. It is enabled with `net_setup(..., markers=NetMarkersCfg())` in a Direct env, the `net_markers` field of the fleet env cfgs, or `NetManagerCfg(markers=NetMarkersCfg())`:

- a line from every robot to its serving gNB, coloured by the SINR of the step in five bins (below 0 dB red, 0–5 dB orange, 5–10 dB yellow, 10–20 dB light green, 20 dB and above green). Links whose line of sight is blocked are dimmer and dashed.
- a mast at every gNB with a translucent coverage disc. The disc radius is where the nominal SNR of the radio's path-loss law falls to `coverage_snr_db`, without shadowing. With the Isaac radio it follows the per-env parameters, so network domain randomization moves and resizes the discs.
- a vertical bar above every robot. Its height grows linearly with the age of information up to `aoi_bar_height_m` at `aoi_max_s`, and its colour is one of five AoI bins.
- a sphere above every robot that is idle, in random access, DRX-dormant or in radio link failure. These need the access model of level `L2`, and nothing is drawn without it.

<!-- ![Viewport overlays of the fleet env](img/isaac_markers.png) -->

*Screenshot to come: `docs/img/isaac_markers.png`, taken on the lab box with `python -m isaac_net.examples.isaac_markers_demo --task Isaac-NetFleet-Direct-v0 --num_envs 4 --viz kit`. Uncomment the image line above once the file exists.*

All shapes are prototypes of one `isaaclab.markers.VisualizationMarkers` (21 cylinders and spheres, sized through the instance scale, quaternions in Isaac Lab 3.0's x, y, z, w order), so they work in the Kit viewport and in the Newton visualizers. With `line_backend="debug_draw"` the links are drawn through the Kit debug-draw interface (`isaacsim.util.debug_draw`) instead, which shows them only in the Kit viewport. The geometry is computed in torch on the network's device (`isaac_net/isaac/marker_geometry.py`) every `update_every` env steps, for the envs in `env_ids` (by default the first `max_envs`, 4), and moved to the host in one copy. When nothing displays the stage, `NetMarkers` is inert: no GUI, no visualizer and no offscreen rendering, which is the default headless run. Its `update` then returns at once, so nothing is computed or transferred and nothing from Isaac Lab is imported.

| `NetMarkersCfg` field | Default | Meaning |
|:---|:---|:---|
| `enable`, `links`, `gnbs`, `coverage`, `aoi`, `states` | `True` | what is drawn |
| `env_ids`, `max_envs` | `None`, 4 | envs drawn |
| `update_every` | 1 | redraw interval in env steps |
| `line_backend` | `"markers"` | `"markers"` (cylinders) or `"debug_draw"` (Kit lines) |
| `sinr_edges_db`, `sinr_colors` | 0, 5, 10, 20 dB; red to green | the 5-bin SINR colormap |
| `aoi_edges_s`, `aoi_colors`, `aoi_max_s`, `aoi_bar_height_m` | 0.2, 0.5, 1, 2 s; green to red; 5 s; 2 m | the AoI bars |
| `nlos_dim`, `nlos_dashes` | 0.35, 4 | blocked links: colour factor and dashes per link |
| `coverage_snr_db`, `coverage_opacity` | 0 dB, 0.12 | coverage discs |

`python -m isaac_net.examples.isaac_markers_demo --task <id> --num_envs 4 --seconds 30 --viz kit` runs any registered task with the overlays on and random actions, and prints the env steps and the number of overlay instances every second.

**Tests.** `tests/test_isaac_ux.py` checks the parts that need no Isaac Lab on a CPU: the registration metadata (ids, entry points defined in their modules, the skrl dicts, the external-callback contract), every manager term against a fake env that holds a real `NetModule`, the overlay geometry from synthetic inputs (link end points, colour bins, dashes, AoI bar heights, coverage radius against the radio, the single host copy), and that the overlays are inert when headless. Its Isaac-marked cases run `tests/scripts/isaac_tasks_check.py`, which builds the registered tasks with `gym.make`, overlays on, steps them and draws one forced overlay frame. Those cases still have to be run on the lab box.

## Next steps

The open items are shorter startup (`clone_in_fabric=True`, one multi-instance asset per env, a Newton backend or a pure-tensor pose integrator for the planar robots), a per-robot parameter-shared policy wrapper that reshapes observations and actions to `[E·R, 12]` and `[E·R, 3]` because a centralized MLP does not scale to R = 128, a downlink `NetModule` that gates commands in `_pre_physics_step`, and a benchmark of the line-of-sight blockage path (`segment_sphere_blocked`, cost O(E · R · G · (R + M)) per pose chunk, with the Warp mesh kernel suggested for R ≥ 64).
