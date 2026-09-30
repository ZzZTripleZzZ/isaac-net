# Benchmark suite

`isaaclab_net.bench` gives network-aware multi-robot learning a common set of tasks, metrics and baselines. Every task runs E environments of R robots on one GPU, talks to the network only through the same `NetModule` the Isaac Lab layer uses, and reports the same per-episode metrics, so a result on one task can be read next to a result on another, and a new method can be compared with the shipped baselines on equal terms. The suite defines what to run and what to measure. It ships baseline sanity numbers that show the pipeline works, and no tuned results.

```bash
python -m isaaclab_net.bench list                                   # tasks, variants, levels, presets, baselines
python -m isaaclab_net.bench run --task coop_map --level L2-legacy --backend triton \
    --baselines random,heuristic,ppo_mlp --seeds 0,1,2 --out results/
python -m isaaclab_net.bench report results/                        # mean ± 95% CI over seeds, as a Markdown table
python -m isaaclab_net.bench calibrate --task all --level L2-legacy --backend triton   # load calibration
```

```python
import torch
from isaaclab_net.bench import TaskConfig, make_task

task = make_task(TaskConfig(task="coop_map", level="L2-legacy", backend="triton", num_envs=64, seed=0), "cuda")
obs = task.reset()                                   # [E, R, task.obs_spec.dim]
cont = torch.zeros(64, 16, task.action_spec.cont_dim, device="cuda")
send = torch.full((64, 16), 1, dtype=torch.long, device="cuda")
obs, reward, done, info = task.step(cont, send)      # reward [E, R], done [E]; info["episodes"]: finished rows
```

## Task API

**`NetTask`** (`isaaclab_net/bench/base.py`) is the base of every task. It is pure torch with kinematic 2-D robots in a 150 m arena and one gNB at the arena corner, unless the NRConfig preset says otherwise. All task state is fixed-shape tensors with leading dims `[E, R]`, and `reset(mask)` resets any subset of envs and leaves the others bit for bit unchanged. Episodes have a fixed length and all envs end at the same step. `step` returns the metric rows of the envs that ended and resets them.

**Configuration.** A `TaskConfig` names everything that decides a run: `task`, `variant`, `level` (any `make_engine` level), `backend` (`reference`, `eager`, `graph`, `compile`, `triton`), `sim` (`torch`, or `isaac` for the Isaac Lab variant), `preset` (the NRConfig preset), `traffic` (the traffic preset), `num_envs`, `num_robots`, `episode_steps`, `seed`, `net_obs` (the network observation features) and `level_params` (a fit file for the fitted levels). The task builds its NRConfig from the preset and then sets the application fields it owns: message sizes, control step, timeout and, for EdgeControl, the edge server. Every result file records the full `TaskConfig` and the task description.

| Preset | NRConfig | Notes |
|:---|:---|:---|
| `default` | `NRConfig()` | the legacy single cell: gNB at the corner, fixed −90 dBm floor |
| `netslot_compat` | `netslot_compat()` | NR engine closest to the legacy slot model |
| `srsran_like`, `oai_like` | fitted to public srsRAN and OAI latency | see the public-data calibration page |
| `lena_like` | the 5G-LENA reference scenario | needs the locally generated 5G-LENA tables |
| `multicell3` | `multicell(3)` | three cells and the engine's own radio; not with `coverage_nav`, whose holes live on the task's radio |

The traffic presets are `policy` (only the messages the policy sends) and `policy+telemetry` (plus 200 B telemetry every 50 ms per robot, generated inside the NR engine, so level `L2` only).

**Observation.** `task.obs_spec` lists named blocks of the per-robot observation `[E, R, D]`: the task blocks first, then one `net:<feature>` block per selected network feature. The network features are the `NetModule` selection with its single normalization (see [Isaac Lab](isaac-lab.md#configuring-the-network)). The default is `aoi`, `sinr`, `queue_len` and `last_delivered`, and `TaskConfig.net_obs` selects any of `delivered_mask`, `msg_delay`, `aoi`, `queue_len`, `queue_bytes`, `sinr`, `rsrp`, `serving_cell`, `last_delivered`, `delay_history` and `blocked`.

**Action.** `task.action_spec` is hybrid: a continuous vector `cont [E, R, cont_dim]` in [−1, 1] and one categorical send choice `send [E, R]` in `range(n_send)`, with the names of the choices in `send_choices`. In three tasks choice 0 sends nothing and choice c ≥ 1 sends one message of class c. In EdgeControl the choice is the upload rate.

**Network step.** Frames are captured at the start-of-step position and the network steps with the end-of-step position, as in the Isaac Lab layer. EdgeControl runs five network steps of 20 ms per task step and goes through `EdgeLoop`, whose edge stage is captured as its own CUDA graph on a GPU.

**Seeds.** A task draws every random number from its own generator seeded by `TaskConfig.seed`, the engine and the radio get seeds derived from it, and the scripted heuristic has a separate generator, so the task's draws do not depend on the policy. `torch.manual_seed` is also set, because the NR engine (`L2`) and the edge loop draw from the global generator. A run with seed s trains on seed s and evaluates on seed s + 10000, the same for every baseline. Two runs with the same seed, level, backend and device give the same episodes; `tests/test_bench.py` checks this bitwise on CPU with the `reference` backend.

## Metrics

Every episode yields one row per env, and every count is per robot:

| Key | Meaning |
|:---|:---|
| task metric | the task's own metric (table below), with its unit and direction in the result file |
| `return` | mean per-robot episode return |
| `sent`, `sent_<choice>` | messages handed to the network, and how often each send choice was taken |
| `refused` | messages the full frame buffer refused |
| `deliveries`, `drops` | messages delivered, and lost (application timeout, RLC UM loss, handover flush) |
| `delivery_ratio` | deliveries / (deliveries + drops) |
| `delay_p50_ms`, `delay_p95_ms` | per-message delay from capture to delivery, quantiles over the episode |
| `aoi_mean_s` | age of the newest delivered capture, averaged over robots and network steps |
| `offered_mbps`, `delivered_mbps` | application bytes submitted and delivered, per env and second |
| `energy_j` | per-robot energy, present only when the step dict carries `energy_j [E, R]` (an energy wrapper) |

Tasks add their own extras: `goals` (per robot and episode) in Fleet-Alert, `stopped_frac`, `slow_frac` and `in_hole_frac` in CoverageNav, and `stale_frac`, `command_age_ms` and `edge_drops` (edge and command losses per robot and episode) in EdgeControl. The delay quantiles come from a log-spaced histogram per env with 240 bins from 0.1 ms to 60 s (about 6% bin width, plus a bin for zero delay), interpolated inside the bin. All accumulators live on the device with leading dim `[E]`, so a partial reset clears only the envs that ended.

## Tasks

| Task | Task metric | Messages | Send choices | Continuous action | Task step |
|:---|:---|:---|:---|:---|:---|
| `fleet_alert` | `hazard_exposure` (share of robot-steps inside a hazard, ↓) | 4,000 / 30,000 B | none, small, large | velocity | 100 ms |
| `coop_map` | `map_age_s` (mean edge-map age, s, ↓) | 1,200 / 9,000 B | none, small, large | velocity | 100 ms |
| `coverage_nav` | `progress_m` (m toward the goals per robot and episode, ↑) | 250 / 2,500 B | none, status, image | velocity | 100 ms |
| `edge_control` | `tracking_error_m` (mean robot-target distance, m, ↓) | 300 B | 10 Hz, 25 Hz, 50 Hz | gain, horizon | 100 ms (5 network steps of 20 ms) |

Episodes last 300 task steps (30 s). The application timeout is 2 s in the first three tasks and 0.5 s in EdgeControl.

**Fleet-Alert** (`fleet_alert`, the example fleet task). Robots drive to random goals. Hazards appear near a random robot, grow for 5 s and last 10 s. A camera frame detects a hazard with probability 0.6 within 25 m (small frame) or 1.0 within 50 m (large frame), and the fleet learns the hazard only when a detecting frame is delivered. Until then robots cannot see it and pay 1 per step inside it, while progress and 2 per reached goal are rewarded. *Mechanism:* detection latency under self-generated contention. Large frames detect more, but when many robots send them together the uplink queues build up, and the detection reaches the fleet late or times out.

**CoopMap** (`coop_map`, cooperative mapping). The edge keeps a 15 × 15 map of 10 m cells. A delivered patch refreshes every cell within 10 m (small patch) or 20 m (large patch) of where the robot was when it captured the patch, stamped with the capture step, and cell age is capped at 10 s. Each robot's reward is minus the fleet's mean cell age plus a credit for the age reduction its own patches caused. *Mechanism:* freshness under contention. A patch describes the place and time of its capture, so queueing delay ages the map it refreshes, and large patches cover more cells but delay everyone's patches.

**CoverageNav** (`coverage_nav`, coverage-aware navigation under a remote supervisor). Robots drive to random goals. The supervisor must hear a status update (any delivered message) from each robot at least every 1 s, or the robot is stopped and may only crawl at 25% speed. A robot drives at full speed only if its newest delivered camera image is at most 1 s old, else at 40%. Each env has 5 coverage holes of 15 m radius with 25 dB extra path loss, drawn at reset, and the observation includes the SNR at four probe points 15 m away. *Mechanism:* coverage-dependent reachability. Inside a hole or at the cell edge the link cannot carry a message before its deadline, so the route decides whether the supervisor hears the robot, and images load the cell for everyone. The holes act through the SNR, so they matter only at levels that read it (`L05` and above, `QA`, `NN`).

**EdgeControl** (`edge_control`, edge-offloaded tracking control). Each robot tracks its own moving target (Ornstein-Uhlenbeck velocity with 2 m/s standard deviation per axis, reflecting walls). The policy acts every 100 ms and picks the state upload rate and two controller parameters carried in every state message, the gain k in [0.5, 5] 1/s and the target prediction horizon h in [0, 0.5] s. The edge (2 servers per env, 2 ms per message, FIFO, 200 ms deadline) computes u = v_tgt + k (p_tgt + h v_tgt − p_robot), clipped to 4 m/s, from the delivered state, and the command returns over the downlink (2 ms plus the transmission time at the robot's SINR). The robot keeps applying its newest command, also after it is older than 0.5 s (hold-last). *Mechanism:* closed-loop latency under the team's own load. A higher upload rate gives fresher commands until the uplink and the shared edge servers queue up, and then every robot's loop latency grows.

### Variants

| Variant | What changes |
|:---|:---|
| `default` | the task as described |
| `light` | negative control: every message size is multiplied by the task's `LIGHT_SCALE` (0.01 Fleet-Alert, 0.025 CoopMap, 0.1 CoverageNav and EdgeControl), so the heaviest choice at every step offers about a tenth of the default cell's capacity. Everything else is identical. |
| `background` | background UEs share the cell. Registered only when `NRConfig` has a `background` field (the background-UE feature) and a builder exists: `bench.register_background(lambda task: <the field's value>)`, or a `BackgroundConfig()` default in `isaaclab_net.core.background`. |

The light variant removes the load the team creates, not the per-robot link. A robot at the cell edge or in a coverage hole still loses messages, which is why CoverageNav keeps a lower delivery ratio in its light variant.

### Load calibration

`bench calibrate` drives every robot with the task's scripted motion and a fixed send choice at every step, once with the lightest choice that sends and once with the heaviest, and reports offered and delivered Mbit/s per env, the delivery ratio and the delay quantiles. The numbers below are for `L2-legacy` on `triton`, 64 envs × 16 robots and one 30 s episode, measured on the shared lab GPU on 2026-09-29.

| Task | Variant | Send choice (every step) | Offered (Mbit/s) | Delivered (Mbit/s) | Delivery ratio | Delay p50 / p95 (ms) | Refused per robot and episode |
|:---|:---|:---|---:|---:|---:|---:|---:|
| fleet_alert | default | small | 5.12 | 2.48 | 0.57 | 508 / 1931 | 34.0 |
| fleet_alert | default | large | 38.4 | 0.442 | 0.02 | 1541 / 1990 | 58.5 |
| fleet_alert | light | small | 0.0512 | 0.0511 | 1.00 | 11 / 62 | 0.2 |
| fleet_alert | light | large | 0.384 | 0.371 | 0.98 | 21 / 919 | 2.7 |
| coop_map | default | small | 1.54 | 0.904 | 0.66 | 37 / 1731 | 25.7 |
| coop_map | default | large | 11.5 | 1.77 | 0.20 | 1037 / 1974 | 51.8 |
| coop_map | light | small | 0.0384 | 0.0383 | 1.00 | 10 / 70 | 0.3 |
| coop_map | light | large | 0.288 | 0.274 | 0.97 | 21 / 763 | 3.4 |
| coverage_nav | default | status | 0.32 | 0.263 | 0.86 | 21 / 1084 | 11.6 |
| coverage_nav | default | image | 3.2 | 1.19 | 0.44 | 187 / 1869 | 39.0 |
| coverage_nav | light | status | 0.032 | 0.0288 | 0.92 | 10 / 167 | 6.4 |
| coverage_nav | light | image | 0.32 | 0.242 | 0.81 | 21 / 1288 | 15.5 |
| edge_control | default | 10Hz | 0.384 | 0.334 | 0.87 | 20 / 277 | 0.0 |
| edge_control | default | 50Hz | 1.92 | 1.13 | 0.70 | 14 / 412 | 239.0 |
| edge_control | light | 10Hz | 0.0384 | 0.0382 | 1.00 | 8 / 38 | 0.0 |
| edge_control | light | 50Hz | 0.192 | 0.186 | 0.99 | 10 / 135 | 23.3 |

At the default sizes every task can load the cell past what it carries. With the default single-cell geometry the legacy engine delivers at most about 2.5 Mbit/s per env in these runs, and small frames at every step already offer more than that in Fleet-Alert. Large frames at every step collapse goodput further (0.44 Mbit/s in Fleet-Alert), because most frames time out after part of them has been sent. The light variants bring the lightest choice to a delivery ratio of 0.92 to 1.00 with a median delay of 8 to 11 ms. Their heaviest choice still has a long p95 tail from robots near the far corners of the arena, which are at the cell edge whatever the load. In CoverageNav the holes keep the light variant's delivery ratio at 0.81 to 0.92 by design.

## Baselines

| Baseline | What it does |
|:---|:---|
| `random` | continuous action uniform in [−1, 1], send choice uniform (own generator) |
| `heuristic` | the task's scripted motion from privileged task state (`task.heuristic()`: goal seeking with hazard or hole avoidance, exploration toward the oldest map cell, or fixed controller gains), with a queue-aware send rule: the task's preferred choice when the robot's uplink queue is empty and at least `min_interval` steps passed since its last send, else the idle choice (nothing, or 10 Hz in EdgeControl) |
| `ppo_mlp` | PPO with one actor shared by every robot: tanh MLP (2 × 128), Gaussian continuous head, categorical send head, separate value MLP. GAE(0.99, 0.95), clip 0.2, 4 epochs × 8 minibatches, entropy 0.005, Adam 3e-4, gradient-norm clip 0.5 |
| `ppo_gru` | the same with a GRU trunk (128) shared by actor and value head, trained on the rollout sequences with the hidden state reset at episode starts |

PPO is trained on `--ppo_envs` envs for `--ppo_iters` iterations of `--ppo_horizon` steps and evaluated deterministically (mean continuous action, most likely send choice). The CLI defaults (30 iterations of 32 steps) are a short run, meant for smoke tests and not for converged baselines.

## Results format

`bench run` writes one JSON file per (task, variant, level, backend, baseline, seed), named `<task>__<variant>__<level>__<backend>__<baseline>__s<seed>[__<label>].json`:

| Field | Content |
|:---|:---|
| `schema` | `"isaaclab-net-bench/1"`; `bench report` reads only files of this schema |
| `label` | free-form tag (for example `sanity`), part of the report's grouping |
| `task`, `variant`, `level`, `backend`, `sim`, `preset`, `traffic`, `baseline`, `seed` | the run's identity |
| `config` | the full `TaskConfig` |
| `task_spec` | description, mechanism, metric (key, unit, direction), observation blocks, action spec, message sizes, episode length, time steps, edge configuration |
| `train` | `null`, or for PPO: architecture, iterations, horizon, learning rate, hidden size, learning curve (per iteration: return, task metric, finished episodes), train seconds, task steps, samples |
| `eval` | evaluation seed, envs, robots, episodes, number of rows, and `metrics`: the mean of every metric over the rows (`null` when no row has a finite value); plus the rows themselves with `run(..., keep_rows=True)` |
| `timing` | build, train and evaluation seconds, seconds per task step, robot-steps per second |
| `environment` | torch version, device, GPU name, host, Python version, git commit, time |

`bench report` groups the files by task, variant, level, backend, preset, traffic, baseline and label, and shows for each metric the mean over seeds and the half-width of the two-sided 95% Student t interval (n = number of seeds, no interval for n = 1). `--json` writes the same aggregate as JSON.

## Adding a task

1. Subclass `NetTask` in `isaaclab_net/bench/tasks/<name>.py`. Set `NAME`, `DESCRIPTION`, `MECHANISM`, `MSG_SIZES`, `SEND_CHOICES`, `CONT_NAMES`, `TASK_BLOCKS` (names and widths of the task features), `METRIC` (a `MetricSpec`) and `METRIC_REDUCE`, and, if they differ from the defaults, `EPISODE_STEPS`, `CONTROL_STEP_MS`, `NET_SUBSTEPS`, `TIMEOUT_STEPS`, `EDGE` and `LIGHT_SCALE`.
2. Implement `_task_reset(mask)` (redraw the state of the masked envs with `self.rand` / `self.randn`), `_task_obs()`, `_step(cont, send)`, which calls `self._net_step(cls, end_positions, ...)` for every network step and returns the reward `[E, R]`, the task metric's per-step value `[E]` and a dict of extras, and `heuristic()`, which returns the scripted continuous action, the preferred send choice and the idle choice.
3. Register the class in `TASKS` (`tasks/__init__.py`).
4. Before using the task, check that the network can matter at all. Run `ORACLE` and `NOCOMM`, which must be far apart, then run `bench calibrate` at the default and light variants and pick `LIGHT_SCALE` so that the light variant's heaviest choice offers about a tenth of the capacity. Document the task here with its mechanism and calibration.
5. Add it to `tests/test_bench.py`. The parametrized tests cover shapes, seeds, partial resets, the bounds and the heuristic.

## Submitting a baseline

A baseline is an object with `reset(task)` and `act(task, obs, done) -> (cont, send)`, where `done` is the previous step's `[E]` mask (`None` at the start). It must use only the observation, unless it is labeled as privileged like `heuristic`. Evaluate it with `bench.runner.evaluate(task, policy, episodes)`, or add it to `make_policy` in `baselines.py` so that `bench run --baselines <name>` works. A submission is the result files (at least 5 seeds, on the default and light variants of every task, at the levels it claims), the command line that produced them, and the training budget in `train`. Use the default `TaskConfig` apart from `level`, `backend`, `seed` and the variant, and report any other change in the label.

## Isaac Lab variant

`TaskConfig(sim="isaac")` runs Fleet-Alert inside Isaac Lab 3.0 through the existing fleet env (`examples/isaac_fleet_env.py`, `NetEnvMixin`) and exposes it with the `NetTask` interface, so the runner, the random and PPO baselines and the metrics apply unchanged (`bench/isaac_adapter.py`). It needs a running Isaac Sim app (launch `isaaclab.app.AppLauncher` first). The heuristic reads privileged task state and runs only on `sim="torch"`. The adapter has not yet been run inside Isaac Lab, and the other three tasks exist only in torch.

## Sanity results

These are pipeline sanity results, not baseline results. They show that every task, variant, baseline, backend and the report run end to end. `benchmarks/suite/sanity.sh` produced them on 2026-09-29 and 2026-09-30 on the shared lab RTX 4090, which other jobs kept 97–98% busy, so the timings are pessimistic. Every run used 64 envs × 16 robots and 2 seeds, one evaluation episode on 64 envs, and for PPO only 20 iterations of 32 steps on 64 envs (655,360 robot-steps), far from convergence. The two tables are checks of two engine configurations, one cheap level on the `graph` backend and `L2-legacy` on `triton`, and are not meant to be compared with each other. Entries are mean ± half-width of the 95% t interval over the 2 seeds (t = 12.7 at n = 2, so the intervals are wide). Counts are per robot and episode, the delay is in ms and the AoI in s. At `L0` the delay does not depend on the message size, so its light rows repeat the default rows.

### L0 on graph

| task | variant | baseline | n | task metric | return | deliveries | drops | delay_p95_ms | aoi_mean_s | train s | eval s/step |
|:---|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fleet_alert | default | random | 2 | hazard_exposure 0.0217 ± 0.028 ↓ | -6.48 ± 8.5 | 200 ± 1.3 | 0 ± 0 | 11.4 ± 0.056 | 0.15 ± 0.0028 | 0 | 0.0376 |
| fleet_alert | default | heuristic | 2 | hazard_exposure 0.000151 ± 0.0019 ↓ | 88.2 ± 0.032 | 300 ± 0 | 0 ± 0 | 11.4 ± 0.064 | 0.1 ± 0 | 0 | 0.0367 |
| fleet_alert | default | ppo_mlp | 2 | hazard_exposure 0.0203 ± 0.015 ↓ | 50.2 ± 3.4 | 80.1 ± 7.3e+02 | 0 ± 0 | 11.4 ± 0.53 | 7.06 ± 74 | 78 | 0.0526 |
| fleet_alert | default | ppo_gru | 2 | hazard_exposure 0.0206 ± 0.0062 ↓ | 33.2 ± 11 | 155 ± 1.8e+03 | 0 ± 0 | 11.5 ± 0.63 | 7.04 ± 88 | 174 | 0.0340 |
| fleet_alert | light | random | 2 | hazard_exposure 0.0217 ± 0.028 ↓ | -6.48 ± 8.5 | 200 ± 1.3 | 0 ± 0 | 11.4 ± 0.056 | 0.15 ± 0.0028 | 0 | 0.0109 |
| fleet_alert | light | heuristic | 2 | hazard_exposure 0.000151 ± 0.0019 ↓ | 88.2 ± 0.032 | 300 ± 0 | 0 ± 0 | 11.4 ± 0.064 | 0.1 ± 0 | 0 | 0.0068 |
| coop_map | default | random | 2 | map_age_s 4.37 ± 0.54 ↓ | -128 ± 17 | 200 ± 1.3 | 0 ± 0 | 11.4 ± 0.056 | 0.15 ± 0.0028 | 0 | 0.0201 |
| coop_map | default | heuristic | 2 | map_age_s 2.87 ± 0.63 ↓ | -82.1 ± 19 | 300 ± 0 | 0 ± 0 | 11.4 ± 0.064 | 0.1 ± 0 | 0 | 0.0211 |
| coop_map | default | ppo_mlp | 2 | map_age_s 5.48 ± 2.5 ↓ | -162 ± 74 | 36.8 ± 1.6e+02 | 0 ± 0 | 11.4 ± 0.38 | 4.28 ± 9.7 | 46 | 0.0279 |
| coop_map | default | ppo_gru | 2 | map_age_s 4.32 ± 0.18 ↓ | -126 ± 5.3 | 300 ± 0 | 0 ± 0 | 11.4 ± 0.064 | 0.1 ± 0 | 137 | 0.0303 |
| coop_map | light | random | 2 | map_age_s 4.37 ± 0.54 ↓ | -128 ± 17 | 200 ± 1.3 | 0 ± 0 | 11.4 ± 0.056 | 0.15 ± 0.0028 | 0 | 0.0043 |
| coop_map | light | heuristic | 2 | map_age_s 2.87 ± 0.63 ↓ | -82.1 ± 19 | 300 ± 0 | 0 ± 0 | 11.4 ± 0.064 | 0.1 ± 0 | 0 | 0.0052 |
| coverage_nav | default | random | 2 | progress_m 0.0146 ± 0.31 ↑ | 0.0312 ± 0.32 | 200 ± 1.3 | 0 ± 0 | 11.4 ± 0.056 | 0.15 ± 0.0028 | 0 | 0.0314 |
| coverage_nav | default | heuristic | 2 | progress_m 85 ± 0.6 ↑ | 86.5 ± 0.52 | 300 ± 0 | 0 ± 0 | 11.4 ± 0.064 | 0.1 ± 0 | 0 | 0.0367 |
| coverage_nav | default | ppo_mlp | 2 | progress_m 61.4 ± 10 ↑ | 61.5 ± 10 | 300 ± 0.55 | 0 ± 0 | 11.4 ± 0.064 | 0.1 ± 0.00051 | 50 | 0.0377 |
| coverage_nav | default | ppo_gru | 2 | progress_m 42.2 ± 41 ↑ | 42.2 ± 41 | 291 ± 42 | 0 ± 0 | 11.4 ± 0.033 | 0.142 ± 0.018 | 121 | 0.0294 |
| coverage_nav | light | random | 2 | progress_m 0.0146 ± 0.31 ↑ | 0.0312 ± 0.32 | 200 ± 1.3 | 0 ± 0 | 11.4 ± 0.056 | 0.15 ± 0.0028 | 0 | 0.0065 |
| coverage_nav | light | heuristic | 2 | progress_m 85 ± 0.6 ↑ | 86.5 ± 0.52 | 300 ± 0 | 0 ± 0 | 11.4 ± 0.064 | 0.1 ± 0 | 0 | 0.0072 |
| edge_control | default | random | 2 | tracking_error_m 0.502 ± 0.027 ↓ | -75.2 ± 4.1 | 850 ± 4.1 | 0 ± 0 | 2.28 ± 0.0044 | 0.0358 ± 0.00025 | 0 | 0.2456 |
| edge_control | default | heuristic | 2 | tracking_error_m 0.354 ± 0.032 ↓ | -53.1 ± 4.9 | 1.5e+03 ± 0 | 0 ± 0 | 2.28 ± 0.01 | 0.02 ± 0 | 0 | 0.2612 |
| edge_control | default | ppo_mlp | 2 | tracking_error_m 0.393 ± 0.67 ↓ | -59 ± 1e+02 | 676 ± 3.2e+03 | 0 ± 0 | 2.28 ± 0.0099 | 0.045 ± 0.09 | 199 | 0.2365 |
| edge_control | default | ppo_gru | 2 | tracking_error_m 0.395 ± 0.34 ↓ | -59.2 ± 51 | 1.5e+03 ± 18 | 0 ± 0 | 2.28 ± 0.01 | 0.02 ± 0.00035 | 140 | 0.1071 |
| edge_control | light | random | 2 | tracking_error_m 0.502 ± 0.027 ↓ | -75.2 ± 4.1 | 850 ± 4.1 | 0 ± 0 | 2.28 ± 0.0044 | 0.0358 ± 0.00025 | 0 | 0.0898 |
| edge_control | light | heuristic | 2 | tracking_error_m 0.354 ± 0.032 ↓ | -53.1 ± 4.9 | 1.5e+03 ± 0 | 0 ± 0 | 2.28 ± 0.01 | 0.02 ± 0 | 0 | 0.1067 |

### L2-legacy on triton

| task | variant | baseline | n | task metric | return | deliveries | drops | delay_p95_ms | aoi_mean_s | train s | eval s/step |
|:---|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fleet_alert | default | random | 2 | hazard_exposure 0.0217 ± 0.028 ↓ | -6.48 ± 8.5 | 52.3 ± 19 | 133 ± 21 | 1.99e+03 ± 0.79 | 5.97 ± 1.5 | 0 | 0.0095 |
| fleet_alert | default | heuristic | 2 | hazard_exposure 0.00105 ± 0.0044 ↓ | 88.1 ± 0.36 | 27 ± 11 | 5.38 ± 2.9 | 1.53e+03 ± 1.3e+02 | 3.62 ± 1.8 | 0 | 0.0092 |
| fleet_alert | default | ppo_mlp | 2 | hazard_exposure 0.021 ± 0.012 ↓ | 49.9 ± 9.1 | 47.9 ± 2.5e+02 | 172 ± 98 | 1.95e+03 ± 83 | 8.29 ± 22 | 13 | 0.0101 |
| fleet_alert | default | ppo_gru | 2 | hazard_exposure 0.0215 ± 0.00035 ↓ | 38 ± 29 | 25.1 ± 1.3e+02 | 128 ± 5.3e+02 | 1.95e+03 ± 1.1e+02 | 9.22 ± 12 | 95 | 0.0233 |
| fleet_alert | light | random | 2 | hazard_exposure 0.0217 ± 0.028 ↓ | -6.48 ± 8.5 | 196 ± 2.9 | 3.4 ± 2.2 | 342 ± 39 | 0.255 ± 0.13 | 0 | 0.0157 |
| fleet_alert | light | heuristic | 2 | hazard_exposure 0.000151 ± 0.0019 ↓ | 88.2 ± 0.032 | 270 ± 8.7 | 0.0259 ± 0.031 | 127 ± 23 | 0.148 ± 0.0091 | 0 | 0.0127 |
| coop_map | default | random | 2 | map_age_s 6.24 ± 0.14 ↓ | -185 ± 4.2 | 109 ± 23 | 81.1 ± 23 | 1.93e+03 ± 1.9e+02 | 3.34 ± 1.6 | 0 | 0.0193 |
| coop_map | default | heuristic | 2 | map_age_s 4.28 ± 0.61 ↓ | -126 ± 19 | 82.6 ± 33 | 4.27 ± 2 | 614 ± 2.5e+02 | 3.36 ± 1.2 | 0 | 0.0224 |
| coop_map | default | ppo_mlp | 2 | map_age_s 9.69 ± 3.9 ↓ | -291 ± 1.2e+02 | 1.46 ± 19 | 0.154 ± 2 | 1.06e+03 | 14.9 ± 2.3 | 42 | 0.0293 |
| coop_map | default | ppo_gru | 2 | map_age_s 8.29 ± 0.65 ↓ | -248 ± 20 | 189 ± 2.4 | 81.3 ± 1.8 | 1.49e+03 ± 9.6e+02 | 2.67 ± 0.14 | 141 | 0.0177 |
| coop_map | light | random | 2 | map_age_s 4.42 ± 0.61 ↓ | -130 ± 19 | 197 ± 3.2 | 2.21 ± 2.1 | 238 ± 1.9e+02 | 0.214 ± 0.035 | 0 | 0.0150 |
| coop_map | light | heuristic | 2 | map_age_s 2.81 ± 0.26 ↓ | -80.6 ± 8 | 270 ± 5 | 0.0332 ± 0.062 | 126 ± 7.7 | 0.152 ± 0.0048 | 0 | 0.0159 |
| coverage_nav | default | random | 2 | progress_m -0.0163 ± 0.36 ↑ | -0.000703 ± 0.36 | 133 ± 17 | 59.5 ± 16 | 1.93e+03 ± 1.3e+02 | 2.76 ± 0.49 | 0 | 0.0238 |
| coverage_nav | default | heuristic | 2 | progress_m 58.9 ± 11 ↑ | 59.8 ± 11 | 123 ± 53 | 3.32 ± 1.1 | 656 ± 3.9e+02 | 2.84 ± 0.86 | 0 | 0.0261 |
| coverage_nav | default | ppo_mlp | 2 | progress_m 24.8 ± 1.7 ↑ | 24.8 ± 1.8 | 30.5 ± 26 | 0.0249 ± 0.16 | 102 ± 1.7e+02 | 13 ± 1.8 | 45 | 0.0363 |
| coverage_nav | default | ppo_gru | 2 | progress_m 20.3 ± 8.9 ↑ | 20.3 ± 8.9 | 60.5 ± 3.7e+02 | 2.78 ± 7.4 | 713 ± 2.7e+03 | 10.8 ± 20 | 141 | 0.0302 |
| coverage_nav | light | random | 2 | progress_m -0.0116 ± 0.29 ↑ | 0.005 ± 0.3 | 173 ± 7.7 | 24.6 ± 6.3 | 548 ± 9.8e+02 | 1.5 ± 0.9 | 0 | 0.0173 |
| coverage_nav | light | heuristic | 2 | progress_m 74.4 ± 1.4 ↑ | 75.6 ± 1.6 | 227 ± 8.7 | 2.16 ± 0.71 | 117 ± 60 | 1.58 ± 0.52 | 0 | 0.0221 |
| edge_control | default | random | 2 | tracking_error_m 7.47 ± 6.4 ↓ | -1.12e+03 ± 9.6e+02 | 585 ± 1.7e+02 | 230 ± 1.6e+02 | 451 ± 44 | 1.04 ± 1.8 | 0 | 0.1650 |
| edge_control | default | heuristic | 2 | tracking_error_m 3.37 ± 2.4 ↓ | -506 ± 3.6e+02 | 872 ± 2.6e+02 | 42 ± 37 | 180 ± 66 | 0.719 ± 1.6 | 0 | 0.2245 |
| edge_control | default | ppo_mlp | 2 | tracking_error_m 6.21 ± 33 ↓ | -931 ± 4.9e+03 | 576 ± 3.6e+03 | 219 ± 1.6e+03 | 376 ± 4.5e+02 | 0.969 ± 3.5 | 139 | 0.2245 |
| edge_control | default | ppo_gru | 2 | tracking_error_m 5.14 ± 27 ↓ | -771 ± 4.1e+03 | 398 ± 1.7e+03 | 124 ± 1.1e+03 | 351 ± 1.2e+03 | 1.05 ± 3.2 | 210 | 0.2176 |
| edge_control | light | random | 2 | tracking_error_m 0.693 ± 0.54 ↓ | -104 ± 81 | 839 ± 22 | 8.5 ± 22 | 69.9 ± 1e+02 | 0.0537 ± 0.032 | 0 | 0.1885 |
| edge_control | light | heuristic | 2 | tracking_error_m 0.433 ± 0.064 ↓ | -65 ± 9.5 | 1.33e+03 ± 55 | 1.53 ± 0.72 | 32.1 ± 11 | 0.0375 ± 0.0085 | 0 | 0.1569 |

**Timings.** One evaluation step of 64 × 16 robots took 0.004–0.05 s in the three 100 ms tasks and 0.09–0.26 s in EdgeControl, which runs five network steps and the edge stage per task step. PPO training took 13–199 s with the MLP and 95–210 s with the GRU, whose update replays the rollout sequences step by step. The whole script (96 runs and the calibration) took 6,245 s.
