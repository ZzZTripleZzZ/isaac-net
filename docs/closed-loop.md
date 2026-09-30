# Closed loop: one fixed controller over every network model

This page runs one scripted controller in a closed loop against each network model and records what the network delivers to it. The controller never changes and nothing is trained: the arms differ only in the network. The open-loop comparisons in [fidelity-vs-lena.md](fidelity-vs-lena.md) replay fixed traffic, while here the controller's sending depends on what was delivered before, so the traffic itself can differ between arms. The code is in `benchmarks/closedloop/` and every number below is in a CSV under `benchmarks/results/closedloop/`.

## Setup

**Task and controller.** The Fleet-Alert task of the [benchmark suite](benchmark-suite.md), reimplemented in `run_closedloop.py` with a smaller arena. Robots drive to random goals at 3 m/s. A hazard appears near a random robot, grows to a 15 m radius and lasts 10 s, and the fleet learns about it only when a camera frame captured within 50 m of it is delivered. The controller drives each robot to its goal and away from known hazards, and sends one 30 kB camera frame when the robot has no frame in flight and at least 5 control steps (0.5 s) have passed since its last send. A frame counts as lost if it has not arrived 2 s (20 control steps) after capture. The per-step reward is the progress to the goal in metres, minus 1 inside an active hazard, plus 2 per goal reached. The task return is its sum over the episode, averaged over robots.

**Geometry.** The example task's 150 m arena puts many robots below 5G-LENA's MCS-0 point, so in that arena the ns-3 reference delivers only 0–4% of what the engines deliver ([bridges.md](bridges.md)). This run therefore uses a 60 m × 60 m arena with the gNB at the corner and the radio of the 5G-LENA validation ([validation-5g-lena.md](validation-5g-lena.md)): path loss 40 + 35 log10(d) dB, 23 dBm UE power, thermal noise with a 7 dB noise figure (−101.44 dBm per 10-PRB subband), 6 dB per-robot shadowing drawn at reset, fading off, and UE power over the whole band. The validation's coverage rule (single-subband full-power SNR of at least 7 dB) is applied to the shadowing: a draw that would put the robot below 7 dB anywhere in the arena is redrawn. Every robot therefore stays inside the validated SNR range at every pose (at least 7 dB, 16.9 dB at the far corner without shadowing).

**Arms.** Every engine arm gets the same per-robot SNR each step, computed from the end-of-step pose. ns-3 gets that pose and the same shadowing and computes the same path loss itself.

| Arm | Model | Backend |
|:---|:---|:---|
| ideal | level `ORACLE`: every frame arrives in its capture step with zero delay and no loss | graph |
| L2 | NR engine with `lena_validation_v2()` | graph |
| L0 | i.i.d. lognormal delay and i.i.d. loss, fitted to the pooled L2 marginals (median 2.55 steps, log sigma 0.229, loss 0.0005; `l0_fit.csv`) | graph |
| L1 | fluid level, default parameters | graph |
| L2-legacy | the prototype slot-level engine | triton |
| ns-3 | ns-3.48 with 5G-LENA v5.1 through the lockstep bridge, one process per env over TCP | CPU |

**Scale.** 8 envs × 16 robots × 300 control steps (30 s), seeds 0 to 4, on the lab box (RTX 4090, 32 CPU cores, lightly loaded). One generator per seed draws every task random number with a fixed number of draws per step. Every arm therefore sees the same initial poses, goals, shadowing and hazard draws, and the runs differ only through what the network delivers. The L0 fit uses the L2 arm's delivered frames of all five seeds. The L2 arm counts a frame the engine loses after HARQ exhaustion (RLC UM) as in flight until its 2 s deadline, as the application sees it in ns-3.

## Results

Mean over the 5 seeds ± 95% t interval. Delay is capture to delivery over delivered frames. Delivery ratio is delivered / (delivered + lost), over frames resolved by the end of the episode. AoI is the age of each robot's newest delivered frame, sampled at every step. Wall time is the whole run of the arm (5 episodes, including engine construction and the task's own Python), and ms per step is per control step of all 8 envs (`per_seed.csv`, `summary.csv`).

| Arm | Delivery ratio | Delay p50 (ms) | Delay p95 (ms) | AoI mean (s) | AoI p95 (s) | Task return | Frames sent | ms per step | Wall (s) |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ideal | 1 | 0 | 0 | 0.350 ± 0 | 0.60 ± 0 | 82.9 ± 2.6 | 6400 ± 0 | 1.9 | 2.9 |
| L2 | 0.9995 ± 0.0003 | 257 ± 13 | 409 ± 51 | 0.583 ± 0.017 | 0.86 ± 0.07 | 82.9 ± 2.4 | 6393 ± 4 | 41.9 | 63.1 |
| L0 | 0.9996 ± 0.0002 | 255 ± 1 | 372 ± 4 | 0.560 ± 0.002 | 0.80 ± 0 | 82.6 ± 2.4 | 6394 ± 2 | 2.1 | 3.1 |
| L1 | 1 | 270 ± 0 | 289 ± 10 | 0.548 ± 0.002 | 0.80 ± 0 | 82.9 ± 2.4 | 6400 ± 0 | 3.4 | 5.1 |
| L2-legacy | 1 | 279 ± 2 | 320 ± 10 | 0.558 ± 0.004 | 0.80 ± 0 | 82.9 ± 2.4 | 6400 ± 0 | 2.0 | 3.0 |
| ns-3 | 0.9990 ± 0.0007 | 252 ± 12 | 399 ± 33 | 0.582 ± 0.011 | 0.90 ± 0 | 82.9 ± 2.5 | 6386 ± 10 | 19.1 | 29.8 |

Distances between the delay distributions, pooled over the 5 seeds (`distances.csv`). W1 is the Wasserstein-1 distance in ms and KS the Kolmogorov-Smirnov statistic. The last two rows split one arm into its even and odd seeds, as a reference for how far two samples of the same model lie apart when their trajectories differ:

| Pair | W1 (ms) | KS |
|:---|---:|---:|
| L0 vs L2 | 17.5 | 0.161 |
| L1 vs L2 | 49.5 | 0.575 |
| L2 vs ns-3 | 6.8 | 0.113 |
| L2-legacy vs ns-3 | 42.5 | 0.460 |
| L0 vs ns-3 | 13.4 | 0.103 |
| L1 vs ns-3 | 52.4 | 0.576 |
| L2 even vs odd seeds | 21.5 | 0.101 |
| ns-3 even vs odd seeds | 16.0 | 0.075 |

![Delay CDF of every arm](img/closed-loop-delay-cdf.png)

*Delay CDF over all delivered frames of the 5 seeds. The ideal arm, a step at 0 ms, is not drawn.*

**What the network delivers.** With the same controller, L2 and ns-3 give nearly the same delivery ratio (0.9995 and 0.9990), delay quantiles (p50 257 and 252 ms, p95 409 and 399 ms) and AoI (0.583 and 0.582 s). Their pooled delay distributions are 6.8 ms apart in W1 with a KS of 0.11, which is within the distance between two seed halves of either arm alone. The pooled pair shares seeds and therefore trajectories, which is why it can lie closer than the split-half reference. L1 and L2-legacy deliver every frame and put almost all delays in a narrow band (L1 270–289 ms at p50–p95, L2-legacy 279–320 ms), while L2 and ns-3 spread from about 150 to 600 ms. That is a KS of 0.46–0.58 against both. L0 matches the L2 median and delivery ratio by construction, but its lognormal shape has a shorter tail (p95 372 ms against 409 ms) and a KS of 0.16 against L2.

**What the controller does with it.** The load stays below the point where the send rule reacts to the network: every arm sends 6386–6400 of the 6400 frames the 0.5 s interval allows, because almost every frame arrives before the robot's next send. The AoI therefore follows the delay (0.35 s ideal, 0.55–0.58 s for the other arms). The task return is 82.6–82.9 in every arm, including the ideal link, and the robots spend about 0.1% of their steps inside a hazard. In this setup the controller's return does not depend on the network model, so the return column says nothing about the models. The delay, delivery and AoI columns are the comparison.

**Wall time.** Including the task's own Python (1.9 ms per step with the ideal link), L0, L1 and L2-legacy need 2.0–3.4 ms per control step, and L2 with the `lena_validation_v2` switches on the graph backend needs 42 ms. ns-3 needs 19 ms per step with 8 processes in parallel on a lightly loaded box, about 5 s per 300-step episode. At E = 8 the ns-3 reference is faster than the L2 graph backend. The GPU engines scale to thousands of envs per step ([performance.md](performance.md)), and ns-3 needs one CPU process per env.

## Caveats

- **One controller.** A single scripted controller with a fixed send rule. A controller that sends more often, or reacts to delay, would load the cell differently, and the distances above would change with it.
- **One geometry.** A single cell, a 60 m arena kept inside the validated SNR range, fading off. In the example task's 150 m arena the ns-3 reference loses most frames to the coverage gap, and none of the numbers here carry over to it.
- **Small scale and moderate load.** 8 × 16 robots, 5 seeds of 30 s, one message size, and a load at which almost every frame arrives before the next send. The loaded regime, where the send rule would react to queueing, is not covered, and the task return is not sensitive to the network at this load.
- **L0 is fitted to this run.** Its parameters come from the L2 arm of the same seeds, so its agreement with L2 on the median and the delivery ratio is by construction.

## Reproduce

On a machine with a CUDA GPU, the locally generated 5G-LENA tables (`ISAACLAB_NET_LENA_TABLES`) and the lockstep bridge built as in [bridges.md](bridges.md) (`NS3BRIDGE_ROOT`, `NS3_TOOLCHAIN_ENV`):

```bash
python benchmarks/closedloop/run_closedloop.py --out benchmarks/results/closedloop
python benchmarks/closedloop/plot_cdf.py benchmarks/results/closedloop docs/img/closed-loop-delay-cdf.png
```

`--arms` selects a subset (L0 needs L2 earlier in the list, since it is fitted to it). The result folder holds `per_seed.csv` (every metric per arm and seed), `summary.csv` (means and 95% intervals), `delay_cdf.csv` (pooled CDF on a log grid from 0.1 ms to 2 s), `distances.csv`, `l0_fit.csv`, `frames.csv.gz` (every delivered frame: arm, seed, env, robot, capture step, delay), `setup.json` and the run log.
