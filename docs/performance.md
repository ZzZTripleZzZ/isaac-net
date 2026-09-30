# Performance

This page explains how the fast backends are shown to compute the same thing as the readable reference engine, and then reports what they cost: per level and per backend, for the NR and multi-cell engines, inside Isaac Lab at up to 1,048,576 robots, and against running ns-3 5G-LENA on CPUs.

## Read this first: every number was measured under contention

No quiet GPU window came up during this work. Every GPU timing on this page was taken on one RTX 4090 (driver 617.14) in the shared lab box while other jobs kept it **95–99% busy** as reported by `nvidia-smi` for the whole device, with 2–17 GB of device memory held by other processes. Every ns-3 timing was taken on the same 32-core box at load averages between 10 and 68. Each table below repeats its own contention context.

Contention inflates launch- and sync-bound code the most. The original eager slot-level engine ran at 77 ms per control step at 16×16 on a quiet GPU in an earlier measurement, and at about 1,400 ms under the contention of these runs, so absolute times here are pessimistic by up to about 20× at small sizes. Speed-up ratios are therefore optimistic at small E, where the reference suffers most, and closer to fair at large E, where every version is bandwidth-bound. Ratios measured in the same run are the meaningful quantities. An uncontended re-benchmark of the headline rows is open item 2 in [STATUS.md](STATUS.md).

Sizes are written E × R: E environments with R robots each. One control step is 100 ms of simulated time, which for the slot-level engines is 40 uplink slots.

## Backends

Every prototype level (`L0`, `L0DR`, `L05`, `L05Q`, `L1`, `L2-legacy`) has an eager reference in `isaaclab_net/core/proto/netsim.py` and graph-safe fast versions in `netsim_fast.py`. The reference is slow because each control step of the slot-level engine runs 40 UL slots of about 25 small operations plus a 5-iteration PF loop, about 3,000 kernel launches, and its boolean-mask writes call `nonzero` and force a host sync each. At 16×16 the GPU work per kernel is a few microseconds, so the step is launch- and sync-bound.

The fast rewrite replaces every mask write with `torch.where` or a one-hot select, replaces `argsort` compaction with a stable cumsum scatter that yields the same permutation, computes the step index and slot times on the device, and keeps all state in persistent buffers updated in place so captured graph addresses stay valid across `reset`. Three backends are built on it:

- **`graph`** captures the rewritten eager operations in CUDA graphs. It is bitwise identical to the reference and is meant for scientific runs.
- **`triton`** runs all 40 slots of a control step in one hand-written Triton kernel, one program per env, with the robot × subband and robot × frame-slot state held in registers and Philox RNG inside the kernel. It matches the reference to rounding and is meant for scale.
- **`compile`** compiles the slot body with `torch.compile` (Inductor, fullgraph, static shapes) and captures the step in a CUDA graph. It is not suitable for training runs, for the reason below.

PF over subbands has no closed form, because the greedy choice for a subband depends on the remaining need and the power-headroom count after earlier subbands, so every backend keeps a sequential 5-step loop. The NR engine (`L2`, one or several cells) and the legacy multi-cell engine have only the `reference` backend so far.

## Equivalence methodology

**Same random draws.** Before each step the test generates that step's noise from a seeded generator: the fading innovations `[40, E, R, S, 2]` and the TB decode draws `[40, E, R]`. The unmodified reference reads them through a context manager that temporarily replaces `torch.randn_like` and `torch.rand_like` with readers that pop one slot's slice and assert that exactly 40 of each are consumed. The fast engine (`inject=True`) receives the same tensors through `set_noise`, and the initial fading is copied from the reference.

**Workload.** A synthetic driver cycles every 100 steps through idle, medium, all-large burst and medium-small phases. It exercises SR, grants, HARQ exhaustion and the RLC wait, buffer overflow, the 2 s timeout, tag changes and SNR drift in [−10, 40] dB.

**What is compared, every step, with exact equality** (`torch.equal`, infinities included): the step outputs, per-frame finish times, all 9 per-frame fields and all MAC state (BSR, SR time, PF average, OLLA offset, wait, HARQ count, fading), and at the end the full logged statistics.

**Two modes.** In free-running mode each engine evolves its own state for 300 steps. In teacher-forced mode the reference state is copied into the fast engine before every step, which counts per-step disagreements from an identical state and separates per-step errors from chaotic divergence. The mismatch unit is an active robot-step: a robot with a frame queued or finishing where any finish time or the newest-delivered output differs.

| Backend | E × R | Mode | Result |
|:---|:---|:---|:---|
| graph | 16 × 16 | free-running | **bitwise identical** every step; 16,807 frames, 0 finish-time mismatches, statistics identical |
| graph | 256 × 16 | free-running | **bitwise identical**; 274,680 frames |
| graph | 64 × 100 | free-running | **bitwise identical**; 96,442 delivered, 625,458 timeouts |
| triton | 16 × 16 | teacher-forced | 0 of 53,409 robot-steps mismatch |
| triton | 256 × 16 | teacher-forced | 2 of 849,113 robot-steps (2.4e-6); aggregate delivered, timeouts and mean delay identical |
| triton | 64 × 100 | teacher-forced | 5 of 1,631,514 robot-steps (3.1e-6); delivered 96,443 vs 96,442 |
| triton | 16 × 16 | free-running | drift from step 0 by float rounding (fading ≤ 1e-6); delivered 16,810 vs 16,807, mean delay 8.8978 vs 8.8976 steps |
| compile | 16 × 16 | teacher-forced | 6,229 of 53,409 robot-steps (11.7%) mismatch |
| compile | 16 × 16 | free-running | delivered 16,661 vs 16,807, mean delay 9.16 vs 8.90 steps |

**Why `triton` is equal only to rounding.** The kernel uses the same CUDA libdevice math as ATen but a different reduction and scan order and fused multiply-adds. State then differs by float rounding, at most about 1e-6 in the fading state and up to about 0.1 byte in remaining bytes within one step. A discrete decision (TB decode, PF winner or frame completion) flips in about 3 per million robot-steps. In free-running mode a flip changes that robot's future, so trajectories decorrelate while the aggregate statistics stay equal to 4–5 significant digits.

**Why `compile` is unfit.** From identical inputs, compiled and eager slot bodies agree on every discrete variable but differ in the last bits of the fading state (from FMA contraction of the AR(1) update) and of the remaining bytes (a different cumsum order). Across 40 slots these differences hit the reference's `≤ 1e-3`-byte "frame finished" threshold: at byte counts of about 30,000 the float32 ulp is about 0.002, so a fully served frame can leave a 0.002-byte residue and complete one slot later. That flips finish times in about 12% of active robot-steps and biases the free-running mean delay by +3% at 16×16. The Triton kernel computes FIFO service as `max(cum − b, 0) − max(cum − rem − b, 0)`, which cancels exactly for fully served frames and avoids most of these flips. Compile would need bitwise-controlled Inductor numerics to be usable, and it is also slower than `triton` at every size.

**After the engine API change** (partial resets, per-env clocks, dict outputs, merge `fbfcb44`), the same scripts were rerun at every level. The new reference is bitwise equal to the frozen original (`tests/scripts/netsim_v0.py`) at all 6 levels in both the legacy and new API, `graph` is bitwise equal to the reference at every level with 59–79 env resets per run, and `test_reset.py` passes 14/14 for reference, `graph` and `triton`: envs that are never reset stay bitwise equal to a run without resets, reset rows equal their initial values, no buffer is reallocated, and a reset does not perturb the other envs' random stream. After the package move, `test_equiv.py --rung all --backend graph` was bitwise identical at every level (12/12 runs, with and without partial resets). These checks run by hand from `tests/scripts/`, and `pytest -m gpu` runs the smaller `graph` and `triton` checks as tests.

## Benchmarks per level and backend

Network step time in ms per control step (`submit` + `step` with dict outputs), final engine-API code. **Contention: RTX 4090 at 98–99% utilization from other jobs in every window; absolute numbers are pessimistic.** The reference was timed over 5 steps at the small sizes and 2 steps at 4096 × 100, and the fast backends over 50 (20 at scale).

| Level | 16×16 ref | graph | triton | 256×16 ref | graph | triton | 4096×100 ref | graph | triton |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `L0` | 10.1 | 2.05 | | 10.3 | 2.06 | | 19.5 | 45.1 | |
| `L0DR` | 10.0 | 2.02 | | 10.0 | 2.09 | | 19.2 | 44.9 | |
| `L05` | 13.9 | 2.03 | | 13.5 | 2.11 | | 23.3 | 46.2 | |
| `L05Q` | 13.8 | 1.61 | | 14.3 | 2.13 | | 21.1 | 46.5 | |
| `L1` | 107 | 6.80 | 1.44 | 118 | 7.32 | 2.09 | 458 | 474 | 47.5 |
| `L2-legacy` | 765 | 30.5 | 2.16 | 769 | 32.2 | 2.09 | 1,102 | 566 | 70.8 |

Every fast backend has a floor of about 2 ms at small sizes on this GPU, set by contention and per-call host overhead. At 4096 × 100 the `graph` versions of the delay levels are about 2× slower than the eager reference, because the fixed-shape path scatters 9 `[E, R, F]` fields during compaction and copies outputs into static buffers, while the eager reference only touches new and finished frames. `graph` gives no speed-up for `L1` at scale either, and `triton` (about 10×) is the scale path there. Peak memory at 4096 × 100 was about 713 MiB for the reference and about 1.7 GiB for `graph` and `triton`. Resetting 1% of envs every step at 256 × 16 costs 0.3–2 ms extra (for example `L2-legacy` 34.1 / 3.8 ms for `graph` / `triton`), and the dict outputs add 0.4–0.8 ms per step over the legacy tuple outputs at 256 × 100.

**The legacy slot-level engine at larger sizes**, measured before the engine-API change (network only, `add_frames` + `step`, torch 2.14 / CUDA 12.6). **Contention: 95–99% utilization in every timing window, with the GPU shared by about 8 other jobs.** Fast versions were timed over 20–50 steps after warm-up, the original over 3–10. Peak MiB is `max_memory_allocated` above the baseline, CUDA-graph pool included.

| E | R | orig ms | graph ms | compile ms | triton ms | orig / graph | orig / triton | peak MiB orig / graph / compile / triton |
|---:|---:|---:|---:|---:|---:|---:|---:|:---|
| 16 | 16 | 1370 | 48.4 | 5.0 | 2.14 | 28× | 640× | 0 / 1 / 1 / 1 |
| 256 | 16 | 1478 | 54.6 | 5.8 | 2.02 | 27× | 730× | 6 / 15 / 13 / 12 |
| 256 | 100 | 1279 | 87.3 | 6.7 | 4.29 | 15× | 298× | 38 / 93 / 131 / 143 |
| 1024 | 16 | 1253 | 80.9 | 5.1 | 1.47 | 15× | 854× | 24 / 60 / 106 / 116 |
| 1024 | 100 | 1474 | 243.3 | 15.2 | 16.42 | 6× | 90× | 154 / 375 / 322 / 357 |
| 4096 | 16 | 1405 | 168.9 | 9.6 | 6.41 | 8× | 219× | 100 / 239 / 210 / 248 |
| 4096 | 64 | 1369 | 499.0 | 47.4 | 25.20 | 3× | 54× | 392 / 955 / 820 / 800 |
| 4096 | 100 | 1586 | 817.7 | 87.3 | 60.15 | 2× | 26× | 617 / 1509 / 1291 / 1220 |

At 4096 × 100 (409,600 robots) `triton` needs 60 ms per 100 ms simulated step, 26× faster than the original under the same contention, while `graph` loses most of its advantage (2×) because it replays about 100 unfused kernels per slot that each touch `[E, R, F]` or `[E, R, S]` tensors. A less contended 256 × 16 run (24–59% utilization during the timing window) gave `graph` 13.4 ms, `compile` 1.01 ms and `triton` 0.43 ms, which shows how much of the absolute cost is contention. With the example fleet task in the loop (env + network, random actions, 256 × 16, 96–99% utilization), a full step took 607 ms with the original, 35.7 ms with `graph`, 11.9 ms with `compile` and 10.9 ms with `triton`, so with a fast network the env's own small operations dominate the step.

The `triton` kernel pads robots to a power of two (at least 16), so R above about 256 would need a tiled PF reduction. Its int32 RNG offsets limit it to about E × R ≈ 4.9M robots, and at R = 100 it is limited by register pressure. Each fast instance captures its graphs for its own (E, R), in about 1–3 s for `graph` and `triton` and 5–120 s of Inductor compilation for `compile`, so a different E needs a new instance, and state attributes must never be reassigned because the graphs hold their addresses.

## NR engine and multi-cell engine

The NR engine (`L2`) runs on the `reference` backend only. The table compares it with the legacy reference in ms per control step at R = 16 (eager PyTorch). **Contention: 98% utilization and 10–16 GB of other jobs' memory, so absolute numbers are inflated by about 20× and only the ratios within a column are meaningful.** The first attempt hit CUDA OOM from other tenants.

| Engine | E = 16 | 64 | 128 | 256 |
|:---|---:|---:|---:|---:|
| legacy reference | 1004 | 894 | 844 | 866 |
| NR `netslot_compat` (1 HARQ process, 5 subbands) | 2812 | 2611 | 2717 | 2502 |
| NR `lena_like` (16 HARQ, EESM, wideband PF, UL) | 2488 | 2592 | 2544 | 2614 |
| NR default μ = 1, 20 MHz (13 RBGs, EESM) | 3137 | 3188 | 3348 | 3375 |
| NR `lena_like` with downlink | 11751 | 12063 | 11714 | 11309 |

The uplink-only NR engine costs about 2.6–3.9× the legacy reference and is flat in E up to 256, so it is launch-bound. 13 RBGs instead of 5 cost about 25% more, and the downlink adds about 4.5× from its 160 D and S slots per step. The step has no host syncs and fixed shapes, so capturing a control step in a CUDA graph is the natural next step (open item 1).

The legacy multi-cell engine was timed with the PyTorch profiler, whose kernel counts and GPU time do not depend on contention. Wall-clock time under 98% GPU utilization and CPU load averages of 10–31 was 660–830 ms per step for the single-cell engine and 740–1080 ms for the multi-cell variants.

| E × R | Single cell, kernels / GPU ms | Multi-cell C = 1 | C = 3 | C = 7 |
|:---|:---|:---|:---|:---|
| 64 × 16 | 9235 / 12.8 | 9242 / 13.0 | 11127 / 16.0 | 11127 / 16.3 |
| 256 × 32 | 9316 / 14.0 | 9323 / 14.3 | 11208 / 17.2 | 11208 / 17.6 |

Multi-cell costs about 20% more kernel launches and 20–27% more GPU time, independent of the number of cells and nearly independent of E × R. At about 280 kernels per UL slot the engine is launch-bound, with only 13–18 ms of GPU work per control step.

## Isaac Lab scale

The full scaling tables, the conditions and the adapter fixes are in [isaac-lab.md](isaac-lab.md). In summary, with Isaac Sim 6.1 PhysX in the loop and the legacy slot-level engine on the `triton` backend, the largest configuration run was 8,192 envs × 128 robots = 1,048,576 robots, at 1.27 control steps per second (1.33 M robot-steps/s) against 1.40 with the network off, within 16.4 GB of device memory. From 131k to 1M robots the network costs 9–20% of the step, and its isolated cost is 16–75 ms per step. **Contention: the GPU was at 96–99% utilization before and during every run, and repeated identical configurations varied by about ±40% over the evening, so only the network-on vs network-off ratios within a row group and the network-only times are meaningful. Absolute throughput is a lower bound.** These runs used the demo's copy of the slot-level engine, which is the code `L2-legacy` now wraps.

| E × R | Robots | Network off (steps/s) | `triton` (steps/s) | `triton` / off | `graph` (steps/s) |
|:---|---:|---:|---:|---:|---:|
| 4096 × 32 | 131,072 | 3.29 | 2.68 | 0.81 | 1.34 |
| 2048 × 128 | 262,144 | 4.38 | 3.96 | 0.90 | 1.78 |
| 4096 × 128 | 524,288 | 2.90 | 2.58 | 0.89 | 1.23 |
| 8192 × 128 | 1,048,576 | 1.40 | 1.27 | 0.91 | not run |

## Cost of CPU co-simulation with ns-3 5G-LENA

**This comparison is between different models.** The GPU numbers are for the slot-level abstraction (NetSlot, `L2-legacy`), and the CPU numbers are for ns-3.48 with 5G-LENA v5.1, a packet-level simulator with far more detail. The model differences are listed in [validation-5g-lena.md](validation-5g-lena.md). This is a cost-of-fidelity comparison, not a like-for-like speed-up. **Contention: both sides were measured on the shared lab box, the GPU at 95–99% utilization and the CPU at load averages of 30–35 (CPU 63–94% busy) during the filtered pool sweep, with at most 12 workers allowed.**

The process pool runs one single-threaded ns-3 process per env (details in [bridges.md](bridges.md)). Its per-env cost per 100 ms step at W = 1 and p = 0.1, with the UE-to-UE spectrum filter that removes an O(R²) cost without changing any per-frame outcome, is 15 ms at R = 8, 26 ms at 16, 75 ms at 32, 107 ms at 64 and 183 ms at 100, close to linear (about R^0.91). Without the filter it is 25, 52, 181, 566 and 1,527 ms (about R^1.57). With 12 workers the shared box reaches about 2,000–2,900 robot-steps/s at every R, at a parallel efficiency of 0.3–0.5.

| Comparison | ns-3 5G-LENA (CPU) | Slot-level abstraction (GPU) | Ratio |
|:---|:---|:---|:---|
| Per robot-step, 16 × 16 | 3–7 CPU-ms per robot-step (lockstep bridge, sum over processes) | `triton` 2.1 ms per step for 256 robots, about 0.008 ms per robot-step | ns-3 about 400–800× slower per robot |
| Per robot-step against the eager original | same | original 77 ms per step at 16 × 16 on a quiet GPU, about 0.3 ms per robot-step | ns-3 about 10–20× slower |
| 256 × 16, whole step | pool at W = 12: about 125 env-steps/s | example task + `triton` network, 10.9 ms per step: about 23,500 env-steps/s | about 185× |
| 256 × 16, 2.46 M env-steps | about 5.4 h of network simulation | about 105 s | about 185× |
| 4096 × 100 at 60 ms per step (the `triton` time) | about 12,500 fully efficient cores (about 32,000 at the measured efficiency of 0.39) and 308 GB of RAM with the filter; about 104,000 cores without it | one GPU | |
| 4096 × 100 at real time (100 ms per step) | about 7,500 cores with the filter, about 62,600 without | one GPU | |

The extrapolated core counts assume perfectly linear scaling in envs and ignore the per-episode ns-3 restart. The pool's efficiency would rise on dedicated cores, but even at efficiency 1 about 12,500 cores are needed to match the `triton` step at 4096 × 100. ns-3 is therefore a validation and evaluation tool at small E, not a training backend.
