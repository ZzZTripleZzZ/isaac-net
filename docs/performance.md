# Performance

This page explains how the fast backends are shown to compute the same thing as the readable reference engine, and then reports what they cost on an otherwise idle GPU: per level and per backend, for the NR engine and the wrappers, with the pure-torch fleet task in the loop, inside Isaac Lab at up to 1,048,576 robots, on a dedicated data-center GPU, and against running ns-3 5G-LENA on CPUs.

## Measurement conditions

Every speed number on this page comes from one campaign on 2026-09-30 (scripts in `benchmarks/uncontended/` and `benchmarks/isaac/run_uncontended.ps1`, results in `benchmarks/results/uncontended/`). It replaces the earlier tables, which were all measured on a GPU that other jobs kept 90–100% busy (see [Contended vs uncontended](#contended-vs-uncontended) at the end of the page).

| | |
|:---|:---|
| GPU | NVIDIA GeForce RTX 4090, 24 GB, driver 617.14 (Windows 11 host) |
| CPU | Intel Core i9-13900KF (8 performance and 16 efficiency cores, 32 threads), Windows power plan "High performance" |
| Network-only and fleet-task runs | WSL2 Ubuntu 20.04, Python 3.11, torch 2.14.0+cu126, Triton 3.8.0 |
| Isaac Lab runs | native Windows, Isaac Sim 6.1.0.0, Isaac Lab 3.0, torch 2.12.0+cu130, `triton-windows` 3.8.0.post29 ([isaac-lab.md](isaac-lab.md)) |
| Dedicated second GPU | Hazel cluster, NVIDIA L40 allocated whole to the job, torch 2.14.0+cu126, Triton 3.8.0 |

**Idle GPU.** Only one benchmark process ran at a time. Before every case the driver waited until `nvidia-smi` showed under 5% utilization in three samples 0.5 s apart, and recorded what it saw. For all 516 network-only and fleet-task processes the GPU was at 0% utilization with 36 MiB of device memory in use before the case started, and for all 9 Isaac Lab runs it was at 0% with 36 MiB. Every CSV row carries these values (`idle_util_max`, `idle_mem_used_mib`) next to the utilization and device memory during the timed windows (`run_util_mean`, `run_mem_used_max_mib`). The CPU was not reserved: one single-threaded CPU job of another user ran on the box throughout, and the load average outside our runs was 1–2.

**Timing.** Sizes are written E × R: E environments with R robots each. One control step is 100 ms of simulated time, which is 40 uplink slots for the slot-level engines and 200 slots (40 UL and 160 DL data slots) for the NR engine at μ = 1. A network-only step is `submit` (plus `add_dl_frames` with a downlink) followed by `step` with dict outputs. Each case ran in a fresh process that built the engine, ran the warm-up (3 steps for eager references, 20 for the fast backends, which covers CUDA-graph capture and Triton compilation), and then timed 3 windows of about 2 s each (at least 2 steps for references and 5 otherwise, at most 100), synchronizing the device at the end of each window. Each case ran in 3 such processes on three passes through the whole grid. **A reported value is the median over the 3 processes of each process's median window.** The spread, given as ±x% where it reaches 10%, is (largest − smallest process value) / median. Inputs (sends, detection flags, SNR drifting around a draw in [0, 25] dB, 6 kB downlink messages, poses random-walking in the 150 m arena) come from a pool of 16 precomputed sets on the device, so input generation is not timed. Every robot sends w.p. 0.3 per step, a third of them 30 kB frames and the rest 4 kB, which saturates the uplink from R = 16 on. Peak memory is `torch.cuda.max_memory_allocated` above the baseline before the engine was built, CUDA-graph pools included. L05, L05Q, TR, GE, QA and NN use synthetic parameters of the shapes their fits produce, which set the cost but not the values that matter for fidelity.

**Host-bound rows vary between processes.** Within a process the three windows agree to a few percent. Between processes, rows whose cost is dominated by host work (the eager references at small and medium sizes, and the eager edge loop) differed by up to 2–3× in some cases at 1024 × 32 and 4096 × 16: two of the three passes had a slow period there, and the third did not. The GPU was idle throughout, so the cause is on the CPU side, possibly scheduling on the hybrid CPU's efficiency cores or the other user's CPU job. The median of three is robust to one slow process, and the spreads are printed so that no single number hides this. The fast backends at 4096 × 100, which are bound by GPU work, vary by under 5% between processes, and the fast backends at the smaller sizes, where a step takes about 1 ms and host overhead still matters, by up to 12%.

## Backends

Every prototype level (`L0`, `L0DR`, `L05`, `L05Q`, `L1`, `L2-legacy`) has an eager reference in `isaaclab_net/core/proto/netsim.py` and graph-safe fast versions in `netsim_fast.py`. The reference is slow because each control step of the slot-level engine runs 40 UL slots of about 25 small operations plus a 5-iteration PF loop, about 3,000 kernel launches, and its boolean-mask writes call `nonzero` and force a host sync each. At 16×16 the GPU work per kernel is a few microseconds, so the step is launch- and sync-bound.

The fast rewrite replaces every mask write with `torch.where` or a one-hot select, replaces `argsort` compaction with a stable cumsum scatter that yields the same permutation, computes the step index and slot times on the device, and keeps all state in persistent buffers updated in place so captured graph addresses stay valid across `reset`. Three backends are built on it:

- **`graph`** captures the rewritten eager operations in CUDA graphs. It is bitwise identical to the reference and is meant for scientific runs.
- **`triton`** runs all 40 slots of a control step in one hand-written Triton kernel, one program per env, with the robot × subband and robot × frame-slot state held in registers and Philox RNG inside the kernel. It matches the reference to rounding and is meant for scale.
- **`compile`** compiles the slot body with `torch.compile` (Inductor, fullgraph, static shapes) and captures the step in a CUDA graph. It is not suitable for training runs, for the reason below.

PF over subbands has no closed form, because the greedy choice for a subband depends on the remaining need and the power-headroom count after earlier subbands, so every backend keeps a sequential 5-step loop. The NR engine (`L2`) has its own `graph` and `triton` backends, described in the NR section below. The legacy multi-cell engine has only the `reference` backend.

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

## NR engine (`L2`): reference, `graph` and `triton`

The NR engine has three backends, all behind `make_engine("L2", ..., backend=...)` and the Isaac `NetModule`:

- **`reference`** (also `eager`) is the readable engine in `nr_engine.py` and `mac*.py`, run eagerly.
- **`graph`** (`nr_fast.NRGraphEngine`) captures the reference step itself in CUDA graphs, so it runs the same operations in the same order.
  - The control step enters as a device scalar, and slot times, HARQ timers and completion times (float64, as the reference computes them) are derived from it on the device.
  - Host decisions depend on time only through the TDD / SR / CQI schedule key and the fading step of the first slot. One graph per (key, first step, input kind) therefore serves every control step: 2–4 graphs in practice.
  - Every state tensor is a persistent buffer. The reference code reassigns its attributes, so after each captured step and each eager call (`submit`, `reset`, traffic injection) the new tensors are copied into the buffers and the attributes pointed back at them. Partial resets are exact and nothing is reallocated.
  - PHY constants are cached device tables, so there are no host-to-device copies or host syncs inside the step.
  - Statistics (`log_stats`) are exported by the graph and appended on the host after the replay.
  - Traffic models with sub-step arrivals, per-robot Doppler, several cells, handover and both directions are all covered.
- **`triton`** (`nr_fast.NRTritonEngine`, `nr_triton.nr_step_kernel`) runs every scheduled slot of a control step in one fused kernel, one program per env, with the robot × {HARQ process, frame, RBG} state in registers.
  - What the kernel covers:
    - fading
    - SR/BSR and grants
    - retransmission admission
    - PF / max C/I / round-robin RBG allocation with the power-headroom cap
    - EESM link adaptation over the MCS table
    - exact TBS and code-block tables, precomputed from `phy.py` for every (symbols, PRBs, MCS)
    - bilinear BLER lookup
    - HARQ chase / IR combining
    - OLLA
    - RLC-AM retransmission or RLC-UM loss
    - in-order delivery
    - DL CQI
    - the traffic arrival gate
    - per-robot Doppler
  - With a downlink the step runs as two kernels, DL then UL. Both replay the same fading trajectory from the same start state and keyed draws, so each carries only one link's state.
  - Prologue and epilogue (input SINR, power control, deadlines, compaction, outputs) are the reference's torch code, and the whole step is captured in CUDA graphs.
  - Single cell for now.

**Engine RNG.** Both fast backends need `NRConfig.rng = "engine"`, the default. Every draw of the NR engine is then a pure function of (seed, env, episode, control step of the env since its reset, slot, draw site, element), from the counter hash of `proto/rng.py`, keyed as `nr_rng.py` documents. The draws are:

- the fading innovations
- the TB decodes in both directions
- the initial fading at reset

A policy's use of the global torch RNG never changes the network. A partial reset re-seeds only its envs, and an env's draws do not depend on E. The reference and `graph` draw identical numbers. The kernel inlines the same hash: uniforms are bitwise equal and normals equal to float rounding. `rng="global"` restores the earlier stream (global torch RNG for stepping), and the frozen-engine regression test runs with it.

### Equivalence

The harness is `tests/nr_equiv.py`; `pytest -m gpu tests/test_nr_fast.py` runs short versions. The reference and the fast engine are built with the same config and seed and driven in lockstep:

- The workload cycles idle / medium / burst / medium-small traffic phases, with detection flags.
- SNR drifts in [−10, 40] dB. The input kind alternates every step (SNR, per-subband SNR with a DL SNR, poses through the engine radio, SNR with a per-subband DL SNR); with several cells, robots random-walk through the cells.
- Random partial resets hit about 5% of envs on 10% of steps, as index tensors or bool masks.

Compared at every step with exact equality: every output of `step()` and every state tensor of the engine (MAC, HARQ, queues, fading, association, interference estimates, counters, RNG counters, clocks). At the end, `collect()` statistics and `counters()` are compared.

Results. Every run is 300 control steps (30 s of simulated time), 25–42 partial resets, `log_stats` on, seed 7. Configs:

- `ul` is `NRConfig()`.
- `ul_dl` adds the downlink.
- `cells3` is `multicell(3, dl=True)`: three cells with UL and DL interference and handover.
- `ul_lena` is 16 HARQ processes with RLC UM, PDCP discard, wideband PF, 5G-LENA IR combining, no OLLA and no PHR cap.
- `ul_doppler` adds per-robot Doppler and round robin.
- `traffic` is periodic 600 B every 10 ms with 2 ms jitter, bursty 1.4 kB traffic on robot 0 and the policy, with the downlink.
- `traffic_c3` is the same traffic on three cells.

| Backend | Config | E × R | Mode | Result |
|:---|:---|:---|:---|:---|
| graph | `ul` | 64 × 16 | free-running | **bitwise identical** at every step, every output and state tensor; 59,720 frames delivered, statistics and counters identical |
| graph | `ul` | 256 × 32 | free-running | **bitwise identical**; 300,811 frames |
| graph | `ul_dl` | 64 × 16 | free-running | **bitwise identical**; 60,553 UL frames, 960,324 DL TBs |
| graph | `ul_dl` | 256 × 32 | free-running | **bitwise identical**; 307,728 UL frames, 9.87 M DL TBs |
| graph | `cells3` | 64 × 16 | free-running | **bitwise identical**; 77,095 frames |
| graph | `cells3` | 256 × 32 | free-running | **bitwise identical**; 373,995 frames |
| graph | `ul_lena` | 64 × 16 | free-running | **bitwise identical**; 11,566 frames |
| graph | `ul_doppler` | 64 × 16 | free-running | **bitwise identical**; 36,041 frames |
| graph | `traffic` | 64 × 16 | free-running | **bitwise identical**; 1,586,982 messages |
| graph | `traffic_c3` | 64 × 16 | free-running | **bitwise identical**; 1,873,046 messages |
| triton | `ul` | 64 × 16 | teacher-forced | 3 of 218,442 active robot-steps differ (1.4e-5); delivered 59,721 vs 59,720 |
| triton | `ul_dl` | 64 × 16 | teacher-forced | 3 of 217,861 (1.4e-5); TB decodes identical in count (3,205,361 UL, 960,324 DL) |
| triton | `ul` | 256 × 32 | teacher-forced | 35 of 1,910,974 (1.8e-5); delivered 300,815 vs 300,811 |
| triton | `traffic` | 64 × 16 | teacher-forced | 13 of 307,200 (4.2e-5); 1,586,977 vs 1,586,982 messages |
| triton | `ul` | 64 × 16 | free-running | delivered 59,710 vs 59,720, mean delay 7.6145 vs 7.6147 steps, decoded TBs 3,245,288 vs 3,246,344 (−0.03%) |
| triton | `ul_dl` | 64 × 16 | free-running | delivered 60,553 vs 60,553, mean delay 7.5502 vs 7.5512 steps, decoded UL TBs +0.003% |

These runs used two earlier states of the branch: the first eight graph rows before its rebase on the traffic and prototype-RNG merges, the traffic and triton rows after it. `pytest -m gpu tests/test_nr_fast.py` reran the short versions on the final code, including the energy wrapper's per-slot tap on `graph`.

**Why `triton` is equal only to rounding.** The kernel's reductions (EESM log-sum-exp, the PRB-weighted means) and its fused multiply-adds round differently from ATen. The fading state and effective SINRs therefore differ in the last bits. A discrete decision flips when a value sits on a threshold: an MCS boundary, a TB decode draw, or a PF tie. In free running a flip changes that robot's future, so trajectories decorrelate while the aggregates agree. From an identical state 1–4 in 100,000 active robot-steps differ. That is about 5–10× the legacy kernel's rate, because the NR step has more thresholds: 29 MCS, the EESM of every candidate MCS, and the HARQ combining. In free running, delivered frames, mean delay and decoded TBs agree to 0.03% or better over 300 steps.

## Network cost per level and backend

Milliseconds per control step, network only, on the idle RTX 4090 (conditions [above](#measurement-conditions)). ±x% marks a spread between the three processes of 10% or more. The last column is the peak memory at the largest size, and `benchmarks/results/uncontended/net_rtx4090.csv` has the peak memory, the three per-process values and the GPU state of every case.

| Level | Backend | 256 × 16 | 1024 × 32 | 4096 × 16 | 4096 × 100 | Peak MiB, 4096 × 100 |
|:---|:---|---:|---:|---:|---:|---:|
| `L0` | reference | 1.23 ±11% | 1.27 | 1.31 ±203% | 4.75 | 707 |
| `L0` | graph | 0.36 | 0.95 | 1.59 | 10.6 | 1,695 |
| `L0DR` | reference | 1.36 | 1.26 | 1.57 ±155% | 4.72 | 707 |
| `L0DR` | graph | 0.36 | 0.93 | 1.59 | 10.6 | 1,695 |
| `L05` | reference | 1.57 ±12% | 1.65 | 1.59 ±221% | 5.08 | 707 |
| `L05` | graph | 0.40 | 1.00 | 1.66 | 10.9 | 1,695 |
| `L05Q` | reference | 1.52 | 1.65 | 1.51 ±201% | 4.98 | 707 |
| `L05Q` | graph | 0.40 | 0.94 | 1.67 | 10.9 | 1,695 |
| `L1` | reference | 15.1 | 15.5 | 22.9 ±56% | 107 | 708 |
| `L1` | graph | 1.94 | 9.51 | 17.3 | 108 | 1,714 |
| `L1` | triton | 0.36 | 0.95 ±12% | 1.75 | 11.1 | 1,662 |
| `L2-legacy` | reference | 94.0 | 94.3 ±110% | 101 ±86% | 191 | 1,291 |
| `L2-legacy` | graph | 9.71 | 18.2 | 27.3 | 135 | 2,076 |
| `L2-legacy` | triton | 0.46 | 1.19 ±10% | 2.10 | 17.7 | 1,718 |
| `TR` | reference | 2.00 | 1.87 | 2.34 ±65% | 10.9 | 1,176 |
| `TR` | graph | 0.40 | 0.94 | 1.63 | 10.6 | 1,687 |
| `GE` | reference | 2.03 | 2.17 ±20% | 2.36 ±114% | 11.1 | 1,177 |
| `GE` | graph | 0.40 | 0.95 | 1.68 | 10.8 | 1,687 |
| `QA` | reference | 2.07 | 2.09 | 2.98 ±80% | 15.4 | 1,180 |
| `QA` | graph | 0.42 | 1.30 | 2.32 | 15.0 | 1,691 |
| `NN` | reference | 3.15 | 3.00 | 3.23 ±150% | 15.6 | 1,189 |
| `NN` | graph | 0.52 | 1.32 | 2.31 | 15.2 | 1,703 |
| `WIFI` | reference | 300 | 311 | 305 ±129% | 923 | 1,208 |
| `WIFI` | graph | 34.2 | 89.0 | 147 | 792 | 1,736 |

Reading the table:

- **`triton` is the scale path for the slot-level levels.** At 4096 × 100 (409,600 robots) `L2-legacy` on `triton` takes 17.7 ms per 100 ms step, 11× faster than its eager reference and 8× faster than `graph`, and `L1` takes 11.1 ms. At 256 × 16 both kernels take under 0.5 ms, about 200× (`L2-legacy`) and 40× (`L1`) faster than the reference.
- **`graph` removes launch overhead, not work.** For `L2-legacy` it is 9.7× faster than the reference at 256 × 16 and 1.4× at 4096 × 100, where it replays about 100 unfused kernels per slot that each touch whole `[E, R, F]` or `[E, R, S]` tensors. For `L1` it gains nothing at 4096 × 100.
- **The delay levels are cheap at every size.** `L0` to `L05Q` cost 0.4–1.7 ms on `graph` up to 4096 × 16. At 4096 × 100 the fixed-shape `graph` versions (10.6–10.9 ms) are about 2× slower than the eager references (4.7–5.1 ms), because the fixed-shape path scatters 9 `[E, R, F]` fields during compaction and copies outputs into static buffers, while the eager reference only touches new and finished frames. Use the reference backend for these levels at large R.
- **The surrogates cost about as much as the delay levels.** `TR`, `GE`, `QA` and `NN` run the same graph-safe operations on both backends, so `graph` gains 1.6–6× at small sizes from fewer launches and nothing at 4096 × 100 (10.6–15.2 ms).
- **`WIFI` is the most expensive level without a fused kernel.** Its step solves a mean-field contention model in each of 100 sub-steps of 1 ms, and costs 34 ms on `graph` at 256 × 16 and 0.8 s at 4096 × 100, which is close to its reference there.
- The eager references of `L0` to `NN` are host-bound at 1.2–3.2 ms per step up to 1024 × 32, whatever the size. The `L2-legacy` reference is host-bound at about 94 ms per step up to 4096 × 16 (about 3,000 kernel launches and a host sync per mask write).

**Memory.** At 4096 × 100 the eager references of the delay levels peak at 707 MiB, the surrogates at about 1.15 GiB, and every `graph` and `triton` version at 1.6–2.0 GiB, including the CUDA-graph pools. Memory grows linearly with E × R (for example `L2-legacy` `triton`: 17 MiB at 256 × 16, 138 at 1024 × 32, 273 at 4096 × 16).

**Kernel limits of the legacy `triton` backend.** The kernel pads robots to a power of two (at least 16), so R above about 256 would need a tiled PF reduction. Its int32 RNG offsets limit it to about E × R ≈ 4.9M robots, and at R = 100 it is limited by register pressure. Each fast instance captures its graphs for its own (E, R), in about 1–3 s for `graph` and `triton`, so a different E needs a new instance, and state attributes must never be reassigned because the graphs hold their addresses. The `compile` backend was not re-measured: it is unfit for training (see above).

### NR engine (`L2`)

Config `ul` is `NRConfig()` (μ = 1, 20 MHz, 13 RBGs, 16 HARQ processes, EESM); `ul_dl` adds the downlink; `c3` is `multicell(3)` (three cells with interference and handover) and `c3_dl` is `multicell(3, dl=True)`. The fused kernel is single-cell, so `c3` and `c3_dl` have no `triton` row. Same conditions and format as above.

| Config | Backend | 256 × 16 | 1024 × 32 | 4096 × 16 | 4096 × 100 | Peak MiB, 4096 × 100 |
|:---|:---|---:|---:|---:|---:|---:|
| `ul` | reference | 330 | 328 ±153% | 388 ±96% | 2,310 | 3,604 |
| `ul` | graph | 50.2 | 138 | 264 | 2,228 | 6,111 |
| `ul` | triton | 1.68 | 5.58 ±11% | 13.3 | 67.8 | 3,726 |
| `ul_dl` | reference | 1,540 | 1,557 ±153% | 1,865 ±96% | 11,467 | 4,383 |
| `ul_dl` | graph | 240 | 685 | 1,315 | 11,084 | 8,306 |
| `ul_dl` | triton | 5.66 | 22.3 | 57.0 | 295 | 5,676 |
| `c3` | reference | 327 | 329 ±125% | 317 ±70% | 1,712 | 2,368 |
| `c3` | graph | 45.3 | 102 | 182 | 1,620 | 4,094 |
| `c3_dl` | reference | 1,450 | 1,429 ±158% | 1,439 ±23% | 8,318 | 2,699 |
| `c3_dl` | graph | 206 | 488 | 884 | 7,948 | 4,903 |

Reading the table:

- **`triton` is the scale path.** At 4096 × 100 the uplink takes 67.8 ms per 100 ms step, 34× faster than the reference and 33× faster than `graph`. With the downlink it takes 295 ms, 39× faster than the reference. At 256 × 16 the uplink takes 1.7 ms and UL + DL 5.7 ms.
- **`graph` removes the launch overhead but not the work.** At 256 × 16 it is 6.6× faster than the reference (50 against 330 ms uplink, 240 against 1,540 ms with the downlink). From about 50,000 robots on, both are bound by memory traffic: the reference step is tens of thousands of small kernels per control step (about 27,000 at 64 × 16 in the profiler count of the NR multi-cell work), each reading and writing whole `[E, R, P]`, `[E, R, F]` or `[E, R, S]` tensors. At 4096 × 100 the two cost the same (2.2–2.3 s uplink, 11.1–11.5 s with the downlink), and `graph` needs 1.7–1.9× the memory of the reference for its pool. Use `graph` for bitwise-reproducible runs at moderate scale and `triton` for training at scale.
- **The downlink** multiplies the cost by 4.7–5.0× on the reference and 3.4–4.3× on `triton`, from its 160 DL data slots per step against 40 UL slots.
- **Three cells cost no more than one.** `c3` costs 0.7–1.0× the single-cell `ul` on the reference and on `graph`, likely because each cell's scheduler handles only the robots it serves.
- **Against the legacy engine**, the NR uplink costs 3.5–3.8× the `L2-legacy` reference up to 4096 × 16 and 3.7–6.3× the legacy `triton` kernel at every size. The NR step has 29 MCS, the EESM of every candidate MCS, 16 HARQ processes and the RLC.
- **Kernel limits.**
  - One program per env, with robots padded to a power of two, so R above about 128 would need a tiled PF reduction.
  - At R = 64–100 the kernel still spills registers, especially with the downlink (Triton reports 8–344 spilled values per thread after the rewrite that runs link adaptation one MCS at a time; before it, 184–1,372).
  - Per-slot work grows with RBGs × MCS (EESM for every MCS of every new TB).

### Edge and energy wrappers

`EdgeLoop` (default `EdgeConfig()`: one FIFO server per env, 10 ms deterministic service, instant return) and `EnergyLoop` (default `EnergyConfig()`, airtime approximation) over `L2-legacy` on `triton`, eager or with the wrapper's own step captured in a CUDA graph (`graph=True`). Same conditions and format as above.

| Wrapper | Wrapper step | 256 × 16 | 1024 × 32 | 4096 × 16 | 4096 × 100 | Peak MiB, 4096 × 100 |
|:---|:---|---:|---:|---:|---:|---:|
| none (`L2-legacy` `triton`) | | 0.46 | 1.19 ±10% | 2.10 | 17.7 | 1,718 |
| EdgeLoop | eager | 33.4 | 42.8 ±119% | 36.1 | 107 | 2,057 |
| EdgeLoop | graph | 8.99 | 18.7 | 17.7 | 391 | 2,520 |
| EnergyLoop | eager | 0.78 | 1.52 ±37% | 2.43 | 18.4 | 1,723 |
| EnergyLoop | graph | 0.63 | 1.37 | 2.29 | 18.4 | 1,723 |

`EnergyLoop` adds 0.2–0.7 ms per step at every size, which is under 5% at 4096 × 100. `EdgeLoop` is expensive: its exact event loop runs up to 2R + 2 × servers + 8 iterations of vectorized tensor work per step (210 at R = 100). Eager, it stops as soon as every env has run out of events (checked every 4 iterations), and costs 33–43 ms up to 4096 × 16 and 107 ms at 4096 × 100. In a CUDA graph it cannot stop early and always replays the full event budget, which is 2–4× cheaper at small sizes (9–19 ms) but 3.7× more expensive at 4096 × 100 (391 ms). At large R, lower `max_events_per_step` (an env that runs out continues exactly at the next step), or use the eager loop.

### Legacy multi-cell engine

The legacy multi-cell engine (NetSlotMC, `L2-legacy` with several cells, `reference` backend only) was profiled earlier with the PyTorch profiler, whose kernel counts and GPU time do not depend on the load of the GPU:

| E × R | Single cell, kernels / GPU ms | Multi-cell C = 1 | C = 3 | C = 7 |
|:---|:---|:---|:---|:---|
| 64 × 16 | 9235 / 12.8 | 9242 / 13.0 | 11127 / 16.0 | 11127 / 16.3 |
| 256 × 32 | 9316 / 14.0 | 9323 / 14.3 | 11208 / 17.2 | 11208 / 17.6 |

Multi-cell costs about 20% more kernel launches and 20–27% more GPU time, independent of the number of cells and nearly independent of E × R. At about 280 kernels per UL slot the engine is launch-bound, with only 13–18 ms of GPU work per control step.

## Full environment step with the fleet task

The pure-torch fleet task (`isaaclab_net/examples/fleet_task.py`: E envs of R robots, goals, hazards, detection frames over the uplink, random actions that send w.p. 0.15) with the network off (a stub that never delivers) or with an engine behind its `add_frames` / `step` calls. Milliseconds per full env step, same conditions and format as above (`benchmarks/uncontended/bench_env.py`, results in `env_rtx4090.csv`):

| Network | Backend | 256 × 16 | 1024 × 32 | 4096 × 16 | 4096 × 100 | Peak MiB, 4096 × 100 |
|:---|:---|---:|---:|---:|---:|---:|
| off | - | 1.66 ±12% | 1.71 ±11% | 1.65 | 1.72 | 101 |
| L0 | reference | 2.91 | 2.94 | 2.88 | 5.55 | 620 |
| L0 | graph | 2.02 ±11% | 2.12 ±12% | 2.92 ±10% | 11.8 | 1,732 |
| L2-legacy | triton | 1.82 | 2.35 | 3.41 | 19.1 | 1,757 |
| L2 | triton | 3.04 ±12% | 6.99 | 14.9 | 69.2 | 3,759 |

The env alone costs about 1.7 ms per step at every size, because it is host-bound (a host sync per step for hazard spawning, plus small tensor operations). A fast network overlaps with that host work: `L2-legacy` on `triton` adds 0.2–0.6 ms up to 1024 × 32, 1.8 ms at 4096 × 16 and 17 ms at 4096 × 100, which is about its network-only cost there. The NR `triton` uplink adds 1.4 ms at 256 × 16 and 67 ms at 4096 × 100, so at 409,600 robots the NR network is 97% of the step. The eager `L0` reference adds about 1.2 ms up to 4096 × 16, and its `graph` version is again slower at 4096 × 100 (11.8 against 5.6 ms).

## Isaac Lab scale

The Isaac Lab fleet env (`NetFleetEnv`, Isaac Sim 6.1 PhysX, dt = 1/50 s, decimation 5, so one env step is one 100 ms control step) on the Windows install of [isaac-lab.md](isaac-lab.md), run as one-shot SYSTEM scheduled tasks (`benchmarks/isaac/run_uncontended.ps1`). Random actions send on about two thirds of robot-steps, half of them large frames, so the uplink is saturated. Each configuration ran in its own process: 10 warm-up steps, then 3 windows of 50 steps, and the table gives the median window. Network-only is 3 windows of 20 isolated `submit` + `step` calls on the live module at the end of the run, median window. Before each run the GPU was at 0% utilization with 36 MiB in use and no other compute process (Windows `nvidia-smi`), and no WSL job ran. Results are in `benchmarks/results/uncontended/isaac_scale_rtx4090.csv`.

| E × R | Robots | Network | Control steps/s | Robot-steps/s | On / off | Network share of step | Network only (ms) | Device memory, max (GiB) | Startup (s) |
|:---|---:|:---|---:|---:|---:|---:|---:|---:|---:|
| 2,048 × 128 | 262,144 | off | 5.19 | 1,360,034 |  |  |  | 4.3 | 259 |
| 2,048 × 128 | 262,144 | `L2-legacy` `triton` | 4.08 | 1,070,764 | 0.79 | 21% | 10.9 | 5.5 | 272 |
| 2,048 × 128 | 262,144 | `L2` `triton` | 4.15 | 1,087,601 | 0.80 | 20% | 36.8 | 6.8 | 259 |
| 4,096 × 128 | 524,288 | off | 2.58 | 1,351,859 |  |  |  | 6.0 | 538 |
| 4,096 × 128 | 524,288 | `L2-legacy` `triton` | 2.39 | 1,252,064 | 0.93 | 7% | 22.9 | 8.3 | 510 |
| 4,096 × 128 | 524,288 | `L2` `triton` | 2.44 | 1,281,282 | 0.95 | 5% | 72.3 | 10.9 | 548 |
| 8,192 × 128 | 1,048,576 | off | 1.50 | 1,574,483 |  |  |  | 9.4 | 1084 |
| 8,192 × 128 | 1,048,576 | `L2-legacy` `triton` | 1.51 | 1,586,506 | 1.01 | 0% | 45.2 | 13.9 | 1067 |
| 8,192 × 128 | 1,048,576 | `L2` `triton` | 1.32 | 1,384,203 | 0.88 | 12% | 145.6 | 19.0 | 1038 |

The windows of a run agree to within 10%, except the first window of the network-off run at 2,048 × 128 (3.65 against 5.19–5.72 steps/s). "Network share of step" is (step time on − step time off) / step time on, and is shown as 0% where the network-on run was as fast as the network-off run.

- **One million robots.** 8,192 × 128 = 1,048,576 robots run at 1.51 control steps per second with `L2-legacy` on `triton` (1.59 M robot-steps/s) and at 1.32 with the NR uplink on `triton` (1.38 M robot-steps/s), against 1.50 with the network off. The device held at most 13.9 GiB (`L2-legacy`) and 19.0 GiB (NR) of its 24 GiB, all of it this process.
- **The Isaac step is bound by host work.** Device-wide GPU utilization during the timed windows was 22–35%, and the network-off step takes 193, 388 and 666 ms at 262k, 524k and 1M robots. The GPU work of the network therefore overlaps with PhysX and Python host work. `L2-legacy` on `triton` costs 11, 23 and 45 ms per step in isolation but adds 7% of the step at 524k robots and nothing measurable at 1M. The NR uplink costs 37, 72 and 146 ms per step in isolation, 3.2× the legacy kernel, and adds 5–20% of the step.
- At 2,048 × 128 both networks add about 50 ms per step (20–21% of the step), more than their isolated cost of 11–37 ms. The Isaac side of the network (the radio's SNR over 4 interpolated poses, the observation features and the module's bookkeeping) adds host work of its own, and the noisy network-off run at this size possibly also overstates the gap.
- **Startup** grows by about 1.0 ms per robot (PhysX cloning of R distinct objects): 259–272 s at 262k, 510–548 s at 524k and 1,038–1,084 s at 1M robots.

Taking 2 control steps per second as the bar for interactive use (a 24-step PPO rollout in at most 12 s), every configuration up to 524k robots meets it with either network, and 1M robots runs at 1.3–1.5 steps/s.

## Dedicated L40 (Hazel)

To check the RTX 4090 numbers on different hardware, the `L2-legacy` fast backends and the NR engine ran on one NVIDIA L40 (48 GB, driver 595.58.03) of the Hazel cluster, allocated whole to a Slurm job (`benchmarks/uncontended/hazel_nr.sbatch`, results in `net_hazel_l40.csv`). The script, workload and timing are the same as above, with one process per case (3 windows), so there is no between-process spread. The GPU was at 0% utilization with 3 MiB in use and no compute process before every case, and every case's windows agree to within 5%. Milliseconds per control step:

| Level / config | Backend | 256 × 16 | 1024 × 32 | 4096 × 16 | 4096 × 100 | RTX 4090, 4096 × 100 |
|:---|:---|---:|---:|---:|---:|---:|
| `L2-legacy` | graph | 10.9 | 19.2 | 28.4 | 142 | 135 |
| `L2-legacy` | triton | 0.39 | 1.08 | 2.14 | 19.1 | 17.7 |
| `L2` `ul` | reference | 342 | 334 | 361 | 2,846 | 2,310 |
| `L2` `ul` | graph | 55.4 | 145 | 288 | 2,834 | 2,228 |
| `L2` `ul` | triton | 1.50 | 5.68 | 13.6 | 72.6 | 67.8 |
| `L2` `ul_dl` | reference | 1,541 | 1,670 | 1,589 | 14,196 | 11,467 |
| `L2` `ul_dl` | graph | 266 | 722 | 1,444 | 14,125 | 11,084 |
| `L2` `ul_dl` | triton | 5.71 | 23.8 | 58.3 | 309 | 295 |

The L40 is 5–27% slower than the RTX 4090 at 4096 × 100 (5–8% for the `triton` kernels and legacy `graph`, 23–27% for the NR reference and `graph`) and about as fast on the small rows. Its eager NR references (334–361 ms up to 4096 × 16) match the fast processes on the RTX 4090 (327–388 ms), which supports the reading that the slow processes there were host-side. An earlier dedicated L40 run of `benchmarks/bench.py` ([isaac-lab-linux.md](isaac-lab-linux.md)) gave `L2-legacy` `triton` 0.53 / 0.71 / 2.21 / 2.36 ms at 256 × 16 / 1024 × 16 / 4096 × 16 / 1024 × 64 and the eager legacy reference 119–122 ms at 16 × 16 and 256 × 16, consistent with these rows.

## Cost of CPU co-simulation with ns-3 5G-LENA

**This comparison is between different models.** The GPU numbers are for the slot-level abstraction (`L2-legacy`) and the NR engine (`L2`), and the CPU numbers are for ns-3.48 with 5G-LENA v5.1, a packet-level simulator with far more detail. The NR engine is the closer of the two (its replay of the 5G-LENA sweep matches the drop rate to a mean absolute difference of 0.029), but the model differences listed in [validation-5g-lena.md](validation-5g-lena.md) remain. This is a cost-of-fidelity comparison, not a like-for-like speed-up. **The ns-3 side was not rerun:** its numbers are those of [bridges.md](bridges.md), measured on the same 32-core box while it was shared with other jobs (load average 30–35 during the pool sweep), and the bridge page estimates that this load can inflate them 2–3×. Only the GPU side below is new.

The process pool runs one single-threaded ns-3 process per env. With the UE-to-UE spectrum filter (which removes an O(R²) cost without changing any per-frame outcome) and p = 0.1, one worker spends 26 ms of CPU per 100 ms step at R = 16 and 183 ms at R = 100; the fit t ≈ 7.5 + 1.14 R + 0.0063 R² + 0.38 R p ms extrapolates to 261 ms at R = 128, beyond the measured range. With 12 workers the pool reached 129 env-steps/s at R = 16 and 25 env-steps/s at R = 100, a parallel efficiency of 0.30–0.39.

| Comparison | ns-3 5G-LENA (CPU) | GPU, idle RTX 4090 | Ratio |
|:---|:---|:---|:---|
| Per robot-step at R = 16 | 1.6 CPU-ms (26 ms per env-step of 16 robots, one worker) | 256 × 16 network only: `L2-legacy` `triton` 0.11 µs, NR `triton` 0.41 µs | about 14,000× / 4,000× |
| 256 × 16, 2.46 M env-steps, fleet task in the loop | pool at W = 12, 129 env-steps/s: about 5.3 h | fleet task + `L2-legacy` `triton` 1.82 ms per step: 17.5 s; + NR `triton` 3.04 ms: 29 s | about 1,090× / 650× |
| Same budget, network only | same | 0.46 ms per step: 4.4 s; NR 1.68 ms: 16 s | about 4,300× / 1,180× |
| 4096 × 100 (409,600 robots), network only | 750 CPU-s per control step (4096 × 183 ms); 23.4 s per step on all 32 cores at perfect efficiency | `L2-legacy` `triton` 17.7 ms, NR `triton` 67.8 ms | about 1,320× / 350× |
| Cores to keep pace at 4096 × 100 | at efficiency 1: about 42,000 (`L2-legacy`) / 11,000 (NR); at the measured 0.39: about 109,000 / 28,000; for real time (100 ms per step): 7,500 / 19,000 at 0.39; about 190–310 GB of RAM (46–77 MB per worker) | one GPU, 1.7 / 3.6 GiB | |
| 4096 × 128 (524,288 robots), Isaac Lab in the loop | 1,071 CPU-s per control step (4096 × 261 ms, R extrapolated from 100 to 128); 33.5 s per step on 32 cores at efficiency 1, plus the Isaac step without network (0.39 s) | Isaac Lab + `L2-legacy` `triton` 0.42 s per step; + NR `triton` 0.41 s | about 81× / 83× |
| 4096 × 128, network only | 33.5 s per step on 32 cores | `L2-legacy` `triton` 22.9 ms, NR `triton` 72.3 ms (measured inside Isaac) | about 1,460× / 460× |

The Isaac-in-the-loop ratio (about 80×) is the one that matters for training time on this box: with the network on the GPU, the Isaac step itself dominates, so a faster network no longer shortens training much. The network-only ratio (350–1,460× at 409k–524k robots, depending on the engine) compares the two network simulators alone. Both assume ns-3 at perfect parallel efficiency on all 32 cores, which the pool never reached (0.30–0.39 at 12 workers), and ns-3 CPU times that were measured under load; the ns-3 side could be up to 2–3× faster on a quiet box and is up to 2.6× slower at the measured efficiency.

The extrapolated core counts assume scaling linear in envs and ignore the per-episode ns-3 restart. ns-3 is therefore a validation and evaluation tool at small E, not a training backend.

## Contended vs uncontended

Before this campaign every timing on this page was taken while other jobs kept the RTX 4090 90–100% busy (2–24 GB of its memory held by other processes). The old tables are kept in git history and in `benchmarks/nr/results/bench_nr_fast.csv`. Contention inflated the launch- and sync-bound rows most. Isaac Lab throughput was not inflated at all, because the Isaac step is bound by host work and the GPU was mostly idle within it:

| Row | Contended (earlier tables) | Idle GPU (this page) | Inflation |
|:---|---:|---:|---:|
| `L0` reference, 256 × 16 | 10.3 ms | 1.23 ms | 8.4× |
| `L2-legacy` reference, 256 × 16 | 769 ms | 94.0 ms | 8.2× |
| `L2-legacy` `graph`, 256 × 16 | 32.2 ms | 9.71 ms | 3.3× |
| `L2-legacy` `triton`, 256 × 16 | 2.09 ms | 0.46 ms | 4.5× |
| `L2-legacy` `triton`, 4096 × 100 | 70.8 ms | 17.7 ms | 4.0× |
| `L1` `triton`, 4096 × 100 | 47.5 ms | 11.1 ms | 4.3× |
| NR `ul` reference, 256 × 16 | 577 ms | 330 ms | 1.7× |
| NR `ul` `graph`, 256 × 16 | 97.8 ms | 50.2 ms | 1.9× |
| NR `ul` `triton`, 256 × 16 | 7.11 ms | 1.68 ms | 4.2× |
| NR `ul` `graph`, 4096 × 100 | 5,085 ms | 2,228 ms | 2.3× |
| NR `ul` `triton`, 4096 × 100 | 120 ms | 67.8 ms | 1.8× |
| NR `ul_dl` `triton`, 4096 × 100 | 503 ms | 295 ms | 1.7× |
| Isaac Lab, 4096 × 128, network off | 2.90 steps/s | 2.58 steps/s | none |
| Isaac Lab, 4096 × 128, `L2-legacy` `triton` | 2.58 steps/s | 2.39 steps/s | none |
| Isaac Lab, 4096 × 128, `L2-legacy` network only | 52 ms | 22.9 ms | 2.3× |

Contention inflated the network-only times 1.7–8.4×, most for the host-bound eager references of the prototype levels and for the small fast rows. Speed-up ratios measured in the same contended run were therefore off by up to about 2× in either direction (for example `L2-legacy` reference / `triton` at 256 × 16 was 370× contended and is 205× on the idle GPU). The earlier Isaac Lab throughput was within the ±40% run-to-run noise it reported, and it was not a lower bound as the earlier text assumed, while its network costs of 9–20% of the step overstated the idle-GPU cost of the legacy kernel (0–7% from 524k robots up).
