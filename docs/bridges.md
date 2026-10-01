# ns-3 co-simulation bridges

The bridges connect an environment to ns-3.48 with 5G-LENA v5.1, so the same task can run against a packet-level reference instead of the GPU engine. They exist for **validation and evaluation only** and are never needed for training: one ns-3 process simulates one cell of one environment on one CPU core, which is orders of magnitude slower than the GPU engine (see [performance.md](performance.md)). Using them requires a local ns-3 + 5G-LENA build, which is not part of this package, and binaries linked against ns-3 are GPL-covered and are not distributed.

The bridges were built and tested on 2026-09-29 on the lab box (WSL) and moved into the package in `651a93e`. In the package they are ported to the per-env-clock engine base with full resets only, and `Ns3Net` (lockstep) and `PoolNet` (pool) were smoke-tested through the package against the existing lab builds. The validation scripts in `tests/bridges/` were import-rewired but not rerun after the port.

| Variant | Package path | C++ program | What it is for |
|:---|:---|:---|:---|
| Lockstep over TCP, Unix socket or ns3-ai shared memory | `isaac_net/bridges/ns3_lockstep/` | `bridges/ns3/lockstep/netslot-bridge.cc` | closed-loop co-simulation of a few envs, including the Isaac `NetModule` API and a stdlib-only Windows client |
| Process pool | `isaac_net/bridges/ns3_pool/` | `bridges/ns3/pool/netslot-bridge.cc` | one ns-3 process per env, the CPU co-simulation baseline |
| Offline trace-driven replay | `isaac_net/bridges/ns3_offline/` | the pool program in file mode | run ns-3 after the fact on a recorded rollout; also holds `lena_replay.py`, the NR engine's replay of the 5G-LENA sweep |
| Real-time mode | pool program with `--io=rt:PORT` | same | wall-clock emulation experiments |

Both C++ programs are the reference scenario `netslot-ref.cc` described in [validation-5g-lena.md](validation-5g-lena.md) plus a step interface. They are our own code written against the ns-3 APIs, with no ns-3 or 5G-LENA source copied. Each compiles against an existing ns-3 build with that build's exact flags in 15 s to 1 min, with no reconfigure.

## How a lockstep step works

Each 100 ms control step, the client sends the robot poses (constant, or as a 4-sub-step waypoint), optional per-UE shadowing and the frames to send. The server applies them, queues the frames and runs the simulator to 1 ns before the next 100 ms boundary. Frames are sent by the UE's own periodic tick event, at exactly the event position the standalone scenario uses, which is why the bridge reproduces the standalone run exactly. The reply carries the completed frames with their completion times, per-UE UL SINR, MCS, TB, retransmission, corrupt-TB and lost-TB counts and good bytes (from the gNB PHY trace, de-duplicated), DL RSRP, RLC buffer bytes from the UE SR/BSR trace, and the server's wall time inside `Run()`. A RESET rebuilds the world in-process with a new run number, which required resetting `RngSeedManager::ResetNextStreamIndex()` and the IPv4 address pool.

Several envs run either as one process per env stepped in parallel (`procs`: send to all, then receive all), as one process holding E cells that each have their own band and spectrum channel and are placed 20 km apart (`single`), or as a mix of the two. On the Python side, `Ns3Net` is a drop-in for the example task's network, reading robot positions from the env it serves (or from `bind_env(env)`) and by default deriving per-UE shadowing from the env's own SNR, so ns-3's large-scale SNR equals the engine radio's. `Ns3NetModule` implements the Isaac `NetModule` API, with the raw ns-3 statistics in `out.ns3`.

**ns3-ai.** The latest ns3-ai (main `b8c9858`, v1.2.0 plus fixes, 2025-01-23) works with ns-3.48 without source changes. The build issues were all environmental: Boost headers need `-isystem` and `Boost_DIR` with the conda cross-compiler, the system protoc and libprotobuf versions disagree (a conda protobuf fixes it), the glibc-2.17 sysroot needs `-lrt`, the example pybind module needs `LD_PRELOAD` of libns3-core, and examples other than a-plus-b fail at configure unless their lte or wifi modules are enabled. Two design limits remain. ns3-ai keeps its segment in a function-local static, so one template instance owns one segment per process lifetime, and the bridge instantiates it 64 times, which caps a Python process at **64 shared-memory channels over its lifetime**. Its semaphores are pure spin-waits, so both sides burn a core while waiting.

## Correctness

The standalone scenario ran first, and its frame schedule was then replayed through each bridge with the same seed. Per-frame completion times were compared at the 6 significant digits the scenario prints.

| Case | Transport | Frames complete (standalone / bridge) | Exact per-frame match |
|:---|:---|:---|:---|
| 8 UEs, 20–90 m, 4 kB, p = 0.5, 20 s | lockstep TCP | 214 / 214 | 214 / 214 |
| same, run 3 | lockstep Unix | 215 / 215 | 215 / 215 |
| same, run 1 | lockstep ns3-ai | 214 / 214 | 214 / 214 |
| random drop, 6 dB shadowing, coverage rule | lockstep TCP | 297 / 297 | 297 / 297 |
| 30 kB, p = 0.25, 10–45 m | lockstep TCP | 156 / 156 | 156 / 156 |
| 16 UEs, RLC AM | lockstep TCP | 622 / 616 | 616 / 616; the 6 missing frames have delays of 9.5–10 s and are cut by the bridge's 10 s frame-registry purge (the application deadline is 2 s) |
| 8 UEs, 20–90 m, 4 kB, p = 0.5, UM | pool: file mode with and without FlowMonitor, TCP, Python pool at W = 1 | 123 of 423 sent | identical in all four paths |
| 16 UEs, random drop, coverage rule, run 3 | same four paths | 158 of 782 sent | identical |
| 8 UEs, 20–50 m, 30 kB, p = 0.3, **AM** | same four paths | 83 of 243 sent | identical |

For the pool, "identical" means the same set of completed frames with every delay within the print precision (at most 4e-6 s, or 3e-5 s for times above 10 s). In the single-process multi-cell mode, env 0 delivers the same frame set whether env 1 is saturated or idle, with times differing by at most 3.6 µs through the shared core link, and a RESET with the same run number reproduces an episode bit for bit. Single mode draws different RNG streams than `procs`, so their sample paths differ.

**The UE-to-UE filter.** The largest ns-3 cost in this single-cell TDD uplink is UE-to-UE signal propagation in the spectrum channel, which is O(R²) and never affects decoding. A `SpectrumTransmitFilter` (`--ueUeFilter=1`) removes it. Per-frame delays with the filter on and off are identical to the standalone scenario, and a step becomes 1.6× faster at R = 8 and 8.4× faster at R = 100. The pool uses the filter by default, and without it the cost comparison with the GPU engine would partly be a strawman.

**Closed loop.** In a closed-loop run of the example fleet task under a fixed policy and a random policy (E up to 4, R up to 16), the TCP, Unix and shared-memory transports give identical trajectories. In the task's 150 m arena with the gNB at the corner, however, 5G-LENA delivers only 0–4% of what the slot-level engine delivers, with a mean UL SINR of about −19 dB at the end of an episode. This is the coverage gap described in [validation-5g-lena.md](validation-5g-lena.md) (no OLLA, no power-headroom scheduling, UEs below the MCS-0 point), not a bridge bug: robots parked 15–40 m from the gNB get frames through at a p50 of about 35–40 ms, and every frame ns-3 completes is matched. Closed-loop comparisons in that arena are therefore nearly empty, and the offline-replay study below uses a 60 m arena.

## The process pool

`PoolNet` runs one ns-3 process per env with R UEs in one cell and one TCP connection each, and every step scatters one line to each worker and gathers one line back, so the pool has a barrier per step. The env's own SNR, including its correlated shadowing, is passed as a per-UE path-loss override (`loss = 113 − snr_db`), so 5G-LENA sees exactly the engine's single-subband full-power SNR while fading, AMC, HARQ, SR/BSR, PF, RLC and UDP stay 5G-LENA's own. Positions only feed the fading geometry. ns-3 cannot rewind, so `reset()` restarts all workers, which takes 0.15–0.6 s with the filter, once per episode, and **partial reset of individual envs is not supported**.

**Scaling.** Contention context: 32-core box shared with about 9 other agents, load average 30–35 and CPU 63–94% busy during the filtered sweep, workers capped at 12. Throughput in env-steps/s with parallel efficiency in parentheses, filtered pool, p = 0.1:

| R \ workers | 1 | 2 | 4 | 8 | 12 |
|:---|:---|:---|:---|:---|:---|
| 8 | 58.0 (1.00) | 97.1 (0.84) | 127.8 (0.55) | 195.1 (0.42) | 278.5 (0.40) |
| 16 | 35.4 (1.00) | 56.2 (0.79) | 88.8 (0.63) | 113.7 (0.40) | 129.1 (0.30) |
| 32 | 12.7 (1.00) | 25.9 (1.01) | 48.0 (0.94) | 53.7 (0.53) | 72.9 (0.48) |
| 64 | 9.2 (1.00) | 15.6 (0.85) | 22.8 (0.62) | 38.3 (0.52) | 45.2 (0.41) |
| 100 | 5.4 (1.00) | 8.3 (0.77) | 15.1 (0.70) | 20.7 (0.48) | 25.2 (0.39) |

At 12 workers the slowest worker takes on average about 50 ms longer than the mean worker, the protocol costs 2–6 ms per step and the environment on CPU 2–5 ms, so the barrier and the box contention, not the protocol, cause the efficiency loss. The unfiltered pool reaches 270, 102, 31 and 9.1 env-steps/s at R = 8, 16, 32 and 64 with 12 workers.

**Cost model.** Per-worker compute per 100 ms step at one worker and p = 0.1 is 15, 26, 75, 107 and 183 ms at R = 8, 16, 32, 64 and 100 with the filter, and 25, 52, 181, 566 and 1,527 ms without it, with resident memory of 46–77 MB and 48–425 MB respectively. The unfiltered R = 100 point measured 3.9–4.3 s at a load average of 63, so contention alone can inflate these numbers 2–3×. A model linear in R times load does not fit (R² about −0.06 filtered), because offered load barely matters: the cost is per UE per slot (control, SR/BSR, CQI, fading and the scheduler), not per byte. The fits are t ≈ 7.5 + 1.14 R + 0.0063 R² + 0.38 R p ms with the filter (R² = 0.97, about R^0.91) and t ≈ 14.4 + 0.12 R + 0.147 R² + 1.49 R p ms without it (R² = 0.995, about R^1.57), where the R² term is the UE-pair channel. The extrapolation to 4096 × 100 and the comparison with the GPU engine are in [performance.md](performance.md).

## Offline trace-driven replay and its limitation

The offline path records a rollout (positions, SNR and the frames that entered the network buffer), runs ns-3 on each env's recorded trace afterwards in file mode (at most 12 at a time), and replays the outcomes by (env, robot, frame id) through `ReplayNet`. On a miss it falls back to the nearest recorded frame of the same robot and size class, then of the env, and otherwise treats the frame as lost.

**The outcomes are valid only for the policy that was recorded.** Replaying policy A's own outcomes under A reproduces the closed loop exactly (0 error on every metric, 2,700–3,470 delivered frames per seed identical). For any other policy the replay is open loop: it cannot see that the policy's own sending changes the queues and therefore the delays. The size of this error was measured with 8 envs × 16 robots × 300 steps and 3 seeds in a 60 m arena, with policy A random (send none, small or large with probability 0.7, 0.25 and 0.05) and policy B a greedy heuristic that heads to its goal and sends whenever its own queue is empty. The truth is each policy run in closed loop with the pool, and the error is replay minus truth for the same policy, as the mean over seeds with [min, max]:

| Replay | KS of delay CDF | Delivery | p50 delay | p95 delay | Mean delay | Mean AoI | Sends per robot-step |
|:---|:---|:---|:---|:---|:---|:---|:---|
| A on A's own outcomes (sanity) | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| A on A, another ns-3 run (noise floor) | 0.04 [0.03, 0.06] | −0.1 pp | +13% | +4% | +7% | −2% | 0 |
| **B on A's outcomes** | **0.38** [0.37, 0.40] | **+13 pp** [+9, +20] | **+63%** [+58, +73] | **+174%** [+119, +239] | **+107%** [+78, +140] | **+33%** [+14, +47] | **+39%** [+19, +68] |
| A on B's outcomes (reverse) | 0.34 [0.29, 0.36] | −0.1 pp | −47% | −81% | −69% | −39% | 0 |

In closed loop, B delivers 64% of its frames with a p50 of 29 ms and a p95 of 106 ms, while A delivers 26% with a p50 of 88 ms and a p95 of 938 ms. When B is evaluated on A's outcomes, 76% of its lookups miss A's recorded frames and fall back, and its replayed delays come from A's heavier offered load, so the replay overstates B's delays about 2× and misstates B's own behavior (+39% sends), because B's sending depends on a queue the replay does not track. The error is about 10× the run-to-run noise floor. In the default 150 m arena the same comparison is dominated by the coverage gap (delivery 1.5% and 5%), and the noise floor there is already KS 0.12. **Use offline replay only to evaluate the policy that produced the trace.**

## Real time

In real-time mode (`--io=rt:PORT`) the worker runs ns-3's `RealtimeSimulatorImpl` in BestEffort mode, a reader thread injects send and move commands, deliveries stream back as they happen, and the lag of simulated time behind wall time is sampled every 10 ms. The test client places R robots in a 60 m square, random-walking 0.3 m per step, and streams frames at a wall-clock 10 Hz. A run passes if, after a 1.5 s warm-up, both the p99 and the final lag are below 20 ms over 20 s. The load average was 16–24.

| R | 1 | 2 | 4 | 8 | 12 | 16 | 24 | 32 | 48 | 64 |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| run 1, p99 lag (ms) | 0.8 ✓ | 0.3 ✓ | 0.3 ✓ | 0.6 ✓ | 1.6 ✓ | 913 | 1,683 | 796 | 511 | 2,551 (drifting) |
| run 2, p99 lag (ms) | 27 | 2.6 ✓ | 91 | 451 | 1,636 | 1,406 | 905 | 939 | 929 | 3,869 (drifting) |

On this shared box `RealtimeSimulatorImpl` holds real time reliably for only about 2 robots, up to 12 in favorable runs, and never at 16 or more. Further R = 12 runs gave p99 lags of 390–1,373 ms, and even with no traffic R = 12 reached 156 ms (67 ms with fading off). The lag builds in ramps without jumps above 50 ms and then recovers, which points to ns-3 compute bursts amplified by the real-time scheduler's per-event synchronization rather than OS stalls. When the worker is on time, wall-clock delay equals simulated delay plus 0.5–1 ms.

For a live demo the better option is the step-synchronous pool worker paced at 10 Hz by the simulator, without `RealtimeSimulatorImpl`. Its per-step compute over 250 steps stays under 100 ms at the 99th percentile up to 48 robots in one cell:

| R | 4 | 8 | 16 | 24 | 32 | 48 | 64 |
|:---|:---|:---|:---|:---|:---|:---|:---|
| p50 / p99 / max (ms) | 5 / 11 / 11 | 14 / 26 / 39 | 21 / 32 / 38 | 27 / 44 / 172 | 37 / 61 / 107 | 53 / 90 / 151 | 76 / 381 / 1,024 |

One outlier of 100–170 ms occurs per 250 steps.

## Lockstep speed and the Windows path

Lockstep timing over 40 steps with robots random-walking 20–80 m from the gNB, p = 0.3 and 70% 4 kB / 30% 30 kB frames. **Contention: load average 30–68 throughout, with another agent running 8 ns-3 processes.** RTT is ms per control step, ns-3 is the maximum over processes of the time inside `Run()`, and overhead (RTT minus ns-3) covers serialization, transport and parsing:

| E × R | TCP procs: RTT (ns-3, overhead) | Unix procs: RTT (overhead) | shared memory procs: RTT (overhead) | TCP single process: RTT (overhead) |
|:---|:---|:---|:---|:---|
| 1 × 8 | 25.6 (25.3, 0.20) | 22.4 (0.15) | 23.6 (0.09) | 24.4 (0.25) |
| 1 × 16 | 55.1 (54.7, 0.21) | 52.4 (0.14) | 52.2 (0.10) | 56.1 (0.20) |
| 4 × 8 | 49.5 (48.1, 0.45) | 54.6 (0.54) | 33.6 (0.12) | 108.6 (0.24) |
| 4 × 16 | 122 (120, 0.48) | 123 (0.44) | 81.9 (0.16) | 293 (0.31) |
| 8 × 16 | 188 (184, 3.0) | 128 (2.8) | | 848 (0.45) |

ns-3 dominates, and bridge overhead is 0.1–0.5 ms, under 2% of the step, except with 8 parallel processes on the overloaded box, where it is 2–3 ms. Shared memory has the lowest overhead (about 0.1 ms) and was fastest end to end at E ≥ 2, perhaps because the spinning server stays scheduled, at the cost of one spinning core per side. Single-process time grows linearly with E, while `procs` parallelizes. RESET costs 0.05–1 s and process startup 0.14–1.2 s. One process runs about 4× faster than real time at R = 8, and 8 × 16 in `procs` runs at about real time.

**Windows to WSL.** The Isaac side runs on Windows while ns-3 runs in WSL. The stdlib-only client `win_client.py` on Windows reaches the TCP bridge in WSL through WSL2 localhost forwarding, with a median overhead of 0.40–0.46 ms per step on Windows against 0.20–1.05 ms for the same client inside WSL (p95 up to 5.6 ms at 8 × 8), so crossing the boundary adds about 0.2 ms per step, negligible next to 25–190 ms of simulation. For the pool, `launcher="wsl"` spawns the workers through `wsl.exe` and connects through localhost forwarding. A PowerShell smoke test (`scripts/windows/win_smoke.ps1`, 4 UEs, 18 ms per step) passed, but the Python launcher branch has not been run on Windows. `Ns3NetModule` was tested only in WSL, and using it from Isaac's Python needs numpy and torch there.

## 5G-LENA is sensitive to heap layout

With the same command file, seed and binary, changing only the length of a command-line path string makes the replies differ from step 12 on, while repeated runs with the same argv are bit-identical. As far as it was isolated, the outcome depends on the process heap layout (a long argv string is heap-allocated and a short one is not), and 5G-LENA has pointer-ordered containers, for example `NrGnbPhy::m_csiRsOffsetToUes` is a `std::set<Ptr<NrUeNetDevice>>`. The effect is a butterfly: a one-slot shift in one frame and a different sample path from then on. The offline runner therefore invokes each env with the pool's exact argument list inside its own directory, which is what made the offline-equals-closed-loop check exact (the first version differed in 3 of 24 env-runs). **Bit-exact comparisons between ns-3 runs, including lockstep against pool, need an identical argv shape, and every other per-frame comparison between differently invoked runs, including 5G-LENA against the GPU engine, is only meaningful statistically.**

## Known limits

Frames that reach the 2 s client timeout stay in the UE's RLC queue and keep using air time, because stock 5G-LENA does not purge them (mismatch 14 in [validation-5g-lena.md](validation-5g-lena.md)), and the bridges do not purge them either. In single-process lockstep mode a partial reset is logical only: old-episode frames are ignored but still occupy the queue, so per-env resets need `procs`. The 3GPP channel update period defaults to 0, the standalone scenario's setting and required for exact equivalence, so path loss and shadowing follow the robots but the small-scale channel does not regenerate as they move. `--chanUpdateMs` exists but is untested. Lab paths remain the defaults and are overridable through `NS3BRIDGE_ROOT`, `BRIDGE_ROOT` and `NS3_TOOLCHAIN_ENV`.
