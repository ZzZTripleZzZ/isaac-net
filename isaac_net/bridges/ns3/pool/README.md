# bridges/parallel: ns-3 / 5G-LENA co-simulation bridges (pool, offline replay, real time)

> **In isaac_net.** `ns3pool.py` and `poolnet.py` are `isaac_net.bridges.ns3_pool`; `rollout.py`,
> `offline_ns3.py`, `replaynet.py`, `replay_error.py` and `analyze_replay.py` are `isaac_net.bridges.ns3_offline`;
> `bench_scaling.py`, `analyze_scaling.py` and `rt_bench.py` are in `benchmarks/ns3/`; the checks are in
> `tests/bridges/pool/`; `ns3/netslot-bridge.cc` and `build_bridge.sh` are this directory. `vendor/` and `_paths.py`
> are gone: the modules import the package. PoolNet and ReplayNet follow the package's NetBase (per-env clocks,
> full resets only). Paths below are the lab-box layout; `BRIDGE_ROOT` and `NS3_TOOLCHAIN_ENV` override them.
> The C++ program is our own code, written against the ns-3 / 5G-LENA APIs; no ns-3 or 5G-LENA source is included.

Three ways to put ns-3.48 + 5G-LENA NR v5.1 (the `ns3ref` scenario `netslot-ref`) under the example
robot fleet. Each exposes the `netsim.NetBase` API, so it drops into `FleetEnv`, `train.py` and `evaluate`.

| Variant | Class / tool | Loop | Use |
|---|---|---|---|
| 1. Parallel pool | `poolnet.PoolNet` (front-end) + `ns3pool.Ns3Pool` (workers) | closed | "best-effort CPU co-simulation" baseline |
| 2. Offline trace-driven | `rollout.py` (record) + `offline_ns3.py` (ns-3 on traces) + `replaynet.ReplayNet` (replay) | open | evaluation without a closed loop; quantifies its flaw |
| 3. Real-time emulation | `netslot-bridge --io=rt:PORT` + `rt_bench.py` | wall clock | live Isaac GUI demo with a few robots |

Lab box: `/home/zzhang66/experiments/bridge_parallel` (WSL Ubuntu 20.04). Local mirror: this directory.
The report with all numbers is `scratchpad/bridge_parallel_report.md`.

## Build

`ns-3.48/` is a `cp -a` copy of `ns3ref/ns-3.48` including its build; the shared libraries' RPATH was
re-pointed to the copy with `patchelf`, so nothing loads from `ns3ref`. The toolchain is `ns3ref/env`, read-only.

```bash
bash ns3/build_bridge.sh      # compiles ns3/netslot-bridge.cc -> bin/netslot-bridge (about 1 min),
                              # copies the unmodified netslot-ref -> bin/netslot-ref; installs atomically
```

`build_bridge.sh` reuses the exact compile and link flags of the ns-3 build of `netslot-ref`, so no ns-3
reconfigure is needed.

## The worker: `ns3/netslot-bridge.cc`

It is `netslot-ref.cc` plus an external step interface. With `--io` empty it is exactly `netslot-ref`.

```
--io=tcp:PORT        one client over TCP (127.0.0.1; --bindAll=1 for 0.0.0.0, needed from Windows)
--io=stdio           stdin / stdout
--io=file:IN:OUT     offline, commands from a file
--io=rt:PORT         real-time emulation (RealtimeSimulatorImpl, BestEffort)
--init=x:y:loss,...  initial UE positions and path-loss overrides (loss < 0: the model's own loss)
--mobility=hold|waypoint   file mode only: hold = position held per step, waypoint = WaypointMobilityModel
--ueUeFilter=1       drop UE->UE signals in the spectrum channel (identical outcomes, 1.3-2.6x faster; pool default)
--flowmon=0          no FlowMonitor (pool default)
```

Step protocol (one ASCII line each way per 100 ms control step):

```
client -> worker   S <t> <np> {<ue> <x> <y> <lossDb>}*np <nf> {<ue> <fid> <bytes>}*nf      |  Q
worker -> client   D <t> <wallUs> <nd> {<ue> <fid> <genS> <lastS>}*nd
```

Step t covers sim time [0.5 + 0.1 t, 0.5 + 0.1 (t+1)). The frames are sent by each UE's `FrameSender`
tick at the start of the step, at exactly the event position the standalone sender uses: the worker's
`Simulator::Stop` for step t is scheduled before the tick event for step t, so `Run()` returns just
before the tick and the tick then sends the queued frames. The reply lists the frames whose last packet
reached the remote host during the step. This is what makes the bridge bit-identical to `netslot-ref`.

## 1. Parallel pool (`poolnet.py`, `ns3pool.py`)

```python
from poolnet import PoolNet
net = PoolNet(E, R, device, TASK_SIZES["T1"])     # E <= 12 workers on the shared box (Ns3Pool.MAX_PROCS)
env = FleetEnv(E, R, net, device); net.attach_env(env)   # attach: real robot xy for the fading geometry
```

- One ns-3 process per env, R UEs in one cell. Each `_transmit` scatters one line per worker (all R
  positions + path-loss overrides, plus the new frames) and gathers one line per worker: a barrier per step.
- Large-scale loss: the env's own SNR is passed as a per-UE path-loss override,
  `loss = P_TX - NI - snr_db = 113 - snr_db`, so ns-3 sees exactly NetSlot's single-subband full-power SNR,
  including the env's correlated shadowing field. Fast fading, AMC, HARQ, SR/BSR and scheduling are 5G-LENA's.
- Frame id = capture step, since there is at most one frame per robot per step. A frame dropped by
  `NetBase` (buffer overflow, 2 s timeout) is simply never matched. Overflowed frames are never sent to ns-3.
- ns-3 cannot rewind, so `reset()` restarts the workers (about 0.15-0.4 s at R <= 16, 1.5-4 s at R = 64).
  Env e in episode k uses ns-3 RNG run `seed_run + k*E + e`.
- From Windows (Isaac Sim): `PoolNet(..., launcher="wsl")` spawns the workers through `wsl.exe` with
  `--bindAll=1` and connects to `127.0.0.1` through WSL2 localhost forwarding. The raw path is tested by
  `windows/win_smoke.ps1` (4 UEs, 18 ms/step). The Python `launcher="wsl"` branch itself has not been run on
  Windows yet.

## 2. Offline trace-driven replay

```python
rec = rollout("random", PoolNet(...) or any NetBase, E, R, seed)   # records pos, SNR, accepted sends
outcomes, _ = offline_ns3.run_offline(rec, workdir, run_base)        # ns-3 per env in file mode, <= 12 at a time
net = ReplayNet(E, R, device, sizes, outcomes)                       # looks up (env, robot, frame id)
```

- The trace keeps only frames that entered the robot's buffer, because NetBase drops overflow before the
  network sees it. An earlier version sent overflowed frames too, and the offline run diverged.
- `ReplayNet` misses (another step, robot or size class than recorded) fall back to the nearest recorded
  frame of the same robot and class, then of the env, else lost. The hit mix is reported.
- `replay_error.py` runs the whole experiment (closed A, closed B, offline A, four replays), and
  `analyze_replay.py` prints the tables.

## 3. Real-time emulation

`netslot-bridge --io=rt:PORT` runs `RealtimeSimulatorImpl` in BestEffort mode. A reader thread injects
`F <ue> <fid> <bytes>` (send now) and `P <ue> <x> <y> <loss>` (move now) with `ScheduleWithContext`.
Deliveries stream back as `D ue fid gen last wall` as they happen. At the end the worker reports its lag
(wall - sim, sampled every 10 ms, after a 1.5 s warm-up). `--rtLagFile` dumps the lag series, and
`rt_bench.py` sweeps R.

On the shared box this holds real time reliably only for about 2 robots (up to 12 in good runs), because
the lag builds up in ramps of 0.1-1.5 s even without traffic. **For the live Isaac demo, use the
step-synchronous TCP worker instead, paced by Isaac at 10 Hz** (one `S` line per 100 ms of wall time).
Its per-step compute is p99 32 ms at 16 robots, 61 ms at 32 and 90 ms at 48 (`tests/step_jitter.py`).

## Tests

| Test | Command | Result |
|---|---|---|
| Correctness vs standalone | `python3 tests/test_correctness.py DIR <netslot-ref args>` | 3 cases x 4 paths (file+flowmon, file, TCP, `Ns3Pool` W=1): identical frame sets, delays equal to netslot-ref's printed precision |
| UE-UE filter is outcome-neutral | `python3 tests/test_filter.py DIR <args>` | identical per-frame delays, 2.6x (16 UE) / 1.3x (8 UE) faster |
| Heap-layout sensitivity | `bash tests/heap_sensitivity.sh CMDS INIT RUN` | the same inputs diverge at step 12 when only an argv length changes |
| Offline == closed loop | `replay_error.py` case `replay_random|random_hold` | 0 error, about 3,000 delivered frames per seed identical |
| Scaling | `bench_scaling.py OUT --W .. --R .. --p .. [--extra=--ueUeFilter=0]`; `analyze_scaling.py tag=files` | report, section 3 |
| Replay error | `replay_error.py OUT --E 8 --R 16 --seeds 0 1 2 --L 60`; `analyze_replay.py OUT/replay_error.jsonl` | report, section 4 |
| Real time | `rt_bench.py OUT --R 1 2 4 .. --dur 20 --L 60 [--static] [--lagdir D]`; `tests/step_jitter.py R STEPS` | report, section 5 |

## Caveat: 5G-LENA is heap-layout sensitive

Identical inputs, seed and binary give different (statistically equivalent) sample paths when the process
heap layout differs. Changing only the length of a command-line string (for example a longer output path)
is enough (`tests/heap_sensitivity.sh`). The likely cause is pointer-ordered containers (for example
`NrGnbPhy::m_csiRsOffsetToUes` is a `std::set<Ptr<NrUeNetDevice>>`). Consequences:

- Bit-for-bit comparisons need identical argv lengths. `offline_ns3.py` therefore runs each env in its own
  directory as `--io=file:c:r`, with the same argument list as a pool worker.
- Any other per-frame comparison between ns-3 runs (lockstep vs pool, LENA vs NetSlot) is only
  meaningful statistically. The noise floor is about the size of a different RNG run (report, section 2).

## Merge notes (no edits to repo-root files)

- `netfactory.make_net` could add `if rung == "NS3POOL": return PoolNet(E, R, device, sizes)`, plus
  `net.attach_env(env)` after `FleetEnv(...)` in `train.py`/`evaluate`. It is optional, because
  without it positions are derived from the SNR.
- PoolNet runs on CPU. Tensors are copied to the device of the NetBase buffers.

## Files

`ns3/netslot-bridge.cc`, `ns3/build_bridge.sh`, `ns3pool.py`, `poolnet.py`, `rollout.py`, `offline_ns3.py`,
`replaynet.py`, `replay_error.py`, `bench_scaling.py`, `rt_bench.py`, `analyze_scaling.py`, `analyze_replay.py`,
`tests/`, `windows/win_smoke.ps1`, `vendor/` (snapshot of netsim.py and env.py at the commit in `vendor/COMMIT`),
`results/` (small JSON results; `replay_v0_overflowbug/` and `rt_v0_nowarmup/` are superseded runs kept for the record).
