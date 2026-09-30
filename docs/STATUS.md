# Project status

*State of 2026-09-29: `main` at `971fc12`, plus three work-in-progress branches.*

`isaaclab-net` is an early research prototype that is now packaged as `isaaclab_net`. The engine, its fast backends, the configurable NR engine, the legacy multi-cell uplink, a snapshot of the Isaac Lab layer and the ns-3 bridges are all merged on `main`, and the CPU test suite passes on a clean checkout. Work outside the repository has produced a 5G-LENA reference sweep, a public-data calibration and an Isaac Lab 3.0 installation that ran 1,048,576 robots with the network in the loop. The remaining work is mostly integration (three branches), speed for the NR engine, and a formal fidelity comparison. A paper describing the engine is in preparation.

## What is done

Each row below is on `main`. Merge commits are listed first, with the component commits after them.

| Component | Commits | How it was verified |
|:---|:---|:---|
| Prototype engine: levels `L0`, `L0DR`, `L05`, `L05Q`, `L1` and the slot-level uplink now called `L2-legacy`, with `eager`, `graph`, `compile` and `triton` backends | `e1afdb9` | Graph backend bitwise identical to the reference at E×R up to 256×16 and 64×100 over 300 steps ([performance.md](performance.md)) |
| pytest suite (67 CPU + 7 GPU tests), CI workflow, GPU test runner | `ac7f1f4` (`9f0ea1b`, `d0729bd`) | 67 CPU tests pass in a fresh venv; GPU tests pass on the lab GPU (one intermittent OOM caused by other tenants) |
| Engine API: partial `reset(env_ids)`, per-env clocks, `submit` / `step` dict outputs, fast backends for every level, PF average floor | `fbfcb44` (`c8d0daf`, `f2c94de`, `64261d8`, `4cb7a82`, `b22f594`, `c624245`) | New reference bitwise equal to the frozen original at all 6 levels; graph bitwise equal to the reference with random partial resets; untouched envs bitwise unaffected by resets (14/14 cases) |
| Package layout `isaaclab_net/` with `prototype/` shims | `1bb4ad1`, lint fix `51f5279` | `ruff` clean on every commit |
| Configurable NR engine as `L2`, one `NRConfig` with presets, `make_engine`, `NREngine`, multi-cell `NetSlotMC` on `L2-legacy`, Sionna BLER tables | `a7f78a7` | `test_nr_phy` (TS 38.214 examples, 564,300-case TBS cross-check against Sionna), `test_nr_harq` (H1–H11), `test_multicell` (C = 1 bitwise equal to NetSlot), `test_engine_api` |
| Isaac Lab layer snapshot (NetModule, DirectRLEnv mixin, mdp terms), demo env, Windows install scripts | `ed20a25` | superseded by `feat/isaac` (below) |
| ns-3 bridges: lockstep, process pool, offline replay, and the NR engine's 5G-LENA replay tool | `651a93e` | Import tests; `Ns3Net` and `PoolNet` smoke-tested through the package against the lab ns-3 builds |
| README, ARCHITECTURE, CONTRIBUTING, tests/README | `664c3a5`; merge of the package `971fc12` | README quick-start snippets run verbatim on the lab GPU |

On a clean `git archive` of the package branch, `pip install -e ".[dev]"` followed by `pytest` gives 165 passed and 2 skipped (Sionna not installed, 5G-LENA tables not generated) in 4 min 43 s. The quick pass `-m "not gpu and not slow"` gives 157 passed in 48 s, and `pytest -m gpu` gives 10 passed. The CI workflow is in `.github/workflows/ci.yml`. Whether it has run green on GitHub since `main` was pushed was not checked for this page.

The following results were produced outside the package code, with scripts that live in the project's research workspace and the lab box, and are documented here:

- **5G-LENA reference.** ns-3.48 + 5G-LENA v5.1, a 186-run sweep, and the replay of 153 of those runs in the NR engine, with a mean absolute drop-rate difference of 0.029 ([validation-5g-lena.md](validation-5g-lena.md)).
- **Public-data calibration.** Uplink latency fits to srsRAN and OAI measurements, a contention fit on ColO-RAN and channel fits on POWDER drive tests, which became the `srsran_like` and `oai_like` presets ([calibration-public-data.md](calibration-public-data.md)).
- **Isaac Lab 3.0 on Windows.** A native install, a demo env, and a scale sweep up to 8,192 envs × 128 robots ([isaac-lab.md](isaac-lab.md)).

## In progress

Three branches carry work that is not yet on `main`. Each one has its own git worktree on the development machine.

**Multi-cell NR engine (`feat/nr-multicell`, rebased on `deb1b35`).** This branch adds multiple cells to the NR engine (`L2`) by porting the design of the legacy `NetSlotMC` onto the NR MAC. `NRNet` accepts up to 7 cells, with per-link fading `[E,R,C,S,2]`, one PF scheduler and one set of HARQ processes per cell, uplink and downlink same-slot interference through `MacLink.sinr_hook`, fractional uplink power control on by default for more than one cell, and A3 handover whose time-to-trigger and interruption are counted in NR slots. At `n_cells = 1` none of the new code runs, and a frozen copy of the single-cell engine (971fc12) in `tests/nr_frozen/` checks that every output and state tensor stays bitwise equal. The new tests are in `tests/test_nr_multicell.py`, and a sweep script is in `benchmarks/multicell/sweep_nr.py`. The multi-cell design is described in [multicell.md](multicell.md).

**Surrogate and bound levels (merged into `main` at `4e73df5`).** This work ported the fitted network surrogates and adds two bounds as `make_engine` levels under `isaaclab_net/core/levels/`, with a fit script `python -m isaaclab_net.tools.fit_levels`. The surrogates are `TR` (each env replays one recorded `L2` env-episode, open loop), `GE` (a 3-state Markov-modulated delay and loss chain per env), `QA` (an analytic processor-sharing queue per control step with FIFO service and SR delay) and `NN` (a stateful MLP that predicts a drop probability and 32 delay quantiles from send-time features). The bounds are `ORACLE` (every message delivered at capture with delay 0) and `NOCOMM` (no message ever delivered). Fitted parameters are written outside the repository. The surrogates were first fitted with the pre-package code on logs of the legacy `L2` engine under a fixed data-collection policy of the example fleet task, and their held-out, teacher-forced network-level fit was as follows (frame-weighted Wasserstein-1 distance of delay per lookup cell in ms, and drop calibration over 123k held-out frames):

| Level | W1, `L05` cells | W1, `L05Q` cells | W1 pooled | Drop ECE | Brier | Log-loss |
|:---|---:|---:|---:|---:|---:|---:|
| `NN` | 12.7 | 16.3 | 9.1 | 0.0034 | 0.065 | 0.204 |
| `L05Q` | 18.9 | 18.1 | 8.7 | 0.0026 | 0.092 | 0.291 |
| `L05` | 30.3 | 464.7 | 11.3 | 0.0066 | 0.126 | 0.392 |
| `GE` | 269.4 | 491.4 | 25.9 | 0.0109 | 0.194 | 0.572 |
| `L0` | 984.5 | 1234.2 | 805.4 | 0.0076 | 0.241 | 0.675 |

A rerun with different random draws moved the `L05` value by about 3 ms, so differences of that size are noise. `TR` and `QA` have no per-cell teacher-forced number in that report. These numbers come from the pre-port code and have not been re-measured on the branch.

**`feat/isaac` (validated, ready to merge).** This branch rebuilds the Isaac layer on `make_engine` and `NRConfig`. `NetModule` wraps any level, keeps the Isaac-side radio (per-env domain randomization, several gNBs, blockage, pose chunks), the per-message tag and the freshness outputs, and keeps `NetConfig` only as a thin alias. The registry engine is retired, and the fleet env, benchmark and training scripts select levels. It passes its tests on WSL and on the Windows Isaac Lab 3.0 install, including an in-Isaac bitwise replay check through partial resets, and the README quick start runs verbatim ([isaac-lab.md](isaac-lab.md#rebuilt-layer-on-make_engine)).

## Open items, in priority order

| # | Item | Why it matters | Pointer |
|:---|:---|:---|:---|
| 1 | `graph` / `triton` backends for the NR engine (`L2`) | The NR engine runs only on the `reference` backend and costs about 2.6–3.9× the legacy reference per step, which is launch-bound and flat in E. Its step has no host syncs and fixed shapes, so CUDA-graph capture of one control step is the first option, and the multi-cell step on `feat/nr-multicell` is also sync-free apart from the per-TB SINR log. Downlink would also benefit from batching the D slots of one TDD period. A new backend needs an eager-vs-graph bitwise test like `test_gpu.py` | [performance.md](performance.md), ARCHITECTURE follow-up 1 |
| 2 | Uncontended re-benchmark | Every timing so far ran on a GPU 95–99% busy with other jobs, which inflates launch-bound code most (the legacy reference was 77 ms per step at 16×16 on a quiet GPU and about 1,400 ms under contention). The headline rows to repeat are the backend table at 256×16 and 4096×100 and the Isaac scale rows at 4096×32/64/128, network off vs `triton` | [performance.md](performance.md), `benchmarks/bench.py`, `benchmarks/isaac/run_scale.ps1` |
| 3 | Formal fidelity comparison against 5G-LENA | The replay so far compares drop rate, p50/p95 and PRB use. Still to do: delay CDFs by KS distance per (N, S, p), per-UE goodput, the HARQ redundancy-version histogram, BSR quantization and grant padding (a `bsr_table` knob) to close the 1.3× PRB-use gap at light load, and the TBS and overhead check against LENA's `NrUlMacStats.txt` | [validation-5g-lena.md](validation-5g-lena.md) |
| 4 | Measure the SR-to-grant delay in 5G-LENA | `lena_validation()` uses 40 slots (about 20 ms), which was inferred from the replay because it aligns p50 and p95. It needs a hook on the UE SR transmission and on the first UL DCI in the ns-3 scenario | [validation-5g-lena.md](validation-5g-lena.md) |
| 5 | Merge the remaining branches | `feat/nr-multicell` routes `make_engine("L2", config=multicell(n))` to the NR engine (the CPU suite and the multicell GPU tests pass on the branch), `feat/levels` is merged (220 CPU and 19 GPU tests pass), and `feat/isaac` is validated on the lab box | above |
| 6 | Run Isaac jobs as a console user | CUDA is unavailable from the ssh session on the Windows lab box, so every GPU run so far was a one-shot SYSTEM scheduled task. Once someone logs on at the console, Interactive-logon tasks can replace SYSTEM with no other change | [isaac-lab.md](isaac-lab.md) |
| 7 | Multi-cell defaults against 5G-LENA | P0 and α of the uplink power control are set from a sanity sweep, not calibrated against a reference | [multicell.md](multicell.md) |
| 8 | 16 HARQ processes at overload in the `netslot_compat` geometry | With 16 processes, loss is lower at light load (0.353 vs 0.390) but higher at overload (0.797 vs 0.724). The likely cause is UEs in trouble that keep consuming PRBs with new TBs while their retransmissions block in-order delivery. Worth checking against 5G-LENA, which uses 16 processes natively | [validation-5g-lena.md](validation-5g-lena.md) |
| 9 | Delay levels at scale | At 4096×100 the `graph` versions of `L0`–`L05Q` (about 45 ms) are about 2× slower than the eager reference (about 20 ms), because the fixed-shape path is memory-bound. Options: a Triton kernel for the delay levels, or a `clone_outputs=False` option | [performance.md](performance.md) |
| 10 | Isaac startup and memory at scale | Scene startup grows at about 1.1 ms per robot (1,131 s at 1M robots). Options: `clone_in_fabric=True`, one multi-instance asset per env instead of an R-object `RigidObjectCollection`, or a kinematic pose integrator | [isaac-lab.md](isaac-lab.md) |
| 11 | Bridge follow-ups | The `tests/bridges/` checks and `bridges/ns3_offline/lena_replay.py` were import-rewired but not re-run after the package port. Stale frames are not purged from the ns-3 RLC queue, partial reset is supported only in the process-per-env mode, `--chanUpdateMs` is untested, and the Windows Python launcher of the pool is untested | [bridges.md](bridges.md) |
| 12 | Smaller PHY and MAC gaps | MIESM is not implemented (EESM with calibrated betas only), proactive grants carry no padding model, the UL DCI-slot direction constraint for K2 is not modelled, and the Sionna 2.2.0 table defects found here could be reported upstream | [validation-5g-lena.md](validation-5g-lena.md) |
| 13 | Reference scenario in the repository | The standalone 5G-LENA scenario `netslot-ref.cc`, the crash-guard patch script and the sweep scripts live outside this repository. Only the two bridge programs derived from the scenario are in `isaaclab_net/bridges/ns3/` | [validation-5g-lena.md](validation-5g-lena.md) |
| 14 | Lab measurements | Public data cannot validate NR uplink multi-UE contention, per-TB BLER vs SINR, the configured SR period and proactive grants, latency with large frames under load, an indoor channel, or fading correlation at robot speeds. These need the lab gNB (up to 4 UEs) or POWDER (up to 2 UEs) | [calibration-public-data.md](calibration-public-data.md) |

## Suggested starter tasks

A good first contribution touches one module, comes with a test, and does not need the shared lab GPU for long. The tasks below are ordered roughly from least to most context needed.

1. **Run the suite and read one engine end to end.** Install the package on a CPU machine, run `python -m pytest -m "not gpu and not slow"`, and read `isaaclab_net/core/proto/netsim.py` (the eager reference of every prototype level) alongside `tests/test_mac_l2.py`, which pins the HARQ and RLC timeline slot by slot.
2. **Quiet-GPU benchmark.** On any free CUDA machine, run `benchmarks/bench.py` for the 256×16 and 4096×100 rows of every level and backend and report them with the GPU model and utilization. This directly addresses open item 2 and needs no code change.
3. **Bring the reference scenario into the repository.** Move `netslot-ref.cc`, `parse_run.py` and the crash-guard patch script into `isaaclab_net/bridges/ns3/ref/` next to the bridge programs, with a build script that follows the recipe in [validation-5g-lena.md](validation-5g-lena.md). No ns-3 or 5G-LENA source may be copied in.
4. **SR-to-grant hook.** In that scenario, trace the UE SR transmission and the first UL DCI per UE, and measure the delay across the sweep's light-load runs. The result either confirms the inferred 40 slots in `lena_validation()` or replaces it.
5. **BSR quantization.** Add a `bsr_table` option to `NRConfig` with the TS 38.321 buffer-size levels, apply it in `mac_ul.py`, and rerun `python -m isaaclab_net.bridges.ns3_offline.lena_replay` to see whether the 1.3× PRB-use gap at light load closes.
6. **CUDA graph for the NR engine.** Capture one control step of `NRNet` in a CUDA graph for a fixed configuration and add a bitwise eager-vs-graph test modelled on `tests/test_gpu.py`. This is the largest item and the one with the highest payoff.
