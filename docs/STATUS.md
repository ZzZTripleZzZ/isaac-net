# Project status

*State of 2026-09-30: `main` packaged as version 0.1.0; all planned work items are merged.*

`isaaclab-net` is a research prototype packaged as `isaaclab_net`. Every feature branch of the third round of work is merged on `main`: fast backends for the NR engine, channel models, traffic models, the edge loop, background users, the energy model, multi-GPU sharding, the Wi-Fi level, adaptive fidelity, differentiable models, radio maps from USD scenes, the benchmark suite, the OAI rfsim bridge, the Linux / HPC recipe and the MuJoCo Playground / MJX backend. The speed and scale tables come from an uncontended benchmark campaign on an idle GPU ([performance.md](performance.md)), and the 5G-LENA load-gap mechanisms found in [fidelity-load-gap.md](fidelity-load-gap.md) are NR engine switches with the `lena_match_v2` preset. [CHANGELOG.md](https://github.com/ZzZTripleZzZ/isaaclab-net/blob/main/CHANGELOG.md) lists what 0.1.0 contains, and [RELEASE.md](https://github.com/ZzZTripleZzZ/isaaclab-net/blob/main/RELEASE.md) how a release is cut. A paper describing the engine is in preparation.

## What is merged

Each row is on `main`. The commits are the merge or the main component commits, oldest first within a row.

| Area | Component | Commits | How it was verified |
|:---|:---|:---|:---|
| Engine | Prototype levels `L0`, `L0DR`, `L05`, `L05Q`, `L1` and the slot-level uplink `L2-legacy`, with `eager`, `graph`, `compile` and `triton` backends | `e1afdb9` | `graph` bitwise equal to the reference at E×R up to 256×16 and 64×100 over 300 steps ([performance.md](performance.md)) |
| Engine | Engine API: partial `reset(env_ids)`, per-env clocks, `submit` / `step` dict outputs, fast backends for every level | `fbfcb44` | New reference bitwise equal to the frozen original at all 6 levels; `graph` bitwise equal with random partial resets; untouched envs bitwise unaffected |
| Engine | Package layout, configurable NR engine as `L2`, one `NRConfig` with presets, `make_engine`, multi-cell `NetSlotMC`, Sionna BLER tables | `971fc12` | `test_nr_phy` (TS 38.214 examples, 564,300-case TBS cross-check against Sionna), `test_nr_harq`, `test_multicell`, `test_engine_api` |
| Engine | Surrogate levels `TR`, `GE`, `QA`, `NN` and the bounds `ORACLE`, `NOCOMM`, fit tool | `4e73df5` | `test_levels`; `graph` bitwise equal to the reference |
| Engine | Multi-cell NR engine: per-cell PF and HARQ, UL and DL same-slot interference, fractional power control, A3 handover | `096e86a`, `661f780` | `test_nr_multicell`; at one cell bitwise equal to a frozen single-cell engine |
| Engine | Engine-owned random streams and configurable frame buffer, timeout and control step for the prototype, surrogate and bound levels | `308fdfe` | `test_rng_levels` |
| Engine | NR engine: scheduler variants (`pf`, `pf_wideband`, `maxci`, `rr`), engine RNG, `graph` (one or several cells) and `triton` (one cell, UL and DL) backends | `e5ffc42` … `b79578f` | `test_nr_sched`, `test_nr_fast` (300-step `graph` == reference bitwise, `triton` to rounding, traffic sub-steps) |
| Radio | Selectable channel models: log-distance variants, TR 38.901 (8 scenarios, spatially consistent LOS, O2I), radio maps, blockage, per-robot Doppler | `388a7fb`, `753619f` | `test_channels` (hand-computed path loss, LOS share, spatial consistency, default bitwise equal to the old radio, CUDA-graph capture) |
| Radio | Radio maps from USD scenes: exporter with ITU-R P.2040 materials, Sionna RT bake CLI, Isaac hook with a scene-hash cache, warehouse demo | `a7ba517` | `test_scene_radio_map`, `test_scene_bake` (`slow`, needs Sionna RT); synthetic scenes within about 1 dB of closed forms ([scene-radio-map.md](scene-radio-map.md)) |
| Traffic and stages | Traffic models (periodic, bursty, video, event) inside the `L2` step | `1d533e9` | `test_traffic`, including the `graph` test |
| Traffic and stages | Edge-computing loop `EdgeLoop` over any level | `0f8aaf9` | `test_edge` (conservation, eager == graph bitwise with partial resets) |
| Traffic and stages | Background users, radio energy model, multi-GPU `ShardedEngine` | `5b64ca4` | `test_background`, `test_energy`, `test_sharded` (two shards bitwise equal to one engine, [background-energy-sharding.md](background-energy-sharding.md)) |
| Levels | Level `WIFI`: mean-field 802.11 DCF / EDCA, event simulator, validation tables | `2134085` | `test_wifi`; ns-3 802.11ax saturation within 3.1% ([wifi.md](wifi.md)) |
| Levels | Differentiable fluid models `L1D`, `QAD`, neural-proxy recipe | `512f4d1` | `test_diff` (exact at temperature 0, gradcheck, finite differences) |
| Levels | Adaptive and mixed fidelity (`AdaptiveEngine`) | `2d96580` … `2400ed9` | `test_adaptive` (threshold 0 / ∞ bitwise equal to the plain levels, handoffs exact, graph == eager) |
| Validation | ns-3 bridges (lockstep, pool, offline replay) and the 5G-LENA replay | `651a93e` | Import tests; lockstep and pool smoke-tested against the lab ns-3 builds |
| Validation | Formal comparison against 5G-LENA, and the load-gap diagnosis with the `nr_loadfix.py` prototype | `aac4e17`, `e4726f1`, `5ff3eb9` | CSVs and report scripts in `benchmarks/fidelity/` ([fidelity-vs-lena.md](fidelity-vs-lena.md), [fidelity-load-gap.md](fidelity-load-gap.md)); `test_nr_loadfix` (all switches off bitwise equal to the engine) |
| Validation | OAI 5G rfsim bridge, campaign, `oai_rfsim` preset | `432c46d` … `41b31e7` | `tests/bridges/oai` (fake stack); campaign results in `benchmarks/oai/results/` ([bridges-oai.md](bridges-oai.md)) |
| Validation | Measurement protocol and tools for a lab gNB and POWDER | `ec0037f`, `5738520` | `test_measure_parsers`, `test_measure_calibrate` on fixture logs |
| Simulators | Isaac Lab layer on `make_engine` and `NRConfig`, `IsaacNetCfg`, observation selection, network domain randomization | `2a98bc3`, `848bcfa` | `test_isaac_layer`; `test_isaac_env` on the Windows Isaac Lab 3.0 install (in-env bitwise replay through partial resets) |
| Simulators | Linux / HPC recipe: kit-less Isaac Lab on Newton or OV PhysX in Apptainer, Slurm templates | `ddfa6ed` | 9/9 `isaac` tests on both physics backends ([isaac-lab-linux.md](isaac-lab-linux.md)) |
| Simulators | MuJoCo Playground / MJX backend, MJX fleet env, Brax PPO | `b871ef3`, `9075a09` | `tests/mjx` (in-env == direct torch replay bitwise, zero-copy check) |
| Benchmark suite | `isaaclab_net.bench`: four tasks, metrics, baselines, runner, result format, load calibration | `4ecb4f5`, `32eb045` | `test_bench` (reproducibility, partial-reset isolation, GPU `graph` == reference episodes) |
| Docs and packaging | Docs site, tutorials, API reference, licensing page; 0.1.0 packaging with extras and console scripts | `8cbd1a9`, `eaa9185`, release branch | `mkdocs build --strict`; wheel installed in a fresh CPU venv, suite run from outside the source tree |

The CPU suite runs in CI (ruff, `pytest -m "not gpu"`, strict docs build). Installed from the 0.1.0 wheel into a fresh venv with the CPU build of torch and run from outside the source tree, `pytest -m "not gpu"` gives 690 passed and 21 skipped (Isaac Lab, JAX, pxr, Sionna and the 5G-LENA tables absent). The one other test, the check of the `prototype/` shims, needs the source tree and fails there by design. The last full GPU run on the lab box gave 75 GPU tests passed. Whether CI has run green on GitHub since the round-3 merges was not checked for this page.

The following results were produced outside the package code, with scripts in the project's research workspace and on the lab box, and are documented here: the ns-3.48 + 5G-LENA v5.1 reference sweep of 186 runs ([validation-5g-lena.md](validation-5g-lena.md)), the public-data calibration that produced the `srsran_like` and `oai_like` presets ([calibration-public-data.md](calibration-public-data.md)), and the Isaac Lab 3.0 scale sweep up to 8,192 envs × 128 robots on Windows ([isaac-lab.md](isaac-lab.md)).

## Recently merged

| Area | What | Evidence |
|:--|:--|:--|
| Performance | Uncontended benchmark campaign on an idle RTX 4090 and a dedicated L40: every level and backend, the fleet task, Isaac Lab scale up to 1,048,576 robots with the NR engine, the ns-3 cost recomputation | CSVs and raw JSONL in `benchmarks/results/uncontended/` ([performance.md](performance.md)) |
| Validation | 5G-LENA load-gap mechanisms as `NRConfig` switches (`pf_update`, `pf_avg_idle`, `ul_retx_sched`, `ul_amc_alloc`, `ul_grant_model`), defaults bitwise unchanged, presets `lena_match_v2` and `lena_validation_v2` with no fitted parameter; `graph` equals the reference exactly, `triton` covers all but the BSR grant pipeline | 153-run replay in `benchmarks/fidelity/results/loadfix_v2/` ([fidelity-vs-lena.md](fidelity-vs-lena.md), [fidelity-load-gap.md](fidelity-load-gap.md)); `test_nr_loadfix` matches a frozen copy of the prototype exactly |

## Open items, in priority order

| # | Item | Why it matters | Pointer |
|:---|:---|:---|:---|
| 1 | NR engine `triton` for several cells, and at large R | `graph` covers one or several cells and `triton` one cell. The fused kernel holds one env's robots in one program, so large R needs a tiled kernel, and several cells need per-cell schedulers and interference inside it | [performance.md](performance.md), ARCHITECTURE follow-up 1 |
| 2 | Remaining differences to 5G-LENA after the load-gap fixes | The engine's UL MCS is one step above 5G-LENA's for about half the UEs at 3–15 dB SNR, and the first-transmission BLER differs by a few tenths of a point, causes not isolated. The TBS and overhead check against `NrUlMacStats.txt` is open | [fidelity-load-gap.md](fidelity-load-gap.md) |
| 3 | Lab measurement campaign | Public data cannot validate multi-UE uplink contention, per-TB BLER against SINR, the configured SR period and proactive grants, latency with large frames under load, an indoor channel, or fading correlation at robot speeds. The protocol and tools are ready for the lab gNB (up to 4 UEs) and POWDER (up to 2 UEs) | [measurement-protocol.md](measurement-protocol.md) |
| 4 | Multi-UE tails against OAI | No preset reproduces OAI's multi-UE delay tails or its saturation at 4.8 Mbit/s of cell load | [bridges-oai.md](bridges-oai.md) |
| 5 | Multi-cell against a reference | P0 and α of the uplink power control come from a sanity sweep, and multi-cell has not been compared with any reference simulator. The downlink has no power control or interference coordination | [multicell.md](multicell.md) |
| 6 | 16 HARQ processes at overload in the `netslot_compat` geometry | With 16 processes, loss is lower at light load (0.353 vs 0.390) but higher at overload (0.797 vs 0.724), likely because UEs in trouble keep taking PRBs for new TBs while retransmissions block in-order delivery | [validation-5g-lena.md](validation-5g-lena.md) |
| 7 | Isaac Sim (Kit) on Linux clusters | Kit-less Isaac Lab works in Apptainer, but the only Isaac Lab 3.0 Kit container is a release candidate on which the fleet env fails. It needs the final `isaac-lab:3.0.0` image or a custom image | [isaac-lab-linux.md](isaac-lab-linux.md) |
| 8 | Isaac startup and memory at scale | Scene startup grows by about 1.1 ms per robot (19 minutes at 1M robots). Options: `clone_in_fabric=True`, one multi-instance asset per env, or a kinematic pose integrator | [isaac-lab.md](isaac-lab.md) |
| 9 | Run Isaac jobs on Windows as a console user | CUDA is unavailable from the ssh session on the Windows lab box, so every GPU run there was a one-shot SYSTEM task | [isaac-lab.md](isaac-lab.md) |
| 10 | MJX host sync and NetModule capture | A masked reset in the core would remove the one host sync per step, and one CUDA graph for the whole `NetModule` step would cut its host-side launches | [backends-mjx.md](backends-mjx.md#open-items) |
| 11 | Differentiable models at full scale | The full benchmark runs (`benchmarks/diff/run_all.sh`), the sign and magnitude comparison against `L2-legacy` finite differences, and the fit quality of the neural proxy are pending | [differentiable.md](differentiable.md) |
| 12 | Shard invariance of the NR engine | `ShardedEngine` runs `L2`, but its stepping draws depend on the split, so two shards are not bitwise one engine | [background-energy-sharding.md](background-energy-sharding.md) |
| 13 | Wi-Fi scope | The mean-field model loses accuracy for small contention windows (VO-only saturation, −41% at 10 stations) and has no downlink, OFDMA or MU-MIMO | [wifi.md](wifi.md) |
| 14 | Bridge follow-ups | Stale frames are not purged from the ns-3 RLC queue, partial reset works only in the process-per-env mode, and the Windows launcher of the pool is untested | [bridges.md](bridges.md) |
| 15 | Smaller PHY and MAC gaps | MIESM, FR2 numerology, FDD, MIMO layers and DL power control are not modelled, and the UL DCI-slot constraint for K2 is missing | [configurability.md](configurability.md) |
| 16 | Reference scenario in the repository | The standalone 5G-LENA scenario `netslot-ref.cc`, its crash-guard patch and the sweep scripts live outside this repository | [validation-5g-lena.md](validation-5g-lena.md) |
| 17 | Public release | PyPI publishing, the public repository URL and the citation wait until the project is public | [RELEASE.md](https://github.com/ZzZTripleZzZ/isaaclab-net/blob/main/RELEASE.md) |

## Suggested starter tasks

A good first contribution touches one module, comes with a test, and does not need the shared lab GPU for long. The tasks below are ordered roughly from least to most context needed.

1. **Run the suite and read one engine end to end.** Install the package on a CPU machine, run `python -m pytest -m "not gpu and not slow"`, and read `isaaclab_net/core/proto/netsim.py` (the eager reference of every prototype level) alongside `tests/test_mac_l2.py`, which pins the HARQ and RLC timeline slot by slot.
2. **A benchmark-suite baseline.** Train a new baseline on one task of `isaaclab_net.bench` and submit its result files as described in [benchmark-suite.md](benchmark-suite.md#submitting-a-baseline). No engine change is needed.
3. **Bring the reference scenario into the repository.** Move `netslot-ref.cc`, `parse_run.py` and the crash-guard patch script into `isaaclab_net/bridges/ns3/ref/` next to the bridge programs, with a build script that follows the recipe in [validation-5g-lena.md](validation-5g-lena.md). No ns-3 or 5G-LENA source may be copied in.
4. **A scheduler or PHY gap.** Pick one of the gaps in item 15, for example DL power control or FDD as an all-uplink pattern with a paired DL carrier, add it behind an `NRConfig` field whose default keeps the engine bitwise, and test it next to `tests/test_nr_sched.py`.
5. **Masked reset.** Add `reset_mask(mask [E])` to one level next to `reset(env_ids)`, drawing for all envs and keeping the masked rows, with a test that the kept rows match `reset(env_ids)` in distribution and that untouched envs are bitwise unaffected (item 10).
6. **Tiled NR `triton` kernel.** Split the robots of one env over several programs in `nr_triton.py` and check the result against the reference with the harness in `tests/nr_equiv.py`. This is the largest item and the one with the highest payoff at scale.
