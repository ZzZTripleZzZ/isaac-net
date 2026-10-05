# Tests

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or your CUDA build
pip install -e ".[dev]"
python -m pytest -m "not gpu"          # CPU suite (what CI runs)
python -m pytest -m "not gpu and not slow"   # quick CPU pass
scripts/run_gpu_tests.sh               # GPU suite on a CUDA machine
ruff check .                           # lint (light config in pyproject.toml)
```

Markers: `gpu` tests are skipped automatically when CUDA is unavailable, `isaac` tests when Isaac Lab is not installed; `slow` marks longer runs
(`-m "not slow"` to skip them). Torch runs single-threaded in tests (`ISAAC_NET_TEST_THREADS` overrides).
The suite also runs in a plain checkout without `pip install -e .` (pyproject.toml puts the repo root on the path).

## Layout

`engine_api.py` is the only place the prototype-level tests call the engine API (construction, `add_frames` /
`step`, stats) and reach into engine internals (`_transmit`, module-level `serve_fifo`, `randn_like` /
`rand_like` for noise injection). When the prototype engine API changes, update this file first.

| File | What it checks |
|---|---|
| `test_fifo.py` | `serve_fifo` serves in FIFO order and removes min(granted, queued) bytes; `_compact` keeps live frames in order and permutes all fields together; the fast backend's scatter compaction equals the reference argsort; the FIFO stays compacted and ordered at every level |
| `test_conservation.py` | accepted = delivered + timed out + queued (frames and bytes) at every level; no frame is delivered or removed twice; overflow count; per-slot byte service; a delivered frame had exactly its size served |
| `test_timeout.py` | an undelivered frame is dropped after exactly `TIMEOUT` steps, never earlier or later, at every level |
| `test_delay.py` | delay = finish time − capture step, the finish time lies inside the reporting step (slot-aligned for L1/L2), logged delays match, `newest` and `det_env` match the delivered frames |
| `test_lookup.py` | L05/L05Q: the bin a fit computes from logged features equals the bin queried at arrival (delay and drop tables) |
| `test_mac_l2.py` | exact HARQ/RLC timeline under forced failures, HARQ reset on success and on empty queue, OLLA saturation and bounds, power-headroom cap on subbands (reference and fast eager) |
| `test_determinism.py` | seeded runs are bitwise reproducible at every level; different seeds differ |
| `test_equivalence_cpu.py` | fast `eager` backend (the ops the `graph` backend captures) is bitwise equal to the reference on CPU |
| `test_env_cpu.py` | smoke test of the example task |
| `test_gpu.py` | `graph` == reference bitwise; `triton` statistically close (same draws, and own RNG as `slow`); seeded determinism of `graph`/`triton`; example task with both backends |
| `test_package.py` | every public module imports on CPU without Isaac Lab, ns-3, Sionna or Triton; the prototype shims alias the package modules; only the Sionna tables ship; `make_engine` rejects bad levels, backends and configs |
| `test_engine_api.py` | `make_engine` at every level: dict outputs, per-env clocks, partial resets that leave other envs bitwise unaffected (NR and legacy), `L2-legacy` == prototype NetSlot, NR legacy calls and clock checks, poses vs SNR input, downlink outputs, the SINR hook |
| `test_nr_phy.py` | TS 38.214 TBS / segmentation examples and config tables; Sionna cross-check (`slow`, needs sionna); 5G-LENA semantics (tables only if generated locally); BLER tables monotone, gap to Shannon, cross-table agreement; link adaptation meets the BLER target |
| `test_nr_harq.py` | multi-process HARQ and RLC in-order delivery against a host-side reference rebuilt from the TB trace (H1–H11: byte tiling, completion times, no HOL blocking, RLC UM loss, downlink, discard modes, partial reset, calibration knobs) |
| `test_nr_compat.py` | NR engine with `netslot_compat()` close to the legacy NetSlot at light load (`slow`) |
| `test_traffic.py` | traffic models: arrival counts, slots, jitter bounds, GOP sizes, burst structure and rates, event triggers; generator partial reset isolation; CUDA-graph capture == eager (`gpu`); engine byte conservation with every model; a 10 ms periodic model at a 100 ms step equals policy messages at a 10 ms step (delays to 1e-4 ms, with and without fading); delay counted from the arrival slot; engine partial reset isolation; submit extras and deadline misses; `policy()` alone bitwise equal to no traffic; every other level refuses traffic models |
| `test_multicell.py` | NetSlotMC at C = 1 bitwise equal to NetSlot (CPU, and GPU as `gpu`); RadioMC draws Radio's field; partial reset coverage, values and isolation at C = 3 with handovers; drive-through handover count, position and interruption |
| `test_edge.py` | `EdgeLoop`: FIFO and PS completion times against hand-computed schedules (multi-server, class table, step boundaries), queue-full and deadline drops, the return-delay formula, newest-action and in-flight replacement, exact times when the event budget runs out, conservation (arrived = completed + dropped + at the edge) at L0 / L1 / L2-legacy / L2, partial resets that leave other envs bitwise unaffected, `make_engine` wrapping, the NR downlink return path; CUDA-graph capture of the edge stage around the `L2-legacy` graph backend bitwise equal to eager (`gpu`) |
| `test_levels.py` | `TR`, `GE`, `QA`, `NN`, `ORACLE`, `NOCOMM`: dict outputs, per-env clocks, poses and legacy calls, partial resets (index and mask) that leave other envs bitwise unaffected in outputs and state, reset draws from the engine generator, conservation and exact timeouts; ORACLE delivers at capture and NOCOMM never; the exact TR matching rule, GE transitions and state-dependent loss, QA's analytic finish time and contention, NN's FIFO clamp and self-generated history; parameter loading (file path, fit dict, size check) and rejections; a CPU smoke fit from `L2-legacy` saved outside the repo; the rollout logger's features equal the engine's; `graph` == reference bitwise with injected draws and random partial resets, and with the default CUDA RNG (`gpu`) |
| `test_nr_multicell.py` | NR engine with several cells: C = 1 bitwise equal to the single-cell NR engine before the multi-cell MAC (971fc12, frozen copy in `tests/nr_frozen/`; CPU, and GPU as `gpu`) over five MAC configs with a partial reset; partial reset at C = 3 (UL + DL, handovers) covers every per-env tensor and leaves the other envs bitwise unaffected; drive-through handover count, fire slot (first A3 slot + TTT), no transmission in the interruption, `flush` vs `carry`; the user SINR hook runs after the engine's interference; sanity sweep and NetSlotMC cross-check (`slow`) |
| `test_nr_sched.py` | NR scheduler variants: `pf_wideband` bitwise equal to `pf_metric="wideband"`; with a fixed PSD max C/I carries the most bytes, then PF, then round robin, and max C/I is the least fair and starves the weakest robot; round robin rotates |
| `test_nr_fast.py` | NR engine RNG (`rng="engine"`: independent of the global RNG, partial resets and E, scalar key reference, CUDA vs torch draws) and fast backends (`gpu`): `graph` bitwise equal to the reference on eight configs incl. three cells, per-robot Doppler and sub-step traffic, with partial resets; `triton` teacher-forced and free-running; NetModule with `graph` / `triton`; the energy wrapper's per-slot tap on `graph`; `graph` bitwise and `triton` teacher-forced with the 5G-LENA MAC switches (G7, G2). `nr_equiv.py` is the harness (CLI for the 300-step runs); the short pytest runs start its traffic cycle at the medium phase (`phase_offset`) so they run under load |
| `test_nr_loadfix.py` | 5G-LENA MAC switches of the NR engine (`pf_update`, `pf_avg_idle`, `ul_retx_sched`, `ul_amc_alloc`, `ul_grant_model`; CPU): every switch set bitwise equal to the frozen load-gap prototype (`tests/nr_frozen/loadfix_proto.py`, on the frozen base in `tests/nr_frozen/base/`) through a partial reset; defaults and the v2 presets; BSR levels; the SR / bootstrap / BSR / RLC-tail timeline; per-RBG PF spreading; TDMA retransmissions; previous-PUSCH AMC; partial reset of the extra state; conservation; three cells with every switch; triton refuses the BSR pipeline |
| `test_channels.py` | channel models: default `log_distance` bitwise equal to the pre-channel `RadioMC` (draws, values, partial reset); shadowing 1/e distance for the legacy band and the exponential ACF, white-component variance split; TR 38.901 path loss of every scenario against hand-computed Table 7.4.1-1 values (functions and through `RadioMC`), LOS probability, `k_subsce` and O2I wall losses against hand values; LOS share over envs equals Pr_LOS, spatial consistency of the LOS state, O2I statistics; partial-reset isolation; per-robot Doppler (config rule, speed from poses and velocities, reset floor, equality with the global model at equal speed, decorrelation order); radio-map bilinear sampling, file round trip, shipped synthetic map, cell checks, engine run; blockage geometry and loss; CUDA-graph capture of the models (`gpu`) |
| `test_rng_levels.py` | engine-owned RNG (`rng="engine"`): runs do not depend on the global RNG (every level, multi-cell too), per-env streams independent of E and of other envs' resets, a reset re-seeds its env deterministically, `rng="global"` keeps the old dependence, fast eager == reference without injection; `mix32` and `CounterRNG.uniform` pinned to hard-coded values; F / timeout / control step / UL slots from the config (F = 32, timeout 40, 50 ms / 20 slots, and an odd variant) on every level with frame and byte conservation and exact timeouts; the fit tool records these values and loading refuses a mismatch; GPU: graph == reference bitwise without injection (defaults and non-default configs, partial resets, global-RNG disturbance), `compile` == reference to rounding (L0DR, L1, L2-legacy), Triton hash == torch hash, triton close to the reference on the same engine draws |
| `test_isaac_layer.py` | Isaac `NetModule` on `make_engine`: equals the reference engine driven directly, bitwise, at L0, L0DR, L1, L2-legacy (eager) and L2 (NR), through a mid-run partial reset (CPU, and GPU); `graph` bitwise with injected draws; `graph`/`triton` partial-reset invariants (GPU); triton free running with and without a mid-run partial reset bitwise equal on the untouched envs (GPU); Isaac radio vs engine radio, blockage and DR parameters; MessageHistory first capture; mixin, mdp terms and the `NetConfig` alias without Isaac Lab |
| `test_isaac_env.py` | marker `isaac` (needs Isaac Lab 3.0): inside Isaac Lab, the fleet env's network equals a reference-engine replay bitwise through partial resets made by DirectRLEnv (`tests/scripts/isaac_fleet_check.py`); every level and backend steps the fleet env |
| `test_sharded.py` | `ShardedEngine`: two shards equal the unsharded engine with partial resets across the shard boundary, background load and energy; bitwise on CUDA, and on CPU bitwise for integers and booleans and to float32 rounding for the float outputs of L1 / L2-legacy (CPU log / exp kernels may round differently for a different tensor shape, seen on macOS arm64) |

`tests/nr_frozen/` holds the frozen golden references of the NR engine (the single-cell engine of 971fc12 for
`test_nr_multicell.py` M1, the load-gap prototype for `test_nr_loadfix.py` L0) and, in `base/`, a copy of the
package modules they build on (`nr_engine`, `mac*`, `phy`, `queues`, `nr_rng`, `proto/rng`) at one commit
(`base.FROZEN_COMMIT`). After an intentional change of the live MAC, re-freeze it with
`python tests/scripts/refreeze_nr.py --commit <rev>` (procedure in that script); `--check` verifies the copy.

`tests/scripts/` holds the command-line equivalence scripts (`test_equiv.py`, `test_reset.py`,
`test_regress.py` against the frozen `netsim_v0.py`, and `regress_main.py`, which dumps every level and backend at
the defaults with `rng="global"` and compares two checkouts bitwise), run by hand on a GPU; they are not collected by
pytest.
`tests/bridges/` holds the ns-3 bridge checks, which need a local ns-3 + 5G-LENA build and the lab paths in
the bridge READMEs; they are not collected either.

## Adding tests

A change to a prototype level goes into `isaac_net/core/proto/netsim.py` first; `test_equivalence_cpu.py`
and the GPU `graph` test then show whether `netsim_fast.py` followed. Keep GPU tests small (E ≤ 64); the lab
GPU is shared.
