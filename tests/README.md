# Tests

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or your CUDA build
pip install -e ".[dev]"
python -m pytest -m "not gpu"          # CPU suite (what CI runs)
python -m pytest -m "not gpu and not slow"   # quick CPU pass
scripts/run_gpu_tests.sh               # GPU suite on a CUDA machine
ruff check .                           # lint (light config in pyproject.toml)
```

Markers: `gpu` tests are skipped automatically when CUDA is unavailable; `slow` marks longer runs
(`-m "not slow"` to skip them). Torch runs single-threaded in tests (`ISAACLAB_NET_TEST_THREADS` overrides).
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
| `test_env_cpu.py`, `test_netmodule.py` | smoke tests of the example task and of the registry `NetModule` incl. partial reset |
| `test_gpu.py` | `graph` == reference bitwise; `triton` statistically close (same draws, and own RNG as `slow`); seeded determinism of `graph`/`triton`; example task with both backends |
| `test_package.py` | every public module imports on CPU without Isaac Lab, ns-3, Sionna or Triton; the prototype shims alias the package modules; only the Sionna tables ship; `make_engine` rejects bad levels, backends and configs |
| `test_engine_api.py` | `make_engine` at every level: dict outputs, per-env clocks, partial resets that leave other envs bitwise unaffected (NR and legacy), `L2-legacy` == prototype NetSlot, NR legacy calls and clock checks, poses vs SNR input, downlink outputs, the SINR hook |
| `test_nr_phy.py` | TS 38.214 TBS / segmentation examples and config tables; Sionna cross-check (`slow`, needs sionna); 5G-LENA semantics (tables only if generated locally); BLER tables monotone, gap to Shannon, cross-table agreement; link adaptation meets the BLER target |
| `test_nr_harq.py` | multi-process HARQ and RLC in-order delivery against a host-side reference rebuilt from the TB trace (H1–H11: byte tiling, completion times, no HOL blocking, RLC UM loss, downlink, discard modes, partial reset, calibration knobs) |
| `test_nr_compat.py` | NR engine with `netslot_compat()` close to the legacy NetSlot at light load (`slow`) |
| `test_multicell.py` | NetSlotMC at C = 1 bitwise equal to NetSlot (CPU, and GPU as `gpu`); RadioMC draws Radio's field; partial reset coverage, values and isolation at C = 3 with handovers; drive-through handover count, position and interruption |
| `test_isaac_layer.py` | Isaac `NetModule`: fast eager == registry engine through a partial reset (CPU, and GPU); `graph` partial reset (GPU); the mixin and mdp terms without Isaac Lab |

`tests/scripts/` holds the command-line equivalence scripts (`test_equiv.py`, `test_reset.py`,
`test_regress.py` against the frozen `netsim_v0.py`), run by hand on a GPU; they are not collected by pytest.
`tests/bridges/` holds the ns-3 bridge checks, which need a local ns-3 + 5G-LENA build and the lab paths in
the bridge READMEs; they are not collected either.

## Adding tests

A change to a prototype level goes into `isaaclab_net/core/proto/netsim.py` first; `test_equivalence_cpu.py`
and the GPU `graph` test then show whether `netsim_fast.py` followed. Keep GPU tests small (E ≤ 64); the lab
GPU is shared.
