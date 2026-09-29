# Tests

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or your CUDA build
pip install -e ".[dev]"
python -m pytest -m "not gpu"          # CPU suite (what CI runs), about 2 min
scripts/run_gpu_tests.sh               # GPU suite on a CUDA machine
ruff check .                           # lint (light config in pyproject.toml)
```

Markers: `gpu` tests are skipped automatically when CUDA is unavailable; `slow` marks longer runs
(`-m "not slow"` to skip them). Torch runs single-threaded in tests (`ISAACLAB_NET_TEST_THREADS` overrides).

## Layout

`engine_api.py` is the only place that calls the engine API (construction, `add_frames` / `step`,
stats) and the only place that reaches into engine internals (`_transmit`, module-level `serve_fifo`,
`randn_like` / `rand_like` for noise injection). When the engine API changes, update this file first.

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
| `test_env_cpu.py`, `test_netmodule.py` | smoke tests of the example env and of the `NetModule` skeleton incl. partial reset |
| `test_gpu.py` | `graph` == reference bitwise; `triton` statistically close (same draws, and own RNG as `slow`); seeded determinism of `graph`/`triton`; example env with both backends |

## Adding tests

A change to the L2 model goes into `netsim.py` first; `test_equivalence_cpu.py` and the GPU
`graph` test then show whether `netsim_fast.py` followed. Keep GPU tests small (E ≤ 64); the lab GPU is shared.
