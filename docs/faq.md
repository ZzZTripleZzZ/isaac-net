# FAQ

Questions new users ask in their first week, with short answers and the page that has the details.

## Is the default `NRConfig()` validated against 5G-LENA?

No. `NRConfig()` is a reasonable 3GPP configuration (30 kHz subcarriers, 20 MHz, `DDDSU`, 16 HARQ processes, EESM), but replaying the ns-3 5G-LENA sweep with it puts the median delay 29–76% low. For a claim about realistic delay, use `lena_validation_v2()`, whose replay has a median p50 delay error of −0.2% (absolute 3.4%), on the `reference` or `graph` backend. For training at scale on `triton`, use "v2 minus BSR", `lena_validation_v2(ul_grant_model="lumped", sr_grant_delay_slots=40)`, with a median p50 error of −3.5% / −5.7% / −0.8% at light / moderate / saturated load ([Fidelity vs 5G-LENA](fidelity-vs-lena.md#scale-configurations), [Cookbook recipe 5](cookbook.md#5-the-5g-lena-validated-configuration-and-the-scale-configuration)).

## Why does `make_engine` say the 5G-LENA tables are missing?

The validation presets (`lena_like`, `lena_validation`, `lena_match_v2`, `lena_validation_v2`) use `bler_source="lena"`, the BLER tables of 5G-LENA. These are GPL-2.0 data, so they are never shipped or committed. Generate them once from your own 5G-LENA checkout:

```bash
git clone https://gitlab.com/cttc-lena/nr.git ~/src/nr
python -m isaac_net.tools.extract_lena_tables ~/src/nr     # writes ~/.cache/isaac_net/lena_eesm_tables.npz
```

`ISAAC_NET_LENA_TABLES` points the engine to another location. Keep the file out of anything you redistribute ([Licensing](licensing.md)).

## Why do my numbers differ from the README?

Usually for one of four reasons. First, the README speed tables come from an idle RTX 4090 with the median of three processes ([Performance](performance.md#measurement-conditions)), and a GPU that other jobs keep busy is slower: on a GPU kept 90–100% busy by other jobs the same rows took 1.7–8.4× longer ([Contended vs uncontended](performance.md#contended-vs-uncontended)). Second, the eager `reference` backends are host-bound, so their cost varies by up to 2–3× between processes with the CPU's state. Third, the tutorials and the cookbook use tiny sizes to finish on a CPU, and their delays and counts show how to read the outputs, not what the engine measures. Fourth, the configuration matters: the scale rows run "v2 minus BSR" (`ul_v2l`), which costs 17–39% more per step than `NRConfig()` and gives different delays.

## How fast is it on my GPU?

Time it on your own hardware, with the GPU otherwise idle. A minimal timing loop:

```python
import time, torch
from isaac_net import NRConfig, Requests, make_engine

E, R = 1024, 32
net = make_engine("L2", E, R, "cuda", NRConfig(), backend="triton", seed=0)
pos, send = torch.rand(E, R, 2, device="cuda") * 150, Requests((torch.rand(E, R, device="cuda") < 0.3).long())
for _ in range(20):                                 # warm-up: Triton compilation, queues filling up
    net.submit(None, send); net.step(None, pos)
torch.cuda.synchronize(); t0 = time.perf_counter()
for _ in range(100):
    net.submit(None, send); net.step(None, pos)
torch.cuda.synchronize(); print(f"{(time.perf_counter() - t0) * 10:.2f} ms per control step")
```

From a clone, `benchmarks/uncontended/bench_net.py` repeats the measurement protocol of [Performance](performance.md) for one configuration, and `python benchmarks/bench.py --grid 256x16 --backends graph,triton` sweeps the prototype levels. If your version ships a diagnostic tool (an environment check and a short benchmark), the README's Install section names it. `isaac-net-bench` is a different tool: it runs the [benchmark suite](benchmark-suite.md) of network-aware tasks, not a speed test.

## Why does `triton` refuse my config?

The fused kernel holds one cell's scheduler and HARQ state per env and implements a fixed set of MAC features. For level `L2` it refuses several cells (and with them handover and radio link failure), the SR / BSR grant pipeline (`ul_grant_model="bsr"`, which `lena_validation_v2()` turns on), rank-2 MIMO, mini-slot grants, SINR hooks (which the energy and background-user wrappers install), and debug traces. It refuses them before anything is built, with one message that names every offending field, and `NRTritonEngine.refusals(cfg)` gives the same list without building ([NR engine backends](configurability.md#nr-engine-backends)). Run such configs on `graph`, which accepts every `L2` feature except the debug traces. `triton` also exists only for `L1`, `L2-legacy` and `L2`, not for the delay levels, the surrogates, the bounds or `WIFI`.

## Can I use it without Isaac Lab?

Yes. The engine is plain PyTorch: `make_engine` takes robot positions or an SNR from any simulator, or from none. The [benchmark suite](benchmark-suite.md) tasks are pure torch, `isaac_net.isaac.NetModule` and `NetEnvMixin` import no Isaac code ([Tutorial 04](tutorials/04_isaac_lab_integration.ipynb) runs them on a CPU), and `isaac_net.mjx` runs the network inside MuJoCo Playground / MJX ([MJX backend](backends-mjx.md)). Only `isaac_net/examples/isaac_fleet_env.py` and the `isaac`-marked tests need Isaac Lab.

## Does it run on a CPU or a Mac?

The `reference` backend of every level runs on a CPU, which is enough for the tutorials, the cookbook and the CPU test suite (`pytest -m "not gpu"`). The `graph` backend of `L2` and `WIFI` needs CUDA, and `triton` needs an NVIDIA GPU (on Windows through the community `triton-windows` build). The reference NR engine is slow (about 10 s for 50 control steps of 64 × 8 on a laptop CPU), so a CPU is for development, not training.

## How do I make results independent of the batch size?

Keep `NRConfig.rng = "engine"`, the default. Every draw is then keyed by (seed, env id, episode of that env, step), so env 7 in its third episode draws the same numbers whether `E` is 16 or 4096, and whatever the other envs do. The same keying makes `ShardedEngine` bitwise equal to one engine. Three things break it: `rng="global"` (stepping draws from the global torch RNG), traffic models or background users on `L2` (their generator is sequential per engine), and an edge loop with exponential service or return jitter. `shard_invariant(level, cfg)` in `isaac_net.core.sharded` tells whether a config is batch-independent. Results also differ in the last bits between a CPU and a GPU, and between `triton` and `graph` ([Configurability](configurability.md#engine-owned-randomness-and-the-application-constants-featprotolevels)).

## Why is a reset bitwise isolated?

`reset(env_ids)` re-initializes only the listed envs: their queues, MAC and HARQ state, link adaptation, fading, shadowing field and clock. It writes those rows in place with fixed-shape masked operations, and the new random state comes from each env's own stream, keyed by its id and a new episode number. No other env's tensors are touched, and no shared generator advances, so the other envs continue bit for bit as if nothing happened. The test suite checks this for every level and backend, through random partial resets ([Concepts](concepts.md#per-env-clocks-and-partial-resets)).

## My config change does nothing. Why?

Every level reads only some `NRConfig` fields. The prototype levels (`L0` to `L1`, `L2-legacy`) have a fixed radio and MAC, so for example `n_harq` or `mcs_table` changes nothing there, and by default the field is ignored silently. `cfg.unused_fields(level)` lists what a level ignores, and `make_engine(..., strict=True)` raises instead ([Cookbook recipe 10](cookbook.md#10-check-a-config-before-a-long-run)). Also check that you built a new engine: an engine reads its config once, at construction.

## Why are the step outputs `[E, R, F]`, and what does `delivered[e, r, f]` mean?

Each robot has a fixed buffer of `F` message slots (16 by default), and the per-message outputs refer to those slots as they stood before the step: `delivered[e, r, f]` is about the message that sat in slot `f` of robot `r`'s queue, and `cap` and `cls` tell you which message that was (`-1` and `0` for an empty slot). `delay` is in control steps from capture to delivery. Fixed shapes are what let thousands of envs run as one GPU program ([Concepts](concepts.md#fixed-shape-state)).

## Why do delays come in steps of 2.5 ms?

The slot-level engines advance time in slots, and a message completes at the end of the slot in which its last byte is decoded. `L2-legacy` has 40 uplink slots per 100 ms control step, one every 2.5 ms. `L2` follows its frame structure: with `DDDSU` at 30 kHz the uplink data slots are also 2.5 ms apart. Effects below one slot are not modelled ([Concepts](concepts.md#slot-synchronous-stepping-instead-of-discrete-events)).

## Can I change the control step, the buffer depth or the timeout?

Yes, for every level: `NRConfig(control_step_ms=50.0, frame_buffer=32, timeout_steps=40)`. Delays and timeouts stay in control steps. A fitted surrogate records the values it was fitted with, and `make_engine` refuses a fit whose values differ from the config ([Configurability](configurability.md#engine-owned-randomness-and-the-application-constants-featprotolevels)). Inside Isaac Lab, `control_step_ms` must match the env step, or set `IsaacNetCfg.net_decimation` or `net_substeps`.

## Which GPU, driver and PyTorch build do I need?

Python 3.10 to 3.12 and PyTorch 2.7 or later. On Linux, the CUDA build of PyTorch from PyPI includes Triton. Isaac Lab 3.0 pins torch 2.12 with CUDA 13.0, which needs NVIDIA driver 580.65.06 or newer on Linux and 580.88 or newer on Windows. The core package alone also works with the CUDA 12.6 build (`--index-url https://download.pytorch.org/whl/cu126`), which runs on older drivers. The results in these docs were measured on an RTX 4090 and an L40 ([Isaac Lab on Linux](isaac-lab-linux.md#versions)). The repository's `docker/` image sets up the Linux stack with these versions.

## How do I cite it?

Cite the paper: Zifan Zhang, Mingzhe Han, Kannan Athreya and Yuchen Liu, "Network-in-the-Loop at Scale: GPU-Batched 5G Simulation for Massively Parallel Robot Learning", [arXiv:2610.02370](https://arxiv.org/abs/2610.02370), 2026 (Zifan Zhang and Mingzhe Han contributed equally).

```bibtex
@article{zhang2026isaacnet,
  title   = {Network-in-the-Loop at Scale: GPU-Batched 5G Simulation for Massively Parallel Robot Learning},
  author  = {Zhang, Zifan and Han, Mingzhe and Athreya, Kannan and Liu, Yuchen},
  journal = {arXiv preprint arXiv:2610.02370},
  year    = {2026},
  note    = {Zifan Zhang and Mingzhe Han contributed equally}
}
```

`CITATION.cff` in the repository carries the same entry, so GitHub's "Cite this repository" button gives it too. If you report numbers, also state the package version (`isaac_net.__version__`), the level, the backend and the preset.
