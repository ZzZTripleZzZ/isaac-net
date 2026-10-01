# Documentation

These pages are for collaborators joining the project. The top-level [README](../README.md) shows how to install the package and put a network into an environment, and [ARCHITECTURE.md](../ARCHITECTURE.md) defines the package layout and the interface contract every engine honors. The pages below record what exists beyond that: how each part was validated, how fast it runs, and what is still open. They describe the state of `main` at `2400ed9` (2026-09-30, version 0.1.0) and the two items in flight listed in the status page.

| Page | What it covers |
|:---|:---|
| [concepts.md](concepts.md) | Slot-synchronous stepping, fixed-shape state, per-env clocks and partial resets, backends and the equivalence guarantee, fidelity levels, and a glossary for networking and robotics readers |
| [tutorials/](tutorials/index.md) | Five tutorials as executed notebooks (scripts in [`tutorials/`](../tutorials/)): first network, choosing fidelity, configuring NR, Isaac Lab integration, validating against ns-3 |
| [reference/](reference/index.md) | API reference: engine contract and outputs, `NRConfig` with every field, traffic, fidelity levels, Isaac Lab layer, ns-3 bridges |
| [licensing.md](licensing.md) | What ships under which license, the locally generated 5G-LENA tables, GPL-bound bridge binaries, Isaac Sim and the Omniverse EULA, public datasets |
| [STATUS.md](STATUS.md) | What is merged (with commits), what is in flight, the prioritized open items, and suggested starter tasks |
| [validation-5g-lena.md](validation-5g-lena.md) | The ns-3.48 + 5G-LENA v5.1 reference, the 186-run sweep, the model mismatch table and which side was changed, and the NR engine's replay of the sweep |
| [fidelity-vs-lena.md](fidelity-vs-lena.md) | The formal NR-engine vs 5G-LENA comparison: per-run and sweep-level KS / W1 / quantile / drop / goodput / HARQ / PRB errors, the SR-delay fit and hold-out split, ablations, the legacy engine and fading arm, and speed |
| [fidelity-heldout.md](fidelity-heldout.md) | One held-out 5G-LENA scenario (10 MHz carrier, 2 RBGs, new UE drops) replayed with `lena_validation_v2` unchanged, and the BLER and MCS-step residuals on it |
| [calibration-public-data.md](calibration-public-data.md) | Public datasets and their licenses, the data problems found, fitted parameters, the `lena_match` / `srsran_like` / `oai_like` presets, and what public data cannot validate |
| [performance.md](performance.md) | Backend equivalence methodology, per-level and per-backend benchmarks, Isaac Lab scale results, and the ns-3 CPU co-simulation cost comparison |
| [isaac-lab.md](isaac-lab.md) | Windows-native Isaac Sim 6.1 / Isaac Lab 3.0 install, running CUDA jobs with nobody logged in, the demo env, the 1,048,576-robot scale table and the adapter fixes |
| [isaac-lab-linux.md](isaac-lab-linux.md) | Linux / HPC recipe (tested on RHEL 9 with Slurm and Apptainer): core GPU tests, kit-less Isaac Lab 3.0 on Newton and OV PhysX, offline asset mirror, sbatch templates, dedicated-GPU timings, failure modes |
| [backends-mjx.md](backends-mjx.md) | The second simulator backend: MuJoCo Playground / MJX (JAX) versions, the zero-copy `buffer_callback` interop, the MJX fleet env, bitwise validation against a direct replay, Brax PPO and throughput |
| [multicell.md](multicell.md) | Multi-cell design, interference timing, handover, the power-control default and the sweep findings |
| [bridges.md](bridges.md) | The ns-3 co-simulation bridges (lockstep, process pool, offline replay, real time), their correctness checks, costs and known limits |
| [channels.md](channels.md) | The selectable channel models (log-distance, TR 38.901, radio map), blockage and per-robot Doppler, their sources, simplifications and cost, and how to bake a Sionna RT radio map |
| [configurability.md](configurability.md) | Every hard-coded choice in the engines, a feature matrix against 5G-LENA, Sionna SYS and Simu5G, the proposed modes and switches, and the config fields added so far |
| [benchmark-suite.md](benchmark-suite.md) | The benchmark suite: task API, the four tasks and their variants, metrics, baselines, result format, load calibration, how to add a task or submit a baseline |
| [scene-radio-map.md](scene-radio-map.md) | Radio maps from USD scenes: export with ITU-R materials, the Sionna RT bake, the Isaac hook and its cache, validation on synthetic scenes |
| [wifi.md](wifi.md) | Level `WIFI`: the mean-field 802.11 DCF / EDCA model, what it leaves out, its validity range, and its validation against Bianchi, an event simulator and ns-3 |
| [background-energy-sharding.md](background-energy-sharding.md) | Background users per cell, the radio energy and battery model, and multi-GPU sharding with its invariance guarantee |
| [adaptive-fidelity.md](adaptive-fidelity.md) | `AdaptiveEngine`: static mixes, load-triggered switching with queue handoff, curricula, accuracy against cost |
| [differentiable.md](differentiable.md) | The differentiable fluid models `L1D` and `QAD`, the neural-proxy recipe, their checks and limits |
| [fidelity-load-gap.md](fidelity-load-gap.md) | Why the NR engine is optimistic under load against 5G-LENA, mechanism by mechanism, and the prototype that closes the gap |
| [bridges-oai.md](bridges-oai.md) | The OAI 5G rfsim bridge: deployment, virtual clock, measured access, HARQ and contention, and the comparison with the engine's presets |
| [measurement-protocol.md](measurement-protocol.md) | The runbook for measurements on a lab gNB and POWDER, with the parsers, probe and calibration tools (`isaac-net-measure`) |

**A note on every timing in these pages.** All GPU numbers were measured on one RTX 4090 in a shared lab box while other jobs kept it 95–99% busy, and all ns-3 numbers on a 32-core box with load averages between 10 and 68. Absolute times are therefore pessimistic, and only ratios measured in the same run are meaningful. An uncontended re-benchmark is the second item on the open list in [STATUS.md](STATUS.md).

**The docs site.** These pages also build into a site with a generated API reference: `pip install -e ".[docs]"`, then `mkdocs serve` for a live preview or `mkdocs build --strict`, which CI runs. The site's home page is [index.md](index.md), and this README is left out of it.
