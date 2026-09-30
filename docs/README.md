# Documentation

These pages are for collaborators joining the project. The top-level [README](../README.md) shows how to install the package and put a network into an environment, and [ARCHITECTURE.md](../ARCHITECTURE.md) defines the package layout and the interface contract every engine honors. The pages below record what exists beyond that: how each part was validated, how fast it runs, and what is still open. They describe the state of `main` at `971fc12` (2026-09-29) and the three work-in-progress branches listed in the status page.

| Page | What it covers |
|:---|:---|
| [concepts.md](concepts.md) | Slot-synchronous stepping, fixed-shape state, per-env clocks and partial resets, backends and the equivalence guarantee, fidelity levels, and a glossary for networking and robotics readers |
| [tutorials/](tutorials/index.md) | Five tutorials as executed notebooks (scripts in [`tutorials/`](../tutorials/)): first network, choosing fidelity, configuring NR, Isaac Lab integration, validating against ns-3 |
| [reference/](reference/index.md) | API reference: engine contract and outputs, `NRConfig` with every field, traffic, fidelity levels, Isaac Lab layer, ns-3 bridges |
| [licensing.md](licensing.md) | What ships under which license, the locally generated 5G-LENA tables, GPL-bound bridge binaries, Isaac Sim and the Omniverse EULA, public datasets |
| [STATUS.md](STATUS.md) | What is done (with commits), what is in progress on which branch, the prioritized open items, and suggested starter tasks |
| [validation-5g-lena.md](validation-5g-lena.md) | The ns-3.48 + 5G-LENA v5.1 reference, the 186-run sweep, the model mismatch table and which side was changed, and the NR engine's replay of the sweep |
| [calibration-public-data.md](calibration-public-data.md) | Public datasets and their licenses, the data problems found, fitted parameters, the `lena_match` / `srsran_like` / `oai_like` presets, and what public data cannot validate |
| [performance.md](performance.md) | Backend equivalence methodology, per-level and per-backend benchmarks, Isaac Lab scale results, and the ns-3 CPU co-simulation cost comparison |
| [isaac-lab.md](isaac-lab.md) | Windows-native Isaac Sim 6.1 / Isaac Lab 3.0 install, running CUDA jobs with nobody logged in, the demo env, the 1,048,576-robot scale table and the adapter fixes |
| [multicell.md](multicell.md) | Multi-cell design, interference timing, handover, the power-control default and the sweep findings |
| [bridges.md](bridges.md) | The ns-3 co-simulation bridges (lockstep, process pool, offline replay, real time), their correctness checks, costs and known limits |
| [configurability.md](configurability.md) | Every hard-coded choice in the engines, a feature matrix against 5G-LENA, Sionna SYS and Simu5G, the proposed modes and switches, and the config fields added so far |

**A note on every timing in these pages.** All GPU numbers were measured on one RTX 4090 in a shared lab box while other jobs kept it 95–99% busy, and all ns-3 numbers on a 32-core box with load averages between 10 and 68. Absolute times are therefore pessimistic, and only ratios measured in the same run are meaningful. An uncontended re-benchmark is the second item on the open list in [STATUS.md](STATUS.md).

**The docs site.** These pages also build into a site with a generated API reference: `pip install -e ".[docs]"`, then `mkdocs serve` for a live preview or `mkdocs build --strict`, which CI runs. The site's home page is [index.md](index.md), and this README is left out of it.
