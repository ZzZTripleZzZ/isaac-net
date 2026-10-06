# Tutorials

A Colab quick start and five tutorials take you from a first network to validation against ns-3. Tutorials 01–05 are Python scripts in `tutorials/` at the repository root, and the pages here render the same scripts as executed notebooks, so you can read the outputs without running anything. The quick start is a notebook only, written for Google Colab.

| Tutorial | What you learn | Runs on |
|:---|:---|:---|
| [00 Quick start in Colab](00_quickstart_colab.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ZzZTripleZzZ/isaac-net/blob/main/docs/tutorials/00_quickstart_colab.ipynb) | `pip install`, an `L2` engine for 64 × 8 robots, the step dict, a delay CDF and the AoI over time, `NRConfig()` against a validated preset, strict mode | free Colab, CPU or GPU, under 3 minutes |
| [01 Your first network](01_first_network.ipynb) | `make_engine`, `submit` and `step`, reading delay, AoI and SINR, partial reset | CPU, seconds |
| [02 Choosing fidelity](02_choosing_fidelity.ipynb) | `L0`, `L1`, `L2-legacy` and `L2` on the same traffic, the `ORACLE` and `NOCOMM` bounds, fitting and loading a surrogate level | CPU, about a minute |
| [03 Configuring NR](03_configuring_nr.ipynb) | `NRConfig`, presets, TDD patterns, HARQ processes, downlink, multiple cells, strict mode | CPU, about a minute |
| [04 Isaac Lab integration](04_isaac_lab_integration.ipynb) | `NetModule`, `MessageHistory`, `NetEnvMixin`, network domain randomization | CPU for the executed cells; the `DirectRLEnv` listing needs Isaac Lab 3.0 and is not executed |
| [05 Validating against ns-3](05_validating_against_ns3.ipynb) | building and running the lockstep bridge, the Isaac-style ns-3 module, the 5G-LENA sweep replay | needs an ns-3 + 5G-LENA build; not executed here |

The tutorials use tiny sizes (a few environments and robots) so that they finish quickly on a CPU. Their numbers show how to read each output. They are not measurements of the engine or comparisons of the levels.

## Running them yourself

```bash
python tutorials/01_first_network.py                 # any tutorial runs as a plain script
```

To regenerate the notebooks rendered here, install `jupytext`, `nbconvert` and `ipykernel` next to the package and run

```bash
bash tutorials/build_notebooks.sh                    # converts every script and executes 01-04 on the CPU
```

which writes `docs/tutorials/*.ipynb`. Tutorial 05 is converted without execution unless `NS3BRIDGE_ROOT` points to an ns-3 bridge build.

The script does not touch the quick start, which has no script. To re-execute it on a CPU against your checkout:

```bash
CUDA_VISIBLE_DEVICES="" PYTHONPATH=. jupyter nbconvert --to notebook --execute --inplace docs/tutorials/00_quickstart_colab.ipynb
```
