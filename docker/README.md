# Docker image (Linux, NVIDIA GPU)

`docker/Dockerfile` builds `isaac-net` for Linux with an NVIDIA GPU in two stages:

| Stage | What it holds | Python env |
|:---|:---|:---|
| `core` (default) | Ubuntu 22.04 with the CUDA 12.6 runtime, uv, Python 3.11, PyTorch from the cu126 index (Triton included), `isaac-net[dev]` | `/opt/venv` |
| `isaaclab` | `core` plus Isaac Lab 3.0 kit-less (`release/3.0.0` at `60e28c1`) with Newton and OV PhysX physics and rsl-rl, no Isaac Sim, and `isaac-net` installed into it | `/opt/lab` (Isaac Lab pins torch 2.12+cu130) |

The `isaaclab` stage follows the kit-less recipe of [docs/isaac-lab-linux.md](../docs/isaac-lab-linux.md) (`uv sync --extra rsl-rl --extra ovphysx` in an Isaac Lab checkout, then `uv pip install --no-deps` of `isaac-net` and `pytest`), with the cluster paths replaced by `/opt/IsaacLab` and `/opt/lab`. On the cluster the recipe ran inside an Apptainer container to get glibc 2.35 or newer. Ubuntu 22.04 has glibc 2.35, so the image needs no second container.

## Requirements

- Docker 23 or newer (BuildKit is the default builder; the Dockerfile uses a bind mount during the build), and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) for `--gpus all`.
- An NVIDIA driver that supports CUDA 12.6 for the `core` stage. The `isaaclab` stage runs Isaac Lab's cu130 build of torch, which needs driver 580.65.06 or newer.
- Internet access during the build, and at run time for the Isaac Lab tasks, which load their USD assets from NVIDIA's servers (see [Offline GPU nodes](../docs/isaac-lab-linux.md#offline-gpu-nodes-and-cloud-assets) for a machine without it).

## Build

Build from the repository root, which is the build context:

```bash
docker build -f docker/Dockerfile -t isaac-net .                                        # isaac-net[dev] from PyPI
docker build -f docker/Dockerfile --build-arg ISAAC_NET_INSTALL=editable -t isaac-net:dev .   # this checkout, pip install -e
docker build -f docker/Dockerfile --target isaaclab -t isaac-net:isaaclab .             # plus Isaac Lab 3.0 kit-less
```

| Build argument | Default | Meaning |
|:---|:---|:---|
| `ISAAC_NET_INSTALL` | `pypi` | `pypi`: `pip install "isaac-net[dev]>=0.2.0"` and copy only `tests/`, `benchmarks/` and `pyproject.toml` to `/src/isaac-net`; `editable`: copy the checkout to `/src/isaac-net` and `pip install -e ".[dev]"` |
| `ISAAC_NET_SPEC` | `>=0.2.0` | version specifier for the PyPI install |
| `TORCH_INDEX_URL` | `https://download.pytorch.org/whl/cu126` | the PyTorch wheel index; `.../whl/cu128` or `.../whl/cpu` work too |
| `BASE_IMAGE` | `nvidia/cuda:12.6.3-runtime-ubuntu22.04` | any Ubuntu 22.04 or newer image; `nvidia/cuda:12.6.3-base-ubuntu22.04` is enough, because the torch wheels bring their own CUDA libraries |
| `ISAACLAB_BRANCH`, `ISAACLAB_COMMIT` | `release/3.0.0`, `60e28c1` | the Isaac Lab checkout (`isaaclab` stage); `ISAACLAB_COMMIT=""` builds the branch head |

With `editable`, the isaaclab stage installs the same checkout into `/opt/lab`. With `pypi`, it installs the same `isaac-net` version from PyPI with `--no-deps`.

## Run

```bash
docker run --rm -it --gpus all isaac-net                                     # a shell in /src/isaac-net
docker run --rm --gpus all isaac-net nvidia-smi                              # the GPU is visible
docker run --rm --gpus all isaac-net python -c "import torch, isaac_net; print(isaac_net.__version__, torch.cuda.is_available())"
```

Mount a host directory to keep results, for example `-v "$PWD/results:/results"`. Use `--shm-size=8g` (or `--ipc=host`) for multi-process training, since Docker's default 64 MB of shared memory is small for PyTorch data loaders.

### The test suites

Inside the `core` image, from `/src/isaac-net`:

```bash
docker run --rm --gpus all isaac-net python -m pytest -m "not gpu"     # CPU suite, what CI runs (several minutes)
docker run --rm --gpus all isaac-net python -m pytest -m gpu           # backend equivalence on the GPU
```

The `isaac` tests skip in `core`, since it has no Isaac Lab, and so do the tests that need Sionna, JAX or the locally generated 5G-LENA tables. With `ISAAC_NET_INSTALL=pypi` the suite runs against the installed package, as in step 7 of [RELEASE.md](../RELEASE.md), and the one test that needs the `prototype/` directory skips itself.

In the `isaaclab` image, the Isaac tests run on either kit-less physics backend:

```bash
docker run --rm --gpus all isaac-net:isaaclab env ISAAC_NET_PHYSICS=newton python -m pytest -m isaac tests/test_isaac_env.py
docker run --rm --gpus all isaac-net:isaaclab env ISAAC_NET_PHYSICS=ovphysx python -m pytest -m isaac tests/test_isaac_env.py
docker run --rm --gpus all isaac-net:isaaclab isaaclab train --rl_library rsl_rl --task Isaac-Cartpole-Direct --num_envs 16 --max_iterations 10 physics=newton_mjwarp
```

These are the commands of [docs/isaac-lab-linux.md](../docs/isaac-lab-linux.md#the-kit-less-recipe). The project's cluster templates also set `OMNI_KIT_ACCEPT_EULA=YES`, which records acceptance of the NVIDIA Omniverse EULA. The image does not set it: read the EULA, and pass `-e OMNI_KIT_ACCEPT_EULA=YES` yourself if a run asks for it. Whether kit-less runs need the variable was not tested.

### Benchmarks

```bash
docker run --rm --gpus all isaac-net isaac-net-bench list                    # the benchmark suite: tasks, levels, presets
docker run --rm --gpus all isaac-net isaac-net-bench run --task coop_map --level L2-legacy --backend triton \
    --baselines random,heuristic --seeds 0 --device cuda --out /results       # add -v "$PWD/results:/results"
docker run --rm --gpus all isaac-net python benchmarks/bench.py --grid 256x16 --backends graph,triton   # network cost
docker run --rm --gpus all isaac-net:isaaclab env ISAAC_NET_PHYSICS=newton \
    python benchmarks/isaac/bench.py --num_envs 256 --num_robots 16 --level L2-legacy --backend triton
```

`isaac-net-bench` runs network-aware tasks and baselines ([docs/benchmark-suite.md](../docs/benchmark-suite.md)). The cost of the network per control step comes from `benchmarks/bench.py`, or from `benchmarks/uncontended/bench_net.py` with the protocol of [docs/performance.md](../docs/performance.md). Speed numbers measured inside a container match the bare-metal ones only if the GPU is otherwise idle.

## Image size

The image is large because PyTorch's CUDA wheels bundle CUDA, cuDNN, cuBLAS and NCCL. Estimates, not measured: the `core` stage is roughly 6 to 9 GB (the CUDA runtime base image is about 1.5 GB compressed and the torch cu126 stack about 5 GB installed), and `BASE_IMAGE=nvidia/cuda:12.6.3-base-ubuntu22.04` saves 2 to 3 GB of it. The `isaaclab` stage adds a second torch build (cu130) and Isaac Lab's dependencies, for roughly 15 to 20 GB in total. The `isaac-net` package itself is under 1 MB.

## What was verified

The Dockerfile passes `hadolint` (2.15.1) with the apt version-pinning rule DL3008 waived on its line. It has **not** been built or run: no Docker was available where it was written. The Isaac Lab commands are those that passed on the Hazel cluster inside Apptainer ([docs/isaac-lab-linux.md](../docs/isaac-lab-linux.md#results-hazel-l40-dedicated)), and the docs page's own section on Ubuntu without a container notes that this route was not run either. Report a failing build as an issue, with the output of `docker build --progress=plain`.
