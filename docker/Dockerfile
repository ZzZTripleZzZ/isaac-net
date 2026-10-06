# syntax=docker/dockerfile:1
#
# isaac-net on Linux with an NVIDIA GPU: the core package (stage "core", the default target) and, optionally,
# Isaac Lab 3.0 kit-less with the Newton and OV PhysX physics backends (stage "isaaclab"), following
# docs/isaac-lab-linux.md. Build from the repository root; docker/README.md has the commands.
#
#   docker build -f docker/Dockerfile -t isaac-net .                                   # core, isaac-net from PyPI
#   docker build -f docker/Dockerfile --build-arg ISAAC_NET_INSTALL=editable -t isaac-net:dev .   # this checkout
#   docker build -f docker/Dockerfile --target isaaclab -t isaac-net:isaaclab .        # plus Isaac Lab kit-less
#
# The torch wheels bundle their own CUDA libraries, so the base image only needs a glibc of 2.35 or newer (Isaac
# Lab's omniverseclient wheels are manylinux_2_35) and the NVIDIA container environment; the host driver comes in
# through the NVIDIA Container Toolkit (docker run --gpus all).

ARG BASE_IMAGE=nvidia/cuda:12.6.3-runtime-ubuntu22.04

# ---------------------------------------------------------------------------------------------- core
FROM ${BASE_IMAGE} AS core

# ISAAC_NET_INSTALL: "pypi" installs isaac-net[dev] from PyPI; "editable" installs this checkout with -e.
ARG ISAAC_NET_INSTALL=pypi
ARG ISAAC_NET_SPEC=">=0.2.0"
# torch 2.7 or later; the cu126 build is the one the Linux results in docs/performance.md were measured with
# (torch 2.14.0+cu126, Triton 3.8.0 bundled), and it runs on drivers that support CUDA 12.6.
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONNOUSERSITE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_INSTALL_DIR=/opt/uv-python \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:/usr/local/bin:/usr/bin:/bin

# gcc and libc6-dev: Triton compiles a small C launcher per kernel at run time. git: Isaac Lab's git-sourced
# dependencies (isaaclab stage). curl and ca-certificates: downloads. Distribution packages are left unpinned so that
# security updates apply (hadolint DL3008).
# hadolint ignore=DL3008
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl gcc git libc6-dev \
    && rm -rf /var/lib/apt/lists/*

# uv, the installer docs/isaac-lab-linux.md uses (0.12.21 there)
COPY --from=ghcr.io/astral-sh/uv:0.12.21 /uv /uvx /usr/local/bin/

# Python 3.11 (uv-managed CPython, which ships the headers Triton's launcher build needs) and the venv
RUN uv python install 3.11 \
    && uv venv --python 3.11 --seed /opt/venv

RUN uv pip install --no-cache torch --index-url "${TORCH_INDEX_URL}"

# The repository is bind-mounted for this step only. "editable" copies the whole checkout to /src/isaac-net;
# "pypi" copies only the test suite and the benchmark scripts there (they are not in the wheel), so pytest and
# benchmarks/bench.py run against the installed package.
WORKDIR /src/isaac-net
RUN --mount=type=bind,target=/ctx \
    if [ "${ISAAC_NET_INSTALL}" = "editable" ]; then \
        cp -a /ctx/. /src/isaac-net/ && uv pip install --no-cache -e "/src/isaac-net[dev]"; \
    elif [ "${ISAAC_NET_INSTALL}" = "pypi" ]; then \
        uv pip install --no-cache "isaac-net[dev]${ISAAC_NET_SPEC}" \
        && cp -a /ctx/tests /ctx/benchmarks /ctx/pyproject.toml /src/isaac-net/; \
    else \
        echo "ISAAC_NET_INSTALL must be 'pypi' or 'editable', got '${ISAAC_NET_INSTALL}'" >&2; exit 1; \
    fi \
    && python -c "import torch, triton, isaac_net; print('isaac_net', isaac_net.__version__, '| torch', torch.__version__, '| cuda', torch.version.cuda, '| triton', triton.__version__)"

CMD ["bash"]

# ---------------------------------------------------------------------------------------------- isaaclab
# Isaac Lab 3.0 kit-less (no Isaac Sim): docs/isaac-lab-linux.md, "The kit-less recipe" and
# scripts/hazel/install_isaaclab_kitless.sbatch, with the cluster paths replaced by /opt/IsaacLab and /opt/lab.
# The container here supplies glibc 2.35 (Ubuntu 22.04), so no Apptainer is needed. Isaac Lab pins torch 2.12+cu130
# in /opt/lab, which needs NVIDIA driver 580.65.06 or newer on the host; the core venv /opt/venv stays as it is.
FROM core AS isaaclab

ARG ISAACLAB_BRANCH=release/3.0.0
# The commit docs/isaac-lab-linux.md validated; set ISAACLAB_COMMIT="" to build the branch head instead.
ARG ISAACLAB_COMMIT=60e28c1

ENV UV_PROJECT_ENVIRONMENT=/opt/lab

# git-lfs holds only docs media and test images, so the clone skips it
RUN GIT_LFS_SKIP_SMUDGE=1 git -c filter.lfs.smudge= -c filter.lfs.required=false \
        clone --branch "${ISAACLAB_BRANCH}" https://github.com/isaac-sim/IsaacLab.git /opt/IsaacLab \
    && if [ -n "${ISAACLAB_COMMIT}" ]; then git -C /opt/IsaacLab checkout --detach "${ISAACLAB_COMMIT}"; fi \
    && git -C /opt/IsaacLab log -1 --format='IsaacLab %H %cd'

WORKDIR /opt/IsaacLab
RUN uv sync --no-cache --extra rsl-rl --extra ovphysx

# isaac-net into Isaac Lab's venv; --no-deps keeps Isaac Lab's torch build
RUN if [ -f /src/isaac-net/isaac_net/__init__.py ]; then \
        uv pip install --no-cache --python /opt/lab/bin/python --no-deps -e /src/isaac-net; \
    else \
        uv pip install --no-cache --python /opt/lab/bin/python --no-deps "isaac-net$(python -c 'import isaac_net; print("==" + isaac_net.__version__)')"; \
    fi \
    && uv pip install --no-cache --python /opt/lab/bin/python pytest

ENV PATH=/opt/lab/bin:/usr/local/bin:/usr/bin:/bin \
    VIRTUAL_ENV=/opt/lab
WORKDIR /src/isaac-net
CMD ["bash"]
