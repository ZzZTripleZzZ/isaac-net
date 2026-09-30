"""MJX backend tests (marker `mjx`): need JAX with CUDA, MuJoCo MJX and MuJoCo Playground, and a GPU.
Skipped automatically without them. JAX must not preallocate the GPU, or torch cannot allocate the engine."""
import importlib.util
import os

import pytest

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


def pytest_configure(config):
    config.addinivalue_line("markers", "mjx: needs JAX (CUDA), MuJoCo MJX and MuJoCo Playground (skipped without them)")


def _mjx_ready():
    if any(importlib.util.find_spec(m) is None for m in ("jax", "mujoco_playground", "torch")):
        return False
    import torch
    if not torch.cuda.is_available():
        return False
    import jax
    return any(d.platform == "gpu" for d in jax.devices())


def pytest_collection_modifyitems(config, items):
    if _mjx_ready():
        return
    skip = pytest.mark.skip(reason="JAX with CUDA / MuJoCo Playground not installed")
    for item in items:
        if "mjx" in item.keywords:
            item.add_marker(skip)
