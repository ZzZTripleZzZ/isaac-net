"""Shared pytest setup.

* Puts prototype/ and prototype/fast/ on sys.path (the package restructure is a later milestone).
* Registers the `gpu` and `slow` markers and skips `gpu` tests when CUDA is unavailable.
"""
import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (os.path.join(ROOT, "prototype"), os.path.join(ROOT, "prototype", "fast"), os.path.dirname(__file__)):
    if p not in sys.path:
        sys.path.insert(0, p)


def pytest_configure(config):
    # The tensors are tiny; intra-op threading only adds overhead (and contends on shared machines).
    torch.set_num_threads(int(os.environ.get("ISAACLAB_NET_TEST_THREADS", "1")))
    config.addinivalue_line("markers", "gpu: needs a CUDA GPU (skipped automatically without one)")
    config.addinivalue_line("markers", "slow: longer runs (deselect with -m 'not slow')")


def pytest_collection_modifyitems(config, items):
    if torch.cuda.is_available():
        return
    skip = pytest.mark.skip(reason="CUDA not available")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def cpu():
    return torch.device("cpu")


@pytest.fixture
def cuda():
    return torch.device("cuda")


@pytest.fixture
def seeded():
    """Seed the global torch RNGs (CPU and CUDA) so each test is reproducible on its own."""
    torch.manual_seed(1234)
    return 1234
