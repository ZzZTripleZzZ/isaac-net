"""Shared pytest setup.

* Makes `engine_api` (tests/) importable; the package itself comes from `pip install -e .` or, in a plain
  checkout, from the repo root that pyproject.toml puts on the path.
* Registers the `gpu` and `slow` markers and skips `gpu` tests when CUDA is unavailable.
* tests/scripts/ (command-line equivalence scripts) and tests/bridges/ (need an ns-3 build) are not collected.
"""
import os
import sys

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

collect_ignore = ["scripts", "bridges"]


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
