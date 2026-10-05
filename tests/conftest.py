"""Shared pytest setup.

* Makes `engine_api` (tests/) importable; the package itself comes from `pip install -e .` or, in a plain
  checkout, from the repo root that pyproject.toml puts on the path.
* Registers the `gpu`, `slow` and `isaac` markers; skips `gpu` tests when CUDA is unavailable and `isaac` tests
  when Isaac Lab is not installed.
* tests/scripts/ (command-line equivalence scripts) and tests/bridges/ (need an ns-3 build) are not collected.
"""
import importlib.util
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
    torch.set_num_threads(int(os.environ.get("ISAAC_NET_TEST_THREADS", "1")))
    config.addinivalue_line("markers", "gpu: needs a CUDA GPU (skipped automatically without one)")
    config.addinivalue_line("markers", "slow: longer runs (deselect with -m 'not slow')")
    config.addinivalue_line("markers", "isaac: needs Isaac Lab 3.0 and a GPU (skipped automatically without them)")


def pytest_collection_modifyitems(config, items):
    has_isaac = importlib.util.find_spec("isaaclab") is not None
    skip_gpu = pytest.mark.skip(reason="CUDA not available")
    skip_isaac = pytest.mark.skip(reason="Isaac Lab not installed")
    for item in items:
        # get_closest_marker, not `"gpu" in item.keywords`: keywords also hold the names of the directories on the
        # path, so a checkout inside a directory called `isaac` or `gpu` would otherwise skip every test.
        if item.get_closest_marker("isaac") is not None and not has_isaac:
            item.add_marker(skip_isaac)
        elif item.get_closest_marker("gpu") is not None and not torch.cuda.is_available():
            item.add_marker(skip_gpu)


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
