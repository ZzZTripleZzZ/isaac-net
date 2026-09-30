"""Compatibility shim: this module moved to `isaaclab_net.isaac.netmodule` (see ARCHITECTURE.md)."""
import importlib as _importlib
import os as _os
import sys as _sys

try:
    import isaaclab_net as _pkg  # noqa: F401  (installed with `pip install -e .`)
except ImportError:  # plain checkout: put the repo root on the path
    _sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", "..")))

_sys.modules[__name__] = _importlib.import_module("isaaclab_net.isaac.netmodule")
