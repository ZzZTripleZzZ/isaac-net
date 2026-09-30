"""Compatibility shim: this module moved to `isaaclab_net.examples.fleet_task` (see ARCHITECTURE.md).

`import env` returns that module object itself, so attributes set on it (tests patch
module-level functions) are seen by the package code.
"""
import importlib as _importlib
import os as _os
import sys as _sys

try:
    import isaaclab_net as _pkg  # noqa: F401  (installed with `pip install -e .`)
except ImportError:  # plain checkout: put the repo root on the path
    _sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), *[".."] * 1)))

_sys.modules[__name__] = _importlib.import_module("isaaclab_net.examples.fleet_task")
