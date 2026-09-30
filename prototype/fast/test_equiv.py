"""Compatibility shim: this script moved to `tests/scripts/test_equiv.py`; the arguments are passed through."""
import os
import runpy
import sys

_target = os.path.abspath(os.path.join(os.path.dirname(__file__), *[".."] * 2, "tests/scripts/test_equiv.py"))
sys.argv[0] = _target
runpy.run_path(_target, run_name="__main__")
