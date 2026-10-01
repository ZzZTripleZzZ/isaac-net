"""Loads the pybind11 module built by bridges/ns3/lockstep/build_pyshm.sh (ns3-ai msg-interface, Python side)."""
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))
_lib = os.environ.get("NS3BRIDGE_PYSHM", os.path.join(os.environ.get("NS3BRIDGE_ROOT", os.path.join(_here, "..", "..")), "bin"))
if _lib not in sys.path:
    sys.path.insert(0, _lib)
import ns3ai_bridge_py as _m  # noqa: E402

_open = {}
_next = [0]


def create(slot, name):
    """slot is ignored: every channel gets a never-used template instance (see ns3ai_bridge_py.cc)."""
    k = _next[0]
    if k >= _m.NSLOT:
        raise RuntimeError(f"ns3-ai: at most {_m.NSLOT} shm channels per Python process lifetime")
    _next[0] += 1
    ch = _m.create(k, name)
    _open[name] = ch
    return ch
