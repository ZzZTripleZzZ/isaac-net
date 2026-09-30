"""ns-3 5G-LENA lockstep bridges (TCP / Unix socket / ns3-ai shared memory). Validation only.

    Ns3Lockstep   drives E envs over one or several ns-3 processes (bridges/ns3/lockstep/netslot-bridge.cc)
    Ns3Net        NetBase drop-in (lockstep_net.py) for the prototype FleetEnv / training loop
    Ns3NetModule  the isaac.netmodule NetModule API (netmodule_ns3.py)

Environment: NS3BRIDGE_ROOT = directory with bin/netslot-bridge and the ns-3.48 build (see
bridges/ns3/lockstep/README.md), NS3_TOOLCHAIN_ENV = conda env whose lib/ the ns-3 build links against.
"""
from .core import Ns3Lockstep  # noqa: F401
