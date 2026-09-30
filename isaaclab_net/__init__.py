"""isaaclab_net: GPU-batched 5G network simulation for massively parallel robot learning.

    from isaaclab_net import make_engine, NRConfig, Requests

See README.md and ARCHITECTURE.md. The backend-agnostic engines live in `isaaclab_net.core`; the Isaac Lab
layer in `isaaclab_net.isaac` imports Isaac Lab only in the files that need it; `isaaclab_net.bridges` holds the
ns-3 co-simulation bridges (validation only).
"""
from .core import LEVELS, NRConfig, Requests, make_engine

__version__ = "0.1.0"
__all__ = ["make_engine", "NRConfig", "Requests", "LEVELS", "__version__"]
