"""isaac_net: GPU-batched 5G network simulation for massively parallel robot learning.

    from isaac_net import make_engine, NRConfig, Requests
    from isaac_net import warehouse_private_5g        # scenario presets (core/scenarios.py)

See README.md and ARCHITECTURE.md. The backend-agnostic engines live in `isaac_net.core`; the Isaac Lab
layer in `isaac_net.isaac` imports Isaac Lab only in the files that need it; `isaac_net.bridges` holds the
ns-3 co-simulation bridges (validation only).
"""
from .core import NRConfig, Requests, make_engine
from .core import UnusedFieldsWarning, factory_inf, outdoor_campus, urllc_control, warehouse_private_5g
from .core import LEVELS as _CORE_LEVELS
from .core.engine import WIFI_LEVELS

LEVELS = _CORE_LEVELS + WIFI_LEVELS          # every level make_engine accepts (core.LEVELS: the 5G NR ones)

__version__ = "0.2.1.dev0"
__all__ = ["make_engine", "NRConfig", "Requests", "LEVELS", "__version__", "warehouse_private_5g", "factory_inf",
           "outdoor_campus", "urllc_control", "UnusedFieldsWarning"]
