"""Wi-Fi level "WIFI": a slot-synchronous, fixed-shape mean-field approximation of the IEEE 802.11 DCF / EDCA
uplink with the engine API of the 5G levels (docs/wifi.md).

    config.py    WifiConfig (NRConfig.wifi): PHY, EDCA, access points, hidden nodes, model knobs
    phy.py       MCS rates, SNR thresholds for rate adaptation, PPDU durations, channel time of one access
    meanfield.py Bianchi-style fixed point on tensors (DomainView / MatrixView) and the scalar reference
    engine.py    WifiNet: the level (reference and graph backends)
    eventsim.py  event-driven CSMA/CA reference simulator (one BSS, CPU) for validation
    validate.py  `python -m isaac_net.core.wifi.validate`: the comparison tables of docs/wifi.md
"""
from .config import AC_NAMES, EDCA, WifiConfig
from .engine import WifiNet, make_wifi
from .phy import AccessTiming, mcs_table

__all__ = ["WifiConfig", "WifiNet", "make_wifi", "AccessTiming", "mcs_table", "EDCA", "AC_NAMES"]
