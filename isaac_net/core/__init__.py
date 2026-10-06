"""Backend-agnostic network engines (pure PyTorch / Triton, no simulator imports).

    from isaac_net.core import make_engine, NRConfig, Requests
    net = make_engine("L2", E, R, "cuda", NRConfig())          # or any level in LEVELS, see engine.py

Modules: config (NRConfig and presets), scenarios (scenario presets: warehouse_private_5g, factory_inf,
outdoor_campus, urllc_control), schema (the step-dict key registry behind every engine's output_schema()), engine (make_engine, NREngine), nr_engine (NRNet), phy, queues,
mac / mac_ul / mac_dl (NR MAC), radio (RadioMC, CellAssociation), traffic (Requests, TrafficModel), proto (prototype levels,
L2-legacy NetSlot and its multi-cell NetSlotMC), levels (fitted surrogates TR / GE / QA / NN and the ORACLE /
NOCOMM bounds), edge (EdgeLoop: edge compute and return path on top of any engine).
"""
from .config import (EdgeConfig, NRConfig, UnusedFieldsWarning, lena_like, lena_match, lena_match_v2, lena_validation,
                     lena_validation_v2, multicell, netslot_compat, oai_like, srsran_like)
from .edge import EdgeLoop
from .engine import BACKENDS, FAST_BACKENDS, LEVELS, SIM_LEVELS, NREngine, make_engine, resolve_backend
from .scenarios import factory_inf, outdoor_campus, urllc_control, warehouse_private_5g
from .schema import STEP_KEYS
from .levels import BOUND_LEVELS, SURROGATE_LEVELS
from .nr_engine import NRNet
from .phy import MCS_TABLES, PHY, lena_tables_path, segment, tbs_38214
from .radio import CellAssociation, Radio, RadioMC
from .traffic import Requests, TrafficModel

__all__ = ["NRConfig", "EdgeConfig", "EdgeLoop", "netslot_compat", "lena_like", "lena_match", "lena_match_v2", "lena_validation",
           "lena_validation_v2", "srsran_like", "oai_like",
           "multicell", "make_engine", "NREngine", "NRNet", "LEVELS", "SIM_LEVELS", "SURROGATE_LEVELS",
           "BOUND_LEVELS", "BACKENDS", "FAST_BACKENDS", "PHY",
           "MCS_TABLES", "tbs_38214", "segment", "lena_tables_path", "RadioMC", "CellAssociation", "Radio",
           "Requests", "TrafficModel", "UnusedFieldsWarning", "resolve_backend", "warehouse_private_5g", "factory_inf",
           "outdoor_campus", "urllc_control", "STEP_KEYS"]

from .background import BackgroundConfig, BackgroundLoop  # noqa: E402
from .energy import EnergyConfig, EnergyLoop  # noqa: E402
from .sharded import ShardedEngine  # noqa: E402

__all__ += ["BackgroundConfig", "BackgroundLoop", "EnergyConfig", "EnergyLoop", "ShardedEngine"]
