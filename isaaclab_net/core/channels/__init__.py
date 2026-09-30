"""Channel models behind radio.RadioMC, selected by NRConfig.channel (see docs/channels.md).

  log_distance  PL = pl_const_db + 10 n log10(d), plus a correlated shadowing field (legacy band or exponential ACF,
                shadow_dcorr_m) and an optional short-range white component (shadow_white_frac); the default is the
                legacy radio bit for bit. Lives in RadioMC.
  tr38901       TR 38.901 RMa / UMa / UMi / InH / InF-SL / DL / SH / DH: LOS probability with a spatially consistent
                LOS state, LOS / NLOS path loss, shadow fading and O2I (models.TR38901Channel).
  radio_map     precomputed gain map [C,H,W] sampled bilinearly (models.RadioMapChannel, radio_map.RadioMap).
  blockage      add-on to every model: robot bodies as spheres on the robot-gNB segment (blockage.blocked_links).
  doppler       per-robot AR(1) fading correlation for the NR engine (doppler.install_per_robot_fading).
"""
from .blockage import blocked_links
from .doppler import install_per_robot_fading, rho_per_ms_from_speed
from .fields import LEGACY_DCORR_M, PlaneWaveField, draw_plane_waves, eval_plane_waves
from .models import RadioMapChannel, TR38901Channel
from .radio_map import SYNTHETIC_MAP, RadioMap, make_synthetic_map, synthetic_gnb_xy

__all__ = ["blocked_links", "install_per_robot_fading", "rho_per_ms_from_speed", "LEGACY_DCORR_M", "PlaneWaveField",
           "draw_plane_waves", "eval_plane_waves", "RadioMapChannel", "TR38901Channel", "SYNTHETIC_MAP", "RadioMap",
           "make_synthetic_map", "synthetic_gnb_xy"]
