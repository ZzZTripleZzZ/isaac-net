"""NRConfig: the one configuration dataclass shared by every isaac_net module.

It fixes the NR numerology, carrier, TDD pattern and MAC timing (NR engine, nr_engine.py and mac_*.py), the
radio and cell layout (radio.py, the NR engine and the multi-cell legacy engine proto/netsim_mc.py), and the application
fields (frame buffer, timeout, control step, message sizes) that every fidelity level reads. Every derived
quantity is a plain Python value or a small CPU list so the batched engines can precompute their schedules.

Sources
- N_RB per bandwidth and SCS: TS 38.101-1 Table 5.3.2-1 (FR1 maximum transmission bandwidth).
- RBG size P: TS 38.214 Table 5.1.2.2.1-1 (PDSCH) = Table 6.1.2.2.1-1 (PUSCH), configurations 1 and 2.
- Slot duration 1 ms / 2^mu, 14 OFDM symbols per slot (normal CP): TS 38.211 Sec. 4.3.2.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, replace
import math

# TS 38.101-1 Table 5.3.2-1, FR1: {scs_khz: {bw_mhz: N_RB}}
NRB_FR1 = {
    15: {5: 25, 10: 52, 15: 79, 20: 106, 25: 133, 30: 160, 35: 188, 40: 216, 45: 242, 50: 270},
    30: {5: 11, 10: 24, 15: 38, 20: 51, 25: 65, 30: 78, 35: 92, 40: 106, 45: 119, 50: 133,
         60: 162, 70: 189, 80: 217, 90: 245, 100: 273},
    60: {10: 11, 15: 18, 20: 24, 25: 31, 30: 38, 35: 44, 40: 51, 45: 58, 50: 65,
         60: 79, 70: 93, 80: 107, 90: 121, 100: 135},
}

# TS 38.214 Table 5.1.2.2.1-1: (bwp_max_prb, P_config1, P_config2)
RBG_TABLE = [(36, 2, 4), (72, 4, 8), (144, 8, 16), (275, 16, 16)]


def rbg_size_38214(n_prb, config=1):
    """RBG size P in PRBs for a bandwidth part of n_prb PRBs (TS 38.214 Table 5.1.2.2.1-1, configuration 1 or 2)."""
    for hi, p1, p2 in RBG_TABLE:
        if n_prb <= hi:
            return p1 if config == 1 else p2
    raise ValueError(n_prb)


J0_FIRST_ZERO = 2.404825557695773       # first positive zero of the Bessel function J0


def fading_rho_from_speed(speed_mps, carrier_ghz=3.5, anchor_ms=2.5):
    """AR(1) fading correlation per ms for a UE speed: the Jakes correlation J0(2 pi f_D anchor) over one legacy
    UL slot spacing (2.5 ms), spread geometrically over its milliseconds. f_D = v f_c / c. 3 m/s at 3.5 GHz gives
    0.9697 per ms (J0 = 0.926 per 2.5 ms), close to the default 0.93 per 2.5 ms. Monotone non-increasing in speed:
    from the first zero of J0 on (2 pi f_D anchor >= 2.405, about 13.1 m/s at 3.5 GHz) consecutive 2.5 ms samples
    are treated as uncorrelated (rho = 0), and the oscillating tail of J0 past that zero is not used."""
    import torch
    fd = speed_mps * carrier_ghz * 1e9 / 299_792_458.0
    x = 2 * math.pi * fd * anchor_ms * 1e-3
    if x >= J0_FIRST_ZERO:
        return 0.0
    j0 = float(torch.special.bessel_j0(torch.tensor(x, dtype=torch.float64)))
    return min(max(j0, 0.0), 1.0 - 1e-12) ** (1 / anchor_ms)


# NRConfig fields by the part of the package that reads them (see fields_read_by / unused_fields). The prototype
# levels, the surrogates and the bounds read only APP and PROTO (and their own level block); their radio is the fixed
# legacy one. The channel fields (radio group) are read by radio.RadioMC, which L2 and NetSlotMC use; per-robot Doppler
# (fading_doppler) needs the NR engine's fading, so it sits in the NR group. The NR engine (L2) reads APP and rng from
# PROTO (its slots per step come from the numerology and TDD pattern, not proto_ul_slots_per_step).
FIELD_GROUPS = {
    "app": ("control_step_ms", "frame_buffer", "timeout_steps", "msg_sizes", "edge", "seed"),
    "proto": ("rng", "proto_ul_slots_per_step"),
    "l0": ("l0_delay_median_steps", "l0_delay_log_sigma", "l0_loss"),
    "l0dr": ("dr_delay_median_steps", "dr_delay_log_sigma", "dr_loss"),
    "l1": ("l1_eta",),
    "frame": ("mu", "bandwidth_mhz", "n_prb", "rbg_size", "rbg_config", "tdd_pattern", "special_split",
              "special_dl_data", "special_ul_data", "dl_ctrl_symbols", "ul_data_symbols"),
    "nr": ("dmrs_re_per_prb", "overhead_re_per_prb", "k1", "k2", "gnb_proc_slots", "sr_period_slots",
           "sr_grant_delay_slots", "ul_harq_rtt_slots", "cqi_period_slots", "proactive_grant", "proc_offset_ms",
           "n_harq", "max_harq_tx", "harq_combining", "harq_fail", "rlc_retx_slots", "discard", "mcs_table",
           "eff_sinr", "bler_source", "tbs_mode", "lena_ref_sc_per_rb", "bler_target", "ul_mcs_max", "dl_mcs_max",
           "olla", "olla_up_db", "scheduler", "pf_metric", "pf_window", "retx_priority", "ul_power", "phr_cap",
           "pf_update", "pf_avg_idle", "ul_retx_sched", "ul_amc_alloc", "ul_grant_model", "sr_boot_slots",
           "sr_boot_bytes", "bsr_delay_slots", "bsr_hdr_bytes", "bsr_est_hdr_bytes", "rlc_tail_bytes", "rlc_tail_timer_ms",
           "phr_min_db", "fading", "fading_rho_per_ms", "ue_speed_mps", "carrier_ghz", "fading_doppler",
           "doppler_min_speed_mps", "fading_rician", "rician_k_db", "rician_k_from_los", "rician_k_ramp_slots",
           "dl_snr_offset_db",
           "gnb_tx_dbm", "ue_nf_db", "tb_overhead_bytes", "pkt_payload_bytes", "pkt_overhead_bytes", "ul", "dl"),
    "link": ("snr_ref_prbs", "noise_model", "ni_fixed_dbm", "gnb_nf_db", "ue_tx_dbm"),
    "radio": ("pl_const_db", "pathloss_exp", "shadow_sigma_db", "shadow_modes", "shadow_dcorr_m", "shadow_white_frac",
              "shadow_acf", "shadow_white_dcorr_m", "channel", "tr38901_scenario", "tr38901_los", "gnb_height_m",
              "ue_height_m", "o2i_indoor_frac", "o2i_model", "inf_clutter_density", "inf_clutter_size_m",
              "inf_clutter_height_m", "radio_map_path", "blockage", "blockage_radius_m", "blockage_loss_db",
              "los_source", "los_raycast_samples", "los_diffraction", "los_soft", "nlos_extra_loss_db",
              "blockage_model", "blocker_size_m", "blockage_max_db",
              "n_cells", "cell_layout", "cell_positions_m", "cell_isd_m", "cell_center_m", "cell_arena_m"),
    "multicell": ("ul_interference", "li_alpha", "ul_pc", "ul_pc_p0_dbm", "ul_pc_alpha", "a3_offset_db",
                  "a3_hyst_db", "a3_ttt_ms", "ho_interruption_ms", "ho_rlc", "a3_min_target_rsrp_dbm"),
    # read by the NR engine only (NetSlotMC has no downlink and no radio link failure)
    "nr_multicell": ("dl_interference", "rlf", "rlf_qout_db", "rlf_qin_db", "n310", "n311", "t310_ms", "t311_ms",
                     "reest_delay_ms", "rlf_rlc"),
    "access": ("rach", "rach_occasion_slots", "rach_preambles", "rach_rar_window_slots", "rach_msg3_slots",
               "rach_backoff_ms", "rach_max_attempts", "rach_initial", "rach_release_after_ms", "drx",
               "drx_inactivity_ms", "drx_cycle_ms", "drx_on_ms", "drx_short_cycle_ms", "drx_short_cycles",
               "drx_start_offset_ms", "drx_ul_wake"),   # core/access.py, level "L2" only
    "traffic": ("traffic",),
    "wrappers": ("background", "energy"),      # make_engine wrappers (core/background.py, core/energy.py)
    "wifi": ("wifi",),                         # level WIFI (core/wifi), which also reads app, proto (rng) and radio
    "fidelity": ("fidelity",),     # read by core/adaptive.py (make_adaptive), not by make_engine's levels
}

# fields read only by ul_grant_model="bsr" (the 5G-LENA grant pipeline of mac_ul.UlMac)
BSR_PIPELINE_FIELDS = ("sr_boot_slots", "sr_boot_bytes", "bsr_delay_slots", "bsr_hdr_bytes", "bsr_est_hdr_bytes",
                       "rlc_tail_bytes", "rlc_tail_timer_ms")
# the 5G-LENA MAC switches and their default (engine before the switches) values
LENA_MAC_DEFAULTS = {"pf_update": "slot", "pf_avg_idle": "decay", "ul_retx_sched": "ofdma", "ul_amc_alloc": "current",
                     "ul_grant_model": "lumped"}

CHANNELS = ("log_distance", "tr38901", "radio_map")
# channel="tr38901_<scenario>" is shorthand for channel="tr38901", tr38901_scenario=<scenario>
TR38901_SHORT = {"tr38901_rma": "RMa", "tr38901_uma": "UMa", "tr38901_umi": "UMi", "tr38901_inh": "InH",
                 "tr38901_inf_sl": "InF-SL", "tr38901_inf_dl": "InF-DL", "tr38901_inf_sh": "InF-SH",
                 "tr38901_inf_dh": "InF-DH"}


# fields read only under some switches (fields_read_by drops them when the switch is off)
DL_ONLY_FIELDS = ("k1", "cqi_period_slots", "dl_mcs_max", "dl_snr_offset_db", "dl_interference", "gnb_tx_dbm",
                  "ue_nf_db")                                 # NR engine downlink (dl=True)
HANDOVER_FIELDS = ("a3_offset_db", "a3_hyst_db", "a3_ttt_ms", "ho_interruption_ms", "ho_rlc",
                   "a3_min_target_rsrp_dbm", "rlf")                           # n_cells > 1
RLF_FIELDS = ("rlf_qout_db", "rlf_qin_db", "n310", "n311", "t310_ms", "t311_ms", "reest_delay_ms", "rlf_rlc")  # rlf
# the A3 admission floor is an RSRP: converting it reads the gNB EPRE (gnb_tx_dbm over 12 nprb subcarriers)
A3_FLOOR_FIELDS = ("gnb_tx_dbm", "mu", "bandwidth_mhz", "n_prb")             # a3_min_target_rsrp_dbm set
INTERFERENCE_FIELDS = ("ul_interference", "dl_interference", "li_alpha")    # n_cells > 1 and noise_model="thermal"
UL_PC_FIELDS = ("ul_pc_p0_dbm", "ul_pc_alpha")               # ul_pc_on
TR38901_FIELDS = ("tr38901_scenario", "tr38901_los", "o2i_indoor_frac", "o2i_model", "inf_clutter_density",
                  "inf_clutter_size_m", "inf_clutter_height_m")             # channel="tr38901"
BLOCKAGE_FIELDS = ("blockage_radius_m", "blockage_loss_db", "blockage_model", "blocker_size_m",
                   "blockage_max_db")                                       # blockage=True
SPHERE_FIELDS = ("blockage_radius_m", "blockage_loss_db")                   # blockage_model="sphere"
LOS_SOURCES = ("stochastic", "map", "raycast", "callback")
FADING_FIELDS = ("fading_rho_per_ms", "ue_speed_mps", "fading_doppler", "doppler_min_speed_mps")   # fading=True
RICIAN_FIELDS = ("rician_k_db", "rician_k_from_los", "rician_k_ramp_slots")   # fading and fading_rician
RACH_FIELDS = tuple(f for f in FIELD_GROUPS["access"] if f.startswith("rach_"))     # rach=True
DRX_FIELDS = tuple(f for f in FIELD_GROUPS["access"] if f.startswith("drx_"))       # drx=True
# frame fields the multi-cell legacy engine (NetSlotMC) reads: only through ul_slot_ms, which converts the A3
# time-to-trigger and the handover interruption to UL slots (ttt_slots, ho_int_slots), so only with n_cells > 1
NETSLOTMC_FRAME_FIELDS = ("mu", "tdd_pattern", "special_split", "special_ul_data", "ul_data_symbols")


def _obstacle_unread(cfg):
    """Obstacle fields (channels/los.py, channels/blockage.py) that cfg's switches leave unread."""
    off = set()
    if cfg.los_source != "raycast":
        off |= {"los_raycast_samples", "los_diffraction"}
    if not (cfg.channel == "tr38901" and cfg.los_source == "stochastic"):
        off.add("los_soft")
    if not (cfg.channel == "log_distance" and cfg.los_source != "stochastic"):
        off.add("nlos_extra_loss_db")
    if cfg.los_source != "stochastic":
        off.add("tr38901_los")                     # the geometric source decides the state
    if cfg.blockage_model != "sphere":
        off |= set(SPHERE_FIELDS)
    if cfg.blockage_model != "screen":
        off.add("blocker_size_m")
    if cfg.blockage_model == "sphere":
        off.add("blockage_max_db")
    return off


def _switch_unread(cfg, nr):
    """Fields of the radio / cell / NR groups that cfg's switches leave unread (nr: the NR engine, else NetSlotMC)."""
    off = set()
    if nr and not cfg.dl:
        off |= set(DL_ONLY_FIELDS)
    if cfg.n_cells == 1:
        off |= set(HANDOVER_FIELDS) | set(INTERFERENCE_FIELDS)
    elif cfg.noise_model != "thermal":
        off |= set(INTERFERENCE_FIELDS)
    if cfg.n_cells == 1 or not cfg.rlf:
        off |= set(RLF_FIELDS)
    if not cfg.ul_pc_on:
        off |= set(UL_PC_FIELDS)
    if cfg.noise_model == "thermal":
        off.add("ni_fixed_dbm")
    else:
        off |= {"gnb_nf_db", "ue_nf_db"}
    if cfg.channel != "tr38901":
        off |= set(TR38901_FIELDS)
    if cfg.channel != "radio_map" and cfg.los_source not in ("map", "raycast"):
        off.add("radio_map_path")              # map / raycast LOS sources read los_prob / obstacle_z from the map
    if not cfg.blockage:
        off |= set(BLOCKAGE_FIELDS)
    off |= _obstacle_unread(cfg)
    if nr:
        if not cfg.fading:
            off |= set(FADING_FIELDS) | {"fading_rician"} | set(RICIAN_FIELDS)
        elif cfg.fading_doppler != "per_robot":
            off.add("doppler_min_speed_mps")
        if cfg.fading and not cfg.fading_rician:
            off |= set(RICIAN_FIELDS)
        elif cfg.fading and cfg.rician_k_db is not None:      # fixed K: no LOS state, no ramp
            off |= {"rician_k_from_los", "rician_k_ramp_slots"}
        elif cfg.fading and not cfg.rician_k_from_los:        # K = 0 (Rayleigh)
            off.add("rician_k_ramp_slots")
        if cfg.rician_mode == "los":                          # mu_K / sigma_K of the scenario, on any channel
            off.discard("tr38901_scenario")
        if cfg.tbs_mode != "lena":
            off.add("lena_ref_sc_per_rb")
        if not cfg.rach:
            off |= set(RACH_FIELDS)
        if not cfg.drx:
            off |= set(DRX_FIELDS)
        elif cfg.drx_short_cycle_ms is None:
            off.add("drx_short_cycles")
    return off


def fields_read_by(level, cfg=None):
    """NRConfig fields that the engine make_engine(level, ..., cfg) actually reads. Without cfg: the fields a level
    may read under some configuration (whole groups); with cfg: only those its switches make it read (DL fields only
    with dl=True, handover and interference fields only with several cells, the noise fields of the noise model, the
    fields of the selected channel model, blockage and fading fields only when on, lena_ref_sc_per_rb only with
    tbs_mode="lena")."""
    groups = {"L0": ("app", "proto", "l0"), "L0DR": ("app", "proto", "l0dr"), "L1": ("app", "proto", "l1"),
              "L2": ("app", "frame", "nr", "link", "radio", "multicell", "nr_multicell", "traffic", "access")}.get(
        level, ("app", "proto"))
    netslot_mc = level == "L2-legacy" and cfg is not None and not cfg.is_legacy_cell()
    if netslot_mc:
        groups = ("app", "proto", "link", "radio", "multicell")        # NetSlotMC
    read = {f for g in groups for f in FIELD_GROUPS[g]} | set(FIELD_GROUPS["wrappers"])
    if netslot_mc and cfg.n_cells > 1:
        read |= set(NETSLOTMC_FRAME_FIELDS)
    if level == "L2":
        read.add("rng")                # engine RNG of the NR engine (nr_rng.py)
        if cfg is not None:            # the UL grant model reads either the lumped SR delay or the BSR pipeline
            read -= set(BSR_PIPELINE_FIELDS) if cfg.ul_grant_model == "lumped" else {"sr_grant_delay_slots",
                                                                                     "proactive_grant"}
    if cfg is not None and (level == "L2" or netslot_mc):
        read -= _switch_unread(cfg, nr=level == "L2")
        if cfg.n_cells > 1 and cfg.a3_min_target_rsrp_dbm is not None:
            read |= set(A3_FLOOR_FIELDS)
    if cfg is not None and cfg.traffic is not None and not any(m.generates for m in cfg.traffic):
        read.add("traffic")            # policy() only: the submit() path every level has
    return read


@dataclass
class EdgeConfig:
    """Edge-computing loop (core/edge.py, EdgeLoop): the edge server that processes delivered uplink messages and
    the return path of the result (command) to the robot. Times in ms; EdgeLoop converts them to control steps.

    Edge compute stage (one server pool per env, shared by its R robots):
      servers_per_env    servers (FIFO) or total service capacity (processor sharing)
      discipline         "fifo": first come first served, non-preemptive, one message per server;
                         "ps": processor sharing, each of n messages in service gets min(1, servers / n)
      service_dist       "deterministic" or "exponential" (mean service_ms)
      service_ms         mean service time per message; a tuple gives one value per message class (1, 2, ...)
      queue_cap          waiting room per env; the edge holds at most queue_cap + servers_per_env messages and an
                         arriving message that finds it full is dropped
      deadline_ms        per-message deadline from capture; a message that has not started service by then (FIFO)
                         or not finished (PS) is dropped. None = no deadline
      max_events_per_step  event budget of the exact event loop per control step (None: 2 R + 2 servers + 8); an env
                         that runs out continues exactly from where it stopped at the next step (edge_lag)
    Return path (result -> robot):
      return_path        "instant" (at edge completion), "delay" (ret_fixed_ms + U(0, ret_jitter_ms) + transmission
                         time of cmd_bytes at a rate from the robot's SINR), or "nr_dl" (the result is submitted as a
                         downlink message of cmd_bytes to the NR engine, level L2 with dl=True, and the command
                         arrives when the real DL scheduler delivers it)
      ret_rate_eta, ret_share, ret_snr_offset_db  "delay" rate = eta * share * bandwidth * log2(1 + SNR_dl),
                         SNR_dl = step SINR + offset; share None = 1 / R (the robots split the downlink)
      ret_inflight       commands in flight per robot; a new command replaces the oldest when all are busy
    """
    servers_per_env: int = 1
    discipline: str = "fifo"
    service_dist: str = "deterministic"
    service_ms: float | tuple = 10.0
    queue_cap: int = 32
    deadline_ms: float | None = None
    max_events_per_step: int | None = None
    return_path: str = "instant"
    ret_fixed_ms: float = 1.0
    ret_jitter_ms: float = 0.0
    cmd_bytes: int = 100
    ret_rate_eta: float = 0.75
    ret_share: float | None = None
    ret_snr_offset_db: float = 10.0
    ret_inflight: int = 4

    def __post_init__(self):
        assert self.servers_per_env >= 1 and self.queue_cap >= 0
        assert self.discipline in ("fifo", "ps"), "discipline: 'fifo' or 'ps'"
        assert self.service_dist in ("deterministic", "exponential")
        svc = self.service_ms if isinstance(self.service_ms, (tuple, list)) else (self.service_ms,)
        assert len(svc) >= 1 and all(s > 0 for s in svc), "service_ms must be > 0"
        assert self.deadline_ms is None or self.deadline_ms > 0
        assert self.return_path in ("instant", "delay", "nr_dl")
        assert self.ret_fixed_ms >= 0 and self.ret_jitter_ms >= 0 and self.cmd_bytes > 0 and self.ret_inflight >= 1
        assert self.ret_share is None or 0 < self.ret_share <= 1

    def service_table(self, n_classes):
        """Mean service time (ms) per message class 1..n_classes."""
        if isinstance(self.service_ms, (tuple, list)):
            svc = tuple(float(s) for s in self.service_ms)
            assert len(svc) >= n_classes, f"service_ms has {len(svc)} classes, the config has {n_classes}"
            return svc[:n_classes]
        return (float(self.service_ms),) * n_classes


@dataclass
class NRConfig:
    """The one configuration of every isaac_net module: numerology, carrier and TDD pattern, MAC timing, HARQ
    and RLC, PHY tables and link adaptation, radio and cell layout, and the application fields.

    Every field has a default, so NRConfig() is a complete configuration (one cell, DDDSU at 30 kHz, 20 MHz).
    The presets (netslot_compat, lena_like, lena_validation, srsran_like, oai_like, multicell) return an
    NRConfig and take keyword overrides; cfg.with_(**changes) returns a modified copy. Derived quantities
    (nprb, rbg, slots_per_step, ...) are properties computed from the fields. Each level reads only some fields
    (fields_read_by); unused_fields(level) lists the non-default fields a level ignores, and
    make_engine(..., strict=True) raises on them.
    """
    # ---- numerology and carrier ----
    mu: int = 1                          # SCS = 15 * 2^mu kHz, mu in {0, 1, 2}
    bandwidth_mhz: int = 20
    n_prb: int | None = None             # override of the 38.101 N_RB (e.g. 50 to match LENA RbOverhead=0.1)
    rbg_size: int | None = None          # override of the 38.214 RBG size (subband = RBG)
    rbg_config: int = 1                  # 38.214 RBG configuration 1 or 2
    # ---- TDD ----
    tdd_pattern: str = "DDDSU"           # one letter per slot, D / S / U, repeated
    special_split: tuple = (10, 2, 2)    # S slot symbols (DL, guard, UL)
    special_dl_data: bool = True         # S slot DL symbols carry PDSCH
    special_ul_data: bool = False        # S slot UL symbols carry PUSCH (LENA: no)
    dl_ctrl_symbols: int = 1             # PDCCH symbols at the start of D / S slots
    ul_data_symbols: int = 12            # PUSCH symbols in a U slot (14 - PUCCH - SRS)
    dmrs_re_per_prb: int = 12            # N_DMRS^PRB in the 38.214 TBS formula (1 DMRS symbol, type 1)
    overhead_re_per_prb: int = 0         # N_oh^PRB (xOverhead)
    # ---- MAC timing (slots) ----
    k1: int = 2                          # PDSCH -> HARQ-ACK (minimum; ACK goes to next UL-capable slot)
    k2: int = 2                          # UL DCI -> PUSCH
    gnb_proc_slots: int = 1              # gNB decode / scheduling latency
    sr_period_slots: int = 5             # SR opportunity: first UL-capable slot of every period window
    sr_grant_delay_slots: int | None = None   # SR -> first PUSCH; default gnb_proc + k2
    ul_harq_rtt_slots: int | None = None      # PUSCH -> earliest retx; default gnb_proc + k2
    cqi_period_slots: int = 10           # DL CQI report period (reports ride the next UL-capable slot)
    proactive_grant: str = "off"         # "off" | "every_ul_slot" | "per_period": UL grant without SR at the first
                                         # UL data slot of each TDD period (OAI-like); unused grants are not modelled
    proc_offset_ms: float = 0.0          # fixed stack processing delay added to every frame completion
    # ---- HARQ / RLC ----
    n_harq: int = 16                     # HARQ processes per UE per direction (1 = head-of-line mode)
    max_harq_tx: int = 4                 # transmissions per TB including the first (LENA maxHarqReTx=3)
    harq_combining: str = "cc"           # "cc" (chase: add linear effective SINR), "ir_lena"
                                         # (EESM over all tx + equivalent MCS at R/ntx, 5G-LENA NrEesmIr), "none"
    harq_fail: str = "rlc_am"            # "rlc_am": resend after rlc_retx_slots, "drop": RLC UM loss
    rlc_retx_slots: int = 50
    discard: str = "purge"               # "purge": drop queued frames older than timeout (legacy NetSlot);
                                         # "none": never purge; "pdcp_arrival": drop ARRIVING frames while the
                                         # head-of-line frame is older than timeout (5G-LENA UM PDCP discard)
    # ---- PHY abstraction / link adaptation ----
    mcs_table: int = 1                   # 38.214 Table 5.1.3.1-1 (64QAM) or -2 (256QAM)
    eff_sinr: str = "eesm"               # "eesm" or "mean_db" (legacy NetSlot)
    bler_source: str = "pdsch"           # "pdsch": Sionna PDSCH curves both directions; "lena": 5G-LENA EESM
                                         # tables (local extraction); "sionna_label": audit only
    tbs_mode: str = "38214"              # "38214" exact TBS, or "lena": floor((12-ref_sc)*RB*sym*Qm*R/8) - CRC
    lena_ref_sc_per_rb: int = 1          # NrAmc NumRefScPerRb (only for tbs_mode="lena")
    bler_target: float = 0.1
    ul_mcs_max: int | None = None        # MCS index cap (public-data calibration: bench open-source UL SE ~2.4-3)
    dl_mcs_max: int | None = None
    olla: bool = True
    olla_up_db: float = 0.05             # down step = up * (1 - target) / target
    scheduler: str = "pf"                # "pf" proportional fair (metric from pf_metric), "pf_wideband" (= pf with
                                         # pf_metric="wideband"), "maxci" (max rate, no fairness), "rr" (round robin:
                                         # the robot served longest ago first, channel-blind)
    pf_metric: str = "subband"           # "subband" (frequency-selective) or "wideband" (5G-LENA OFDMA PF)
    pf_window: float = 100.0             # EWMA window in scheduled slots of that direction
    retx_priority: bool = True
    # ---- 5G-LENA MAC behavior under load (docs/fidelity-load-gap.md); defaults = the engine before these switches ----
    pf_update: str = "slot"              # "slot": PF metric fixed within a slot, average updated with the served
                                         # bytes; "rbg": 5G-LENA OFDMA PF, the winner's average is updated with its
                                         # granted bytes after every RBG (RBGs spread over backlogged UEs)
    pf_avg_idle: str = "decay"           # "decay": every UE's PF average moves every data slot; "freeze": only UEs
                                         # with new data to schedule update it (5G-LENA active list)
    ul_retx_sched: str = "ofdma"         # "ofdma": UL retx share the slot with new data; "tdma": one UL retx per
                                         # slot (oldest NACK first) on every data symbol, nothing else in its slot
    ul_amc_alloc: str = "current"        # UL MCS for the PRBs of the current grant, or "previous": for the PRBs
                                         # of the UE's previous PUSCH (5G-LENA NrAmc), TB sized for the current one
    ul_grant_model: str = "lumped"       # "lumped": SR -> grant after sr_grant_delay_slots, BSR = exact buffer;
                                         # "bsr": 5G-LENA pipeline (SR bootstrap grant, 38.321 short-BSR levels
                                         # rounded up with a report delay, padded over-grants, RLC tail stall);
                                         # makes sr_grant_delay_slots and proactive_grant unused
    sr_boot_slots: int = 6               # bsr: SR -> first PUSCH slot the bootstrap grant may use
    sr_boot_bytes: int = 17              # bsr: bootstrap grant after an SR (5G-LENA srGrantSize 12 + 2 RLC + 3 MAC)
    bsr_delay_slots: int = 10            # bsr: PUSCH carrying a BSR -> first PUSCH scheduled with it (2 UL slots)
    bsr_hdr_bytes: int = 8               # bsr: UE adds short BSR (5) + MAC subheader (3) to the reported buffer
    bsr_est_hdr_bytes: int = 5           # bsr: scheduler adds RLC (2) + MAC subheader (3) to the reported buffer
    rlc_tail_bytes: int = 16             # bsr: RLC header residue the draining TB leaves behind (0 = no tail stall)
    rlc_tail_timer_ms: float = 10.0      # bsr: RLC buffer-status timer that reports the residue
    # ---- radio model inside the engine ----
    snr_ref_prbs: int = 10               # input UL SNR = full UE power over this many PRBs
    ul_power: str = "allocated"          # "allocated": UE power split over its grant; "whole_band": fixed PSD
    phr_cap: bool = True                 # UL power-headroom cap on RBGs per UE (NetSlot; 5G-LENA has none)
    phr_min_db: float = 3.0              # cap rule: per-PRB SNR (at the snr_ref_prbs split) >= this
    fading: bool = True
    fading_rho_per_ms: float = 0.93 ** (1 / 2.5)   # AR(1) per-subband Rayleigh; 0.93 per 2.5 ms (35 Hz)
    ue_speed_mps: float | None = None    # set: fading_rho_per_ms = J0(2 pi f_D 2.5 ms)^(1/2.5), f_D = v f_c / c
    carrier_ghz: float = 3.5             # carrier for the Doppler of ue_speed_mps (3 m/s at 3.5 GHz = 35 Hz) and of
                                         # the tr38901 path loss
    fading_doppler: str = "global"       # "global": one rho for all robots; "per_robot": rho from each robot's own
                                         # speed (velocity input or consecutive poses), NR engine with pose input
    doppler_min_speed_mps: float = 0.0   # per_robot: speed floor (0 = a still robot's fading is frozen)
    fading_rician: bool = False          # Rician fast fading (NR engine, docs/channels.md): per subband
                                         # |sqrt(K/(K+1)) e^{j phi} + sqrt(1/(K+1)) h|^2, h the AR(1) Rayleigh state
    rician_k_db: float | None = None     # fading_rician: fixed K (dB) for every link; None = K from the LOS state
    rician_k_from_los: bool = True       # K log-normal per link (TR 38.901 Table 7.5-6, tr38901_scenario) where the
                                         # radio reports LOS and no blockage, else K = 0; False (or no LOS state): K = 0
    rician_k_ramp_slots: int = 4         # K moves linearly to a new target over this many slots (0 = at once)
    dl_snr_offset_db: float = 10.0       # step(): default DL per-PRB SNR = UL input SNR + offset
    # ---- link budget for step_rx() (shared with multicell/: pathgain + interference in, SINR out) ----
    noise_model: str = "fixed"           # "fixed": ni_fixed_dbm over snr_ref_prbs PRBs; "thermal": -174 dBm/Hz + NF
    ni_fixed_dbm: float = -90.0          # noise + interference floor over snr_ref_prbs PRBs (legacy NetSlot)
    gnb_nf_db: float = 5.0
    ue_nf_db: float = 9.0
    ue_tx_dbm: float = 23.0
    gnb_tx_dbm: float = 43.0             # DL total power, spread evenly over all PRBs
    # ---- overheads ----
    tb_overhead_bytes: int = 0           # MAC subheaders / MAC CEs / RLC header per TB
    pkt_payload_bytes: int = 1400        # application frame split into packets of this payload
    pkt_overhead_bytes: int = 0          # per-packet bytes on the air (frame header, UDP/IP, PDCP, RLC SDU hdr)
    # ---- radio (radio.py: large-scale gain per robot-cell link; the MAC takes SNR / pathgain inputs) ----
    pl_const_db: float = 40.0            # PL = pl_const_db + 10 pathloss_exp log10(d), d >= 1 m
    pathloss_exp: float = 3.5
    shadow_sigma_db: float = 6.0         # one spatially correlated field per (env, cell), sum of plane waves
    shadow_modes: int = 8                # plane waves per field (every channel model's fields)
    shadow_dcorr_m: float = 10.3         # 1/e decorrelation distance of the log_distance shadowing (10.3 = the legacy
                                         # 20-60 m wavelength band, bitwise; POWDER fit: 20-40 m)
    shadow_acf: str = "sos"              # "sos": legacy band scaled to shadow_dcorr_m; "exp": exp(-r / shadow_dcorr_m)
    shadow_white_frac: float = 0.0       # share of the shadowing variance in a component decorrelated within
    shadow_white_dcorr_m: float = 1.0    # this distance (exp ACF); POWDER: about 0.5 of the variance
    # ---- channel model (radio.py; see docs/channels.md) ----
    channel: str = "log_distance"        # "log_distance" (above) | "tr38901" | "radio_map"; "tr38901_inf_sh" etc.
                                         # is shorthand for channel="tr38901", tr38901_scenario="InF-SH"
    tr38901_scenario: str = "InF-SH"     # RMa, UMa, UMi, InH (mixed office), InF-SL, InF-DL, InF-SH, InF-DH
    tr38901_los: str = "stochastic"      # "stochastic" (Table 7.4.2-1, spatially consistent) | "los" | "nlos"
    gnb_height_m: float | None = None    # None: the scenario default (tr38901), else 2-D geometry (other models)
    ue_height_m: float = 1.5             # robot antenna height (tr38901 and blockage)
    o2i_indoor_frac: float = 0.0         # tr38901 UMa / UMi / RMa: share of robots indoors (O2I penetration)
    o2i_model: str = "low"               # "low" | "high" (Table 7.4.3-2; RMa allows only "low")
    inf_clutter_density: float | None = None   # InF r, d_clutter, h_c; None = Table 7.8-7 values of the scenario
    inf_clutter_size_m: float | None = None
    inf_clutter_height_m: float | None = None
    radio_map_path: str | None = None    # channel="radio_map": .npz / .pt with gain_db [C,H,W] and bounds (x0,y0,x1,y1);
                                         # "synthetic" = the shipped 2-cell test map
    blockage: bool = False               # add-on to every channel: other robots are spheres that block the link
    blockage_radius_m: float = 0.3
    blockage_loss_db: float = 20.0       # extra loss on a link blocked by at least one robot body
    # ---- obstacles and NLOS (channels/los.py, channels/blockage.py; see docs/obstacles.md) ----
    los_source: str = "stochastic"       # LOS state: "stochastic" (tr38901 Pr_LOS, today) | "map" (radio map los_prob)
                                         # | "raycast" (radio map obstacle_z height map) | "callback" (a blocked_fn,
                                         # RadioMC.set_los_callback); map / raycast read radio_map_path
    los_raycast_samples: int = 32        # raycast: samples along each robot-gNB segment
    los_diffraction: bool = False        # raycast: ITU-R P.526 knife-edge loss from the clearance (LOS -> NLOS ramp)
    los_soft: bool = False               # tr38901 with los_source="stochastic": Sec. 7.6.3.3 soft LOS blend
    nlos_extra_loss_db: float = 0.0      # log_distance with a geometric LOS state: extra loss on NLOS links
    blockage_model: str = "sphere"       # blockage=True: "sphere" (above) | "screen" (38.901 model B, robots and
                                         # step(blockers=) as screens) | "stochastic" (38.901 model A)
    blocker_size_m: tuple = ((0.6, 1.5), (0.3, 1.7), (4.8, 1.4))   # screen (w, h) m per class: robot, human,
                                         # vehicle (Table 7.6.4.2-5 for human and vehicle)
    blockage_max_db: float = 40.0        # screen / stochastic: cap of the summed blockage loss per link
    # ---- cells: layout, uplink interference and power control, association and handover ----
    # Default: one gNB at the origin with the fixed noise floor, which is the legacy NetSlot geometry.
    # multicell() gives the multi-cell preset. n_cells > 1 runs on level "L2" (NR engine, UL and DL
    # interference) and on "L2-legacy" (NetSlotMC, UL only). TTT and interruption are in ms: NetSlotMC counts
    # them in UL slots (ttt_slots, ho_int_slots), the NR engine in slots of slot_ms.
    n_cells: int = 1                     # 1..7
    cell_layout: str = "custom"          # "hex" | "grid" | "custom"
    cell_positions_m: tuple = ((0.0, 0.0),)   # custom: one (x, y) per cell, env-local metres
    cell_isd_m: float = 100.0            # hex inter-site distance (>= 100 m: see multicell())
    cell_center_m: tuple = (75.0, 75.0)  # hex: cluster centroid
    cell_arena_m: float = 150.0          # grid: square arena tiled by ceil(sqrt(C)) columns
    ul_interference: bool = True         # thermal noise only: add same-slot other-cell UL interference
    dl_interference: bool = True         # thermal noise only: same for the DL (NR engine)
    li_alpha: float = 1.0                # link adaptation uses an EWMA of the measured N+I; 1 = previous slot
    ul_pc: bool | None = None            # UL open-loop fractional power control P = min(Pmax - split, P0 + alpha PL)
                                         # per subband; None = ON when n_cells > 1, OFF for one cell (legacy)
    ul_pc_p0_dbm: float = -88.0          # per subband (10 PRB); about 15 dB SNR per subband at alpha = 1
    ul_pc_alpha: float = 1.0
    a3_offset_db: float = 0.0            # A3: neighbour > serving + offset + hysteresis ...
    a3_hyst_db: float = 3.0
    a3_ttt_ms: float = 300.0             # ... held for the time-to-trigger
    ho_interruption_ms: float = 40.0     # robot cannot be scheduled after a handover
    ho_rlc: str = "carry"                # "carry": lossless, queued frames continue at the target; "flush": dropped
    a3_min_target_rsrp_dbm: float | None = None   # A3 target admission (5G-LENA MinTargetRsrpDbm): a neighbour whose
                                         # RSRP (gNB EPRE + path gain, dBm per subcarrier) is below it is not a
                                         # target; also the floor of the RLF cell search. None = no floor
    # radio link failure (NR engine, n_cells > 1; docs/multicell.md): serving-link SINR (the step's sinr_db) against
    # Qout / Qin once per control step; n310 out-of-sync in a row start T310, n311 in-sync in a row stop it; at T310
    # expiry the robot is in RLF (unschedulable) and re-establishes at the strongest suitable cell (SINR >= Qin, RSRP
    # >= a3_min_target_rsrp_dbm) reest_delay_ms after that cell is found, or goes idle (queue dropped) if T311 expires
    # first
    rlf: bool = False
    rlf_qout_db: float = -8.0            # out-of-sync SINR (TS 38.133 Qout: 10% hypothetical PDCCH BLER, about -8 dB)
    rlf_qin_db: float = -6.0             # in-sync SINR (TS 38.133 Qin: 2% hypothetical PDCCH BLER, about -6 dB)
    n310: int = 1                        # TS 38.331 N310 / N311 (network-configured; n1 is the smallest value)
    n311: int = 1
    t310_ms: float = 1000.0              # TS 38.331 T310 (network-configured, ms0 to ms6000)
    t311_ms: float = 3000.0              # TS 38.331 T311 (network-configured, ms1000 to ms30000)
    reest_delay_ms: float = 40.0         # cell found -> service at that cell (random access and RRC re-establishment)
    rlf_rlc: str | None = None           # on RLF: "carry" (PDCP AM recovery) or "flush"; None = follow ho_rlc
    # ---- access state machine (core/access.py, level "L2"): RACH / connection setup and connected-mode DRX ----
    # Off by default (every robot always connected and awake, the engine before these fields, bitwise).
    rach: bool = False                   # contention-based random access (TS 38.321 Sec. 5.1) before a robot is served
    rach_occasion_slots: int = 20        # RACH occasion (RO) period; ROs at the first UL-capable slot of every period
                                         # window (a multiple of the TDD period; 20 slots = 10 ms at mu = 1)
    rach_preambles: int = 64             # contention-based preambles per cell and RO (TS 38.211: 64 per cell)
    rach_rar_window_slots: int = 10      # preamble -> RAR (the RAR is taken at the end of ra-ResponseWindow)
    rach_msg3_slots: int = 10            # RAR -> contention resolution (Msg3 on PUSCH, Msg4); then the robot is served
    rach_backoff_ms: float = 20.0        # backoff after a collision: uniform in [0, rach_backoff_ms] (38.321 Sec.
                                         # 5.1.4, backoff indicator; 20 ms = BI index 2 of Table 7.2-1)
    rach_max_attempts: int = 10          # preambleTransMax; a procedure that reaches it fails and restarts
    rach_initial: str = "connected"      # state after a reset: "connected" (as without RACH) or "idle" (power-on)
    rach_release_after_ms: float | None = None   # RRC release to idle after this inactivity; None = never
    drx: bool = False                    # connected-mode DRX (TS 38.321 Sec. 5.7)
    drx_inactivity_ms: float = 100.0     # drx-InactivityTimer: awake this long after the last scheduling activity
    drx_cycle_ms: float = 160.0          # drx-LongCycle
    drx_on_ms: float = 10.0              # drx-onDurationTimer
    drx_short_cycle_ms: float | None = None   # drx-ShortCycle; None = no short cycle
    drx_short_cycles: int = 2            # drx-ShortCycleTimer in short cycles
    drx_start_offset_ms: float = 0.0     # drx-StartOffset (on-durations start at slot offset mod cycle)
    drx_ul_wake: str = "sr"              # UL data while dormant: "sr" wakes the robot at once (a pending SR is Active
                                         # Time, 38.321 Sec. 5.7), "on_duration" waits for the next on-duration
    # ---- directions and application ----
    ul: bool = True
    dl: bool = False
    control_step_ms: float = 100.0
    frame_buffer: int = 16               # frames per robot per direction
    timeout_steps: int = 20              # application deadline in control steps
    msg_sizes: tuple = (4000.0, 30000.0) # bytes of traffic classes 1, 2, ... (Requests.send = class index)
    traffic: tuple | None = None         # traffic models run inside the step (traffic.TrafficModel), level L2 only;
                                         # None = policy messages only. A model or a list is turned into a tuple
    # ---- delay levels (make_engine fills the level params from these when params is None) ----
    l0_delay_median_steps: float = 0.05  # L0: i.i.d. lognormal delay, median in control steps
    l0_delay_log_sigma: float = 0.5      # L0: sigma of the log delay
    l0_loss: float = 0.0                 # L0: i.i.d. loss probability
    dr_delay_median_steps: tuple = (0.05, 10.0)   # L0DR: per-env median delay range, log-uniform at every reset
    dr_delay_log_sigma: tuple = (0.2, 1.2)        # L0DR: per-env sigma of the log delay, uniform
    dr_loss: tuple = (0.0, 0.2)                   # L0DR: per-env loss probability, uniform
    l1_eta: float = 0.9                  # L1: goodput factor on 0.75 log2(1 + SNR) (all backends, triton included)
    # ---- randomness and prototype timing ----
    seed: int | None = None              # engine seed (make_engine(seed=...) overrides it); None = drawn from the
                                         # global torch RNG at construction
    rng: str = "engine"                  # every level (the NR engine too): "engine" = every draw from the
                                         # engine's counter-based streams keyed by (seed, env, episode, call), so a
                                         # policy's use of the global RNG never changes the network; "global" = the
                                         # earlier behavior (stepping draws from the global torch RNG)
    proto_ul_slots_per_step: int | None = None   # prototype levels (L1, L2-legacy, QA): UL slots per control step;
                                                 # None = control_step_ms / 2.5 ms (the legacy DDDSU UL spacing)
    # ---- edge-computing loop (core/edge.py): make_engine wraps any level in EdgeLoop when set ----
    edge: EdgeConfig | None = None
    background: object = None            # core.background.BackgroundConfig: non-robot UEs sharing each cell
    energy: object = None                # core.energy.EnergyConfig: per-robot radio energy and battery
    wifi: object | None = None           # WifiConfig (core/wifi/config.py) for level "WIFI"; None = WifiConfig()
    fidelity: object = None              # adaptive / mixed fidelity: a core.adaptive.FidelityConfig for make_adaptive

    # ---------------- derived ----------------
    def __post_init__(self):
        assert self.mu in (0, 1, 2), "mu must be 0, 1 or 2 (FR1)"
        assert set(self.tdd_pattern) <= set("DSU") and self.tdd_pattern
        assert self.harq_combining in ("cc", "ir_lena", "none") and self.harq_fail in ("rlc_am", "drop")
        assert self.eff_sinr in ("eesm", "mean_db") and self.pf_metric in ("subband", "wideband")
        assert self.scheduler in ("pf", "pf_wideband", "maxci", "rr"), "scheduler: pf, pf_wideband, maxci or rr"
        assert self.discard in ("purge", "none", "pdcp_arrival") and self.tbs_mode in ("38214", "lena")
        assert self.bler_source in ("pdsch", "lena", "sionna_label") and self.noise_model in ("fixed", "thermal")
        assert not (self.harq_combining == "ir_lena" and self.eff_sinr != "eesm")
        assert self.ul_power in ("allocated", "whole_band")
        assert self.proactive_grant in ("off", "every_ul_slot", "per_period")
        assert self.pf_update in ("slot", "rbg") and self.pf_avg_idle in ("decay", "freeze")
        assert self.pf_update == "slot" or self.scheduler in ("pf", "pf_wideband"), "pf_update='rbg' needs a PF scheduler"
        assert self.ul_retx_sched in ("ofdma", "tdma") and self.ul_amc_alloc in ("current", "previous")
        assert self.ul_grant_model in ("lumped", "bsr")
        assert self.sr_boot_slots >= 1 and self.bsr_delay_slots >= 1 and self.sr_boot_bytes >= 1
        assert self.rlc_tail_bytes >= 0 and self.rlc_tail_timer_ms > 0
        assert self.mcs_table in (1, 2) and self.n_harq >= 1 and self.max_harq_tx >= 1
        assert sum(self.special_split) == 14
        assert self.cell_layout in ("hex", "grid", "custom") and self.ho_rlc in ("carry", "flush")
        assert self.rlf_rlc in (None, "carry", "flush"), "rlf_rlc: None (follow ho_rlc), 'carry' or 'flush'"
        assert self.rlf_qin_db >= self.rlf_qout_db, "rlf_qin_db must not be below rlf_qout_db"
        assert self.n310 >= 1 and self.n311 >= 1, "n310 and n311 count indications (>= 1)"
        assert min(self.t310_ms, self.t311_ms, self.reest_delay_ms) >= 0, "RLF timers must be >= 0"
        assert 1 <= self.n_cells <= 7, "1 to 7 cells"
        if self.cell_layout == "custom":
            assert len(self.cell_positions_m) == self.n_cells, (
                f"cell_layout='custom' needs one position per cell ({self.n_cells}); use cell_layout='hex' or "
                "'grid', or the multicell() preset")
        assert len(self.msg_sizes) >= 1
        try:
            from .traffic import normalize_traffic
        except ImportError:  # config.py loaded as a standalone module (the docs hook, no package, no torch)
            normalize_traffic = None
        if normalize_traffic is not None:
            self.traffic = normalize_traffic(self.traffic)
        assert 0 < self.dr_delay_median_steps[0] <= self.dr_delay_median_steps[1]
        assert self.dr_delay_log_sigma[0] <= self.dr_delay_log_sigma[1] and 0 <= self.dr_loss[0] <= self.dr_loss[1] <= 1
        assert self.l0_delay_median_steps > 0 and 0 <= self.l0_loss <= 1 and self.l1_eta > 0
        assert self.rng in ("engine", "global"), "rng must be 'engine' or 'global'"
        assert self.frame_buffer >= 1 and self.timeout_steps >= 1 and self.control_step_ms > 0
        assert self.rach_initial in ("connected", "idle"), "rach_initial: 'connected' or 'idle'"
        assert self.rach_occasion_slots >= 1 and self.rach_preambles >= 1 and self.rach_max_attempts >= 1
        assert min(self.rach_rar_window_slots, self.rach_msg3_slots) >= 0 and self.rach_backoff_ms >= 0
        assert self.rach_release_after_ms is None or self.rach_release_after_ms > 0
        assert self.drx_ul_wake in ("sr", "on_duration"), "drx_ul_wake: 'sr' or 'on_duration'"
        assert self.drx_inactivity_ms >= 0 and 0 < self.drx_on_ms <= self.drx_cycle_ms and self.drx_start_offset_ms >= 0
        assert self.drx_short_cycle_ms is None or 0 < self.drx_on_ms <= self.drx_short_cycle_ms <= self.drx_cycle_ms
        assert self.drx_short_cycles >= 1
        assert self.proto_ul_slots_per_step is None or self.proto_ul_slots_per_step >= 1
        if self.ue_speed_mps is not None:
            self.fading_rho_per_ms = fading_rho_from_speed(self.ue_speed_mps, self.carrier_ghz)
        if self.channel in TR38901_SHORT:
            self.channel, self.tr38901_scenario = "tr38901", TR38901_SHORT[self.channel]
        assert self.channel in CHANNELS, f"channel must be one of {CHANNELS} or {tuple(TR38901_SHORT)}"
        assert self.shadow_acf in ("sos", "exp") and 0.0 <= self.shadow_white_frac <= 1.0
        assert self.shadow_dcorr_m > 0 and self.shadow_white_dcorr_m > 0 and self.shadow_modes >= 1
        assert self.fading_doppler in ("global", "per_robot") and self.doppler_min_speed_mps >= 0
        assert self.rician_k_ramp_slots >= 0, "rician_k_ramp_slots must be >= 0"
        assert self.tr38901_los in ("stochastic", "los", "nlos") and self.o2i_model in ("low", "high")
        assert 0.0 <= self.o2i_indoor_frac <= 1.0 and self.blockage_radius_m > 0
        assert self.los_source in LOS_SOURCES, f"los_source must be one of {LOS_SOURCES}"
        assert self.los_raycast_samples >= 2 and self.nlos_extra_loss_db >= 0 and self.blockage_max_db > 0
        assert not self.los_diffraction or self.los_source == "raycast", "los_diffraction needs los_source='raycast'"
        assert not self.los_soft or (self.channel == "tr38901" and self.los_source == "stochastic"
                                     and self.tr38901_los == "stochastic"), \
            "los_soft needs channel='tr38901', los_source='stochastic' and tr38901_los='stochastic'"
        assert self.blockage_model in ("sphere", "screen", "stochastic"), \
            "blockage_model must be 'sphere', 'screen' or 'stochastic'"
        self.blocker_size_m = tuple(tuple(float(x) for x in wh) for wh in self.blocker_size_m)
        assert len(self.blocker_size_m) >= 1 and all(len(wh) == 2 and wh[0] > 0 and wh[1] > 0
                                                     for wh in self.blocker_size_m), "blocker_size_m: ((w, h), ...)"
        if self.channel == "tr38901":
            from .channels.tr38901 import SCENARIOS, scenario_name
            self.tr38901_scenario = scenario_name(self.tr38901_scenario)
            if self.o2i_indoor_frac > 0:
                allowed = SCENARIOS[self.tr38901_scenario].o2i
                assert self.o2i_model in allowed, (
                    f"O2I model {self.o2i_model!r} is not defined for {self.tr38901_scenario} (allowed: {allowed})")

    @property
    def rician_mode(self):
        """Rician K source of the NR engine: None (Rayleigh: fading off, fading_rician off, or K = 0 because
        rician_k_from_los is off and no rician_k_db is set), "fixed" (rician_k_db) or "los" (from the LOS state)."""
        if not (self.fading and self.fading_rician):
            return None
        if self.rician_k_db is not None:
            return "fixed"
        return "los" if self.rician_k_from_los else None

    @property
    def scs_khz(self):
        """Subcarrier spacing in kHz, 15 * 2^mu."""
        return 15 * 2 ** self.mu

    @property
    def slot_ms(self):
        """Slot duration in ms, 1 / 2^mu."""
        return 1.0 / 2 ** self.mu

    @property
    def nprb(self):
        """PRBs of the carrier: n_prb if set, else TS 38.101-1 N_RB for bandwidth_mhz at scs_khz."""
        if self.n_prb is not None:
            return self.n_prb
        tab = NRB_FR1[self.scs_khz]
        if self.bandwidth_mhz not in tab:
            raise ValueError(f"{self.bandwidth_mhz} MHz not defined at {self.scs_khz} kHz")
        return tab[self.bandwidth_mhz]

    @property
    def rbg(self):
        """RBG size in PRBs (one subband): rbg_size if set, else the TS 38.214 value for nprb."""
        return self.rbg_size if self.rbg_size is not None else rbg_size_38214(self.nprb, self.rbg_config)

    @property
    def n_subbands(self):
        """Number of RBGs (subbands) of the carrier, ceil(nprb / rbg)."""
        return math.ceil(self.nprb / self.rbg)

    @property
    def subband_prbs(self):
        """PRBs per RBG (the last RBG may be smaller, 38.214 Sec. 5.1.2.2.1)."""
        s = [self.rbg] * self.n_subbands
        s[-1] = self.nprb - self.rbg * (self.n_subbands - 1)
        return s

    @property
    def slots_per_step(self):
        n = self.control_step_ms / self.slot_ms
        assert abs(n - round(n)) < 1e-9, "control step must be a whole number of slots"
        return int(round(n))

    @property
    def proto_slots_per_step(self):
        """UL slots per control step of the prototype levels: proto_ul_slots_per_step, or control_step_ms / 2.5 ms
        (100 ms -> 40, the prototype value; 50 ms -> 20)."""
        if self.proto_ul_slots_per_step is not None:
            return int(self.proto_ul_slots_per_step)
        n = self.control_step_ms / 2.5
        if abs(n - round(n)) > 1e-9 or round(n) < 1:
            raise ValueError(f"control_step_ms={self.control_step_ms} is not a whole number of 2.5 ms UL slots; set "
                             "proto_ul_slots_per_step for the prototype levels")
        return int(round(n))

    def slot_symbols(self, pos):
        """(dl_data_symbols, ul_data_symbols) of the slot at pattern position pos."""
        c = self.tdd_pattern[pos % len(self.tdd_pattern)]
        if c == "D":
            return 14 - self.dl_ctrl_symbols, 0
        if c == "U":
            return 0, self.ul_data_symbols
        dl = self.special_split[0] - self.dl_ctrl_symbols if self.special_dl_data else 0
        ul = self.special_split[2] if self.special_ul_data else 0
        return max(dl, 0), ul

    def ul_capable(self, pos):
        """Slot carries UL symbols (PUCCH for SR / HARQ-ACK / CQI) even if no PUSCH data."""
        c = self.tdd_pattern[pos % len(self.tdd_pattern)]
        return c == "U" or (c == "S" and self.special_split[2] > 0)

    def first_ul_in_window(self, g, period):
        """SR / CQI opportunity: g is the first UL-capable slot of its window [k*period, (k+1)*period)."""
        if not self.ul_capable(g):
            return False
        return not any(self.ul_capable(x) for x in range(period * (g // period), g))

    def next_ul_capable(self, g):
        """Smallest slot index >= g that is UL-capable."""
        P = len(self.tdd_pattern)
        for d in range(P):
            if self.ul_capable(g + d):
                return g + d
        raise ValueError("pattern has no UL-capable slot")

    @property
    def sr_delay(self):
        """Slots from a scheduling request to the first PUSCH: sr_grant_delay_slots, default gnb_proc_slots + k2."""
        return self.sr_grant_delay_slots if self.sr_grant_delay_slots is not None else self.gnb_proc_slots + self.k2

    @property
    def rlc_tail_timer_slots(self):
        """ul_grant_model="bsr": RLC buffer-status timer rlc_tail_timer_ms in slots (at least one)."""
        return max(1, int(round(self.rlc_tail_timer_ms / self.slot_ms)))

    def lena_mac_switches(self):
        """The 5G-LENA MAC switches (LENA_MAC_DEFAULTS) set away from their defaults, {field: value}."""
        return {k: getattr(self, k) for k, v in LENA_MAC_DEFAULTS.items() if getattr(self, k) != v}

    @property
    def ul_rtt(self):
        """Slots from a PUSCH to its earliest retransmission: ul_harq_rtt_slots, default gnb_proc_slots + k2."""
        return self.ul_harq_rtt_slots if self.ul_harq_rtt_slots is not None else self.gnb_proc_slots + self.k2

    @property
    def ul_slots_per_step(self):
        """Slots per control step that carry uplink data symbols."""
        return sum(1 for g in range(self.slots_per_step) if self.slot_symbols(g)[1] > 0)

    @property
    def dl_slots_per_step(self):
        """Slots per control step that carry downlink data symbols."""
        return sum(1 for g in range(self.slots_per_step) if self.slot_symbols(g)[0] > 0)

    def summary(self):
        """One-line human-readable summary of the frame structure, HARQ, PHY and scheduler settings."""
        return (f"mu={self.mu} ({self.scs_khz} kHz), {self.bandwidth_mhz} MHz -> {self.nprb} PRB, "
                f"RBG {self.rbg} -> {self.n_subbands} subbands {self.subband_prbs}, TDD {self.tdd_pattern} "
                f"S={self.special_split}, {self.slots_per_step} slots/step "
                f"({self.ul_slots_per_step} UL, {self.dl_slots_per_step} DL data slots), HARQ {self.n_harq}x"
                f"{self.max_harq_tx}tx {self.harq_fail}, MCS table {self.mcs_table}, {self.eff_sinr}, "
                f"OLLA {'on' if self.olla else 'off'}, PF {self.pf_metric}")

    def noise_dbm_per_prb(self, rx="gnb"):
        """Noise PSD per PRB (dBm). fixed: ni_fixed_dbm spread over snr_ref_prbs PRBs."""
        if self.noise_model == "fixed":
            return self.ni_fixed_dbm - 10 * math.log10(self.snr_ref_prbs)
        nf = self.gnb_nf_db if rx == "gnb" else self.ue_nf_db
        return -174.0 + 10 * math.log10(12 * self.scs_khz * 1e3) + nf

    def unused_fields(self, level):
        """Fields set away from their defaults that make_engine(level, ..., self) ignores (silently, unless
        make_engine(..., strict=True))."""
        ref = NRConfig()
        read = fields_read_by(level, self)
        return sorted(f.name for f in fields(self)
                      if f.name not in read and getattr(self, f.name) != getattr(ref, f.name))

    def with_(self, **kw):
        """A copy of this config with the given fields replaced (dataclasses.replace). An explicit
        fading_rho_per_ms clears an inherited ue_speed_mps, which would otherwise recompute it."""
        if "fading_rho_per_ms" in kw and "ue_speed_mps" not in kw:
            kw["ue_speed_mps"] = None
        return replace(self, **kw)

    # ---------------- cells ----------------
    def gnb_xy(self):
        """gNB positions (env-local metres). hex: centre site then the first ring at 0, 60, ..., 300 deg,
        first n_cells sites, centroid at cell_center_m (C = 3: equilateral triangle, C = 7: full cluster).
        grid: centres of a ceil(sqrt(C))-column tiling of the arena. custom: cell_positions_m."""
        n = self.n_cells
        if self.cell_layout == "custom":
            return [tuple(map(float, p)) for p in self.cell_positions_m]
        if self.cell_layout == "hex":
            isd = self.cell_isd_m
            pts = [(0.0, 0.0)] + [(isd * math.cos(math.pi / 3 * j), isd * math.sin(math.pi / 3 * j)) for j in range(6)]
            pts = pts[:n]
            cx, cy = sum(p[0] for p in pts) / n, sum(p[1] for p in pts) / n
            return [(p[0] - cx + self.cell_center_m[0], p[1] - cy + self.cell_center_m[1]) for p in pts]
        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)
        dx, dy = self.cell_arena_m / cols, self.cell_arena_m / rows
        return [((i % cols + 0.5) * dx, (i // cols + 0.5) * dy) for i in range(n)]

    @property
    def ul_pc_on(self):
        """Whether uplink fractional power control is on: ul_pc, or by default on exactly when n_cells > 1."""
        return self.n_cells > 1 if self.ul_pc is None else bool(self.ul_pc)

    @property
    def ul_slot_ms(self):
        """UL slot spacing of the multi-cell legacy engine NetSlotMC: control_step_ms / proto_slots_per_step (2.5 ms
        by default). Converts the handover times to NetSlotMC's slots. The NR engine counts them in slot_ms
        instead (CellAssociation(slot_ms=...)), so its values do not depend on this."""
        return self.control_step_ms / self.proto_slots_per_step

    @property
    def rlf_flush(self):
        """Whether a radio link failure drops the robot's queued frames: rlf_rlc, or ho_rlc when it is None."""
        return (self.ho_rlc if self.rlf_rlc is None else self.rlf_rlc) == "flush"

    @property
    def ttt_slots(self):
        """A3 time-to-trigger a3_ttt_ms in NetSlotMC's UL slots (ul_slot_ms)."""
        return int(round(self.a3_ttt_ms / self.ul_slot_ms))

    @property
    def ho_int_slots(self):
        """Handover interruption ho_interruption_ms in NetSlotMC's UL slots (ul_slot_ms)."""
        return int(round(self.ho_interruption_ms / self.ul_slot_ms))

    @property
    def subband_noise_dbm(self):
        """Noise per snr_ref_prbs-PRB subband at the gNB (fixed: ni_fixed_dbm; thermal: -174 dBm/Hz + NF)."""
        return self.noise_dbm_per_prb("gnb") + 10 * math.log10(self.snr_ref_prbs)

    def is_legacy_cell(self):
        """One gNB at the origin with the fixed -90 dBm noise floor: the geometry of the prototype levels."""
        return (self.n_cells == 1 and self.gnb_xy() == [(0.0, 0.0)] and self.noise_model == "fixed"
                and self.ni_fixed_dbm == -90.0 and self.ue_tx_dbm == 23.0 and self.pl_const_db == 40.0
                and self.pathloss_exp == 3.5 and self.shadow_sigma_db == 6.0 and self.shadow_modes == 8
                and self.channel == "log_distance" and self.shadow_acf == "sos" and self.shadow_dcorr_m == 10.3
                and self.shadow_white_frac == 0.0 and not self.blockage)


def netslot_compat(**kw):
    """Closest NRConfig to the legacy NetSlot (netsim.py L2): single HARQ process with head-of-line
    blocking, DDDSU at mu=1, 50 PRB in 5 subbands of 10 PRB, 12 data symbols without DMRS
    overhead, SR -> grant 2 UL slots, HARQ RTT 4 UL slots, 10 extra UL slots after exhaustion,
    OLLA on, per-subband PF, legacy mean-dB effective SINR. PHY stays 3GPP MCS/TBS/BLER."""
    base = dict(mu=1, bandwidth_mhz=20, n_prb=50, rbg_size=10, tdd_pattern="DDDSU",
                special_ul_data=False, ul_data_symbols=12, dmrs_re_per_prb=0, n_harq=1,
                max_harq_tx=4, harq_fail="rlc_am", rlc_retx_slots=50, sr_period_slots=5,
                sr_grant_delay_slots=10, ul_harq_rtt_slots=20, olla=True, pf_metric="subband",
                eff_sinr="mean_db", harq_combining="cc", dl=False)
    base.update(kw)
    return NRConfig(**base)


def lena_like(**kw):
    """Matches the ns3ref 5G-LENA v5.1 scenario (netslot-ref.cc) as far as the model allows:
    50 PRB (RbOverhead 0.1) in 5 RBGs of 10 PRB, DDDSU, 13 UL data symbols (SRS off), 16 HARQ
    processes, max 4 tx, retx at the next UL slot, RLC UM loss on HARQ exhaustion, no OLLA, no PHR
    cap, wideband PF, 5G-LENA EESM tables + NrEesmIr combining + LENA TB size (bler_source="lena"
    needs the local extraction; pass bler_source="pdsch", tbs_mode="38214", harq_combining="cc" to
    run without it), PDCP discard of arriving SDUs, thermal noise with gNB NF 18.44 dB (= -90 dBm
    per 10 PRB). Overheads are first estimates to calibrate against ns3ref (18 B frame header +
    28 B UDP/IP + 2 B PDCP + 2 B RLC per 1400 B packet, 6 B MAC per TB)."""
    base = dict(mu=1, bandwidth_mhz=20, n_prb=50, rbg_size=10, tdd_pattern="DDDSU", ul_data_symbols=13,
                dmrs_re_per_prb=0, n_harq=16, max_harq_tx=4, harq_fail="drop", olla=False,
                pf_metric="wideband", eff_sinr="eesm", bler_source="lena", tbs_mode="lena",
                harq_combining="ir_lena", phr_cap=False, discard="pdcp_arrival", noise_model="thermal",
                gnb_nf_db=18.44, tb_overhead_bytes=6, pkt_payload_bytes=1400, pkt_overhead_bytes=50,
                frame_buffer=64)
    base.update(kw)
    return NRConfig(**base)


lena_match = lena_like


def _ul_slots(n_ul, cfg_pattern="DDDSU", mu=1):
    """UL-slot units of the calibration fits -> slots (DDDSU: one UL slot per 5 slots)."""
    n_u = cfg_pattern.count("U")
    if n_u == 0:
        raise ValueError(f"tdd_pattern {cfg_pattern!r} has no U slot; the calibrated UL-slot delays need one")
    return int(round(n_ul * len(cfg_pattern) / n_u))


def _calibrated(base, kw, sr_period_ms, sr_grant_ul, harq_rtt_ul):
    """Merge kw into base, then fill the calibrated timers that kw does not set from the merged mu and tdd_pattern:
    SR period sr_period_ms in slots, SR -> grant and HARQ RTT in UL-slot units of the fits (_ul_slots)."""
    base.update(kw)
    mu, pat = base.get("mu", 1), base.get("tdd_pattern", "DDDSU")
    derived = {"sr_period_slots": lambda: int(round(sr_period_ms * 2 ** mu)),
               "sr_grant_delay_slots": lambda: _ul_slots(sr_grant_ul, pat),
               "ul_harq_rtt_slots": lambda: _ul_slots(harq_rtt_ul, pat)}
    for k, f in derived.items():
        if k not in kw:
            base[k] = f()
    return NRConfig(**base)


def srsran_like(**kw):
    """Fitted to srsRAN UL one-way delay (Zenodo 13754300, calib/params_latency.json 'srs'): SR period
    20 ms, SR -> grant 1 UL slot, HARQ RTT 4 UL slots, BLER target 0.5%, processing offset 2.5 ms, no
    proactive grants (W1 1.25 ms, KS 0.12 on the fit set). UL MCS capped at 15 (SE 2.41) for the
    measured bench UL SE ceiling of about 2.4-3 (calib/params_link.json). The slot counts follow the mu and
    tdd_pattern of the call (fit: mu=1, DDDSU)."""
    base = dict(mu=1, tdd_pattern="DDDSU", bler_target=0.005, proc_offset_ms=2.5, proactive_grant="off",
                ul_mcs_max=15)
    return _calibrated(base, kw, sr_period_ms=20, sr_grant_ul=1, harq_rtt_ul=4)


def oai_like(**kw):
    """Fitted to OAI UL one-way delay, 5/10-slot TDD periods (params_latency.json 'oai'): proactive UL
    grant once per TDD period, BLER target 0.5%, processing offset 2.25 ms (OAI 20-slot periods need
    about 7.25 ms), SR period 20 ms / SR -> grant 10 UL slots (irrelevant with proactive grants),
    HARQ RTT 4 UL slots; UL MCS capped at 15 (bench OAI UL SE ceiling median 2.36). The slot counts follow the
    mu and tdd_pattern of the call (fit: mu=1, DDDSU)."""
    base = dict(mu=1, tdd_pattern="DDDSU", proactive_grant="per_period", bler_target=0.005, proc_offset_ms=2.25,
                ul_mcs_max=15)
    return _calibrated(base, kw, sr_period_ms=20, sr_grant_ul=10, harq_rtt_ul=4)


def multicell(n_cells=3, **kw):
    """Multi-cell preset (merged from multicell/): netslot_compat() MAC and PHY, a hex cluster of n_cells gNBs
    at 100 m inter-site distance centred in the 150 m arena, thermal noise (NF 5 dB: -103.4 dBm per 10-PRB
    subband) with same-slot UL interference, UL fractional power control (P0 -88 dBm per subband, alpha 1),
    A3 handover (3 dB hysteresis, 300 ms TTT, 40 ms interruption) and lossless RLC carry-over.
    Without power control, full-power robots near their own gNB dominate the interference and three cells
    at 60 m ISD carry less than one (multicell_report); keep ul_pc on or the ISD at 100 m or more.
    Runs on level "L2" (NR engine, this MAC with 3GPP MCS/TBS/BLER) and on "L2-legacy" (NetSlotMC)."""
    base = dict(n_cells=n_cells, cell_layout="hex", cell_isd_m=100.0, cell_center_m=(75.0, 75.0),
                noise_model="thermal", gnb_nf_db=5.0, ul_pc=True, ul_pc_p0_dbm=-88.0, ul_pc_alpha=1.0)
    base.update(kw)
    return netslot_compat(**base)


def lena_validation(**kw):
    """lena_like() in the ns3ref validation geometry (ns3ref_report 5.1 / 5.5): fading off, UE power
    over the whole band, thermal noise NF 7 dB (-101.44 dBm per 10-PRB subband), per-UE link budgets
    fed directly (tools/lena_replay.py), frames every 100 ms in phase."""
    # sr_grant_delay_slots=40: LENA's SR -> first-grant delay is unmeasured (ns3ref 6.3); replaying the
    # N <= 4 sweep with 3/13/23/33/43 slots moves the light-load median delay linearly (-17.6 .. +2.4 ms
    # vs LENA), and about 40 slots (20 ms) aligns p50 and p95. Inferred, not measured.
    base = dict(fading=False, ul_power="whole_band", gnb_nf_db=6.99697, frame_buffer=128, sr_grant_delay_slots=40)
    base.update(kw)
    return lena_like(**base)


# 5G-LENA MAC behavior under load (docs/fidelity-load-gap.md): per-RBG PF averages frozen while idle, TDMA UL
# retransmissions, UL MCS for the previous PUSCH's PRBs, the SR/BSR grant pipeline with the RLC tail stall, 8 bytes of
# MAC overhead per TB (3 B subheader + 5 B short BSR). No fitted parameter: every value is 5G-LENA's own.
LENA_MAC_V2 = dict(pf_update="rbg", pf_avg_idle="freeze", ul_retx_sched="tdma", ul_amc_alloc="previous",
                   ul_grant_model="bsr", tb_overhead_bytes=8)


def lena_match_v2(**kw):
    """lena_like() (= lena_match) with the 5G-LENA MAC behavior under load (LENA_MAC_V2): the version that also
    matches 5G-LENA in loaded cells. lena_match / lena_like stay as they were, so earlier results stay reproducible.
    The lumped SR-to-grant delay (sr_grant_delay_slots, proactive_grant) is replaced by the grant pipeline."""
    base = dict(LENA_MAC_V2)
    base.update(kw)
    return lena_like(**base)


def lena_validation_v2(**kw):
    """lena_validation() geometry (fading off, whole-band UE power, NF 7 dB, 128-frame buffers) on lena_match_v2().
    The fitted 40-slot SR-to-grant delay of lena_validation() is not used: the grant pipeline and the RLC tail stall
    model what it stood in for (fidelity-vs-lena.md, "v2")."""
    base = dict(fading=False, ul_power="whole_band", gnb_nf_db=6.99697, frame_buffer=128)
    base.update(kw)
    return lena_match_v2(**base)
