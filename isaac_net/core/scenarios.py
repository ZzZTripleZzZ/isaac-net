"""Scenario presets: representative NRConfig starting points for common robot deployments.

    from isaac_net import make_engine, warehouse_private_5g
    cfg = warehouse_private_5g(radio_map_path="hall.npz")       # **kw overrides any field
    print(cfg.describe())                                       # what is on, which backends can run it
    net = make_engine("L2", E, R, "cuda", cfg, backend="auto")

Every preset returns a plain NRConfig, so cfg.with_(...), cfg.diff(...) and cfg.describe() work on it, and every
keyword argument overrides the field it names (it is applied last). The presets are representative, not calibrated:
no value below is fitted to or measured in a real deployment. What each preset turns on, and why, is in its docstring.

What is validated and what is not. All four presets start from the MAC of lena_validation_v2() (config.py): per-RBG
proportional fair scheduling with averages frozen while idle, TDMA uplink retransmissions, uplink MCS for the PRBs of
the previous PUSCH, the 5G-LENA SR / BSR grant pipeline with the RLC tail stall, 16 HARQ processes, RLC UM loss on
HARQ exhaustion, PDCP discard at arrival, 50 PRB in 10-PRB RBGs, 8 bytes of MAC overhead per transport block, no OLLA
and no power-headroom cap. That MAC was compared with ns-3 5G-LENA on a single-cell uplink sweep with fading off and
the 5G-LENA PHY tables (docs/fidelity-vs-lena.md, "v2"). Everything a preset adds on top (the channel model, fading,
several cells, power control, QoS classes, mini-slots, downlink) is outside that comparison, and so is the PHY: the
presets use the shipped Sionna PDSCH BLER curves and the TS 38.214 transport block size (bler_source="pdsch",
tbs_mode="38214", harq_combining="cc"), so they run without the local 5G-LENA table extraction. Pass
bler_source="lena", tbs_mode="lena", harq_combining="ir_lena" to use the 5G-LENA tables when they are installed.

Backends. The triton kernel refuses the SR / BSR grant pipeline, several cells, rank-2 MIMO and mini-slots
(NRTritonEngine.refusals), so every preset below runs on "reference" and "graph" only, as cfg.describe() reports.
warehouse_private_5g(ul_grant_model="lumped", sr_grant_delay_slots=40) is the triton-runnable variant ("v2 minus BSR"
of docs/fidelity-vs-lena.md, whose 40-slot SR-to-grant delay was inferred from the 5G-LENA sweep, not measured).
"""
from __future__ import annotations

import math

from .config import lena_validation_v2

# The PHY the presets use instead of the local 5G-LENA tables (always available, see the module docstring).
SHIPPED_PHY = dict(bler_source="pdsch", tbs_mode="38214", harq_combining="cc")


def _build(fields, kw):
    """lena_validation_v2() with fading on and the shipped PHY, then the scenario's fields, then the caller's kw."""
    base = dict(fading=True, **SHIPPED_PHY)
    base.update(fields)
    base.update(kw)
    return lena_validation_v2(**base)


def _face_centroid(cfg):
    """Boresight azimuth per cell (deg) toward the centroid of the gNBs (0 for a gNB at the centroid)."""
    xy = cfg.gnb_xy()
    cx, cy = sum(p[0] for p in xy) / len(xy), sum(p[1] for p in xy) / len(xy)
    az = []
    for x, y in xy:
        dx, dy = cx - x, cy - y
        az.append(0.0 if math.hypot(dx, dy) < 1e-9 else round(math.degrees(math.atan2(dy, dx)) % 360.0, 6))
    return tuple(az)


def _sector_cells(fields, kw):
    """Build the config, then point the sector antennas at the cluster centroid unless kw sets cell_azimuth_deg."""
    cfg = _build(fields, kw)
    if cfg.gnb_antenna == "sector" and "cell_azimuth_deg" not in kw:
        cfg = cfg.with_(cell_azimuth_deg=_face_centroid(cfg))
    return cfg


def warehouse_private_5g(**kw):
    """Indoor warehouse with a private 5G cell: n78 (3.5 GHz), 20 MHz TDD at 30 kHz, one ceiling-mounted gNB.

    Representative, not calibrated. Turns on, beyond lena_validation_v2() (module docstring):
      fading=True                    the validation geometry runs fading off; a warehouse link fades
      channel="tr38901_inf_sh"       TR 38.901 InF-SH path loss and LOS probability: a gNB above the clutter (8 m)
                                     and sparse clutter of large regular objects with open aisles between them, which
                                     we take as the closer InF sub-scenario to rows of racks; InF-DH (dense clutter,
                                     LOS-state correlation 1 m) suits a production floor and is used by factory_inf
      los_source="raycast"           only when radio_map_path is given: the LOS state comes from the map's height map
                                     (obstacle_z, a ray march per pose, docs/obstacles.md), so a robot behind a rack is
                                     NLOS; without a map the TR 38.901 stochastic LOS state stays. Pass
                                     los_source="map" for a map that has only los_prob
      blockage=True, blockage_model="screen"   TR 38.901 blockage model B: the other robots are screens, and
                                     people or forklifts can be added per step with step(blockers=)
      fading_rician=True             Rician fast fading with K drawn per link from the LOS state (K = 0 on NLOS or
                                     blocked links, TR 38.901 Table 7.5-6 InF values on LOS links)
      fading_freq_corr=True          frequency-correlated subbands with a delay spread drawn per link from the LOS
                                     state (Table 7.5-6 InF lgDS), so the subband scheduler does not see independent
                                     subbands
      ul_pc=True, ul_tpc=True        uplink open-loop power control (P0 -88 dBm per subband, alpha 1) with closed-loop
                                     TPC on top (accumulated 38.213 steps toward the default SINR target)
    Kept off: sector antennas (one gNB, isotropic), the QoS scheduler, RACH and DRX (robots always connected and
    awake), the downlink (dl=False). Rician K and the delay spread follow the LOS state only with pose input
    (step(t, poses)); with SNR input K is 0 and the delay spread is the NLOS draw. The gNB sits at the origin
    (cell_positions_m=((x, y),) moves it) at the InF-SH height of 8 m; robot antennas stay at 1.5 m, below the 2 m
    InF-SH clutter height that the scenario's LOS probability needs. Triton refuses it (SR / BSR grant pipeline); with
    ul_grant_model="lumped" it runs on triton too.
    """
    fields = dict(carrier_ghz=3.5, bandwidth_mhz=20, mu=1, duplex="tdd", channel="tr38901_inf_sh",
                  blockage=True, blockage_model="screen", fading_rician=True, rician_k_from_los=True,
                  fading_freq_corr=True, fading_ds_from_los=True, n_cells=1, gnb_antenna="isotropic",
                  ul_pc=True, ul_tpc=True, scheduler="pf", rach=False, drx=False)
    if kw.get("radio_map_path") is not None and "los_source" not in kw:
        fields["los_source"] = "raycast"
    return _build(fields, kw)


def factory_inf(n_cells=3, **kw):
    """Factory floor with dense clutter and several ceiling-mounted cells: TR 38.901 InF-DH at 3.5 GHz, 20 MHz TDD.

    Representative, not calibrated. Turns on, beyond lena_validation_v2() (module docstring):
      fading=True                    fading on (the validation geometry runs it off)
      channel="tr38901_inf_dh"       InF-DH: dense clutter (machinery, 6 m clutter height) below gNBs at 8 m, so most
                                     links beyond a few metres are NLOS (LOS-state correlation 1 m)
      n_cells=3 (argument), cell_layout="hex", cell_isd_m=30   a cluster of cells 30 m apart around the default
                                     arena centre (75, 75); set cell_isd_m / cell_center_m / cell_positions_m to match
                                     the hall
      gnb_antenna="sector"           TR 38.901 Table 7.3-1 element per cell (8 dBi, 65 deg), each boresight pointed at
                                     the cluster centroid (cell_azimuth_deg overrides it)
      handover                       A3 with the package defaults (3 dB hysteresis, 300 ms time-to-trigger, 40 ms
                                     interruption, lossless RLC carry-over)
      rlf=True                       radio link failure: T310 / N310 against Qout / Qin, re-establishment or idle
                                     (only with n_cells > 1, where the multi-cell association runs)
      olla=True                      outer-loop link adaptation, which lena_validation_v2() runs off (5G-LENA has
                                     none): with several cells the scheduler's MCS uses the N+I measured in the
                                     previous slot, and OLLA absorbs the mismatch with the actual same-slot
                                     interference (nr_engine.py); without it many more transport blocks exhaust HARQ
      ul_pc=True                     uplink fractional power control (on by default with several cells; set here so
                                     it stays on for n_cells=1)
      fading_rician=True, fading_freq_corr=True   Rician K and the delay spread per link from the LOS state, as in
                                     warehouse_private_5g
    Interference: thermal noise with same-slot UL interference (and DL interference with dl=True) between the cells.
    Kept off: blockage, closed-loop TPC, QoS, RACH / DRX, downlink. Robot antennas stay at 1.5 m, below the InF-DH
    clutter height. Triton refuses it (several cells, and the SR / BSR grant pipeline).
    """
    fields = dict(carrier_ghz=3.5, bandwidth_mhz=20, mu=1, duplex="tdd", channel="tr38901_inf_dh",
                  n_cells=n_cells, cell_layout="hex", cell_isd_m=30.0, gnb_antenna="sector", rlf=n_cells > 1,
                  ul_pc=True, olla=True, fading_rician=True, rician_k_from_los=True, fading_freq_corr=True,
                  fading_ds_from_los=True)
    return _sector_cells(fields, kw)


def outdoor_campus(n_cells=3, **kw):
    """Outdoor campus with several street-level cells: TR 38.901 UMi at 3.5 GHz, 20 MHz TDD.

    Representative, not calibrated. Turns on, beyond lena_validation_v2() (module docstring):
      fading=True                    fading on (the validation geometry runs it off)
      channel="tr38901_umi"          UMi street canyon: gNBs at 10 m, LOS probability and shadowing of Table 7.4.2-1
                                     and 7.4.1-1; robots outdoors (o2i_indoor_frac=0)
      n_cells=3 (argument), cell_layout="hex", cell_isd_m=200   sites 200 m apart (the UMi inter-site distance of
                                     the TR 38.901 calibration layout) around the default arena centre (75, 75)
      gnb_antenna="sector"           one sector per site pointed at the cluster centroid
      handover, rlf=True             A3 handover with the package defaults and radio link failure (rlf only with
                                     n_cells > 1)
      olla=True                      outer-loop link adaptation, which lena_validation_v2() runs off (5G-LENA has
                                     none): with several cells the scheduler's MCS uses the N+I measured in the
                                     previous slot, and OLLA absorbs the mismatch with the actual same-slot
                                     interference (nr_engine.py), as in
                                     factory_inf; without it many more transport blocks exhaust HARQ
      ul_pc=True                     uplink fractional power control
      blockage=True, blockage_model="stochastic"   TR 38.901 blockage model A: angular blocking regions around each
                                     robot (people, vehicles) for a scene without blocker geometry
      fading_rician=True, fading_freq_corr=True   Rician K and the delay spread per link from the LOS state (UMi
                                     values of Table 7.5-6)
    Kept off: closed-loop TPC, QoS, RACH / DRX, downlink. UMi needs robot antennas of at least 1.5 m (the default).
    Triton refuses it (several cells, and the SR / BSR grant pipeline).
    """
    fields = dict(carrier_ghz=3.5, bandwidth_mhz=20, mu=1, duplex="tdd", channel="tr38901_umi", o2i_indoor_frac=0.0,
                  n_cells=n_cells, cell_layout="hex", cell_isd_m=200.0, gnb_antenna="sector", rlf=n_cells > 1,
                  ul_pc=True, olla=True, blockage=True, blockage_model="stochastic", fading_rician=True,
                  rician_k_from_los=True, fading_freq_corr=True, fading_ds_from_los=True)
    return _sector_cells(fields, kw)


# 3GPP 5QI characteristics (TS 23.501 Table 5.7.4-1) of the two classes: 5QI 82 (delay-critical GBR, discrete
# automation: priority level 19, 10 ms packet delay budget) and 5QI 2 (GBR conversational video: priority 40, 150 ms)
URLLC_QOS = dict(qos_classes=2, qos_priority=(19, 40), qos_pdb_ms=(10.0, 150.0))


def urllc_control(base=None, **kw):
    """Low-latency robot control: a 10 ms control step, 2-symbol mini-slot grants and QoS classes for commands and
    video. Adds these switches to `base` (an NRConfig, for example warehouse_private_5g()), or to the MAC of
    lena_validation_v2() with fading on (module docstring) when base is None.

    Representative, not calibrated. Turns on:
      control_step_ms=10             a 100 Hz control loop (20 slots per step at mu=1)
      ul_mini_slot_symbols=2, dl=True, mini_slot_dl=True   every UL and DL data slot is split into 2-symbol
                                     occasions, each with its own grant and transport block, so a short command
                                     finishes before the slot ends (docs/configurability.md "Mini-slot grants")
      scheduler="qos", qos_classes=2  the 5G-LENA QoS scheduler with two message classes: class 0 = commands
                                     (priority 0 in submit(..., priority=) or a TrafficModel, the default for policy
                                     messages) with 5QI 82 characteristics (priority level 19, 10 ms delay budget),
                                     class 1 = video (priority=1) with 5QI 2 characteristics (priority level 40,
                                     150 ms budget)
      ul_grant_model="lumped", proactive_grant="every_ul_slot"   an uplink grant in every UL slot without a
                                     scheduling request, a stand-in for configured grants; it replaces the SR / BSR
                                     pipeline, so the uplink grant timing here is not part of the 5G-LENA comparison
    Not modelled: MCS table 3 and BLER targets far below 10 %, PDCCH monitoring per mini-slot, preemption, and
    configured grants sized to a command. Video comes from a traffic model with priority=1, for example
    NRConfig.traffic=[TrafficModel.video(fps=30, mean_frame_bytes=4000, priority=1)]. Triton refuses it (mini-slots).
    """
    fields = dict(control_step_ms=10.0, ul_mini_slot_symbols=2, dl=True, mini_slot_dl=True, scheduler="qos",
                  ul_grant_model="lumped", proactive_grant="every_ul_slot", **URLLC_QOS)
    if base is None:
        return _build(fields, kw)
    fields.update(kw)
    return base.with_(**fields)


SCENARIOS = {"warehouse_private_5g": warehouse_private_5g, "factory_inf": factory_inf,
             "outdoor_campus": outdoor_campus, "urllc_control": urllc_control}

__all__ = ["warehouse_private_5g", "factory_inf", "outdoor_campus", "urllc_control", "SCENARIOS", "SHIPPED_PHY",
           "URLLC_QOS"]
