# Configurability

Users should be able to choose the network model they train against, not inherit the choices baked into the prototype. This page records an audit of the package at `deb1b35` (2026-09-29). It lists every hard-coded choice found in the engines, compares the user-selectable features with ns-3 5G-LENA, Sionna SYS and Simu5G, and proposes the modes and switches to add, grouped by cost. The last section describes what the branch `feat/config-audit` already plumbed into `NRConfig`.

## How configuration works today

One `NRConfig` dataclass (`isaac_net/core/config.py`) configures every module, and `make_engine(level, E, R, device, config, backend)` builds every fidelity level. The configurable NR engine (`L2`) reads almost all of it. The other levels read much less. The prototype levels (`L0` to `L1`, `L2-legacy`), the fitted surrogates and the bounds read only the application fields (frame buffer, timeout, control step and UL slots per step, message sizes) and the randomness fields (`seed`, `rng`); since `feat/protolevels` they accept any values of these (see "Engine-owned randomness and the application constants"). `L2-legacy` with more than one cell or thermal noise runs `NetSlotMC`, which reads the radio, cell, power-control and handover fields but none of the NR MAC fields. `L2` reads those cell fields too, plus `dl_interference`. Until this branch, a field that a level does not read was ignored without notice. For example, `make_engine("L0", config=NRConfig(pathloss_exp=3.0))` runs with the fixed legacy radio, and `make_engine("L2-legacy", config=NRConfig(olla_up_db=0.1))` runs with the legacy OLLA steps. `NRConfig.unused_fields(level)` now lists such fields and `make_engine(..., strict=True)` refuses them (see the last section).

## Inventory of hard-coded choices

"In NRConfig" says whether a field already controls the choice, and for which engine. "Fast backends" says whether changing it is safe for the `graph` and `triton` backends of the prototype levels: *constexpr* means the Triton kernel already receives it as a compile-time constant, so a new value only triggers a recompile. *Literal* means it is written into the kernel or the graph body and needs a code change first.

### Application and time

| Choice | Where | Value | In NRConfig | Fast backends |
|:---|:---|:---|:---|:---|
| Frame buffer depth | `proto/netsim.py` (`F`, default of `fb`) | 16 frames per robot | `frame_buffer`, every level | constexpr (`F_`, padded to a power of two `FB`) |
| Application timeout | `proto/netsim.py` (`TIMEOUT`, default of `timeout`) | 20 control steps (2 s) | `timeout_steps`, every level | argument of the graph bodies |
| Control step | `proto/netsim.py` (`UL_PER_STEP`, default of `ul_per_step`) | 100 ms, 40 UL slots | `control_step_ms` (every level) and `proto_ul_slots_per_step` (default `control_step_ms / 2.5`) | constexpr (`K`) |
| Message size classes | `config.py` `msg_sizes` | (4000, 30000) B | yes, every level | per-call tensor, safe |
| Isaac-side message sizes | `isaac/netmodule.py:54` | (1500, 12000) B | yes since 9c642ce: the Isaac layer reads `NRConfig.msg_sizes` (4000, 30000) | n/a |
| Policy messages per robot per step | `traffic.py`, `Requests.send` | at most one, class index 0, 1, 2, ... | no; generated traffic: `traffic` (see [Traffic models](#traffic-models)), `L2` only | fixed shape `[E,R]` |
| Generated traffic direction | `traffic.py` `TrafficModel.direction` | uplink | `direction="dl"` (or `.downlink()`) per model, mixed freely with UL models; `L2` with `dl=True` (see [Downlink models](#downlink-models)) | reference and `graph` backends |
| Per-message tag | `Requests.det` / `hid`; `submit(..., tag=, priority=, deadline_ms=)` | one bool per message, one id per env; tag, priority and deadline per message on `L2` | no | safe |
| Stack processing offset | `config.py` `proc_offset_ms` | 0 ms | yes, `L2` | n/a |
| Stepping randomness | every engine | engine-owned counter streams (`proto/rng.py`; `nr_rng.py` on `L2`) | `seed`, `rng` (`"engine"` default, `"global"` = earlier behavior) | hashed inside the graph and the Triton kernel |

### Radio

| Choice | Where | Value | In NRConfig | Fast backends |
|:---|:---|:---|:---|:---|
| Path loss model | `proto/netsim.py:170` (legacy), `radio.py`, `channels/` | log-distance, 40 + 35 log10(d) | `channel` = `log_distance` (`pl_const_db`, `pathloss_exp`), `tr38901` (8 scenarios) or `radio_map`, for `L2` and `NetSlotMC` ([channels.md](channels.md)); the prototype `Radio` is fixed | radio runs outside the engine graph; its ops are fixed-shape and capture in a CUDA graph |
| Minimum distance, 2-D distance | `radio.py:71-72`, `proto/netsim.py:169` | d ≥ 1 m, height ignored | no | safe |
| Shadowing | `channels/fields.py`, `proto/netsim.py:161` | 6 dB, 8 plane waves, wavelengths uniform in 20–60 m | `shadow_sigma_db`, `shadow_modes`, `shadow_dcorr_m`, `shadow_acf` (legacy band or exponential ACF), `shadow_white_frac`, `shadow_white_dcorr_m` (`L2`, `NetSlotMC`); TR 38.901 sigma and correlation per scenario | safe |
| gNB placement | `config.py` `cell_positions_m` | one gNB at the arena corner (0, 0) | yes (`cell_layout`, hex / grid / custom) for `L2` and `NetSlotMC`; `L1` and `QA` require the corner | safe |
| Noise floor | `proto/netsim.py:27`, `config.py` | −90 dBm per 10-PRB subband, includes interference | `noise_model`, `ni_fixed_dbm`, `gnb_nf_db`, `ue_nf_db` for `L2` and `NetSlotMC` | legacy constant |
| UE and gNB power | `proto/netsim.py:26`, `config.py` | 23 dBm, 43 dBm | `ue_tx_dbm`, `gnb_tx_dbm` (`L2`, `NetSlotMC`) | legacy constant |
| Fast fading | `proto/netsim.py:35`, `config.py`, `channels/doppler.py`, `nr_engine.py` | AR(1) Rayleigh per subband, 0.93 per 2.5 ms (3 m/s at 3.5 GHz) | `fading`, `fading_rho_per_ms`, `ue_speed_mps`, `carrier_ghz`, per-robot Doppler from each robot's speed (`fading_doppler="per_robot"`) and Rician fading with a K-factor from the LOS state (`fading_rician`, `rician_k_db`, `rician_k_from_los`, `rician_k_ramp_slots`) and frequency correlation across the subbands from a delay spread (`fading_freq_corr`, `fading_delay_spread_ns`, `fading_ds_from_los`, `fading_ds_grid`, `inf_hall_volume_m3`, `inf_hall_surface_m2`, `inf_lg_ds`) for `L2`; legacy fixed | constexpr (`RHO`); Rician: kernel inputs `k_ptr`, `phi_ptr`, constexpr `RICIAN`; frequency correlation: kernel inputs `fcl_ptr`, `fci_ptr`, constexpr `FCORR` |
| DL SNR from UL SNR | `config.py` `dl_snr_offset_db` | UL + 10 dB when no DL SNR is given | yes, `L2` | n/a |
| LOS blockage | `isaac/netmodule.py:84`, `channels/blockage.py` | 20 dB when the ray is blocked | Isaac `ParamRanges`; robot bodies as spheres in `RadioMC` (`blockage`, `blockage_radius_m`, `blockage_loss_db`, off by default); TR 38.901 model B screens with per-step `blockers=` or model A regions (`blockage_model`, `blocker_size_m`, `blockage_max_db`); the Isaac `blocked_fn` also drives the engine radio through `los_source="callback"` ([obstacles.md](obstacles.md)) | fixed-shape `[E,R,R,C]` test; screens `[E,R,C,R+M]` |
| LOS state from geometry | `channels/models.py` (stochastic only) | TR 38.901 probability, threshold field | `los_source` = `map` (baked `los_prob`) / `raycast` (2.5-D march over `obstacle_z`, `los_raycast_samples`) / `callback`; `los_diffraction` (ITU-R P.526 knife edge), `los_soft` (§7.6.3.3), `nlos_extra_loss_db` ([obstacles.md](obstacles.md)) | fixed-shape `[E,R,C,N]`, no host sync; radio runs eagerly before the captured step |
| gNB antenna pattern | `channels/antenna.py`, `radio.py` (`RadioMC.rx_dbm`) | isotropic (0 dBi) at every gNB | `gnb_antenna="sector"`: TR 38.901 Table 7.3-1 element with `cell_azimuth_deg`, `cell_tilt_deg`, `gnb_antenna_gain_dbi` (`L2`, `NetSlotMC`); the UE stays isotropic | input side only (path gain), so every backend, `triton` included |

### PHY and link adaptation

| Choice | Where | Value | In NRConfig | Fast backends |
|:---|:---|:---|:---|:---|
| Legacy spectral efficiency | `proto/netsim.py:64` | 0.75 log2(1 + SNR), clamped to [0.2, 5.5] | no | constexpr (`SE_MIN`, `SE_MAX`); 0.75 literal |
| Legacy BLER | `proto/netsim.py:539` | logistic, slope 1.5 around the required SNR | no | literal |
| Legacy chase-combining gain | `proto/netsim.py:513` | +3 dB per retransmission | no | literal |
| L1 goodput factor | `proto/netsim.py:464`, `triton_slot.py` | 0.9 | now `l1_eta` | now constexpr (`ETA`) |
| Subbands and PRBs (legacy) | `proto/netsim.py:24-25` | 5 × 10 PRB, 12 symbols | NR: `bandwidth_mhz`, `n_prb`, `rbg_size`; legacy fixed | constexpr (`S_`, `BYTES`) |
| MCS / TBS / BLER tables | `phy.py` | 38.214 tables 1 and 2, Sionna or local 5G-LENA BLER | `mcs_table`, `bler_source`, `tbs_mode`, `eff_sinr`, `bler_target`, `*_mcs_max` | `L2` has no fast backend |
| Link-adaptation reference | `phy.py:203-204` | threshold SINR at a 1524-bit code block on one 10-PRB RBG | no | n/a |
| OLLA steps | `proto/netsim.py:546`, `proto/netsim_mc.py:257`, `mac.py:254` | legacy +0.05 / −0.45 dB; NR `olla_up_db`; clamp ±10 dB everywhere | NR yes (except the clamp); legacy and `NetSlotMC` no | literal |
| Layers / antennas | `phy.py:46` | one layer, one antenna | no (`tbs_38214` already has `layers`) | n/a |

### MAC

| Choice | Where | Value | In NRConfig | Fast backends |
|:---|:---|:---|:---|:---|
| Scheduler | `mac.py` (`MacLink.slot`), `proto/netsim.py:522-529` | proportional fair (NR default); legacy engine PF only | NR `scheduler` = `"pf"`, `"pf_wideband"`, `"maxci"`, `"rr"`, `"qos"` ([QoS scheduling](#qos-scheduling)), `pf_metric` subband / wideband, `pf_window`, `retx_priority` | constexpr (`SCHED`, `PF_A`, `PF_B`, `QOS_G`); `qos` reads per-robot weights `qw` |
| PF average initial value and floor | `mac.py:48, 56`, `proto/netsim.py:37, 475`, `netsim_mc.py:142` | 100 B/slot; floor 1 B/slot on the legacy engine, 1e-9 B/slot on the NR engine (`AVG_MIN`, as 5G-LENA) | no | constexpr (`PF_MIN`, `AVG_MIN`) |
| SR to grant delay (legacy) | `proto/netsim.py:31` | 2 UL slots | NR: `sr_period_slots`, `sr_grant_delay_slots`; legacy fixed | constexpr |
| HARQ RTT, max transmissions, RLC retry (legacy) | `proto/netsim.py:32-34` | 4 UL slots, 4 tx, +10 UL slots | NR: `ul_harq_rtt_slots`, `max_harq_tx`, `n_harq`, `harq_fail`, `rlc_retx_slots`; legacy fixed | constexpr |
| Power-headroom cap | `proto/netsim.py:38` | 3 dB per subband | NR `phr_cap`, `phr_min_db`; legacy fixed | constexpr (`PHR`) |
| UL power control | `mac_ul.py` (`pc_backoff`, `_pc`), `proto/netsim_mc.py:222-224` | fractional open loop, on by default with more than one cell; closed-loop TPC off | `ul_pc*` (`L2` and multi-cell legacy); closed loop `ul_tpc*` (`L2`, see [below](#closed-loop-power-control-cqi-table-and-sector-antennas)) | open loop: every backend; `ul_tpc`: reference and graph (`triton` refuses it) |
| Retransmission priority | `mac.py:243-290` | admitted retransmissions win RBGs before new data (`retx_priority`); off: they compete on the PF metric and RBGs of short ones are released | `retx_priority` on/off | same rule in the `triton` kernel |
| Duplexing | `config.py` pattern helpers (`slot_symbols`, `ul_capable`, `next_ul_capable`) | TDD, one carrier for both directions | `duplex="fdd"` with `dl_n_prb` / `dl_bandwidth_mhz` (`L2`); see [Duplexing](#duplexing-tdd-and-fdd) | reference and graph; `triton` refuses FDD |
| DL CQI | `mac_dl.py` (`cqi_report`), `phy.py` (`CQI_T1`, `CQI_T2`, `cqi_tables`) | best MCS per subband, mapped back to its threshold, reported every 10 slots | `cqi_period_slots`; `cqi_table="38214"` quantizes to the 4-bit CQI of TS 38.214 Table 5.2.2.1-2 / -3 instead | `"mcs"`: every backend; `"38214"`: reference and graph (`triton` refuses it) |

### Fidelity levels, surrogates and bounds

| Choice | Where | Value | In NRConfig | Fast backends |
|:---|:---|:---|:---|:---|
| L0 delay and loss | `proto/netsim.py` `NetDelay` | from `params`; no default (make_engine raised) | now `l0_delay_median_steps`, `l0_delay_log_sigma`, `l0_loss` | per-call floats |
| L0DR ranges | `proto/netsim.py` `L0DR_RANGES` (fixed in `_reset_state` on `main`) | median 0.05–10 steps log-uniform, σ 0.2–1.2, loss 0–0.2 | now `dr_delay_median_steps`, `dr_delay_log_sigma`, `dr_loss` | reset draws run eagerly, safe |
| L05 / L05Q bins | `proto/netsim.py:39-41` | active robots {2, 5, 9}, SNR {0, 10, 20, 30} dB, own queue {0, 1, 3} | no; fixed in the fit format | device tensors |
| QA efficiency and PF gain | `levels/__init__.py:23` | η = 0.9, PF diversity on | through `params` only | graph-safe |
| NN history and features | `levels/surrogates.py:23, 207` | 4-step history, SNR / 40, own queue / 16 | no; part of the fit format | graph-safe |
| Surrogate message semantics | `levels/base.py:10` | F = 16, 20-step timeout, 100 ms step | yes; the fit file records them and loading refuses a mismatch | graph-safe |
| ORACLE / NOCOMM | `levels/bounds.py` | delay 0 and never lost; never delivered, FIFO fills and overflows | no | graph-safe |

### Isaac layer and example task

| Choice | Where | Value | Notes |
|:---|:---|:---|:---|
| Observation features | `isaac/net_module.py:191-198`, `isaac/netmodule.py:518-526` | AoI clamped at 50 steps, SNR / 40 (or / 30), queue / 16, delivered flag, blocked flag | two different normalizations in the two Isaac modules; no feature selection |
| SNR sampling within a step | `isaac/netmodule.py:50` | `pose_chunks = 4` | `IsaacNetCfg` |
| Domain randomization | `isaac/netmodule.py:77-88`, `isaac/mdp/events.py` | per-env uniform ranges of p_tx, noise, path loss, shadowing, blockage, background load, L0 delay | only on the Isaac engine; `make_engine` levels have no per-env parameter tensors |
| Background load | `isaac/netmodule.py:85` | fraction of subbands taken by other UEs | Isaac engine only |
| Fleet task | `examples/fleet_task.py:3-19` | imports `F`, `TIMEOUT` and the prototype `Radio` whatever the engine | task constants are class attributes |

## Feature matrix

The comparison was checked line by line against the official documentation of ns-3 5G-LENA `v5.1` (NR manual sources and `FEATURES.md` at that tag), NVIDIA Sionna SYS `v2.2.0` (API docs and tutorials) and Simu5G `v1.7.0` (simu5g.org user's guide and the repository at that tag), all accessed on 2026-09-29. [feature-matrix-sources.md](feature-matrix-sources.md) gives, for every (tool, feature) cell, the status, the feature name the tool's documentation uses, and the URL and section. The status words there are *supported*, *partial* and *not supported*. A cell confirmed only by source code, not by a manual, is marked as such there. "Have" means a user can select it today through `NRConfig` or `make_engine`. Sionna SYS is a set of system-level blocks on top of Sionna PHY and RT, so its cells name the companion package when the capability lives there.

| Feature | isaac-net | 5G-LENA | Sionna SYS | Simu5G | Our status and reason |
|:---|:---|:---|:---|:---|:---|
| Numerology | μ = 0, 1, 2 (`L2`) | μ = 0–4 (FR1, FR2) | no numerology model; any subcarrier spacing via Sionna PHY `ResourceGrid` | μ = 0–4, one per component carrier | **partial**: FR2 (μ = 3) missing, cheap once tables allow |
| Bandwidth / PRBs | any 38.101 FR1 value (`L2`) | any, up to 275 PRBs per BWP | any (`ResourceGrid`, scheduler `num_freq_res`) | any (`numBands` per carrier) | **have** (`L2`); legacy fixed at 50 PRB |
| TDD / FDD | any TDD string, or FDD with a paired DL carrier (`duplex="fdd"`, `dl_n_prb` / `dl_bandwidth_mhz`) (`L2`) | TDD and FDD | none; one direction (DL or UL) per run | TDD (fixed DL/UL symbol split) and FDD | **have** (`duplex="fdd"`, `L2` reference and `graph`): one subband grid shared by the two carriers; `triton` and the legacy levels are TDD only ([Duplexing](#duplexing-tdd-and-fdd)) |
| Schedulers | PF (subband or wideband), max C/I, round robin, and a QoS-aware PF (`scheduler="qos"`, QoS / slicing row) | PF, RR, MR in TDMA and OFDMA, QoS-aware, random, RL-based | PF (SU-MIMO) | Max C/I (and variants), PF, DRR, QoS-aware PF | **partial**: RR and max-C/I are one line each in `mac.py` |
| HARQ | multi-process, chase or IR, max tx | IR and CC, multi-process, max retx | ACK/NACK feedback to link adaptation, no retransmissions | yes (processes, max retx) | **have** (`L2`) |
| RLC | AM retry or UM loss, PDCP discard | UM, AM, TM | none | UM, AM, TM | **partial**: no RLC segmentation timers or status reports |
| Link adaptation | BLER target, OLLA, MCS caps; DL CQI per MCS or on the 38.214 4-bit CQI table (`cqi_table`) | AMC, error-model or Shannon based | inner and outer loop | CQI-based AMC | **have**; OLLA clamp and legacy steps fixed |
| Power control | UL fractional open loop (`L2` and legacy multi-cell) and closed-loop TPC, accumulated or absolute (`ul_tpc`, `L2`) | UL open and closed loop; DL uniform power allocation only | UL open loop, DL fair power | none documented (fixed transmit powers) | **have** UL open loop (`ul_pc`) and closed-loop TPC (`ul_tpc`, `L2` reference and `graph`); **missing**: DL power control, PUCCH / SRS loops, TPC on the `triton` backend |
| Antenna patterns | gNB sector element of TR 38.901 Table 7.3-1 per cell, boresight and downtilt (`gnb_antenna`); UE isotropic | 3GPP UPAs, dual polarization, multi-panel, isotropic / cosine / parabolic | antenna arrays and patterns via Sionna PHY | isotropic or directional per node | **have** (`gnb_antenna="sector"`, every backend): one element per cell; **missing**: arrays and beamforming; the Sionna RT bake applies the pattern per grid point along the direct direction, not per traced ray ([scene-radio-map.md](scene-radio-map.md#sector-antennas-at-bake-time)) |
| MIMO / beamforming | none (one layer) | SU-MIMO up to rank 4, analog beamforming | SU-MIMO streams; precoding via Sionna PHY | none (incomplete MIMO removed in v1.4.3) | **missing**: layers could scale TBS and SINR (moderate); beamforming is large |
| Channel model | log-distance with correlated and white shadowing; TR 38.901 RMa, UMa, UMi, InH, InF-SL/DL/SH/DH path loss with spatially consistent LOS state and O2I; precomputed radio maps (Sionna RT baking tool); robot-body blockage (obstacles and the TR 38.901 blockage models: next row); AR(1) Rayleigh with per-robot Doppler and an optional Rician K-factor (row after next) | 3GPP TR 38.901 (RMa, UMa, UMi, InH, V2V, NTN), NYUSIM (incl. InF), FTR, Sionna RT | TR 38.901 via Sionna PHY (UMi, UMa, RMa, InH, InF); ray tracing via Sionna RT | 3GPP TR 36.814, 36.873, 38.901 path loss, shadowing, Rayleigh or Jakes fading | **have** large-scale models ([channels.md](channels.md)); **missing**: 38.901 cluster fast fading, online ray tracing |
| Obstacles / LOS blockage | geometric LOS state from a baked LOS map or a 2.5-D ray march over the scene's height map, or an Isaac callback (`los_source`); ITU-R P.526 knife-edge diffraction (`los_diffraction`); TR 38.901 soft LOS (`los_soft`); TR 38.901 blockage model B screens for robots and per-step blockers and model A regions (`blockage_model`) (`RadioMC`, per-step `blockers=` on `L2`; [obstacles.md](obstacles.md)) | supported (source only, ns-3-dev): LOS from `Building` boxes (`BuildingsChannelConditionModel`) and blockage model A in `ThreeGppChannelModel`; no model B in the source | not assessed | not assessed | **have** (`los_source`, `los_diffraction`, `los_soft`, `blockage_model`): 2.5-D height map, one map for all envs; **missing**: exact 3-D mesh LOS inside the graph backend, multi-edge diffraction |
| Rician fading / K-factor | per-link K fixed or from the LOS state (TR 38.901 Table 7.5-6 log-normal per scenario, 0 when NLOS or blocked), linear ramp on LOS changes, all three `L2` backends ([channels.md](channels.md#rician-fading)) | K-factor inside the TR 38.901 cluster model | inside the TR 38.901 CDL / TDL models of Sionna PHY | not assessed | **have** (`fading_rician`, `L2`): specular term on the AR(1) Rayleigh state; no LOS-path Doppler; subbands independent unless `fading_freq_corr` |
| Fast fading: frequency selectivity | subband (RBG) fading correlated by an exponential power-delay profile, `1 / sqrt(1 + (2π Δf τ_rms)^2)`, with one delay spread or a per-link log-normal draw from TR 38.901 Table 7.5-6 by LOS state, all three `L2` backends ([channels.md](channels.md#frequency-selective-fading)) | cluster delays and angles of the TR 38.901 model; TDL-A / TDL-D in the PHY manual | TR 38.901 CDL / TDL models of Sionna PHY | not assessed | **have** (`fading_freq_corr`, `fading_delay_spread_ns`, `fading_ds_from_los`, `L2`): correlation only, no tap structure or angles; **pending**: PF subband gain against 5G-LENA with `McsCsiSource=AVG_MCS` |
| Mobility | from the simulator's poses | ns-3 mobility models | random UT velocities in the topology generators; trajectories user-coded | INET mobility models, Veins | **have**: poses come from Isaac Lab, which is the point of the package |
| Traffic | policy messages (one per robot per step, size classes), periodic / bursty / video / event generators in the uplink or the downlink (`L2`); DL bytes | NGMN, 3GPP XR, FTP Model 1, HTTP generators | none (scheduler takes rates only) | any INET application | **partial**: periodic, bursty, video and event generators on `L2` only; DL generators (`direction="dl"`) on the reference and `graph` backends, not on `triton` ([Downlink models](#downlink-models)) |
| UL / DL / sidelink | UL, DL (`L2`) | UL, DL; sidelink only in a separate v3.1-based branch | UL, DL | UL, DL, network-assisted D2D (prototype) | **partial**: sidelink out of scope for now |
| QoS / slicing | `scheduler="qos"`: per-message class from `priority`, 5G-LENA QoS metric with 3GPP priority level and packet delay budget per class, class-ordered byte assignment (`L2`) | 5QI QoS schedulers, BWP-based slicing | none | 5QI QoS flows, SDAP, QoS-aware PF; no slicing | **have** QoS scheduling (`scheduler="qos"`, every `L2` backend): one queue per robot, reordered by class once per step, so a new class-0 message waits behind at most one partly sent message (option (b), [QoS scheduling](#qos-scheduling)); **missing**: GBR rate guarantee, slicing |
| Multi-cell / handover | 1–7 cells, per-cell PF and HARQ, A3 handover with interruption (`L2` and legacy) | multi-cell, X2 handover, hex wraparound | multi-cell hex layouts, wraparound; no handover | multi-cell, X2 handover, background cells | **partial**: no wraparound, no X2 data forwarding model |
| Radio link failure | N310/N311/T310/T311 on the serving-link SINR, re-establishment at the best suitable cell after a fixed delay or, with `rach=True`, through contention-based RACH, queue carry or flush; A3 target admission `a3_min_target_rsrp_dbm` (`L2`, several cells; [multicell.md](multicell.md#radio-link-failure)) | *unverified*: RLF via the ns-3 LTE RRC; A3 `MinTargetRsrpDbm` | not assessed | not assessed | **have** (`rlf`, `L2` with several cells, reference and `graph`): uplink SINR once per control step; no contention-free RACH, T301 or handover-failure model |
| Interference | same-slot UL and DL per RBG (`L2`), UL (legacy) | all co-channel transmitters, incl. DL–UL cross-link | inter-cell, in the post-equalization SINR | inter-cell DL and UL (configurable), background cells | **partial** |
| Wi-Fi (802.11) uplink | level `WIFI`: mean-field DCF / EDCA contention per sub-step, 802.11ax / ac / a rates with SNR-threshold rate adaptation, A-MPDU, RTS/CTS, several APs with RSSI association and co-channel sharing, optional hidden nodes ([wifi.md](wifi.md)) | not in 5G-LENA; ns-3 has a separate `wifi` module | none | not in Simu5G; INET, which Simu5G builds on, has 802.11 models | **partial**: a validated mean-field abstraction, not a packet-level 802.11 model; no downlink, OFDMA or MU-MIMO |
| RACH / connection setup | contention-based RACH per cell: RO period, 64 preambles, collisions per RO, RAR and Msg3 delays, backoff, preambleTransMax; idle or connected start, RRC release after inactivity (`L2`, [access.md](access.md)) | contention-based RACH (preamble, RAR, Msg3), ideal or real RRC | not assessed | not assessed | **have** (`rach`, `L2` reference and `graph`): no Msg3 capture, no paging delay |
| DRX | connected-mode DRX: inactivity timer, long and short cycles, on-duration, UL wake by SR or at the on-duration, sleep power in the energy model (`L2`, [access.md](access.md)) | not supported | not assessed | not assessed | **have** (`drx`, `L2` reference and `graph`), a differentiator: 5G-LENA has no DRX, and isaac-net couples it to battery energy (`EnergyConfig.drx_sleep_power_w`) |
| Fidelity per env | cheap and expensive level side by side, per env: static mix, load-triggered switching with queue handoff, curriculum ([adaptive-fidelity.md](adaptive-fidelity.md)) | — | — | — | **have** (`core/adaptive.py`, `NRConfig.fidelity`) |
| Carrier aggregation / BWP | none | CA and BWPs | none | CA; BWPs not documented | **out of scope** for robot fleets on one carrier |
| Differentiable KPIs | fluid relaxations L1D / QAD: gradients of delay, delivery, AoI and energy w.r.t. send probability, message size, transmit power and position ([differentiable.md](differentiable.md)) | not assessed | not assessed | not assessed | **partial (exploratory)**: fluid models only; the L2 scheduler and HARQ are not differentiated |
| Real-stack validation | OAI 5G in rfsim mode behind a lockstep bridge, 1–10 UEs ([bridges-oai.md](bridges-oai.md)) | n/a | n/a | n/a | **have**, for validation only (not a model feature) |

## Traffic models

`NRConfig(traffic=...)` takes one `TrafficModel` or a list of them (`isaac_net.core.traffic`). They run inside the engine step of level `L2` and put messages into the uplink queues (or, for [downlink models](#downlink-models), the robots' DL queues) without the policy emitting them; the policy's `submit()` keeps working next to them.

```python
from isaac_net.core import NRConfig, make_engine
from isaac_net.core.traffic import TrafficModel as TM

cfg = NRConfig(traffic=[
    TM.periodic(200, period_ms=10, jitter_ms=1).on(range(4)),                  # telemetry, 10 per 100 ms step
    TM.video(fps=25, mean_frame_bytes=6_000, gop=(30_000, 4_000, 15)).on(4),   # I/P frame pattern
    TM.bursty(1_400, rate_hz=40, burst_size=4, on_off=(0.5, 1.0)).on(5),      # Markov on/off
    TM.event(4_000, trigger="alarm", det=True, deadline_ms=50),                # task-driven
    TM.policy(),                                                               # submit(), always on
])
net = make_engine("L2", E, R, "cuda", cfg, seed=0)
out = net.step(None, snr, triggers={"alarm": alarm_mask})                      # alarm_mask [E,R] or [E] bool
```

| Model | Arrivals | Notes |
|:---|:---|:---|
| `periodic(size_bytes, period_ms, jitter_ms=0, phase="random" \| "aligned")` | one message per period; the period may be shorter than the control step | `random`: each robot's first message uniform in `[0, period)` after a reset; jitter is a uniform `[0, jitter_ms)` delay around the nominal time, without drift |
| `bursty(size_bytes, rate_hz, burst_size, on_off=(mean_on_s, mean_off_s))` | exponential ON and OFF periods, Poisson bursts at `rate_hz` while ON, `burst_size` messages per burst at the same slot | stationary start after a reset; `mean_off_s = 0` is an always-on Poisson source |
| `video(fps, mean_frame_bytes, gop=(I_bytes, P_bytes, gop_len))` | one frame every `1000 / fps` ms, frame `k % gop_len == 0` is an I frame | with both `mean_frame_bytes` and `gop`, the I and P sizes are scaled to that mean |
| `event(size_bytes, trigger)` | one message at the start of each step whose trigger is set | trigger: a name in `step(..., triggers={name: mask})`, or a callable `f(clock [E]) -> mask`; `det=True` marks it like `Requests.det` |
| `policy()` | the policy's own `submit()` messages | always on; listing it documents the mix |

Every constructor also takes `tag` (default: 1 + the model's position in the list; policy messages have tag 0), `priority`, `deadline_ms` and `max_msgs_per_step`. `.on(robots)` restricts a model to robot indices, so a robot class or one robot can have its own generator, and several models can feed the same robot.

**Arrival offsets.** Each generated message has an arrival slot inside the control step. The engine enqueues all of a step's messages in arrival order and then opens the per-robot byte stream slot by slot: SR/BSR and the MAC see a message's bytes only from its arrival slot on. `step()` then reports `delay` from the arrival slot (not from the start of the step), and adds `arrival` (env clock including the offset), `arrival_slot`, `tag`, `priority`, `bytes` (on the air) and `deadline_miss` per frame, plus `gen_accepted` and `gen_bytes` per robot. A 10 ms periodic model at a 100 ms step gives every message the same delay that the policy's message gets when it submits one message every step at a 10 ms control step; `tests/test_traffic.py` checks this to 1e-4 ms. Arrivals are quantized to slot starts. Admission (frame-buffer room, PDCP discard) is decided when the step starts, and the application timeout still counts in whole control steps from the capture step.

**Fixed shapes and graphs.** A model reserves `max_msgs_per_step` arrivals per robot per step (by default the most a periodic or video model can produce, and the Poisson mean + 4σ for bursty). The generator returns `[E, R, M]` tensors, uses no data-dependent shapes or host syncs, and updates its state in place, so `TrafficGen.step` can be captured in a CUDA graph (the GPU test checks graph == eager). Arrivals past the reserved width are deferred to the next step, never dropped, and counted in `net.traffic.deferred`. The NR engine itself has no graph backend yet.

**Randomness and resets.** The engine owns the traffic generator and seeds it from its own `seed`. Step draws and reset draws use two generators, so policy sampling does not shift the traffic, the traffic does not shift the network's draws, and `reset(env_ids)` leaves every other env bit-for-bit unaffected (tested). A reset redraws the reset envs' phases, on/off states and GOP positions.

**Levels.** Only `L2` runs traffic models. Every other level (`L0` to `L1`, `L2-legacy`, the surrogates and the bounds) raises a `ValueError` that names the models it would ignore, whether or not `strict` is set, and `NRConfig.unused_fields(level)` lists `traffic`. A config with only `policy()` works everywhere. Traffic models, `submit()` extras and the extra outputs cost nothing when unused: without them the engine runs exactly its earlier ops.

**Limits.** `priority` is carried and reported. With the default schedulers the MAC serves each robot's queue in FIFO order. With `scheduler="qos"` the priority selects the message class, which sets the robot's scheduling weight and the order in which its queued messages get bytes ([QoS scheduling](#qos-scheduling)). `deadline_ms` is reported as `deadline_miss` and does not drop messages. Traffic models work with one cell and with several NR cells (`n_cells > 1`). The engine gates arrivals through four hooks on its `UlMac` instance (`sr_step`, `slot`, `end_step`, and `handover`, so that a `ho_rlc="flush"` handover spares messages that arrive later in the step). If a future `NRNet` stops calling the first three, `step()` raises instead of silently mis-timing.

### Downlink models

`direction="dl"` (or `model.downlink()`) turns any generating model into a downlink source: the gNB sends its messages to the robot through the NR engine's DL queue (`NRConfig(dl=True)`, otherwise `make_engine` raises a `ValueError`), with the same sizes, `.on(robots)`, tags, priorities, deadlines and arrival offsets as an uplink model. A config may mix UL and DL models in one list.

```python
cfg = NRConfig(dl=True, traffic=[
    TM.periodic(200, period_ms=10).on(range(4)),                 # UL telemetry
    TM.periodic(1_500, period_ms=20).downlink().on(range(4)),    # DL setpoints / map updates
    TM.video(fps=10, mean_frame_bytes=20_000).downlink().on(4),  # DL video to an operator robot
])
```

- **Arrivals.** The step enqueues a DL model's messages in arrival order and opens the robot's DL byte stream slot by slot, exactly as for the uplink (hooks on the `DlMac` instance: `slot`, `end_step`, `handover`). The DL scheduler cannot send a message's bytes before its arrival slot, and its delay counts from that slot.
- **Outputs.** With DL models, `step()` adds per DL frame `[E, R, Fd]`: `dl_delivered`, `dl_lost` (timed out or dropped), `dl_delay` (control steps from the arrival slot, NaN otherwise), `dl_tag`, `dl_bytes` (on the air), `dl_generated` (the frame came from a DL model) and `dl_deadline_miss`, plus `gen_dl_accepted` and `gen_dl_bytes` per robot. `net.traffic_stats_dl` counts generated, accepted and refused DL messages and bytes.
- **Randomness.** DL models draw from a second generator (`TrafficGen(direction="dl")`) seeded from the engine seed, so adding a DL model leaves the UL models' arrivals, and every UL output, bitwise unchanged (tested). A model's default tag is still 1 + its position in the whole list.
- **Sharing the DL queue with the edge loop.** `EdgeConfig(return_path="nr_dl")` and `add_dl_frames()` use the same per-robot DL queue (`frame_buffer` frames), so generated DL traffic competes with the commands for queue room and DL RBGs, which is the intended load. The two never mix: an edge command carries `cls` = capture step + 1 (at least 1) and `EdgeLoop` matches a delivered DL frame to its command by `cls - 1`, while generated DL frames carry `cls = -tag` (at most -1), which no command matches. `dl_generated` reports the generated ones.
- **Backends and levels.** DL models run on the reference and `graph` backends of `L2`. On `graph` the step's DL messages are generated and enqueued eagerly before the replay, and the in-step arrival gate (the hooks on `net.dl`) reads static buffers that `step()` refills before each replay, as for the UL models, so the graph backend is bitwise equal to the reference (`tests/test_limits_closed.py`). `triton` refuses them ([NR engine backends](#nr-engine-backends)). The other levels refuse them like any traffic model.

Example: [`isaac_net/examples/traffic_models.py`](https://github.com/ZzZTripleZzZ/isaac-net/blob/main/isaac_net/examples/traffic_models.py).

## Duplexing (TDD and FDD)

| Field | Default | Meaning |
|:---|:---|:---|
| `duplex` | `"tdd"` | `"tdd"`: both directions share one carrier and `tdd_pattern` assigns each slot; `"fdd"`: the UL carrier is an all-`U` pattern and a paired DL carrier is an all-`D` pattern, so every slot carries `14 - dl_ctrl_symbols` DL data symbols and `ul_data_symbols` UL data symbols |
| `dl_n_prb` | `None` | FDD only: PRBs of the DL carrier (override) |
| `dl_bandwidth_mhz` | `None` | FDD only: DL carrier bandwidth, TS 38.101-1 N_RB at `scs_khz`; `None` (with `dl_n_prb=None`) = the UL carrier |

FDD lives in the pattern helpers of `NRConfig`. `slot_symbols(pos)` returns both directions in every slot, `ul_capable(pos)` is always true and `next_ul_capable(g) = g`, so the schedule, `ul_slots_per_step = dl_slots_per_step = slots_per_step`, the SR opportunities (first slot of each `sr_period_slots` window), the DL CQI reports (first slot of each `cqi_period_slots` window) and the DL HARQ-ACK (exactly `k1` slots after the PDSCH) follow without engine changes. K2, the UL HARQ RTT and the SR grant delay are counted in slots as before. `cfg.dl_nprb` and `cfg.dl_subband_prbs` describe the DL carrier.

What the two carriers share. The NR engine keeps one subband grid for both directions: the per-subband fading state, the CQI and the SINR inputs are `[E, R, S]` with `S = n_subbands` of the UL carrier. A DL carrier of its own width is therefore split into the same `S` RBGs (`dl_subband_prbs`, as even as possible), and the DL MAC sizes its transport blocks with those PRB counts. The gNB power `gnb_tx_dbm` is spread over the DL carrier's PRBs, so the default per-PRB DL SINR moves by `-10 log10(dl_nprb / nprb)`: through `step_rx` (pose and path-gain input), the multi-cell DL PSD, and the default of the SNR input (`snr + dl_snr_offset_db`); an explicit `dl_snr_db` is taken as given. Both carriers use the same per-subband fading process (statistically the same, not reciprocal), and there is no interference between them.

Limits: `proactive_grant="per_period"` needs a TDD period and is refused with FDD (use `"every_ul_slot"`); the `triton` kernel assumes one TDD carrier and refuses `duplex="fdd"`; `L2-legacy`, `NetSlotMC` and the prototype levels have no FDD and list `duplex` in `unused_fields`. With `duplex="fdd"` the TDD fields (`tdd_pattern`, `special_split`, `special_dl_data`, `special_ul_data`) are unused on `L2`; `dl_n_prb` and `dl_bandwidth_mhz` are unused without `dl=True`, and setting them with TDD raises. `EdgeConfig(return_path="delay")` sizes the command rate on the DL carrier, `dl_nprb` PRBs (the UL carrier's `nprb` with TDD or without a DL carrier of its own).

## NR engine backends

`make_engine("L2", ..., backend=...)` builds the NR engine on `reference` (= `eager`), `graph` (the reference step captured in CUDA graphs, bitwise equal to the reference, `core/nr_fast.NRGraphEngine`) or `triton` (the slots of a step in one fused kernel, equal to the reference to float rounding, `NRTritonEngine`). `graph` runs every `L2` feature except the debug traces. What the fused kernel does not implement is refused in one place, `NRTritonEngine.__init__` (`NRTritonEngine.refusals(cfg)` lists the offending fields), before anything is built, with one message that names every such feature of the config and the full list below. The exception, `TritonUnsupported`, is both a `NotImplementedError` and a `ValueError`. SINR hooks are installed after construction and are refused at the first step.

| Feature | `reference` | `graph` | `triton` |
|:---|:---|:---|:---|
| several cells (`n_cells > 1`), so A3 handover and radio link failure (`rlf`) | yes | yes | refused |
| SR / BSR grant pipeline (`ul_grant_model="bsr"`, the presets `lena_match_v2`, `lena_validation_v2`) | yes | yes | refused |
| closed-loop UL power control (`ul_tpc`) | yes | yes | refused |
| 38.214 CQI table (`cqi_table="38214"`) | yes | yes | refused |
| RACH and DRX (`rach`, `drx`) | yes | yes | refused |
| FDD (`duplex="fdd"`) | yes | yes | refused |
| DL traffic models (`TrafficModel(..., direction="dl")`) | yes | yes | refused |
| SINR hooks (`set_sinr_hook`, `core.slot_tap` wrappers such as energy and background) | yes | yes | refused at the first step |
| debug traces (`trace_frames`, `log_sinr`, link traces) | yes | no | no |
| the other 5G-LENA MAC switches (`pf_update`, `pf_avg_idle`, `ul_retx_sched`, `ul_amc_alloc`), Rician fading, the gNB sector antenna, UL traffic models | yes | yes | yes |

Mirrors of the refused features in the fused kernel are open work ([STATUS.md](STATUS.md), items 1 and 21).

## Closed-loop power control, CQI table and sector antennas

Three switches added after the feature-gap analysis against 5G-LENA. Each defaults to the earlier engine bit for bit (`tests/test_tpc_cqi_antenna.py` T1), and its other fields count as read only while it is on (`unused_fields`).

**Closed-loop uplink power control** (`ul_tpc=True`, TS 38.213 Sec. 7.1.1, `mac_ul.UlMac`, level `L2`). It corrects the open-loop power `P0 + alpha PL`, so it needs `ul_pc` on (automatic with several cells, `ul_pc=True` at one cell). The UE transmits at `P0 + alpha PL + f`, where `f` is a per-robot offset. After every PUSCH the gNB measures that transmission's SINR, the per-PRB SINR at its transmit PSD averaged in linear scale over every RBG of the carrier (an SRS-like wideband measurement against the gNB's latest noise-plus-interference estimate). If none of the robot's commands is in flight, the gNB sends the step of the command set closest to the error, and the command takes effect `ul_tpc_delay_slots` after the PUSCH (default `k2`). Keeping one command in flight stops the loop from reacting to its own delay.

| Field | Default | Meaning |
|:---|:---|:---|
| `ul_tpc` | `False` | closed loop on (NR engine only; `NetSlotMC` lists it as unused) |
| `ul_tpc_mode` | `"accumulate"` | `"accumulate"`: `f += step`; `"absolute"`: `f = ` the step closest to `f + error` |
| `ul_tpc_target_db` | `None` | target PUSCH SINR per PRB; `None` = the 10% BLER SINR of the middle MCS (MCS 14) of `mcs_table` on one 10-PRB RBG: 6.3 dB for table 1 and 12.8 dB for table 2 with the default `bler_source="pdsch"` |
| `ul_tpc_steps_db` | `None` | command set; `None` = (−1, 0, 1, 3) dB for accumulation, (−4, −1, 1, 4) dB for absolute (38.213 Table 7.1.1-1) |
| `ul_tpc_delay_slots` | `None` | slots from the PUSCH to the slot the command applies; `None` = `k2` |
| `ul_tpc_range_db` | 20 | the offset is clamped to ±this value |

The offset enters wherever the open-loop backoff `pc_backoff` already did (`UlMac._pc`): the scheduler's estimate, the power split of `_split`, the other cells' interference (`NRNet._ul_ici` calls `_split`) and the energy tap. The power-headroom cap is unchanged. In accumulation mode a positive step is dropped while the robot transmitted at full power (the 38.213 rule against wind-up). A handover resets the offset and drops a command in flight. The state (`tpc_f`, `tpc_cmd`, `tpc_at`, and the last measured SINR `tpc_sinr`) is per robot, fixed-shape and reset with its env. With fading off, a 10 dB path-loss step is corrected in four commands (3, 3, 3 and 1 dB), and absolute mode jumps to the set value with one command.

**38.214 CQI table** (`cqi_table="38214"`, `mac_dl.DlMac.cqi_report`, `phy.cqi_tables`, needs `dl=True`). With the default `"mcs"`, the UE reports the highest MCS that meets the BLER target on each RBG. With `"38214"`, it reports the 4-bit CQI of TS 38.214 Table 5.2.2.1-2 (Table 5.2.2.1-3 with `mcs_table=2`). CQI k is reported from the 10% BLER SINR of its (Qm, R), which is the threshold of the MCS with the same spectral efficiency. CQI 1 (QPSK, R = 78/1024) has no MCS row, so its threshold is MCS 0's shifted by the Shannon-gap difference, as `build_bler_table` fills missing curves. CQI 0 means out of range. The gNB maps the CQI to the highest MCS (up to `dl_mcs_max`) whose spectral efficiency does not exceed the CQI's, and keeps that MCS's threshold as its estimate. The report period, delay and per-RBG reporting are unchanged, so only the quantization grid differs: 15 levels instead of 29 (or 28) MCSs, and the estimate is never above the per-MCS one.

**gNB sector antenna** (`gnb_antenna="sector"`, `channels/antenna.py`, applied in `RadioMC.rx_dbm`): see [channels.md](channels.md#antenna-patterns). Fields: `cell_azimuth_deg` (one boresight per cell; `None` = 30, 150 and 270 degrees cycled over the cells), `cell_tilt_deg` (one downtilt or one per cell, degrees below the horizon) and `gnb_antenna_gain_dbi` (8 dBi).

**Backends.** The antenna changes only the path gain the engine receives, so it runs on every backend. `ul_tpc` and `cqi_table="38214"` change the per-slot loop. They run on the reference and on the `graph` backend, which captures the reference step, and the `triton` backend refuses them ([NR engine backends](#nr-engine-backends)) until the fused kernel mirrors them (the kernel recomputes the power split from a per-step `pc` input and the CQI from the MCS thresholds).

## QoS scheduling

`scheduler="qos"` (level `L2`, every backend) is modelled on 5G-LENA's `NrMacSchedulerOfdmaQos`, with the weights of `NrMacSchedulerUeInfoQos` and the logical-channel byte assignment of `NrMacSchedulerLcQos` (5G-LENA `v5.1`). Each message gets a class from its `priority` (from `submit(..., priority=)` or a traffic model's `priority`), clamped to `0 .. qos_classes - 1`. A class plays the role of a 5QI flow on its own logical channel. Messages without a priority are class 0. The default is off, and with any other scheduler the engine runs exactly its earlier ops (`tests/test_qos.py` Q1 compares it with the frozen engine bit for bit).

| Field | Default | Meaning |
|:---|:---|:---|
| `scheduler` | `"pf"` | `"qos"` turns the QoS scheduler on |
| `qos_classes` | 2 | number of message classes Q |
| `qos_priority` | (10, 70) | 3GPP priority level P per class (1 to 99, lower is more important); the default pairs 5QI 5 (IMS signalling) with 5QI 7 (voice, video, interactive gaming) |
| `qos_pdb_ms` | (inf, inf) | packet delay budget per class in ms; a finite value makes the class delay-critical (the delay-budget factor below), inf keeps the factor at 1 |
| `qos_gamma` | 1.0 | rate exponent of the metric (5G-LENA `m_alpha`) |

**Metric.** On RBG s the scheduler ranks robot u by

    w(u, s) = qw(u) * r(u, s)^gamma / R(u)

where r is the achievable rate on the RBG, R is the PF average (the same `avg` as `"pf"`, floored at `AVG_MIN`) and qw is the robot's class weight. With `pf_update="rbg"` the average moves after every RBG, as for `"pf"`. The classes that count are those with unsent bytes in the robot's queue. In the downlink qw is the sum over these classes of (100 − P_c) · D_c (5G-LENA `CalculateDlWeight`). In the uplink the gNB knows the classes from the buffer status report, and qw is (100 − P_c) · D_c of the class with the lowest priority level (5G-LENA `CompareUeWeightsUl`). A robot with no such class, which happens only with padding grants of the BSR pipeline, gets the weight of the least important class.

**Delay-budget factor.** D_c follows 5G-LENA's `CalculateDelayBudgetFactor`. Let HOL be the age of the class's oldest message with unsent bytes, in ms. Then D_c = PDB / (PDB − HOL) while HOL < PDB, and D_c = PDB / 0.1 once HOL ≥ PDB. So D_c is 1 for a fresh message, grows as the message nears its budget, and is very large after it, which serves an expired message first. 5G-LENA applies D only to delay-critical GBR flows. Here a finite `qos_pdb_ms` marks the class as such, and the uplink gets the same factor (5G-LENA has none in the uplink, so `qos_pdb_ms=inf` there reproduces it). Example with the default priorities and a 30 ms budget on class 1: a class-1 message beats a fresh class-0 message on the same channel once 30 · 30 / (30 − HOL) > 90, that is, once it has waited more than 20 ms (Q3).

**Byte assignment by class.** When a robot gets a transport block, its bytes come from the most important class first, and each class stays in arrival order (as `NrMacSchedulerLcQos` serves logical channels by priority). The robot keeps one queue, a byte stream with in-order RLC delivery (`FrameQueue`). Once per control step, before its slots, `MacLink.qos_prepare` stably reorders the messages whose bytes are all unsent and already arrived, so that their bytes are laid out in class order (`FrameQueue.reorder`). Messages that already have bytes in a HARQ process keep their place, and so do messages behind the arrival gate of the traffic models.

The alternative was one queue per class, `[E, R, Q]` with a stream pointer per class. It would let a new class-0 message pre-empt the rest of a partly sent class-1 message. We chose the single reordered queue because it keeps the MAC state `[E, R]`, every HARQ and RLC rule, and the fused Triton kernel unchanged. Its limit: a new class-0 message waits for the partly sent message ahead of it, at most one message, and a message that arrives inside a step is reordered at the start of the next step. The class weights are also computed once per step, from the queue at the step start.

**Backends.** The reorder and the weights qw [E, R] are torch code that runs before the slot loop, so the `graph` backend captures them and the `triton` kernel reads qw as a per-robot input and multiplies its PF metric by it (constexpr `SCHED=3`, `QOS_G`). The retransmission rules are unchanged: admitted retransmissions still win RBGs first (`retx_priority`), and among themselves they are ranked by the weighted metric.

**Limits.** No guaranteed bit rate (5G-LENA's GBR resource type only selects the delay factor here), no per-class PRB quota, no slicing. The PDCP discard of `discard="pdcp_arrival"` checks the message at the head of the byte stream, which is the most important class after a reorder, not necessarily the oldest message. A message purged by `discard="purge"` from the middle of the stream leaves a gap that is skipped at the next step start.

## Proposed modes and switches

Every proposal keeps today's behavior as the default, so existing results and the bitwise backend tests stay valid. A field that a level cannot honor must appear in `unused_fields(level)`, or the level must refuse it, never ignore it silently.

### (a) Trivially exposable (plumbing only)

| Switch | API | Levels and backends |
|:---|:---|:---|
| L0 delay and loss, L0DR ranges | `NRConfig(l0_delay_median_steps=..., dr_delay_median_steps=(lo, hi), ...)` | **done** on this branch; all backends |
| L1 goodput factor | `NRConfig(l1_eta=0.8)` | **done**; reference, graph, compile, triton |
| Fading from speed | `NRConfig(ue_speed_mps=1.5, carrier_ghz=3.5)` | **done**; `L2` (legacy keeps its constexpr `RHO`); rho = 0 from the first zero of J0 (about 13.1 m/s at 3.5 GHz) |
| Ignored-field check | `cfg.unused_fields(level)`, `make_engine(..., strict=True)` | **done**; every level; switch-aware: fields gated by a switch (`dl`, `n_cells`, `noise_model`, `channel`, `blockage`, `blockage_model`, `los_source`, `fading`, `fading_doppler`, `fading_rician`, `fading_freq_corr`, `ul_pc`, `tbs_mode`, `gnb_antenna`, `ul_tpc`) count as read only when the switch makes the engine read them |
| Scheduler metric | `NRConfig(scheduler="pf" \| "rr" \| "maxci")`: the metric in `mac.py:150` becomes `rate / avg`, `1 / (slots since served)` or `rate` | `L2`; deferred until the NR multi-cell merge, which edits `mac.py` (now in) |
| OLLA clamp, PF initial average | `NRConfig(olla_max_db=10.0, pf_avg_init=100.0)` | `L2`; same deferral |
| Shadowing correlation distance | `NRConfig(shadow_dcorr_m=30.0, shadow_acf="exp", shadow_white_frac=0.5)` | **done** on `feat/channel` (`L2`, `NetSlotMC`); the default field is bitwise unchanged |
| Legacy MAC constants | pass `SR_DELAY`, `HARQ_RTT`, `HARQ_MAX`, `RLC_EXTRA`, `RHO`, `PF_T`, `PHR_MIN_DB` from `NRConfig` into `NetSlot`, `NetFast` and the kernel, which already takes them as constexprs | `L2-legacy`; the eager reference and the graph bodies read module globals, so each needs instance attributes; small but touches the frozen engine |

### (b) Moderate (new code that fits the tensor design)

| Switch | API | Interaction with levels and backends |
|:---|:---|:---|
| Engine-owned step RNG | `NRConfig(seed=..., rng="engine")` | **done** on `feat/protolevels` for every level except `L2`, which has its own engine RNG (`nr_rng.py`) |
| Traffic generators | **done** on `feat/traffic`: `NRConfig(traffic=[TrafficModel.periodic(...), .bursty(...), .video(...), .event(...), .policy()])`, see [Traffic models](#traffic-models) | `L2` only; they run inside the engine step because sub-step arrivals must gate the MAC, so the other levels refuse them |
| DL traffic generators | **done** on `feat/dltraffic`: `TrafficModel.<kind>(...).downlink()` / `direction="dl"`, see [Downlink models](#downlink-models) | `L2` with `dl=True`, reference and graph backends; triton refuses |
| FDD | **done** on `feat/dltraffic`: `NRConfig(duplex="fdd", dl_bandwidth_mhz=...)`, see [Duplexing](#duplexing-tdd-and-fdd) | `L2` reference and graph; triton refuses; the subband grid is shared by the two carriers |
| REM export | **done** on `feat/dltraffic`: `python -m isaac_net.tools.rem` / `isaac-net-rem` samples `RadioMC` on a grid ([rem.md](rem.md)) | any channel model, CPU |
| Several messages per robot per step | **done** for generated traffic (fixed `max_msgs_per_step` per model, arrival offset in slots); policy `Requests` stay one per step | `L2`; a multi-message `Requests(send=[E,R,M])` for the policy is still open |
| Configurable F, timeout and step for every level | `NRConfig(frame_buffer=32, timeout_steps=40, control_step_ms=50.0)` | **done** on `feat/protolevels`: every level and backend; fits record them |
| 38.901 path loss and LOS probability | `NRConfig(channel="tr38901_inf_sh")` (also `rma`, `uma`, `umi`, `inh`, `inf_sl`, `inf_dl`, `inf_dh`) | **done** on `feat/channel`: `RadioMC` model with LOS state, shadow fading and O2I; fast fading and MAC unchanged |
| Radio-map input | `NRConfig(channel="radio_map", radio_map_path="map.npz")` or `RadioMC(..., radio_map=RadioMap(gain [C, H, W], bounds))` | **done** on `feat/channel`: bilinear lookup in `RadioMC`; `tools/bake_radio_map_sionna.py` bakes a map with Sionna RT |
| Radio map from a USD scene | `IsaacNetCfg(scene_map=SceneRadioMapCfg(...))` at env creation, or `python -m isaac_net.tools.scene.bake --usd scene.usd --tx X Y Z --out map.pt` | **done** on `feat/usdmap`: USD export with ITU materials from semantic labels, material and prim names, Sionna RT bake, cache keyed by a stage hash; the engine's radio samples the map ([scene-radio-map.md](scene-radio-map.md)) |
| Robot blockage and per-robot Doppler | `NRConfig(blockage=True, fading_doppler="per_robot")` | **done** on `feat/channel`; per-robot Doppler needs the NR engine (`L2`) and pose input |
| Obstacles and NLOS | `NRConfig(los_source="raycast", radio_map_path="map.npz", los_diffraction=True, blockage=True, blockage_model="screen")`, `step(..., blockers=)` | **done** on `feat/obstacles` ([obstacles.md](obstacles.md)): `RadioMC` and `NetSlotMC`; `blockers=` and the `los` / `blocked` step keys on the NR engine (`L2`, reference and graph); no Triton change (per-step pathgain inputs only) |
| 5G-LENA MAC behavior under load | `NRConfig` fields `pf_update="rbg"`, `pf_avg_idle="freeze"`, `ul_retx_sched="tdma"`, `ul_amc_alloc="previous"`, `ul_grant_model="bsr"` (with `sr_boot_*`, `bsr_*`, `rlc_tail_*`); all on in the presets `lena_match_v2()` / `lena_validation_v2()` ([fidelity-load-gap.md](fidelity-load-gap.md)) | `L2`, reference and graph backends (graph bitwise), one or several cells; triton runs every switch except `ul_grant_model="bsr"`; defaults = the engine before the switches, bitwise |
| MIMO layers | `NRConfig(n_layers=2)` | TBS already takes `layers`; SINR per layer needs a rank model |
| Unified Isaac config | **Done (9c642ce).** The Isaac layer takes the same `NRConfig` as `make_engine`, with Isaac-only settings in `IsaacNetCfg` (`isaac/config.py`); observation features are chosen by name through `obs_features`, with one normalization and `obs_dim()`; domain-randomization ranges live in `IsaacNetCfg.dr_ranges` and `dr_support()` reports which levels honor them. `NetConfig` remains only as a deprecated alias with no defaults of its own. | closed |
| Background users, radio energy, multi-GPU sharding | **done** on `feat/bgenergy`: `NRConfig(background=BackgroundConfig(n_background=8), energy=EnergyConfig())`, `ShardedEngine(level, E, R, devices)`; see [background-energy-sharding.md](background-energy-sharding.md) | background: ghost robots on `L2`, an offered-load approximation on `L1` / `L2-legacy` (every backend), refused elsewhere; energy: per-slot on `L2`, airtime approximation elsewhere, every level and backend; sharding: bitwise shard-invariant for the engine-RNG levels, including `L2` without traffic models or background users |
| Per-env domain randomization for `make_engine` levels | `NRConfig` field ranges resolved per env at reset | needs per-env parameter tensors in the radio and MAC |

### (c) Large (needs design)

| Feature | Why it needs design |
|:---|:---|
| `graph` / `triton` backends for `L2` | every NR feature is reference-only, so none of them is usable at the scale the package is for |
| Slicing | per-slice PRB quotas or BWPs change the MAC state layout (QoS classes are done: [QoS scheduling](#qos-scheduling)) |
| Beamforming | per-beam gains and beam management interact with the scheduler and the interference model |
| RLC segmentation with status reports | the byte-stream queue would need per-SDU RLC state |

### (d) Out of scope

Carrier aggregation and bandwidth parts, dual connectivity and sidelink are out of scope for now. A robot fleet in one arena is served by one carrier, and each of these would multiply the state per robot for little effect on the tasks the package targets. Core-network and transport modeling beyond a fixed processing offset (`proc_offset_ms`) stays with the ns-3 bridges.

## What this branch plumbed

All new fields default to the earlier behavior. `tests/test_config_defaults.py` compares the defaults against inline copies of the old formulas, and a GPU run of `main` (`deb1b35`) against this branch was bitwise equal for `L0DR` (reference, graph, compile), `L1` (reference, graph, triton, compile) and `L2-legacy` (graph), with a partial reset in the middle of the run. The CPU suite passes (227 tests).

- `l0_delay_median_steps`, `l0_delay_log_sigma`, `l0_loss`: `make_engine("L0")` without `params` now takes them from the config. Before, it raised.
- `dr_delay_median_steps`, `dr_delay_log_sigma`, `dr_loss`: the per-env ranges `L0DR` draws at every reset, on every backend. `params` with the same keys override them.
- `l1_eta`: the `L1` goodput factor, on every backend. The Triton fluid kernel takes it as a constexpr.
- `ue_speed_mps`, `carrier_ghz`: when a speed is set, `fading_rho_per_ms` is derived from the Jakes correlation J0(2π f_D · 2.5 ms), spread over the milliseconds of that 2.5 ms interval. At 3 m/s and 3.5 GHz it gives 0.926 per 2.5 ms against the default 0.93. A set speed overrides an explicit `fading_rho_per_ms`.
- `FIELD_GROUPS`, `fields_read_by(level, cfg)`, `NRConfig.unused_fields(level)` and `make_engine(..., strict=True)`: each field belongs to one group, each level reads a known set of groups, and strict mode refuses a config that sets fields the level would ignore. `L2` reads the multi-cell group and `dl_interference` (group `nr_multicell`).

## Edge-computing loop

Until this addition a message counted as done when it reached the edge. `EdgeConfig` (in `NRConfig.edge`) adds what happens next: the edge processes the message, and the result travels back to the robot as a command. `make_engine` wraps the engine in `EdgeLoop` (`isaac_net/core/edge.py`) when `edge` is set, so a task switches the loop on through its config alone. `EdgeLoop(engine, EdgeConfig(...))` wraps an engine by hand. The wrapper reads only the engine's step dict (`delivered`, `cap`, `cls`, `delay`, `t`, `sinr_db`), so it works at every level and backend and leaves the engines untouched.

```python
from isaac_net import NRConfig, make_engine
from isaac_net.core import EdgeConfig

edge = EdgeConfig(servers_per_env=2, discipline="fifo", service_dist="exponential", service_ms=(15.0, 60.0),
                  queue_cap=16, deadline_ms=300.0, return_path="delay", ret_fixed_ms=2.0, cmd_bytes=200)
net = make_engine("L2-legacy", E, R, dev, NRConfig(edge=edge), backend="graph")
out = net.step(None, pos)            # engine keys + edge_* + act_* keys
obs_age = out["act_age"]             # age of the action each robot holds, in control steps
```

| Field | Default | Meaning |
|:---|:---|:---|
| `servers_per_env` | 1 | servers of the env's edge pool, shared by its R robots (PS: total service capacity) |
| `discipline` | `"fifo"` | `"fifo"`: non-preemptive, first come first served; `"ps"`: processor sharing, each of n messages gets min(1, servers / n) |
| `service_dist` | `"deterministic"` | or `"exponential"`, with mean `service_ms` |
| `service_ms` | 10.0 | mean service time; a tuple gives one mean per message class |
| `queue_cap` | 32 | waiting room; an arrival that finds `queue_cap + servers_per_env` messages at the edge is dropped |
| `deadline_ms` | None | from capture; a message not started (FIFO) or not finished (PS) by then is dropped |
| `max_events_per_step` | 2R + 2 servers + 8 | event budget of the loop per control step (see below) |
| `return_path` | `"instant"` | `"instant"`, `"delay"` or `"nr_dl"` |
| `ret_fixed_ms`, `ret_jitter_ms` | 1.0, 0.0 | `"delay"`: fixed part and uniform jitter |
| `cmd_bytes` | 100 | command size (`"delay"` transmission time, `"nr_dl"` DL message) |
| `ret_rate_eta`, `ret_share`, `ret_snr_offset_db` | 0.75, 1/R, 10 dB | `"delay"` rate = eta · share · bandwidth · log2(1 + SNR), SNR = the step's SINR + offset |
| `ret_inflight` | 4 | commands in flight per robot; a new one replaces the oldest when all are busy |

**Edge stage.** Arrivals are the messages the engine delivered, at time `cap + delay` on the env clock. The stage is an exact continuous-time event simulation. Every loop iteration moves each env to its next event (an arrival, a completion, a deadline, or the end of the control step), and all envs advance together with fixed-shape `[E, J]` job tables (J = `queue_cap + servers_per_env`). The loop runs at most `max_events_per_step` iterations. An env with more events than that stops early and resumes at the next step from the exact time it reached (`edge_lag`), so completion times stay exact and only their reporting slips by a step. Eager runs on a GPU leave the loop once every env has reached the end of the step (one host check every four iterations). Inside a CUDA-graph capture the loop always runs the full budget, and the extra iterations change nothing.

**Return path.** The newest result a robot gets in a step becomes a command. `"instant"` delivers it at the completion time. `"delay"` adds the fixed part, the jitter and the transmission time of `cmd_bytes` at a rate taken from the robot's SINR, so a robot at the cell edge gets its commands later. `"nr_dl"` needs level `L2` with `NRConfig(dl=True)`: the command becomes a downlink message of `cmd_bytes` in the NR engine, is scheduled by its DL MAC (HARQ, CQI, PF), and reaches the robot when that message completes. The NR engine takes new messages once per control step, so a command waits for the next step boundary before it enters the DL queue. `EdgeLoop` observes the DL frames through an instance-level wrapper of the DL link's `end_step`, which reads their completion times before the queue is compacted. A DL loss, a full DL queue or a replaced in-flight command counts in `cmd_dropped`.

**Loop accounting.** `act_cap` is the capture step of the newest action a robot received (the highest capture step, so a late older command never replaces a newer one), `act_age = t + 1 − act_cap` is its age at the end of the step, and `act_latency = act_ul_delay + act_edge_delay + act_ret_delay` splits capture-to-arrival into its three stages. `counters()` returns per-robot cumulative counts with `arrived == completed + dropped_full + dropped_deadline + at_edge`. How a task treats a stale action is its own choice. `examples/edge_control.py` compares holding the last command with stopping once `act_age` passes a threshold.

**Resets and graphs.** `reset(env_ids)` resets the engine and clears the edge state, the commands in flight and the counters of those envs only. `EdgeLoop(engine, cfg, graph=True)` captures the edge stage (not the engine) in a CUDA graph on its first step. It is bitwise equal to the eager stage with deterministic service, including partial resets, around the `graph` backend of `L2-legacy` (`tests/test_edge.py`). Random service and jitter draw from the default device generator, as the engines' graph backends do. `"nr_dl"` runs eagerly, since the NR engine has only its reference backend.

**Limits.** An arrival is admitted at its own time, but when the engine reports a message after the edge clock has passed its arrival (only possible after a lagged step), the edge admits it at its current time. Messages are processed one per job; batching at the edge is not modelled. The legacy `step(t, snr, hid)` form passes through without the edge stage. The `"delay"` return path computes its rate from the DL carrier, `dl_nprb` PRBs, which under `duplex="fdd"` may be wider or narrower than the UL carrier.

## Engine-owned randomness and the application constants (`feat/protolevels`)

`NRConfig.rng = "engine"` (the default) makes every draw of the prototype, surrogate and bound levels, on every backend, a pure function of (seed, env id, episode of that env, channel, call counter of that env, draw site, element), computed by a counter-based 32-bit hash (`proto/rng.py`; one Triton kernel per draw on CUDA, a torch fallback elsewhere). The seed is `make_engine(seed=...)` or `NRConfig.seed`. A policy's use of the global torch RNG never changes the network, an env's randomness in its k-th episode depends only on (seed, env, k) and its own inputs, and `reset(env_ids)` re-seeds exactly those envs. The draws are computed inside the captured graphs from device counters that `submit` / `step` advance before each replay, so the `graph` backend stays bitwise equal to the reference without injected draws; the `triton` backend hashes the same counters in its kernel (same uniforms, normals to float rounding). `rng="global"` keeps the earlier behavior (stepping draws from the global RNG, resets from the engine generator) and is bitwise equal to `main` at `1d533e9` for every level and backend (`tests/scripts/regress_main.py`). The low-level constructors (`netsim.make_net`, `NetFast`, `LevelNet`) default to `"global"`, so code that builds them directly and the injection tests are unchanged. `RadioMC` draws its reset randomness from the same engine streams, keyed by (seed, env id, episode).

`frame_buffer`, `timeout_steps`, `control_step_ms` and `proto_ul_slots_per_step` (default `control_step_ms / 2.5`, the legacy UL slot spacing) now configure every level and backend; the defaults resolve to the prototype constants (16, 20, 100 ms, 40). Delays, timeouts and the L0 / L0DR delay parameters stay in control steps, and the per-UL-slot MAC constants of `L2-legacy` (SR delay, HARQ RTT, fading correlation) stay per UL slot. `python -m isaac_net.tools.fit_levels --frame-buffer 32 --timeout-steps 40 --control-step-ms 50` fits under any values and records them in `meta`; `make_engine` refuses a fit whose values differ from the config (a fit file without them counts as the prototype values).
