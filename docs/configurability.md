# Configurability

Users should be able to choose the network model they train against, not inherit the choices baked into the prototype. This page records an audit of the package at `deb1b35` (2026-09-29). It lists every hard-coded choice found in the engines, compares the user-selectable features with ns-3 5G-LENA, Sionna SYS and Simu5G, and proposes the modes and switches to add, grouped by cost. The last section describes what the branch `feat/config-audit` already plumbed into `NRConfig`.

## How configuration works today

One `NRConfig` dataclass (`isaaclab_net/core/config.py`) configures every module, and `make_engine(level, E, R, device, config, backend)` builds every fidelity level. The configurable NR engine (`L2`) reads almost all of it. The other levels read much less. The prototype levels (`L0` to `L1`, `L2-legacy`), the fitted surrogates and the bounds read only the application fields, and they refuse a config whose frame buffer, timeout or control step differs from their compiled constants. `L2-legacy` with more than one cell or thermal noise runs `NetSlotMC`, which reads the radio, cell, power-control and handover fields but none of the NR MAC fields. Until this branch, a field that a level does not read was ignored without notice. For example, `make_engine("L0", config=NRConfig(pathloss_exp=3.0))` runs with the fixed legacy radio, and `make_engine("L2-legacy", config=NRConfig(olla_up_db=0.1))` runs with the legacy OLLA steps. `NRConfig.unused_fields(level)` now lists such fields and `make_engine(..., strict=True)` refuses them (see the last section).

## Inventory of hard-coded choices

"In NRConfig" says whether a field already controls the choice, and for which engine. "Fast backends" says whether changing it is safe for the `graph` and `triton` backends of the prototype levels: *constexpr* means the Triton kernel already receives it as a compile-time constant, so a new value only triggers a recompile. *Literal* means it is written into the kernel or the graph body and needs a code change first.

### Application and time

| Choice | Where | Value | In NRConfig | Fast backends |
|:---|:---|:---|:---|:---|
| Frame buffer depth | `proto/netsim.py:29` (`F`), `levels/base.py:33` | 16 frames per robot | `frame_buffer`, read by `L2` only; all other levels refuse other values | constexpr (`FB`); surrogates allocate `[E,R,16]` |
| Application timeout | `proto/netsim.py:30` (`TIMEOUT`) | 20 control steps (2 s) | `timeout_steps`, `L2` only | used in graph bodies as a Python constant |
| Control step | `proto/netsim.py:23` (`UL_PER_STEP`), `engine.py` check | 100 ms, 40 UL slots | `control_step_ms`, `L2` only | constexpr (`K`) |
| Message size classes | `config.py` `msg_sizes` | (4000, 30000) B | yes, every level | per-call tensor, safe |
| Isaac-side message sizes | `isaac/netmodule.py:54` | (1500, 12000) B | no: separate `NetConfig` with a different default | n/a |
| One message per robot per step | `traffic.py`, `Requests.send` | class index 0, 1, 2, ... | no | fixed shape `[E,R]` |
| Per-message tag | `Requests.det` / `hid` | one bool per message, one id per env | no | safe |
| Stack processing offset | `config.py` `proc_offset_ms` | 0 ms | yes, `L2` | n/a |
| Stepping randomness | every engine; `seed` covers reset draws only | global torch RNG | no | graph backends draw from the device's default Philox generator |

### Radio

| Choice | Where | Value | In NRConfig | Fast backends |
|:---|:---|:---|:---|:---|
| Path loss model | `proto/netsim.py:170` (legacy), `radio.py:73` | log-distance, 40 + 35 log10(d) | `pl_const_db`, `pathloss_exp` for `L2` and `NetSlotMC`; the prototype `Radio` is fixed | radio runs outside the graph |
| Minimum distance, 2-D distance | `radio.py:71-72`, `proto/netsim.py:169` | d ≥ 1 m, height ignored | no | safe |
| Shadowing | `radio.py:39-48`, `proto/netsim.py:161` | 6 dB, 8 plane waves, wavelengths uniform in 20–60 m | `shadow_sigma_db`, `shadow_modes`; `shadow_dcorr_m` and `shadow_white_frac` exist but no engine reads them | safe |
| gNB placement | `config.py` `cell_positions_m` | one gNB at the arena corner (0, 0) | yes (`cell_layout`, hex / grid / custom) for `L2` and `NetSlotMC`; `L1` and `QA` require the corner | safe |
| Noise floor | `proto/netsim.py:27`, `config.py` | −90 dBm per 10-PRB subband, includes interference | `noise_model`, `ni_fixed_dbm`, `gnb_nf_db`, `ue_nf_db` for `L2` and `NetSlotMC` | legacy constant |
| UE and gNB power | `proto/netsim.py:26`, `config.py` | 23 dBm, 43 dBm | `ue_tx_dbm`, `gnb_tx_dbm` (`L2`, `NetSlotMC`) | legacy constant |
| Fast fading | `proto/netsim.py:35`, `config.py` | AR(1) Rayleigh per subband, 0.93 per 2.5 ms (3 m/s at 3.5 GHz) | `fading`, `fading_rho_per_ms` for `L2`; now also `ue_speed_mps`, `carrier_ghz`; legacy fixed | constexpr (`RHO`) |
| DL SNR from UL SNR | `config.py` `dl_snr_offset_db` | UL + 10 dB when no DL SNR is given | yes, `L2` | n/a |
| LOS blockage | `isaac/netmodule.py:84` | 20 dB when the ray is blocked | Isaac `ParamRanges` only | n/a |

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
| Scheduler | `mac.py:142-150`, `proto/netsim.py:522-529` | proportional fair only | `pf_metric` subband / wideband, `pf_window`, `retx_priority` (NR) | constexpr (`PF_A`, `PF_B`) |
| PF average initial value and floor | `mac.py:30`, `proto/netsim.py:37, 475`, `netsim_mc.py:142` | 100 B/slot; floor 1 B/slot on the legacy engine only | no | constexpr (`PF_MIN`) |
| SR to grant delay (legacy) | `proto/netsim.py:31` | 2 UL slots | NR: `sr_period_slots`, `sr_grant_delay_slots`; legacy fixed | constexpr |
| HARQ RTT, max transmissions, RLC retry (legacy) | `proto/netsim.py:32-34` | 4 UL slots, 4 tx, +10 UL slots | NR: `ul_harq_rtt_slots`, `max_harq_tx`, `n_harq`, `harq_fail`, `rlc_retx_slots`; legacy fixed | constexpr |
| Power-headroom cap | `proto/netsim.py:38` | 3 dB per subband | NR `phr_cap`, `phr_min_db`; legacy fixed | constexpr (`PHR`) |
| UL power control | `proto/netsim_mc.py:222-224` | fractional, `NetSlotMC` only | `ul_pc*` (multi-cell legacy only) | reference only |
| Retransmission priority | `mac.py:160` | +1e9 on the PF metric | `retx_priority` on/off | n/a |
| DL CQI | `mac_dl.py:16-20` | best MCS per subband, mapped back to its threshold, reported every 10 slots | `cqi_period_slots`; the quantization rule is fixed (not the 38.214 CQI table) | n/a |

### Fidelity levels, surrogates and bounds

| Choice | Where | Value | In NRConfig | Fast backends |
|:---|:---|:---|:---|:---|
| L0 delay and loss | `proto/netsim.py` `NetDelay` | from `params`; no default (make_engine raised) | now `l0_delay_median_steps`, `l0_delay_log_sigma`, `l0_loss` | per-call floats |
| L0DR ranges | `proto/netsim.py` `L0DR_RANGES` (fixed in `_reset_state` on `main`) | median 0.05–10 steps log-uniform, σ 0.2–1.2, loss 0–0.2 | now `dr_delay_median_steps`, `dr_delay_log_sigma`, `dr_loss` | reset draws run eagerly, safe |
| L05 / L05Q bins | `proto/netsim.py:39-41` | active robots {2, 5, 9}, SNR {0, 10, 20, 30} dB, own queue {0, 1, 3} | no; fixed in the fit format | device tensors |
| QA efficiency and PF gain | `levels/__init__.py:23` | η = 0.9, PF diversity on | through `params` only | graph-safe |
| NN history and features | `levels/surrogates.py:23, 207` | 4-step history, SNR / 40, own queue / 16 | no; part of the fit format | graph-safe |
| Surrogate message semantics | `levels/base.py:10` | F = 16, 20-step timeout, 100 ms step | no; refused by the fit tool otherwise | graph-safe |
| ORACLE / NOCOMM | `levels/bounds.py` | delay 0 and never lost; never delivered, FIFO fills and overflows | no | graph-safe |

### Isaac layer and example task

| Choice | Where | Value | Notes |
|:---|:---|:---|:---|
| Observation features | `isaac/net_module.py:191-198`, `isaac/netmodule.py:518-526` | AoI clamped at 50 steps, SNR / 40 (or / 30), queue / 16, delivered flag, blocked flag | two different normalizations in the two Isaac modules; no feature selection |
| SNR sampling within a step | `isaac/netmodule.py:50` | `pose_chunks = 4` | Isaac `NetConfig` only |
| Domain randomization | `isaac/netmodule.py:77-88`, `isaac/mdp/events.py` | per-env uniform ranges of p_tx, noise, path loss, shadowing, blockage, background load, L0 delay | only on the Isaac engine; `make_engine` levels have no per-env parameter tensors |
| Background load | `isaac/netmodule.py:85` | fraction of subbands taken by other UEs | Isaac engine only |
| Fleet task | `examples/fleet_task.py:3-19` | imports `F`, `TIMEOUT` and the prototype `Radio` whatever the engine | task constants are class attributes |

## Feature matrix

The comparison was checked line by line against the official documentation of ns-3 5G-LENA `v5.1` (NR manual sources and `FEATURES.md` at that tag), NVIDIA Sionna SYS `v2.2.0` (API docs and tutorials) and Simu5G `v1.7.0` (simu5g.org user's guide and the repository at that tag), all accessed on 2026-09-29. [feature-matrix-sources.md](feature-matrix-sources.md) gives, for every (tool, feature) cell, the status, the feature name the tool's documentation uses, and the URL and section. The status words there are *supported*, *partial* and *not supported*. A cell confirmed only by source code, not by a manual, is marked as such there. "Have" means a user can select it today through `NRConfig` or `make_engine`. Sionna SYS is a set of system-level blocks on top of Sionna PHY and RT, so its cells name the companion package when the capability lives there.

| Feature | isaaclab-net | 5G-LENA | Sionna SYS | Simu5G | Our status and reason |
|:---|:---|:---|:---|:---|:---|
| Numerology | μ = 0, 1, 2 (`L2`) | μ = 0–4 (FR1, FR2) | no numerology model; any subcarrier spacing via Sionna PHY `ResourceGrid` | μ = 0–4, one per component carrier | **partial**: FR2 (μ = 3) missing, cheap once tables allow |
| Bandwidth / PRBs | any 38.101 FR1 value (`L2`) | any, up to 275 PRBs per BWP | any (`ResourceGrid`, scheduler `num_freq_res`) | any (`numBands` per carrier) | **have** (`L2`); legacy fixed at 50 PRB |
| TDD / FDD | any TDD string (`L2`) | TDD and FDD | none; one direction (DL or UL) per run | TDD (fixed DL/UL symbol split) and FDD | **partial**: FDD missing (an all-`U` pattern with a paired DL carrier) |
| Schedulers | PF (subband or wideband) | PF, RR, MR in TDMA and OFDMA, QoS-aware, random, RL-based | PF (SU-MIMO) | Max C/I (and variants), PF, DRR, QoS-aware PF | **partial**: RR and max-C/I are one line each in `mac.py` |
| HARQ | multi-process, chase or IR, max tx | IR and CC, multi-process, max retx | ACK/NACK feedback to link adaptation, no retransmissions | yes (processes, max retx) | **have** (`L2`) |
| RLC | AM retry or UM loss, PDCP discard | UM, AM, TM | none | UM, AM, TM | **partial**: no RLC segmentation timers or status reports |
| Link adaptation | BLER target, OLLA, MCS caps | AMC, error-model or Shannon based | inner and outer loop | CQI-based AMC | **have**; OLLA clamp and legacy steps fixed |
| Power control | UL fractional (legacy multi-cell only) | UL open and closed loop; DL uniform power allocation only | UL open loop, DL fair power | none documented (fixed transmit powers) | **partial**: not in the NR engine |
| MIMO / beamforming | none (one layer) | SU-MIMO up to rank 4, analog beamforming | SU-MIMO streams; precoding via Sionna PHY | none (incomplete MIMO removed in v1.4.3) | **missing**: layers could scale TBS and SINR (moderate); beamforming is large |
| Channel model | log-distance, plane-wave shadowing, AR(1) Rayleigh | 3GPP TR 38.901 (RMa, UMa, UMi, InH, V2V, NTN), NYUSIM (incl. InF), FTR, Sionna RT | TR 38.901 via Sionna PHY (UMi, UMa, RMa, InH, InF); ray tracing via Sionna RT | 3GPP TR 36.814, 36.873, 38.901 path loss, shadowing, Rayleigh or Jakes fading | **partial**: no 38.901 scenarios, no LOS probability, no radio map |
| Mobility | from the simulator's poses | ns-3 mobility models | random UT velocities in the topology generators; trajectories user-coded | INET mobility models, Veins | **have**: poses come from Isaac Lab, which is the point of the package |
| Traffic | one message per robot per step, size classes; DL bytes | NGMN, 3GPP XR, FTP Model 1, HTTP generators | none (scheduler takes rates only) | any INET application | **partial**: no periodic, bursty or video generators |
| UL / DL / sidelink | UL, DL (`L2`) | UL, DL; sidelink only in a separate v3.1-based branch | UL, DL | UL, DL, network-assisted D2D (prototype) | **partial**: sidelink out of scope for now |
| QoS / slicing | none | 5QI QoS schedulers, BWP-based slicing | none | 5QI QoS flows, SDAP, QoS-aware PF; no slicing | **missing** |
| Multi-cell / handover | 1–7 cells, A3 handover (legacy); NR in progress | multi-cell, X2 handover, hex wraparound | multi-cell hex layouts, wraparound; no handover | multi-cell, X2 handover, background cells | **partial**: NR multi-cell on `feat/nr-multicell` |
| Interference | same-slot UL (legacy MC); NR hook | all co-channel transmitters, incl. DL–UL cross-link | inter-cell, in the post-equalization SINR | inter-cell DL and UL (configurable), background cells | **partial** |
| Carrier aggregation / BWP | none | CA and BWPs | none | CA; BWPs not documented | **out of scope** for robot fleets on one carrier |

## Proposed modes and switches

Every proposal keeps today's behavior as the default, so existing results and the bitwise backend tests stay valid. A field that a level cannot honor must appear in `unused_fields(level)`, or the level must refuse it, never ignore it silently.

### (a) Trivially exposable (plumbing only)

| Switch | API | Levels and backends |
|:---|:---|:---|
| L0 delay and loss, L0DR ranges | `NRConfig(l0_delay_median_steps=..., dr_delay_median_steps=(lo, hi), ...)` | **done** on this branch; all backends |
| L1 goodput factor | `NRConfig(l1_eta=0.8)` | **done**; reference, graph, compile, triton |
| Fading from speed | `NRConfig(ue_speed_mps=1.5, carrier_ghz=3.5)` | **done**; `L2` (legacy keeps its constexpr `RHO`) |
| Ignored-field check | `cfg.unused_fields(level)`, `make_engine(..., strict=True)` | **done**; every level |
| Scheduler metric | `NRConfig(scheduler="pf" \| "rr" \| "maxci")`: the metric in `mac.py:150` becomes `rate / avg`, `1 / (slots since served)` or `rate` | `L2`; deferred to avoid conflicts with the NR multi-cell merge, which edits `mac.py` |
| OLLA clamp, PF initial average | `NRConfig(olla_max_db=10.0, pf_avg_init=100.0)` | `L2`; same deferral |
| Shadowing correlation distance | make `RadioMC` draw wavelengths from `shadow_dcorr_m` (today a fixed 20–60 m) | `L2`, `NetSlotMC`; needs a mapping that keeps the default field bitwise, so it waits for the `radio.py` owner |
| Legacy MAC constants | pass `SR_DELAY`, `HARQ_RTT`, `HARQ_MAX`, `RLC_EXTRA`, `RHO`, `PF_T`, `PHR_MIN_DB` from `NRConfig` into `NetSlot`, `NetFast` and the kernel, which already takes them as constexprs | `L2-legacy`; the eager reference and the graph bodies read module globals, so each needs instance attributes; small but touches the frozen engine |

### (b) Moderate (new code that fits the tensor design)

| Switch | API | Interaction with levels and backends |
|:---|:---|:---|
| Engine-owned step RNG | `make_engine(..., seed=..., step_generator=True)` | the draws already sit at fixed positions per robot; they would come from `net.gen` instead of the global RNG, so a policy's own sampling no longer shifts the network's random stream. Graph capture needs a registered generator (`graph.register_generator_state`) |
| Traffic generators | `traffic=TrafficModel.periodic(period_steps=5, cls=1, phase="random")`, `.poisson(rate)`, `.bursty(on, off)`, `.video(fps, gop)` returning `Requests` each step | pure torch outside the engine, so every level and backend works unchanged |
| Several messages per robot per step | `Requests(send=[E,R,M])` | enqueue loops over M; fixed shape keeps graphs valid |
| Configurable F, timeout and step for every level | `NRConfig(frame_buffer=32, timeout_steps=10)` accepted by `L0` to `L1` and the surrogates | the kernel takes `FB` and `K` as constexprs; the eager and graph bodies need instance fields; surrogate fits must record them |
| 38.901 path loss and LOS probability | `NRConfig(channel="log_distance" \| "tr38901_inf_sh" \| "tr38901_umi" \| "tr38901_inh")` | a new `RadioMC` path-loss function per scenario; fast fading and MAC unchanged |
| Radio-map input | `channel="radio_map"`, `radio_map=RadioMap(tensor [C, H, W], origin, resolution)` | bilinear lookup of a precomputed Sionna RT map in `RadioMC.rx_dbm`; the map is data outside the repository |
| UL power control in the NR engine | `NRConfig(ul_pc=True)` for `L2` | port of the `NetSlotMC` rule into `UlMac._split` |
| MIMO layers | `NRConfig(n_layers=2)` | TBS already takes `layers`; SINR per layer needs a rank model |
| Unified Isaac config | Isaac `NetConfig` folded into `NRConfig`; observation features chosen by name, `obs=("aoi", "sinr", "queue", "delivered")` | removes the two normalizations and the differing size defaults |
| Per-env domain randomization for `make_engine` levels | `NRConfig` field ranges resolved per env at reset | needs per-env parameter tensors in the radio and MAC |

### (c) Large (needs design)

| Feature | Why it needs design |
|:---|:---|
| `graph` / `triton` backends for `L2` | every NR feature is reference-only, so none of them is usable at the scale the package is for |
| QoS classes and slicing | priority queues per robot, a QoS-aware scheduler and per-class PRB quotas change the FIFO and the MAC state layout |
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
- `FIELD_GROUPS`, `fields_read_by(level, cfg)`, `NRConfig.unused_fields(level)` and `make_engine(..., strict=True)`: each field belongs to one group, each level reads a known set of groups, and strict mode refuses a config that sets fields the level would ignore. The map must be updated when the NR multi-cell merge makes `L2` read the multi-cell group.
