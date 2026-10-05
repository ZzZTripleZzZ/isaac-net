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
| Fast fading | `proto/netsim.py:35`, `config.py`, `channels/doppler.py` | AR(1) Rayleigh per subband, 0.93 per 2.5 ms (3 m/s at 3.5 GHz) | `fading`, `fading_rho_per_ms`, `ue_speed_mps`, `carrier_ghz`, and per-robot Doppler from each robot's speed (`fading_doppler="per_robot"`) for `L2`; legacy fixed | constexpr (`RHO`) |
| DL SNR from UL SNR | `config.py` `dl_snr_offset_db` | UL + 10 dB when no DL SNR is given | yes, `L2` | n/a |
| LOS blockage | `isaac/netmodule.py:84`, `channels/blockage.py` | 20 dB when the ray is blocked | Isaac `ParamRanges`; robot bodies as spheres in `RadioMC` (`blockage`, `blockage_radius_m`, `blockage_loss_db`, off by default) | fixed-shape `[E,R,R,C]` test |

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
| PF average initial value and floor | `mac.py:48, 56`, `proto/netsim.py:37, 475`, `netsim_mc.py:142` | 100 B/slot; floor 1 B/slot on the legacy engine, 1e-9 B/slot on the NR engine (`AVG_MIN`, as 5G-LENA) | no | constexpr (`PF_MIN`, `AVG_MIN`) |
| SR to grant delay (legacy) | `proto/netsim.py:31` | 2 UL slots | NR: `sr_period_slots`, `sr_grant_delay_slots`; legacy fixed | constexpr |
| HARQ RTT, max transmissions, RLC retry (legacy) | `proto/netsim.py:32-34` | 4 UL slots, 4 tx, +10 UL slots | NR: `ul_harq_rtt_slots`, `max_harq_tx`, `n_harq`, `harq_fail`, `rlc_retx_slots`; legacy fixed | constexpr |
| Power-headroom cap | `proto/netsim.py:38` | 3 dB per subband | NR `phr_cap`, `phr_min_db`; legacy fixed | constexpr (`PHR`) |
| UL power control | `mac_ul.py` (`pc_backoff`), `proto/netsim_mc.py:222-224` | fractional, on by default with more than one cell | `ul_pc*` (`L2` and multi-cell legacy) | reference only |
| Retransmission priority | `mac.py:243-290` | admitted retransmissions win RBGs before new data (`retx_priority`); off: they compete on the PF metric and RBGs of short ones are released | `retx_priority` on/off | same rule in the `triton` kernel |
| DL CQI | `mac_dl.py:16-20` | best MCS per subband, mapped back to its threshold, reported every 10 slots | `cqi_period_slots`; the quantization rule is fixed (not the 38.214 CQI table) | n/a |

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
| TDD / FDD | any TDD string (`L2`) | TDD and FDD | none; one direction (DL or UL) per run | TDD (fixed DL/UL symbol split) and FDD | **partial**: FDD missing (an all-`U` pattern with a paired DL carrier) |
| Schedulers | PF (subband or wideband) | PF, RR, MR in TDMA and OFDMA, QoS-aware, random, RL-based | PF (SU-MIMO) | Max C/I (and variants), PF, DRR, QoS-aware PF | **partial**: RR and max-C/I are one line each in `mac.py` |
| HARQ | multi-process, chase or IR, max tx | IR and CC, multi-process, max retx | ACK/NACK feedback to link adaptation, no retransmissions | yes (processes, max retx) | **have** (`L2`) |
| RLC | AM retry or UM loss, PDCP discard | UM, AM, TM | none | UM, AM, TM | **partial**: no RLC segmentation timers or status reports |
| Link adaptation | BLER target, OLLA, MCS caps | AMC, error-model or Shannon based | inner and outer loop | CQI-based AMC | **have**; OLLA clamp and legacy steps fixed |
| Power control | UL fractional (`L2` and legacy multi-cell) | UL open and closed loop; DL uniform power allocation only | UL open loop, DL fair power | none documented (fixed transmit powers) | **partial**: no DL power control |
| MIMO / beamforming | none (one layer) | SU-MIMO up to rank 4, analog beamforming | SU-MIMO streams; precoding via Sionna PHY | none (incomplete MIMO removed in v1.4.3) | **missing**: layers could scale TBS and SINR (moderate); beamforming is large |
| Channel model | log-distance with correlated and white shadowing; TR 38.901 RMa, UMa, UMi, InH, InF-SL/DL/SH/DH path loss with spatially consistent LOS state and O2I; precomputed radio maps (Sionna RT baking tool); robot-body blockage; AR(1) Rayleigh with per-robot Doppler | 3GPP TR 38.901 (RMa, UMa, UMi, InH, V2V, NTN), NYUSIM (incl. InF), FTR, Sionna RT | TR 38.901 via Sionna PHY (UMi, UMa, RMa, InH, InF); ray tracing via Sionna RT | 3GPP TR 36.814, 36.873, 38.901 path loss, shadowing, Rayleigh or Jakes fading | **have** large-scale models ([channels.md](channels.md)); **missing**: 38.901 fast fading (clusters, K-factor), online ray tracing |
| Mobility | from the simulator's poses | ns-3 mobility models | random UT velocities in the topology generators; trajectories user-coded | INET mobility models, Veins | **have**: poses come from Isaac Lab, which is the point of the package |
| Traffic | policy messages (one per robot per step, size classes), periodic / bursty / video / event generators (`L2`); DL bytes | NGMN, 3GPP XR, FTP Model 1, HTTP generators | none (scheduler takes rates only) | any INET application | **partial**: periodic, bursty, video and event generators on `L2` only; no downlink generators |
| UL / DL / sidelink | UL, DL (`L2`) | UL, DL; sidelink only in a separate v3.1-based branch | UL, DL | UL, DL, network-assisted D2D (prototype) | **partial**: sidelink out of scope for now |
| QoS / slicing | none | 5QI QoS schedulers, BWP-based slicing | none | 5QI QoS flows, SDAP, QoS-aware PF; no slicing | **missing** |
| Multi-cell / handover | 1–7 cells, per-cell PF and HARQ, A3 handover with interruption (`L2` and legacy) | multi-cell, X2 handover, hex wraparound | multi-cell hex layouts, wraparound; no handover | multi-cell, X2 handover, background cells | **partial**: no wraparound, no X2 data forwarding model |
| Radio link failure | N310/N311/T310/T311 on the serving-link SINR, re-establishment at the best suitable cell after a fixed delay, queue carry or flush; A3 target admission `a3_min_target_rsrp_dbm` (`L2`, several cells; [multicell.md](multicell.md#radio-link-failure)) | *unverified*: RLF via the ns-3 LTE RRC; A3 `MinTargetRsrpDbm` | not assessed | not assessed | **partial**: uplink SINR once per control step, no RACH contention or handover-failure model |
| Interference | same-slot UL and DL per RBG (`L2`), UL (legacy) | all co-channel transmitters, incl. DL–UL cross-link | inter-cell, in the post-equalization SINR | inter-cell DL and UL (configurable), background cells | **partial** |
| Wi-Fi (802.11) uplink | level `WIFI`: mean-field DCF / EDCA contention per sub-step, 802.11ax / ac / a rates with SNR-threshold rate adaptation, A-MPDU, RTS/CTS, several APs with RSSI association and co-channel sharing, optional hidden nodes ([wifi.md](wifi.md)) | not in 5G-LENA; ns-3 has a separate `wifi` module | none | not in Simu5G; INET, which Simu5G builds on, has 802.11 models | **partial**: a validated mean-field abstraction, not a packet-level 802.11 model; no downlink, OFDMA or MU-MIMO |
| Fidelity per env | cheap and expensive level side by side, per env: static mix, load-triggered switching with queue handoff, curriculum ([adaptive-fidelity.md](adaptive-fidelity.md)) | — | — | — | **have** (`core/adaptive.py`, `NRConfig.fidelity`) |
| Carrier aggregation / BWP | none | CA and BWPs | none | CA; BWPs not documented | **out of scope** for robot fleets on one carrier |
| Differentiable KPIs | fluid relaxations L1D / QAD: gradients of delay, delivery, AoI and energy w.r.t. send probability, message size, transmit power and position ([differentiable.md](differentiable.md)) | not assessed | not assessed | not assessed | **partial (exploratory)**: fluid models only; the L2 scheduler and HARQ are not differentiated |
| Real-stack validation | OAI 5G in rfsim mode behind a lockstep bridge, 1–10 UEs ([bridges-oai.md](bridges-oai.md)) | n/a | n/a | n/a | **have**, for validation only (not a model feature) |

## Traffic models

`NRConfig(traffic=...)` takes one `TrafficModel` or a list of them (`isaac_net.core.traffic`). They run inside the engine step of level `L2` and put messages into the uplink queues without the policy emitting them; the policy's `submit()` keeps working next to them.

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

**Limits.** Uplink only. `priority` is carried and reported, but the MAC serves each robot's queue in FIFO order. `deadline_ms` is reported as `deadline_miss` and does not drop messages. Traffic models work with one cell and with several NR cells (`n_cells > 1`). The engine gates arrivals through four hooks on its `UlMac` instance (`sr_step`, `slot`, `end_step`, and `handover`, so that a `ho_rlc="flush"` handover spares messages that arrive later in the step). If a future `NRNet` stops calling the first three, `step()` raises instead of silently mis-timing.

Example: [`isaac_net/examples/traffic_models.py`](https://github.com/ZzZTripleZzZ/isaac-net/blob/main/isaac_net/examples/traffic_models.py).

## Proposed modes and switches

Every proposal keeps today's behavior as the default, so existing results and the bitwise backend tests stay valid. A field that a level cannot honor must appear in `unused_fields(level)`, or the level must refuse it, never ignore it silently.

### (a) Trivially exposable (plumbing only)

| Switch | API | Levels and backends |
|:---|:---|:---|
| L0 delay and loss, L0DR ranges | `NRConfig(l0_delay_median_steps=..., dr_delay_median_steps=(lo, hi), ...)` | **done** on this branch; all backends |
| L1 goodput factor | `NRConfig(l1_eta=0.8)` | **done**; reference, graph, compile, triton |
| Fading from speed | `NRConfig(ue_speed_mps=1.5, carrier_ghz=3.5)` | **done**; `L2` (legacy keeps its constexpr `RHO`); rho = 0 from the first zero of J0 (about 13.1 m/s at 3.5 GHz) |
| Ignored-field check | `cfg.unused_fields(level)`, `make_engine(..., strict=True)` | **done**; every level; switch-aware: fields gated by a switch (`dl`, `n_cells`, `noise_model`, `channel`, `blockage`, `fading`, `fading_doppler`, `ul_pc`, `tbs_mode`) count as read only when the switch makes the engine read them |
| Scheduler metric | `NRConfig(scheduler="pf" \| "rr" \| "maxci")`: the metric in `mac.py:150` becomes `rate / avg`, `1 / (slots since served)` or `rate` | `L2`; deferred until the NR multi-cell merge, which edits `mac.py` (now in) |
| OLLA clamp, PF initial average | `NRConfig(olla_max_db=10.0, pf_avg_init=100.0)` | `L2`; same deferral |
| Shadowing correlation distance | `NRConfig(shadow_dcorr_m=30.0, shadow_acf="exp", shadow_white_frac=0.5)` | **done** on `feat/channel` (`L2`, `NetSlotMC`); the default field is bitwise unchanged |
| Legacy MAC constants | pass `SR_DELAY`, `HARQ_RTT`, `HARQ_MAX`, `RLC_EXTRA`, `RHO`, `PF_T`, `PHR_MIN_DB` from `NRConfig` into `NetSlot`, `NetFast` and the kernel, which already takes them as constexprs | `L2-legacy`; the eager reference and the graph bodies read module globals, so each needs instance attributes; small but touches the frozen engine |

### (b) Moderate (new code that fits the tensor design)

| Switch | API | Interaction with levels and backends |
|:---|:---|:---|
| Engine-owned step RNG | `NRConfig(seed=..., rng="engine")` | **done** on `feat/protolevels` for every level except `L2`, which has its own engine RNG (`nr_rng.py`) |
| Traffic generators | **done** on `feat/traffic`: `NRConfig(traffic=[TrafficModel.periodic(...), .bursty(...), .video(...), .event(...), .policy()])`, see [Traffic models](#traffic-models) | `L2` only; they run inside the engine step because sub-step arrivals must gate the MAC, so the other levels refuse them |
| Several messages per robot per step | **done** for generated traffic (fixed `max_msgs_per_step` per model, arrival offset in slots); policy `Requests` stay one per step | `L2`; a multi-message `Requests(send=[E,R,M])` for the policy is still open |
| Configurable F, timeout and step for every level | `NRConfig(frame_buffer=32, timeout_steps=40, control_step_ms=50.0)` | **done** on `feat/protolevels`: every level and backend; fits record them |
| 38.901 path loss and LOS probability | `NRConfig(channel="tr38901_inf_sh")` (also `rma`, `uma`, `umi`, `inh`, `inf_sl`, `inf_dl`, `inf_dh`) | **done** on `feat/channel`: `RadioMC` model with LOS state, shadow fading and O2I; fast fading and MAC unchanged |
| Radio-map input | `NRConfig(channel="radio_map", radio_map_path="map.npz")` or `RadioMC(..., radio_map=RadioMap(gain [C, H, W], bounds))` | **done** on `feat/channel`: bilinear lookup in `RadioMC`; `tools/bake_radio_map_sionna.py` bakes a map with Sionna RT |
| Radio map from a USD scene | `IsaacNetCfg(scene_map=SceneRadioMapCfg(...))` at env creation, or `python -m isaac_net.tools.scene.bake --usd scene.usd --tx X Y Z --out map.pt` | **done** on `feat/usdmap`: USD export with ITU materials from semantic labels, material and prim names, Sionna RT bake, cache keyed by a stage hash; the engine's radio samples the map ([scene-radio-map.md](scene-radio-map.md)) |
| Robot blockage and per-robot Doppler | `NRConfig(blockage=True, fading_doppler="per_robot")` | **done** on `feat/channel`; per-robot Doppler needs the NR engine (`L2`) and pose input |
| 5G-LENA MAC behavior under load | `NRConfig` fields `pf_update="rbg"`, `pf_avg_idle="freeze"`, `ul_retx_sched="tdma"`, `ul_amc_alloc="previous"`, `ul_grant_model="bsr"` (with `sr_boot_*`, `bsr_*`, `rlc_tail_*`); all on in the presets `lena_match_v2()` / `lena_validation_v2()` ([fidelity-load-gap.md](fidelity-load-gap.md)) | `L2`, reference and graph backends (graph bitwise), one or several cells; triton runs every switch except `ul_grant_model="bsr"`; defaults = the engine before the switches, bitwise |
| MIMO layers | `NRConfig(n_layers=2)` | TBS already takes `layers`; SINR per layer needs a rank model |
| Unified Isaac config | **Done (9c642ce).** The Isaac layer takes the same `NRConfig` as `make_engine`, with Isaac-only settings in `IsaacNetCfg` (`isaac/config.py`); observation features are chosen by name through `obs_features`, with one normalization and `obs_dim()`; domain-randomization ranges live in `IsaacNetCfg.dr_ranges` and `dr_support()` reports which levels honor them. `NetConfig` remains only as a deprecated alias with no defaults of its own. | closed |
| Background users, radio energy, multi-GPU sharding | **done** on `feat/bgenergy`: `NRConfig(background=BackgroundConfig(n_background=8), energy=EnergyConfig())`, `ShardedEngine(level, E, R, devices)`; see [background-energy-sharding.md](background-energy-sharding.md) | background: ghost robots on `L2`, an offered-load approximation on `L1` / `L2-legacy` (every backend), refused elsewhere; energy: per-slot on `L2`, airtime approximation elsewhere, every level and backend; sharding: bitwise shard-invariant for the engine-RNG levels, including `L2` without traffic models or background users |
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

**Limits.** An arrival is admitted at its own time, but when the engine reports a message after the edge clock has passed its arrival (only possible after a lagged step), the edge admits it at its current time. Messages are processed one per job; batching at the edge is not modelled. The legacy `step(t, snr, hid)` form passes through without the edge stage.

## Engine-owned randomness and the application constants (`feat/protolevels`)

`NRConfig.rng = "engine"` (the default) makes every draw of the prototype, surrogate and bound levels, on every backend, a pure function of (seed, env id, episode of that env, channel, call counter of that env, draw site, element), computed by a counter-based 32-bit hash (`proto/rng.py`; one Triton kernel per draw on CUDA, a torch fallback elsewhere). The seed is `make_engine(seed=...)` or `NRConfig.seed`. A policy's use of the global torch RNG never changes the network, an env's randomness in its k-th episode depends only on (seed, env, k) and its own inputs, and `reset(env_ids)` re-seeds exactly those envs. The draws are computed inside the captured graphs from device counters that `submit` / `step` advance before each replay, so the `graph` backend stays bitwise equal to the reference without injected draws; the `triton` backend hashes the same counters in its kernel (same uniforms, normals to float rounding). `rng="global"` keeps the earlier behavior (stepping draws from the global RNG, resets from the engine generator) and is bitwise equal to `main` at `1d533e9` for every level and backend (`tests/scripts/regress_main.py`). The low-level constructors (`netsim.make_net`, `NetFast`, `LevelNet`) default to `"global"`, so code that builds them directly and the injection tests are unchanged. `RadioMC` draws its reset randomness from the same engine streams, keyed by (seed, env id, episode).

`frame_buffer`, `timeout_steps`, `control_step_ms` and `proto_ul_slots_per_step` (default `control_step_ms / 2.5`, the legacy UL slot spacing) now configure every level and backend; the defaults resolve to the prototype constants (16, 20, 100 ms, 40). Delays, timeouts and the L0 / L0DR delay parameters stay in control steps, and the per-UL-slot MAC constants of `L2-legacy` (SR delay, HARQ RTT, fading correlation) stay per UL slot. `python -m isaac_net.tools.fit_levels --frame-buffer 32 --timeout-steps 40 --control-step-ms 50` fits under any values and records them in `meta`; `make_engine` refuses a fit whose values differ from the config (a fit file without them counts as the prototype values).
