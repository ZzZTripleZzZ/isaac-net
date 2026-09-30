# Calibration against public data

Public measurement data cannot validate a whole 5G uplink model, but it can pin down individual layers: the latency structure of real open-source stacks, the throughput a PF scheduler delivers under contention, the statistics of large-scale fading, and the shape of the link abstraction. This page records which datasets were used, the problems found in them, the fitted parameters, the presets that came out of the fits, and the layers public data could not reach. The calibration was run on 2026-09-29 against the legacy slot-level engine (NetSlot, now `L2-legacy`) through a subclass whose extra knobs default to bit-exact NetSlot behavior. The raw data (about 15 GB) stays on the lab box, and the scripts and small outputs (`params_*.json`, per-task CSVs) are in the project's research workspace.

## Datasets and licenses

| Dataset | License (how verified) | What was used |
|:---|:---|:---|
| Zenodo 13754300, 5G campus QoS, OAI and srsRAN (Raffeck et al., CNSM 2024) | CC BY 4.0 (Zenodo API) | two `all_Packets` CSVs (2.75 + 2.82 GB), throughput CSVs and the README; the redundant `Packets_with_IATs` files were skipped |
| ColO-RAN (wineslab) | GPL-3.0 (LICENSE file read) | the git repository, 7.6 GB of CSV on disk |
| POWDER, Zenodo 19463551 (NR drive test, UE RSRP with position) | CC BY 4.0 (Zenodo API) | 108 kB |
| POWDER, Zenodo 18272105 (Viavi CBRS LTE scan with GPS) | CC BY 4.0 (Zenodo API) | 2.2 MB |
| Lumos5G v1.0 CSV (GitHub mirror `LaveshM/lumos5g` of the IEEE DataPort release) | CC BY 4.0 per the DataPort page; the mirror itself has no LICENSE file | 8.5 MB |
| AERPAW Ericsson 5G (Dryad) | CC0 (API metadata) | **not obtained**: the file download needs a Dryad bearer token |
| Berlin V2X, AI4Mobile iV2I+ | IEEE DataPort | **not obtained**: a DataPort login is required and there is no direct link |

No dataset is redistributed with the package, and no fitted parameter file is committed. ColO-RAN in particular is GPL-3.0, and none of its data is copied into the repository. The facts taken from the CNSM paper are μ = 1 (0.5 ms slots), a single Quectel RM520N-GL UE on the same host as the gNB (so UE and gNB captures share one clock), the UL slots per TDD pattern from its Table II, and traffic that is either deterministic at 1.7 ms or exponential.

## Data problems found

**Whole-second timestamps in the Zenodo campus data.** Every NTNU srsRAN run and three NTNU OAI 40 MHz configurations carry whole-second timestamps, so their one-way delay (OWD) is always 0 or 1000 ms. These configurations are flagged and excluded, and the WUE site is clean for both stacks.

**ColO-RAN has no uplink contention.** ColO-RAN is srsLTE FDD downlink with 4 Mb/s CBR eMBB traffic, and its uplink rates are about 0.02 Mb/s. It was therefore used only to check the PF and queue core in a downlink-equivalent setup, and its KPIs are 250 ms windows without per-TTI data.

**No public NR uplink with more than one UE.** None of the reachable datasets has multi-UE NR uplink traffic, per-TB BLER against SINR, or position-tagged channel data at robot speeds.

**Sionna 2.2.0 BLER tables.** These are not measurement data, but the defects found in them while building the PHY tables are listed in [validation-5g-lena.md](validation-5g-lena.md).

## Uplink latency (Zenodo 13754300)

The per-packet OWD from UE egress to GTP at the gNB was compared with a one-robot engine run for each uplink configuration. The harness steps the engine once per UL slot and maps slot indices to wall time with the paper's TDD table, merges 35-byte packets that arrive inside one UL gap into one frame, and uses a static bench UE at 20 dB SNR. The fit set is the 32 WUE configurations with deterministic arrivals, and the held-out set is the WUE exponential-arrival configurations plus the clean NTNU OAI configurations (36 configurations). The metrics are the Wasserstein-1 distance W1 in ms and the two-sample KS statistic, with a constant processing offset d0 fitted as a nuisance parameter. The grid covered the SR delay (1–16 UL slots), K2 as a grant lead (0–2 slots), the HARQ RTT (4 or 6), and three knobs the legacy engine lacked: the SR period (every UL slot, 5, 10 or 20 ms), the OLLA BLER target (10%, 2% or 0.5%) and proactive grants (off, every UL slot, or once per TDD period).

The measured OWD differs sharply between the stacks. OAI with 5-slot TDD periods has a p50 of 3.3–4.3 ms and a p95 of about 5 ms (the exception, 5-slot 2:1, has a p95 of 34 ms). OAI with 10-slot periods has a p50 of 5.0 ms at 4:1 and 9.6 ms at 2:1, and with 20-slot periods a p50 of 16.7 ms and a p99 of 54–79 ms. srsRAN has a p50 of 12.4–16 ms and a p95 of about 21.6 ms for every pattern, almost independent of the TDD pattern.

| Stack / subset | Engine defaults | Best shared fit | Fitted parameters |
|:---|:---|:---|:---|
| srsRAN, fit set | W1 2.8 ms, KS 0.33 | W1 1.25 ms, KS 0.12 | SR delay 1 UL slot, SR period 20 ms, BLER target 0.5%, d0 2.5 ms |
| srsRAN, held out | W1 2.7 ms, KS 0.33 | W1 2.0–2.9 ms, KS 0.19–0.28 | same |
| OAI, 5/10-slot periods, fit set | W1 3.3 ms, KS 0.40 | W1 2.5 ms, KS 0.26 | proactive grant once per TDD period, BLER target 0.5%, d0 2.0 ms |
| OAI, 5/10-slot periods, held out | W1 3.6 ms, KS 0.45 | W1 3.1 ms, KS 0.29 | same |
| OAI, 20-slot periods | W1 6.5 ms, KS 0.45 | W1 3.0 ms, KS 0.15 | SR delay 6, SR period 20 ms, 10% BLER target, d0 7.25 ms |

The legacy SR model, a fixed delay with an SR opportunity in every UL slot, matches neither stack. srsRAN is explained by an SR and grant cycle of about 20 ms, which sets its 13 ms median, and OAI at short periods is explained by proactive UL grants. Tuning the SR delay alone never fits, with a best W1 2–3× worse than the shared fits. A 10% OLLA target also makes the HARQ tail too heavy for a bench link, while a 0.5% target reproduces OAI's p90 of about 5–6 ms at 5-slot periods. K2 and the HARQ RTT change W1 by less than 0.1 ms, so these data cannot identify them. OAI is not one model: the best parameters differ by configuration (per-configuration KS 0.07–0.23), the 5-slot 2:1 pattern has a 30–50 ms tail that no setting reproduces, and 20-slot periods need an extra 5 ms of processing offset, which suggests an implementation effect. The latency is also invariant to the number of environments, with p50 of 5.12, 5.03 and 5.16 ms at E = 1, 8 and 64 (W1 against E = 1 of 0.17 and 0.10 ms, which is seed noise). At DDDSU the engine defaults give a p50 of about 5 ms, between OAI's 4.0 ms at 5-slot 4:1 and srsRAN's 12.4 ms.

These fits rest on one UE, one bench channel and 35-byte packets. They ignore the UL symbols of the special slot, and d0 lumps together the UE modem, the gNB L2 and GTP.

## Contention (ColO-RAN)

The contention unit is one eMBB slice of 2 UEs sharing 6–42 PRBs, 2.2 M windows in total. The engine was configured to match: a 1 ms TTI, subbands of one 3-PRB RBG, no UL power split and no SR (the eNB knows the downlink buffer), a HARQ RTT of 8 TTIs, and 2 UEs per slice with SNR drawn from the reported `dl_snr`. Because slicing fixes 2 UEs per slice, the contention axis is PRBs per UE, which under homogeneous PF is equivalent to UE count at fixed PRBs. The table gives per-UE throughput in Mb/s:

| Slice PRBs | 6 | 12 | 18 | 24 | 30 | 36 | 42 |
|:---|---:|---:|---:|---:|---:|---:|---:|
| data | 0.67 | 1.44 | 2.17 | 2.81 | 3.35 | 3.75 | 4.02 |
| engine, η = 1.0 | 0.93 | 1.84 | 2.77 | 3.63 | 3.94 | 4.00 | 4.00 |
| engine, η = 0.8 | 0.75 | 1.47 | 2.22 | 2.96 | 3.63 | 3.90 | 3.99 |

A capacity scale of η = 0.8 gives a count-weighted MAPE of 6%, against 28% at the default η = 1.0, so the uncalibrated engine overestimates capacity by about 25–35% below saturation. Jain fairness per window is 0.964 in the data and 0.978–0.99 in the engine. The qualitative pattern is reproduced: throughput grows linearly in PRB share, both buffers sit at their cap until the demand knee, and the buffer then collapses. Two mismatches remain. The measured knee is gradual (p10 per-UE throughput is still 3.35 Mb/s at 42 PRBs, against 3.84 in the engine) and the engine underestimates the window-to-window spread, and the engine is slightly too fair. Both are consistent with the engine missing the slow channel and attach dynamics of the emulator.

## Channel (POWDER drive tests)

The only downloadable position-tagged RSRP sets are two POWDER outdoor campus drives: 2 NR cells over about 200 × 200 m, and 17 CBRS LTE cells over about 1.5 km. The base-station positions are unpublished, so each cell was fitted jointly for its position, intercept and exponent by grid search, and the exponent is reported with a profile interval (residual standard deviation within 0.25 dB of the best).

| Parameter | Engine | NR cells (2) | CBRS cells (17), median [IQR] |
|:---|---:|:---|:---|
| path-loss exponent n | 3.5 | 1.6 and 4.3, profiles [1.5, 2.0] and [3.25, 6.0] | 3.97 [2.1, 5.6], profiles typically span 1.5–6 |
| shadowing σ (dB), n free | 6.0 | 4.2, 5.8 | 6.4 [5.6, 7.7] |
| σ at n = 3.5 (dB) | 6.0 | 5.5, 6.0 | 6.9 [5.7, 7.7] |
| decorrelation distance, 1/e (m) | 10.3 | 18, 26 | 39 [28, 55] |
| autocorrelation at 2.5 m | 0.95 | 0.46 | about 0.4–0.6 |

σ = 6 dB lies inside the measured range and stays. The exponent cannot be identified from these data, so they give no evidence against n = 3.5. The engine's shadowing field (8 plane waves with wavelengths uniform in 20–60 m) decorrelates 2–4× faster than the measured shadowing and has a negative lobe of −0.18 at 22 m that the data lack. The data also put about half of the variance into a component that is already decorrelated at 2.5 m, from fast fading, measurement noise or GPS misalignment, which the engine does not have. The suggested change is an exponential-autocorrelation field with a decorrelation distance of about 20–40 m plus a white component of about 0.5σ². `NRConfig(shadow_acf="exp", shadow_dcorr_m=30.0, shadow_white_frac=0.5)` applies it (see [channels.md](channels.md)). These are outdoor 3.5 GHz campus drives with unknown base-station locations, far from an indoor warehouse, and residual trend inflates the decorrelation distance.

## Link abstraction

On ColO-RAN (LTE downlink, 114k UE-windows), the achieved bits per PRB-TTI rise from 130 at 9 dB to 285 at 19–21 dB, a steady 0.39–0.46 of the legacy engine's 0.9 · 144 · 0.75 · log2(1 + SNR). A gap fit gives bits = 51 · log2(1 + SNR / 2.0 dB), a Shannon gap of 3.0 dB. The shape matches, and the scale partly reflects LTE's roughly 120 data REs per PRB against the engine's 144 and srsLTE's SNR report. The Spearman correlation between SNR and MCS is only 0.58. On the Zenodo uplink iperf3 data, the peak spectral efficiency that would make the engine's peak UL goodput equal the measurement is 2.3–3.4 for OAI with 5/10-slot periods, 0.6–0.95 for OAI with 20-slot periods, and 2.4–5.1 for srsRAN at 20 MHz (inflated because the special-slot UL symbols are ignored), while several srsRAN 40 MHz configurations show about 0.1, which means a broken uplink. The legacy SE_MAX of 5.5 is therefore about 2× optimistic for open-source uplink on a bench. On Lumos5G (mmWave downlink, per-second samples), normalized throughput against SINR follows the engine's normalized SE curve with a mean absolute error of 0.054, and is steeper between 3 and 9 dB (Spearman 0.46). This is shape evidence only. The logistic BLER slope of 1.5 dB⁻¹ was **not** validated, because no public set has per-TB BLER against SINR.

## Presets

The fits became presets of `NRConfig` in `isaaclab_net/core/config.py`, which the NR engine (`L2`) uses directly:

| Preset | Settings | Source |
|:---|:---|:---|
| `srsran_like()` | SR period 20 ms (40 slots), SR-to-grant 1 UL slot, HARQ RTT 4 UL slots, BLER target 0.5%, processing offset 2.5 ms, no proactive grants, UL MCS ≤ 15 | srsRAN latency fit (W1 1.25 ms, KS 0.12 on the fit set); the MCS cap gives SE 2.41, the measured bench UL ceiling of 2.4–3 |
| `oai_like()` | proactive UL grant once per TDD period, BLER target 0.5%, processing offset 2.25 ms, SR period 20 ms and SR-to-grant 10 UL slots (irrelevant with proactive grants), HARQ RTT 4 UL slots, UL MCS ≤ 15 | OAI 5/10-slot latency fit; OAI with 20-slot periods needs about 7.25 ms of offset instead |
| `lena_match` = `lena_like()` | the 5G-LENA scenario: 50 PRB in 5 RBGs, 16 HARQ processes, 4 transmissions, UM loss, no OLLA, no PHR cap, wideband PF, 5G-LENA EESM tables and IR combining, LENA TB size, PDCP arrival discard, thermal noise with a gNB NF of 18.44 dB, 50 bytes of overhead per 1400-byte packet | the ns-3 reference, not public data ([validation-5g-lena.md](validation-5g-lena.md)); `lena_validation()` adds the validation geometry |

The capacity scale η ≈ 0.8 from ColO-RAN and the proposed shadowing-field change are not applied in any preset. They are recorded as calibration knobs, and the shadowing change is available through `shadow_acf`, `shadow_dcorr_m` and `shadow_white_frac`.

## What public data could not validate

Seven layers need the lab gNB (up to 4 UEs) or POWDER (up to 2 UEs).

1. **NR uplink multi-UE contention.** No public dataset has more than one NR UE on the uplink, and ColO-RAN is LTE downlink. The lab gNB with 2–4 UEs can anchor PF sharing, buffer dynamics and SR collisions at small N, and many-UE behavior still needs 5G-LENA.
2. **Per-TB BLER against SINR, the OLLA target and HARQ statistics.** Only coarse BLER is public. gNB MAC logs are needed.
3. **SR period, proactive grants and K2 as actually configured.** The Zenodo fit infers them, and lab gNB configurations plus MAC traces could confirm the values in the presets.
4. **Latency under load and with large frames.** Zenodo has only 35-byte packets every 1.7 ms from one UE, far from 4–30 kB robot frames and multi-UE queueing.
5. **Indoor warehouse channel.** Exponent, σ and decorrelation distance were fitted only on outdoor drives. RSRP/SINR maps on a grid in a robot-like space are needed.
6. **Fading correlation against robot speed.** No public trace has per-slot channel data at robot speeds.
7. **Robot uplink traffic.** There is no public trace of robot traffic over 5G.
