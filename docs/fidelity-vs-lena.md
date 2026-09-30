# Fidelity of the NR engine against ns-3 5G-LENA

This page is the formal comparison between the NR engine (level `L2`, `isaaclab_net/core/nr_engine.py`) and ns-3.48 with 5G-LENA v5.1. It replays every run of the 5G-LENA sweep described in [validation-5g-lena.md](validation-5g-lena.md) with identical per-UE link budgets and identical offered traffic, compares delay, delivery, throughput, HARQ and PRB use per run and over the sweep, splits the one fitted parameter into a fit set and a hold-out set, ablates the alignment switches one at a time, adds the legacy slot-level engine (`L2-legacy`) and the 5G-LENA fading arm as separate rows, and reports simulation speed. It makes no claim about RL policies. Every number below is in a CSV under `benchmarks/fidelity/results/`, and `python benchmarks/fidelity/report.py` regenerates every table from those files.

## Summary

Over the 153 runs of the primary (no-fading) arm, with the SR-to-grant delay fitted on seeds 1 and 3 only:

- **Delay.** The median p95-delay error is −7.8% (median absolute error 9.2%, 90th percentile 36%), and the median p50 error is −7.2% (absolute 9.7%). On the 51 held-out runs (seed 2) the median p95 error is −6.8% and the median p50 error −0.2%. The median Wasserstein-1 distance between the delay distributions is 12.8 ms.
- **Delivery and throughput.** The median drop-rate difference is −0.63 percentage points (90th percentile of the absolute difference 8.1 pp), and the median absolute cell-goodput error is 0.6%.
- **HARQ.** First-transmission BLER agrees to a median of +0.12 pp (LENA 0.28%, NR 0.46%), and the retransmitted-TB fraction to +0.09 pp.
- **PRB use.** The engine grants 9.7% fewer PRBs than 5G-LENA (median), 13% fewer at light load and the same at saturation.
- **Where it fails.** The gap is concentrated in loaded cells. In runs where 5G-LENA drops 1–20% of frames (moderate) the engine's p50 is 27% low and it drops 4.1 pp fewer frames, and in saturated runs (5G-LENA drop ≥ 20%) the p50 is 35% low and the drop rate 6.9 pp low. The engine is optimistic in both cases.
- **KS distance.** The median KS distance is 0.28 against an engine replica-to-replica floor of 0.007, so the two delay distributions remain statistically distinguishable even where quantiles agree to a few percent (see the KS note under Caveats).
- **Legacy engine.** The slot-level engine that the scale results use (`L2-legacy`) has a median p95 error of −19% (absolute 36%) and a KS distance of 0.50 on the same runs, and in saturated runs it drops 32 pp fewer frames than 5G-LENA. The NR engine is the better LENA match on every delay metric and on saturation.
- **Speed.** The NR engine's reference backend is not faster than 5G-LENA for one small cell: 0.47–0.81 s of wall time per simulated second on one CPU thread (E = 1) against 0.015–0.93 s for 5G-LENA, with break-even at N = 64. Batching 16 envs on one thread brings it to 0.03–0.17 s per env-second.

## Method

### What is replayed

The reference is the primary arm of the 5G-LENA sweep: N ∈ {1, 2, 4, 8, 16, 32, 64} UEs, frame size S ∈ {4000, 30000} bytes, nominal load f ∈ {0.1, 0.3, 0.6, 0.9, 1.2} and 3 seeds (drops), 30 s of traffic each, 153 runs. The scenario, the coverage-conditioned drop and the reasons for the frequency-flat channel are in [validation-5g-lena.md](validation-5g-lena.md).

For every run the NR engine receives:

- **the same link budget per UE**: the single-subband full-power SNR `snr1_db` from the run's `ues.csv` (shadowing included), fed as the engine's fixed-SNR input;
- **the same traffic, frame by frame**: the exact list of (UE, 100 ms instant, S) of every frame the 5G-LENA run generated, read from the run's `frames.csv`, rather than a fresh Bernoulli draw with the same p. The engine therefore offers the same frames as 5G-LENA in every run, and all remaining differences come from the network models;
- **the same duration**: 300 traffic steps (0.5–30.5 s) plus 2.2 s of drain, the 5G-LENA simulation end.

The configuration is `lena_validation()` (the `lena_match` preset in the validation geometry, [validation-5g-lena.md](validation-5g-lena.md) lists every switch) on the `reference` backend, which is the only backend of the NR engine. The runs execute on CPU (2 threads per process, 8 processes, on the shared lab box). Each run is replayed with 4 engine replicas that differ only in the engine's random draws (TB decoding). The torch seed depends on N only, so every ablation arm sees the same random stream (common random numbers) and arm-to-arm differences are not sampling noise. The 5G-LENA side is one ns-3 run per seed.

### What is measured

A frame is on time if all its packets arrive within 2 s of generation. The drop rate counts late, lost, discarded and unfinished frames, as in the 5G-LENA parser (`parse_run.py`). For each run, with the four NR replicas pooled:

| Quantity | Definition |
|:---|:---|
| KS | two-sample Kolmogorov–Smirnov distance between the delays of on-time frames, from 5G-LENA's per-frame log (not from its 101-point CDF) |
| W1 | Wasserstein-1 distance between the same two samples, in ms |
| p50 / p95 / p99 error | (NR − LENA) / LENA of the delay quantile |
| Drop Δ | NR drop rate minus LENA drop rate, in percentage points |
| Goodput error | relative error of cell goodput (on-time bytes / 30 s); per UE the same per UE |
| BLER Δ, retx Δ | difference in first-transmission BLER and in the fraction of TBs that are retransmissions |
| retx TB count error | relative error of the number of retransmitted TBs, only where 5G-LENA has at least 20 |
| PRB error | relative error of PRB utilization over the UL slots of the 30 s traffic window |
| Floor | the same KS, W1 and p95 error between two NR replicas of the same run |

Aggregates are medians over runs (signed, and of the absolute value) and 90th percentiles of the absolute value. Runs are grouped by N, by nominal load f, by frame size and by load regime, which is defined on the 5G-LENA side: **light** (drop < 1%, 79 runs), **moderate** (1–20%, 36 runs) and **saturated** (≥ 20%, 38 runs). UEs are split into thirds of the pooled `snr1_db` distribution of all 3,621 primary-arm UEs: **edge** (below 16.3 dB), **mid** and **centre** (at or above 22.5 dB). Every UE has at least 7 dB by construction of the drop.

**Per-UE HARQ counts on the 5G-LENA side were recomputed.** The per-UE TB columns of the sweep's `per_ue.csv` attribute every TB to UE 0, because the `nodeId` column of the UE buffer trace is constant and the RNTI-to-UE map built from it collapses. With fading off and whole-band power, every UL TB of a UE is received at exactly that UE's whole-band SNR, so `lena_extract.py` maps each RNTI to the UE whose `snr_bw_db` equals the RNTI's median PHY SINR. The match error is 0.000 dB in all 153 runs, with no unmapped RNTI.

### Which numbers are fitted

One engine parameter was fitted to 5G-LENA's outputs: the SR-to-first-PUSCH delay `sr_grant_delay_slots`, set to 40 slots (20 ms) in `lena_validation()` from a sensitivity sweep over the N ≤ 4 runs of all seeds. Every other alignment value comes from the 5G-LENA configuration (numerology, TDD, noise figure, HARQ, RLC, discard, scheduler) or was read from 5G-LENA's traces without reference to the compared KPIs (50 bytes of overhead per 1400-byte packet, from the 4150-byte buffer of a 4000-byte frame). To remove the in-sample fit, the delay was refitted here on a grid of 10 values using **seeds 1 and 3 only** (102 runs), minimizing the median absolute p50 error plus the median absolute p95 error, and evaluated on **seed 2** (51 runs), which the refit never sees:

| SR→PUSCH slots | Fit objective | Fit p50 | Fit p95 | Fit KS | Hold-out p50 | Hold-out p95 | Hold-out \|p50\| | Hold-out \|p95\| | Hold-out KS | Hold-out W1 ms |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 (1.5 ms), engine default | 0.610 | -38.5% | -19.4% | 0.465 | -41.3% | -20.9% | 41.3% | 20.9% | 0.539 | 21.0 |
| 8 (4.0 ms) | 0.572 | -35.4% | -18.2% | 0.462 | -37.6% | -20.2% | 37.6% | 20.2% | 0.505 | 18.6 |
| 14 (7.0 ms), measured SR→PUSCH | 0.522 | -30.6% | -17.1% | 0.450 | -31.4% | -17.8% | 31.4% | 17.8% | 0.471 | 16.0 |
| 20 (10.0 ms) | 0.457 | -24.8% | -15.9% | 0.422 | -25.1% | -16.4% | 25.1% | 16.4% | 0.442 | 13.5 |
| 26 (13.0 ms) | 0.388 | -21.1% | -14.4% | 0.393 | -18.9% | -16.0% | 18.9% | 16.0% | 0.425 | 11.0 |
| 32 (16.0 ms) | 0.266 | -14.6% | -10.3% | 0.310 | -7.4% | -9.9% | 7.6% | 9.9% | 0.400 | 6.6 |
| 36 (18.0 ms) | 0.266 | -14.6% | -10.3% | 0.310 | -7.4% | -9.9% | 7.6% | 9.9% | 0.400 | 6.6 |
| **40 (20.0 ms), chosen on the fit seeds** | **0.205** | -10.0% | -7.8% | 0.267 | **-0.2%** | **-6.8%** | **7.5%** | **8.5%** | 0.351 | 5.2 |
| 44 (22.0 ms) | 0.221 | -5.4% | -5.4% | 0.271 | +3.6% | -5.1% | 12.8% | 6.1% | 0.351 | 5.9 |
| 50 (25.0 ms) | 0.225 | -2.0% | -4.5% | 0.305 | +7.3% | -4.3% | 17.9% | 6.5% | 0.378 | 7.6 |

The refit on seeds 1 and 3 selects the same 40 slots, and the held-out errors (p50 −0.2%, p95 −6.8%) are no worse than the in-sample ones. 32 and 36 slots give identical results because the SR is raised at the special slot and the grant can only take effect at a UL data slot, so the delay acts in steps of the 5-slot TDD period. In the rest of this page, **pre-fit** means the engine's default of 3 slots (gNB processing + K2), and **post-fit** means 40 slots. All tables below are post-fit unless they say otherwise. The primary-arm tables use all 153 runs, and the hold-out table repeats the headline subsets on seed 2 alone.

**The fitted 20 ms is not 5G-LENA's SR delay.** The SR-to-grant step was measured directly from the MAC control-message traces: from each SR a UE transmits to the next UL DCI that UE receives, the per-run median is 4.5 ms (interquartile range over runs 4.5–6.0 ms, the same at light and heavy load, `lena_sr_dci.csv`). With about 2.5 ms from DCI to PUSCH in the inspected trace, that is about 7 ms or 14 slots from SR to PUSCH. Using that measured value (`sr_measured14`) leaves the p50 31% low. The remaining 13 ms of light-load access delay therefore comes from other parts of 5G-LENA's grant pipeline, for example the small first grants and the BSR round trip visible in the traces (in the inspected trace, one 4150-byte frame went out as six TBs of 750–1611 bytes, about 6.2 KB in total, over six UL slots). The engine has no model of that pipeline, and the 40-slot SR delay is a lumped stand-in for it. It aligns light-load delays but cannot reproduce the capacity that 5G-LENA's grant padding costs at high load, which is the saturated-regime gap below.

## Results

### Sweep-level accuracy (post-fit, all 153 runs)

| Metric | Runs | Median (signed) | Median \|·\| | 90th pct \|·\| |
|:---|---:|---:|---:|---:|
| KS distance, delay of on-time frames | 153 | 0.278 | 0.278 | 0.529 |
| KS, engine replica vs replica (floor) | 153 | 0.007 | 0.007 | 0.019 |
| Wasserstein-1, delay (ms) | 153 | 12.8 | 12.8 | 172.7 |
| p50 delay rel. error | 153 | -7.2% | 9.7% | 54.5% |
| p95 delay rel. error | 153 | -7.8% | 9.2% | 35.8% |
| p99 delay rel. error | 153 | -6.8% | 7.7% | 32.1% |
| drop rate NR − LENA | 153 | -0.63 pp | 0.63 pp | 8.12 pp |
| cell goodput rel. error | 153 | +0.6% | 0.6% | 13.4% |
| per-UE goodput \|rel. error\| (median over UEs) | 153 | 0.0% | 0.0% | 14.9% |
| first-tx BLER NR − LENA | 153 | +0.12 pp | 0.50 pp | 1.92 pp |
| retx TB fraction NR − LENA | 153 | +0.09 pp | 0.51 pp | 2.34 pp |
| retx TB count rel. error (LENA ≥ 20 retx) | 111 | -35.5% | 87.5% | 146.3% |
| PRB utilization rel. error | 153 | -9.7% | 9.7% | 28.9% |

The per-UE goodput error is 0 for most UEs because most UEs in light and moderate runs deliver every frame on both sides. The retransmitted-TB count error is large while the retransmission fraction agrees, because 5G-LENA sends about twice as many TBs as the engine for the same traffic (median NR/LENA TB-count ratio 0.55, 10th–90th percentile 0.22–0.85), so counts are not comparable and fractions are the meaningful HARQ metric.

### By load regime, UE count, offered load and frame size

KS is shown with the replica-to-replica floor in parentheses. Errors are signed medians over runs, and goodput is the median absolute error.

| Subset | Runs | KS (floor) | W1 ms | p50 err | p95 err | p99 err | Drop Δ | Goodput \|err\| | PRB err |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all runs | 153 | 0.278 (0.007) | 12.8 | -7.2% | -7.8% | -6.8% | -0.63 pp | 0.6% | -9.7% |
| light (LENA drop < 1%) | 79 | 0.257 (0.005) | 3.5 | -0.2% | -9.2% | -9.7% | +0.00 pp | 0.0% | -13.3% |
| moderate (1–20%) | 36 | 0.316 (0.007) | 66.0 | -26.9% | -15.2% | -12.8% | -4.09 pp | 4.5% | -5.4% |
| saturated (≥ 20%) | 38 | 0.379 (0.012) | 132.9 | -35.2% | -2.6% | +0.0% | -6.89 pp | 11.8% | +0.0% |
| N = 1 | 9 | 1.000 (0.000) | 0.1 | -0.1% | -0.1% | -0.1% | +0.00 pp | 0.0% | -12.5% |
| N = 2 | 12 | 0.498 (0.010) | 4.8 | -4.0% | -5.3% | -5.0% | +0.00 pp | 0.0% | -11.7% |
| N = 4 | 21 | 0.364 (0.005) | 5.0 | -7.3% | -5.7% | -5.3% | +0.00 pp | 0.0% | -11.2% |
| N = 8 | 24 | 0.266 (0.006) | 6.2 | -3.8% | -7.8% | -10.9% | +0.00 pp | 0.0% | -10.9% |
| N = 16 | 27 | 0.252 (0.007) | 17.5 | -0.2% | -14.1% | -10.7% | -3.29 pp | 3.4% | -9.2% |
| N = 32 | 30 | 0.231 (0.009) | 46.5 | -17.0% | -11.1% | -6.1% | -4.38 pp | 5.0% | -4.4% |
| N = 64 | 30 | 0.247 (0.009) | 98.7 | -22.6% | -11.0% | -6.6% | -5.52 pp | 7.9% | -2.7% |
| f = 0.1 | 42 | 0.226 (0.008) | 2.6 | -0.1% | -6.4% | -9.3% | +0.00 pp | 0.0% | -17.3% |
| f = 0.3 | 36 | 0.167 (0.006) | 6.7 | -0.2% | -12.3% | -11.6% | +0.00 pp | 0.0% | -14.6% |
| f = 0.6 | 30 | 0.329 (0.007) | 56.0 | -26.6% | -14.2% | -12.2% | -2.88 pp | 3.0% | -6.4% |
| f = 0.9 | 24 | 0.385 (0.008) | 117.6 | -35.0% | -4.8% | -1.7% | -6.82 pp | 8.6% | +0.0% |
| f = 1.2 | 21 | 0.356 (0.011) | 124.1 | -23.1% | -2.3% | +0.0% | -6.54 pp | 11.9% | +0.0% |
| S = 4000 | 63 | 0.279 (0.005) | 5.8 | -0.2% | -10.8% | -11.5% | -0.11 pp | 0.1% | -23.5% |
| S = 30000 | 90 | 0.274 (0.010) | 27.3 | -12.5% | -4.9% | -3.6% | -1.33 pp | 1.9% | -7.6% |

Small-N rows cover less of the load range, because p saturates at 1 (N = 1 with 4 KB reaches only 4% load), so the N and f rows are not independent. The cross table N × regime is in `summary_primary.csv` (group `N_x_regime`). It shows the saturated gap at every N ≥ 4: the saturated p50 error is −58% at N = 4 (4 runs), −42% at N = 8 (4), −39% at N = 16 (4), −29% at N = 32 (11) and −26% at N = 64 (13), with drop differences between −5.6 and −8.9 pp.

**Hold-out seed only (post-fit, seed 2, never used in the refit):**

| Subset | Runs | KS (floor) | W1 ms | p50 err | p95 err | p99 err | Drop Δ | Goodput \|err\| | PRB err |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all hold-out runs | 51 | 0.351 (0.007) | 5.2 | -0.2% | -6.8% | -5.3% | +0.00 pp | 0.0% | -11.3% |
| light | 34 | 0.266 (0.006) | 2.6 | -0.1% | -7.6% | -6.8% | +0.00 pp | 0.0% | -13.9% |
| moderate | 8 | 0.357 (0.007) | 90.7 | -30.7% | -18.3% | -11.9% | -4.09 pp | 4.5% | -1.6% |
| saturated | 9 | 0.396 (0.011) | 175.2 | -37.2% | -3.1% | -0.9% | -6.49 pp | 11.0% | +0.0% |

Seed 2 has more light-load runs (34 of 51, against 45 of 102 on the fit seeds), which is why its all-run medians look better than the full sweep's. Within each regime the held-out errors match the full sweep's.

**What drives the loaded-cell gap.** In moderate runs 5G-LENA drops a median 5.4% of frames and the engine 0.04%, and in saturated runs 39% against 33%, while 5G-LENA uses more PRBs to do it (0.97 against 0.90 in moderate runs). Two 5G-LENA behaviors that the engine does not have account for this, as far as the traces show.

- **Grant overhead.** 5G-LENA carries the same traffic in about twice as many TBs (TB-count ratio 0.55) and grants about 15% more PRBs at light load (the engine's PRB error there is −13%). When the cell is full, that overhead is lost capacity, so 5G-LENA saturates earlier and queues longer. This matches the open BSR-quantization and grant-padding item in [validation-5g-lena.md](validation-5g-lena.md) and is the most likely main cause.
- **A link-adaptation failure mode for some UEs.** 38 UEs in 23 runs have a 5G-LENA first-transmission BLER above 10% (at least 50 TBs). In the clearest case (seed 1, UE 12, whole-band SNR 6.35 dB) 5G-LENA's AMC picks MCS 10 for 1181-byte TBs whose TB error rate the PHY trace gives as 0.886, and the three retransmissions report the same 0.886, so HARQ combining does not help and 196 TBs are lost after 4 transmissions in one run. The engine never shows this (0.25 TBs lost per replica over the whole sweep, against 1,145 for 5G-LENA). Removing these UEs from the comparison shrinks the moderate-regime delivery gap from 4.1 to 3.1 pp and leaves the saturated gap unchanged.

### Cell edge and cell centre

Per-UE errors are medians over UEs, and the pooled KS is the median over runs of the KS between all on-time frames of that class.

| Class | UEs | snr1 median dB | UE p50 err | UE p95 err | UE \|p95 err\| | Pooled KS | UE goodput \|err\| 90th pct | retx TBs LENA / NR | TBs NR / LENA |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| edge | 1207 | 13.0 | -10.6% | -5.9% | 11.4% | 0.266 | 141.7% | 10824 / 3931 | 0.34 |
| mid | 1203 | 19.5 | -12.6% | -7.4% | 14.1% | 0.364 | 31.1% | 2928 / 3133 | 0.33 |
| centre | 1211 | 28.5 | -24.3% | -19.7% | 27.2% | 0.568 | 0.0% | 692 / 1268 | 0.41 |

Delay error is largest at the cell centre, not at the edge. Centre UEs use high MCS, so the engine sends their frames in few large TBs, while 5G-LENA's delay for them is set by its grant pipeline rather than by the link, which the engine does not model. At the edge the delays agree better, but per-UE goodput spreads widely in loaded runs (90th percentile of the absolute error 142%), because the engine and 5G-LENA starve different weak UEs once the cell is full. 5G-LENA retransmits about 2.8× more edge TBs than the engine, which includes the AMC failure mode above, while the engine retransmits more at the centre. The total retransmission fractions agree within 0.1 pp.

### Ablations

Each arm changes one switch of `lena_validation()` and replays all 153 runs with the same random stream. Signed and absolute medians over runs:

| Arm | Switch | KS | W1 ms | p50 err | \|p50\| | p95 err | \|p95\| | \|drop Δ\| | \|goodput\| | BLER Δ | PRB err |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| primary | none (post-fit) | 0.278 | 12.8 | -7.2% | 9.7% | -7.8% | 9.2% | 0.63 pp | 0.6% | +0.12 pp | -9.7% |
| olla_on | `olla=True` (10% BLER target) | 0.237 | 10.2 | -3.8% | 9.7% | -1.2% | 5.4% | 1.37 pp | 1.5% | +8.72 pp | -3.4% |
| phr_cap_wholeband | `phr_cap=True` | 0.278 | 12.8 | -7.2% | 9.7% | -7.8% | 9.2% | 0.63 pp | 0.6% | +0.12 pp | -9.7% |
| ul_power_alloc | `ul_power="allocated"` | 0.245 | 12.6 | -0.1% | 9.9% | -9.7% | 12.4% | 0.40 pp | 0.4% | +0.32 pp | -12.8% |
| ul_power_alloc_phr | `ul_power="allocated"`, `phr_cap=True` | 0.240 | 18.3 | -0.2% | 9.4% | -15.9% | 16.9% | 0.40 pp | 0.4% | +0.38 pp | -16.8% |
| harq1 | `n_harq=1` | 0.278 | 12.8 | -7.2% | 9.7% | -7.8% | 9.2% | 0.63 pp | 0.6% | +0.12 pp | -9.7% |
| sr_default3 | `sr_grant_delay_slots=None` (3, pre-fit) | 0.499 | 28.2 | -39.2% | 40.0% | -20.6% | 21.1% | 0.63 pp | 0.6% | +0.08 pp | -9.7% |
| sr_measured14 | `sr_grant_delay_slots=14` | 0.461 | 23.2 | -30.9% | 30.9% | -17.8% | 19.3% | 0.63 pp | 0.6% | +0.10 pp | -9.7% |
| bler_sionna | `bler_source="pdsch"` (Sionna tables) | 0.422 | 35.2 | -18.2% | 19.4% | -32.2% | 32.2% | 0.63 pp | 0.6% | +1.25 pp | -33.2% |

Attribution, in order of effect:

1. **SR-to-grant delay** is the largest single term for delay: without it (pre-fit) the p50 is 39% low and the p95 21% low. It changes nothing else, because it only shifts when data starts.
2. **BLER tables** are the largest term for PRB use and the tail: the Sionna AWGN curves put every MCS threshold 2.4–5.7 dB lower than 5G-LENA's EESM tables, so the engine picks higher MCS, uses 33% fewer PRBs than 5G-LENA and gets a p95 that is 32% low. Drop and goodput do not change, because both tables keep every UE of this drop decodable.
3. **OLLA** improves some delay medians but is wrong on the mechanism: with a 10% BLER target the engine's first-transmission BLER rises by 8.7 pp to about 9%, against 0.3% in 5G-LENA, which has no OLLA. Its better p95 probably comes from the larger TBs of a more aggressive MCS rather than from a better model, and it doubles the drop error.
4. **UE power split** (`allocated` instead of whole-band power) lowers per-RB SINR for multi-RBG grants, which moves the p50 closer and the tail further away, and with the power-headroom cap the p95 error doubles to −16%.
5. **Power-headroom cap alone and 16 vs 1 HARQ processes are inert in this geometry**, and bit-identical to the primary arm on every run. The PHR cap only acts on the `allocated` power path (the whole-band path grants every RBG). With DDDSU at μ = 1, the UL HARQ round trip (3 slots) is shorter than the UL-slot spacing (5 slots) and retransmissions take priority, so a UE never has a second TB in flight and additional HARQ processes are never used.

The remaining post-fit gap is not attributable to any of these switches. From the traces its main candidates are 5G-LENA's grant overhead and its AMC failure mode, described above.

### Legacy engine (`L2-legacy`)

The legacy slot-level engine replays the same 153 runs through `make_engine("L2-legacy", ..., backend="graph")` on the GPU (bitwise equal to its reference backend, [performance.md](performance.md)). It is frozen, so its closest configuration is its only one: 1 HARQ process with head-of-line blocking, OLLA, power split with PHR cap, logistic BLER, AR(1) Rayleigh fading at 0.93 per UL slot that cannot be switched off, −90 dBm noise-plus-interference (irrelevant here, as the SNR is fed directly), a 16-frame FIFO and a 2 s purge. The only adaptation is the frame size on the air (4150 / 31100 bytes, 5G-LENA's 50 bytes per packet), and it has no HARQ or PRB counters.

| Subset | Runs | KS (floor) | W1 ms | p50 err | p95 err | p99 err | Drop Δ | Goodput \|err\| |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| all runs | 153 | 0.498 (0.027) | 32.3 | -31.7% | -19.1% | -8.2% | -0.36 pp | 1.0% |
| light | 79 | 0.438 (0.033) | 17.9 | -15.1% | -5.7% | +6.5% | +0.00 pp | 0.0% |
| moderate | 36 | 0.464 (0.028) | 95.6 | -33.6% | -43.3% | -44.6% | -4.82 pp | 5.3% |
| saturated | 38 | 0.627 (0.021) | 399.5 | -67.0% | -33.1% | -8.6% | -32.23 pp | 53.0% |
| N = 1 | 9 | 0.875 (0.060) | 14.5 | +10.4% | +17.9% | +36.0% | +0.00 pp | 0.0% |
| N = 8 | 24 | 0.418 (0.021) | 18.5 | -38.6% | -12.4% | +0.3% | +0.00 pp | 0.0% |
| N = 64 | 30 | 0.631 (0.023) | 267.6 | -67.0% | -53.9% | -50.4% | -12.53 pp | 14.3% |

The absolute p95 error is 36% (median over runs) for the legacy engine against 9.2% for the NR engine. The legacy engine drops 32 pp fewer frames than 5G-LENA once the cell saturates, likely because its logistic PHY is optimistic at high spectral efficiency ([validation-5g-lena.md](validation-5g-lena.md)), which gives it more capacity than 5G-LENA once the cell is full. Any result that relies on the legacy engine's delays or losses near saturation should be read with this gap in mind. The full tables are `summary_legacy_graph.csv` and `per_run_legacy_graph.csv`.

### Fading-on arm

The 33 fading runs of the sweep (N ∈ {4, 16, 64}, f ∈ {0.3, 0.9}, both frame sizes, 3 seeds) use stock 3GPP TR 38.901 UMi NLOS fading at 3 m/s. **5G-LENA's own uplink link adaptation does not work in this arm**: it drops a median 70% of frames (16–99%) with a first-transmission BLER of 30%, against 0.6% and 0.3% in the primary arm, because its AMC picks the MCS from the SINR of the RBGs of the last PUSCH, then grants other RBGs, and has no OLLA to absorb the error. This arm is therefore a record of stock 5G-LENA's behavior, not a reference for fading, and it is why the primary arm is frequency-flat. The engine has no 38.901 channel, so it is replayed with its own per-subband AR(1) Rayleigh fading at 3 m/s:

| Engine arm | KS | W1 ms | p50 err | p95 err | Drop Δ | Goodput \|err\| | PRB err | NR drop / BLER (median) |
|:---|---:|---:|---:|---:|---:|---:|---:|:---|
| NR, LENA-matched (OLLA off), `fading=True, ue_speed_mps=3` | 0.244 | 105.2 | -18.8% | +0.9% | -17.5 pp | 82.3% | +11.2% | 0.56 / 0.40 |
| NR, OLLA on | 0.196 | 105.1 | -2.1% | +10.0% | -35.8 pp | 158.3% | +12.7% | 0.32 / 0.11 |
| legacy (`L2-legacy`, OLLA, fading always on) | 0.597 | 270.1 | -67.8% | -72.3% | -68.1 pp | 228.8% | – | 0.00 / – |

5G-LENA's median drop is 0.70. With the same no-OLLA AMC, the NR engine reproduces the failure qualitatively (median drop 56%, BLER 40%), but not quantitatively, since the channel models differ. With OLLA the engine recovers most frames, and the legacy engine delivers all of them. The per-N and per-load tables are in `fade_summary_*.csv`. None of these numbers validates fading. That still needs a link-level check against 5G-LENA's EESM curves or a patched 5G-LENA with OLLA, as [validation-5g-lena.md](validation-5g-lena.md) notes.

### Speed

Wall time per simulated second at the 5G-LENA timing points (60% nominal load, 10 s of traffic, 12.7 s simulated, same p per (N, S) as the 5G-LENA timing runs). 5G-LENA: one process, MAC traces off, measured at a load average of about 21 on the 32-core box (`timing_lena.csv`, from the sweep). NR engine: **`reference` backend, the readable eager engine, not a fast backend**, on the same box with one CPU thread, measured at load averages of 6–13 while other agents' jobs ran (`timing_nr.csv`). E is the number of envs stepped together. The GPU was also tried, but with other jobs holding it at 94% utilization one env took 28 s per simulated second, so that measurement was stopped and is not reported.

| N | S (B) | 5G-LENA s/s | NR ref. s/s (E = 1) | NR / LENA | NR ref. s/s per env (E = 16) |
|:---|---:|---:|---:|---:|---:|
| 1 | 4000 | 0.015 | 0.511 | 34.3× | 0.040 |
| 1 | 30000 | 0.035 | 0.562 | 16.0× | 0.036 |
| 2 | 4000 | 0.025 | 0.521 | 20.9× | 0.036 |
| 2 | 30000 | 0.085 | 0.537 | 6.3× | 0.033 |
| 4 | 4000 | 0.044 | 0.475 | 10.9× | 0.038 |
| 4 | 30000 | 0.072 | 0.601 | 8.3× | 0.039 |
| 8 | 4000 | 0.092 | 0.535 | 5.8× | 0.049 |
| 8 | 30000 | 0.105 | 0.543 | 5.2× | 0.048 |
| 16 | 4000 | 0.142 | 0.592 | 4.2× | 0.060 |
| 16 | 30000 | 0.170 | 0.656 | 3.9× | 0.065 |
| 32 | 4000 | 0.418 | 0.735 | 1.8× | 0.098 |
| 32 | 30000 | 0.256 | 0.676 | 2.6× | 0.097 |
| 64 | 4000 | 0.928 | 0.807 | 0.9× | 0.168 |
| 64 | 30000 | 0.692 | 0.713 | 1.0× | 0.153 |

The reference NR engine costs about 0.5 s per simulated second whatever N is, because each 100 ms step runs 200 slots of small tensor operations whose overhead does not depend on the batch size. 5G-LENA's cost grows about linearly with N. For one cell the reference engine is therefore slower than 5G-LENA below N = 64. Its advantage is batching: 16 envs of 64 UEs cost 0.15–0.17 s per env-second on one thread, about 5× less than one 5G-LENA process, and a GPU or a fused backend would batch far more envs (the fast backends of the legacy engine are in [performance.md](performance.md), and a graph backend for the NR engine is open). In the replay campaign itself, the 30 N = 64 runs × 4 replicas (120 envs) took 22 min of wall time per arm on 2 threads with 8 jobs in parallel, about 0.34 s per env-second. The 5G-LENA sweep runs took a median of 0.21 s per simulated second (8 in parallel). Both machines were shared, so only the same-run ratios above are meaningful.

## Caveats

- **The SR-to-grant value is a fitted proxy.** 40 slots is chosen on seeds 1 and 3 and holds on seed 2, but the measured SR-to-DCI delay is 4.5 ms (about 7 ms to PUSCH). The fit absorbs 5G-LENA grant-pipeline latency that the engine does not model. It is a calibration, not a model of the SR procedure, and it will not transfer to another scheduler or TDD pattern without refitting.
- **Per-frame agreement is only statistical.** 5G-LENA's sample path depends on the process heap layout ([bridges.md](bridges.md), "5G-LENA is sensitive to heap layout"): a different argv length changes the run from the first few steps on. The engine cannot reproduce 5G-LENA frame by frame, and this page compares distributions and quantiles, never individual frames. It also means one 5G-LENA run per seed carries sampling noise of its own that this study cannot measure, and the replica floor covers only the engine's noise.
- **Fading is off in the reference.** The primary arm is a frequency-flat channel with whole-band UE power, because stock 5G-LENA's uplink AMC fails with frequency selectivity (70% median drop in the fading arm). The comparison validates the MAC, HARQ, RLC and PHY abstraction on a flat channel. It says nothing about the engine's fading model.
- **Coverage.** The validation drop redraws any UE below 7 dB `snr1_db` (about 5% at N = 64), so every validated UE is above 5G-LENA's MCS-0 point. The example fleet task uses a 150 m arena with the gNB at the corner, and by the Monte Carlo in `coverage_check.py`, 24% of positions there are below 7 dB with thermal noise and 73% with the legacy −90 dBm noise-plus-interference floor (6% in the 100 m validation arena). In that arena, 5G-LENA delivers only 0–4% of what the slot-level engine delivers in closed loop, with a mean UL SINR of about −19 dB ([bridges.md](bridges.md)), and 0 of 820 frames for a random 8-UE drop in the kill-test link budget. The engines are optimistic below the MCS-0 point, since they keep scheduling with OLLA and a power-headroom rule where stock 5G-LENA sends MCS-0 TBs that fail. This study does not validate that region. A task run in the 150 m arena operates mostly outside the validated range, and its results should not be presented as 5G-LENA-equivalent.
- **The KS distance saturates on narrow distributions.** At N = 1 both sides deliver every frame at a near-constant delay (for example 52.57 ms in 5G-LENA and 52.50 ms in the engine), so KS is 1.0 while W1 is 0.07 ms. KS is reported for completeness, and the quantile errors and W1 are the meaningful measures at light load.
- **TB counts are structurally different.** 5G-LENA sends about twice as many TBs, many of them small padded grants, so retransmission counts differ by 35% (median) while retransmission fractions agree. The engine's retransmission count per UE is failures minus HARQ exhaustions, which ignores TBs still in flight at the end of the run.
- **The 5G-LENA tables are local.** `bler_source="lena"` needs the locally generated 5G-LENA EESM tables (GPL-2.0 data, never shipped). They were available on the lab box and every NR arm except `bler_sionna` uses them.
- **Shared machine.** All runs shared the lab box with other agents' jobs (load average 6–51 during the campaign). This affects only the timing numbers.

## Reproducing

On the lab box, from a working directory of your choice (each arm takes about 40 minutes of wall time on 2 threads per job with 8 jobs in parallel, dominated by N = 64):

```bash
export ISAACLAB_NET_LENA_TABLES=<path to lena_eesm_tables.npz> REPO=<isaaclab-net checkout> PYTHONPATH=$REPO
python $REPO/benchmarks/fidelity/lena_extract.py <ns3ref>/sweep/nofade data        # 5G-LENA per-frame / per-UE / SR data
python $REPO/benchmarks/fidelity/lena_extract.py <ns3ref>/sweep/fade data_fade
bash $REPO/benchmarks/fidelity/run_all.sh        # primary + 8 ablation arms, 4 replicas, CPU, reference backend
bash $REPO/benchmarks/fidelity/run_extra.sh      # SR grid for the fit / hold-out split, fading arm
python $REPO/benchmarks/fidelity/legacy_replay.py data replay legacy_graph 4 cuda graph
python $REPO/benchmarks/fidelity/legacy_replay.py data_fade replay_fade legacy_graph 4 cuda graph
python $REPO/benchmarks/fidelity/compare.py data replay results
python $REPO/benchmarks/fidelity/compare.py data_fade replay_fade results fade_
python $REPO/benchmarks/fidelity/timing.py <ns3ref>/timing.csv results/timing_nr.csv cpu 1 1
python $REPO/benchmarks/fidelity/timing.py <ns3ref>/timing.csv results/timing_nr.csv cpu 16 1
python $REPO/benchmarks/fidelity/report.py results                               # the tables on this page
```

`compare.py` copies nothing, so `lena_runs.csv`, `lena_per_ue.csv`, `lena_sr_dci.csv` (and their `_fade` versions) and the 5G-LENA `timing.csv` (as `timing_lena.csv`) are copied into `results/` by hand. The per-frame `.npz` files (about 38 MB for the 5G-LENA side and all replays) stay on the lab box under `/home/zzhang66/experiments/lenafid/`.

### Result files

| File | Content |
|:---|:---|
| `per_run_<arm>.csv`, `fade_per_run_<arm>.csv` | one row per run: every metric above on both sides, the replica floor, and per-class (edge / mid / centre) KS and quantile errors |
| `summary_<arm>.csv`, `fade_summary_<arm>.csv` | medians and 90th percentiles by all / regime / N / load / S / N × regime |
| `ablation.csv`, `fade_ablation.csv` | one sweep-level row per arm |
| `sr_fit_holdout.csv` | the SR grid on the fit seeds (1, 3) and the hold-out seed (2), with the objective and the chosen value |
| `per_ue_primary.csv` | one row per UE and run: SNR, class, goodput, p50 / p95, TB and retransmission counts on both sides |
| `edge_centre_primary.csv`, `ue_classes.csv` | the class table above and the SNR tercile boundaries |
| `delay_quantiles_primary.csv` | 101 delay quantiles per run for both sides, for CDF figures |
| `lena_runs.csv`, `lena_per_ue.csv`, `lena_sr_dci.csv`, `*_fade.csv` | the 5G-LENA side as extracted (per-UE HARQ counts with the corrected RNTI map, SR-to-DCI delays) |
| `timing_lena.csv`, `timing_nr.csv` | speed measurements |
| `replay_groups.txt` | wall time of every replay job, with its overrides |
