# Why the NR engine is optimistic under load, and a fix

The formal comparison in [fidelity-vs-lena.md](fidelity-vs-lena.md) found that the NR engine (level `L2`, `lena_validation()`) matches 5G-LENA at light load but is optimistic once the cell is loaded: in runs where 5G-LENA drops 1–20% of frames (moderate) the engine's median delay is 27% low and it drops 4.1 percentage points fewer frames, and in saturated runs (5G-LENA drop ≥ 20%) the median delay is 35% low and the drop rate 6.9 pp low. This page attributes that gap mechanism by mechanism, from 5G-LENA's own MAC, RLC and PHY traces and from engine replays of the same 153 runs, and measures a prototype switch for each missing mechanism. All of them are now `NRConfig` switches of the engine itself, turned on together by the presets `lena_match_v2()` and `lena_validation_v2()`, with defaults that leave the engine bitwise unchanged ([Status](#status-in-the-engine-v2)). The tables below keep the prototype's switch names; the Status section maps them to the fields. Every number below is in a CSV under `benchmarks/fidelity/results/loadfix/` (the study) and `benchmarks/fidelity/results/loadfix_v2/` (the engine-integrated rerun).

## Summary

- **The main cause is the scheduler's sharing discipline, not grant overhead.** 5G-LENA's OFDMA PF scheduler updates a UE's average throughput after every RBG it assigns and freezes the average of idle UEs. Under load it therefore gives each backlogged UE about one RBG per slot, so frames of different UEs are served side by side (processor sharing). The engine's PF metric is fixed within a slot and every UE's average decays while it is idle, so one UE takes every RBG it needs and frames are served nearly one at a time. At the same capacity, side-by-side service raises the median delay and makes more frames miss the 2 s deadline. The two PF changes alone bring the moderate-load median-delay error from −27% to −11% and the saturated one from −35% to −7%.
- **Grant padding is real but small under load.** 5G-LENA's quantized buffer status reports, its two-UL-slot report delay and the SR bootstrap grant pad 9% of its granted RBGs at light load, which is the engine's −13% PRB error there. In moderate and saturated cells the padding is 1.2% and 0.02% of the carrier, because backlogged UEs use their grants. Modeling the grant pipeline removes most of the light-load PRB error (−13% → −5%) and replaces the fitted 20 ms SR-to-grant delay with measured timing, but alone it moves the loaded-cell medians little.
- **An RLC tail stall is where the fitted 20 ms went.** In 5G-LENA, the last few bytes of nearly every frame whose UE drains its buffer (5–17 bytes of RLC header that the UE's MAC does not account for) wait for the 10 ms RLC buffer-status timer and a new SR, which adds about 15 ms to the frame. 100% of the frames of light runs complete this way. The engine's fitted 40-slot SR delay was a stand-in for this stall plus the SR/BSR round trips. The prototype models the stall and the round trips directly and matches the light-load delays with no fitted parameter (N = 1: 52.5 ms against 5G-LENA's 52.57 ms).
- **All mechanisms together meet the target.** With every switch on, the median p50 error is −1.2% at moderate load (absolute 4.6%) and −0.6% at saturation (absolute 4.5%), the light-load match is kept and tightened (p50 −0.2%, absolute 1.3%; p95 −9.2% → −2.5%), the drop gap shrinks to −1.4 pp and −1.7 pp, the PRB error to −3%, the KS distance in loaded cells from 0.32–0.38 to 0.09–0.13, and the engine sends as many TBs as 5G-LENA (TB-count ratio 0.23–0.71 → 0.97–1.02). The combination has no fitted parameter, so the seed-2 hold-out is not a separate check here; it is shown anyway.
- **Terms that only matter together.** TDMA retransmissions (5G-LENA gives every UL retransmission the whole slot) and 5G-LENA's link adaptation for the PRB count of the previous PUSCH change almost nothing on their own, because the engine's TBs usually span the whole carrier. Once RBGs are spread over UEs, a retransmitted one-RBG TB blocks the other four, and leaving either switch out of the combination costs 3–5 points of moderate-load p50 and up to 2.2 pp of drop. The MAC overhead per TB (8 instead of 6 bytes) and the fluid-versus-per-TB accounting of partially served frames are negligible.

## Method

Both sides are the ones of [fidelity-vs-lena.md](fidelity-vs-lena.md): the 153 primary-arm runs of the ns3ref 5G-LENA sweep (fading off, whole-band UE power, per-UE `snr1_db`, the exact frame schedule), replayed in the NR engine with 4 replicas per run on the reference backend (CPU), common random numbers across arms, and compared by `benchmarks/fidelity/compare.py` with the same regimes (light: 5G-LENA drop < 1%, 79 runs; moderate: 1–20%, 36; saturated: ≥ 20%, 38). The engine side of every arm is `benchmarks/fidelity/loadfix/loadfix_replay.py`, which is `nr_replay.py` with `LoadFixNet`; its `base` arm is bitwise the study's `primary` arm (checked on every N = 2 run).

The 5G-LENA side of the diagnosis is `benchmarks/fidelity/loadfix/lena_pipeline.py`, which reads each run's raw traces read-only and joins every UL grant (`NrUlMacStats.txt`), every decoded TB (`RxPacketTrace.txt`), every RLC PDU the UE built into a grant (`NrUlRlcTxStats.txt`), every SR (`TxedUeMacCtrlMsgsTrace.txt`) and the RBG use of every UL slot (`gnb_slots.csv`). The RBG count of a grant is recovered by inverting 5G-LENA's TB-size formula at 13 symbols. A grant's data RBGs are the fewest RBGs whose TB fits the RLC bytes it carried plus the 8 bytes of MAC subheader and short BSR, and the rest of its RBGs are padding. The mechanisms were also read in the 5G-LENA v5.1 source used by the sweep (`contrib/nr/model`, with the ns3ref patches): `NrMacSchedulerOfdma::AssignULRBG`, `NrMacSchedulerUeInfoPF`, `NrMacSchedulerHarqRr::ScheduleUlHarq`, `NrMacSchedulerNs3::DoScheduleUl` and `DoScheduleUlSr`, `NrUeMac::SendNewData` and `SendBufferStatusReport`, `NrRlcUm::DoTransmitBufferStatusReport`, `NrAmc::CreateCqiFeedbackSiso` and the BSR level table in `nr-common.cc`.

## 5G-LENA's grant pipeline, from its traces

Medians over runs (`lena_pipeline_by_regime.csv`). Utilizations are fractions of all RBG-slots of the 30 s traffic window.

| Regime | Runs | Padded share of granted RBGs | Empty grants (share of TBs) | Data | Padding | Retx | Blocked by TDMA retx | TBs per frame | Tail stalls per frame | Stall gap ms |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| light | 79 | 8.9% | 7.3% | 31.2% | 3.1% | 0.27% | 0.05% | 12.7 | 1.00 | 15.0 |
| moderate | 36 | 1.6% | 1.3% | 92.6% | 1.2% | 0.65% | 0.15% | 21.8 | 1.00 | 15.0 |
| saturated | 38 | 0.02% | 0.01% | 99.6% | 0.02% | 0.06% | 0.11% | 49.4 | 0.59 | 15.0 |

By frame size, the light-load padding is 17% of granted RBGs for 4 KB frames and 5.6% for 30 KB frames (a fixed cost per burst weighs more on a small frame). The engine's PRB errors follow the same pattern (−24% for 4 KB, −8% for 30 KB).

A single 30 KB frame of the N = 1 run `n1_s30000_f0.3_r1` shows the whole pipeline (times from the frame's generation at 0.500 s):

| ms | Event |
|---:|:---|
| 0.0 | 31,100 B enter the RLC buffer, the UE sends an SR |
| 5.0 | bootstrap PUSCH: one RBG, 38 B TB at MCS 0 carrying 30 B and a short BSR (the scheduler owes 17 B after an SR) |
| 10.0 | first data PUSCH, 2698 B (MCS 19, 5 RBGs): the BSR sent at 5.0 ms is used two UL slots later |
| 10.0–37.5 | 11 full TBs, then 1480 B in a full 2698 B TB |
| 40.0 | a 2156 B grant with nothing to send: it was sized from the BSR of 35.0 ms (1480 B, quantized up to 1552 B), which reached the scheduler after the 37.5 ms grant had been decided, so the 1480 B were granted twice |
| 46.0 | the RLC buffer-status timer (10 ms after the last transmission opportunity) reports 17 B that the UE MAC had not counted (RLC headers of the concatenated PDUs), and the UE sends a new SR |
| 52.5 | a 538 B TB (one RBG) carries the 17 B, the frame completes (52.57 ms) |

`tail_check.py` over all runs: 99–100% of the frames of light runs, 50–100% of moderate runs and 1–26% of saturated runs complete with such a tail PDU sent at least 5 ms after the UE's previous PDU. In a saturated cell a UE rarely drains its buffer, so the stall happens less often.

## Mechanism by mechanism

Each row is one prototype switch (Status maps it to its `NRConfig` field), replayed alone on all 153 runs (`arms_by_regime.csv`). Signed medians over runs; the |p50| column is the median absolute error.

| Arm | Switch | Regime | p50 err | \|p50\| | p95 err | Drop Δ pp | PRB err | TB ratio | KS |
|:---|:---|:---|---:|---:|---:|---:|---:|---:|---:|
| primary | none (the engine) | light | -0.2% | 4.5% | -9.2% | +0.00 | -13.3% | 0.71 | 0.257 |
| primary | none (the engine) | moderate | -26.9% | 26.9% | -15.2% | -4.09 | -5.4% | 0.38 | 0.316 |
| primary | none (the engine) | saturated | -35.2% | 35.7% | -2.6% | -6.89 | +0.0% | 0.23 | 0.379 |
| pf_intra | `pf_intra_slot` | light | -0.2% | 4.2% | -9.2% | +0.00 | -13.3% | 0.92 | 0.257 |
| pf_intra | `pf_intra_slot` | moderate | -28.0% | 28.0% | -14.0% | -3.66 | -3.9% | 0.85 | 0.308 |
| pf_intra | `pf_intra_slot` | saturated | -31.0% | 32.6% | -2.2% | -5.52 | +0.0% | 0.95 | 0.357 |
| pf_active | `pf_active_only` | light | -3.9% | 4.9% | -11.7% | +0.00 | -13.3% | 0.71 | 0.246 |
| pf_active | `pf_active_only` | moderate | -14.6% | 14.6% | -17.5% | -4.41 | -5.5% | 0.38 | 0.186 |
| pf_active | `pf_active_only` | saturated | -8.8% | 9.8% | -2.1% | -6.90 | +0.0% | 0.22 | 0.101 |
| pf | both PF switches | light | -4.0% | 5.1% | -11.6% | +0.00 | -13.3% | 0.87 | 0.246 |
| pf | both PF switches | moderate | -11.3% | 11.3% | -17.9% | -3.66 | -3.6% | 0.90 | 0.168 |
| pf | both PF switches | saturated | -6.7% | 7.7% | -1.8% | -5.25 | +0.0% | 1.00 | 0.110 |
| retx | `retx_tdma` | light | -0.2% | 4.9% | -9.2% | +0.00 | -13.3% | 0.71 | 0.257 |
| retx | `retx_tdma` | moderate | -26.9% | 26.9% | -14.6% | -3.88 | -5.4% | 0.38 | 0.315 |
| retx | `retx_tdma` | saturated | -34.6% | 35.4% | -3.3% | -6.86 | -0.0% | 0.23 | 0.378 |
| amc | `amc_prev_alloc` | light | -0.2% | 4.9% | -8.5% | +0.00 | -13.3% | 0.71 | 0.252 |
| amc | `amc_prev_alloc` | moderate | -26.9% | 26.9% | -13.7% | -3.86 | -5.2% | 0.38 | 0.315 |
| amc | `amc_prev_alloc` | saturated | -34.1% | 34.4% | -3.0% | -6.76 | +0.0% | 0.23 | 0.378 |
| oh8 | `tb_overhead_bytes=8` | light | -0.2% | 4.7% | -9.2% | +0.00 | -13.3% | 0.71 | 0.257 |
| oh8 | `tb_overhead_bytes=8` | moderate | -26.9% | 26.9% | -15.0% | -3.95 | -5.2% | 0.38 | 0.314 |
| oh8 | `tb_overhead_bytes=8` | saturated | -34.1% | 34.8% | -2.8% | -6.88 | +0.1% | 0.23 | 0.380 |
| pipe | `grant_pipeline` + 8 B | light | -0.2% | 3.3% | -6.4% | +0.00 | -5.0% | 0.85 | 0.250 |
| pipe | `grant_pipeline` + 8 B | moderate | -26.9% | 26.9% | -13.6% | -3.32 | -3.0% | 0.41 | 0.302 |
| pipe | `grant_pipeline` + 8 B | saturated | -26.7% | 28.6% | -2.6% | -5.96 | +0.1% | 0.23 | 0.329 |

### (a) BSR granularity and grant padding

**Evidence.** 5G-LENA pads 8.9% of its granted RBGs at light load, 7.3% of its new-data grants carry no data at all, and each frame costs one bootstrap grant (1.0 per frame at light load). Three rules produce this: the UE reports its buffer with the 38.321 short-BSR levels rounded up (up to 17% above the true value, `BufferSizeLevelBsr`), the report reaches the scheduler two UL slots after the PUSCH that carries it and then overwrites the scheduler's count, so the grant already issued for the next slot is counted again, and an SR buys a 17 B bootstrap grant of one RBG. In loaded cells padding falls to 1.2% (moderate) and 0.02% (saturated) of the carrier.

**Predicted effect.** Most of the engine's light-load PRB error (−13%) and TB-count ratio (0.71); a capacity loss of at most 1.2% in moderate cells, hence little change of the loaded-cell delays.

**Model change.** `grant_pipeline=True`: SR at the next SR opportunity, a bootstrap grant `sr_boot_slots` (6) later that the PF scheduler serves like any other need of 17 B, a short BSR on every PUSCH with the buffer left after that TB plus 8 B, quantized up to the level table and applied `bsr_delay_slots` (10, two UL slots of DDDSU) after its PUSCH as the new estimate (plus 5 B of RLC and MAC header), which grants reduce as they are issued. RBGs are allocated until the TB covers the estimate, the UE fills them with what it has, and the rest is padding. It replaces the lumped SR-to-grant delay, whose fitted value (40 slots) is no longer read.

**Measured.** The `pipe` arm: light-load PRB error −13.3% → −5.0%, p95 −9.2% → −6.4%, light p50 unchanged at −0.2% with no fitted parameter, TB ratio 0.71 → 0.85; the saturated p50 improves from −35% to −27% and the moderate p50 not at all.

### (b) TBS/MCS selection and BLER tables (Sionna vs 5G-LENA, EESM beta)

**Evidence.** The validation arm already uses 5G-LENA's own EESM tables, TB-size rule and IR combining, and the channel is flat with whole-band power, so the effective SINR equals the per-RB SINR and the EESM beta has no effect on any run of this sweep (it weighs subbands of different SINR, and here all are equal). The Sionna-table ablation of the fidelity study (−33% PRB, −32% p95) is therefore a table effect that the validation configuration has already removed. What remains is the MCS rule: 5G-LENA's UL AMC (`NrAmc::CreateCqiFeedbackSiso`) picks the highest MCS whose TB error rate is at most 10% for the SINR and the PRB count of the UE's last PUSCH, then sizes the TB for the new allocation. For 49% of the 3,583 UEs with a first transmission the engine's MCS (chosen for the current allocation) is one step above the median first-transmission MCS 5G-LENA used, for 3% one below, for 48% equal (`mcs_check.py`; the engine rule at 1, 2 and 5 RBGs gives the same split, and the difference sits in the 3–15 dB SNR bands). Neither a constant SINR offset nor the "first failing MCS minus one" search order explains it.

**Predicted effect.** About half an MCS step on average, a few percent of spectral efficiency and hence of capacity in saturated cells.

**Model change.** `amc_prev_alloc=True`: MCS for the PRB count of the previous PUSCH, TB for the current allocation.

**Measured.** Alone, the `amc` arm moves the saturated p50 by one point (−35.2% → −34.1%) and the drop gap by 0.1 pp. With the other switches on it matters more: leaving it out of `all` moves the moderate p50 from −1.2% to −4.3% and the drop gaps by 0.4–0.8 pp. The half-step disagreement is not fully explained by this rule either and is an open item.

### (c) HARQ round trip and retransmission scheduling

**Evidence.** The UL HARQ round trip agrees: 5G-LENA retransmits a NACKed TB 2.5 ms (the next UL slot) after its first transmission in every run (`retx_gap_ms`), and the engine's `ul_rtt` (3 slots) also lands on the next UL slot. The scheduling differs: 5G-LENA schedules UL retransmissions TDMA (`NrMacSchedulerHarqRr::ScheduleUlHarq`), so a retransmission takes every data symbol of its slot whatever its RBG count, at most one is sent per slot, and no new data shares that slot. The traces show 0.65% of the carrier spent on retransmissions in moderate cells and a further 0.15% blocked by them.

**Predicted effect.** Below 1% of capacity with the engine's scheduler, where a UE's TB usually has all five RBGs anyway. Larger once RBGs are spread over UEs, because a retransmitted one-RBG TB then blocks four RBGs.

**Model change.** `retx_tdma=True`: one retransmission per slot, the oldest NACK first, and nothing else in its slot.

**Measured.** Alone (`retx`): within 0.6 points of the base arm on every regime median. With the PF changes on it is the second-largest term for the drop rate: leaving it out of `all` moves the moderate p50 from −1.2% to −6.3%, the saturated p50 from −0.6% to −4.1% and the saturated drop gap from −1.7 to −3.9 pp.

### (d) gNB processing and pipeline delay, and queue build-up

**Evidence.** Two parts. First, the access and completion pipeline of a frame (SR → bootstrap PUSCH 5 ms, BSR round trip 5 ms, RLC tail stall about 15 ms, see the trace above) is what the fitted 20 ms SR delay stood in for; at light load it is sequential and a lumped delay works. Second, and larger under load, the scheduler: in 5G-LENA's OFDMA PF (`AssignULRBG` with `NrMacSchedulerUeInfoPF`) the RBG winner's average throughput is updated after every RBG, `(1 − 1/99)·last + (1/99)·TB(k)/symbols`, before the next RBG is ranked, and only UEs with data update their average. Under load this spreads the RBGs of a slot over up to five UEs. The trace evidence is the TB count: 5G-LENA sends a median 49 TBs per frame in saturated cells, about 4.4 times as many as the engine (TB-count ratio 0.23), and in the saturated N = 64, 4 KB run 12,351 of the 12,505 scheduled UL slots carry exactly five UEs. The engine's PF metric is fixed within a slot, so the top UE takes every RBG until its need is covered, and every UE's average decays every UL slot, so a UE that returns from idle outranks the backlogged ones. The result is FIFO-like service in the engine against processor sharing in 5G-LENA: at the same carried load, processor sharing gives a higher median delay and more frames past the deadline.

**Predicted effect.** The bulk of the loaded-cell p50 gap and part of the drop gap; no effect at light load, where frames rarely overlap.

**Model change.** `pf_intra_slot=True` (per-RBG average update with the granted TB bytes) and `pf_active_only=True` (the average moves only in slots where the UE has data).

**Measured.** Together (`pf`): moderate p50 −26.9% → −11.3%, saturated −35.2% → −6.7%, KS 0.32 → 0.17 and 0.38 → 0.11, TB ratio → 0.90 and 1.00, drop gap −4.1 → −3.7 pp and −6.9 → −5.3 pp. Most of it comes from the frozen average (`pf_active` alone: −14.6% and −8.8%), which stops returning UEs from grabbing the carrier; the per-RBG update alone (`pf_intra`) spreads RBGs (TB ratio 0.95) but moves the medians little, and the two together give the processor-sharing service that matches 5G-LENA. With `pf` the light-load p50 drops to −4.0%, which the grant pipeline brings back (next section).

### (e) RLC segmentation, padding and header overhead

**Evidence.** Per TB, 5G-LENA spends 3 B of MAC subheader and 5 B of short BSR (`NrUeMac::SendNewData`), against the engine's `tb_overhead_bytes = 6`. RLC header bytes are already in the engine's 50 B per 1400 B packet, which was read from 5G-LENA's buffer traces. The RLC effect that matters is not the byte count but the tail stall of (d): the RLC reports its buffer as bytes plus 2 B per SDU, the concatenated PDUs spend a different number of header bytes, and the difference (5–17 B per frame in the traces) is only reported by the 10 ms RLC buffer-status timer.

**Predicted effect.** Header bytes: 2 B per TB, below 0.4% of a one-RBG TB. Tail stall: about 15 ms on every frame whose UE drains its buffer.

**Model change.** `tb_overhead_bytes=8`; the tail stall is part of `grant_pipeline` (`rlc_tail_bytes = 16` held back by the TB that drains new data, reported `rlc_tail_timer_slots = 20` later or at the next arrival, then an SR).

**Measured.** `oh8` alone: within 1.1 points of the base arm everywhere. The stall is measured with the grant pipeline.

### (f) Fluid versus per-TB accounting of partially served frames

**Evidence.** The engine does not use a fluid model: it hands out byte ranges of a per-UE stream to TBs, a TB may carry the end of one frame and the start of the next, and a frame completes at the end of the slot in which its last byte is decoded, in order (`mac.py`). 5G-LENA concatenates RLC SDUs into one PDU per grant and delivers a frame when its last packet is reassembled. In the N = 1 runs, where queueing is absent, the engine and 5G-LENA complete frames 0.07 ms apart (52.50 against 52.57 ms, the 0.1 ms from PUSCH end to the application), and the per-TB byte counts of the trace above are reproduced exactly by the grant-pipeline arm.

**Predicted effect and change.** None; no switch.

## Everything together, and leave-one-out

| Arm | Regime | Runs | p50 err | \|p50\| | p95 err | \|p95\| | Drop Δ pp | Goodput \|err\| | PRB err | TB ratio | KS | W1 ms |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| primary | light | 79 | -0.2% | 4.5% | -9.2% | 9.5% | +0.00 | 0.0% | -13.3% | 0.71 | 0.257 | 3.5 |
| primary | moderate | 36 | -26.9% | 26.9% | -15.2% | 16.8% | -4.09 | 4.5% | -5.4% | 0.38 | 0.316 | 66.0 |
| primary | saturated | 38 | -35.2% | 35.7% | -2.6% | 3.5% | -6.89 | 11.8% | +0.0% | 0.23 | 0.379 | 132.9 |
| primary | light, seed 2 | 34 | -0.1% | 4.2% | -7.6% | 8.5% | +0.00 | 0.0% | -13.9% | 0.72 | 0.266 | 2.6 |
| primary | moderate, seed 2 | 8 | -30.7% | 30.7% | -18.3% | 18.3% | -4.09 | 4.5% | -1.6% | 0.35 | 0.357 | 90.7 |
| primary | saturated, seed 2 | 9 | -37.2% | 37.2% | -3.1% | 3.1% | -6.49 | 11.0% | +0.0% | 0.23 | 0.396 | 175.2 |
| pf | light | 79 | -4.0% | 5.1% | -11.6% | 11.6% | +0.00 | 0.0% | -13.3% | 0.87 | 0.246 | 3.9 |
| pf | moderate | 36 | -11.3% | 11.3% | -17.9% | 22.4% | -3.66 | 4.2% | -3.6% | 0.90 | 0.168 | 38.4 |
| pf | saturated | 38 | -6.7% | 7.7% | -1.8% | 2.9% | -5.25 | 9.0% | +0.0% | 1.00 | 0.110 | 49.6 |
| pf | light, seed 2 | 34 | -0.2% | 4.2% | -7.6% | 8.9% | +0.00 | 0.0% | -13.9% | 0.88 | 0.257 | 2.5 |
| pf | moderate, seed 2 | 8 | -14.0% | 14.0% | -25.5% | 25.5% | -3.66 | 4.2% | -1.1% | 0.96 | 0.175 | 44.0 |
| pf | saturated, seed 2 | 9 | -7.7% | 8.2% | -1.7% | 2.8% | -5.24 | 9.3% | +0.0% | 1.00 | 0.079 | 48.4 |
| pipe | light | 79 | -0.2% | 3.3% | -6.4% | 6.5% | +0.00 | 0.0% | -5.0% | 0.85 | 0.250 | 3.3 |
| pipe | moderate | 36 | -26.9% | 26.9% | -13.6% | 16.7% | -3.32 | 3.5% | -3.0% | 0.41 | 0.302 | 46.6 |
| pipe | saturated | 38 | -26.7% | 28.6% | -2.6% | 3.9% | -5.96 | 9.4% | +0.1% | 0.23 | 0.329 | 123.3 |
| pipe | light, seed 2 | 34 | -0.2% | 3.1% | -5.2% | 5.4% | +0.00 | 0.0% | -4.2% | 0.90 | 0.292 | 2.4 |
| pipe | moderate, seed 2 | 8 | -33.6% | 33.6% | -21.5% | 28.0% | -2.50 | 2.7% | -0.5% | 0.39 | 0.400 | 77.4 |
| pipe | saturated, seed 2 | 9 | -31.6% | 31.6% | -3.3% | 3.7% | -6.03 | 9.3% | +0.1% | 0.23 | 0.367 | 163.6 |
| pf_pipe | light | 79 | -1.3% | 3.3% | -4.7% | 5.4% | +0.00 | 0.0% | -3.8% | 1.01 | 0.250 | 2.2 |
| pf_pipe | moderate | 36 | -5.7% | 7.4% | -9.3% | 15.3% | -3.04 | 3.3% | -1.3% | 0.99 | 0.140 | 26.3 |
| pf_pipe | saturated | 38 | -4.0% | 6.6% | -1.3% | 1.6% | -3.84 | 7.4% | +0.1% | 1.00 | 0.118 | 37.0 |
| pf_pipe | light, seed 2 | 34 | -0.2% | 0.2% | -3.3% | 3.5% | +0.00 | 0.0% | -2.4% | 1.04 | 0.258 | 1.1 |
| pf_pipe | moderate, seed 2 | 8 | -3.8% | 6.0% | -21.6% | 21.6% | -2.65 | 2.8% | -0.3% | 1.03 | 0.139 | 38.6 |
| pf_pipe | saturated, seed 2 | 9 | -6.6% | 6.7% | -1.4% | 1.5% | -4.52 | 7.2% | +0.1% | 1.00 | 0.079 | 39.7 |
| all | light | 79 | -0.2% | 1.3% | -2.5% | 3.3% | +0.00 | 0.0% | -3.2% | 1.02 | 0.248 | 2.0 |
| all | moderate | 36 | -1.2% | 4.6% | -5.2% | 8.6% | -1.44 | 1.8% | -4.7% | 0.99 | 0.130 | 17.6 |
| all | saturated | 38 | -0.6% | 4.5% | +0.4% | 1.5% | -1.73 | 4.0% | -3.1% | 0.97 | 0.088 | 30.6 |
| all | light, seed 2 | 34 | -0.2% | 0.2% | -2.2% | 3.1% | +0.00 | 0.0% | -2.4% | 1.05 | 0.270 | 1.0 |
| all | moderate, seed 2 | 8 | -1.1% | 9.0% | -1.8% | 17.7% | -1.58 | 1.7% | -2.3% | 1.04 | 0.131 | 41.0 |
| all | saturated, seed 2 | 9 | -2.8% | 6.7% | +1.1% | 2.2% | -3.52 | 5.5% | -2.4% | 0.98 | 0.054 | 28.5 |

Leave-one-out from `all` (one switch off at a time), median over runs:

| Arm | Moderate p50 | Moderate \|p50\| | Moderate drop Δ pp | Saturated p50 | Saturated \|p50\| | Saturated drop Δ pp | Light p50 | Light p95 | Light PRB |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| all | -1.2% | 4.6% | -1.44 | -0.6% | 4.5% | -1.73 | -0.2% | -2.5% | -3.2% |
| all_no_intra | -9.5% | 9.5% | -2.40 | -7.4% | 9.5% | -5.81 | -3.5% | -5.7% | -4.7% |
| all_no_active | -19.9% | 20.6% | -0.93 | -15.4% | 18.0% | -1.45 | -0.2% | -0.2% | -2.7% |
| all_no_pipe | -6.2% | 6.5% | -2.49 | -0.5% | 5.1% | -2.62 | -3.3% | -8.7% | -12.9% |
| all_no_retx | -6.3% | 7.5% | -2.23 | -4.1% | 6.6% | -3.90 | -2.7% | -3.5% | -3.0% |
| all_no_amc | -4.3% | 7.0% | -2.23 | -0.8% | 6.8% | -2.12 | -0.2% | -4.5% | -3.9% |
| all_no_oh8 | -1.8% | 5.6% | -1.78 | -0.4% | 6.1% | -1.87 | -0.2% | -3.0% | -3.2% |

By N, the `all` arm's p50 error lies between −5.4% (N = 2) and −0.1% (N = 1) with drop differences of −1.1 pp or less (`summary_all.csv`), against −22.6% and −5.5 pp at N = 64 before. The first-transmission BLER difference to 5G-LENA grows from +0.12 pp to +0.45 pp; the cause was not isolated. On the seed-2 runs alone the saturated drop gap is larger (−3.5 pp over 9 runs) than on the full sweep, while the p50 errors stay within 3 points.

Every switch pays for itself in the leave-one-out table. The frozen PF average is the largest delay term (without it the loaded p50 is 15–20% low again), the per-RBG PF update and the TDMA retransmissions are the largest drop terms at saturation (−5.8 and −3.9 pp without them), and the grant pipeline carries the light-load PRB and p95 match. Removing the frozen average lowers the drop gap slightly (−0.9 and −1.5 pp) at the price of the delay match, so the two PF switches trade a little drop accuracy for a lot of delay accuracy.

**Remaining gap.** The drop rate stays 1.4–1.7 pp low in loaded cells. Candidates, in order of the evidence: 5G-LENA's link-adaptation failure mode for a few edge UEs (about 1 pp of the moderate gap, see [fidelity-vs-lena.md](fidelity-vs-lena.md)), which the engine does not reproduce; the half-MCS disagreement of (b); and the SR re-sends of 5G-LENA while backlogged (2.7–12.9 SRs per frame), each of which re-arms a 17 B bootstrap owed until the next BSR, which the prototype ignores.

## Status: in the engine (v2)

Every mechanism is an `NRConfig` field of the engine (`mac.py`, `mac_ul.py`). The defaults reproduce the engine before these switches bitwise, and the preset `lena_match_v2()` (the `lena_match` scenario) and its validation-geometry form `lena_validation_v2()` turn all of them on. `lena_match`, `lena_like` and `lena_validation` are unchanged, so the published fidelity numbers stay reproducible.

| Prototype switch | `NRConfig` field | Links |
|:---|:---|:---|
| `pf_intra_slot` | `pf_update="rbg"` (default `"slot"`) | UL and DL, PF schedulers |
| `pf_active_only` | `pf_avg_idle="freeze"` (default `"decay"`) | UL and DL |
| `retx_tdma` | `ul_retx_sched="tdma"` (default `"ofdma"`) | UL, one retransmission per slot and cell |
| `amc_prev_alloc` | `ul_amc_alloc="previous"` (default `"current"`) | UL |
| `grant_pipeline` | `ul_grant_model="bsr"` (default `"lumped"`) with `sr_boot_slots` 6, `sr_boot_bytes` 17, `bsr_delay_slots` 10, `bsr_hdr_bytes` 8, `bsr_est_hdr_bytes` 5, `rlc_tail_bytes` 16, `rlc_tail_timer_ms` 10 | UL |
| `tb_overhead_bytes=8` | `tb_overhead_bytes=8` (existing field; the v2 presets set it) | UL and DL |

Under `ul_grant_model="bsr"` the fields `sr_grant_delay_slots` and `proactive_grant` are not read, and `unused_fields("L2")` reports them when they are set; under `"lumped"` it reports the pipeline fields instead. The extra per-robot state (`est`, `boot`, the four reports in flight `rep_v` / `rep_g`, `hid`, `hid_until`, `enq_seen`, `armed`, and `last_nprb` for the previous-PUSCH AMC) is part of `UlMac.STATE` only when its switch is on, so it has the fixed `[E, R, ...]` shape and the partial `reset(env_ids)` of every other MAC state. The switches work with several cells: TDMA admits one retransmission per cell and slot, and a handover hands the target the robot's quantized buffer while reports in flight are lost. With any switch on, the MAC counts granted TB bytes, empty grants and slots blocked by a TDMA retransmission (`counters()`: `tb_bytes`, `tb_empty`, `retx_block`). `core/nr_loadfix.py` is now a compatibility shim that maps the prototype's `LoadFixConfig` onto these fields for the study's scripts.

**Backends.** The graph backend captures the switches like any other MAC code and stays bitwise equal to the reference with them on (`tests/test_nr_fast.py` G7: `lena_match_v2`, per-RBG PF with the BSR pipeline on UL and DL, and every switch at three cells). The fused Triton kernel implements `pf_update`, `pf_avg_idle`, `ul_retx_sched` and `ul_amc_alloc` (teacher-forced test G2) but not yet the BSR grant pipeline: `make_engine(..., backend="triton")` refuses `ul_grant_model="bsr"`, and with it the v2 presets, with a message that points to `graph`. Porting the pipeline means carrying the nine extra per-robot arrays (four of them 4 deep) through the kernel, whose registers already spill at R = 64–100; that is a follow-up.

**Checks.** `tests/test_nr_loadfix.py` L0 runs every switch set against a frozen copy of the prototype (`tests/nr_frozen/loadfix_proto.py`) and finds every output and state tensor bitwise equal, through a partial reset, with the global and with the engine RNG. On the full replay, `lena_validation_v2(rng="global")` reproduces the prototype's `all` arm bitwise in all 153 runs (every per-frame delay, TB and PRB count), so the numbers above carry over to the engine unchanged.

**Replay with the engine RNG** (`lena_validation_v2()` as it runs by default; `primary_eng` is `lena_validation()` on the same RNG, the before; medians over runs):

| Arm | Regime | Runs | p50 err | \|p50\| | p95 err | \|p95\| | Drop Δ pp | Goodput \|err\| | PRB err | TB ratio | KS | W1 ms |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| primary_eng | light | 79 | -0.2% | 4.2% | -9.2% | 9.5% | +0.00 | 0.0% | -13.4% | 0.71 | 0.256 | 3.5 |
| primary_eng | moderate | 36 | -26.9% | 26.9% | -15.4% | 16.9% | -4.05 | 4.4% | -5.4% | 0.38 | 0.316 | 66.8 |
| primary_eng | saturated | 38 | -33.7% | 34.8% | -3.3% | 4.2% | -6.84 | 11.8% | +0.0% | 0.23 | 0.380 | 135.2 |
| v2 | light | 79 | -0.2% | 2.5% | -2.5% | 3.5% | +0.00 | 0.0% | -3.2% | 1.02 | 0.245 | 2.0 |
| v2 | moderate | 36 | -1.2% | 4.8% | -6.1% | 8.7% | -1.42 | 1.7% | -4.9% | 1.00 | 0.128 | 17.4 |
| v2 | saturated | 38 | -0.1% | 4.4% | +0.1% | 1.4% | -1.70 | 3.9% | -3.0% | 0.97 | 0.088 | 30.3 |

The engine RNG changes the draws but not the result: moderate p50 −1.2% (prototype −1.2%), saturated −0.1% (−0.6%), light p50 −0.2% unchanged, drop gaps −1.4 and −1.7 pp as before. On the reference backend (CPU, 2 threads, shared lab box) the v2 arm took 1.11 times the wall time of `primary_eng` over the whole sweep (1.03 times at N = 64).

## Reproducing

On the lab box, in a directory holding the `data/` of `lena_extract.py` and a checkout in `repo/`:

```bash
export ISAACLAB_NET_LENA_TABLES=<path to lena_eesm_tables.npz> REPO=$PWD/repo PYTHONPATH=$PWD/repo
python repo/benchmarks/fidelity/loadfix/lena_pipeline.py <ns3ref>/sweep/nofade data results      # 5G-LENA grant accounting
mkdir -p replay && ln -s <fidelity study replay>/primary replay/primary                          # the base arm
bash repo/benchmarks/fidelity/loadfix/run_loadfix.sh                                              # 9 arms (prototype names, via the shim)
python repo/benchmarks/fidelity/loadfix/loadfix_replay.py data replay all_no_retx all 4 64 lf.retx_tdma=False   # one leave-one-out job
python repo/benchmarks/fidelity/compare.py data replay results
python repo/benchmarks/fidelity/loadfix/report_loadfix.py results results/loadfix
bash repo/benchmarks/fidelity/loadfix/run_v2.sh              # v2 (engine RNG), v2_global, primary_eng; about 15 min
python repo/benchmarks/fidelity/compare.py data replay results
python repo/benchmarks/fidelity/loadfix/report_loadfix.py results results/loadfix_v2
```

`run_v2.sh` calls `nr_replay.py` with `NRF_PRESET=lena_validation_v2`, so the v2 arm is the plain engine, not the shim.

### Result files (`benchmarks/fidelity/results/loadfix/`)

| File | Content |
|:---|:---|
| `lena_pipeline_per_run.csv`, `lena_pipeline_by_regime.csv` | 5G-LENA grant accounting per run and by regime and frame size: new TBs, TB and RLC bytes, data / padding / empty / bootstrap RBGs and TBs, retransmissions and slots blocked by them, tail stalls, SRs, grant lead time, HARQ gap |
| `arms_by_regime.csv` | every arm by regime (and seed 2 alone): signed and absolute medians of p50 / p95 / p99 error, drop difference, goodput, PRB error, TB ratio, KS, W1 |
| `per_run_<arm>.csv`, `summary_<arm>.csv` | `compare.py` outputs of every arm, same columns as the fidelity study |
| `tail_check.csv`, `mcs_check.txt` | share of frames completed by an RLC tail PDU per run (`tail_check.py`); engine vs 5G-LENA UL MCS per UE (`mcs_check.py`) |
| `doc_tables.md` | the arm tables of this page, as `report_loadfix.py` writes them |
| `../loadfix_v2/arms_by_regime.csv`, `per_run_*.csv`, `summary_*.csv`, `ablation.csv` | the engine-integrated rerun: arms `v2`, `v2_global` (bitwise the prototype's `all`) and `primary_eng` |
