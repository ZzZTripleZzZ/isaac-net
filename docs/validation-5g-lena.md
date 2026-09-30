# Validation against ns-3 5G-LENA

The configurable NR engine (`L2`) is validated against ns-3 with the 5G-LENA NR module, the most widely used open 5G system-level simulator. This page describes the reference setup, the sweep that was run on it, the model differences that had to be resolved and how, and how closely the NR engine reproduces the sweep when it is fed the same per-UE link budgets. The reference work was done on 2026-09-29 on the lab box (WSL, Ubuntu 20.04). The standalone scenario, its patch script and the sweep scripts live in the project's research workspace, not yet in this repository (open item 13 in [STATUS.md](STATUS.md)). The two bridge programs derived from the scenario are in `isaaclab_net/bridges/ns3/` and are described in [bridges.md](bridges.md).

## Reference setup

| Component | Version |
|:---|:---|
| 5G-LENA (nr module) | v5.1, 2026-08-06, the latest tag (DOI 10.5281/zenodo.21770621), from `gitlab.com/cttc-lena/nr` |
| ns-3 | ns-3.48, the version the v5.1 `RELEASE_NOTES.md` declares compatible, from `gitlab.com/nsnam/ns-3-dev` |
| Compiler | conda-forge g++ 14.4.0 |
| CMake / Ninja / Python | 3.30.5 / conda ninja / 3.11.16 |
| Eigen3, SQLite | enabled |
| Local patch | a 6-line crash guard in `nr-amc.cc` (below) |

The NR README's install section is out of date (it still names ns-3.46 with v4.1.1), so the release notes are the authority. ns-3.48 needs g++ ≥ 11.1, CMake ≥ 3.25, Python ≥ 3.10 and C++23, which the lab box's system toolchain does not provide, so the build uses a user-level conda environment and needs neither sudo nor apt packages. The build is configured with `-d optimized --enable-examples --disable-tests --disable-python-bindings -G Ninja --enable-modules "nr;flow-monitor;applications;point-to-point;internet;mobility"` and takes about 10 minutes at `-j8`. Two pitfalls cost time. With gcc 13 the link fails on `__cxa_call_terminate@CXXABI_1.3.15` from the conda-forge ICU that SQLite pulls in, and gcc 14 fixes it. Two v5.1 API renames also break older examples: the RLC mapping attribute is now `ns3::NrGnbRrc::QosFlowToRlcMapping`, and `NrUeMac::UeMacStateMachineTrace` gained a leading `uint64_t imsi` argument. The Sionna-RT coupling example was not set up.

**The crash guard does not change behavior.** In RLC UM runs, the v5.1 uplink sub-band CQI path can pass an SINR vector that is exactly zero on every RB of the matched allocation. `NrAmc::CreateCqiFeedbackSiso` then hands the EESM model an empty RB map, and the run dies with `NS_ABORT` ("number of allocated RBs cannot be 0") 6–13 s into a 20 s run. The patch makes the AMC return MCS 0 / CQI 0 when no RB was measured. It only replaces an abort with MCS 0 in a case that would otherwise abort the run, so every run that completes without the patch is unchanged. Disabling SRS did not avoid the bug, and SRS is off in all runs anyway.

## Scenario

The scenario is a single-cell uplink written to mirror the engine's abstraction as closely as stock 5G-LENA allows.

| Item | Setting |
|:---|:---|
| Topology | 1 gNB at (0, 0, 10 m), N UEs at 1.5 m, remote host over 100 Gb/s with zero core delay |
| Carrier | 3.5 GHz, 20 MHz, μ = 1 (30 kHz), `RbOverhead` 0.1 giving 50 PRBs, `NumRbPerRbg` 10 giving 5 RBGs |
| TDD | `DL\|DL\|DL\|S\|UL\|` twice, one UL slot per 2.5 ms, 13 UL data symbols with SRS off |
| Path loss | custom `LogDistShadowLoss`: 40 + 35 log10(d2D), d ≥ 1 m, 6 dB log-normal shadowing frozen per link |
| MAC | `NrMacSchedulerOfdmaPF` (UL OFDMA PF), HARQ with at most 4 transmissions, real SR/BSR, UL AMC in ErrorModel mode, error model `NrEesmIrT1` (MCS table 1) |
| RLC | UM with PDCP discard at 2000 ms, or AM |
| Traffic | every 100 ms, all UEs in phase, each UE sends a frame of S bytes with probability p, as ⌈S/1400⌉ UDP packets with an 18-byte frame header |
| Outputs | per-frame delay and completion status, per-UE link budget (`ues.csv`), per-slot RB × symbol usage, NR PHY and MAC traces |

Every run exports each UE's shadowing and its single-subband full-power SNR (`snr1_db` in `ues.csv`), which is the quantity the engine takes as input. This is what allows the NR engine to be run on identical per-UE link budgets.

## Why this geometry

The validation geometry was chosen by testing candidate configurations with 3 seeds, 8 UEs, 4 KB frames and p = 0.5 in a 100 m arena with thermal noise:

| Configuration | First-transmission BLER | Frames delivered |
|:---|:---|:---|
| power over used RBs, fading on (stock default) | 0.39–0.40 | 45–51% |
| power over used RBs, no fading | 0.25–0.27 | 53–57% |
| whole-band power, fading on (3 m/s) | 0.40–0.41 | 48–49% |
| whole-band power, static frequency-selective channel | 0.18–0.65 | 44–84% |
| **whole-band power, no fading** | **0.01** | **100%** |

Stock 5G-LENA uplink AMC cannot track a frequency-selective channel. It picks the MCS from the SINR measured on the RBGs of the UE's last PUSCH, then grants other RBGs, and it has no outer-loop link adaptation (OLLA) to absorb the error, so first-transmission BLER sits near 40% even on a static channel. With power spread over the allocated RBs, a change in grant size also shifts SINR by up to 7 dB with the same effect. The only stock configuration with sane link adaptation is therefore a frequency-flat channel with whole-band UE power, and that is the primary arm. A secondary arm keeps stock 3GPP UMi fading at 3 m/s as a record of what stock 5G-LENA does with fading.

Cell-edge UEs below the MCS-0 point were the second problem. Without a coverage rule, a 16–64-UE drop almost always contains one deeply shadowed UE, which sends MCS-0 TBs that all fail. In one example, a single UE at −4.2 dB sent about 6,500 such TBs in 30 s and occupied about half of all UL slots, and the cell showed 25–43% BLER at 10% load, a PRB cap near 0.65 and repeated random access. The validation drop therefore redraws any UE whose SNR with 23 dBm over all 5 RBGs, shadowing included, is below 0 dB, which is equivalent to a single-subband full-power SNR of at least 7 dB. About 5% of UEs are redrawn at N = 64. With this rule, 64 UEs at 60% load reach BLER ≤ 0.01 and full PRB use. The same rule is why the validation arena is 100 m rather than the 150 m arena of the example fleet task: in the larger arena with a corner gNB, stock 5G-LENA leaves many robots uncovered (see the coverage finding in [bridges.md](bridges.md)).

The resulting validation configuration is a 100 m × 100 m arena with the gNB at the corner, thermal noise of −174 dBm/Hz with a 7 dB noise figure (−101.44 dBm per 10-PRB subband), no interference, 6 dB per-UE shadowing, whole-band UE power, fading off, and otherwise stock MAC and RLC (OFDMA PF, HARQ with 4 transmissions, SR/BSR, EESM IR T1, ErrorModel AMC, SRS off, RLC UM with 2 s discard).

## Sweep design

The primary arm crosses N ∈ {1, 2, 4, 8, 16, 32, 64} UEs, frame size S ∈ {4000, 30000} bytes, nominal load f ∈ {0.1, 0.3, 0.6, 0.9, 1.2} and 3 seeds, with 30 s of traffic per run. The send probability is p = min(1, f · C / (N · S · 80 b/s)) with C = 8 Mb/s, the PHY-level saturated throughput measured in this configuration (6.0–9.6 Mb/s with 4 UEs, 30 KB, p = 1). When p saturates at 1, which happens for small N, only the first such point is kept, so small-N rows cover less of the load range (N = 1 with 4 KB reaches only 4%). The measured frame-level saturation goodput is about 5–6.6 Mb/s, so relative to delivered capacity the true load is about 1.3× the nominal f. The summary file carries both offered and delivered rates, so any capacity definition can be applied afterwards.

The sweep summary contains 186 runs: 153 in the primary arm and 33 in the fading arm (N ∈ {4, 16, 64}, f ∈ {0.3, 0.9}, both frame sizes, 3 seeds, with one saturated point dropped). The written report states 150 and 36; the counts here are taken from `sweep_summary.csv`. At most 8 runs ran in parallel.

## Reference results

Seed means from the primary arm (excerpt):

| N | S | Nominal load | Drop | p50 / p95 delay (ms) | First-tx BLER | PRB use |
|---:|:---|---:|---:|:---|---:|---:|
| 8 | 4 KB | 0.30 | 0.00 | 42 / 77 | 0.003 | 0.59 |
| 8 | 30 KB | 0.61 | 0.07 | 121 / 460 | 0.004 | 0.75 |
| 8 | 30 KB | 1.21 | 0.32 | 503 / 1636 | 0.000 | 1.00 |
| 16 | 4 KB | 0.60 | 0.06 | 58 / 157 | 0.004 | 0.98 |
| 16 | 30 KB | 0.88 | 0.19 | 335 / 1440 | 0.001 | 1.00 |
| 32 | 4 KB | 0.90 | 0.24 | 136 / 1431 | 0.001 | 1.00 |
| 64 | 4 KB | 0.10 | 0.00 | 33 / 69 | 0.010 | 0.25 |
| 64 | 4 KB | 0.60 | 0.09 | 157 / 1031 | 0.001 | 1.00 |
| 64 | 4 KB | 1.20 | 0.46 | 322 / 1728 | 0.001 | 1.00 |
| 64 | 30 KB | 1.18 | 0.61 | 1145 / 1881 | 0.001 | 1.00 |

Below 30% nominal load, frames arrive in 31–113 ms at the median with no drops, BLER is 0.4–4% and HARQ retransmissions are 0–6% of TBs. From about 60% nominal load, which is about 80% of delivered capacity, PRB use reaches 0.93–1.0 and delays grow toward the 2 s deadline, and drops then come from RLC UM discard and from frames later than 2 s. With N ≤ 2 and 30 KB frames, 14–24% of frames drop at 30–60% cell load because a single UE's MCS, not the cell, is the bottleneck. The fading arm drops 52–98% of frames even at 30% load with BLER 0.23–0.32, which confirms that stock 5G-LENA with fading is not a usable reference without OLLA. The wall-clock cost of these runs is in [performance.md](performance.md).

## Model mismatches and which side was changed

Nineteen aspects were compared between the legacy slot-level model (NetSlot, now `L2-legacy`) and 5G-LENA as configured. The first recommendation was to patch 5G-LENA for power headroom, OLLA and stale-frame purging. The decision taken instead was to keep **5G-LENA as the stock reference with no behavior patches**, the crash guard being the only change, and to move **every row on the engine side**. The NR engine gained a switch for each mismatch, and the preset `lena_validation()` (built on `lena_like()`, also exported as `lena_match`) sets all of them.

| # | Aspect | Legacy engine | 5G-LENA as configured | How the NR engine matches |
|---:|:---|:---|:---|:---|
| 1 | Path loss | 40 + 35 log10(d2D) | same, custom model | already equal |
| 2 | Shadowing | 6 dB, spatially correlated field | 6 dB, frozen per link, independent across UEs | per-UE `snr1_db` fed directly, engine shadowing unused |
| 3 | Fast fading | i.i.d. Rayleigh per subband, AR(1) 0.93 per UL slot | 38.901 UMi NLOS | `fading=False`, fading off in both |
| 4 | Noise | −90 dBm per subband | thermal, single cell | `noise_model="thermal"` with the LENA noise figure |
| 5 | Grid | 50 PRB, 12 data symbols | 50 PRB, 13 data symbols minus DMRS inside the TBS | `ul_data_symbols=13`, `tbs_mode="lena"` |
| 6 | TDD | DDDSU | DDDSU, S slots carry no UL data | already equal |
| 7 | Power split and headroom | power split over won subbands, PHR cap | power spread over allocated RBs, no headroom limit in the UL scheduler | `ul_power="whole_band"`, `phr_cap=False` |
| 8 | PF metric | per-subband rate / EWMA | wideband MCS rate / average throughput | `pf_metric="wideband"` |
| 9 | Link adaptation | OLLA, 10% target | highest MCS with BLER ≤ 0.1 at the last PUSCH SINR, no OLLA | `olla=False`, MCS at 10% BLER on the known SINR |
| 10 | BLER model | logistic in SINR gap, SE floor, +3 dB per retransmission | EESM with LDPC curves, IR combining | `bler_source="lena"`, `eff_sinr="eesm"`, `harq_combining="ir_lena"` |
| 11 | HARQ | 1 process with head-of-line blocking, RTT 4 UL slots | 16 processes, retransmission on the next UL slot | `n_harq=16`, retransmission at the next UL slot |
| 12 | After 4 failed transmissions | wait, then resend (no loss) | RLC UM loses the TB and every frame in it | `harq_fail="drop"` |
| 13 | SR / BSR | minimal grant after exactly 2 UL slots, exact BSR | PUCCH SR, BSR MAC CE and K2 emerge from the MAC | `sr_grant_delay_slots=40`, **inferred** (below) |
| 14 | 2 s deadline | purge frames older than 2 s | drop *arriving* SDUs while head-of-line is over 2 s; frames over 2 s counted as drops afterwards | `discard="pdcp_arrival"` |
| 15 | Queue cap | 16 frames per robot | 100 MB RLC buffer | `frame_buffer=128`, irrelevant below overload |
| 16 | Bytes on air | payload only | about 4% header overhead | 50 bytes per 1400-byte packet, confirmed by LENA's 4150-byte buffer for a 4000-byte frame |
| 17 | Frame completion | when the byte counter passes the frame end | when the last packet reaches the remote host | already equal |
| 18 | Mobility | moving robots | static drops | validate per static drop |
| 19 | Access | always connected | ideal RRC; some UEs repeat random access when SR goes unanswered | not modelled; worth watching at high N |

## NR engine replay of the sweep

`python -m isaaclab_net.bridges.ns3_offline.lena_replay` replays all 153 primary-arm runs in the NR engine, 4 replicas each. Each run's per-UE `snr1_db` is the SNR input, the configuration is `lena_validation()`, and traffic is in phase every 100 ms for 30 s plus a 2.5 s drain. This covers every matched-configuration switch above except a TBS and overhead check against LENA's `NrUlMacStats.txt`, which was not available locally; `tbs_mode="lena"` implements LENA's TBS formula instead. Differences are NR engine minus 5G-LENA, per run:

| Subset | Runs | Drop rate | p50 delay | p95 delay | PRB use, NR / LENA | First-tx BLER, LENA / NR |
|:---|---:|:---|---:|---:|---:|:---|
| all | 153 | mean \|Δ\| 0.029 (max 0.15) | −20 ms | −30 ms | 0.83 | 0.0077 / 0.0051 |
| light (LENA drop < 1%) | 79 | −0.001 | −18 ms | −25 ms | 0.75 | 0.0091 / 0.0041 |
| saturated (LENA drop ≥ 20%) | 38 | −0.07 | −179 ms | −48 ms | 0.97 | 0.0007 / 0.007 |

Drop rate, BLER and saturation match, and the residual at light load is a near-constant access delay of about 18–20 ms. The mean absolute drop difference (0.029, max 0.146) and the median p50 offset (−20 ms) were rechecked against the replay CSVs for this page. The table was produced before the SR-to-grant default below was set. A sensitivity sweep over the SR-to-grant delay (N ≤ 4, 42 runs) moves the median p50 offset linearly: −17.6 ms at 3 slots, −12.6 ms at 13, −7.6 ms at 23, −2.6 ms at 33 and +2.4 ms at 43. About 40 slots (about 20 ms) aligns p50 and p95, and that is now the `lena_validation()` default. It is marked as inferred in the code, because 5G-LENA's SR-to-first-grant delay itself has not been measured.

The median KS distance between the delay distributions stays at 0.45–0.5. At light load both distributions are very narrow, so an offset of a few ms saturates the KS statistic, and p50/p95 offsets are the more meaningful measure until the access delay is measured. 5G-LENA also grants about 1.3× more PRBs at light load and sends TB bytes of about 1.5× the offered load, which fits BSR quantization and grant padding that the engine does not model yet.

**The PHY tables alone move the cell edge by several dB.** At equal MCS, 5G-LENA's 10%-BLER points lie +2.4 to +5.7 dB (mean +3.4 dB) above the Sionna AWGN curves that ship with the package. MCS 0, for example, sits at −0.65 dB in 5G-LENA and −6.4 dB in Sionna, and the legacy engine's logistic model lies between them at the low end (MCS 0 equivalent −4.7 dB) and above both at high spectral efficiency. Any fidelity comparison with 5G-LENA must therefore use `bler_source="lena"`, or the gap measures the tables and not the MAC. The 5G-LENA tables are GPL-2.0 data and are generated locally from the user's own checkout (see the README), and the lookup reproduces `NrEesmErrorModel` up to its 0.05 dB grid. While exporting the Sionna 2.2.0 tables, several data defects were found: `PUSCH_table1.json` keys 17–27 hold the PDSCH curves of MCS 18–28, `PUSCH_table2.json` is not 38.214 Table 5.1.3.1-2 (its 10% points lie about 8 dB below the Shannon limit), PDSCH table 2 MCS 27 never reaches 10% BLER within its range, and Sionna clamps to the edge value outside its grid. The package therefore uses the PDSCH curves in both directions, extends the grid, fills missing rows and forces monotonicity, after which every threshold lies 0.56–2.76 dB above Shannon.

## What remains

The formal comparison, with a fit and hold-out split of the SR-to-grant delay, ablations, the legacy engine and the fading arm, is now in [fidelity-vs-lena.md](fidelity-vs-lena.md). It also measures the SR-to-DCI delay (4.5 ms median), which shows that the fitted 20 ms is a proxy for 5G-LENA's grant pipeline rather than its SR delay. The remaining validation work is the formal comparison: delay CDFs by KS distance and p50/p95 per (N, S, p), per-UE goodput, the HARQ redundancy-version histogram and PRB use, after the SR-to-grant delay is measured and BSR quantization is added. The TBS and overhead check against `NrUlMacStats.txt` is also open. Validating fading needs a different reference, because stock 5G-LENA's uplink AMC fails with frequency selectivity: either a link-level check against LENA's EESM curves (`cttc-error-model`), or a patched 5G-LENA with OLLA as a separate, clearly labeled arm. Sionna-RT coupling would inherit the same AMC limitation and has low priority. Multi-cell has not been compared with 5G-LENA at all.
