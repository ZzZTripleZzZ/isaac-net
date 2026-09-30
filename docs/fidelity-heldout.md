# Held-out 5G-LENA scenario (10 MHz carrier)

The `lena_validation_v2` preset has no fitted continuous parameter, but its four MAC switches and `tb_overhead_bytes = 8` were chosen while looking at the 153 runs of the primary sweep ([fidelity-load-gap.md](fidelity-load-gap.md)), all on one carrier, one TDD pattern and one arena. This page replays one 5G-LENA scenario that none of those choices was checked against, with the preset unchanged except for the carrier fields. The code is in `benchmarks/fidelity/heldout/` and every number below is in a file under `benchmarks/results/fidelity_heldout/`.

## Scenario

Everything is as in the validation scenario of [validation-5g-lena.md](validation-5g-lena.md) (single cell, 100 m arena, flat channel, whole-band UE power, thermal noise with a 7 dB noise figure, 6 dB shadowing, the ≥ 7 dB coverage rule, OFDMA PF, SR/BSR, HARQ with 4 transmissions, EESM IR T1, RLC UM with a 2 s discard), except for the rows below.

| Item | Primary sweep | Held-out scenario |
|:---|:---|:---|
| Carrier | 20 MHz, `RbOverhead` 0.1: 50 PRBs in 5 RBGs of 10 PRBs | 10 MHz, `RbOverhead` 0.279: 20 PRBs in 2 RBGs of 10 PRBs |
| UE drops | seeds 1, 2, 3 | seed 4 |
| Grid | N ∈ {1, …, 64}, nominal load 0.1–1.2 of 8 Mb/s | N ∈ {8, 16, 32}, nominal load 0.1–1.8 of 3.2 Mb/s (8 Mb/s × 20/50) |
| Runs | 153, three seeds | 39, one seed |

Both frame sizes (4 kB and 30 kB), the 100 ms in-phase schedule, the send-probability rule p = min(1, f · C / (N · S · 80 b/s)) and the 30 s of traffic are those of the sweep. When p saturates at 1, only the first such point is kept, as in the sweep, which removes one point at N = 8 with 4 kB.

**Why the carrier.** It is the cheapest change the reference program accepts on its command line (`--bw`, `--rbOverhead`), whereas its TDD pattern is fixed in the source. It still changes most of what the v2 switches act on: the TB sizes, the number of RBGs the PF scheduler hands out per UL slot (2 instead of 5), the per-PRB SNR under whole-band power (about 4 dB higher for the same UE) and the cell capacity (2.1–3.5 Mb/s of goodput in saturated runs, against 5–6.6 Mb/s in the sweep). The seed gives new UE positions and shadowing.

**Why `RbOverhead` 0.279.** 5G-LENA uses ⌊B (1 − overhead) / 360 kHz⌋ PRBs and ⌊PRBs / RBG size⌋ RBGs. With the sweep's overhead of 0.1 a 10 MHz carrier has 25 PRBs, of which the scheduler can grant only the 20 in 2 full RBGs while the UE spreads its power over all 25. An overhead of 0.279 gives exactly 20 PRBs, so the granted and the powered bandwidth agree, as they do in the sweep and in the engine. The per-PRB noise does not change (5G-LENA reports the same 6.997 dB noise figure). The 5G-LENA slot trace (`gnb_slots.csv`) reports 20 available RBs in every UL slot.

**Engine side.** `nr_replay.py` with `NRF_PRESET=lena_validation_v2` and the overrides `bandwidth_mhz=10 n_prb=20`, nothing else. As in the sweep, each run is replayed four times on 5G-LENA's per-UE `snr1_db` (full UE power in one 10-PRB subband) and its exact message schedule, and `compare.py` computes the per-run errors.

## Results

Medians over runs, NR engine minus 5G-LENA, with the regime set by 5G-LENA's drop rate (light < 1%, moderate 1–20%, saturated ≥ 20%). The sweep rows are the v2 rows of the fidelity table, recomputed from the sweep's `per_run_v2.csv` with the same script (`summarize.py`). The last column is the median first-transmission BLER of each side.

| Set | Regime | Runs | p50 | p95 | Drop (pp) | KS | W1 (ms) | First-tx BLER, 5G-LENA / NR |
|:---|:---|---:|---:|---:|---:|---:|---:|---:|
| sweep, 20 MHz | light | 79 | −0.2% | −2.5% | 0.00 | 0.25 | 2.0 | 0.48% / 0.91% |
| | moderate | 36 | −1.2% | −6.1% | −1.42 | 0.13 | 17.4 | 0.31% / 1.02% |
| | saturated | 38 | −0.1% | +0.1% | −1.70 | 0.09 | 30.3 | 0.03% / 0.84% |
| **held-out, 10 MHz** | light | 22 | −4.7% | −3.7% | 0.00 | 0.15 | 4.3 | 0.08% / 0.72% |
| | moderate | 6 | −10.0% | −7.8% | −1.28 | 0.14 | 35.1 | 0.02% / 0.73% |
| | saturated | 11 | −6.8% | 0.0% | −4.09 | 0.08 | 41.5 | 0.00% / 0.65% |
| | all | 39 | −5.7% | −2.9% | 0.00 | 0.14 | 12.5 | 0.02% / 0.71% |

**The load behavior carries over.** In loaded cells the delay distributions stay as close as in the sweep (KS 0.14 and 0.08 against 0.13 and 0.09), and the p95 errors are of the same size (−7.8% and 0.0% against −6.1% and +0.1%). The v2 switches were derived on a 5-RBG slot, and with 2 RBGs per slot the per-RBG PF update and the grant pipeline still produce 5G-LENA's delay shape under load.

**The median delay is 5–10% low at every load.** On the held-out carrier the engine is faster than 5G-LENA at the median in all three regimes (−4.7%, −10.0% and −6.8%), where the sweep had −0.2%, −1.2% and −0.1%. W1 is 1.4 to 2.2 times the sweep's value. In saturated cells the engine also drops 4.1 pp too few messages, against 1.7 pp in the sweep. The two largest per-run errors are the saturated N = 16, 4 kB runs, where the engine's p50 is 125 ms against 229 and 321 ms (−45% and −61%) and it carries 4–11% more goodput than 5G-LENA (3.52–3.54 against 3.18–3.39 Mb/s). A likely cause is the MCS residual below, since one MCS step is a larger share of a 20-PRB cell's capacity, but this has not been isolated.

## Residuals: BLER and MCS

Two residuals of the sweep show up again on the held-out carrier, in the same direction.

- **First-transmission BLER.** The engine fails 0.7 pp more first transmissions than 5G-LENA (0.71% against 0.02%), where the sweep gave about 0.5 pp (0.95% against 0.28% over all runs).
- **UL MCS one step high.** On the sweep, the engine's UL MCS is one step above 5G-LENA's median first-transmission MCS for 49% of 3,583 UEs, one step below for 3% and equal for 48%, with the difference in the 3–15 dB SNR bands ([fidelity-load-gap.md](fidelity-load-gap.md), `mcs_check.py`; open item 2 in [STATUS.md](STATUS.md)). On the held-out carrier, `mcs_heldout.py` applies the engine's rule to the SINR that 5G-LENA measured for each of 752 RNTIs: the engine picks a higher MCS for 55% of them with a 1-RBG allocation and 52% with 2 RBGs, a lower one for none, and it is higher for 62–87% of the RNTIs between 9 and 21 dB (`mcs_heldout.txt`).

The two residuals fit together: 5G-LENA's link adaptation settles one MCS below the engine's for about half the UEs and therefore almost never fails a first transmission, while the engine runs closer to its 10% target. The cause of the step is not isolated. Neither a constant SINR offset nor the search order explains it (section (b) of [fidelity-load-gap.md](fidelity-load-gap.md)).

## Caveats

- **One scenario, one seed.** 39 runs, 6 of them moderate. A second TDD pattern or arena has not been run, because the TDD pattern is fixed in the reference program's source.
- **Per-UE columns.** The reference program writes `snr_bw_db` in `ues.csv` for 5 RBGs, so at 10 MHz it is 4 dB below the SINR that 5G-LENA sees. `lena_extract.py` maps RNTIs to UEs by that value, and the map fails for some UEs of this scenario (`rnti_map_max_err_db` up to 35 dB in `lena_runs.csv`). The per-UE TB and cell-edge columns of `per_run_v2.csv` are therefore not reliable here. The five columns above and the BLER come from run-level totals and are unaffected, and `mcs_heldout.py` needs no map.
- **Cost.** The 39 ns-3 runs take under a minute at 8 in parallel on the lab box, and the CPU replay 50–103 s per N group (`replay_groups.csv`).

## Reproduce

On the lab box, with the 5G-LENA reference program of the validation built (see [validation-5g-lena.md](validation-5g-lena.md)) and the local 5G-LENA tables:

```bash
export REPO=/path/to/isaaclab-net ISAACLAB_NET_LENA_TABLES=/path/to/lena_eesm_tables.npz
export NETSLOT_REF_BIN=/path/to/ns3.48-netslot-ref-optimized PARSE_RUN=/path/to/parse_run.py
mkdir heldout && cd heldout && bash $REPO/benchmarks/fidelity/heldout/run_heldout.sh
```

The folder then holds `manifest.txt`, the ns-3 run directories under `sweep/`, `data/` (`lena_extract.py`), `replay/v2/`, `results/per_run_v2.csv` and `summary_v2.csv` (`compare.py`), `table.md` (`summarize.py`) and `mcs_heldout.txt`. The copies in `benchmarks/results/fidelity_heldout/` are these files without the raw traces.
