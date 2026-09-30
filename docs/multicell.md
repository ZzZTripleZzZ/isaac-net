# Multi-cell networks

A fleet that spans a warehouse or a campus is served by several gNBs, so robots see inter-cell interference and hand over between cells as they move. The engine supports up to 7 cells per environment, batched over the cell dimension with no Python loop over cells. The multi-cell model runs on the NR engine (`make_engine("L2", config=multicell(n))`, uplink and downlink) and on the legacy slot-level engine (`NetSlotMC` in `isaaclab_net/core/proto/netsim_mc.py`, reached through `make_engine("L2-legacy", config=multicell(n))`, uplink only). The NR version ports the legacy design onto the NR MAC and adds downlink interference. This page describes the design shared by both, the choices behind the defaults, and what a sanity sweep and a handover check showed.

## Design

**Configuration.** The cells block of `NRConfig` holds the layout (`cell_layout` hex, grid or custom, 1–7 cells, `cell_isd_m`, `cell_center_m`), the link budget and noise model, the optional fractional uplink power control (`ul_pc`, `ul_pc_p0_dbm`, `ul_pc_alpha`), and the handover settings in ms (`a3_offset_db`, `a3_hyst_db`, `a3_ttt_ms`, `ho_interruption_ms`, `ho_rlc`). The default cell is the legacy single cell (gNB at the origin, a fixed −90 dBm noise-plus-interference floor), so single-cell behavior is unchanged everywhere. The preset `multicell(n)` gives a hexagonal cluster at 100 m inter-site distance centered in the 150 m arena, thermal noise with a 5 dB gNB noise figure (−103.4 dBm per 10-PRB subband), same-slot uplink interference, power control with P0 = −88 dBm per subband and α = 1, and A3 handover with 3 dB hysteresis, 300 ms time-to-trigger, a 40 ms interruption and lossless RLC carry-over.

**Radio and association** live in `isaaclab_net/core/radio.py` and are shared by both engines. `RadioMC` gives per-link path loss plus one shadowing field per (env, cell), and at C = 1 it draws the same field as the legacy radio bit for bit. `CellAssociation` handles max-RSRP attach, the A3 event with hysteresis and time-to-trigger fired at its exact slot, and the interruption window. Both support partial `reset(env_ids)` without host syncs.

**Scheduling.** Each cell runs its own PF scheduler over the same carrier, batched as a masked max over `[E, C, R]`. In the NR engine, each robot's MAC state stays per robot, `MacLink.member [E, C, R]` runs retransmission admission and PF per RBG once per cell, and robots inside a handover interruption are masked out.

**Interference timing.** Uplink interference at a gNB comes from the robots that other cells scheduled on the same subband in the same slot, at their transmit power after the power split and power control, through each interfering link's own fading. The rule for simultaneity is that **link adaptation (the MCS choice and the power-headroom cap) uses the noise plus interference measured in the previous slot, and decoding uses the actual same-slot noise plus interference**. A real gNB cannot know the next slot's interference when it picks the MCS, and OLLA absorbs the resulting error. An EWMA over the measured interference (`li_alpha`, 1 = previous slot) is available. In the NR engine, the downlink mirrors this with interference from every other gNB that transmits on the RBG, and link adaptation there covers the scheduler estimate, MCS, PHR cap and downlink CQI. Interference needs `noise_model="thermal"`, because the fixed −90 dBm floor already includes interference.

**Handover.** When A3 fires, the robot moves to the target cell and cannot be scheduled for the interruption time. With `ho_rlc="carry"` (the default) its queued frames, including a partly sent head-of-line frame, continue at the target cell, and with `flush` they are dropped and logged as drops. At the target, OLLA, HARQ combining, the pending SR and the PF average reset, and the BSR is known. In the NR engine, HARQ processes with undecoded bytes restart their combining at the target under RLC AM and are lost under RLC UM or with `flush`.

**Equivalence to the single-cell engine.** At C = 1 with the fixed noise floor, `NetSlotMC` is bitwise identical to the single-cell legacy engine at every step (all frame fields, MAC state, fading, finish times, outputs and logged statistics) at 64 × 16, 64 × 32 and 256 × 16 over 300 heavily loaded steps. This is `tests/test_multicell.py`, on CPU and GPU. In the NR engine none of the multi-cell code runs at C = 1, and a frozen copy of the single-cell NR engine (971fc12) in `tests/nr_frozen/` checks that every output and state tensor stays bitwise equal. A partial reset at C = 3 with interference and 366 handovers leaves the 60 non-reset envs bitwise identical to a run without the reset, and introspection finds no state tensor that the reset misses.

## Why power control is on by default

**Uplink fractional power control is on by default whenever `n_cells > 1`** (`ul_pc=None` means automatic, and `ul_pc=True/False` overrides it). The sweep below shows why. At 60 m inter-site distance with every robot at 23 dBm and no power control, three cells deliver *less* than one cell at moderate load: 114 kB per env-step with 8.7% drops against 127 kB with 0.8% drops at 4 kB per robot-step. A single central cell already puts most robots near the spectral-efficiency cap (median SNR about 24 dB, where the cap needs about 22 dB), and reuse-1 neighbours at full power cost more than the extra bandwidth adds, with a mean interference-over-thermal of about 25 dB and a 10th-percentile SINR of about −3 dB. Interference also changes from slot to slot as neighbours reschedule. A slow EWMA on the measured interference (`li_alpha = 0.05`) did not help (106 kB, 14% drops), so the loss comes from the interference level itself and not from the previous-slot link-adaptation rule. With α = 1 power control (P0 = −88 dBm per subband, which targets about 15 dB SNR per subband) capacity rises monotonically with cell count at every load. The preset also uses 100 m spacing, and the alternative to power control is an inter-site distance of at least 100 m. P0 and α have not been calibrated against a reference simulator.

## Sanity sweep

The sweep ran the legacy multi-cell engine with R = 32 robots, E = 64 envs and 120 steps, robots random-walking at up to 3 m/s, and a hexagonal layout at **60 m** inter-site distance centered in the 150 m arena. Offered load is in kB per robot-step, and "delivered" is the bytes of completed frames per env-step. IoT is the interference over thermal. The sweep was run before the package port, which added the per-env clock and the PF-average floor, so the numbers may shift slightly on the current code.

| Configuration | Load | Delivered (kB/env-step) | Delay p50 / p95 (ms) | Drops | SINR p10 / p50 / p90 (dB) | SNR − SINR p50 (dB) | SINR edge / centre (dB) | IoT (dB) |
|:---|---:|---:|:---|---:|:---|---:|:---|---:|
| 1 cell, fixed floor, corner gNB (legacy) | 4k | 37.7 | 468 / 1973 | 66% | −13.9 / 4.4 / 17.3 | 0 | | |
| 1 cell, centre, thermal | 4k | 126.6 | 70 / 200 | 0.8% | 12.9 / 23.6 / 33.2 | 0 | | |
| 1 cell, centre, thermal | 8.4k | 141.5 | 1643 / 1980 | 19% | 17.7 / 23.9 / 31.6 | 0 | | |
| 3 cells, no interference | 8.4k | 268.9 | 30 / 100 | 0% | 16.9 / 26.7 / 41.9 | 0 | 22.3 / 33.3 | |
| 3 cells, no interference | 30k | 373.9 | | 55% | | | | |
| 3 cells | 1.2k | 38.3 | 35 / 75 | 0% | −0.6 / 12.1 / 28.2 | 13.6 | 9.2 / 16.2 | 17.8 |
| 3 cells | 4k | 114.1 | 60 / 1293 | 8.7% | −2.9 / 11.0 / 23.3 | 20.1 | 7.5 / 14.1 | 25.2 |
| 3 cells | 8.4k | 151.7 | 335 / 1968 | 31% | −3.9 / 12.1 / 24.1 | 21.9 | 7.2 / 15.6 | 26.8 |
| 3 cells | 30k | 118.7 | | 85% | | 23.1 | 6.6 / 18.9 | 28.1 |
| 7 cells | 4k | 126.5 | 50 / 125 | 0.9% | −1.4 / 8.9 / 20.8 | 26.0 | 6.4 / 12.4 | 27.7 |
| 7 cells | 8.4k | 217.9 | 68 / 1880 | 12% | −1.9 / 8.7 / 20.1 | 26.4 | 5.7 / 12.7 | 29.1 |
| 7 cells | 30k | 298.4 | | 64% | | 28.6 | 6.0 / 15.9 | 31.4 |
| 1 cell, centre, power control | 8.4k | 70.0 | | 45% | 11.9 / 17.1 / 20.8 | 0 | | |
| 3 cells, power control | 8.4k | 94.6 | | 38% | 2.0 / 10.4 / 17.4 | 5.5 | 12.4 / 8.9 | 9.4 |
| 7 cells, power control | 8.4k | 176.3 | 875 / 1973 | 17% | −0.5 / 6.7 / 13.3 | 8.0 | 8.4 / 5.3 | 10.5 |

"Edge" is robots whose geometry SINR is below 3 dB and "centre" above 10 dB. Empty cells were not reported in the sweep summary.

**Interference responds to load and to cell count.** The median SNR − SINR gap grows with load (3 cells: 13.6, 20.1, 21.9 and 23.1 dB from 1.2k to 30k) and with cell count (about 26–29 dB with 7 cells), and the mean IoT follows. Turning interference off raises the median SINR by 15–18 dB.

**Cell-edge robots see lower SINR without power control**, 7.5 against 14.1 dB with 3 cells at 4k and 5.7 against 12.7 dB with 7 cells at 8.4k. With α = 1 power control the ordering flips (edge 12.4 against centre 8.9 dB), because full path-loss compensation equalizes received power, so a victim's SINR no longer depends on its own geometry, and edge robots show up as the loud interferers instead.

**More cells give more capacity when resources are the limit.** At 30 kB per robot-step, goodput rises from 33 (1 cell) to 119 (3 cells) to 298 (7 cells) kB per env-step, and with power control it is monotone at every load (70, 95 and 176 kB at 8.4k). Without interference, 3 cells reach 374 kB, 2.6× one cell.

**The legacy single-cell setting is a very different network.** With the corner gNB and the fixed −90 dBm floor, far-corner robots up to 212 m away sit near −8 dB SNR, and the cell loses 34% of frames already at 1.2 kB per robot-step. A thermal-noise cell at the arena centre serves the same robots easily. Random-walking robots hand over about 0.1 (3 cells) and 0.2 (7 cells) times per robot-minute.

## Handover check

One robot drives along y = 75 m from x = 0 to 150 m at 0.3 m per step through cells at x = 25, 75 and 125 m (ideal boundaries at 50 and 100 m), sending 30 kB every step, while 15 static robots send small frames at 0.4 per step. There are 32 envs with different shadowing and 500 steps.

| Variant | Handovers per traversal (mean, range) | Exactly 2 | Ping-pong within 1 s | Handover position p10 / p50 / p90 (m) | Served in interruption | Backlogged handover to first service |
|:---|:---|---:|---:|:---|---:|:---|
| 3 dB hysteresis, 300 ms TTT, 40 ms interruption | 2.09 (2–4) | 94% | 0 | 45.5 / 50.8 / 60.4 and 96.2 / 104.4 / 112.2 | 0 | 42.5 ms (all) |
| no hysteresis, no TTT | 2.19 (2–4) | 88% | 0 | 43.4 / 48.5 / 57.8 and 92.1 / 101.1 / 109.8 | 0 | 42.5 ms p50, 52.5 ms p90 |
| default with `flush` | 2.09 | 94% | 0 | same as default | 0 | queue flushed (15 frames dropped) |
| default with 0 ms interruption | 2.09 | 94% | 0 | same as default | | 2.5 ms (next UL slot) |

Two boundary crossings give two handovers in 94% of envs, and the extra ones come from shadowing and happen more than 1 s apart. Hysteresis plus time-to-trigger delays the handover by about 2–4 m, which matches 300 ms at 0.3 m per step plus the 3 dB margin. The interruption is exact: the robot is never served inside it, and a backlogged robot is served in the first UL slot after it (16 slots plus one, 42.5 ms). With lossless carry-over the driving robot still delivers its full 30 kB per step.

## NR engine against the legacy engine

`benchmarks/multicell/sweep_nr.py` runs the sweep above on the NR engine, and with `--engine mc` on `NetSlotMC`, with the same poses, traffic, shadowing and config (E = 16, R = 32, 150 steps). Association does not depend on the MAC, so serving cells and handover counts are identical. Without power control the two engines agree to about 1 dB in interference over thermal, SINR and SNR−SINR gap, and to a few percent in delivered goodput; at 100 m spacing and 8.4 kB per robot-step, three cells deliver 200 against 195 kB per env-step with 22.2 dB mean interference over thermal in both. With power control the NR engine delivers more (three cells at 8.4 kB: 160 against 117 kB), likely because power control also enters its scheduler estimate and its power-headroom cap. The NR engine takes the headroom cap from the noise-only SNR, since power headroom does not depend on interference; the legacy engine uses the SINR. Capacity with power control rises with the cell count (15 kB per robot-step: 44, 76 and 185 kB for 1, 3 and 7 cells). In decoded bytes, α = 1 power control at reuse 1 is interference-limited and nearly flat in the cell count, and the extra cells mainly help edge robots complete their frames. The downlink at reuse 1 with 43 dBm gNBs sees 23–31 dB interference over thermal, growing with load and cells. `tests/test_nr_multicell.py` checks these trends at test size (`slow`) and cross-checks the NR engine against `NetSlotMC`.

## Cost

With the PyTorch profiler, whose counts do not depend on GPU contention, the legacy multi-cell engine launches about 20% more kernels than the single-cell engine and uses 20–27% more GPU time, independent of whether there are 3 or 7 cells and nearly independent of E × R. C = 1 costs almost nothing extra. The NR engine with several cells launches about 19% more kernels than with one (32k against 27k per control step at 64 × 16, uplink only), again independent of 3 or 7 cells, and the downlink multiplies the work by about 4.5. Wall-clock times, measured with the GPU 98% busy, and the full table are in [performance.md](performance.md). The multi-cell engines run on the `reference` backend only, but the slot loop is sync-free, so the `graph` and `triton` backends should carry the same relative overhead.

## Open items

P0 and α should be calibrated against a reference such as 5G-LENA, and multi-cell has not been compared with any reference simulator yet. The downlink has no power control or interference coordination.
