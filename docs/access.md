# Access: RACH, connection setup and DRX

By default every robot is connected and awake for the whole episode. `NRConfig(rach=True)` and `NRConfig(drx=True)` turn on a per-robot access state machine on level `L2` (`core/access.py`). A robot is scheduled, in either direction, only while it is connected and awake. Its messages wait in the queue otherwise, so the frame buffer, the timeouts and the AoI apply as usual.

```python
from isaac_net.core import NRConfig, make_engine
from isaac_net.core.energy import EnergyConfig

cfg = NRConfig(rach=True, rach_initial="idle", rach_release_after_ms=2000,
               drx=True, drx_inactivity_ms=100, drx_cycle_ms=160, drx_on_ms=10,
               energy=EnergyConfig(drx_sleep_power_w=0.002))
net = make_engine("L2", E, R, "cuda", cfg, backend="graph")
out = net.step(None, poses)
out["access_state"]        # [E, R] 0 idle, 1 RACH, 2 connected, 3 DRX-dormant (at the step's last slot)
out["rach_attempts"]       # [E, R] preambles sent this step
out["access_sleep_frac"]   # [E, R] share of the step dormant or idle (read by EnergyLoop)
net.counters()["access"]   # rach_attempts, rach_collisions, rach_successes, rach_failures, rrc_releases
```

## State machine

```
            UL or DL data arrives
   IDLE (0) ----------------------> RACH (1) --- preamble alone at its RO ---> RAR + Msg3/Msg4 ---> CONNECTED (2)
      ^                              |   ^                                                         |    ^
      |                              |   | backoff U[0, rach_backoff_ms], next RO                   |    |
      |                              +---+ same preamble as another robot of the cell (collision)   |    |
      |                                    rach_max_attempts failures: the procedure restarts       |    |
      |                                                                                             |    |
      +------------------ rach_release_after_ms without activity (RRC release) -------------------+    |
                                                                                                        |
   CONNECTED (2) <---- next on-duration, or UL data with drx_ul_wake="sr" ----> DORMANT (3)            |
                 ----- drx_inactivity_ms without scheduling activity ------>                -----------+
```

## Fields

| Field | Default | Meaning | Source |
|:---|:---|:---|:---|
| `rach` | False | contention-based random access before a robot is served | TS 38.321 §5.1 |
| `rach_occasion_slots` | 20 | RACH occasion (RO) period; the RO is the first UL-capable slot of each window. Must be a multiple of the TDD period | TS 38.211 §6.3.3.2 (PRACH configuration period) |
| `rach_preambles` | 64 | contention-based preambles per cell and RO | TS 38.211 §6.3.3.1 (64 preambles per cell) |
| `rach_rar_window_slots` | 10 | preamble to RAR; the RAR is taken at the end of the window | TS 38.321 §5.1.4 (ra-ResponseWindow) |
| `rach_msg3_slots` | 10 | RAR to contention resolution (Msg3 on PUSCH, Msg4) | TS 38.321 §5.1.5 |
| `rach_backoff_ms` | 20 | backoff after a collision, uniform in [0, value] | TS 38.321 §5.1.4, Table 7.2-1 |
| `rach_max_attempts` | 10 | preambleTransMax | TS 38.321 §5.1.4, TS 38.331 |
| `rach_initial` | "connected" | state after a reset: "connected" (as without RACH) or "idle" (a fleet that powers on) | — |
| `rach_release_after_ms` | None | RRC release to idle after this much inactivity; None = never | TS 38.331 (RRCRelease), network inactivity timer |
| `drx` | False | connected-mode DRX | TS 38.321 §5.7 |
| `drx_inactivity_ms` | 100 | drx-InactivityTimer | TS 38.321 §5.7, TS 38.331 DRX-Config |
| `drx_cycle_ms` | 160 | drx-LongCycle | same |
| `drx_on_ms` | 10 | drx-onDurationTimer | same |
| `drx_short_cycle_ms` | None | drx-ShortCycle; None = no short cycle | same |
| `drx_short_cycles` | 2 | drx-ShortCycleTimer, in short cycles | same |
| `drx_start_offset_ms` | 0 | drx-StartOffset | same |
| `drx_ul_wake` | "sr" | UL data while dormant: "sr" wakes the robot at once (a pending SR counts as Active Time), "on_duration" waits for the next on-duration | TS 38.321 §5.7 (Active Time) |

`EnergyConfig.drx_sleep_power_w` (default None = `idle_power_w`) is the power while DRX-dormant or idle; see [background-energy-sharding.md](background-energy-sharding.md#radio-energy).

## How it is modelled

**Gating.** The access stage ANDs its mask into `MacLink.sched_ok`, the mask the handover interruption already uses, for every UL and DL data slot. The MAC code is unchanged. With several cells the mask is combined with the handover mask. The SR state machine keeps running while a robot is not schedulable, so its grant is ready when it becomes schedulable, which stands for the buffer status that Msg3 carries.

**RACH.** UL data (a `submit()` before the step or a traffic-model message at its arrival slot) or DL data for an idle robot starts the procedure. The robot sends a preamble at the first RO at or after the arrival. Each robot draws one of `rach_preambles` preambles from the engine's counter RNG. Two or more robots of the same env and cell that draw the same preamble at the same RO collide. The counts per (env, cell, preamble) come from one `scatter_add` over `[E, C · preambles]`, with no loop over robots. A collision fails for every robot involved, since Msg3 capture is not modelled. A successful robot is served from RO + `rach_rar_window_slots` + `rach_msg3_slots` on. A colliding robot learns of the failure at the same time, backs off, and retries at the next RO after the backoff. For 32 robots that power on together on 64 preambles, the share whose first preamble succeeds matches (63/64)^31 = 0.614 (tested over 256 envs).

**RLF re-establishment.** With `rach=True` and radio link failure on (`rlf=True`, several cells, [multicell.md](multicell.md#radio-link-failure)), re-establishment goes through this RACH model instead of the fixed `reest_delay_ms`. When the cell search of `CellAssociation` selects a suitable cell, after the RLF declaration or for a robot that went idle at T311 expiry, the robot enters RACH toward that cell and contends with that cell's robots. Contention resolution ends the outage: the robot is served, and attached to the new cell, from RO + `rach_rar_window_slots` + `rach_msg3_slots` on, so `out["rlf"]` covers the access delay, and collisions, backoff and failed procedures count as for any other attempt. The stage draws its ROs at the start of each control step, so a selection made during a step uses the first RO of the next step at the earliest. The interface is two calls on `CellAssociation`: `take_reest_requests()` returns the robots that selected a cell and the slot of the selection, and `rach_connected(mask, g)` reports the slot from which service starts. Both use fixed `[E, R]` masks and no host sync. With `rach=False` the fixed `reest_delay_ms` applies as before (tested bitwise against the engine before the change).

**DRX.** A connected robot is awake (Active Time) while the inactivity timer runs, during the on-duration of its cycle, or, with `drx_ul_wake="sr"`, while it has UL data. The inactivity timer restarts in every slot in which the robot is awake and has data or a waiting HARQ process in that direction. This stands for the PDCCH of a new transmission, so the timer starts once the buffers have drained. On-durations follow the global slot clock (the SFN), so an env's resets do not move them. DL data for a dormant robot waits for the next on-duration.

**Randomness and batching.** The preamble and backoff draws use the engine's counter RNG (sites 16 and 17 of `nr_rng.py`), keyed by seed, env, episode and step. An env's access process therefore does not depend on E, on other envs' resets, or on sharding (tested). All state is fixed-shape `[E, R]`, and nothing syncs with the host. The graph backend registers the stage's state with its other state and captures it. A partial reset returns the reset envs to `rach_initial` and clears their procedures.

## Backends and levels

| Backend | RACH / DRX |
|:---|:---|
| `reference` | yes |
| `graph` | yes (same ops as the reference; GPU equivalence test `test_graph_backend_bitwise_equal_reference`) |
| `triton` | refused in `NRTritonEngine.__init__` (`TritonUnsupported`, a `ValueError` and a `NotImplementedError`): the fused kernel has no schedulable-mask input; see [NR engine backends](configurability.md#nr-engine-backends) |

Every level other than `L2` refuses `rach=True` or `drx=True` (`make_engine` raises), and `unused_fields("L2")` lists the RACH fields as unused while `rach` is False and the DRX fields while `drx` is False.

## What 5G-LENA does

5G-LENA models contention-based RACH (preamble, RAR, Msg3) with ideal or real RRC. It has no DRX. Together with the energy model, DRX here gives the wake-up latency of DL commands and the battery saving of sleeping robots, which 5G-LENA cannot show.

## Limitations

- No Msg3 capture: every collision fails for all robots involved. Preamble detection is otherwise perfect, and the RAR always fits.
- No contention-free RACH, also not at RLF re-establishment, and no RACH on handover. A handover keeps its own interruption model.
- Paging is not modelled: DL data for an idle robot starts RACH at its arrival.
- The RRC release is decided per control step from the step's first data arrival.
- With several cells, a robot contends in the cell it was associated with at the end of the previous step.
- DRX has no separate HARQ RTT or retransmission timers. A waiting HARQ process keeps an awake robot awake, and a dormant robot's retransmission waits for the next on-duration.
- `access_sleep_frac` is sampled at the slots the engine runs, which is every slot with data symbols of an active direction. On an uplink-only config those are the UL slots.
- RLF re-establishment through RACH: T301 (the re-establishment timer) is not modelled, so a robot whose procedures keep failing retries until it connects instead of going idle; while the robot waits for its first RO, `access_state` still shows its state before the failure.
