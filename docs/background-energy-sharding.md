# Background users, radio energy and multi-GPU sharding

Three additions that sit on top of the engines and leave them unchanged. Background users load every cell with UEs the policy does not control (`core/background.py`). The energy model turns each robot's radio activity into joules and a battery state (`core/energy.py`). `ShardedEngine` splits a batch of envs over several GPUs behind the API of one engine (`core/sharded.py`). The first two are set through `NRConfig` and applied by `make_engine`, and all three keep the engine contract (fixed shapes, partial resets; see [the engine reference](reference/engine.md)).

```python
from isaac_net import NRConfig, make_engine
from isaac_net.core import BackgroundConfig, EnergyConfig, ShardedEngine, TrafficModel

cfg = NRConfig(background=BackgroundConfig(n_background=8, traffic=(TrafficModel.video(fps=30, mean_frame_bytes=4000),),
                                           mobility="random_waypoint", speed_mps=1.5),
               energy=EnergyConfig(battery_j=500.0, low_battery_frac=0.2))
net = make_engine("L2", E, R, "cuda", cfg)          # BackgroundLoop, then EnergyLoop on top
out = net.step(None, poses)                         # robot keys [E, R, ...] + bg_* [E, C] + energy_* / battery_* [E, R]
obs = net.energy_obs()                              # [E, R, 2]: state of charge, low-battery flag

big = ShardedEngine("L2-legacy", 16384, 16, ["cuda:0", "cuda:1"], NRConfig(), backend="graph", seed=0)
```

`make_engine` stacks the wrappers as background (innermost), then the edge loop of `NRConfig.edge`, then energy (outermost). Each wrapper passes unknown attributes through to the engine below it.

## Background users

`BackgroundConfig` places `n_background` UEs in every cell. They share the carrier with the robots, but the policy neither sends their messages nor sees their state. Every per-robot output keeps its shape `[E, R, ...]`.

| Field | Default | Meaning |
|:---|:---|:---|
| `n_background` | 0 | background UEs per cell; 0 makes `make_engine` return the plain engine, so the robots' outputs are bitwise unchanged |
| `placement` | `"random"` | `"random"`: uniform in the region at every reset; `"fixed"`: `positions_m`, one (x, y) per UE, cell by cell |
| `region` | `"auto"` | the arena with one cell, a disc of `cell_radius_m` (default `cell_isd_m / 2`) around each gNB with several cells; or `"arena"` / `"cell"` explicitly |
| `arena_m` | `(0, 0, cell_arena_m, cell_arena_m)` | arena bounds (x0, y0, x1, y1) in env-local metres |
| `traffic` | periodic 1000 B every 20 ms | models from `core/traffic.py` run by every background UE: `periodic`, `bursty`, `video` |
| `mobility`, `speed_mps` | `"static"`, 1.0 | or `"random_waypoint"`: straight to a uniform waypoint in the region, then a new one |
| `height_m` | 1.5 | z of the background UEs when the robots' poses are 3-D |
| `max_util` | 0.95 | offered-load levels: cap on a cell's background share |
| `seed` | engine seed | seed of the placement and waypoint draws |

Placement and waypoints come from the wrapper's own counter-based streams (`proto/rng.py`), keyed by (seed, env id, episode). A partial reset therefore redraws the background of exactly the reset envs, and an env draws the same background in any shard of a `ShardedEngine`.

**On `L2`, background users are ghost robots.** The NR engine runs with R + R_bg robots, R_bg = `n_background` × C. Rows R to R + R_bg − 1 carry the background traffic models inside the step. They compete for RBGs in the PF scheduler, hold HARQ processes, interfere with the other cells, and attach and hand over like robots. The wrapper restricts the user's traffic models to rows 0 to R − 1 and pads every `[E, R, ...]` input for the ghost rows: poses get the background positions, an SNR or path-gain input gets the log-distance path loss of those positions (no shadowing), and submits and event triggers get nothing. Every `[E, R + R_bg, ...]` output is sliced back to `[E, R, ...]`. Background traffic is uplink only. Attributes reached through the wrapper (the MAC state, `traffic_stats`) still include the ghost rows. `EdgeConfig(return_path="nr_dl")` cannot be combined with background users.

**On `L1` and `L2-legacy`, the background is an offered load.** The frozen prototype schedulers (reference, graph, compile and Triton backends) have no room for rows with generated traffic, so the background enters as a fixed load term, which is an approximation. Each background UE u offers λ_u bytes per control step, the mean rate of its traffic models. It needs the share ρ_u = λ_u / (K · S · 180 · se(snr_u)) of its cell's RBG-slots, where K is the number of UL slots per step, S = 5 subbands, 180 bytes is one subband-slot at 1 bit/s/Hz, and se is the legacy spectral efficiency at the UE's full-power SNR on one subband (log-distance path loss to the nearest gNB, no shadowing or fading). A cell's load is ρ_c = min(`max_util`, Σ ρ_u). The background then takes the share ρ_c of every resource granted to a robot of that cell. The wrapper does this by scaling every robot message by 1 / (1 − ρ_c) of the robot's serving cell when it is enqueued, writing the engine's queue buffer in place (graph-safe). For `L1`, which gives every backlogged robot an equal share and serves bytes at a rate independent of the backlog, this is exactly a capacity reduction by 1 − ρ_c, and the test checks that. For `L2-legacy`, PF, SR/BSR, HARQ and fading run as before and the robots' queues drain 1 / (1 − ρ_c) times slower. The background's own burstiness and its PF competition are not modelled. ρ_c is re-evaluated every step from the background positions, and a message keeps the factor it was enqueued with. `queue_bytes` is reported in the robots' own bytes. Levels without a capacity model (`L0`, `L0DR`, `L05`, `L05Q`, the surrogates and the bounds) refuse a background config.

Step dict keys, per cell `[E, C]`:

| Key | Ghost robots (`L2`) | Offered load (`L1`, `L2-legacy`) |
|:---|:---|:---|
| `bg_n` | background UEs served by the cell | UEs whose nearest gNB it is |
| `bg_offered_bytes` | accepted generated bytes on the air this step | the mean offered bytes Σ λ_u |
| `bg_util` | PRB-slots of background transmissions / PRB-slots of the step's UL data slots | ρ_c |
| `bg_delivered_bytes`, `bg_lost_bytes`, `bg_queue_bytes` | delivered, timed out or dropped this step, queued after it; offered = delivered + lost + queued over an episode | not reported |

`bg_pos` `[E, R_bg, 2]` gives the background positions after the step.

### Downlink background

Two `BackgroundConfig` fields load the downlink of `L2` (with `NRConfig(dl=True)`). Both default off, and then every output is bitwise the engine without them.

| Field | Default | Meaning |
|:---|:---|:---|
| `dl_traffic` | `()` | DL traffic models (`periodic`, `bursty`, `video`) of every background UE. On `L2` the ghost rows get them as [downlink models](configurability.md#downlink-models): the gNB queues their messages in the ghosts' DL queues and the DL PF scheduler serves ghosts and robots together, so the ghosts take DL RBGs, HARQ processes and, with several cells, cause DL interference like robots |
| `dl_load_frac` | 0.0 | offered-load DL share f, 0 ≤ f < 1: background traffic that is not simulated UE by UE takes the share f of every DL RBG, so each RBG keeps (1 − f) of its PRBs for the simulated UEs (robots and DL ghosts). It scales the DL MAC's PRBs per RBG, hence the DL capacity, by 1 − f. It also works with `n_background = 0`, where it is the only background |

The offered-load mode of `L1` and `L2-legacy` has no downlink to scale (those engines are uplink-only), so they refuse both fields with a `ValueError`, as does `L2` without `dl=True`. `dl_load_frac` changes the DL MAC's PRB table, which the `triton` kernel reads from the config, so `triton` refuses it; `dl_traffic` runs on the reference and graph backends, and on triton through the kernel's DL arrival gate.

Step dict keys added with `dl_traffic`, per cell `[E, C]`: `bg_dl_offered_bytes` (accepted generated DL bytes on the air), `bg_dl_delivered_bytes`, `bg_dl_lost_bytes`, `bg_dl_queue_bytes` (offered = delivered + lost + queued over an episode) and `bg_dl_util` (PRB-slots of the ghosts' DL transport blocks over the DL carrier's PRB-slots of the step, `cfg.dl_nprb × dl_slots_per_step`). The robots' own DL frames are reported by the per-frame `dl_*` keys of the engine, sliced to `[E, R, Fd]`.

## Radio energy

`EnergyConfig` adds per-robot energy and a battery to any level. Per robot and control step:

energy = tx_energy / `pa_efficiency` + `tx_circuit_w` · tx_time + `rx_power_w` · rx_time + `idle_power_w` · step duration + `msg_energy_j` · messages

| Field | Default | Meaning |
|:---|:---|:---|
| `tx_power_dbm` | None | None: the engine's per-slot power on `L2`, `ue_tx_dbm` elsewhere; a value fixes it |
| `pa_efficiency` | 1.0 | radiated power / power drawn for it |
| `tx_circuit_w` | 0.0 | extra power while transmitting |
| `rx_power_w` | 0.1 | while receiving a scheduled DL data slot (`L2` with `dl=True`) |
| `idle_power_w` | 0.02 | baseline over the whole step |
| `msg_energy_j` | 0.001 | per accepted message (policy submits and, on `L2`, generated messages) |
| `battery_j` | 3600 | battery capacity |
| `initial_soc` | 1.0 | state of charge after a reset, or a range (lo, hi) drawn per robot at every reset |
| `low_battery_frac` | 0.2 | threshold of the `low_battery` flag; None turns it off |
| `drx_sleep_power_w` | None | power while DRX-dormant or RRC-idle ([access.md](access.md)); None = `idle_power_w` |

**Sleep power with DRX and RRC idle.** With `NRConfig(drx=True)` or `rach=True` on `L2`, the step dict carries `access_sleep_frac`, the share of the step the robot spent DRX-dormant or idle ([access.md](access.md)). With `drx_sleep_power_w` set, the idle term becomes (`idle_power_w` · (1 − f) + `drx_sleep_power_w` · f) · step duration; with the default None the energy is unchanged.

**Transmit energy on `L2` is counted per slot.** A read-only tap on the MAC's per-slot SINR hook (`core/slot_tap.py`) sees, in every UL data slot, which robots send a transport block and on how many PRBs. For each of those transmissions it adds the transmit power of that slot times the PUSCH duration (slot × data symbols / 14). The power is the engine's own: `ue_tx_dbm` split over the allocation (the total stays at `ue_tx_dbm` with `ul_power="allocated"`), the fixed PSD of `ul_power="whole_band"`, and the fractional power-control backoff when it is on. The tap returns the SINR it receives, so the engine's outputs stay bitwise unchanged (tested), and a user hook set with `set_sinr_hook` through the wrapper is chained after it. `tx_slots` equals the number of transport blocks the MAC sent, retransmissions included (tested against the MAC counters). SR and HARQ-ACK transmissions on PUCCH are not counted.

**Transmit energy on the other levels is an approximation.** Their outputs have no per-slot record, so the wrapper converts delivered bytes into airtime. The on-air bytes of the messages delivered in the step (the `bytes` key, else the size of their class) are sent at the legacy rate at the robot's SNR: n = floor(10^((snr − 3)/10)) subbands in [1, 5] (the power-headroom cap of `L2-legacy`), spectral efficiency se(snr − 10 log10 n), 180 bytes per subband per bit/s/Hz in one slot, and 1 / (1 − `bler_target`) transmissions per TB, at `tx_power_dbm` or `ue_tx_dbm`. The airtime is attributed to the step in which a message is delivered, and messages that time out add none.

Step dict keys `[E, R]`: `energy_j` (this step), `energy_tx_j`, `energy_cum_j` (since the env's last reset), `tx_slots`, `rx_slots`, `battery_j`, `battery_frac`, `low_battery`, `battery_empty`. The battery clamps at 0, and the engine does not gate a robot with an empty battery; what that means is the task's choice. A partial reset restores the reset envs' batteries and zeroes their counters. The energy step has fixed shapes and no host syncs, and `EnergyLoop(engine, cfg, graph=True)` captures it in a CUDA graph. Around the `L2-legacy` graph backend it is bitwise equal to the eager step on the reference engine, through partial resets (`tests/test_energy.py`). `EnergyLoop` needs the dict step form and raises on the legacy `step(t, x, cur_hid)` form, and `EnergyConfig.seed` takes precedence over the engine seed. Under `make_adaptive`, `energy` and `edge` wrap the adaptive engine; background users are refused.

## Multi-GPU sharding

`ShardedEngine(level, E, R, devices, config, backend, seed=..., split=None)` builds one independent engine per entry of `devices` with `make_engine`. Shard i holds the contiguous global envs `offsets[i]` to `offsets[i] + split[i] − 1` (near-equal by default). `locate(e)` returns (shard, local id), and `shard_of` / `local_of` give the whole map. `submit`, `step`, `reset`, `clock`, `queued` and `energy_obs` keep the engine API. Inputs whose first dim is E are sliced and moved to each shard's device, and outputs are concatenated on `devices[0]`. `reset(env_ids)` turns the ids into a mask and gives every shard its slice. A device may repeat, so two shards can share one GPU.

**Shard invariance.** Engine-owned draws (`NRConfig.rng="engine"`, the default) are keyed by env id. Before this change the key was the row index inside the engine, which is offset-based, so an env drew different numbers in shard 1 than in the unsharded batch. `CounterRNG.set_env_offset(offset)` now keys row i by global id offset + i, in the torch path, in the Triton draw kernel, and in the `L2-legacy` Triton slot kernel (a new `env0` argument). `ShardedEngine` sets the offset on every stream of a shard's wrapper chain and redoes the shard's initial reset under the global keys. Every shard shares the seed, so the sharded engine is bitwise equal to the unsharded one. The tests check this with partial resets that cross the shard boundary for `L0`, `L0DR`, `L1`, `L2-legacy`, `ORACLE` and `NOCOMM` on the CPU reference, with background load and the energy model on top, for `L2`, `L2` with `multicell(3)`, multi-cell `L2-legacy` (`multicell(3)`) and `WIFI` on the CPU reference, and on one GPU for the `L2-legacy` reference, graph and Triton backends, `L1` Triton and `L0DR` graph. Two computations were moved to float64 for this: the energy airtime and the background geometry. A CPU elementwise op can round differently in the vectorized body and the scalar tail of a batch, so a float32 result could depend on the batch size. For the same reason the CPU test compares the float outputs of `L1` and `L2-legacy` to float32 rounding; draws, decisions and integer outputs stay exact, and on CUDA the comparison is bitwise.

The invariance does not hold where draws come from a sequential generator whose position depends on the batch: `L2` with traffic models (`NRConfig.traffic`, or background users, which `L2` runs as ghost traffic models; `TrafficGen` uses one generator per engine), any level with an edge loop that draws from the global RNG (`service_dist="exponential"` or `ret_jitter_ms > 0`), and `rng="global"`. There `shard_invariant` is False and every shard gets a distinct derived seed, so shards are independent but results depend on the split. Otherwise `L2` (one or several cells), multi-cell `L2-legacy` and `WIFI` are invariant: their radio (`RadioMC`) draws from the engine RNG keyed by global env id.

**Running on several GPUs.** The shards run one after another from one Python thread. The steps have no host syncs, so kernels on different GPUs overlap. Resets sync once per call. `benchmarks/bench_sharded.py` checks bitwise equality against one engine and times both. `scripts/hazel/sharded_gpu.sbatch` runs `tests/test_sharded.py` and the benchmark (`L2-legacy` graph and Triton, `L1` Triton, E = 4096 and 16384, R = 16) on a Slurm node with two GPUs:

```bash
sbatch scripts/hazel/sharded_gpu.sbatch                       # --gres=gpu:l40:2 by default
python benchmarks/bench_sharded.py --level L2-legacy --backend graph --E 8192 --R 16 --devices cuda:0,cuda:1
python benchmarks/bench_sharded.py --devices cuda:0,cuda:0    # correctness only, one GPU
```

Across two GPUs of the same model the kernels are identical, so the bitwise check applies there too. Across different GPU models, float results can differ in the last bits.

## Tests

`tests/test_background.py`, `tests/test_energy.py` and `tests/test_sharded.py`: n_background = 0 bitwise equal to the plain engine; output shapes; ghost byte conservation; per-cell attachment with several cells; capacity reduction at every supported level; the exact `L1` capacity factor; frame conservation and real-byte queue reporting under the offered load; partial-reset isolation; region and speed bounds; the energy tap bitwise transparent and equal to the MAC's TB count; the energy formula term by term; power control, DL receive slots, battery flag, partial reset, `initial_soc` ranges; `EnergyLoop(graph=True)` around the `L2-legacy` graph backend; the offered load on the graph and Triton backends; the `CounterRNG` offset; sharded == unsharded bitwise on CPU and GPU. `tests/test_dl_traffic_fdd_rem.py` covers the downlink background: ghost DL traffic and `dl_load_frac` lower a robot's DL throughput, ghost DL bytes are conserved, and the levels without a downlink refuse both fields.
