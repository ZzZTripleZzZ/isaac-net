# Channel models

`NRConfig.channel` selects the large-scale channel that `radio.RadioMC` computes for every robot–cell link. The engines see only its output, the path gain `[E, R, C]` in dB, through the existing interfaces (`NREngine.step(t, poses)`, `step_rx(pathgain, ...)`, `step_cells(t, pathgain)` and `NetSlotMC`). Fast fading stays the AR(1) Rayleigh model of the NR engine, and its Doppler can now follow each robot's own speed. Every model keeps fixed-shape state, redraws the rows of reset envs without a host sync, and evaluates with fixed-shape tensor ops, so a step can run inside a CUDA graph (`tests/test_channels.py::test_channel_cuda_graph_capture`).

| `channel` | What it computes | Main fields |
|:---|:---|:---|
| `log_distance` (default) | `pl_const_db + 10 n log10(d)`, a correlated shadowing field and an optional short-range white component | `pl_const_db`, `pathloss_exp`, `shadow_sigma_db`, `shadow_modes`, `shadow_dcorr_m`, `shadow_acf`, `shadow_white_frac`, `shadow_white_dcorr_m` |
| `tr38901` | TR 38.901 path loss, LOS probability with a spatially consistent LOS state, shadow fading and O2I penetration | `tr38901_scenario`, `tr38901_los`, `carrier_ghz`, `gnb_height_m`, `ue_height_m`, `o2i_indoor_frac`, `o2i_model`, `inf_clutter_*` |
| `radio_map` | a precomputed gain map per cell, sampled bilinearly at the robot positions | `radio_map_path` (or `RadioMC(..., radio_map=RadioMap)`) |
| add-on `blockage` | other robots are spheres that cost `blockage_loss_db` when they sit on the robot–gNB segment; or TR 38.901 model B screens (robots plus per-step `blockers=`), or model A angular regions ([obstacles.md](obstacles.md)) | `blockage`, `blockage_radius_m`, `blockage_loss_db`, `blockage_model`, `blocker_size_m`, `blockage_max_db` |
| add-on LOS state from geometry | LOS / NLOS of every link from a baked `los_prob` map, a 2.5-D ray march over an `obstacle_z` height map, or a `blocked_fn` callback, with optional knife-edge diffraction; TR 38.901 soft LOS for the stochastic state ([obstacles.md](obstacles.md)) | `los_source`, `los_raycast_samples`, `los_diffraction`, `los_soft`, `nlos_extra_loss_db` |
| add-on per-robot Doppler | AR(1) fading correlation from each robot's speed (NR engine) | `fading_doppler="per_robot"`, `doppler_min_speed_mps` |

```python
from isaac_net import NRConfig, make_engine

cfg = NRConfig(channel="tr38901_inf_sh", blockage=True, fading_doppler="per_robot")   # indoor factory
eng = make_engine("L2", E, R, "cuda", cfg)
out = eng.step(None, poses)              # poses [E,R,2|3]; velocities from consecutive poses
out = eng.step(None, poses, vel=vel)     # or pass them [E,R,2|3] m/s
```

`channel="tr38901_<scenario>"` (`rma`, `uma`, `umi`, `inh`, `inf_sl`, `inf_dl`, `inf_sh`, `inf_dh`) is shorthand for `channel="tr38901", tr38901_scenario=...`. Levels that do not use `RadioMC` (the prototype levels, the surrogates and the bounds) list the channel fields in `NRConfig.unused_fields(level)`, and `make_engine(..., strict=True)` refuses them. `L2-legacy` with any non-default channel runs `NetSlotMC`, which uses `RadioMC` and so honors every model except per-robot Doppler (its fading is the legacy constant).

## `log_distance`

The default is the earlier radio bit for bit: one field per (env, cell), the sum of `shadow_modes` plane waves with wavelengths uniform in 20–60 m, drawn in the same order from the same generator (`test_default_is_bitwise_legacy` compares against an inline copy of the old code, including a partial reset).

Under `rng="engine"`, the engines draw the fields from their counter RNG (log-distance streams 10–12, equal to the prototype `Radio`), so an env's channel depends only on (seed, env id, episode). The "bit for bit" statement above refers to the generator path (`rng="global"`).

- **Decorrelation distance.** `shadow_dcorr_m` is the 1/e distance of the field's autocorrelation. With `shadow_acf="sos"` the legacy wavelength band is scaled by `shadow_dcorr_m / 10.3`. The default 10.3 m is the value the calibration measured on the legacy field (the analytic 1/e distance of that band is 10.19 m), so the default draws are unchanged. The legacy band has a negative lobe of about −0.18 near 2.1 `shadow_dcorr_m`.
- **Exponential autocorrelation.** `shadow_acf="exp"` draws the wavenumbers from the radial spectrum of `exp(-r / d)` (the Gudmundson model that TR 38.901 uses for spatial consistency): `|k| = sqrt((1 - u)^-2 - 1) / d` with `u ~ U(0, 1)`. The ensemble autocorrelation is then exactly `exp(-r / d)` and has no negative lobe. A single field with few modes is less Gaussian than the legacy one, so raise `shadow_modes` (16–32) when the marginal matters.
- **White component.** `shadow_white_frac = w` moves a share `w` of the variance into a second exponential field with a short decorrelation distance `shadow_white_dcorr_m` (1 m by default). A robot that stands still keeps its value and a robot that moves a few metres sees it decorrelate, which matches the "already decorrelated at 2.5 m" component of the public drive data. The total standard deviation stays `shadow_sigma_db`.

The calibration in [calibration-public-data.md](calibration-public-data.md) suggests `NRConfig(shadow_acf="exp", shadow_dcorr_m=30.0, shadow_white_frac=0.5)` for outdoor campus drives. No preset applies it, because those drives are far from an indoor warehouse.

## `tr38901`

Implemented from TR 38.901 V17.0.0: path loss and shadow-fading sigma from Table 7.4.1-1, LOS probability from Table 7.4.2-1, O2I from Tables 7.4.3-1 and 7.4.3-2, InF parameters from Tables 7.2-4 and 7.8-7, and correlation distances from Tables 7.5-6 and 7.6.3.1-2. The formulas are in `core/channels/tr38901.py` and accept tensors or Python floats.

| Scenario | Default h_BS | LOS-state correlation | SF sigma LOS / NLOS (dB) | SF correlation LOS / NLOS | O2I |
|:---|:---|:---|:---|:---|:---|
| RMa | 35 m | 60 m | 4 (6 beyond d_BP) / 8 | 37 / 120 m | low-loss |
| UMa | 25 m | 50 m | 4 / 6 | 37 / 50 m | low or high |
| UMi (street canyon) | 10 m | 50 m | 4 / 7.82 | 10 / 13 m | low or high |
| InH (mixed office LOS probability) | 3 m | 10 m | 3 / 8.03 | 10 / 6 m | none |
| InF-SL (sparse clutter, low BS) | 1.5 m | d_clutter / 2 = 5 m | 4.3 / 5.7 | 10 / 10 m | none |
| InF-DL (dense clutter, low BS) | 1.5 m | 1 m | 4.3 / 7.2 | 10 / 10 m | none |
| InF-SH (sparse clutter, high BS) | 8 m | 5 m | 4.3 / 5.9 | 10 / 10 m | none |
| InF-DH (dense clutter, high BS) | 8 m | 1 m | 4.3 / 4.0 | 10 / 10 m | none |

The InF LOS probability is `exp(-d_2D / k_subsce)` with `k_subsce = -d_clutter / ln(1 - r)`, multiplied by `(h_BS - h_UT) / (h_c - h_UT)` for SH and DH. The defaults are the calibration values (r = 0.2, d_clutter = 10 m, h_c = 2 m for sparse clutter; 0.6, 2 m and 6 m for dense clutter), so `k_subsce` is 44.8, 2.18, 582.6 and 3.15 m for SL, DL, SH and DH. `inf_clutter_density`, `inf_clutter_size_m` and `inf_clutter_height_m` override them.

**LOS state.** Each (env, cell) has a unit plane-wave field `z(x)` with an exponential autocorrelation at the scenario's LOS-state correlation distance. The field is mapped through the exact CDF of its own marginal (a table of 65,536 sorted samples of a sum of `shadow_modes` cosines), which gives a uniform `u(x)`. The link is LOS where `u(x) < Pr_LOS(d_2D)`. Over many envs the LOS share at each distance equals `Pr_LOS` (`test_los_probability_statistics`), a robot that stays put keeps its state, and a robot that moves about one correlation distance draws a new one. This is the procedure of TR 38.901 Sec. 7.6.3.1 with a hard threshold instead of the soft LOS of Sec. 7.6.3.3. Because the exponential autocorrelation is rough, the state can flicker near the probability threshold when a robot moves a few centimetres (about 2% of links for a 0.1 m move in UMi). `tr38901_los="los"` or `"nlos"` forces the state.

**Shadow fading.** Two unit fields per (env, cell), with the LOS and NLOS correlation distances, scaled by the scenario's sigma and selected by the LOS state.

**O2I.** With `o2i_indoor_frac > 0` (UMa, UMi, RMa) each robot is indoors with that probability. An indoor robot gets its own `d_2D-in = min(U(0, 25), U(0, 25))` metres (10 m for RMa) and its own penetration-loss draw `N(0, sigma_P^2)`, all redrawn at reset. It adds `PL_tw + 0.5 d_2D-in + N(0, sigma_P^2)` (12.7 dB wall loss and sigma_P = 4.4 dB for the low-loss model at 3.5 GHz, 26.8 dB and 6.5 dB for the high-loss model), and its LOS probability uses `d_2D-out = d_2D - d_2D-in`.

**Simplifications.** UMa uses `h_E = 1 m`, which is exact for robots below 13 m. Distances below the applicability ranges (10 m outdoors, 1 m indoors) extrapolate the formulas, with `d_2D >= 1 m`. The optional single-slope NLOS formulas, InH open office, InF-HH and the < 6 GHz backwards-compatible O2I model (Table 7.4.3-3) are not implemented. Heights are constants (`gnb_height_m`, `ue_height_m`): the z of 3-D poses is ignored, as in the other models. The LOS state changes only the path loss and the shadowing. Fast fading stays Rayleigh, with no Ricean K-factor.

Heights are checked against TR 38.901 Table 7.4.1-1: UMa and UMi need 1.5 m <= `ue_height_m` <= 22.5 m (below 1 m the breakpoint distance is not positive), RMa 1–10 m, and InF-SH/DH need `ue_height_m` < `inf_clutter_height_m` < BS height. Out-of-range heights raise `ValueError` when the channel is built instead of extrapolating. For ground robots with antennas below 1.5 m, use InF (InF-SL/DL have no UT-height term), a calibrated `log_distance`, or a radio map.

## `radio_map`

A map is an `.npz` (or `.pt`) with `gain_db [C, H, W]`, the path gain in dB between cell `c` and a robot antenna at grid point `(x_j, y_i)`, and `bounds = (x0, y0, x1, y1)`, the coordinates of the first and last grid points. Rows run along y and columns along x. `RadioMap.sample(pos)` interpolates bilinearly in dB and clamps to the border outside the bounds. A map may carry `gnb_xy` metadata, and `RadioMC` then checks that `cell_positions_m` matches it, because association, interference and blockage use the configured positions. The map has no per-env randomness: a partial reset changes nothing, and fast fading is the only random part of the link.

`radio_map_path="synthetic"` loads the tiny map shipped in `core/data/radio_map_synthetic.npz` (two gNBs at (25, 75) and (125, 75) m, 16 × 16 points over 150 m, log-distance loss plus a 10 dB wall at x = 75 m; `python -m isaac_net.tools.make_synthetic_radio_map` regenerates it):

```python
cfg = NRConfig(channel="radio_map", radio_map_path="synthetic", n_cells=2,
               cell_positions_m=((25.0, 75.0), (125.0, 75.0)), noise_model="thermal")
```

**Baking a map with Sionna RT.** `isaac_net/tools/bake_radio_map_sionna.py` runs Sionna RT's `RadioMapSolver` and writes the file. It needs only numpy and `sionna-rt`, so run it as a script in a separate environment. Without `--scene` it builds a simple warehouse (concrete floor and walls, a row of metal shelves, no ceiling). It was tested on the lab box with sionna-rt 2.2.0 (Mitsuba 3.9.1, Dr.Jit 1.5.0) in a side conda env: a 60 × 40 m map at 1 m cells with two gNBs at 6 m, 2 × 10^6 samples per gNB and 4 bounces took 4.4 s on the CPU (`--variant llvm`). In the open parts of the hall the map lies within 0.4–1.7 dB of free-space loss, as expected for LOS plus reflections. The CUDA variant needs OptiX, which WSL 2 does not expose without extra setup, hence the CPU variant there.

```bash
python -m venv rtenv && rtenv/bin/pip install sionna-rt
rtenv/bin/python isaac_net/tools/bake_radio_map_sionna.py --out warehouse.npz --arena 60 40 \
    --gnb 15 20 6 --gnb 45 20 6 --fc 3.5 --cell 1.0 --samples 2000000 --depth 4 --variant llvm
```

**LOS and obstacle grids.** A map may also carry `los_prob [C, H, W]` (`bake.py --los-map`) and `obstacle_z [H, W]` (`bake.py --obstacle-z`). `RadioMap` keeps them as attributes (`los_prob`, `obstacle_z`) instead of metadata, and files without them load as before. They feed `los_source="map"` and `"raycast"`, also for the `tr38901` and `log_distance` channels. With `channel="radio_map"` the LOS state adds no path loss, because the map already holds the NLOS loss ([obstacles.md](obstacles.md)). `python -m isaac_net.tools.make_synthetic_radio_map --obstacles hall.npz` writes a synthetic hall with both grids.

**From an Isaac Sim USD stage.** `python -m isaac_net.tools.scene.bake --usd scene.usd ...` exports the stage with ITU radio materials and bakes the map, and `IsaacNetCfg(scene_map=...)` does the same at env creation from the running stage. See [scene-radio-map.md](scene-radio-map.md).

## Blockage

With `blockage=True` every other robot of the same env is a sphere of radius `blockage_radius_m` centred at its antenna position `(x, y, ue_height_m)`. A link loses `blockage_loss_db` (20 dB by default, the value of the Isaac layer's blockage) when the segment from the robot's antenna to the gNB antenna passes through at least one sphere. A robot never blocks itself, and a sphere behind the robot or beyond the gNB does not count. The test is one pairwise tensor op of shape `[E, R, R, C]`, cheap up to a few hundred robots per env. The gNB height is the scenario's for `tr38901` and `gnb_height_m` otherwise (unset means 2-D geometry at robot height, where any robot on the line blocks). The loss is one fixed value per link. It does not model diffraction around the body, several blockers adding up, or static obstacles, which belong in a radio map or in the Isaac layer's mesh ray test (`isaac/radio.py`).

This is `blockage_model="sphere"`, the default. `blockage_model="screen"` (TR 38.901 model B, knife-edge screens for robots and for per-step `blockers=` such as people and vehicles) and `"stochastic"` (model A, angular regions for scenes without geometry) are described in [obstacles.md](obstacles.md), together with `RadioMC.los_state()` / `blocked_state()` and the `los` / `blocked` step keys.

## Per-robot Doppler

The NR engine's fading is `h <- rho h + sqrt(1 - rho^2) n` per elapsed interval, with one `rho` per ms for all robots (`fading_rho_per_ms`, or `ue_speed_mps` through `fading_rho_from_speed`). With `fading_doppler="per_robot"` each robot gets `rho_ms = J0(2 pi f_D 2.5 ms)^(1 / 2.5)` from its own speed, with `f_D = v f_c / c`, the same rule applied per robot. The speed is `|vel|` when `step(..., vel=...)` passes velocities, else the pose difference over one control step. Right after a reset, a robot has no previous pose and uses `doppler_min_speed_mps` (0 by default: a still robot's fading is frozen). Per-robot Doppler needs pose input, because SNR input carries no motion. Speeds apply from the step in which they are measured.

`NREngine` installs this by replacing `_evolve` on its own `NRNet` instance (`channels/doppler.install_per_robot_fading`). The replacement makes the same `randn_like` calls in the same order and differs only in using a per-robot `rho` tensor, so with every robot at the global speed it reproduces the global model up to float rounding (`test_per_robot_fading_equals_global_at_equal_speed`). If the NR engine's fading code changes, this hook must follow it. The natural long-term home is an optional `rho` tensor in `NRNet._evolve` itself.

## Cost

`RadioMC.pathgain_db` at `E = 128`, `R = 32` and three hex cells, eager, median of 50 calls on the shared RTX 4090 (82% busy with other jobs): `log_distance` 0.62 ms, with the exponential field and a white component 0.94 ms, `tr38901` InF-SH 1.95 ms, UMi with O2I 5.8 ms, `radio_map` (two cells) 1.3 ms, and `blockage` adds about 1.0 ms. The radio runs once per control step. The reference NR engine's step at the same size took 1,337 ms with the default channel and 1,361 ms with InF-SH, blockage and per-robot Doppler, so the channel is below 0.5% of a step. The times are dominated by kernel launches, so they grow slowly with `E`.
