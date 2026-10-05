# Channel models

`NRConfig.channel` selects the large-scale channel that `radio.RadioMC` computes for every robot–cell link. The engines see only its output, the path gain `[E, R, C]` in dB, through the existing interfaces (`NREngine.step(t, poses)`, `step_rx(pathgain, ...)`, `step_cells(t, pathgain)` and `NetSlotMC`). Fast fading is the AR(1) Rayleigh model of the NR engine, optionally with a Rician specular term whose K-factor follows the link's LOS state and with subbands correlated in frequency by a delay spread, and its Doppler can follow each robot's own speed. Every model keeps fixed-shape state, redraws the rows of reset envs without a host sync, and evaluates with fixed-shape tensor ops, so a step can run inside a CUDA graph (`tests/test_channels.py::test_channel_cuda_graph_capture`).

| `channel` | What it computes | Main fields |
|:---|:---|:---|
| `log_distance` (default) | `pl_const_db + 10 n log10(d)`, a correlated shadowing field and an optional short-range white component | `pl_const_db`, `pathloss_exp`, `shadow_sigma_db`, `shadow_modes`, `shadow_dcorr_m`, `shadow_acf`, `shadow_white_frac`, `shadow_white_dcorr_m` |
| `tr38901` | TR 38.901 path loss, LOS probability with a spatially consistent LOS state, shadow fading and O2I penetration | `tr38901_scenario`, `tr38901_los`, `carrier_ghz`, `gnb_height_m`, `ue_height_m`, `o2i_indoor_frac`, `o2i_model`, `inf_clutter_*` |
| `radio_map` | a precomputed gain map per cell, sampled bilinearly at the robot positions | `radio_map_path` (or `RadioMC(..., radio_map=RadioMap)`) |
| add-on `blockage` | other robots are spheres that cost `blockage_loss_db` when they sit on the robot–gNB segment; or TR 38.901 model B screens (robots plus per-step `blockers=`), or model A angular regions ([obstacles.md](obstacles.md)) | `blockage`, `blockage_radius_m`, `blockage_loss_db`, `blockage_model`, `blocker_size_m`, `blockage_max_db` |
| add-on LOS state from geometry | LOS / NLOS of every link from a baked `los_prob` map, a 2.5-D ray march over an `obstacle_z` height map, or a `blocked_fn` callback, with optional knife-edge diffraction; TR 38.901 soft LOS for the stochastic state ([obstacles.md](obstacles.md)) | `los_source`, `los_raycast_samples`, `los_diffraction`, `los_soft`, `nlos_extra_loss_db` |
| add-on per-robot Doppler | AR(1) fading correlation from each robot's speed (NR engine) | `fading_doppler="per_robot"`, `doppler_min_speed_mps` |
| add-on Rician fading | specular term with a per-link K-factor, fixed or from the LOS state (NR engine) | `fading_rician`, `rician_k_db`, `rician_k_from_los`, `rician_k_ramp_slots` |
| add-on frequency-selective fading | subband fading correlated by an exponential power-delay profile, one delay spread or a per-link draw by LOS state (NR engine) | `fading_freq_corr`, `fading_delay_spread_ns`, `fading_ds_from_los`, `fading_ds_grid`, `inf_hall_volume_m3`, `inf_hall_surface_m2`, `inf_lg_ds` |
| add-on gNB antenna | TR 38.901 sector element per cell, added to every link | `gnb_antenna="sector"`, `cell_azimuth_deg`, `cell_tilt_deg`, `gnb_antenna_gain_dbi` |

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

**Simplifications.** UMa uses `h_E = 1 m`, which is exact for robots below 13 m. Distances below the applicability ranges (10 m outdoors, 1 m indoors) extrapolate the formulas, with `d_2D >= 1 m`. The optional single-slope NLOS formulas, InH open office, InF-HH and the < 6 GHz backwards-compatible O2I model (Table 7.4.3-3) are not implemented. Heights are constants (`gnb_height_m`, `ue_height_m`): the z of 3-D poses is ignored, as in the other models. By default the LOS state changes only the path loss and the shadowing, and fast fading stays Rayleigh. With `fading_rician=True` it also sets the Rician K-factor of the link (see [Rician fading](#rician-fading)).

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
## Antenna patterns

With `gnb_antenna="sector"` every gNB has one TR 38.901 Table 7.3-1 element, and its gain toward each robot is added to the link in `RadioMC.rx_dbm`, after the channel model and blockage. Association, handover, interference, the SINR and the open-loop power control all see it (`channels/antenna.py`). The pattern is

- vertical cut `A_V = -min(12 ((theta - 90 - tilt) / 65)^2, 30)` dB, with `theta` the zenith angle of the link (90 degrees is the horizon) and `tilt` the downtilt `cell_tilt_deg`,
- horizontal cut `A_H = -min(12 (phi / 65)^2, 30)` dB, with `phi` the link azimuth minus the cell's boresight `cell_azimuth_deg`,
- combined `A = -min(-(A_V + A_H), 30)` dB, and gain `gnb_antenna_gain_dbi + A` (8 dBi by default).

The half-power beamwidth is 65 degrees in both planes: the gain is 3 dB below the maximum at ±32.5 degrees and reaches the 30 dB floor at about ±103 degrees. The default boresights, 30, 150 and 270 degrees cycled over the cells, are the 3-sector site convention. Three cells at one position (`cell_layout="custom"`) form a 3-sector site whose sectors meet at 90, 210 and 330 degrees with −2.2 dBi each. With the cells of the hex preset, which sit at different sites, set `cell_azimuth_deg` to match the deployment. The elevation uses the gNB height of the channel (the scenario's for `tr38901`, `gnb_height_m` otherwise, robot height when unset) and `ue_height_m`. Downtilt is applied in the separable form `theta - 90 - tilt`, which is exact for links in the boresight's vertical plane and the usual system-level simplification of the 38.901 Sec. 7.1 rotation. The UE antenna stays isotropic, and there is no array gain or beamforming. With `channel="radio_map"`, `gnb_antenna="sector"` adds the pattern on top of a map baked with isotropic antennas. Alternatively the bake applies the pattern itself (`bake.py --gnb-antenna sector --cell-azimuth ...`, [scene-radio-map.md](scene-radio-map.md#sector-antennas-at-bake-time)) and records it in the map's metadata, and `RadioMC` then refuses `gnb_antenna="sector"` with a `ValueError` so the pattern is never applied twice; such a map runs with the default `gnb_antenna="isotropic"`.

The default `gnb_antenna="isotropic"` adds nothing, so every model stays bitwise unchanged. The gain is one fixed-shape `[E, R, C]` op on the poses, so it is captured in CUDA graphs like the rest of the radio.

## Per-robot Doppler

The NR engine's fading is `h <- rho h + sqrt(1 - rho^2) n` per elapsed interval, with one `rho` per ms for all robots (`fading_rho_per_ms`, or `ue_speed_mps` through `fading_rho_from_speed`). With `fading_doppler="per_robot"` each robot gets `rho_ms = J0(2 pi f_D 2.5 ms)^(1 / 2.5)` from its own speed, with `f_D = v f_c / c`, the same rule applied per robot. The speed is `|vel|` when `step(..., vel=...)` passes velocities, else the pose difference over one control step. Right after a reset, a robot has no previous pose and uses `doppler_min_speed_mps` (0 by default: a still robot's fading is frozen). Per-robot Doppler needs pose input, because SNR input carries no motion. Speeds apply from the step in which they are measured.

`NREngine` installs this by replacing `_evolve` on its own `NRNet` instance (`channels/doppler.install_per_robot_fading`). The replacement makes the same `randn_like` calls in the same order and differs only in using a per-robot `rho` tensor, so with every robot at the global speed it reproduces the global model up to float rounding (`test_per_robot_fading_equals_global_at_equal_speed`). If the NR engine's fading code changes, this hook must follow it. The natural long-term home is an optional `rho` tensor in `NRNet._evolve` itself.

## Rician fading

**Model.** With `fading_rician=True` the NR engine's fading gain of every subband becomes

`|x|^2 = |sqrt(K / (K + 1)) e^{j phi} + sqrt(1 / (K + 1)) h|^2`,

where `h` is the unchanged AR(1) Rayleigh state (`E|h|^2 = 1`, same draws, same Doppler, per-robot Doppler included), `phi` is a fixed phase per link drawn uniformly at reset, and `K` is the linear K-factor of the link (`[E, R]` with one cell, `[E, R, C]` with several). The mean power stays 1, so the link budget is unchanged and only the fade distribution changes. A LOS link at K = 7 dB has 1% of its subband-slots below −9.8 dB, against −20 dB for Rayleigh, which removes most of the deep fades that cause HARQ retransmissions on good links. K = 0 gives exactly the Rayleigh engine, fade for fade. The specular phase is constant, so it carries no Doppler shift of the LOS path. The specular term is the same phasor on every subband, so it is fully correlated in frequency, and the diffuse part `h` fades independently per subband unless `fading_freq_corr` is on (see [Frequency-selective fading](#frequency-selective-fading)).

**Where K comes from.**

| Fields | K of a link |
|:---|:---|
| `rician_k_db=x` | `10^(x / 10)` for every link, LOS or not |
| `rician_k_db=None`, `rician_k_from_los=True` (default), radio with a LOS state | a log-normal draw `10^(N(mu_K, sigma_K) / 10)` per link, fixed for the episode and redrawn at reset, where the radio reports LOS and no blockage; 0 where the link is NLOS or blocked |
| `rician_k_db=None` and no LOS state (SNR input, or a radio without `los_state()`), or `rician_k_from_los=False` | 0, which is plain Rayleigh |

The LOS state comes from `RadioMC.los_state()` and `RadioMC.blocked_state()` (bool `[E, R, C]`), which `NREngine` reads after every path-gain call with pose input and passes to `NRNet.set_los`. A radio without these methods leaves K at 0. `mu_K` and `sigma_K` are those of `tr38901_scenario`, whatever the channel model, so with `channel="radio_map"` and a LOS state the default InF-SH values apply.

| Scenario | mu_K (dB) | sigma_K (dB) |
|:---|:---|:---|
| UMi (street canyon) | 9 | 5 |
| UMa | 9 | 3.5 |
| RMa | 7 | 4 |
| InH (office) | 7 | 4 |
| InF (SL, DL, SH, DH) | 7 | 8 |

Source: TR 38.901 V17.0.0, Table 7.5-6 Parts 1–3 (K-factor, LOS only; N/A for NLOS and O2I). The values were checked against the ETSI TR 138 901 V17.0.0 text on 2026-10-05 and are in `nr_engine.RICIAN_K_DB`. With `sigma_K = 8 dB`, InF links range from almost Rayleigh to almost no fading.

**Ramp.** When a link's target K changes, for example on a LOS to NLOS flip, K moves linearly from its current value to the new target over `rician_k_ramp_slots` slots (4 by default, 2 ms at μ = 1), starting at the first slot of the control step that delivered the new LOS state. A flip back in the middle of a ramp starts a new ramp from the current K. `rician_k_ramp_slots=0` switches K at once. The ramp avoids a one-slot SINR cliff on top of the path-loss step. The first LOS state after a reset applies at once, without a ramp. K in slot `g` is a pure function of `g` and the ramp state (`NRNet._k_at`), so the result does not depend on which slots the schedule evaluates.

**Randomness and backends.** Under `rng="engine"` the phase and the K draw are reset draws of the engine's counter RNG (sites `KPHI` and `KFAC` of `nr_rng.py`), keyed by (seed, env, episode). An env's Rician state therefore does not depend on `E` or on other envs' resets. Under `rng="global"` they come from the engine's generator after the initial fading state, so with Rician on they shift the generator draws that follow. The `graph` backend captures the K state as persistent buffers, and `set_los` runs eagerly before the replay, as the per-robot Doppler input does. The fused `triton` kernel takes the K target, ramp start, ramp slot and phasor as inputs (`k_ptr`, `kf_ptr`, `kg_ptr`, `phi_ptr`, constexpr `RICIAN`) and evaluates the same ramp and gain per slot.

**Validation.** `tests/test_rician.py` checks the following. With `fading_rician=False` the engine is bitwise the pre-feature fading code. K = 0 reproduces the Rayleigh engine bitwise. For K ∈ {0, 3, 7, 10} dB the empirical CDF of `|x|^2` over about 400,000 link-slots matches the Rician power CDF (`2 (K + 1) |x|^2` is noncentral χ² with 2 degrees of freedom and noncentrality `2K`) with a Kolmogorov–Smirnov distance below 0.002, against 0.10–0.29 for a Rayleigh CDF, and the mean power is 1 within 0.3%. The ramp, a flip back mid-ramp, the first state after a reset, the K draw statistics (`mu_K`, `sigma_K`), E-independence and partial-reset isolation are also tested. The GPU equivalence lists (`tests/test_nr_fast.py` G1, G2, G7) include the Rician configs of `tests/nr_equiv.py`. 5G-LENA's fading arm cannot serve as a system-level reference here, because stock 5G-LENA's uplink AMC fails under frequency-selective fading ([validation-5g-lena.md](validation-5g-lena.md)). The model is therefore validated at link level against the Rician distribution and the TR 38.901 K table. A system-level check would need OAI rfsim with a Rician TDL channel ([bridges-oai.md](bridges-oai.md)) or a patched 5G-LENA with OLLA.

## Frequency-selective fading

**Why.** Without this add-on the AR(1) innovation of every subband (RBG) is drawn independently, so the coherence bandwidth is implicitly one RBG. Indoor factories have rms delay spreads of tens of ns, which means coherence bandwidths of a few MHz, so neighbouring RBGs of 1.4–1.8 MHz fade together. The subband PF metric (`pf_metric="subband"`) and per-subband CQI gain from frequency diversity only as far as the subbands actually differ, so with independent subbands they overstate that gain.

**Model.** With `fading_freq_corr=True` the innovation `z [E, R, (C), S, 2]` of the AR(1) recursion becomes `L z`, where `L` is the lower-triangular Cholesky factor of an `[S, S]` correlation matrix `Cm`. The same `L` multiplies the real and the imaginary part. The initial state drawn at reset gets the same treatment, so the process is stationary from the first slot. The recursion stays

`h <- rho h + sqrt(1 - rho^2) L z / sqrt(2)`,

so the temporal correlation `rho^k` of every subband, the Doppler (global or per robot) and the Rician term are unchanged.

**Subband covariance.** An exponential power-delay profile `P(t) = exp(-t / tau) / tau` (`t >= 0`, rms delay spread `tau`) gives the complex frequency correlation

`rho(df) = E[H(f + df) H*(f)] = 1 / (1 + j 2 pi df tau)`.

The exact circular model would give the complex innovation the covariance `R_ij = rho(f_i - f_j)`, which couples the real and imaginary parts through `Im R`. The engine instead keeps the two parts independent with one real correlation matrix, and it uses the magnitude-consistent choice

`Cm_ij = |rho(f_i - f_j)| = 1 / sqrt(1 + (2 pi (f_i - f_j) tau)^2)`.

This choice has three properties, and the docstring of `nr_engine.subband_corr` gives the derivation:

- `Cm_ii = 1`, so every subband keeps unit power (`E|h_s|^2 = 1`) and the AR(1) recursion stays stationary.
- For circular Gaussian subbands the power correlation is `|E[h_i h_j*]|^2`. Here it is `Cm_ij^2 = 1 / (1 + (2 pi df tau)^2)`, the same as in the exact complex model. The Rayleigh gain uses only `|h|^2`, so it has the exact model's statistics up to a phase rotation `arg rho` that no output depends on.
- `Cm` is positive definite, because `1 / sqrt(1 + (2 pi tau f)^2)` is the Fourier transform of a positive function (a modified Bessel function `K_0`), so Bochner's theorem applies.

With Rician fading the specular-diffuse cross term correlates as `Cm_ij / 2` instead of the exact `Re rho / 2`, so it is slightly more correlated than in the exact model.

The subband frequencies `f_s` are the RBG centres of the carrier: PRBs per RBG × 12 × SCS, with the last RBG possibly narrower (`NRConfig.subband_prbs`). Each subband is one fading value sampled at its centre. At the default 20 MHz, μ = 1 carrier (13 RBGs of 1.44 MHz), adjacent subbands correlate at 0.996, 0.91, 0.74 and 0.35 for `tau` = 10, 50, 100 and 300 ns. At 3 µs they are almost independent (0.037). `corr_sqrt` computes `L` in float64 and adds a small diagonal jitter for nearly singular matrices (`tau` far below 1 / bandwidth, where all subbands are almost identical). It then renormalizes the rows, so `L L^T` keeps an exact unit diagonal.

**Where the delay spread comes from.**

| Fields | Delay spread of a link |
|:---|:---|
| `fading_delay_spread_ns=x` | `x` ns for every link: one constant `L` built at construction, one `[S, S]` matmul per slot |
| `fading_delay_spread_ns=None`, `fading_ds_from_los=True` (default) | a log-normal draw `10^(N(mu_lgDS, sigma_lgDS))` s per link and per LOS state, fixed for the episode and redrawn at reset. The radio's LOS state selects the LOS or the NLOS draw. A blocked LOS link counts as NLOS, and a link without a LOS state (SNR input, or a radio without `los_state()`) uses its NLOS draw |

`fading_freq_corr=True` with `fading_delay_spread_ns=None` and `fading_ds_from_los=False` is refused. The LOS state reaches the engine through `NRNet.set_los`, which `NREngine` calls after every path-gain call when Rician K or the delay spread follow the LOS state. A new state applies from the first slot of the step without a ramp, because the AR(1) state carries the change over the coherence time anyway.

`mu_lgDS` and `sigma_lgDS` (`lgDS = log10(DS / 1 s)`, `fc` in GHz) are those of `tr38901_scenario` at `carrier_ghz`, whatever the channel model:

| Scenario | LOS mu / sigma | NLOS mu / sigma | Median DS at 3.5 GHz, LOS / NLOS |
|:---|:---|:---|:---|
| UMi (street canyon) | −0.24 lg(1 + fc) − 7.14 / 0.38 | −0.24 lg(1 + fc) − 6.83 / 0.16 lg(1 + fc) + 0.28 | 50 / 103 ns |
| UMa | −6.955 − 0.0963 lg(fc) / 0.66 | −6.28 − 0.204 lg(fc) / 0.39 | 93 / 364 ns |
| RMa | −7.49 / 0.55 | −7.43 / 0.48 | 32 / 37 ns |
| InH (office) | −0.01 lg(1 + fc) − 7.692 / 0.18 | −0.28 lg(1 + fc) − 7.173 / 0.10 lg(1 + fc) + 0.055 | 20 / 39 ns |
| InF (SL, DL, SH, DH) | lg(26 (V/S) + 14) − 9.35 / 0.15 | lg(30 (V/S) + 32) − 9.44 / 0.19 | SL, DH: 53 / 55 ns; DL, SH: 59 / 61 ns |

Frequency floors: UMa and InH use `fc = 6` below 6 GHz, UMi uses `fc = 2` below 2 GHz. For InF, `V` is the hall volume and `S` its total surface (walls, floor and ceiling). The default hall is the Table 7.8-7 hall of the scenario as Sionna implements it: 120 × 60 × 10 m for SL and DH (V/S = 4.0) and 300 × 150 × 10 m for DL and SH (V/S = 4.55). `inf_hall_volume_m3` and `inf_hall_surface_m2` set your own hall, and `inf_lg_ds` sets the mean lgDS directly for both states.

Sources: TR 38.901 V17.0.0, Table 7.5-6 Parts 1–3 (DS rows and notes 6–7; note 4 of Part 3 for V and S). The values were checked on 2026-10-05 against Sionna's V16.1 parameter files (`src/sionna/phy/channel/tr38901/models/v16_1/*.json`, whose DS rows are those of V17.0.0) and, for RMa, InH and InF, against the itecspec.com mirror of clause 7.5 (Release 19, rows unchanged). Release 19 updated the UMi and UMa rows for its 7–24 GHz study (note 8): UMi LOS −0.18 lg(1 + fc) − 7.28 / 0.39 and NLOS −0.22 lg(1 + fc) − 6.87 / 0.19 lg(1 + fc) + 0.22, UMa LOS −7.067 − 0.0794 lg(fc) / 0.57 + 0.026 lg(fc) and NLOS −6.47 − 0.134 lg(fc) / 0.39. The engine keeps the V17.0.0 values, like the rest of the package. The InF default hall dimensions were checked against Sionna's `inf_scenario.py`, not against the spec text. The values are in `nr_engine.LG_DS` and `lg_ds_params`.

**Grid approximation.** A per-link Cholesky inside the step would break the fixed-shape, graph-capturable step. The engine therefore precomputes `L` for `G` log-spaced delay spreads (`fading_ds_grid = (3, 3000, 16)`: 3 ns to 3 µs, ratio 1.58 between points), keeps each link's nearest grid index for its LOS and NLOS draw (`fc_idx_los`, `fc_idx_nlos`, computed at reset), and selects the live index `fc_idx [E, R, (C)]` in `set_los`. The step gathers `fc_tab[fc_idx]` and applies a batched `[S, S]` matmul. Draws outside the grid are clamped to its ends, where the subbands are already almost fully correlated (3 ns) or almost independent (3 µs) at FR1 RBG widths. Rounding to the grid widens the lgDS spread by `0.2^2 / 12` in variance (0.058 dex rms), small against `sigma_lgDS` = 0.15–0.66.

**Randomness and backends.** Under `rng="engine"` the two normals per link are reset draws of the engine's counter RNG (site `DSPR` of `nr_rng.py`), keyed by (seed, env, episode), so an env's delay spreads depend neither on `E` nor on other envs' resets. Under `rng="global"` they come from the engine's generator right after the initial fading state, so with the per-link mode on they shift the generator draws that follow (the Rician draws among them). The `graph` backend keeps `fc_L` or `fc_tab`, `fc_idx`, `fc_idx_los` and `fc_idx_nlos` as persistent buffers, and `set_los` runs eagerly before the replay. The fused `triton` kernel takes the `L` table (`fcl_ptr`, `[G, S, S]` row-major) and the per-robot grid index (`fci_ptr`, `[E, R]`) as inputs. Its constexpr `FCORR` is 0 (off: the earlier code path), 1 (one shared `L`) or 2 (per-robot index). It correlates the innovation per robot with an `S × S` register product before the AR(1) update. Its normals equal the torch path to float rounding, and so does the product.

**Validation.** `tests/test_freqfade.py` checks the following. With `fading_freq_corr=False` the engine is bitwise the pre-feature fading code (one and three cells, global RNG, partial reset). For `tau` ∈ {10, 50, 100, 300} ns the empirical cross-subband correlation of `h` over about 40,000 link-slots matches `Cm` within 0.01, and the subband power correlation matches `Cm^2` within 0.02. The initial state has the same correlation. At 1 ns all subbands correlate above 0.99, and at 3 µs the off-diagonal entries stay below 0.05. The per-subband power is 1 within 1%, and the AR(1) correlation over 4 slots equals `rho^2` per subband. The subband PF gain over wideband PF drops monotonically with the correlation: on 8 × 6 saturated robots it is 2.13, 2.06, 1.25 and 1.03 for `tau` = 3 µs, 100 ns, 10 ns and 1 ns. The tests also cover the delay-spread draw statistics, the Table 7.5-6 values, the LOS / NLOS switch, E-independence and partial-reset isolation, and the config gating. The GPU lists of `tests/test_nr_fast.py` (G1, G2, and G7 for `ul_fcorr`) include the `ul_fcorr` and `ul_fcorr_rician` configs of `tests/nr_equiv.py`. The system-level check still to do is the PF subband gain against 5G-LENA with `McsCsiSource=AVG_MCS` and a TDL channel of the same delay spread. Because stock 5G-LENA's uplink AMC fails under frequency-selective fading ([validation-5g-lena.md](validation-5g-lena.md)), that comparison is best done in the downlink or with a patched AMC.

**Limits.** The model reproduces the second-order frequency correlation of an exponential power-delay profile. It has no discrete taps, no angles and no per-tap Doppler, so it is not a TDL or CDL channel. All taps share the AR(1) time correlation. The power-delay profile is exponential only (`fading_pdp`), and the delay spread is fixed per link and episode, with no spatial consistency in the delay spread.

## Cost

`RadioMC.pathgain_db` at `E = 128`, `R = 32` and three hex cells, eager, median of 50 calls on the shared RTX 4090 (82% busy with other jobs): `log_distance` 0.62 ms, with the exponential field and a white component 0.94 ms, `tr38901` InF-SH 1.95 ms, UMi with O2I 5.8 ms, `radio_map` (two cells) 1.3 ms, and `blockage` adds about 1.0 ms. The radio runs once per control step. The reference NR engine's step at the same size took 1,337 ms with the default channel and 1,361 ms with InF-SH, blockage and per-robot Doppler, so the channel is below 0.5% of a step. The times are dominated by kernel launches, so they grow slowly with `E`.
