# Obstacles and NLOS

The obstacle stack decides whether each robot–gNB link has line of sight, how much an obstacle edge costs near the shadow boundary, and how much dynamic blockers (other robots, people, vehicles) cost. It adds three things to `radio.RadioMC`:

- a **geometric LOS state** `[E, R, C]` that can come from the scene instead of the TR 38.901 probability (`NRConfig.los_source`),
- a **knife-edge diffraction** loss that turns the LOS → NLOS step at an aisle end into a ramp (`los_diffraction`),
- two **TR 38.901 blockage models**: geometric screens (model B) for dynamic blockers and stochastic angular regions (model A) for scenes without geometry (`blockage_model`).

Every new field defaults to today's behaviour. Outputs at the defaults are bitwise unchanged (see [Validation](#validation)). Code: `core/channels/los.py`, `core/channels/blockage.py`, `core/channels/models.py` (TR 38.901 hooks), `core/radio.py` (`RadioMC`), `tools/scene/heightmap.py`, `tools/make_synthetic_radio_map.py`.

```python
from isaac_net import NRConfig, make_engine

# geometry-driven LOS from a baked map (los_prob and obstacle_z, see "Inputs"), TR 38.901 InF path loss,
# knife-edge ramp, people as screens
cfg = NRConfig(channel="tr38901_inf_sh", radio_map_path="hall.npz", los_source="raycast", los_diffraction=True,
               blockage=True, blockage_model="screen")
eng = make_engine("L2", E, R, "cuda", cfg)
out = eng.step(None, poses, blockers=people)   # people [E, M, 3] = (x, y, class), class 1 = human, -1 = empty slot
out["los"], out["blocked"]                     # [E, R] bool, serving link
eng.radio.los_state(), eng.radio.blocked_state()   # [E, R, C] bool, every link
```

## Fields

| Field | Default | Read when | Meaning |
|:---|:---|:---|:---|
| `los_source` | `"stochastic"` | always (radio group) | where the LOS state comes from: `"stochastic"` (TR 38.901 probability, today), `"map"` (baked `los_prob`), `"raycast"` (2.5-D ray march over `obstacle_z`), `"callback"` (a `blocked_fn`) |
| `los_raycast_samples` | 32 | `los_source="raycast"` | samples per robot–gNB segment |
| `los_diffraction` | False | `los_source="raycast"` | ITU-R P.526 knife-edge loss from the ray march |
| `los_soft` | False | `channel="tr38901"`, `los_source="stochastic"` | TR 38.901 §7.6.3.3 soft LOS blend |
| `nlos_extra_loss_db` | 0 | `channel="log_distance"` with a geometric source | extra loss on NLOS links |
| `blockage_model` | `"sphere"` | `blockage=True` | `"sphere"` (today), `"screen"` (model B), `"stochastic"` (model A) |
| `blocker_size_m` | ((0.6, 1.5), (0.3, 1.7), (4.8, 1.4)) | `blockage_model="screen"` | (w, h) in metres of blocker classes 0 (robot), 1 (human), 2 (vehicle) |
| `blockage_max_db` | 40 | `screen` or `stochastic` | cap on the summed blockage loss of a link |

`radio_map_path` is read for `channel="radio_map"` and for `los_source` `"map"` or `"raycast"`, so a TR 38.901 or log-distance channel can take its LOS state from a map. `tr38901_los` is not read with a geometric source. `blockage_radius_m` and `blockage_loss_db` are read only by the sphere model. `NRConfig.unused_fields(level)` lists every field the switches leave unread, and `make_engine(..., strict=True)` refuses them. The config also refuses combinations that would be silently ignored: `los_diffraction` without `"raycast"`, and `los_soft` outside the stochastic TR 38.901 path.

## LOS state

`channels.los.LosState` owns `los [E, R, C]`, the state of the previous call (`prev`), and a per-link transition counter (`n_trans`; a call right after an env's reset counts no flip). `RadioMC` builds it only when `los_source != "stochastic"`.

**`"map"`.** The bake writes `los_prob [C, H, W]`, the share of k × k points per grid cell with a clear straight segment to the gNB antenna (`tools/scene/bake.py --los-map`). The state is `u(x) < los_prob(x)`, with `los_prob` sampled bilinearly and `u(x)` the spatially consistent uniform field of the stochastic model (a plane-wave field through the exact CDF of its marginal, `fields.uniform_from_field`). With `channel="tr38901"` it reuses the channel's own LOS-state field. Otherwise it draws one field per (env, cell), with a correlation distance equal to the map's grid spacing, from the engine RNG (streams `RADIO_STREAM + 32..34`), so the state is keyed by (seed, env id, episode) like every other radio draw. In a half-shadowed cell, half of the robots are LOS, and a robot that stands still keeps its state. Cost: one gather per link.

**`"raycast"`.** The bake writes `obstacle_z [H, W]` (`--obstacle-z`): the largest z of any triangle over each grid cell, clamped at 0 (the floor). Robots are excluded by the export's `--exclude`, as for the radio map, and a ceiling is dropped with `--z-max`. The ray march takes N samples at `t = (n + 0.5) / N` on the segment from the robot antenna `(x, y, ue_height_m)` to the gNB antenna `(x_c, y_c, h_bs)`, reads the height bilinearly, and sets `blocked = any(z_obstacle > z_ray)`. It also returns the minimum clearance in metres (negative when blocked). The gNB height is `gnb_height_m`, the scenario height for TR 38.901, or the map's `gnb_z` for the 2-D models. The march is deterministic, fixed-shape (`[E, R, C, N]`), has no data-dependent branching and makes no host sync. The sample spacing must stay below the thinnest obstacle: 64 samples on a 36 m link (0.56 m) resolve 1 m deep racks exactly, while 32 samples (1.1 m) step over a few of them (`test_raycast_equals_baked_los_prob`).

**`"callback"`.** `RadioMC.set_los_callback(fn)`, `NREngine.set_los_callback(fn)`, or `NetModule.step(..., blocked_fn=fn)` with `radio="engine"`: `fn(poses)` returns `[E, R, C]` bool, True = blocked, and `los = ~fn(poses)`. The poses are the ones passed to the radio, in the radio frame. This is the signature of the Isaac layer's `blocked_fn`, so the same function, for example `isaac.radio.mesh_blocked_fn(mesh, gnb, to_world)` around the Warp kernel `los_blocked_kernel`, now serves multi-cell `radio="engine"` configs too. It unifies the two blockage paths that existed side by side. The Isaac radio keeps its own `blocked_fn` path unchanged.

### What the state drives in each channel

| `channel` | LOS state selects | Diffraction (`los_diffraction`) |
|:---|:---|:---|
| `tr38901` | `pl_los` / `pl_nlos` and the LOS / NLOS shadow field, exactly as in `TR38901Channel.pathgain_db`; only the source of `self.los` changes | NLOS excess `min(J(v), PL_NLOS − PL_LOS)`, and the shadow field is blended with the same weight |
| `radio_map` | **no path loss** (see below); only the step outputs and the hooks (K-factor) | only the lit-side Fresnel loss `J(min(v, 0))`, 0 to 6 dB |
| `log_distance` | `nlos_extra_loss_db` on NLOS links | excess `min(J(v), nlos_extra_loss_db)` |

**Why the state adds no path loss on a radio map.** A ray-traced map already holds the mean NLOS loss: the bake's `gain_db` is the power that reaches each grid point through every path the solver found, and in a rack's shadow that power is already low. Adding an NLOS term on top would count the obstacle twice. The state therefore only drives what the map cannot hold: the per-link state for the fading (K-factor) and the outputs. With diffraction on, the map gets only the lit-side Fresnel loss (0 to 6 dB), which a bake without `--diffraction` lacks. The shadow side of the ramp is already in the map. A map baked with `--diffraction` (metadata `diffraction=True`) is refused together with `los_diffraction`, because the lit-side loss would then also be counted twice.

## Knife-edge diffraction

`knife_edge_db(v)` is the ITU-R P.526 single-edge approximation `J(v) = 6.9 + 20 log10(sqrt((v − 0.1)^2 + 1) + v − 0.1)` for `v > −0.78`, else 0. It gives 6.03 dB at grazing (`v = 0`). The Fresnel parameter is `v = h sqrt(2 (d1 + d2) / (λ d1 d2))`, with `h` the distance by which the obstacle reaches into the ray (negative for a clear ray).

The ray march computes `v` from two kinds of edges:

- **Obstacle tops.** Per sample, `h` is the ray's height below the obstacle top. The floor (`z ≤ 0.01 m`) is not an edge, because a ground knife-edge would be wrong next to a ground reflection.
- **Vertical edges**, for example the end of a rack at an aisle end. The whole ray is shifted sideways in parallel along a fixed ladder of offsets (0.25 to 5 mid-link Fresnel radii `sqrt(λ d) / 2`). The shift at which the ray changes state, interpolated linearly on the ray's largest gap, is the distance `h` to the edge. `d1` and `d2` are taken at the sample that controls it.

A clear ray takes the nearest edge (largest `v ≤ 0`). A blocked ray takes the easiest way out, over the top or around the side (smallest `v > 0`). An edge farther sideways than the ladder's end is taken at that distance, which bounds `J` above 25 dB and makes the loss saturate there instead of jumping. In the aisle-end test, a robot slides past the end of a 6 m rack at 3.5 GHz: the loss rises from 0 through 4.8 dB at the boundary to 20 dB one metre into the shadow, and then reaches the NLOS level, with no step larger than 2 dB per 5 cm (`test_diffraction_ramp_at_aisle_end`). Diffraction costs about 12 times the plain ray march, because it adds 2 × 7 height lookups per sample (see Cost).

## Blockage model B: geometric screens (`blockage_model="screen"`)

Every blocker is a vertical rectangular screen of width `w` and height `h`. It stands on the floor at its `(x, y)` and is turned to face the horizontal direction of the link. TR 38.901 §7.6.4.2 (eq. 7.6-29 and 7.6-30) gives the loss per screen:

```
L = −20 log10(1 − (F_h1 + F_h2)(F_w1 + F_w2)),
F_k = atan(± (π/2) sqrt((π/λ)(D1_k + D2_k − r))) / π
```

Here `D1_k` and `D2_k` are the distances from the two antennas to edge k (projected onto the screen at the height or lateral position of the direct path), and `r` is the direct distance. The sign is + when the direct path lies on the screen side of edge k and − otherwise. The edges are the floor, the top and the two sides. Losses of several screens add in dB and are clamped at `blockage_max_db`. A link counts as blocked (`blocked_state()`) when the direct path crosses a screen rectangle. Other robots are class-0 screens, and a robot never blocks itself. Extra blockers come per step as `blockers [E, M, 3]` rows `(x, y, class)`, where the class indexes `blocker_size_m` and a negative class is an empty slot (fixed shape). They are passed through `NREngine.step(..., blockers=)`, the graph backend, or `NetModule.step(..., blockers=)` with `radio="engine"` at level L2. The `BackgroundConfig` UE positions or Isaac humans and forklifts are natural sources. The intermediate shape is `[E, R, C, R + M]`.

**What the formula gives at 3.5 GHz.** A 0.3 × 1.7 m person 1 m from the robot on a 20 m link at 1.5 m height costs 4.3 dB (hand value in `test_screen_loss_human_and_vehicle`). The loss falls to 0.1 dB when the person stands 1 m to the side. A 0.3 m body is about one first-Fresnel-zone radius wide at that distance (0.29 m), so the 10 to 25 dB body loss known from mmWave is not reached at 3.5 GHz. The same geometry gives 11 dB at 28 GHz. A 20 × 5 m screen grazed at one side gives 5.1 dB, and the 6 dB half-plane value is reduced by the floor and top edges. Behind its centre the loss is about 18 dB, because the floor and the top edge still let energy through. This is how the model behaves at sub-6 GHz, not a defect.

## Blockage model A: stochastic regions (`blockage_model="stochastic"`)

TR 38.901 §7.6.4.1 is meant for scenes without geometry. Each robot has K = 4 non-self-blocking angular regions. Region k has a centre azimuth φ_k, a centre zenith of 90°, an azimuth span x_k, an elevation span y_k and a distance r, from Table 7.6.4.1-2. The correlation distance is the one of Table 7.6.4.1-4 (UMi, UMa and RMa 10 m for LOS and NLOS, InH 5 m; the 5 m O2I value is not used). All values were checked against ETSI TR 138 901 V17.0.0 (2022-04):

| Family | x_k (deg) | y_k (deg) | r (m) | correlation distance |
|:---|:---|:---|:---|:---|
| indoor (InH; also used for InF and the non-38.901 channels) | U[15, 45] | U[5, 15] | 2 | 5 m |
| outdoor (UMi, UMa, RMa) | U[5, 15] | 5 | 10 | 10 m |

The spans are drawn at reset per (env, region). The centres are spatially and temporally consistent: `φ_k(x, t) = 360 u_k(x + v_b t)`, where `u_k` is a uniform field with the family's correlation distance and `v_b` is a per-env drift of 3 km/h in a random direction. Eq. 7.6-28 sets `t_corr = d_corr / v` with v the blocker speed but gives no value for model A, so we take the human speed of Table 7.6.4.2-5 (up to 3 km/h). ns-3's `BlockerSpeed` defaults to 1 m/s instead. A still robot therefore sees its regions move with a correlation time of about `d_corr / v_b`, which is 6 s indoors. The attenuation of eq. 7.6-22, `L = −20 log10(1 − (F_A1 + F_A2)(F_Z1 + F_Z2))` with `F = atan(± (π/2) sqrt((π/λ) r (1/cos(Δ) − 1))) / π`, is evaluated at the azimuth and zenith of each link's direct path. As the spec states below eq. 7.6-22, a region attenuates only when `|φ_AOA − φ_k| < x_k` and `|θ_ZOA − θ_k| < y_k`, and the signs follow Table 7.6.4.1-3. isaac_net has no clusters, so the loss of the LOS cluster is applied to the whole link: this is the approximation. Losses are summed over the regions and clamped at `blockage_max_db`. The env clocks advance by `control_step_ms` on every `rx_dbm` call, and the engines make one such call per control step. Self-blocking is off, because it is a handset concept (a hand or head next to the antenna). The randomness comes from engine RNG streams `RADIO_STREAM + 48..53`, keyed by (seed, env id, episode).

At 3.5 GHz a 30 × 10° region at 2 m costs about 3.5 dB, and the same region costs about 10 dB at 28 GHz. Indoors, with horizontal links and still robots, a link sits inside a region 29% of the time (400 envs × 30 s). Inside, the loss averages 3.6 dB, the mean over all time is 1.3 dB, and episodes last 1.2 s on average (`test_model_a_statistics`). Episodes are shorter than `d_corr / v_b` because a 30° region needs to move only part of the way to uncover the link. Like model B, model A was calibrated for mmWave, and its effect at sub-6 GHz is mild.

## Soft LOS (`los_soft`)

For the stochastic TR 38.901 state, §7.6.3.3 blends LOS and NLOS:

```
LOS_soft = 1/2 + atan(sqrt(20 / λ)(F − G)) / π,  F = sqrt(2) erfinv(2 Pr_LOS − 1),  G = sqrt(2) erfinv(2u − 1)
```

Here `u` is the uniform of the LOS-state field, so `G` is its Gaussian with the LOS-state correlation distance of Table 7.6.3.1-2. Path loss and shadow field are mixed in dB with weight `LOS_soft`. The spec mixes the channel matrices instead, as `H_LOS LOS_soft + H_NLOS sqrt(1 − LOS_soft²)` (eq. 7.6-19), and the dB blend is our large-scale approximation of it. The boolean state `LOS_soft > 1/2` equals the hard state `u < Pr_LOS`, so the outputs and the K-factor hook see the same state. The blend is continuous in `Pr_LOS` and in position, which removes the threshold flicker of a robot that moves along a state boundary. Eq. 7.6-18 of TR 38.901 V17.0.0 writes `LOS_soft = 1/2 + (1/π) arctan(sqrt(20 / λ)(G + F(d)))` with its own zero-mean Gaussian `G` and `F(d) = sqrt(2) erf⁻¹(2 Pr_LOS(d) − 1)`. With `G → −G`, which has the same law, it is this form. The scale `sqrt(20 / λ)` was checked against the ETSI copy, with λ in metres (the clause names no unit).

## Interface for other modules

- `RadioMC.los_state() -> Optional[Tensor]`: `[E, R, C]` bool of the last `rx_dbm` call. It is the geometric state when `los_source != "stochastic"`, the TR 38.901 stochastic state for `channel="tr38901"`, and otherwise `None` (log-distance or radio map without a geometric source, or before the first call).
- `RadioMC.blocked_state() -> Optional[Tensor]`: `[E, R, C]` bool, True when a dynamic blocker is on the direct path (sphere hit, screen crossed, or model-A region containing the path), or `None` without blockage.
- Step dict of the NR engine (reference and graph backends), with poses as input: `los` and `blocked` `[E, R]` for the serving link. They are added only when the obstacle stack is on (`los_source != "stochastic"` or `blockage=True`) and the radio has the corresponding state, so the dict of every existing default config is unchanged. Contract tests check a subset of keys (`tests/test_engine_api.py`), so extra keys are allowed. `blocked` therefore also appears for an existing `blockage=True` config, as additive information.
- Isaac layer: the `NetModule.step` dict always has `los` `[E, R]`. It is the engine's state with `radio="engine"`, and `~blocked` with the Isaac radio. The observation feature `"los"` (`IsaacNetCfg.obs_features`) exposes it next to `"blocked"`. With `radio="engine"`, `blocked` is now the engine's dynamic-blockage flag instead of always False.

## Validation

All tests are CPU tests in `tests/test_obstacles.py` and use the synthetic hall from `python -m isaac_net.tools.make_synthetic_radio_map --obstacles hall.npz`: 40 × 24 m, four rows of 1 × 24 × 2.5 m racks, gNBs at 6 m on the short walls. Its `los_prob` comes from exact segment–box tests on 3 × 3 points per cell, the same method that `bake.py --los-map` uses with Mitsuba. Its `obstacle_z` comes from the racks' triangles through `heightmap.height_map`, the same path that `--obstacle-z` uses. No USD or Sionna is needed.

| Check | Result |
|:---|:---|
| (a) raycast vs baked `los_prob` at its 1.0 / 0.0 points | identical at 64 samples, on the 95% of grid points that are all-LOS or all-NLOS (`test_raycast_equals_baked_los_prob`) |
| (b) LOS fraction vs distance in random InF clutter | Boolean model of `d_clutter` squares at area density r, axis-aligned links between clutter-free points: the fitted decay length equals `k_subsce = −d_clutter / ln(1 − r)` within 15% for (r, d) = (0.4, 2 m) and (0.2, 3 m), and the fraction follows `exp(−(d − d_clutter) / k_subsce)` within 0.05. The shift by `d_clutter` comes from conditioning on clear endpoints. TR 38.901's `exp(−d / k_subsce)` is this law along a grid axis. For random orientations a Boolean model of squares decays faster, by the mean caliper width `4 d_clutter / π` |
| (c) knife-edge | `J(0) = 6.03 dB`, `J(v ≤ −0.78) = 0`, monotone, continuous at −0.78 (< 0.02 dB), `J(2.4) ≈ 20.5 dB` |
| (d) screens | human at 1 m on a 20 m link: 4.3 dB at 3.5 GHz, falling with lateral distance; 11 dB at 28 GHz |
| (e) model A | indoor, horizontal links: inside a region 29% of the time (test bound 15–40%), 3.6 dB mean inside at 3.5 GHz (bound 1.5–8; at least 4 dB more at 28 GHz), 1.3 dB mean overall and never a gain, episodes of 1.2 s (bound 0.5–20 s), little loss toward an elevated gNB |
| (f) soft LOS | monotone and continuous in `Pr_LOS`, equals the hard state at `λ → 0`, smaller path-loss jumps than the hard state along a line |
| bitwise defaults | before/after script on L2 with poses: log-distance (± blockage), TR 38.901 InF-SH, UMi with O2I, multicell(3) InF-DL with blockage, radio map (both `rng="engine"` and `"global"`), `RadioMC.pathgain_db` alone, and the Isaac `NetModule` on L2 / L2-legacy with both radios: every tensor is bitwise equal. The only differences are the added keys (`los` in the NetModule dict; `los` / `blocked` in the step dict of `blockage=True` configs) |

Other checks: map-source statistics (LOS share at a 50% cell is within 0.03 over 4000 envs, a still robot keeps its state, E-invariance under the engine RNG), the path loss of each channel given the state, config gating, the callback through a multi-cell engine and through `NetModule`, and the map file round trip (old files without the grids load as before).

**Needs the lab box (GPU, Sionna, USD).** The graph backend with poses and the obstacle stack (the radio is evaluated eagerly before the captured step, so no capture change is expected, but nothing ran on CUDA here). GPU cost. `bake.py --obstacle-z` on the `tools/scene/validate.py` scenes against their `--los-map` output. The knife-edge loss past a wall's end against a Sionna RT bake with `--diffraction` (the design target is within about 2 dB). `mesh_blocked_fn` with Warp.

## Cost

Measured on CPU (Apple laptop, 4 threads, eager) at E = 128, R = 32, C = 2. The ray march with 32 samples takes 65 ms, with diffraction 770 ms, and 40 screens per link take 260 ms. The ray march is about `E R C N` bilinear lookups. Diffraction multiplies that by 15 (2 sides × 7 rungs + centre) and holds an `[E, R, C, 7, N, 2]` tensor per side: at E = 4096, R = 32, C = 3, N = 32 that is about 1.4 GB, so lower `los_raycast_samples` or shard the envs (`ShardedEngine`) for large batches. GPU numbers are not measured yet.

## Limitations

- **2.5-D.** A height map cannot represent overhangs or holes. A robot under a mezzanine or a conveyor sees solid obstacle down to the floor. The Warp mesh callback is exact but runs outside CUDA-graph capture.
- **One map for all envs.** `RadioMap` is shared, so layout randomization per env needs `[E_layouts, H, W]` maps and a per-env index, which is not implemented.
- **Heights are constants.** The engine is 2-D: `ue_height_m` is the antenna height of every robot, and the z of the poses is ignored by the ray march (the callback receives the poses as passed).
- **Lateral diffraction** is resolved on a fixed ladder of parallel shifts with linear interpolation. Edges farther than 5 Fresnel radii sideways are taken at that distance. Several edges in a row (rack, aisle, rack) are not combined (no Deygout or Epstein–Peterson). The loss of the strongest edge is used and capped by the NLOS level.
- **Screens are perpendicular to the link**, not to the blocker–antenna line, and the floor is a diffracting edge, not a reflector. Model A applies the LOS-cluster loss to the whole link. Both models were designed for mmWave and give a few dB at 3.5 GHz.
- **Spec check.** The model B equations 7.6-29 and 7.6-30 with their sign rule, the blocker sizes of Table 7.6.4.2-5, the model A Table 7.6.4.1-2 values, eq. 7.6-22 to 7.6-27, the signs of Table 7.6.4.1-3, the correlation distances of Table 7.6.4.1-4 and the soft-LOS eq. 7.6-18 were checked against ETSI TR 138 901 V17.0.0 (2022-04). The check found one error: model A applied its loss at every angle, while the spec limits it to `|φ_AOA − φ_k| < x_k` and `|θ_ZOA − θ_k| < y_k`. The window is now applied, which removes the small residual loss outside it (`blockage_model="stochastic"` only). The 3 km/h model A blocker speed is our choice, because the spec gives no value. Model A has no InF row, so InF uses the InH row.
- The K-factor (Rician fading driven by `los_state()`) is a separate feature. This stack only provides the state.

## Comparison with ns-3 `BuildingsChannelConditionModel`

ns-3's buildings module, which 5G-LENA users get, decides LOS by intersecting the link segment with `Building` objects. These are static, axis-aligned boxes declared in the simulation script. `BuildingsPropagationLossModel` then adds wall-penetration and floor losses by wall type. Compared with that:

- **Geometry source.** ns-3 needs the obstacles retyped as `Building` boxes. Here the same USD stage that Isaac renders is exported once, and its triangles become `obstacle_z` and `los_prob`. Arbitrary shapes are supported within the 2.5-D limit, and the exact Warp mesh test is available through the callback. ns-3 boxes are exact 3-D but axis-aligned only.
- **What NLOS costs.** ns-3 switches the 3GPP path loss to its NLOS formula (or adds wall losses). Here the TR 38.901 channel does the same, the log-distance channel adds a fixed NLOS loss, and a ray-traced map already holds the loss. Neither ns-3 model has a diffraction ramp at the shadow boundary. ns-3's `ThreeGppChannelModel` implements blockage model A (`Blockage`, `NumNonselfBlocking`, `BlockerSpeed`), with per-cluster angles that isaac_net does not have. Model B (geometric screens) was not found in the ns-3 documentation, which is an absence claim from the docs, not from the source.
- **Dynamic obstacles.** ns-3 buildings are static. Here robots and per-step blockers are screens, re-evaluated every control step for thousands of envs at once.
- **Batching and consistency.** ns-3 evaluates one link at a time on the CPU. Here every link of every env is evaluated in one fixed-shape tensor op, and the stochastic parts are spatially consistent and keyed by (seed, env id, episode).
- **What ns-3 has that this lacks.** Exact 3-D boxes with per-wall materials, multi-floor buildings, O2I from geometry, and per-cluster blockage on top of the full TR 38.901 fast-fading model.

The natural cross-check is the warehouse racks exported both as `Building` boxes for ns-3 and as `obstacle_z` here, with fading off on both sides, comparing the LOS share per aisle and the SINR CDF (design notes, validation item 2). This has not been run yet.
