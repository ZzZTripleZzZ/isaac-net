# Radio maps from USD scenes

The shelves, walls and racks of an Isaac Lab scene can become the network's channel. `isaac_net/tools/scene/` exports the meshes of a USD stage to a Sionna RT scene with ITU-R P.2040 radio materials, bakes a path-gain map with Sionna RT's `RadioMapSolver`, and writes the file that `NRConfig(channel="radio_map")` loads ([channels.md](channels.md#radio_map)). Inside Isaac Lab, one field of `IsaacNetCfg` does all of this when the env is created, and a map that was already baked for the same geometry is loaded from a cache instead.

```
USD stage ──export──> Mitsuba XML + PLY per material ──Sionna RT──> gain_db [C, H, W] + bounds ──> RadioMC
 (pxr)                (usd_export.py)                  (bake.py)     (+ los_prob, valid)          (channel="radio_map")
```

The map is static. Moving robots are not part of it: their bodies are handled at run time by the blockage add-on (`NRConfig(blockage=True)`, [channels.md](channels.md#blockage)).

## Recipe

### From a USD file (command line)

```bash
pip install usd-core sionna-rt            # plus torch (CPU is enough): the package imports it
python -m isaac_net.tools.scene.bake --usd warehouse.usd --tx -5 -8 7 --tx 5 10 7 \
    --fc 3.5 --cell 0.5 --samples 4e6 --depth 5 --los-map --variant llvm --out warehouse_map.pt
```

`--tx X Y Z` is one gNB or access point per cell (antenna position in the map frame, metres). `--variant llvm` runs on the CPU. `--variant cuda` needs OptiX, which native Windows and Linux drivers provide and WSL 2 does not. The main options are listed below.

| Option | Default | Meaning |
|:---|:---|:---|
| `--root PRIM` | `/` | subtree to export |
| `--frame PRIM` | stage world | the prim whose frame is the map frame |
| `--offset DX DY DZ` | 0 0 0 | added after the frame change |
| `--map PATTERN=MATERIAL`, `--map-json FILE` | none | user material rules (see below) |
| `--default-material` | `concrete` | material when no rule matches |
| `--thickness MATERIAL=M` | table below | per-face slab thickness |
| `--include`, `--exclude PATTERN` | none | prim-path filters (fnmatch, case-insensitive; a prim matches when it or an ancestor does) |
| `--crop X0 Y0 X1 Y1`, `--z-max Z` | none | drop triangles entirely outside the box or above the height |
| `--bounds X0 Y0 X1 Y1` | footprint of the geometry | area of the map |
| `--cell`, `--ue-height`, `--fc` | 1.0 m, 1.5 m, 3.5 GHz | grid cell, robot antenna height, carrier |
| `--samples`, `--depth` | 1e6, 4 | rays per transmitter, interactions per path |
| `--no-refraction`, `--diffraction` | refraction on, diffraction off | propagation mechanisms |
| `--los-map`, `--los-sub K` | off, 3 | also write `los_prob` from K x K points per cell |
| `--scene-xml FILE` | | bake an exported (or hand-made) Mitsuba scene instead of `--usd` |
| `--scene-dir DIR`, `--export-only` | | keep the exported scene, or export and stop (needs no Sionna) |

Then use the map:

```python
from isaac_net import NRConfig, make_engine
cfg = NRConfig(channel="radio_map", radio_map_path="warehouse_map.pt", n_cells=2, cell_layout="custom",
               cell_positions_m=((-5.0, -8.0), (5.0, 10.0)))
eng = make_engine("L2", E, R, "cuda", cfg)
```

Isaac Lab loads its assets from an S3 bucket (`ISAAC_NUCLEUS_DIR`). Inside Isaac Sim the Omniverse resolver opens those URLs. Plain `usd-core` cannot, so `python -m isaac_net.tools.scene.fetch <url> <dir>` mirrors a USD file and every layer it composes (sublayers, references, payloads, recursively; textures are skipped) and prints the local path.

### Inside Isaac Lab (at env creation)

```python
from isaac_net.isaac import IsaacNetCfg
from isaac_net.isaac.scene_map import SceneRadioMapCfg

isaac = IsaacNetCfg(pose_asset="robots", gnb_pos=((-5.0, -8.0, 7.0), (5.0, 10.0, 7.0)),
                    scene_map=SceneRadioMapCfg(root="/World/envs/env_0/Warehouse", frame="/World/envs/env_0",
                                               cell_m=0.5, samples=4_000_000, depth=5))
self.net_setup("L2-legacy", R, nr, "reference", isaac=isaac)      # in _setup_scene
```

With `scene_map` set, `net_setup` calls `isaac.scene_map.resolve_scene_map` before it builds the network. It exports the subtree `root` of the current stage, hashes the exported geometry and materials together with every bake parameter, and loads `<cache>/<key>.pt` if that file exists. Otherwise it bakes, in the same process when `sionna.rt` imports, or else in the interpreter named by `SceneRadioMapCfg.python` or `$ISAAC_NET_SIONNA_PYTHON` (which must have `sionna-rt` and torch). It then returns an `NRConfig` with `channel="radio_map"` and one cell per gNB at the gNB positions, and an `IsaacNetCfg` with `radio="engine"`, so the engine's radio samples the map. The cache is `$ISAAC_NET_RADIO_MAPS`, or `~/.cache/isaac_net/radio_maps`, and the exported scene is kept next to the map (`scene_<hash>/scene.xml`, `manifest.json`).

`SceneRadioMapCfg` fields that are `None` come from the other configs: the transmitters are `IsaacNetCfg.gnb_positions(nr)` (x, y and mast height), the carrier is `NRConfig.carrier_ghz`, the antenna height is `NRConfig.ue_height_m`, and the offset is `IsaacNetCfg.pose_offset_m`. The default `exclude=("*/robot*",)` keeps the robots out of the static map. All envs share env_0's map, so the envs must be clones of one layout.

## Coordinate convention

A point `p` of a prim is written as

```
p_map = C( p_local · M_prim · M_frame⁻¹ ) + offset
```

`M_prim` is the prim's local-to-world matrix (USD row-vector convention, at the default time code). `M_frame` is the local-to-world matrix of the `frame` prim (identity without one). `C` converts stage units to metres (`metersPerUnit`) and turns a Y-up stage into Z-up (`(x, y, z) → (x, −z, y)`). The result is metres with Z up, which is Sionna's convention and the engine's arena frame. In Isaac Lab the map frame must be the frame of the poses the network sees: env-local coordinates (`frame="/World/envs/env_0"`) shifted by `IsaacNetCfg.pose_offset_m`, which is what the hook does. The map's rows run along y and its columns along x, and `bounds = (x0, y0, x1, y1)` are the first and last cell centres ([channels.md](channels.md#radio_map)). The gNB `z` is used by the bake only, because the engine's radio is 2-D.

## Materials

Each exported prim gets one material. The first rule that gives one wins:

1. **user**: the mapping `{pattern: material}`, fnmatch patterns tested against the prim path, then its semantic labels, then the name of its bound USD material;
2. **semantic**: the prim's or an ancestor's semantic labels (`UsdSemantics` `semantics:labels:*`, or Isaac's older `semantic:*:params:semanticData`) through the keyword table;
3. **material**: the bound USD material's name through the keyword table;
4. **name**: the names in the prim path through the keyword table;
5. **default**: `concrete`.

Names are split into tokens at non-alphanumerics and camel-case humps (`SM_RackShelf_01` → sm, rack, shelf, 01), and a token matches a keyword when it starts with it. Words that name a material win over words that name an object, so `WoodenRack` is wood.

| ITU material | Slab per face | Material words | Object words (warehouse vocabulary) |
|:---|---:|:---|:---|
| `concrete` (default) | 0.1 m | concrete, cement, asphalt, stone, brick, masonry | wall, floor, ground, ceiling, roof, pillar, column, slab, foundation, stair, ramp, dock, curb |
| `metal` | 2 mm | metal, steel, iron, alumin(ium), chrome, zinc, galvan(ized) | rack, shelf, shelving, bracket, forklift, beam, girder, truss, pipe, duct, conveyor, cage, locker, cabinet, container, ladder, railing, bollard, shutter, vent, lamp, light, wire, fence, barrel, drum, fuse (box), extinguisher, cart, trolley, vehicle, truck |
| `wood` | 20 mm | wood, timber, plywood, osb, oak, pine, cardboard, carton, paper, plastic, polymer, rubber | pallet, crate, plank, box, package, parcel, bin, bucket, bottle, cone, note |
| `glass` | 6 mm | glass, glazing | window, windshield |
| `plasterboard` | 12.5 mm | plaster, drywall, gypsum, sheetrock | partition, office |

Any other ITU-R P.2040 material of Sionna RT (`brick`, `ceiling_board`, `chipboard`, `plywood`, `marble`, `floorboard`, the three grounds) can be named in the user mapping. The table has no entry for plastic or cardboard, so both map to wood, the closest low-permittivity material. The slab thickness is the thickness Sionna RT puts behind every face a ray crosses. A closed solid is crossed twice, so the concrete default is half of a 0.2 m wall, as in Sionna's own scenes, and the sheet materials use the sheet thickness. `--thickness` or `SceneRadioMapCfg.thickness` changes it.

On Isaac Lab's `Simple_Warehouse/warehouse_multiple_shelves.usd`, 1,877 prims with 1,248,910 triangles were exported: 197 concrete, 355 metal and 1,325 wood prims. The material came from a semantic label for 1,751 prims, from the USD material name for 12 and from the prim name for 93, and 21 prims (barcode and sign decals) fell back to concrete. The fallback covers 0.005% of the triangles. On `warehouse.usd` (780 prims) and `full_warehouse.usd` (3,471 prims) the fallback covers 0.19% and 1.1% of the triangles, again signs and barcodes. It covered 3% and 20% before the keyword table gained brackets, barrels, bottles, fire extinguishers and cones. The table is meant to be extended, and the user mapping always wins.

## What is exported

Every prim under `root` that is visible, has purpose `default` or `render`, and is not excluded: `UsdGeom.Mesh` (polygons fan-triangulated, holes ignored, subdivision surfaces as their control mesh), the implicit shapes `Cube`, `Sphere`, `Cylinder`, `Cone`, `Capsule` and `Plane` (tessellated), and the instances of a `PointInstancer`. Instance proxies are traversed, so instanceable assets are exported. Guide and proxy geometry, invisible prims and their subtrees are skipped. Triangles are merged into one PLY per material. `manifest.json` records the counts per material and per rule, the bounds, the skipped prims and each prim's material and rule.

The stage hash that keys the cache is the SHA-256 of the exported vertices (rounded to 0.1 mm), faces, materials and thicknesses. The export of the warehouse from the running Isaac Sim stage (Kit, S3 references) and the export of the fetched copy with plain `pxr` gave the same hash.

## Validation

`python -m isaac_net.tools.scene.validate` builds three USD scenes with known geometry (40 m × 20 m, gNB at (10, 10, 4) m, robot antenna at 1.5 m, 3.5 GHz, 0.5 m cells, 4 × 10⁶ rays, depth 4) and compares the baked map with free-space loss and the ITU-R P.2040 single-slab transmission (`materials.slab_transmission_db`, 10.36 dB for 0.1 m of concrete at normal incidence). `tests/test_scene_bake.py` asserts the same comparisons. Behind the wall the values are power averages over 3 × 3 cells, because only about 20 rays reach one cell there. Results with sionna-rt 2.2.0 on the CPU (LLVM, lab box WSL):

| Scene | Points | Map − expected |
|:---|:---|:---|
| free space + one wall as a two-sided quad | 4 in line of sight | +0.05 to +0.11 dB against free space |
| | 4 behind the wall | −0.15 to +0.92 dB against free space − one slab |
| the same wall as a closed 0.2 m box | 4 in line of sight | +0.05 to +0.13 dB |
| | 4 behind the wall | −0.95 to +0.32 dB against free space − two slabs |
| arena with floor and four walls, box wall ending at y = 14 m | 4 in line of sight | +0.44 to +0.60 dB (floor and wall reflections add power) |
| | 1 past the wall's end | +1.97 dB |

The LOS map is 1.00 at every line-of-sight point and 0.00 at every point behind the wall. The CUDA variant on native Windows (RTX 4090, as SYSTEM) gave the same table to 0.01 dB, and the bake tests pass there too. The pipeline also recovers the geometry exactly when the same arena is written in centimetres with Y up, or nested under a translated prim and exported in that prim's frame (`tests/test_scene_radio_map.py`).

## Bake times

| Scene | Where | Export | Bake (radio map + LOS map) |
|:---|:---|---:|---:|
| validation scenes (62 triangles) | lab box WSL, CPU (LLVM) | < 0.1 s | 0.06–0.13 s |
| validation scenes | lab box Windows, CUDA | < 0.1 s | 0.01–0.16 s |
| `warehouse.usd` (470k triangles), 2 gNBs, 1 m cells, 1e6 rays, depth 4 | lab box WSL, CPU | 5.3 s | 1.8 s |
| same, 0.5 m cells, 4e6 rays, depth 5 | lab box WSL, CPU | 4.7 s | 2.1 s |
| `warehouse_multiple_shelves.usd` (1.25M triangles), 1 m / 1e6 / 4 | lab box WSL, CPU | 8.5 s | 1.7 s |
| same, 0.5 m / 4e6 / 5 | lab box WSL, CPU | 8.9 s | 2.5 s |
| `full_warehouse.usd` (677k triangles, 3,471 prims), 1 m / 1e6 / 4 | lab box WSL, CPU | 28 s | 2.1 s |
| `warehouse_multiple_shelves.usd` in Isaac Lab, 0.5 m / 4e6 / 5 | lab box Windows, RTX 4090 (CUDA + OptiX), as SYSTEM | 2.9–4.6 s | 1.0 s in the solver, 8.3 s for the whole subprocess |

The bake is short. Most of the wall time is the export (per-prim material binding and transform queries), the Python and Dr.Jit start-up of a separate interpreter (about 7 s on Windows), and the first JIT compilation. A cached map costs only the export, which is needed for the hash. The command-line runs above took 16–37 s end to end.

## Demo: the fleet task in Isaac Lab's warehouse

`isaac_net/examples/isaac_warehouse_env.py` moves the fleet task of `isaac_fleet_env.py` into `Simple_Warehouse/warehouse_multiple_shelves.usd` (24 m × 38.8 m, 9.3 m high), spawned in every env. The robots are the same velocity-driven spheres. They collide with the shelves, and they spawn and pick goals only in free floor cells, taken from an occupancy grid of the same exported geometry (66% of the floor is free). Two gNBs hang at 7 m, at (−5, −8) and (5, 10) m. `benchmarks/isaac/warehouse_map_demo.py` bakes the map at env creation, then runs one scripted fleet (drive to goal with noise; no message, a 4 kB message or a 30 kB message with probability 0.6, 0.3 and 0.1) three times with the same seeds, under three channels that differ in nothing else:

- `map`: the baked radio map;
- `logdist`: the default log-distance channel (40 + 35 log10 d, 6 dB shadowing), same cells;
- `map_blockage`: the map plus robot-body blockage.

Settings: 8 envs × 8 robots, `L2-legacy` (the multi-cell slot-level engine), thermal noise (so the two cells interfere), 300 control steps (one 30 s episode). Run as a SYSTEM task on the lab box, Windows, Isaac Lab 3.0.

Delivered KPIs per arm. SINR is the engine's `sinr_db`, the full-power SNR of the serving link against the gNB's latest noise-plus-interference estimate (before power control). Delay counts delivered messages only.

| UE power | Channel | Delivered | Delay mean / p95 (ms) | AoI (s) | SINR p5 / p50 (dB) | Robots on cell 0 |
|:---|:---|---:|:---|---:|:---|---:|
| 23 dBm | `map` | 100.0% | 65.9 / 167.5 | 0.271 | 54.4 / 64.8 | 55.5% |
| 23 dBm | `logdist` | 100.0% | 49.6 / 107.5 | 0.261 | 43.5 / 57.7 | 55.8% |
| 23 dBm | `map_blockage` | 99.9% | 63.3 / 160.0 | 0.270 | 52.5 / 64.8 | 53.0% |
| −10 dBm | `map` | 100.0% | 56.0 / 135.0 | 0.265 | 23.1 / 31.9 | 55.5% |
| −10 dBm | `logdist` | 99.8% | 64.0 / 157.5 | 0.279 | 10.5 / 24.7 | 55.8% |
| −10 dBm | `map_blockage` | 100.0% | 56.5 / 137.5 | 0.266 | 19.7 / 31.8 | 53.0% |

The 7,681 messages, the robots' paths (2.3 goals per robot, 2.1 m/s mean speed) and the hazards are the same in every arm, and the fleet knew every hazard in every arm. The channel changes the delay by up to a third, in a direction that depends on the regime:

- **The map is stronger than the default.** Over the free floor cells, the median path gain to the serving gNB is −63.0 dB in the map and −72.2 dB under the log-distance law without shadowing, and the map lies a median 1.2 dB above free space. Most of the exported geometry is wood-class goods (1.12 M of the 1.25 M triangles), and a 20 mm wood slab costs 0.8 dB. The log-distance exponent of 3.5 is too pessimistic for this hall.
- **The map also carries more inter-cell interference.** The median gap between the serving gNB and the other gNB is 7.6 dB in the map and 13.9 dB under log-distance. At 23 dBm, fractional power control holds the received power at its target, so the link is limited by interference, and the delay under the map is 33% higher (mean 65.9 ms against 49.6 ms). At −10 dBm, the robots run out of power, so the link is limited by path gain, and the map's stronger gains lower the delay by 12% (56.0 ms against 64.0 ms) and raise the 5th-percentile SINR from 10.5 dB to 23.1 dB. These two explanations come from the gain statistics above. The engine's per-slot interference was not logged.
- **Robot blockage** lowers the 5th-percentile SINR by 2–3 dB and moves some robots to the other cell. It changes the delay by less than 5%.

Bake and start-up for this run: the first start exported the stage in 2.9 s and baked in 8.3 s (a CUDA subprocess). Later starts found the map in the cache and spent 4.4–5.0 s on the export that computes the hash. Env creation took 24–30 s in all.

## Limitations

- **Static scene.** The map is baked once for the geometry at env creation. Moving robots, forklifts and people are not in it. Robot bodies are handled by the blockage add-on (spheres on the robot–gNB segment, a fixed loss per blocked link). Moving scene objects other than robots are not modelled, and a scene that changes needs a new bake.
- **One map for all envs.** Every env uses env_0's map. Envs with different layouts would need one map per env, which `RadioMap` does not support.
- **2-D map at one height.** The map is a horizontal plane at `ue_height_m`. The z of the robot poses is ignored, as in the other channel models.
- **Cell averages and Monte-Carlo noise.** The solver averages the path gain over each cell, so cells next to a gNB read below the free-space gain at the cell centre, and cells far from a gNB or behind strong obstacles receive few rays. More samples or larger cells reduce the noise. Cells that no ray of a gNB reached (inside solid obstacles, or beyond `max_depth` interactions) are filled from their neighbours for that gNB. The `valid` field marks the cells some ray reached.
- **Materials by name.** The keyword table is a heuristic for warehouse assets. Check `manifest.json` (the rule behind each prim) and add user rules where it guesses wrong. Plastic and cardboard use the wood parameters. Every face is a slab of one thickness, and a closed solid is crossed twice.
- **Mechanisms.** Line of sight, specular reflection and refraction (transmission through walls) are on, diffuse scattering is off, and diffraction is off by default (`--diffraction` turns wedge diffraction on). Antennas are isotropic with vertical polarization, so antenna gains and patterns are not in the map.
- **Export coverage.** NURBS patches, curves, points and volumes are not exported, and neither are USD `Plane` prims on USD versions without that schema. Geometry animated over time is exported at the default time code.
- **Sionna RT on the lab box.** The CUDA variant needs OptiX, which WSL 2 does not expose, so WSL bakes run on the CPU. On native Windows the CUDA variant works as SYSTEM, while the LLVM variant does not load without an LLVM install. Sionna RT registers its radio-material plugin for the Mitsuba variant active when it is imported, so `bake_scene(variant=...)` must run before anything imports `sionna.rt` in that process.
