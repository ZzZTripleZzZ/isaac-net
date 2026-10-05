# Radio environment maps

`isaac_net.tools.rem` samples the large-scale radio of a configuration on an x–y grid and writes a radio environment map (REM), like the REM helper of 5G-LENA. It uses `radio.RadioMC`, the same radio the NR engine and `NetSlotMC` use, so every channel model works (`log_distance`, `tr38901`, `radio_map`) with the gNB layout of `NRConfig.gnb_xy()`.

```bash
python -m isaac_net.tools.rem --out rem.npz --png rem.png --res 2 --preset multicell --set n_cells=3 channel=tr38901_umi
isaac-net-rem --out rem.npz --bounds 0 0 200 100 --res 1 --set channel=radio_map radio_map_path=map.npz n_cells=2 cell_positions_m="((25,75),(125,75))"
```

```python
from isaac_net.core import multicell
from isaac_net.tools.rem import compute_rem, load_rem, save_rem

rem = compute_rem(multicell(3, channel="tr38901_umi"), resolution_m=2.0, seed=0)   # dict of numpy arrays
save_rem(rem, "rem.npz", png="rem.png")                                            # png needs matplotlib
rem = load_rem("rem.npz")
```

| Array | Shape | Meaning |
|:---|:---|:---|
| `x`, `y` | `[W]`, `[H]` | grid cell centres (m, env-local); rows run along y |
| `gnb_xy` | `[C, 2]` | gNB positions |
| `pathgain_db` | `[C, H, W]` | large-scale gain of every gNB-point link: path loss, shadowing and, for `tr38901`, the LOS state (negative dB) |
| `rsrp_dbm` | `[C, H, W]` | DL RSRP per resource element: `gnb_tx_dbm` over the 12 · `dl_nprb` REs of the DL carrier, plus the path gain |
| `sinr_db` | `[H, W]` | best-cell DL SINR per PRB with every gNB transmitting on every PRB (full-buffer interference, as 5G-LENA's REM), over the UE noise floor `noise_dbm_per_prb("ue")` |
| `serving` | `[H, W]` | serving cell = argmax RSRP, the engine's max-RSRP attach |
| `los` | `[C, H, W]` | LOS state, only for channel models that have one (`tr38901`) |
| `meta` | string | JSON: config summary, bounds, resolution, seed, env, noise floor, DL PRBs |

Every grid point is a receiver at `ue_height_m`. The shadowing and LOS fields are those of env `env` of a `RadioMC` whose fields come from a `torch.Generator` seeded with `seed`, so a fixed seed gives the same map on every run (the tool runs on the CPU). The fields are those of the channel model, not of a particular engine run: an engine with the same config draws its own fields from its engine RNG. The map is outdoor coverage without robots: the blockage add-on (other robots as spheres) and O2I (an indoor draw per receiver) are turned off, since grid points are not robots.

Options of the command line: `--preset` names a preset of `isaac_net.core.config` (`multicell`, `lena_like`, ...), `--set FIELD=VALUE ...` overrides `NRConfig` fields (values are Python literals), `--bounds X0 Y0 X1 Y1` (default: the arena `(0, 0, cell_arena_m, cell_arena_m)`), `--res` the grid spacing in m, `--seed`, `--env`, `--png`. From Python, `shape=(H, W)` fixes the grid size instead of the spacing, and `radio_map=` passes a `RadioMap` directly.

Tests: `tests/test_dl_traffic_fdd_rem.py` (grid shape, serving cell = argmax RSRP, determinism for a fixed seed, a different map for another seed, the npz round trip, the command line).
