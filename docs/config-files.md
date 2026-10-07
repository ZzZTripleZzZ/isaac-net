# Config files and presets

An `NRConfig` can be written to a YAML or JSON file, read back exactly, built from a preset by name, and changed field by field from the Isaac Lab command line. This page describes the file format, the preset files that ship with the package, the Hydra override syntax of the registered tasks, and the errors a misspelled field produces.

## Writing and reading a config

```python
from isaac_net import NRConfig, warehouse_private_5g

cfg = warehouse_private_5g(frame_buffer=32)
cfg.to_yaml("warehouse.yaml")                       # every field
cfg.to_yaml("warehouse_diff.yaml", only_changed=True)   # only the fields that differ from NRConfig()
same = NRConfig.from_yaml("warehouse.yaml")
assert same == cfg == NRConfig.from_yaml("warehouse_diff.yaml")

d = cfg.to_dict()                                   # JSON-safe dict
assert NRConfig.from_dict(d) == cfg
cfg.to_json("warehouse.json")                       # strict JSON
NRConfig.from_json("warehouse.json")
```

| Method | What it does |
|:---|:---|
| `cfg.to_dict(only_changed=False)` | A JSON-safe dict of the fields. `only_changed=True` keeps the fields that differ from `NRConfig()` |
| `NRConfig.from_dict(d, base=None)` | The config of a dict. The fields are applied to `base` (`base.with_(...)`) when it is given, else to `NRConfig()` |
| `cfg.to_yaml(path=None, only_changed=False, header=None)` | YAML text, written to `path` when given. `header` becomes `#` comment lines at the top |
| `NRConfig.from_yaml(src, base=None)` | Reads a path, a `pathlib.Path`, an `importlib.resources` object or YAML text |
| `cfg.to_json(...)`, `NRConfig.from_json(...)` | The same with JSON |
| `NRConfig.from_preset(name, **overrides)` | The config of a preset by name (see [Presets](#presets)) |

The round trip is exact: `NRConfig.from_dict(cfg.to_dict()) == cfg` holds for every preset, every scenario and a config with every nested wrapper and traffic model set, and the test suite checks it. A loaded config passes through `__post_init__` once more. Its normalizations are idempotent (the `channel="tr38901_inf_sh"` shorthand, `fading_rho_per_ms` from `ue_speed_mps`, the traffic tuple), so the loaded config equals the written one, and `cfg.with_()` equals `cfg`. `describe()`, `diff()` and `unused_fields()` work on a loaded config as on any other.

YAML needs PyYAML (`pip install pyyaml`; Isaac Lab installs it with Hydra). Without it `to_yaml` writes JSON text, which is also valid YAML, and `from_yaml` reads JSON content only.

## File format

A config file is a mapping from `NRConfig` field names to values:

```yaml
# robots.yaml: a hand-written config, the rest of the fields keep their defaults
mu: 1
bandwidth_mhz: 40
msg_sizes: [4000.0, 30000.0]
qos_pdb_ms: [10.0, inf]
ul_pc: true
ul_tpc: true
cell_positions_m:
- [0.0, 0.0]
traffic:
- __type__: TrafficModel
  kind: periodic
  size_bytes: 200.0
  period_ms: 10.0
  robots: [0, 1]
- kind: video
  size_bytes: 10133.333333333334
  period_ms: 33.333333333333336
  gop: [40000.0, 8000.0, 15]
  priority: 1
edge:
  __type__: EdgeConfig
  service_ms: [5.0, 20.0]
  deadline_ms: 80.0
background:
  n_background: 3
```

The values follow these rules:

- Tuples are written as lists and read back as tuples wherever the field holds a tuple (`msg_sizes`, `cell_positions_m`, `blocker_size_m`, ...).
- `None` is written as `null`.
- `inf`, `-inf` and `nan` are written as the strings `inf`, `-inf` and `nan`, so the files are strict JSON too, and read back as floats wherever the field holds a float.
- A nested config is a mapping with a `__type__` key: `EdgeConfig` (`edge`), `BackgroundConfig` (`background`), `EnergyConfig` (`energy`), `WifiConfig` (`wifi`) or `FidelityConfig` (`fidelity`). In a file you write by hand the key can be left out, because each of these fields holds one type. Inside a nested config, `to_dict(only_changed=True)` keeps the fields that differ from that config's defaults.
- A traffic model (`traffic`, and `traffic` / `dl_traffic` of a `BackgroundConfig`) is a mapping of its `kind` and the fields its constructor set, which are the fields that differ from the `TrafficModel` defaults. A list of them is read back as a tuple. A model is rebuilt from these fields directly, not through its constructor, so a video model's `size_bytes` is already the mean frame size and is not rescaled. An event model with a callable trigger cannot be written, so give the trigger a name (`TrafficModel.event(..., trigger="alarm")` and `engine.step(..., triggers={"alarm": mask})`).
- A `preset` key names a preset. The other keys are then passed to the preset function as keyword overrides, so `{preset: srsran_like, mu: 0}` equals `srsran_like(mu=0)`, whose SR and HARQ timers follow the new `mu`.

## Presets

`NRConfig.from_preset(name, **overrides)` calls the Python preset `name` with the overrides. The Python functions are the source of truth:

| Name | Source |
|:---|:---|
| `netslot_compat`, `lena_like` (alias `lena_match`), `lena_validation`, `lena_match_v2`, `lena_validation_v2`, `srsran_like`, `oai_like`, `multicell` | validation presets, `isaac_net/core/config.py` |
| `warehouse_private_5g`, `factory_inf`, `outdoor_campus`, `urllc_control` | scenario presets, `isaac_net/core/scenarios.py` ([Scenario presets](configurability.md#scenario-presets)) |

```python
from isaac_net import NRConfig
from isaac_net.core.presets import PRESETS, preset_path

cfg = NRConfig.from_preset("factory_inf", n_cells=2, ul_tpc=True)
yaml_copy = NRConfig.from_yaml(preset_path("factory_inf"))      # the shipped file, also from an installed wheel
print(sorted(PRESETS))
```

The folder `isaac_net/core/presets/` holds one YAML copy per preset (`lena_validation_v2.yaml`, `warehouse_private_5g.yaml`, ...), with the fields that differ from `NRConfig()`. Each file starts with a comment that says it is generated and names its source function. `scripts/export_presets.py` writes them, and `tests/test_config_io.py` regenerates every file and fails when one differs, so they cannot drift from the Python presets. After changing a preset or an `NRConfig` default, run:

```bash
python scripts/export_presets.py           # rewrite the files
python scripts/export_presets.py --check   # exit 1 if a file is stale
```

The files are package data, and `preset_path(name)` returns an `importlib.resources` object that `NRConfig.from_yaml` reads directly. Copy one as the starting point of your own config file.

## Hydra overrides in the registered tasks

The fleet tasks (`Isaac-NetFleet-Direct-v0`, `Isaac-NetFleet-Direct-L0-v0`, `Isaac-NetFleet-Direct-Warehouse-v0`) hold an `IsaacNetCfg` in `env.net_isaac`. Its `nr` field is a dict of `NRConfig` field overrides, applied to the task's `NRConfig` when the network is built (`IsaacNetCfg.resolve_nr`, which calls `NRConfig.from_dict(nr, base=task_config)`). Any `NRConfig` field can therefore be set on the train command line with Isaac Lab's `env.` prefix:

```bash
isaaclab train --rl_library rsl_rl --task Isaac-NetFleet-Direct-v0 \
    --external_callback isaac_net.isaac.tasks.register \
    env.net_isaac.nr.ul_pc=true env.net_isaac.nr.ul_tpc=true env.net_isaac.nr.frame_buffer=32

# start from a preset, then change fields of it
python -m isaac_net.isaac.tasks.train --task Isaac-NetFleet-Direct-Warehouse-v0 \
    env.net_isaac.nr.preset=warehouse_private_5g env.net_isaac.nr.frame_buffer=16

# tuples and other literals are Python syntax
python -m isaac_net.isaac.tasks.train --task Isaac-NetFleet-Direct-v0 \
    "env.net_isaac.nr.msg_sizes=(2000.0,20000.0)" env.net_isaac.nr.control_step_ms=100.0
```

How it works: Isaac Lab 3.0 applies every `env.`-prefixed override to the env cfg object itself (`isaaclab_tasks.utils.hydra`), with `true`, `false` and `none` read as Python values and everything else through `ast.literal_eval`, or as a string when that fails. An override of a key inside the `nr` dict adds or replaces that key, so no `+` is needed. Only overrides without the `env.` or `agent.` prefix go through Hydra's composition, which converts the cfg to a dict and back. The `nr` dict survives that round trip unchanged, and `from_dict` turns its lists back into tuples. An `NRConfig` object stored in a cfg comes back with its nested tuples turned into lists, so the registered tasks keep the overrides in the dict.

Rules of the `nr` dict:

- Without a `preset` key, the fields are applied on top of the task's own `NRConfig` (the fleet task's message sizes, 16-frame buffer, 2 s timeout and the control step of the env).
- With `env.net_isaac.nr.preset=<name>` the config is that preset with the other keys as overrides, and the task's own `NRConfig` is not used. A preset sets its own `control_step_ms` (100 ms, or 10 ms for `urllc_control`), so it must match the env step (0.1 s for the fleet tasks), or `net_substeps` / `net_decimation` must cover the difference: `net_setup` checks the rates. A preset with several cells (`factory_inf`, `outdoor_campus`) also needs `env.net_isaac.radio=engine`.
- `IsaacNetCfg(nr=NRConfig(...))` replaces the task's `NRConfig` altogether (Python only).
- The resolved `NRConfig` also sizes the observation space (`finalize_fleet_cfg`), so an override of `frame_buffer` or of the cell count changes the widths consistently. The warehouse task sizes its observation from `env.net_nr` before the overrides are applied, so on that task leave the observation-width fields (`frame_buffer` with `delivered_mask` or `msg_delay`, and the cell count with `serving_cell`) as they are.

Other `IsaacNetCfg` fields take the same prefix, for example `env.net_isaac.log_kpis=false` or `env.net_isaac.obs_history=8` (see [Network KPIs in TensorBoard](isaac-lab.md#network-kpis-in-tensorboard)). The manager-based task keeps its `IsaacNetCfg` in `env.isaac_net.isaac`, so `env.isaac_net.isaac.nr.ul_tpc=true` works there.

## Errors for unknown fields

Reading is strict. A key that is not a field raises `UnknownFieldError` (a `ValueError`) with the closest field names, found with `difflib.get_close_matches`:

```text
>>> NRConfig.from_dict({"ul_tcp": True})
UnknownFieldError: NRConfig has no field 'ul_tcp' (did you mean 'ul_tpc' or 'ul_pc'?)
>>> NRConfig.from_dict({"background": {"n_backgrund": 2}})
UnknownFieldError: NRConfig.background: BackgroundConfig has no field 'n_backgrund' (did you mean 'n_background'?)
>>> NRConfig.from_preset("warehouse_private5g")
UnknownFieldError: unknown preset 'warehouse_private5g' (did you mean 'warehouse_private_5g'?); presets: ...
```

The same check runs on the `nr` dict of an `IsaacNetCfg`: when the cfg is built in Python, and when the network is built after command-line overrides. A typo such as `env.net_isaac.nr.ul_tcp=true` therefore stops the run in `finalize_fleet_cfg`, before the simulation starts, with the message above. Values are validated by `NRConfig.__post_init__` as for any config, so `env.net_isaac.nr.mu=5` fails with the usual assertion.
