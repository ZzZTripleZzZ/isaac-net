# Configuration

One dataclass, `NRConfig`, configures every module: numerology, carrier and TDD pattern, MAC timing, HARQ and RLC, the PHY tables, the radio and cell layout, and the application fields. Do not add a second config class; new options become `NRConfig` fields.

```python
from isaaclab_net import NRConfig, make_engine
from isaaclab_net.core import multicell, netslot_compat

cfg = NRConfig(mu=1, bandwidth_mhz=20, tdd_pattern="DDDSU", n_harq=16, mcs_table=2, dl=True)
print(cfg.summary())                     # 51 PRB in 13 RBGs, 200 slots per 100 ms step, ...
cfg2 = cfg.with_(bler_target=0.01)       # modified copy
net = make_engine("L2", E, R, "cuda", cfg)
```

## Presets

Each preset returns an `NRConfig` and takes keyword overrides, for example `netslot_compat(n_harq=4)`.

| Preset | What it is |
|:---|:---|
| `netslot_compat()` | the geometry and timing of the legacy slot-level model with the 3GPP PHY: one HARQ process, 50 PRBs in 5 subbands of 10, `DDDSU` |
| `lena_like()` (alias `lena_match`) | the ns-3 5G-LENA reference scenario as far as the model allows; needs the locally generated 5G-LENA BLER tables |
| `lena_validation()` | `lena_like()` in the geometry of the 5G-LENA validation sweep (fading off, whole-band UE power, per-UE link budgets) |
| `srsran_like()`, `oai_like()` | uplink latency fitted to public srsRAN and OAI measurements (see [Public-data calibration](../calibration-public-data.md)) |
| `multicell(n)` | a hexagonal cluster of `n` cells at 100 m spacing, thermal noise, same-slot interference, uplink power control and A3 handover; runs on `L2` (up to 7 cells) and on `L2-legacy` |

## Strict mode

Every level reads only some fields, listed by `fields_read_by(level)`. The prototype levels read the application fields and their own parameters, because their radio and MAC are fixed, while `L2` reads the frame, NR, link and radio fields. A field that a level ignores is ignored silently by default. `NRConfig.unused_fields(level)` returns the fields that are set away from their defaults and that the level ignores, and `make_engine(..., strict=True)` raises a `ValueError` naming them:

```python
cfg = NRConfig(mcs_table=2, l0_loss=0.1)
cfg.unused_fields("L0")                  # ['mcs_table']
make_engine("L0", E, R, "cpu", cfg, strict=True)   # ValueError: level L0 ignores these config fields: mcs_table
```

Independently of strict mode, the prototype levels, the surrogates and the bounds refuse a config whose `frame_buffer`, `timeout_steps` or `control_step_ms` differ from their compiled constants (16, 20 and 100 ms), and single-cell levels refuse `n_cells > 1`.

## All fields

The tables below are generated from `isaaclab_net/core/config.py` when the docs are built. "Read by" names the engines that read the field: the application fields are read by every level, and "L2-legacy multi-cell" is `L2-legacy` with a non-legacy cell setting such as `multicell(n)`. [Configurability](../configurability.md) discusses the modelling choices behind many of them.

<!-- NRCONFIG_FIELDS -->

## Class and functions

::: isaaclab_net.core.config.NRConfig
    options:
      heading_level: 3
      members: [summary, with_, unused_fields, is_legacy_cell, gnb_xy, scs_khz, slot_ms, nprb, rbg, n_subbands, subband_prbs, slots_per_step, ul_slots_per_step, dl_slots_per_step, sr_delay, ul_rtt, slot_symbols, ul_capable, noise_dbm_per_prb, subband_noise_dbm, ul_pc_on, ul_slot_ms, ttt_slots, ho_int_slots]

::: isaaclab_net.core.config.fields_read_by
    options:
      heading_level: 3

::: isaaclab_net.core.config.netslot_compat
    options:
      heading_level: 3

::: isaaclab_net.core.config.lena_like
    options:
      heading_level: 3

::: isaaclab_net.core.config.lena_validation
    options:
      heading_level: 3

::: isaaclab_net.core.config.srsran_like
    options:
      heading_level: 3

::: isaaclab_net.core.config.oai_like
    options:
      heading_level: 3

::: isaaclab_net.core.config.multicell
    options:
      heading_level: 3

::: isaaclab_net.core.config.fading_rho_from_speed
    options:
      heading_level: 3

::: isaaclab_net.core.config.rbg_size_38214
    options:
      heading_level: 3
