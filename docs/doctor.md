# Doctor: check an install

`isaac-net-doctor` reports what is installed, runs a short self-test of the engines and checks a configuration against the backends. Run it first after installing, and attach its output (or `--json`) to a bug report.

```bash
isaac-net-doctor                                    # environment report + self-test (CPU, plus CUDA when available)
isaac-net-doctor --quick                            # shorter self-test, no throughput smoke
isaac-net-doctor --no-selftest                      # environment report only
isaac-net-doctor --device cpu                       # CPU checks only, even on a GPU machine
isaac-net-doctor --config lena_validation_v2        # which backends run a config (skips the self-test)
isaac-net-doctor --config "multicell(3, dl=True)"   # an expression over the presets
isaac-net-doctor --config my_cfg.py:CFG             # a config from your own file (my_cfg.py:make(16) works too)
isaac-net-doctor --json > doctor.json               # one JSON document instead of the text report
python -m isaac_net.tools.doctor ...                # the same without the console script
```

The doctor makes no network access. It finds the optional stacks with `importlib` (the module spec and the installed package metadata) without importing them, so a broken Isaac Lab or JAX install cannot crash it. The one exception is Triton, which it imports because the `triton` backend does. A missing GPU is reported, not an error.

## Exit code

| Code | Meaning |
|:---|:---|
| 0 | no self-test check failed and every required component is present |
| 1 | a self-test check failed, or Python, torch, numpy or `isaac_net` is missing or too old |
| 2 | bad command line, or `--config` could not be built |

A missing optional stack (Isaac Lab, Isaac Sim, MuJoCo Playground, JAX, Warp, Sionna, USD, mkdocs, the ns-3 bridge variables) has the status `optional` and never changes the exit code.

## Environment report

Every row has a name, what was found (version, path or `not installed`), a status (`ok`, `missing` or `optional`) and, when not `ok`, a hint that names the pip extra or the page that provides it.

| Row | What it reports |
|:---|:---|
| Python, torch, numpy | versions; torch with its build (CUDA version and cuDNN, or CPU-only). Required: Python 3.10+, torch 2.7+, numpy 1.23+ |
| CUDA, GPU | CUDA runtime of the torch build, NVIDIA driver (from `nvidia-smi`) and the CUDA version it supports; GPU name, memory and compute capability |
| Triton | version, or why the `triton` backend cannot run |
| isaac_net | version, import path and how it got there: wheel install, editable install, or a source tree on `PYTHONPATH`. A source tree that shadows an installed copy is flagged |
| Isaac Lab, Isaac Sim, Newton, OV PhysX | the packages `isaaclab`, `isaacsim`, `newton`, `ovphysx` |
| Isaac Lab mode | Isaac Lab with Isaac Sim (Kit, PhysX through Isaac Sim), or kit-less Isaac Lab with Newton / OV PhysX, where `ISAAC_NET_PHYSICS` selects the fleet env's physics ([Isaac Lab on Linux](isaac-lab-linux.md)) |
| MuJoCo Playground, JAX, Brax | the `mjx` extra ([MJX backend](backends-mjx.md)) |
| Warp, Sionna, USD (pxr), pyarrow, pybind11, mkdocs | the other optional stacks and the extras that install them |
| 5G-LENA tables | whether the locally generated tables exist; only `bler_source="lena"` presets need them ([licensing](licensing.md)) |
| `NS3BRIDGE_ROOT`, `NS3_TOOLCHAIN_ENV`, `ISAAC_NET_PHYSICS`, `ISAAC_NET_LENA_TABLES` | environment variables of the ns-3 bridges ([bridges](bridges.md)), the Isaac Lab physics and the 5G-LENA tables |

## Self-test

The self-test runs by default (not with `--config` alone). On the CPU it takes about 5 s, or about 3 s with `--quick`. With a CUDA device it adds the backend checks, which on the first run include the CUDA-graph capture and the Triton compile.

| Check | Device | What passes |
|:---|:---|:---|
| package data | CPU | the Sionna PHY tables shipped in `core/data` load |
| conservation L0, L1, L2-legacy, L2 | CPU | the reference engine runs 4 × 6 for 40 steps (20 with `--quick`) under light, medium and heavy load, with a partial reset every 10 steps. Every step, each message resolved in that step has exactly one of `delivered`, `timed_out` and `dropped`, no message is resolved twice, and each robot's queue grows by at most the one message it sent. Over the run, accepted = resolved + still queued + cleared by a reset |
| rng E-independence L2 | CPU | with `rng="engine"` and poses through the engine radio, env 0 of an engine with E = 2 equals env 0 with E = 4 bitwise at every step |
| graph == reference L2 | CUDA | `tests/nr_equiv.py` free-running, config `ul`, 8 × 4, 40 steps with partial resets: every output and every state tensor bitwise equal |
| triton ~ reference L2 | CUDA | `tests/nr_equiv.py` teacher-forced, config `ul`, 16 × 8, 20 steps: at most 1 in 1,000 active robot-steps differ, the tolerance of `tests/test_nr_fast.py` (float rounding, see [performance](performance.md#equivalence-methodology)) |
| throughput L2 graph | CUDA | 1,024 × 16, `NRConfig()`, 5 warm-up and 20 timed control steps; reports control steps/s and robot-steps/s. Skipped by `--quick` |

The two equivalence checks reuse the harness in `tests/nr_equiv.py`, which ships in the git checkout and the sdist but not in the wheel. From a wheel install they are skipped with that reason. Run the doctor from a checkout (`pip install -e .`) to get them.

The throughput line is a smoke test, not a benchmark. The numbers in the README and in [performance](performance.md) were measured on an idle RTX 4090 with the validated scale configuration, with warm-up and repeated timing windows. This check uses the default configuration on whatever else runs on the GPU, so expect a different value.

## Config check

`--config` builds the `NRConfig` and prints:

- a summary of the configuration: `NRConfig.describe()` when the installed version has it, otherwise the frame summary and every field set away from its default;
- the slots per control step: NR slots with their UL and DL data slots, and the UL slots of the prototype levels;
- for each L2 backend, whether it accepts the configuration and whether it can run on this machine: `graph` needs CUDA and `rng="engine"`, `triton` also needs Triton and refuses what its kernel does not implement (`NRTritonEngine.refusals`, for example several cells or the BSR grant pipeline of `lena_validation_v2`), and every backend needs the local 5G-LENA tables when `bler_source="lena"`;
- `unused_fields("L2")`, the fields set away from their defaults that level L2 ignores.

The preset names are those of `isaac_net.core` (`default`, `netslot_compat`, `lena_like`, `lena_match`, `lena_match_v2`, `lena_validation`, `lena_validation_v2`, `srsran_like`, `oai_like`, `multicell`, `multicell3`), plus the scenario presets of `isaac_net.core.scenarios` when the installed version has that module.

## Sample output

On a MacBook (arm64, CPU-only torch, source tree on `PYTHONPATH`), shortened:

```text
isaac-net-doctor (isaac_net 0.2.1.dev0)

Environment
  Python                 ok        3.12.2 (CPython, darwin arm64)
  torch                  ok        2.9.1 (CPU-only build)
  CUDA                   optional  not available (torch has no CUDA build)
                                   -> the graph and triton backends need an NVIDIA GPU; the CPU runs the reference engine
  Triton                 optional  not installed
  numpy                  ok        1.26.4
  isaac_net              ok        0.2.1.dev0 (source tree on PYTHONPATH or the working directory, not installed) at .../isaac_net
  Isaac Lab              optional  not installed
                                   -> Isaac Lab 3.0: docs/isaac-lab.md (Windows) or docs/isaac-lab-linux.md (Linux, kit-less)
  ...
  NS3BRIDGE_ROOT         optional  unset
                                   -> ns-3 bridge build tree (docs/bridges.md)

Self-test (cpu)
  PASS  package data             Sionna PHY tables shipped in core/data load  [0.0 s]
  PASS  conservation L0          40 steps, 4 partial resets: 436 accepted = 436 resolved once (436 delivered) + 0 queued + 0 cleared by reset  [0.0 s]
  PASS  conservation L1          40 steps, 4 partial resets: 436 accepted = 323 resolved once (309 delivered) + 55 queued + 58 cleared by reset  [0.1 s]
  PASS  conservation L2-legacy   40 steps, 4 partial resets: 436 accepted = 272 resolved once (239 delivered) + 81 queued + 83 cleared by reset  [0.6 s]
  PASS  conservation L2          40 steps, 4 partial resets: 436 accepted = 315 resolved once (287 delivered) + 57 queued + 64 cleared by reset  [2.9 s]
  PASS  rng E-independence L2    L2, 8 steps, poses input: env 0 bitwise equal at E = 2 and E = 4  [1.1 s]

self-test 6 passed, 0 failed, 0 skipped; 4.7 s; exit code 0
```

The config check of `lena_validation_v2` on the same machine:

```text
Config lena_validation_v2
  summary: mu=1 (30 kHz), 20 MHz -> 50 PRB, RBG 10 -> 5 subbands [10, 10, 10, 10, 10], TDD DDDSU S=(10, 2, 2), 200 slots/step (40 UL, 160 DL data slots), HARQ 16x4tx drop, MCS table 1, eesm, OLLA off, PF wideband
  non-default fields (24): n_prb=50, rbg_size=10, ..., ul_grant_model='bsr', ..., frame_buffer=128
  slots per control step (100.0 ms): 200 NR slots (40 UL, 160 DL data slots); prototype levels 40 UL slots
  backends (level L2):
    reference config accepted, cannot run here (bler_source='lena' needs the local 5G-LENA tables at ...)
    graph     config accepted, cannot run here (needs a CUDA device (none here); bler_source='lena' needs ...)
    triton    refuses this config (1 refusal, listed below; needs a CUDA device (none here); needs triton (not importable here))
    triton refusal: ul_grant_model='bsr' (the 5G-LENA SR / BSR grant pipeline, which lena_match_v2 / lena_validation_v2 turn on) (...)
  unused_fields('L2'): none (every non-default field is read by L2)
```

## Reading a failure

Each check prints one line: `PASS`, `FAIL` or `SKIP`, its name and the reason. A check never stops the others, and an exception inside a check is reported as a `FAIL` with the exception as the reason.

- **`FAIL package data`**: the wheel is incomplete, or `isaac_net` is imported from a tree without `core/data/`. The `isaac_net` row shows which copy was imported.
- **`FAIL conservation <level>`**: the reason names the step, env and robot where a message was flagged twice, resolved twice, or where the queue length did not match accepted minus resolved. This is an engine bug. Report it with the `--json` output.
- **`FAIL rng E-independence L2`**: an env's random draws depend on how many envs the engine holds, so runs with different E are not comparable. The check sets `rng="engine"` itself, so this is an engine bug. Report it with the `--json` output.
- **`FAIL graph == reference L2`**: the reason gives the first step and the outputs or state tensors that differ. The graph backend is supposed to be bitwise equal, so a mismatch is a bug, unless the CUDA or torch version is new and changed a kernel. Rerun the GPU tests (`pytest -m gpu tests/test_nr_fast.py`) to see which configurations are affected.
- **`FAIL triton ~ reference L2`**: more robot-steps differ than float rounding explains. Check that Triton matches the torch build (the Triton row), then run `python tests/nr_equiv.py --backend triton --cfg ul --mode teacher` for the full report.
- **`SKIP`** lines are expected: the CUDA checks skip without a GPU unless `--device cuda` is given (then they fail, since a GPU was requested), the equivalence checks skip without `tests/nr_equiv.py`, and the throughput smoke skips with `--quick`.
- A `missing` row in the environment report makes the exit code 1 even when every check passes. Its hint says what to install.

## JSON output

`--json` prints one document with the keys `tool`, `isaac_net` (version), `environment` (the rows, each with `key`, `name`, `found`, `version`, `status`, `hint`), `config` (with `--config`: `spec`, `backends`, `triton_refusals`, `unused_fields_L2`, `slots`, and `describe` or `summary` with `non_default_fields`), `selftest` (`device`, `quick`, `checks` with `name`, `status`, `reason`, `seconds`, `data`, and the counts `passed`, `failed`, `skipped`), `seconds` and `exit_code`.
