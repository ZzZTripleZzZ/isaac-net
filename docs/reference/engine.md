# Engine

`make_engine(level, E, R, device, config, backend)` builds the network of `E` environments with `R` robots each at one fidelity level. Every engine it returns has the same API, the *engine contract* below, so a task switches fidelity by changing the first argument.

```python
import torch
from isaac_net import NRConfig, Requests, make_engine

net = make_engine("L2-legacy", E=256, R=16, device="cuda", config=NRConfig(), backend="graph", seed=0)
net.submit(None, Requests(send))          # send [E,R] long: 0 nothing, c >= 1 one message of class c
out = net.step(None, poses)               # poses [E,R,2|3] in metres, or an SNR [E,R] in dB
net.reset(done_ids)                       # partial reset of the envs whose episodes ended
```

## The engine contract

| Call | Meaning |
|:---|:---|
| `reset(env_ids=None)` | Re-initialize `env_ids` (all if `None`; an index tensor, a list or a bool mask `[E]`): queues, MAC and HARQ state, link adaptation, fading, radio and clock. Other environments stay bit for bit unaffected, and the reset's random draws come from the engine generator. |
| `submit(t, requests, snr_db=None)` | Enqueue new messages captured at `t`. `requests` is a [`Requests`](traffic.md) or a `send [E, R]` tensor. `snr_db [E, R]` is recorded as a message feature (default: the SNR of the previous step). Returns `accepted [E, R]` bool. On `L2`, the keyword arguments `tag=`, `priority=` and `deadline_ms=` attach per-message extras. |
| `step(t, poses_or_snr)` | Advance every environment from `t` to `t + 1` and return the output dict below. Positions `[E, R, 2]` or `[E, R, 3]` go through the engine's radio; an `[E, R]` tensor is taken as the SNR in dB. |
| `clock` | `[E]` long, control steps since each environment's last reset. |
| `queued()` | `[E, R]` long, messages in each robot's queue. |

`t` is `None` in almost all code, which means "each environment's own clock". The prototype levels also accept an int (the same step for every environment) or an `[E]` tensor. The NR engine accepts an explicit `t` only when it equals its clock, because it cannot jump in time.

The legacy calls `add_frames(t, send, det, hid, snr_db)` and `step(t, snr_db, cur_hid) -> (newest, det_env)` remain as thin wrappers, so code written for the prototype keeps working.

## Step outputs

`step` returns a dict. Per-message entries refer to the robot's `F` message slots *as queued before the step* (`F = NRConfig.frame_buffer`, 16 by default).

| Key | Shape, type | Meaning |
|:---|:---|:---|
| `delivered` | `[E, R, F]` bool | the message in this slot was delivered during the step |
| `timed_out` | `[E, R, F]` bool | the message in this slot hit its deadline (`timeout_steps`) and was dropped |
| `cap`, `cls` | `[E, R, F]` long | capture step and traffic class of the slot (`-1` and `0` if empty) |
| `delay` | `[E, R, F]` float | delivery time minus capture step, in control steps; `NaN` if not delivered |
| `newest` | `[E, R]` long | newest capture step delivered in this step, `-1` if none |
| `det_env` | `[E]` bool | a message carrying the environment's current event id (`Requests.hid`) was delivered |
| `queue_len` | `[E, R]` long | messages queued after the step |
| `queue_bytes` | `[E, R]` float | bytes still queued after the step |
| `sinr_db` | `[E, R]` float | wideband SNR or SINR used for this step |
| `t` | `[E]` long | the clock value of this step; the clock is now `t + 1` |

Some engines add entries:

| Key | Engines | Meaning |
|:---|:---|:---|
| `serving_cell` | `L2`, multi-cell `L2-legacy` | serving cell of each robot `[E, R]` (0 with one cell) |
| `dropped` | `L2` | `[E, R, F]` messages lost under RLC unacknowledged mode (`harq_fail="drop"`) and resolved this step |
| `dl_newest`, `dl_queue_len` | `L2` with `NRConfig(dl=True)` | downlink counterparts of `newest` and `queue_len` |
| `arrival`, `arrival_slot`, `tag`, `priority`, `bytes`, `deadline_miss` | `L2` with traffic models or `submit` extras | per message `[E, R, F]`: arrival time in the env clock including the in-step offset, its slot, the extras, bytes on the air, and whether the deadline was missed; `delay` then counts from the arrival slot |
| `gen_accepted`, `gen_bytes` | `L2` with traffic models | `[E, R]` generated messages and bytes accepted this step |

Capture steps in every output are in the environment's own clock, so after a reset they restart at 0.

## Constants

| Name | Value |
|:---|:---|
| `LEVELS` | every level `make_engine` accepts: `SIM_LEVELS + SURROGATE_LEVELS + BOUND_LEVELS` |
| `SIM_LEVELS` | `("L0", "L0DR", "L05", "L05Q", "L1", "L2", "L2-legacy")` |
| `SURROGATE_LEVELS`, `BOUND_LEVELS` | `("TR", "GE", "QA", "NN")` and `("ORACLE", "NOCOMM")` |
| `BACKENDS` | `("reference", "eager", "graph", "compile", "triton")` |
| `FAST_BACKENDS` | `BACKENDS` without `"reference"` |

Which backend is available for which level is listed on [Fidelity levels](levels.md#backends-per-level).

::: isaac_net.core.engine.make_engine
    options:
      heading_level: 2

## The prototype engine base

The prototype levels (`L0` to `L1`, `L2-legacy`), the surrogates and the bounds all derive from `NetBase`, whose docstrings define the contract calls in detail.

::: isaac_net.core.proto.netsim.NetBase
    options:
      members: [reset, submit, step, attach_radio, add_frames]

## NREngine

`make_engine("L2", ...)` returns an `NREngine`, which wraps the configurable NR engine `NRNet` in the contract API. Attributes it does not define itself, such as the per-direction MAC objects `ul` and `dl`, are forwarded to the wrapped `NRNet`.

::: isaac_net.core.engine.NREngine
    options:
      members: [clock, reset, submit, step, add_dl_frames, attach_radio, set_sinr_hook, add_frames]
