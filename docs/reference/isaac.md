# Isaac Lab layer

`isaaclab_net.isaac` puts a network into an Isaac Lab task. Only `rigid_positions_local` and the `mdp` event term touch Isaac Lab objects, and nothing in the package imports Isaac Lab at import time, so `NetModule`, `MessageHistory` and the mixin hooks also run on a CPU without Isaac ([Tutorial 04](../tutorials/04_isaac_lab_integration.ipynb) does exactly that).

| Name | Role |
|:---|:---|
| `NetModule` | the network of one environment batch, on any `make_engine` level, plus the Isaac radio, per-message tags and freshness outputs |
| `TrafficRequest` | messages per robot for one control step, with an optional per-message tag |
| `NetEnvMixin` | four hook calls that wire a `NetModule` into a `DirectRLEnv` |
| `MessageHistory` | the receiver's delayed view of per-robot payloads |
| `IsaacRadio`, `ParamRanges` | poses to SNR with per-environment parameters, several gNBs and line-of-sight blockage |
| `mdp.randomize_network` | an event term that redraws the per-environment radio parameters (network domain randomization) |
| `net_features` | the compact `[E, R, 4]` network observation |

## Where the calls go in a DirectRLEnv

`DirectRLEnv.step` in Isaac Lab 3.0 calls `_pre_physics_step`, then the physics substeps, then `_get_dones`, `_get_rewards`, `_reset_idx` for finished environments, and finally `_get_observations`. The mixin's calls fit this order:

| Hook | Call | Why there |
|:---|:---|:---|
| `_setup_scene` | `self.net_setup(level, R, config, backend, **kwargs)` | builds the `NetModule` once for all environments |
| `_pre_physics_step` | (task code) read the start-of-step pose and decide what to send | messages are captured at the start of the step |
| `_get_dones` | `out = self.net_step(poses_end, send, tag, cur_tag)` | the first hook after physics: it sees the end-of-step poses and the queues before any reset |
| `_reset_idx` | `self.net_reset(env_ids)` after `super()._reset_idx(env_ids)` | partial reset of the finished environments only |
| `_get_observations` | `self.net_obs()` | `[E, R, 4]` AoI, SNR, queued messages, delivered; zeros for environments that reset this step |

`net_setup("off")` removes the network: `net_step` returns `None` and `net_obs` returns zeros of the same shape, which is the ideal-link baseline. A multi-cell config needs the engine's own radio, `net_setup(..., radio="engine")`.

## NetModule outputs

`NetModule.step` returns a dict whose keys differ from the core engine's in a few places, because it is written for task code:

| Key | Shape | Meaning |
|:---|:---|:---|
| `delivered` | `[E, R]` bool | at least one message of the robot was delivered this step |
| `newest_cap` | `[E, R]` long | newest capture step delivered this step, `-1` if none |
| `last_cap` | `[E, R]` long | newest capture step delivered this episode (0 means the state at reset) |
| `aoi_s` | `[E, R]` float | age of that information at the end of the step, in seconds |
| `queue_len`, `queue_bytes` | `[E, R]` | queue state after the step |
| `sinr_db`, `serving`, `blocked` | `[E, R]` | radio of this step |
| `tag_delivered` | `[E]` bool | a message tagged with the environment's current tag arrived (only if `cur_tag` is given) |
| `msg_delivered`, `timed_out`, `cap`, `cls`, `delay_s` | `[E, R, F]` | per message slot, as in the engine; `delay_s` in seconds |
| `t` | `[E]` long | the clock value of this step |

::: isaaclab_net.isaac.net_module.NetModule
    options:
      heading_level: 2
      members: [clock, reset, submit, step, set_params, sample_params, queued]

::: isaaclab_net.isaac.net_module.TrafficRequest
    options:
      heading_level: 2

::: isaaclab_net.isaac.mixins.NetEnvMixin
    options:
      heading_level: 2
      members: [net_setup, net_step, net_reset, net_obs]

::: isaaclab_net.isaac.mixins.rigid_positions_local
    options:
      heading_level: 2

::: isaaclab_net.isaac.net_module.MessageHistory
    options:
      heading_level: 2
      members: [push, update, reset]

::: isaaclab_net.isaac.net_module.net_features
    options:
      heading_level: 2

::: isaaclab_net.isaac.radio.IsaacRadio
    options:
      heading_level: 2
      members: [reset, set_params, sample_params, params, snr_db]

::: isaaclab_net.isaac.radio.ParamRanges
    options:
      heading_level: 2

::: isaaclab_net.isaac.mdp.events.randomize_network
    options:
      heading_level: 2

::: isaaclab_net.isaac.net_module.NetConfig
    options:
      heading_level: 2
      members: false
