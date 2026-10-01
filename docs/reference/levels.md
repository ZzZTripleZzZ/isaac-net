# Fidelity levels

Every level comes from `make_engine(level, E, R, device, config, backend)` and honors the [engine contract](engine.md#the-engine-contract). A task switches fidelity by changing the level string.

| Level | Model | Parameters | Typical use |
|:---|:---|:---|:---|
| `L0` | i.i.d. lognormal delay and loss per message, no queue interaction | `l0_delay_median_steps`, `l0_delay_log_sigma`, `l0_loss` from the config, or `params={"mu", "sig", "p"}` | the usual randomized-delay baseline |
| `L0DR` | `L0` whose median delay, spread and loss are redrawn per environment at every reset | `dr_*` ranges from the config | domain randomization over delay |
| `L05`, `L05Q` | delay and drop looked up in tables fitted offline from `L2` rollouts, per cell of backlogged robots × SNR × class (`L05Q` also by the robot's own queue) | `params={"q", "pdrop"}` (required) | cheap state-conditioned delay |
| `L1` | fluid slot model: robots with queued data share each uplink slot equally, FIFO queues | `l1_eta` (goodput factor) | contention without MAC detail |
| `L2` | configurable NR MAC and PHY: 3GPP MCS/TBS and BLER tables, multiple HARQ processes, optional downlink, 1 to 7 cells with interference, power control and handover | the whole `NRConfig` | the fidelity model |
| `L2-legacy` | the prototype slot-level MAC and PHY, frozen; also runs 1 to 7 cells with a multi-cell config | application fields; the cell block for multi-cell | reproducing earlier prototype runs, and speed at scale |
| `TR` | trace replay: each environment replays one recorded `L2` or `L2-legacy` env-episode, open loop | fit file (required) | the replayed-trace baseline |
| `GE` | three-state Markov-modulated delay and loss, one chain per environment | fit file (required) | a Gilbert–Elliott-style baseline |
| `QA` | analytic processor-sharing queue per control step, FIFO service, scheduling-request delay | fit file, or an uncalibrated default | contention without slot simulation |
| `NN` | learned stateful surrogate: an MLP predicts a drop probability and delay quantiles from features at submission | fit file (required) | the learned-surrogate baseline |
| `ORACLE` | every message delivered at capture, delay 0, never lost | none | upper bound on what any network gives a task |
| `NOCOMM` | no message is ever delivered | none | lower bound: the task without communication |

## Semantics shared by the levels

- **Message model.** Every level uses the same per-robot FIFO of `F = 16` message slots, the same 20-step (2 s) application deadline and the same output dict. The prototype levels, the surrogates and the bounds are compiled around these constants and a 100 ms control step, so `make_engine` refuses a config that asks for other values. The NR engine `L2` reads `frame_buffer`, `timeout_steps` and `control_step_ms` from the config.
- **Radio.** `L0` to `L1`, the surrogates and the bounds use the fixed legacy radio (one gNB at the origin, log-distance path loss with correlated shadowing, a −90 dBm noise floor). `L1` and `QA` refuse any other cell setting. `L2` and multi-cell `L2-legacy` build their radio from the config, and only they accept `n_cells > 1` (up to 7, with uplink power control on by default).
- **Config fields.** Each level reads only some `NRConfig` fields. `NRConfig.unused_fields(level)` lists the non-default fields a level ignores, and `make_engine(..., strict=True)` raises on them (see [Configuration](config.md#strict-mode)).
- **Resets.** Every level keeps per-env clocks and exact partial resets. The surrogates draw their reset state (the replayed trace of `TR`, the initial state of `GE`) from the engine generator.

## Backends per level

| Level | `reference` | `eager` | `graph` | `compile` | `triton` |
|:---|:---:|:---:|:---:|:---:|:---:|
| `L0`, `L0DR`, `L05`, `L05Q` | ✓ | ✓ | ✓ | ✓ | |
| `L1`, `L2-legacy` (one cell) | ✓ | ✓ | ✓ | ✓ | ✓ |
| `L2-legacy` (multi-cell) | ✓ | | | | |
| `L2` (one or several cells) | ✓ | | | | |
| `TR`, `GE`, `QA`, `NN`, `ORACLE`, `NOCOMM` | ✓ | ✓ | ✓ | | |

For the surrogates and bounds, `reference` and `eager` run the same graph-safe code, and `graph` captures it. The fast backends of the prototype levels need a CUDA GPU.

## Fitting the surrogates

`TR`, `GE` and `NN` exist only as fits, and `QA` is calibrated by the same tool. The fit rolls out `L2` or `L2-legacy` on the example fleet task under a behavior policy, logs every message through the public API, and writes all four surrogates into one file outside the repository:

```bash
python -m isaac_net.tools.fit_levels --source L2-legacy --task T1 --device cuda --backend graph
python -m isaac_net.tools.fit_levels --source L2 --preset netslot_compat --task T1 --out ~/fits/T1_nr.pt
```

The default location is `~/.cache/isaac_net/levels/<source>_<task>.pt`, or `$ISAAC_NET_LEVELS_DIR`. The tool refuses a path inside the source tree, and fitted files are never committed. A JSON summary of the fit is written next to the file. Load the file with `params`:

```python
net = make_engine("NN", E, R, device, params="~/.cache/isaac_net/levels/L2-legacy_T1.pt", backend="graph")
```

The file records the message sizes it was fitted with, and `make_engine` refuses an engine whose `msg_sizes` differ. The surrogates share the prototype constants, so the fit tool refuses a source config with a different frame buffer or timeout.

::: isaac_net.tools.fit_levels.fit_levels
    options:
      heading_level: 3

::: isaac_net.tools.fit_levels.save_fit
    options:
      heading_level: 3

::: isaac_net.core.levels.load_level_params
    options:
      heading_level: 3

## Level classes

These classes are what `make_engine` returns for the surrogate and bound levels. Build them through `make_engine`, which checks the config first.

::: isaac_net.core.levels.surrogates.NetTR
    options:
      heading_level: 3
      members: false

::: isaac_net.core.levels.surrogates.NetGE
    options:
      heading_level: 3
      members: false

::: isaac_net.core.levels.surrogates.NetQA
    options:
      heading_level: 3
      members: false

::: isaac_net.core.levels.surrogates.NetNN
    options:
      heading_level: 3
      members: false

::: isaac_net.core.levels.bounds.NetOracle
    options:
      heading_level: 3
      members: false

::: isaac_net.core.levels.bounds.NetNoComm
    options:
      heading_level: 3
      members: false
