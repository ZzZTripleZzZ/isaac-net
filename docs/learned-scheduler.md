# Learned uplink scheduler (`scheduler="learned"`)

Status: experimental, branch `learned-scheduler`. NR engine (level `L2`), uplink, one cell; backends `reference` and
`graph` (bitwise equal). `triton` refuses the config for now (see the plan below).

## What it does

The UL MAC keeps everything it does under `scheduler="pf"` (retransmission admission, greedy RBG-by-RBG filling up to
the need or the power-headroom cap, HARQ, OLLA, PF averages) and multiplies the PF metric of every robot on every RBG
by `exp(lw[e, r])`:

```
metric[e, r, s] = rate_est[e, r, s] / avg[e, r] * exp(lw[e, r])
lw = clamp(scorer_{group[e]}(x[e, r]) + l_ext[e, r], -learned_logw_clip, +learned_logw_clip)
```

The default clip is `learned_logw_clip = ln 16`, so the weight `w = exp(lw)` is a positive per-robot factor in
[1/16, 16] that multiplies the PF ratio inside the per-slot grant decision, refreshed once per control step. This is
the actuation a deployed controller has on srsRAN/OCUDU with a per-RNTI weight patch (a per-UE positive weight
multiplying the UL PF metric; grant shares there are monotone and sublinear in the weight). `learned_metric="weight"`
(ablation only, not the default) replaces the PF metric by `w` (a channel-blind priority).

`x` is 14 numbers per robot: seven per-robot features and their mean over the env's robots.

| # | feature | scaling |
|:--|:--|:--|
| 0 | backlog: queued stream bytes not yet released | log1p(bytes / 1000) |
| 1 | head-of-line age of the oldest queued frame | ms / 100, clamped to [0, 10] |
| 2 | channel: wideband mean of the scheduler's per-RBG SINR estimate incl. OLLA | dB / 20 |
| 3 | HARQ processes owning undecoded bytes | / n_harq |
| 4 | transmissions so far of the oldest pending TB | / max_harq_tx |
| 5 | class of the robot's last submitted message (0 = none yet) | / number of classes |
| 6 | application hint `l_hint` (e.g. the fleet policy's rate class) | as given |

The scorer is a 14 -> `learned_hidden` (tanh) -> 1 MLP, 257 parameters at the default 16 hidden units, with one
parameter set per group (`learned_groups`); env `e` uses set `group[e]`. All parameters start at zero, and with zero
parameters and a zero external input the engine is bitwise the PF engine (tests L1).

## Timescales

* `learned_timescale="step"` (default): the scorer runs once per control step, on the state at the start of the step,
  and the weights hold for the step's slots. Together with the external input this is timescale (a): a trainer (or a
  near-RT RIC xApp) writes one log-weight per robot per control step with `set_sched_logw`.
* `learned_timescale="slot"`: the scorer runs before every UL data slot on that slot's features (timescale (b),
  deployable in the gNB MAC). Implemented and graph-capturable; not used by the Phase-1 pilot.

## API (NREngine, all backends that accept the config)

```python
cfg = netslot_compat(scheduler="learned", learned_groups=4)
net = make_engine("L2", E, R, dev, cfg, backend="graph")
net.set_sched_groups(torch.arange(E, device=dev) // (E // 4))   # population partition
net.set_sched_params(w1, b1, w2, b2, group=k)                    # [14,H], [H], [H], [] for group k (or [G,...])
x = net.sched_features(snr)                                      # [E,R,7] at the start of the next step
net.set_sched_logw(logw)                                         # [E,R], added to the scorer output
net.set_sched_hint(rate_class / 2)                               # feature 6
out = net.step(None, snr)
net.sched_state()                                                # {"feat", "logw"} of the last evaluation
```

Every setter writes in place into a fixed buffer, so the graph backend replays its captured graphs with the new values
and never captures again for a parameter or input update (checked by `test_gpu_graph_bitwise_with_live_parameter_writes`).
The parameters and inputs are not MAC state: `reset(env_ids)` leaves them, except the per-env last message class.

## Triton plan

The fused kernel already reads a per-robot weight `qw [E,R]` for `scheduler="qos"` (prologue in torch, product with
the PF metric in the kernel). Timescale "step" maps onto that path directly: evaluate `learned_prepare` in the torch
prologue and pass `exp(lw)` as `qw` with `SCHED=3, qos_gamma=1`. Timescale "slot" needs the scorer inside the kernel
(14 x 16 MAC per robot per slot, with the env mean as a reduction over the robot axis the program already holds); the
parameters would be loaded once per program from `[G, ...]` tensors indexed by the env's group.
