# Adaptive and mixed fidelity

`AdaptiveEngine` (`isaaclab_net/core/adaptive.py`) runs a cheap level and an expensive level behind one engine API and decides, per env, which of them simulates each control step. Three uses share the same machinery:

1. **Static mixed fidelity.** A fixed random subset of the envs, for example 10%, runs on `L2-legacy` for the whole run, and the other envs run on `L1`.
2. **Load-triggered switching.** An env runs on the cheap level while its load indicator stays low and moves to the expensive level while the cell is congested, which is where the cheap levels diverge from `L2-legacy`.
3. **Curriculum.** A schedule over training iterations, such as `L1` for the first k iterations and `L2-legacy` afterwards, applied through a callback.

```python
from isaaclab_net.core import NRConfig
from isaaclab_net.core.adaptive import FidelityConfig, FidelityCurriculum, make_adaptive

fid = FidelityConfig(cheap="L1", expensive="L2-legacy", cheap_backend="triton", expensive_backend="triton",
                     mode="load", indicator="backlog", up_threshold=4000.0, active_budget=0.25)
net = make_adaptive(E, R, "cuda", NRConfig(), fid)       # or NRConfig(fidelity=fid) and make_adaptive(E, R, dev, cfg)
net.submit(None, Requests(send))
out = net.step(None, poses)                              # the step dict of make_engine, plus:
out["fidelity"]                                          # [E] 1 where this step ran on L2-legacy
out["fidelity_indicator"]                                # [E] the indicator after the step
net.fidelity_stats()                                     # switches up / down, budget denials, env-steps per level
```

The engine keeps the contract of every level: fixed `[E, R]` shapes, `reset(env_ids)` that leaves other envs bitwise unaffected, per-env clocks, the legacy `add_frames` and `step(t, snr, hid)` calls, and `NRConfig.edge` (the edge loop wraps the adaptive engine).

## The model

Every env has one authoritative level at any time. Its queued messages live in that level's FIFO, and the other level's rows for that env are empty. Each control step runs the cheap instance on all E envs and the expensive instance on its own rows, then merges the two step dicts per env. Poses go through the cheap instance's radio, and both levels receive the same SNR. Because both instances are seeded alike (`rng="engine"`), this radio is the one a plain engine of either level would draw.

Switches happen between two control steps. After step t, the engine computes each env's indicator from the merged step dict and decides the target level for step t + 1, then moves the envs whose target changed.

| Indicator | Per env, after each step | Unit |
|:---|:---|:---|
| `backlog` (default) | queued bytes, averaged over robots | bytes per robot |
| `contention` | share of robots with a non-empty queue | 0 to 1 |
| `offered` | EWMA (weight `offered_ewma`) of the bytes submitted per step, averaged over robots | bytes per robot per step |

An env moves up when its indicator reaches `up_threshold` and moves down when it falls below `down_threshold` (default half the up threshold). After a switch it stays on the new level for at least `min_dwell_steps` steps. `up_threshold = 0` keeps every env on the expensive level for the whole run, and `up_threshold = inf` keeps every env on the cheap level. `decision_period` evaluates switches only every n steps.

Supported pairs: the cheap level is `L0`, `L0DR`, `L05`, `L05Q` or `L1`, each on any of its backends. The expensive level is `L1` or `L2-legacy` on the single legacy cell, on any of their backends, or the NR engine `L2` with one cell, uplink only and no traffic models, on its reference backend. The multi-cell `NetSlotMC` has no handoff rule yet (see [Limits](#limits)).

### Two layouts

**Mask** (`layout="mask"`). The expensive instance has E rows, where row e is env e, and it steps every env every step. Rows of envs on the cheap level hold empty queues, and their outputs are discarded. This layout is the simple, correct baseline, and it costs a full expensive step no matter how few envs need it.

**Subbatch** (`layout="subbatch"`, `active_budget=M`). The expensive instance has M rows, called slots. An env that moves up takes a free slot and keeps it until it moves down. Each env prefers its home slot `e mod M`, so with M = E every env sits in its own row. When more than M envs want the expensive level, the ones with the highest indicator get the slots and the others wait on the cheap level (`fidelity_stats()["denied"]` counts these requests). Only M rows pay for the expensive step. Moving an env is a masked gather and `torch.where` over fixed-shape slot tables (`slot_env [M]`, `env_slot [E]`), so `submit`, `step` and `reset` have no host sync in either layout (with the NR engine, a partial reset of a slot syncs once).

`layout="auto"` (the default) picks subbatch for static mode, with M equal to the static set, and for load mode whenever `active_budget` is set. `active_budget` takes an int or a share of E.

### Graph mode

The routing logic is a few hundred small kernels per step: gathers of the FIFO rows, masked writes, the slot allocation and the merge of the two step dicts. Run eagerly, their launch cost exceeds the GPU time of both levels at the sizes used here. With `graph=True`, the engine captures `submit`, `step` and `reset` in one CUDA graph each after two eager calls. The captured step runs the step regions of both levels (their `graph` or `triton` bodies, not their own graphs), the merge, the indicators and the switching. The captured reset is a masked reset of both levels that takes the same keyed draws as their own `reset`, so a partial reset costs one small mask upload and one graph replay. All routing state is kept in persistent buffers updated in place. `graph="auto"`, the default, turns graph mode on when the device is CUDA and both levels use a fast backend (`eager`, `graph`, `compile` or `triton`). Calls with an explicit `t` or `cur_hid` run eagerly on the same state. A change of mode (`set_fraction`, `set_mode`) makes the engine capture the step and the reset again. Graph mode is bitwise equal to the eager engine through switches, partial resets, poses input and mode changes (`tests/test_adaptive.py`).

### Handoff rules

A handoff moves one env's state from one level's rows to the other's at the boundary between two control steps.

| State | Rule |
|:---|:---|
| Queued messages | Move exactly. Every prototype level stores messages in the same compacted F-slot FIFO, so capture step, class, detection tag, remaining bytes and the lookup features (`f_nact`, `f_snr`, `f_own`) are copied as they are. Deadlines therefore stay exact: a message times out `timeout_steps` after its capture on whichever level it ends. |
| `L2-legacy` MAC, on entry | Steady-state values. The BSR equals the queued bytes, since the gNB of a backlogged robot has a recent buffer report, so no SR is pending. No HARQ process is in flight. The OLLA offset is the robot's own offset when it last left `L2-legacy` in this episode. A robot without one gets the mean of its env's remembered offsets, and an env without any gets the prior `olla_prior_db` (default −3 dB, the steady-state mean measured on `L2-legacy`). The PF average is the robot's delivered-bytes EWMA per UL slot, with the PF time constant of 100 slots and a floor of `PF_AVG_MIN`. The fading state is a fresh draw from its stationary distribution CN(0, 1) per subband, which is exact for the AR(1) process. Every value depends only on the env's own history, so envs stay independent. |
| `L2-legacy` MAC, on exit | Dropped. The queue keeps the bytes of undecoded HARQ transmissions, because `L2-legacy` removes bytes from a message only after a successful decode. |
| `L1` | Has no state beyond the FIFO. |
| `L2` (NR engine), on entry | The FIFO is laid out on a fresh byte stream from offset 0, with air bytes = remaining payload × air(size) / size, where air() adds the per-packet overhead. The MAC starts from its reset values (no HARQ process in flight, OLLA 0), except that the BSR equals the queued air bytes, the CSI equals the gain of a fresh stationary fading draw, and the PF average equals the delivered bytes per UL data slot. The env clock moves through `NREngine.epoch`. |
| `L2` (NR engine), on exit | A frame keeps the payload share of its air bytes above the RLC in-order pointer, and at least 1 byte while it is queued. Bytes of in-flight HARQ transmissions stay in the queue. |
| Delay levels (`L0` to `L05Q`), on entry | Each queued message has waited w = now − capture steps without delivery. Its delivery time is drawn from the level's delay distribution conditioned on delay > w, which is lognormal for `L0`/`L0DR` and the fitted quantile table of the message's own bin for `L05`/`L05Q`. Its loss uses the matching conditional probability p / (p + (1 − p)(1 − F(w))). |
| Resets | An env restarts on the level its mode assigns at zero load, with that level's exact reset state. |

**The entry transient.** The OLLA offset matters most. In `L2-legacy` it settles at about −3.1 dB, with a standard deviation of 1.5 to 2 dB across robots. It is nearly independent of load, lower for robots at low SNR, and persistent (correlation 0.8 over 50 control steps). A robot that enters at 0 dB transmits about 3 dB too aggressively, and its first transport blocks fail and wait for HARQ retransmissions. The OLLA steps (+0.05 / −0.45 dB per transmission) need several control steps to recover. A re-entry experiment measures this directly (`benchmarks/adaptive/transient.py`, arm `reentry-N`). All envs stay on `L2-legacy`, and every N steps each env is handed to the cheap level and straight back between two steps. The queue passes through exactly and the random streams stay those of pure `L2-legacy`, so only the MAC entry state differs. With E = 512, R = 16, load p = 0.3 and N = 5, entering at OLLA 0 moves the delay distribution by W1 = 0.43 steps and KS = 0.115, against a seed-to-seed noise floor of W1 = 0.017 and KS = 0.004. With the remembered offset the shift drops to W1 = 0.024 and KS = 0.016. What remains comes from the PF-average estimate: keeping the old PF average as well gives W1 = 0.004 and KS = 0.001, and keeping every MAC field reproduces pure `L2-legacy` exactly. `min_dwell_steps` (default 5) keeps envs from paying the remaining transient repeatedly. A down-switch hands `L1` a queue that the fluid model serves from the next slot on, without the grant delay or retransmissions that `L2-legacy` would add. This is the cheap level's own bias applied to those bytes.

### Randomness and exactness

The adaptive engine requires `NRConfig.rng = "engine"`, the default. Both instances share the seed, and every draw is keyed by env id, episode and call count. Handoff draws (fading and conditional delays) use their own channel of the same counter streams. Consequently:

* `up_threshold = 0` is bitwise equal to a plain engine of the expensive level, and `up_threshold = inf` to one of the cheap level, in both layouts, through partial resets, with SNR or poses as input (tests on CPU, and on GPU for `graph` and `triton`).
* In the subbatch layout each slot's random keys are set to those of its env, so the subbatch run is bitwise equal to the mask run while envs switch. This holds on the GPU for the reference and `graph` backends, for any slot placement. On the CPU it holds for M = E, where every env sits in its home row. With M < E the CPU run agrees up to float rounding, because the CPU kernels of the prototype engine round transcendental functions differently in the vectorized body and in the scalar tail of a tensor, so a row's position and the tensor size change the last bit. The `triton` kernel of `L2-legacy` keys its draws by row. There a slot run by another env draws from an independent stream (hashed episode key) instead of the one the mask layout would use, so the triton subbatch run is statistically equivalent but not bitwise equal.
* The `graph` backends of both levels are bitwise equal to the reference adaptive engine, and `reset`, `submit` and `step` raise no host synchronization under `torch.cuda.set_sync_debug_mode("error")` for `graph` and `triton` in both layouts.
* The NR engine `L2` draws its stepping randomness from the global torch RNG. With it as the expensive level, `up_threshold = 0` equals a plain `L2` engine bitwise under the same global seed, and `up_threshold = inf` equals the cheap level.

## Static mixed fidelity for training

With `mode="static"` and `fraction=f`, a fixed random subset of round(f·E) envs runs on the expensive level for the whole run, and the other envs run on the cheap level. `set_fraction(f)` changes the subset between iterations, and the subsets are nested, so raising f only adds envs.

**What the policy optimizes.** The policy does not observe which level an env runs on, and every env contributes its samples to each update. The training distribution is therefore the mixture MDP, in which the network is the expensive level with probability f and the cheap level otherwise, and the plain batch average is an unbiased gradient estimate of the mixture objective

J_mix(π) = f · J_exp(π) + (1 − f) · J_cheap(π).

Let ε = sup over policies of |J_cheap(π) − J_exp(π)| be the cheap level's worst-case error on the task objective. Then |J_mix − J_exp| ≤ (1 − f) ε for every policy, and the policy that maximizes J_mix loses at most 2 (1 − f) ε of expensive-level return against the best expensive-level policy. Mixing shrinks the cheap level's bias by the factor (1 − f), while the variance of the gradient estimate stays that of all E envs rather than of the f·E expensive ones.

**Cost against accuracy.** At a fixed compute budget per step, pure expensive training affords E' = E · (c_cheap (1 − f) + c_exp f) / c_exp envs, where c is the cost per env-step. With per-env gradient variance σ², the mixed batch has mean squared error of roughly (1 − f)² ‖Δ‖² + σ²/E, where Δ is the gradient gap between the levels, against σ²/E' for pure expensive training. Mixing pays when the bias term is smaller than the variance it saves, that is, when the cheap level is close to the expensive one on the task and the expensive level is much more costly per env. Since ‖Δ‖ is unknown, the anchor group provides the check.

**The anchor.** The expensive envs form a random sample of the same envs under the reference model, so their group mean of any per-env statistic (episode return, delay quantiles, drop rate) estimates its expensive-level value for the current policy without bias. The difference between the cheap and the expensive group means estimates J_cheap − J_exp at the current policy. This estimate lower-bounds ε, and a large or growing gap says the policy has found behavior on which the cheap level is wrong. `out["fidelity"]` marks the groups. It also allows a reweighted update λ·ĝ_exp + (1 − λ)·ĝ_cheap, which trades bias for variance continuously: λ = f is the plain average, and λ = 1 is unbiased for the expensive objective and uses only the anchor's samples.

Keep the assignment random, as the default permutation does, and keep the fidelity flag out of the observation. Otherwise the policy can condition on the level and learn two behaviors.

## Curriculum

```python
cur = FidelityCurriculum.step_at(net, k)                  # L1 for iterations 0 .. k-1, L2-legacy from k on
cur = FidelityCurriculum(net, [(0, 0.0), (200, 0.25), (400, 1.0)])   # or a callable it -> share
for it in range(n_iterations):
    cur(it)                                               # same as cur.on_iteration(it)
    ...                                                   # rollout and update
```

Each change of the scheduled share is one `set_fraction` call. The switch happens between two control steps through the handoff above, not through a reset, so running episodes continue. Use the mask layout, or a budget at least as large as the final share, since the subbatch layout caps the share at its budget.

## Accuracy against cost

The results in this section come from `benchmarks/adaptive/sweep.py` on the lab box (RTX 4090, shared with other jobs), with `L1` and `L2-legacy` on the `triton` backend.

RESULTS_PLACEHOLDER

## Choosing thresholds

RULES_PLACEHOLDER

## Limits

* The NR engine `L2` is supported as the expensive level with one cell, uplink only and no traffic models, on its reference backend and without graph mode. Its downlink queue, the traffic models' per-message extras and the multi-cell association have no handoff rule yet, and neither has the multi-cell `NetSlotMC`.
* The OLLA prior of the NR engine `L2` is its reset value 0, since its steady-state offset depends on the configured BLER target and tables and has not been measured here. Set `olla_prior_db` when it is known.
* The `backlog` indicator counts air bytes (payload plus overhead) on `L2` and payload bytes on the prototype levels.
* The surrogate levels `TR`, `GE`, `QA` and `NN` are not supported as the cheap level, since their hidden state (the replayed trace position, the Markov state and the learned history) has no defined counterpart in the expensive level.
* With the subbatch layout, an env whose request is denied stays on the cheap level while the budget is full. Size the budget from the share of envs that the workload congests (see the tables).
