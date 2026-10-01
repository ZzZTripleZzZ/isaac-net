# Adaptive and mixed fidelity

`AdaptiveEngine` (`isaac_net/core/adaptive.py`) runs a cheap level and an expensive level behind one engine API and decides, per env, which of them simulates each control step. Three uses share the same machinery:

1. **Static mixed fidelity.** A fixed random subset of the envs, for example 10%, runs on `L2-legacy` for the whole run, and the other envs run on `L1`.
2. **Load-triggered switching.** An env runs on the cheap level while its load indicator stays low and moves to the expensive level while the cell is congested, which is where the cheap levels diverge from `L2-legacy`.
3. **Curriculum.** A schedule over training iterations, such as `L1` for the first k iterations and `L2-legacy` afterwards, applied through a callback.

```python
from isaac_net.core import NRConfig, Requests
from isaac_net.core.adaptive import FidelityConfig, FidelityCurriculum, make_adaptive

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

The routing logic is a few hundred small kernels per step: gathers of the FIFO rows, masked writes, the slot allocation and the merge of the two step dicts. Run eagerly, their launch cost exceeds the GPU time of both levels at the sizes used here. With `graph=True`, the engine captures `submit`, `step` and `reset` in one CUDA graph each after two eager calls. The captured step runs the step regions of both levels (their `graph` or `triton` bodies, not their own graphs), the merge, the indicators and the switching. The captured reset is a masked reset of both levels that takes the same keyed draws as their own `reset`, so a partial reset costs one small mask upload and one graph replay. All routing state is kept in persistent buffers updated in place. `graph="auto"`, the default, turns graph mode on when the device is CUDA and both levels use a fast backend (`eager`, `graph`, `compile` or `triton`). Calls with an explicit `t` or `cur_hid` run eagerly on the same state. A change of mode (`set_fraction`, `set_mode`) makes the engine capture the step and the reset again. With `decision_period` n > 1, the engine captures two steps, one with the switching logic and one without, and replays the first every n-th step, so the routing work is paid once per n steps. Graph mode is bitwise equal to the eager engine through switches, partial resets, poses input and mode changes (`tests/test_adaptive.py`).

### Handoff rules

A handoff moves one env's state from one level's rows to the other's at the boundary between two control steps.

| State | Rule |
|:---|:---|
| Queued messages | Move exactly. Every prototype level stores messages in the same compacted F-slot FIFO, so capture step, class, detection tag, remaining bytes and the lookup features (`f_nact`, `f_snr`, `f_own`) are copied as they are. Deadlines therefore stay exact: a message times out `timeout_steps` after its capture on whichever level it ends. |
| `L2-legacy` MAC, on entry | Steady-state values. The BSR equals the queued bytes, since the gNB of a backlogged robot has a recent buffer report, so no SR is pending. No HARQ process is in flight. The OLLA offset is the robot's own offset when it last left `L2-legacy` in this episode. A robot without one gets the mean of its env's remembered offsets, and an env without any gets the prior `olla_prior_db` (default −3 dB, the steady-state mean measured on `L2-legacy`). The PF average is the robot's PF-average estimate: its actual PF average when it last left `L2-legacy` (the reset value after a reset), continued while the env is on the cheap level as an EWMA, with the PF time constant of 100 slots, of the bytes the cheap level served (delivered bytes on a delay level), with a floor of `PF_AVG_MIN`. The fading state is a fresh draw from its stationary distribution CN(0, 1) per subband, which is exact for the AR(1) process. Every value depends only on the env's own history, so envs stay independent. |
| `L2-legacy` MAC, on exit | Dropped. The queue keeps the bytes of undecoded HARQ transmissions, because `L2-legacy` removes bytes from a message only after a successful decode. |
| `L1` | Has no state beyond the FIFO. |
| `L2` (NR engine), on entry | The FIFO is laid out on a fresh byte stream from offset 0, with air bytes = remaining payload × air(size) / size, where air() adds the per-packet overhead. The MAC starts from its reset values (no HARQ process in flight), except that the BSR equals the queued air bytes, the CSI equals the gain of a fresh stationary fading draw, and the OLLA offset and PF average follow the `L2-legacy` rules (per UL data slot, prior 0 dB). The env clock moves through `NREngine.epoch`. |
| `L2` (NR engine), on exit | A frame keeps the payload share of its air bytes above the RLC in-order pointer, and at least 1 byte while it is queued. Bytes of in-flight HARQ transmissions stay in the queue. |
| Delay levels (`L0` to `L05Q`), on entry | Each queued message has waited w = now − capture steps without delivery. Its delivery time is drawn from the level's delay distribution conditioned on delay > w, which is lognormal for `L0`/`L0DR` and the fitted quantile table of the message's own bin for `L05`/`L05Q`. Its loss uses the matching conditional probability p / (p + (1 − p)(1 − F(w))). |
| Resets | An env restarts on the level its mode assigns at zero load, with that level's exact reset state. |

**The entry transient.** The OLLA offset matters most. In `L2-legacy` it settles at about −3.1 dB, with a standard deviation of 1.5 to 2 dB across robots. It is nearly independent of load, lower for robots at low SNR, and persistent (correlation 0.8 over 50 control steps). A robot that enters at 0 dB transmits about 3 dB too aggressively, and its first transport blocks fail and wait for HARQ retransmissions. The OLLA steps (+0.05 / −0.45 dB per transmission) need several control steps to recover. A re-entry experiment measures this directly (`benchmarks/adaptive/transient.py`, arm `reentry-N`). All envs stay on `L2-legacy`, and every N steps each env is handed to the cheap level and straight back between two steps. The queue passes through exactly and the random streams stay those of pure `L2-legacy`, so only the MAC entry state differs. With E = 512, R = 16, load p = 0.3 and N = 5, entering at OLLA 0 moved the delay distribution by W1 = 0.43 steps and KS = 0.115, against a seed-to-seed noise floor of W1 = 0.017 and KS = 0.004. With the remembered offset the shift dropped to W1 = 0.024 and KS = 0.016. The rest came from the PF average: an estimate built from delivered bytes undercounts at heavy load, where bytes served for messages that later time out count in the PF average but are never delivered (W1 = 0.29 at p = 0.5). The continued PF-average estimate removed it. Keeping every MAC field reproduces pure `L2-legacy` exactly, which confirms that the experiment isolates the entry state. With the final rule the re-entry shift stays below the noise floor at every load, even with a handoff every step (see [The handoff itself](#the-handoff-itself)). A down-switch hands `L1` a queue that the fluid model serves from the next slot on, without the grant delay or retransmissions that `L2-legacy` would add. This is the cheap level's own bias applied to those bytes.

### Randomness and exactness

The adaptive engine requires `NRConfig.rng = "engine"`, the default. Both instances share the seed, and every draw is keyed by env id, episode and call count. Handoff draws (fading and conditional delays) use their own channel of the same counter streams. Consequently:

* `up_threshold = 0` is bitwise equal to a plain engine of the expensive level, and `up_threshold = inf` to one of the cheap level, in both layouts, through partial resets, with SNR or poses as input (tests on CPU, and on GPU for `graph` and `triton`).
* In the subbatch layout each slot's random keys are set to those of its env, so the subbatch run is bitwise equal to the mask run while envs switch. This holds on the GPU for any slot placement (tested with the `graph` backends). On the CPU it holds for M = E, where every env sits in its home row. With M < E the CPU run agrees up to float rounding, because the CPU kernels of the prototype engine round transcendental functions differently in the vectorized body and in the scalar tail of a tensor, so a row's position and the tensor size change the last bit. The `triton` kernel of `L2-legacy` keys its draws by row. There a slot run by another env draws from an independent stream (hashed episode key) instead of the one the mask layout would use, so the triton subbatch run is statistically equivalent but not bitwise equal.
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

The measurements come from `benchmarks/adaptive/` on the lab box (RTX 4090, 2026-09-30); `run_all.sh` reproduces them. The GPU was shared with other jobs at 94–100% utilization (all processes) throughout, and wall times per step changed by up to 2× between runs as that load changed. Every cost comparison below is therefore taken within one run, with all engines timed round-robin, and a second, contention-light view is given as GPU kernel time per step. The complete tables, including every threshold and budget, are in `benchmarks/adaptive/tables.md`.

**Workload.** Random send policies with no learning. E envs of R robots on the legacy cell, positions a random walk inside 5–60 m of the gNB, each robot sending one message per step with probability p (70% 4000 B, 30% 30000 B), and 0.5% of the envs reset per step. The positions keep every robot above about 5 dB SNR, so the backlog comes from load rather than from robots out of coverage. In the 150 m arena of the example task, far robots stay backlogged at any load, and a backlog indicator then tracks coverage instead of contention. In the `mixed` scenario each env alternates between a light regime (p = 0.03) and a heavy one (p uniform in 0.3–0.6), with mean dwell times of 50 and 25 steps, so about a third of the envs are congested at any time. The `uniform-p` scenarios hold one load for every env.

**Metrics.** Each run uses the same seed, poses, sends and resets, so runs differ only in the network model. The metrics are the per-message delay distribution of delivered messages, compared with pure `L2-legacy` by the Wasserstein-1 distance (W1, in control steps) and the Kolmogorov–Smirnov distance (KS), and the drop share (timed out over resolved) as the difference Δdrop from pure `L2-legacy`. Pure `L2-legacy` with another seed gives the noise floor. The runs have 400 steps, of which the last 350 are scored, and E = 4096, R = 16 unless stated otherwise.

### L1 and L2-legacy, both on `triton`

Mixed scenario, one run (accuracy and wall time per step, without and with the 0.5% resets):

| Run | Env-steps on `L2-legacy` | W1 (steps) | KS | Δdrop | ms / step | ms / step with resets |
|:---|---:|---:|---:|---:|---:|---:|
| pure `L2-legacy` (reference) | 100% | 0 | 0 | 0 | 2.20 | 2.49 |
| pure `L2-legacy`, other seed | 100% | 0.008 | 0.001 | −0.000 | | |
| pure `L1` | 0% | 1.030 | 0.118 | −0.024 | 1.81 | 2.04 |
| static, 10% | 10% | 0.934 | 0.107 | −0.021 | 2.45 | 2.98 |
| static, 25% | 25% | 0.781 | 0.089 | −0.018 | 2.72 | 3.30 |
| load, 1000 B, mask | 47% | 0.213 | 0.061 | −0.002 | 4.86 | 5.32 |
| load, 1000 B, budget 50% | 47% | 0.214 | 0.061 | −0.002 | 4.15 | 4.77 |
| load, 1000 B, budget 25% | 25% | 0.654 | 0.108 | −0.007 | 3.59 | 4.17 |
| load, 1000 B, budget 10% | 10% | 0.978 | 0.121 | −0.016 | 3.25 | 3.77 |
| load, 1000 B, mask, `decision_period=5` | 42% | 0.330 | 0.071 | −0.003 | 4.21 | 4.66 |
| load, 1000 B, budget 50%, `decision_period=5` | 42% | 0.331 | 0.071 | −0.003 | 3.35 | 3.99 |
| load, 1000 B, budget 25%, `decision_period=5` | 25% | 0.692 | 0.109 | −0.008 | 2.89 | 3.47 |
| load, 1000 B, budget 10%, `decision_period=5` | 10% | 0.983 | 0.121 | −0.016 | 2.61 | 3.14 |
| load, 4000 B, mask | 30% | 0.467 | 0.095 | −0.006 | | |
| load, 16000 B, mask | 9% | 0.953 | 0.122 | −0.017 | | |

"Env-steps on `L2-legacy`" is the share of env-steps the expensive level simulated. The threshold is the `backlog` indicator in bytes per robot, and "budget" is `active_budget` as a share of E. The load-4000 and load-16000 rows come from the full sweep (another run), so their wall times are not comparable and are omitted.

By load, with the same levels (full sweep):

| Load | pure `L1`: W1 / KS / Δdrop | static 25%: W1 / KS / Δdrop | load 1000 B, mask: share, W1 / KS / Δdrop | load 1000 B, budget 25%: W1 / KS / Δdrop |
|:---|:---|:---|:---|:---|
| p = 0.05 | 0.772 / 0.396 / −0.009 | 0.582 / 0.298 / −0.007 | 17%, 0.573 / 0.347 / −0.003 | 0.574 / 0.347 / −0.003 |
| p = 0.2 | 0.917 / 0.264 / −0.019 | 0.692 / 0.199 / −0.015 | 65%, 0.262 / 0.122 / −0.001 | 0.691 / 0.224 / −0.008 |
| p = 0.5 | 1.314 / 0.116 / −0.033 | 0.991 / 0.088 / −0.025 | 97%, 0.079 / 0.010 / −0.001 | 1.185 / 0.108 / −0.025 |
| mixed | 1.030 / 0.118 / −0.024 | 0.781 / 0.089 / −0.018 | 47%, 0.213 / 0.061 / −0.002 | 0.654 / 0.108 / −0.007 |

What these runs show:

* **Where `L1` is wrong.** `L1` has no scheduling-request delay, grant delay or HARQ, so its whole delay distribution sits early at every load (median 0.15 against 0.42 steps at p = 0.05, KS 0.40). Under load it also misses the queueing tail and the drops (Δdrop −0.033 at p = 0.5). Load switching fixes the second error and not the first. At p = 0.05 it moves only the few congested env-steps (17%) and leaves KS at 0.35.
* **Accuracy per expensive share.** Load switching puts the expensive level where the error is. In the mixed scenario, 47% of env-steps on `L2-legacy` bring W1 from 1.03 to 0.21 and the drop share to within 0.002 of the reference, while a static 25% mix reaches only 0.78. The static mix removes error roughly in proportion to its share, as the (1 − f) bound predicts. A budget below the share the threshold asks for caps the gain: 25% gives 0.65, about what a static 25% mix gives.
* **Cost on `triton`.** The fused `L2-legacy` kernel costs about as much GPU time as the `L1` kernel (2.2 against 1.8 ms per step here, and 2.2 against 1.8 ms of kernel time). Running both levels therefore always costs more than pure `L2-legacy`. Static 10% costs 1.11× pure `L2-legacy`, and load switching costs 1.5–2.2×, depending on the budget. With this pair, adaptive fidelity buys accuracy for `L1`-based training rather than saving time over `L2-legacy`.
* **Subbatch against mask.** At the same threshold the subbatch layout runs the switching engine 1.17× (budget 50%), 1.35× (25%) and 1.50× (10%) faster than the mask layout. With `decision_period=5` the routing runs every fifth step, and the subbatch then takes 3.35, 2.89 and 2.61 ms against 4.86 ms for the mask layout at period 1. The longer period costs some accuracy, because an env reacts up to four steps late (W1 0.33 against 0.21 at the full share).

### A fitted `L05Q` as the cheap level

`L05Q` tables fitted from `L2-legacy` rollouts (`benchmarks/adaptive/fit_l05q.py`, 1.05 M delivered and 37 k timed-out messages from a separate workload seed, E = 1024) on the `graph` backend, with `L2-legacy` on `triton`:

| Scenario | pure `L05Q` | static 10% | static 25% | load 1000 B, budget 25% | load 1000 B, mask (share) |
|:---|:---|:---|:---|:---|:---|
| mixed | 0.149 / 0.035 / −0.001 | 0.133 / 0.031 / −0.001 | 0.110 / 0.026 / −0.001 | 0.185 / 0.022 / −0.001 | 0.055 / 0.007 / +0.000 (60%) |
| p = 0.2 | 0.200 / 0.044 / −0.006 | 0.180 / 0.040 / −0.005 | 0.150 / 0.033 / −0.004 | 0.156 / 0.032 / −0.003 | 0.078 / 0.011 / +0.000 (82%) |
| p = 0.5 | 0.404 / 0.077 / −0.002 | 0.363 / 0.069 / −0.002 | 0.304 / 0.058 / −0.002 | 0.331 / 0.062 / +0.001 | 0.034 / 0.004 / +0.000 (97%) |

Entries are W1 (steps) / KS / Δdrop. A fitted delay level is already close: its drop share matches within 0.006 and its W1 is 0.15–0.40 steps. A static mix improves it in proportion to its share. Load switching with a budget of 25% is no better than the static mix, and in the mixed scenario it is slightly worse (W1 0.185 against 0.110). Messages that move between a queueing level and a delay level mix two kinds of dynamics, so with a fitted delay level the static mix is the better choice. Wall time: pure `L05Q` 1.76 ms, static 25% 2.68 ms and pure `L2-legacy` 2.21 ms per step in that run.

### An expensive level that costs more: `L2-legacy` on `graph`, the NR engine

With `L2-legacy` on the `graph` backend, whose captured graph replays several thousand small kernels, the expensive level is the dominant cost (mixed scenario, `L1` on `triton`). The first two columns come from the sweep at E = 1024 with a threshold of 4000 B, and the kernel times from `kernel_time.py` at E = 4096 with a threshold of 1000 B:

| Run | E = 1024: W1 / KS / Δdrop | E = 1024: ms / step | E = 4096: kernel ms / step |
|:---|:---|---:|---:|
| pure `L2-legacy` (`graph`) | 0 / 0 / 0 | 13.98 | 27.9 |
| pure `L1` (`triton`) | 1.018 / 0.118 / −0.024 | 0.68 | 1.8 |
| load, mask | 0.460 / 0.093 / −0.006 | 14.86 | 30.6 |
| load, budget 50% | 0.460 / 0.093 / −0.006 | 11.27 | 21.6 |
| load, budget 25% | 0.622 / 0.105 / −0.008 | 10.94 | 17.1 |
| load, budget 10% | 0.960 / 0.120 / −0.016 | 11.26 | 13.3 |
| static 10% | 0.917 / 0.107 / −0.022 | 10.88 | 12.8 |

Here the subbatch saves real work. GPU kernel time falls from 30.6 ms (mask) to 21.6, 17.1 and 13.3 ms at budgets of 50, 25 and 10%. Wall time falls only 1.24–1.33× against pure `L2-legacy`, because the replay of the `graph` backend is bound by its kernel count rather than by the number of rows.

The NR engine (`L2`, reference backend, E = 256, R = 8, mixed scenario) is launch-bound at about 340 ms per step on this GPU (90 ms of kernel time), independent of its row count. Pure `L2` costs 338 ms per step, load switching (4000 B, 13% share) 343 ms in the mask layout and 341 ms with a 25% budget, and static 10% 332 ms. The accuracy gains follow the same pattern as with `L2-legacy`: W1 0.431 for pure `L1`, 0.307 for load switching and 0.393 for static 10%, against a noise floor of 0.041. The time savings will come with a batched fast backend for the NR engine.

### The handoff itself

`benchmarks/adaptive/transient.py` (E = 2048, R = 16) separates the handoff from the model difference.

*Re-entry* keeps every env on `L2-legacy` and hands it to the cheap level and straight back every N steps, so only the MAC entry state differs from pure `L2-legacy`. W1 / KS against pure `L2-legacy`:

| Load | noise floor | N = 1 | N = 5 | N = 25 |
|:---|:---|:---|:---|:---|
| p = 0.1 | 0.025 / 0.003 | 0.010 / 0.003 | 0.002 / 0.001 | 0.001 / 0.000 |
| p = 0.3 | 0.029 / 0.003 | 0.018 / 0.003 | 0.004 / 0.001 | 0.001 / 0.000 |
| p = 0.5 | 0.045 / 0.003 | 0.019 / 0.002 | 0.005 / 0.000 | 0.001 / 0.000 |

The entry rule is below the noise floor even when it runs every step. Two earlier versions of the rule were measured on the same experiment (E = 512, p = 0.3 and 0.5, N = 5). Entering with an OLLA offset of 0 gave W1 0.43 and KS 0.12, and a PF average estimated from delivered bytes gave W1 0.29 at p = 0.5. The offset and PF average carried over from the robot's own history fixed both.

*Toggle-N* alternates a random half of the envs and its complement between the levels every N steps. It keeps the same 50% share of env-steps as a static 50% mix, so the difference from the static mix is the effect of switching. W1 against the static mix:

| Cheap level, load | static 50% vs pure `L2-legacy` | N = 1 | N = 5 | N = 25 | N = 50 |
|:---|:---|:---|:---|:---|:---|
| `L1`, p = 0.1 | 0.405 | 0.133 | 0.088 | 0.016 | 0.009 |
| `L1`, p = 0.3 | 0.570 | 0.090 | 0.100 | 0.061 | 0.038 |
| `L1`, p = 0.5 | 0.668 | 0.140 | 0.083 | 0.084 | 0.072 |
| `L05Q`, p = 0.1 | 0.037 | 0.131 | 0.085 | 0.029 | 0.012 |
| `L05Q`, p = 0.3 | 0.213 | 0.453 | 0.280 | 0.067 | 0.034 |
| `L05Q`, p = 0.5 | 0.205 | 0.802 | 0.577 | 0.155 | 0.079 |

With `L1`, switching changes the delay distribution by at most 0.14, much less than the model difference itself (0.4–0.7). Messages that switch mid-flight get part of each level's service. With the fitted `L05Q`, frequent switching is harmful: switching every 1–5 steps moves the distribution by more than the model difference. A message that enters `L05Q` after waiting takes a delay conditioned on its wait but drawn from the table bin of its arrival features, and repeated entries keep re-conditioning it. With 25–50 steps between switches the effect falls to the level of the model difference.

## Choosing thresholds

**First decide whether a mix pays.** The mix saves time only when the expensive level costs clearly more per env than the cheap one. That holds for `L2-legacy` on `graph` against `L1` on `triton`, and for the NR engine once it has a batched backend. With `L2-legacy` on `triton` it does not hold, since the two kernels cost about the same, and pure `L2-legacy` on `triton` is then both cheaper and exact. The mix still helps a run that must stay on a cheap level, since it moves that run's network statistics toward `L2-legacy`, and the anchor group measures the remaining gap.

**Static or load.** With a fitted delay level (`L05`, `L05Q`) as the cheap level, use a static mix: it improves the cheap level in proportion to its share and never switches a message between the two kinds of dynamics. With `L1`, use load switching when the congested envs are a minority, since it places the expensive level where `L1` is wrong. The mixed scenario reaches W1 0.21 at a 47% share, against 0.78 for a static 25% mix. At uniformly light load, `L1`'s remaining error is its missing access and HARQ latency, which no threshold addresses. The static mix, or `L05Q`, is the remedy there.

**Indicator.** Use `backlog` with a cheap level that serves bytes (`L1`) or queues messages (`L05Q`, whose waiting messages count as backlog). Use `offered` with `L0` or `L0DR`: their messages leave at their sampled delivery time, so their backlog does not rise with contention. `contention` (the share of robots with a non-empty queue) reacts to how many robots transmit at once rather than to how much they send.

**Threshold.** The backlog is in bytes per robot. At 1000 B per robot (a quarter of a small message per robot) the engine catches congestion early, and the share it puts on the expensive level follows the load: 17% at p = 0.05, 47% in the mixed scenario, 65% at p = 0.2 and 97% at p = 0.5. At 4000 B it switches later (30% in the mixed scenario, W1 0.47). At 16000 B it misses most congestion (9%, W1 0.95). Start at about a quarter of the smallest message size, and read `out["fidelity"].float().mean()` to see the share the threshold produces on the task's own traffic. `up_threshold = 0` and `inf` reproduce the pure levels exactly and serve as sanity checks.

**Budget.** Size `active_budget` to the share the threshold asks for, measured with the mask layout or read from `fidelity_stats()`. A budget below it caps the accuracy at about that of a static mix of the budget's size, since the most congested envs keep the slots and the others wait (`denied`). A budget above it costs nothing extra in accuracy and saves little time, because the subbatch cost follows the budget, not the share in use.

**Hysteresis and dwell.** The default `down_threshold` of half the up threshold and `min_dwell_steps=5` keep an env on one level for tens of steps in the mixed scenario, where each env switches up about once every 60 steps. With a delay level as the cheap level, raise `min_dwell_steps` to 25 or more (see the toggle table).

**Decision period.** In graph mode `decision_period` makes the routing run only every n steps, and n = 5 cut the step cost of the switching engine by 13–20% at the price of reacting up to n − 1 steps late.

## Limits

* The NR engine `L2` is supported as the expensive level with one cell, uplink only and no traffic models, on its reference backend and without graph mode. Its downlink queue, the traffic models' per-message extras and the multi-cell association have no handoff rule yet, and neither has the multi-cell `NetSlotMC`.
* The OLLA prior of the NR engine `L2` is its reset value 0, since its steady-state offset depends on the configured BLER target and tables and has not been measured here. Set `olla_prior_db` when it is known.
* The `backlog` indicator counts air bytes (payload plus overhead) on `L2` and payload bytes on the prototype levels.
* The surrogate levels `TR`, `GE`, `QA` and `NN` are not supported as the cheap level, since their hidden state (the replayed trace position, the Markov state and the learned history) has no defined counterpart in the expensive level.
* With the subbatch layout, an env whose request is denied stays on the cheap level while the budget is full. Size the budget from the share of envs that the workload congests (see the tables).
