# Differentiable network models

`isaac_net.core.diff` holds differentiable versions of two of the fluid levels, so that gradients can flow from network KPIs back to the decisions that caused them. This is exploratory research infrastructure. It is not an engine level: `make_engine` does not build it, it has no CUDA-graph backend, and the engine levels do not depend on it.

| What | Module | Relaxes | Gradients with respect to |
|:---|:---|:---|:---|
| **L1D** | `DiffFluid(mode="L1")` | `L1` (`NetFluid`): equal subband shares among backlogged robots, spectral efficiency from SNR, FIFO byte service over the UL slots of a control step, buffer overflow, application timeout | per-robot send probability, message size, transmit power, position (through the radio model), deadline |
| **QAD** | `DiffFluid(mode="QA")` | `QA` (`NetQA`): one processor-sharing interval per control step with the PF multi-user-diversity gain, the power-headroom cap and the SR offset | same |
| Neural proxy (recipe) | `diff.proxy` | an MLP fitted to per-robot KPIs of `L2-legacy` rollouts | send probability, message size, SNR (so power and position) |

KPIs (per robot `[E,R]` and as batch means): mean delay of the delivered messages, delivery ratio, age of information (AoI) at the end of each control step, and transmit energy.

## Quick start

```python
import torch
from isaac_net.core.diff import DiffFluid, rollout

E, R, T = 64, 8, 100
p = torch.full((E, R), 0.4, requires_grad=True)          # send probability per robot
B = torch.full((E, R), 12_000.0, requires_grad=True)     # message size, bytes
tx = torch.full((E, R), 23.0, requires_grad=True)        # transmit power, dBm
pos = (40 + 100 * torch.rand(E, R, 2)).requires_grad_() # positions, m (gNB at the origin)

net = DiffFluid(E, R, "cpu", mode="L1", tau=0.05)
kpi = rollout(net, T, p, B, pos=pos, tx_dbm=tx, warmup=10)
loss = kpi["mean_delay"] + 0.5 * kpi["mean_aoi"] + 50.0 * kpi["mean_energy_j"]
loss.backward()                                          # p.grad, B.grad, tx.grad, pos.grad
```

`net.step(send, msg_bytes, snr_db=..., pos=..., tx_dbm=..., deadline=...)` advances one control step and returns the per-step KPIs. `reset(env_ids)` re-initializes some envs without touching the others or their gradient history, and `detach()` truncates backpropagation through time. `reference.discrete_rollout` runs a real engine level with binary sends and returns the same KPIs, which is how the relaxation is checked.

## How the models are made differentiable

**State layout.** A robot enqueues at most one message per control step and a message leaves after at most `timeout` steps, so messages can be indexed by age instead of by FIFO position. Slot `a` of a `[E, R, timeout]` buffer holds the message captured `a` steps ago, and FIFO order is age order. The discrete levels keep a compacted FIFO, which is a permutation and has no gradient. Here the buffer shifts by one slot per step. Each slot carries the expected remaining bytes `x` and a mass `w`, the probability-like weight of the message still being queued. For a sent message at temperature 0, `w = 1`.

**Sends.** The send input is a weight in [0, 1]. With the probability itself (mean field), a message of size `B` sent with probability `p` enters as `p·B` expected bytes with mass `p`. With `relaxed_bernoulli(p, u, lam)`, the send is a binary-concrete sample from fixed uniforms `u`, which is near 0 or 1 and gives a pathwise gradient of the relaxed stochastic model.

**Relaxations.** One temperature `tau` controls every relaxation, and `tau = 0` runs the exact hard operations.

| Hard operation | Relaxation (`relax.py`) | Temperature kind |
|:---|:---|:---|
| FIFO completion: message `a` is done once the bytes served since the step began reach the bytes queued up to and including it | sigmoid of the log ratio `(log S_k − log cum_a) / tau`; the expected finish time weights each UL slot by the increase of this indicator | relative |
| FIFO remainder `max(cum − S, 0)` | `cum − smin(cum, S)` with the log-domain soft minimum `smin(a, b) = exp(−tau·logsumexp(−log a/tau, −log b/tau))`, exact at `S = 0` and for `S ≫ cum` | relative |
| Backlog indicator | 1 minus the completion indicator of the youngest queued byte | relative |
| Share `S / max(n, 1)`, power split `max(share, 1)`, SE cap `min(se, 5.5)` | `smax`, `smin` in the log domain | relative |
| Buffer overflow (at most `F` messages queued) | `sigmoid((F − 0.5 − n) / tau_n)` on the soft message count | additive, in messages |
| Application timeout | the deadline is a continuous per-robot input; at the end of the step a message of age `a` survives with weight `sigmoid((deadline − 0.5 − (a + 1)) / tau_n)` relative to age `a − 1` | additive, in control steps |
| QA power-headroom cap `floor(·)` | sum of sigmoid steps at the integers | additive |
| QA PF gain `H(n)` at an integer `n` | harmonic number at a real `n`, `digamma(n + 1) + γ` | none |
| QA SR offset when the queue was empty | soft "was empty" gate | relative |

A relative temperature acts on log ratios, so the smoothing is proportional to the quantity. A 4 kB and a 30 kB message see the same relative smoothing, which a fixed temperature in bytes would not give. At the crossing point `smin(a, a) = a·2^−tau`, so `tau = 0.05` costs about 3.4 % there and nothing far from it. `tau_n` defaults to `tau`.

**Consistency with the discrete levels.** At `tau = 0` the relaxed models are the discrete levels. `tests/test_diff.py` checks that per-robot delivered counts, delay sums, timeouts, buffer overflows and AoI of L1D equal those of `L1`, and those of QAD equal those of `QA`, including an overload case where the buffer and the deadline both bind. Mass is conserved at every temperature: offered = delivered + timed out + overflow + still queued.

**Energy.** The discrete levels have no energy model, so energy is defined here. A backlogged robot transmits at `tx_dbm` for the fraction `min(share, 1)` of each UL slot in L1D, and of the busy part of the control step in QAD. At `tau = 0` this is the energy of the discrete fluid schedule.

## Checks and measurements

Benchmarks live in `benchmarks/diff/` (`run_all.sh` runs all of them; results in `benchmarks/diff/results/`).

All numbers below were measured on the lab box (WSL, RTX 4090, shared and heavily loaded at the time). The full-scale benchmark runs (`run_all.sh`) are **pending**: only the checks and the small smoke runs listed here have completed.

**Equivalence at `tau = 0`** (`tests/test_diff.py`, E = 8, R = 6, T = 50, normal load and an overload with F = 4 and a 6-step timeout). Per-robot delivered count, timeouts, overflow and AoI of L1D equal those of `L1`, and those of QAD equal those of `QA` (max difference 0 at 1e-4 tolerance, delay sums within 1e-4 relative).

**Convergence as `tau → 0`** (E = 16, R = 8, T = 60, binary sends with p = 0.4, sizes 4 kB or 30 kB, SNR 10–30 dB). Discrete mean delay: 0.382 control steps (L1) and 0.638 (QA).

| tau | L1D mean delay | L1D mean AoI | QAD mean delay | QAD mean AoI |
|---:|---:|---:|---:|---:|
| 0 (= discrete) | 0.382 | 2.501 | 0.638 | 2.725 |
| 0.2 | 0.502 | 2.818 | 0.773 | 3.082 |
| 0.1 | 0.448 | 2.682 | 0.710 | 2.933 |
| 0.05 | 0.420 | 2.602 | 0.687 | 2.834 |
| 0.02 | 0.398 | 2.544 | 0.658 | 2.765 |
| 0.01 | 0.390 | 2.521 | 0.643 | 2.741 |
| 0.005 | 0.385 | 2.507 | 0.642 | 2.735 |
| 0.002 | 0.383 | 2.502 | 0.639 | 2.728 |

The error falls roughly linearly in `tau` (about 10 % on the mean delay at `tau = 0.05`, under 0.5 % at 0.002). The delivery ratio error is below 0.1 % at every `tau` in the table. The relaxation biases delay upward. `benchmarks/diff/convergence.py` produces the same curves at E = 64, T = 200 and three loads (pending).

**Gradient correctness.** `torch.autograd.gradcheck` passes in float64 for delay, delivery, AoI and energy with respect to send probability, message bytes, transmit power and positions (E = 2, R = 3, T = 4, K = 4, small buffer and deadline), for both L1D and QAD. At E = 32, R = 8, T = 60, K = 40 the autograd directional derivative of a KPI mix matches central finite differences to better than 1e-4 relative for every input. The finite-difference error shrinks quadratically with the step (at `tau = 0.05`: 1.7 % at step 1e-2, 3.9e-6 at 1e-5), which confirms the gradient and shows the curvature of the relaxed objective. Signs are physical: more bytes or a higher send probability raise delay, more power lowers delay and raises energy.

**Sensitivity against L2-legacy** (`benchmarks/diff/sensitivity.py`): pending at full scale. A smoke run at E = 16, T = 60, two seeds, gave for d(mean AoI)/d(tx dB) at p = 0.5 a QAD relaxed-Bernoulli gradient of −0.79 ± 0.07 (AoI falls with power), but the matching L2-legacy finite differences were cut off by the smoke timeout, so no sign or magnitude comparison with L2-legacy is reported yet.

**Optimization demo** (`opt_rates.py`, smoke at E = 8, T = 40, 10 gradient iterations against 32 random candidates per env). True cost on the discrete L1 with held-out sends: 5.46 after 10 gradient steps against 15.4 for the best of 32 random decisions per env. On L2-legacy the gradient solution raised delivery from 0.77 to 0.81 and lowered AoI from 7.2 to 6.4 control steps, with delay 2.39 → 2.49 control steps. This is a smoke run with a tiny budget; the full comparison (64 envs, 150 iterations, 256 random candidates) is pending. The placement toy (`placement.py`) has not completed yet.

**Neural proxy** (`nn_proxy.py`): the recipe runs end to end in the CPU smoke test (collect from L2-legacy, fit, per-robot gradients). Fit quality and its gradient comparison with L2-legacy finite differences are pending.

**Cost.** On the loaded lab box, one L1D rollout of T = 60 steps took 10.8 s forward and 11.1 s backward at E = 16 on CPU (22.1 s and 16.5 s at E = 64), and 29–40 s forward on the shared GPU; an `L2-legacy` rollout of the same length took 27–91 s. The box had a load average near 60 on 32 cores, so these are upper bounds. A comparison of the eager slot loop with `compile=True` (`torch.compile` of the per-slot body) is pending.

## Temperature guidance

- `tau = 0.05` is a reasonable default for optimization. The KPI bias against the discrete model is a few percent (see the convergence table), and the gradients are informative across a wide range of decisions.
- Smaller `tau` reduces the bias roughly linearly but makes the objective stiffer: its curvature grows like `1/tau`, the gradient becomes large near completion and deadline boundaries and nearly zero elsewhere, and finite-difference checks need smaller steps (the tests use `1e-6` in float64).
- Larger `tau` (0.1 to 0.3) smooths more and helps when the starting point is far from good, at a bias of 10 % or more. Annealing `tau` from 0.2 to 0.02 during an optimization is a reasonable schedule (`net.tau` can be changed between steps).
- Score the result on the discrete model: L1D at `tau = 0` with binary sends is the discrete L1 level.
- Use float64 for gradient checks. float32 is fine for optimization.

## Limits

- **The discrete scheduling and HARQ are not differentiated.** L1D and QAD are fluid models with the L1 and QA physics. They have no subband-by-subband PF allocation, no SR/BSR timeline in L1D, no fading, no BLER draws, no OLLA and no HARQ or RLC retransmission. Their gradients are gradients of these fluid models, not of `L2-legacy` or `L2`. The sensitivity table above says how well their signs and sizes carry over.
- **Mean-field sends are not Bernoulli sends.** A send probability `p` enters as a fluid of expected bytes, which removes the burstiness of Bernoulli arrivals and underestimates queueing delay. Relaxed-Bernoulli sends keep the burstiness, at the price of gradient variance.
- **Temperature bias.** For `tau > 0` the KPIs are biased (delay upward in the measured cases), and the optimum of the relaxed objective can differ from that of the discrete one. Scoring the result at `tau = 0` catches this.
- **Hard edges remain.** Positions within 1 m of the gNB are clamped by the path-loss model (zero gradient there). The age buffer has `timeout` slots, so a continuous deadline cannot exceed the timeout.
- **The neural proxy** is a regression fit. Its gradients are only as good as its training distribution (stationary Bernoulli sends, the default `L2-legacy` constants, one cell, the ranges in `proxy.RANGES`). Check them against finite differences of the engine before relying on them.
- **Cost.** L1D loops over the UL slots of each control step in Python, and backpropagation stores every slot, so a rollout costs roughly K = 40 small kernels per step forward and again backward. The models are meant for batched studies of tens to hundreds of envs, not for the throughput of the fast engine backends.
