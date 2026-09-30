# Second backend: MuJoCo Playground / MJX (JAX)

The network module is not tied to Isaac Lab. `isaaclab_net.isaac.NetModule` is pure PyTorch (it builds its engine with `make_engine` and has no Isaac imports), and `isaaclab_net.mjx.NetModuleMJX` drives the same object from inside a jitted, vmapped MuJoCo Playground environment. The fleet task of the Isaac demo runs on MJX with the network in the loop, Brax PPO trains it, and the network outputs inside the JAX env are bitwise equal to a direct torch replay of the same poses. This page records how the interop works, how it was validated, what it costs, and the exact versions.

## Versions

Tested on the lab box (WSL2 Ubuntu 20.04 on Windows 11, RTX 4090, Windows driver 617.14, CUDA UMD 13.4), in a fresh venv built from Python 3.11.16:

| Component | Version |
|:---|:---|
| JAX | `jax[cuda12]` 0.9.2 (`jaxlib` 0.9.2, `jax-cuda12-plugin` / `jax-cuda12-pjrt` 0.9.2, bundled CUDA 12 wheels) |
| MuJoCo / MJX | `mujoco` 3.14.0, `mujoco-mjx` 3.14.0 (implementation `warp`, `warp-lang` 1.17.0; `jax` also works) |
| MuJoCo Playground | `playground` 0.2.0 |
| Brax | 0.14.2 (PPO), `flax` 0.12.6 |
| PyTorch | 2.14.0+cu126, `triton` 3.8.0 |

```bash
python3.11 -m venv venv && . venv/bin/activate && pip install uv
uv pip install torch --index-url https://download.pytorch.org/whl/cu126
uv pip install "jax[cuda12]==0.9.2" "flax" "brax==0.14.2" "playground==0.2.0"
uv pip install --no-deps -e .              # isaaclab_net; --no-deps keeps the CUDA build of torch
export XLA_PYTHON_CLIENT_PREALLOCATE=false  # otherwise JAX takes 75% of the GPU before torch allocates the engine
```

JAX 0.10 installs by default but breaks Brax 0.14.2 PPO (`jax.device_put_replicated` was removed), so JAX is pinned to 0.9.2, and `flax` then resolves to 0.12.6. The JAX CUDA 12 wheels and the cu126 build of torch share the `nvidia-*-cu12` packages without conflict (`uv pip check` is clean). The driver is far newer than CUDA 12 needs. JAX prints one harmless error at start-up, `Could not get kernel mode driver version`, because the WSL driver reports its version in a format JAX does not parse.

## How the network runs inside JAX

A Playground env implements `reset(rng)` and `step(state, action)` for one env, and the Brax wrappers vmap them over E envs and scan them inside `jit`. The torch engine is stateful and batched, so it enters that program through `jax.experimental.buffer_callback` (JAX 0.7 and later):

```python
net = NetModuleMJX("L2-legacy", E, R, "cuda", NRConfig(msg_sizes=(4000.0, 30000.0)), backend="triton", seed=0)

def step(self, state, action):                        # one env; vmapped over E by the Brax wrappers
    fresh = state.data.time == 0.0                    # data straight out of reset() (also after an autoreset)
    ...                                               # MJX physics, task logic
    out = net(poses_end, send, tag, cur_tag, fresh)   # dict of JAX arrays, [R, ...] per env
    obs = jnp.concatenate([..., out["feats"].reshape(-1)])
```

- **Batching.** The callback uses `vmap_method="broadcast_all"`, so under `jax.vmap` it is called once per step with the whole batch `[E, R, ...]`, which is what the engine expects.
- **Zero copy.** The callback receives XLA's device buffers and wraps them with `torch.from_dlpack`. The torch tensor has the same device pointer as the XLA buffer (asserted in `tests/mjx`), so the poses, sends and tags reach the engine without a copy and without a host round trip.
- **Stream ordering.** The callback also receives XLA's CUDA stream and runs every torch kernel on it through `torch.cuda.ExternalStream`. The network step is therefore ordered after the MJX physics of the same XLA program and before everything that reads its outputs, with no stream synchronization. XLA used the same stream for every call in the probes. Temporaries that torch allocates inside the callback belong to that stream in torch's caching allocator.
- **Warm-up.** The `graph` and `triton` backends capture CUDA graphs (and compile kernels) on their first call, and a capture synchronizes the device. `NetModuleMJX` does one dummy step and a full reset at construction, before any XLA program runs, and restores the global CUDA RNG afterwards. At interpreter exit it frees the captured graphs before JAX tears down the CUDA context, which otherwise aborts with `context is destroyed`.

**Copies that remain.** None goes through the host. On the device there are an int32 to int64 cast of the send, tag and current-tag arrays (JAX runs without x64 and the engine indexes with int64), and one device-to-device copy per result into the XLA-owned output buffer (`feats [E,R,4]` and a few `[E,R]` vectors). All are O(E·R) elements. If the env passes `[R,2]` poses, NetModule also appends the z column.

**Host synchronization.** There is one per step, and only for partial resets. The per-env reset flags are a device array, and the engine's reset draws (fading, shadowing) depend on how many envs reset, so `reset()` needs the indices on the host. The callback reads them with one `nonzero()`, which waits for the physics of the step to finish. Isaac Lab pays the same sync when it turns `reset_buf` into `env_ids`. A sync-free variant needs a masked reset in the core, which is described under [open items](#open-items).

**Resets.** `BraxAutoResetWrapper` resets a done env by restoring its cached first `mjx.Data` and observation, and it leaves `info` alone. `data.time` is 0 only in data that come out of `reset()`, so the env starts each step by checking `data.time == 0`. For those envs it restores the task state of `reset()` (kept in `info["task0"]`) and passes `reset=True` to the network, which resets them before it submits the step's messages. That is the order of the Isaac layer, which resets after the done step and then submits and steps the next one. The reset observation already carries zero network features, as the Isaac mixin's `net_obs` does for envs that just reset.

**Constraints.** Each `NetModuleMJX` serves one env batch, and its callback checks that the batch has E envs, so a Brax eval env is a second env instance built with `num_eval_envs`. The network state lives in torch, outside the JAX state: every execution of a jitted step advances it once, so a step must not be re-executed for the same state (for example by `jax.checkpoint` recomputation), and gradients through the network are not defined. JAX device 0 and torch `cuda:0` must be the same GPU.

## The MJX fleet env

[`isaaclab_net/examples/mjx_fleet_env.py`](../isaaclab_net/examples/mjx_fleet_env.py) (`MJXFleetEnv`) mirrors the Isaac demo env. One MJX model per env holds R bodies, each with two slide joints (x, y) and two velocity actuators (kv = 20, mass 1 kg), gravity off and no contacts. The time step is 1/50 s with 5 substeps per 0.1 s control step. The task is the same as in the Isaac env: a 150 m × 150 m arena with the gNB on a 6 m mast at the corner, per-robot velocity plus a send choice as the action, hazards that the fleet learns about only when a detecting frame is delivered, and the same 12-value observation per robot. Level `"off"` gives the ideal link with zero network features. Any `make_engine` level and backend works, because the env builds its network through `NetModuleMJX`.

Differences from the Isaac env: robot-robot contacts are off (the Isaac env keeps them), the robots follow the commanded velocity through actuators rather than having it written to the simulator, and a done env restarts from its cached first state (Playground's default autoreset) rather than from fresh random poses. The warp implementation of MJX needs `njmax > 0` even for a model without constraints: with `njmax = 0` nothing moves and no error is raised, so the env sets `njmax = 8`.

## Validation

`tests/mjx/test_mjx_backend.py` (marker `mjx`, 7 tests, all passing on the lab box in about 2 minutes under JAX 0.9.2 and in about 4 minutes under JAX 0.10.2):

1. **Interop.** Under `jit(vmap(...))`, the callback's torch tensors have the XLA buffers' device pointers, torch's current stream is XLA's stream, and the callback sees the whole batch.
2. **In-env network equals a direct replay, bitwise.** The fleet env (E = 24, R = 6) is stepped 70 times by a jitted `lax.scan` of the Brax-wrapped env, which is the PPO rollout path. The episode counters are staggered, so the autoreset wrapper restarts different envs at different steps. The module records the poses, sends, tags, reset indices and CUDA RNG state of every step, and `isaaclab_net.mjx.replay` drives a fresh torch `NetModule` with them, without JAX. Every output (`feats`, `delivered`, `newest_cap`, `last_cap`, `aoi_s`, `queue_len`, `sinr_db`, `tag_delivered`) is equal bitwise at every step for L2-legacy on `graph` (replayed on `graph`), on `triton` (replayed on `triton`) and on `eager` (replayed on the `reference` engine), for L0 on `graph`, and for L1 on `eager` against `reference`. The graph backend is itself bitwise equal to the reference with injected draws (`tests/test_isaac_layer.py`).
3. **Partial resets.** At every step the module reset exactly the envs that the wrapper restarted, and at least 5 steps had a strict subset of envs resetting. A reset env starts its episode with `last_cap = 0` and an AoI of one control step.
4. **Level `"off"`** builds no network and returns zero network features.

Run it with `XLA_PYTHON_CLIENT_PREALLOCATE=false python -m pytest -m mjx tests/mjx`. Without JAX or Playground the tests are skipped.

**PPO.** `benchmarks/mjx/train_ppo.py` trains the env with Playground's Brax PPO (E = 256, R = 16, L2-legacy on `triton`, 5 iterations of 10,240 env steps, episode length 100, a separate 128-env eval env with its own network). It ran end to end in 91 s, 25 s of which were compilation and the first evaluation. The network was called 200 times in training and 600 times in evaluation, with 512 and 768 env resets. Five iterations are a smoke test: the evaluation reward moved from −1.96 to between −1.26 and −1.86, which is not evidence of learning.

## Throughput

`benchmarks/mjx/bench.py` times the jitted `lax.scan` rollout of the wrapped env with random actions. `benchmarks/mjx/run_grid.sh` runs network off, L0 on `graph` and L2-legacy on `triton` at E ∈ {256, 1024, 4096} × R ∈ {16, 32}, twice, interleaved.

Measured on 2026-09-29 on the shared RTX 4090 (JAX 0.9.2, MJX warp, 200 timed steps after warm-up, random actions, episode length 300, so the only network resets are the initial ones). Each cell gives pass 1 / pass 2. The GPU utilization is device-wide and sampled every 0.5 s during the timed window, so it includes this run as well as the other jobs on the GPU, which kept the device between 0% and 98% busy before the runs and changed from minute to minute. Treat the absolute rates as lower bounds and compare within a row.

| E × R | Robots | Off (env steps/s) | L0 `graph` | L2-legacy `triton` | L2-legacy / off | Step with L2-legacy (ms) | NetModule alone (ms) | GPU util. during, off / L0 / L2 (%) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 256 × 16 | 4,096 | 102,542 / 72,797 | 19,522 / 19,455 | 20,799 / 14,982 | 0.20 / 0.21 | 12.3 / 17.1 | 8.2 / 8.7 | 75/29/64 ; 57/98/98 |
| 1,024 × 16 | 16,384 | 270,482 / 267,463 | 62,412 / 75,411 | 55,481 / 64,955 | 0.21 / 0.24 | 18.5 / 15.8 | 10.2 / 5.8 | 39/94/84 ; 50/52/27 |
| 4,096 × 16 | 65,536 | 818,010 / 1,077,486 | 286,113 / 283,092 | 249,409 / 277,179 | 0.30 / 0.26 | 16.4 / 14.8 | 7.9 / 7.0 | 67/57/92 ; 66/51/73 |
| 256 × 32 | 8,192 | 76,624 / 94,831 | 17,715 / 23,646 | 16,128 / 24,420 | 0.21 / 0.26 | 15.9 / 10.5 | 11.4 / 4.8 | 85/89/97 ; 88/97/97 |
| 1,024 × 32 | 32,768 | 261,044 / 320,980 | 53,617 / 83,128 | 52,084 / 80,830 | 0.20 / 0.25 | 19.7 / 12.7 | 6.6 / 5.4 | 99/93/98 ; 98/95/98 |
| 4,096 × 32 | 131,072 | 423,414 / 397,083 | 160,754 / 139,727 | 142,373 / 118,752 | 0.34 / 0.30 | 28.8 / 34.5 | 12.5 / 17.1 | 98/82/77 ; 100/99/98 |

Three observations. First, MJX itself is fast here: with the network off a control step (5 physics substeps for all envs) takes 2.5 to 10 ms, so 4,096 × 16 runs at 0.8 to 1.1 M env steps per second. Second, the network adds a roughly constant 8 to 24 ms per step, and most of it is the NetModule step itself: the torch module alone, with no JAX, takes 5 to 17 ms. That cost is host-bound (kernel launches from Python for the radio's pose chunks, the output dict and the features), which is why it grows little from 256 to 4,096 envs, and why L0 costs as much as L2-legacy on `triton`. The callback adds the rest, 1 to 9 ms per step and typically about 4 ms (the reset sync, the int casts, the output copies and the Python call itself). The NetModule-alone time was measured right after each timed window, under whatever contention held then, so this split is approximate. Third, the network-on throughput therefore reaches 0.25 to 0.28 M env steps per second (4.0 to 4.4 M robot-steps per second) at 4,096 × 16, and network on over off is 0.2 to 0.34, well below the 0.9 of the Isaac Lab runs, because the MJX step of this simple model is much cheaper than the PhysX step of the Isaac runs (which used R = 128 and more robots per env), so the fixed network cost is no longer hidden behind the physics. Capturing the NetModule step in a CUDA graph (open item 2) is the main lever. The torch memory of the network was 25 to 562 MiB (peak allocated, 256 × 16 to 4,096 × 32).

Raw rows: `benchmarks/mjx/run_grid.sh` appends one JSON line per run; the rows of this table are kept with the report of this work.

## Open items

1. **Masked reset in the core.** A `reset_mask(mask [E])` that draws the reset state for all E envs and keeps the rows of the masked ones would let the callback skip the one host sync per step. It changes the reset RNG stream (E draws instead of n), so it has to be a separate call next to `reset(env_ids)`.
2. **Capture the NetModule step.** The callback's cost is dominated by host-side launches of the NetModule step (the radio's pose chunks, the output dict, `net_features`) rather than by the triton kernel, which is why the network time barely changes from 256 to 4096 envs. Capturing the whole NetModule step in one CUDA graph would cut it to a few graph launches. That capture belongs in the `isaac/` layer or in a neutral module, not here.
3. **A neutral home for NetModule.** `NetModuleMJX` imports `isaaclab_net.isaac.net_module`, which is simulator-free but sits under `isaac/`. Moving `NetModule`, `IsaacRadio` and `MessageHistory` to a neutral package (for example `isaaclab_net.module`), with `isaac/` re-exporting them, would make the dependency direction explicit.
4. **Packaging.** An optional extra `mjx = ["jax[cuda12]==0.9.2", "playground==0.2.0", "brax==0.14.2"]` in `pyproject.toml` and the `mjx` marker in its pytest marker list (it is registered in `tests/mjx/conftest.py` for now).
