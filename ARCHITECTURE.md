# Architecture: `isaaclab_net`

The goal is a pip-installable network extension for Isaac Lab. A backend-agnostic core (pure PyTorch/Triton, no Isaac imports) sits under a thin Isaac Lab layer, so the same core can also serve ManiSkill, MuJoCo Playground or plain PyTorch envs.

## Layout

```
isaaclab_net/
  core/
    config.py        # NRConfig: numerology, bandwidth/PRBs, RBG size, TDD pattern, K1/K2, SR period, HARQ
                     #           processes, PHY tables, noise model, cell layout, power control, handover,
                     #           application (frame buffer, timeout, message sizes); presets
    engine.py        # make_engine(level, E, R, device, config, backend); NREngine (contract API over NRNet)
    nr_engine.py     # NRNet: the configurable NR engine (level L2): slot schedule, fading, UL/DL, step_rx
    radio.py         # RadioMC (per-link path loss + shadowing), CellAssociation (attach, A3/TTT handover)
    phy.py           # 3GPP MCS/TBS tables, EESM effective SINR, BLER tables (Sionna, or local 5G-LENA)
    mac.py           # MacLink: per-slot MAC of one direction: multi-process HARQ, PF per RBG, link adaptation
    mac_ul.py        # uplink hooks: SR/BSR, proactive grants, power split / whole-band PSD, PHR cap
    mac_dl.py        # downlink hooks: delayed quantized CQI, K1 HARQ feedback
    queues.py        # fixed-shape frame FIFOs on a per-robot byte stream, RLC in-order delivery, reset helpers
    traffic.py       # Requests (message class per robot per step)
    data/            # Sionna SYS 2.2.0 BLER tables and EESM betas (Apache-2.0)
    proto/           # the prototype engine, frozen: levels L0, L0DR, L05, L05Q, L1 and L2-legacy (NetSlot)
      netsim.py      #   eager reference of every prototype level
      netsim_fast.py #   eager / graph / compile / triton backends, same API
      triton_slot.py #   fused per-step Triton kernels (L1, L2-legacy)
      netsim_mc.py   #   NetSlotMC: multi-cell L2-legacy (per-cell PF, same-slot interference, handover)
    levels/          # levels beyond the simulators, one graph-safe implementation each (reference + graph)
      base.py        #   LevelNet: FIFO and outputs from the prototype's graph-safe bodies, per-env clocks, resets
      surrogates.py  #   TR trace replay, GE Markov-modulated, QA analytic queue, NN learned surrogate
      bounds.py      #   ORACLE and NOCOMM value-of-information bounds
  isaac/
    net_module.py    # NetModule: reset(env_ids), submit(t, req), step(t, poses) -> dict (Isaac-side wrapper)
    netmodule.py     # registry engine L0/L1/L2 with per-env clocks, LOS blockage and parameter DR
    mixins.py        # DirectRLEnv mixin wiring the hooks (_get_dones / _reset_idx / _get_observations)
    mdp/             # event terms (network domain randomization)
  bridges/           # validation only, never in the training loop
    ns3_lockstep/    # TCP / Unix-socket / ns3-ai shared-memory lockstep co-simulation with ns-3 5G-LENA
    ns3_pool/        # one ns-3 process per env (the CPU co-simulation baseline)
    ns3_offline/     # trace-driven, open-loop replay; fixed-SNR 5G-LENA replay of the NR engine
    ns3/             # C++ ns-3 programs (our own, written against the ns-3 APIs) and build scripts
  examples/          # fleet_task.py (pure torch task), isaac_fleet_env.py (Isaac Lab demo env)
  tools/             # Sionna table export, local 5G-LENA table extraction, BLER curve CSVs,
                     # fit_levels.py: fits TR / GE / QA / NN from L2 or L2-legacy rollouts
tests/               # pytest suite; tests/scripts/ equivalence scripts; tests/bridges/ need an ns-3 build
benchmarks/          # engine speed, NR vs legacy, multi-cell, Isaac scaling, ns-3 scaling
prototype/           # compatibility shims: `import netsim` etc. return the package modules
```

## Interface contract (every component honors it)
- All state is fixed-shape tensors with leading dims `[E, R]` (plus a cell dim where needed). There are no Python loops over envs, robots or cells.
- Every state tensor supports partial `reset(env_ids)`, because Isaac Lab resets subsets of envs. A partial reset leaves every other env bit-for-bit unaffected, and its random draws come from the engine's own generator.
- Engine API: `reset(env_ids)`, `submit(t, requests)` for new messages per robot, and `step(t, poses)`. `step` returns a dict with delivered masks, per-message delay, newest delivered capture step (age of information), queue state, SINR and, for the NR and multi-cell engines, the serving cell. Time is a per-env episode clock; `t=None` uses it. The legacy `add_frames` / `step(t, snr, hid)` calls remain as wrappers.
- One shared `NRConfig` dataclass configures every module.
- Backends: `graph` is bitwise equal to the eager reference and is meant for scientific runs; `triton` is for scale. Any new backend must pass `tests/` equivalence against the eager reference.

## Fidelity levels
`L0` i.i.d. delay/loss · `L0DR` randomized delay · `L05` / `L05Q` lookup tables fitted offline from `L2` · `L1` fluid slot model · `L2` configurable NR MAC/PHY · `L2-legacy` the prototype slot-level MAC/PHY (multi-cell capable).
Surrogates fitted from `L2` or `L2-legacy` rollouts: `TR` trace replay · `GE` 3-state Markov-modulated delay/loss · `QA` analytic per-step processor-sharing queue · `NN` learned stateful surrogate.
Value-of-information bounds: `ORACLE` instant lossless delivery · `NOCOMM` nothing delivered.
All levels come from `make_engine` and expose the same API, so a task can switch fidelity with one argument.

`ORACLE` and `NOCOMM` are not network models. They bracket what any level can give a task, and they are the first check in task design: a task in which network fidelity can matter must show a large gap between its `ORACLE` and `NOCOMM` returns. If the gap is small, the policy does not use the information the network carries, and comparing fidelity levels on that task says nothing about the network.

## Decisions

- **One engine API.** `make_engine(level, E, R, device, config, backend)` builds every level. `L2` is the configurable NR engine (NRNet, merged from nrconfig). The prototype NetSlot stays, frozen, as `L2-legacy`: the kill-test results were produced with it, and its `graph` backend is bitwise verified against its reference. `L0` to `L1` keep their prototype implementations.
- **NR engine clock.** NRNet simulates continuous physical time with one global slot clock. `NREngine` keeps that clock and gives every env an episode clock (`clock = T - epoch`, zeroed by `reset(env_ids)`), and converts capture steps in its outputs. HARQ, SR and CQI timers of a reset env are cleared, so the reset is exact. An explicit `t` must equal the engine clock, because the NR engine cannot jump in time.
- **One config.** `NRConfig` carries the NR fields, the cells block merged from multicell, and the application fields including `msg_sizes`. The prototype levels read only the application fields and refuse a config that disagrees with their compiled constants (frame buffer 16, timeout 20 steps, 100 ms step). `L1` also requires the legacy single cell, because its radio is fixed. `L2-legacy` routes to NetSlot for the legacy single cell and to NetSlotMC for any other cell layout or noise model.
- **Multi-cell.** NetSlotMC is a NetSlot subclass and lives in `core/proto/netsim_mc.py`, next to the engine it extends, rather than in the NR engine's `mac_ul.py` as the multicell README proposed. Its radio and association (`RadioMC`, `CellAssociation`) live in `core/radio.py`, shared with the NR engine, which uses `RadioMC` at C = 1 to turn poses into path gains. At C = 1 with the fixed noise floor and power control off (the NRConfig defaults) it is bitwise equal to NetSlot (`tests/test_multicell.py`). Uplink fractional power control (P0 = −88 dBm per subband, α = 1) is on by default whenever `n_cells > 1`.
- **Multi-cell hooks in the NR engine.** `NRConfig.n_cells` and the cells block are shared; NREngine outputs `serving_cell` [E,R]; `NREngine.set_sinr_hook(fn)` installs `MacLink.sinr_hook(g, dir, won, n_prb, sinr) -> sinr`, which sits between allocation and decoding and is where same-slot inter-cell interference goes; `step_rx(pathgain, ul_interf_dbm_prb=...)` takes an external interference trace. NRNet refuses `n_cells > 1` until the multi-cell MAC is merged.
- **Licensing.** Only the Sionna-derived BLER tables (Apache-2.0) ship. The 5G-LENA tables are GPL-derived: `python -m isaaclab_net.tools.extract_lena_tables <your nr checkout>` builds them outside the source tree (`~/.cache/isaaclab_net/` or `$ISAACLAB_NET_LENA_TABLES`), and `.gitignore` blocks them. The C++ bridge programs are our own code written against ns-3 APIs; no ns-3 or 5G-LENA source is vendored.
- **Surrogate and bound levels.** `TR`, `GE`, `QA`, `NN`, `ORACLE` and `NOCOMM` live in `core/levels/`, ported from the kill-test `baselines/` and NetDelay modes onto per-env clocks and partial resets. Each is written once with graph-safe ops (no `nonzero`, no host sync in `submit` / `step`); `reference` runs it eagerly and `graph` captures it, and the FIFO, enqueue and end-of-step bookkeeping reuse the prototype's graph-safe bodies, so every level has the prototype's message semantics (F = 16, 2 s timeout, same dict outputs). Per-robot and per-env draws come from the global RNG in fixed positions, and reset draws (the replayed trace, the initial GE state) from the engine generator, so a partial reset leaves other envs bitwise unaffected. `NN` evaluates its MLP on every robot at every submit to keep shapes fixed. The parameter formats are those of the kill-test fits, which therefore load directly.
- **Fitted parameters stay outside the repository.** `python -m isaaclab_net.tools.fit_levels` rolls out `L2` or `L2-legacy` on the example fleet task under a behavior policy, logs every frame through the public API (so any level can be the source), fits all four surrogates into one file under `~/.cache/isaaclab_net/levels/` (or `$ISAACLAB_NET_LEVELS_DIR`), and refuses a path inside the source tree. `make_engine(level, ..., params=<file>)` loads it with `weights_only=True` and checks the message sizes it was fitted with.
- **Isaac layer.** It is a snapshot of the isaac/demo work of 2026-09-29, which is still in development. Its `NetConfig` remains the Isaac-side configuration until the layer is rebuilt on `make_engine` and `NRConfig`.

## Module status

| Module | Status | Tests |
|:---|:---|:---|
| `core/proto` (L0 ... L1, L2-legacy, fast backends) | frozen; bitwise-verified | `test_equivalence_cpu`, `test_gpu`, `tests/scripts/test_equiv.py`, `test_reset.py`, `test_regress.py` |
| `core/nr_engine`, `mac*`, `phy`, `queues` (L2) | merged; reference backend only | `test_nr_phy`, `test_nr_harq`, `test_nr_compat`, `test_engine_api` |
| `core/engine` (`make_engine`, `NREngine`) | new | `test_engine_api`, `test_package` |
| `core/levels` (TR, GE, QA, NN, ORACLE, NOCOMM) | new; reference and graph backends, graph bitwise equal to reference | `test_levels` |
| `core/config` (`NRConfig`, presets) | merged (nrconfig + multicell cells block) | `test_nr_phy`, `test_multicell` |
| `core/radio`, `core/proto/netsim_mc` (multi-cell) | merged on L2-legacy; NR multi-cell MAC pending | `test_multicell` |
| `isaac/` | snapshot of the demo in development; NetConfig not yet unified | `test_netmodule`, `test_isaac_layer` |
| `examples/fleet_task.py` | stable | `test_env_cpu`, `test_gpu` |
| `examples/isaac_fleet_env.py` | demo in development (needs Isaac Lab 3.0) | none in CI |
| `bridges/` | merged; ported to the per-env-clock NetBase; full resets only; lockstep and pool smoke-tested against the lab ns-3 builds | import tests; `tests/bridges/` need an ns-3 build |
| `tools/` | merged; `fit_levels` new | `test_nr_phy` (LENA path checks), `test_levels` (smoke fit) |

## Follow-ups
1. `graph` / `triton` backends for the NR engine (its step has no host syncs and fixed shapes, so CUDA-graph capture of a control step is the first option).
2. Multi-cell MAC in the NR engine: per-cell PF masking, same-slot interference through `sinr_hook`, handover moving the HARQ and stream state (multicell README, merge plan steps 3 to 5).
3. Rebuild the Isaac layer on `make_engine` and fold its `NetConfig` into `NRConfig`.
4. Surrogate fits from the NR engine with a non-default frame buffer or timeout: the surrogate levels share the prototype constants (F = 16, 20-step timeout, 100 ms step), so the fit tool refuses such configurations for now.
5. Re-sync `isaac/` with isaac/demo once that work settles (this is a 2026-09-29 snapshot).
