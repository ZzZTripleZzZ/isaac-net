# Architecture: `isaaclab_net`

The goal is a pip-installable network extension for Isaac Lab. A backend-agnostic core (pure PyTorch/Triton, no Isaac imports) sits under a thin Isaac Lab layer, so the same core can also serve ManiSkill, MuJoCo Playground or plain PyTorch envs. MuJoCo Playground / MJX (JAX) runs as the second simulator backend (`mjx/`).

## Layout

```
isaaclab_net/
  core/
    config.py        # NRConfig: numerology, bandwidth/PRBs, RBG size, TDD pattern, K1/K2, SR period, HARQ
                     #           processes, PHY tables, noise model, cell layout, power control, handover,
                     #           application (frame buffer, timeout, message sizes); presets
    engine.py        # make_engine(level, E, R, device, config, backend); NREngine (contract API over NRNet)
    nr_engine.py     # NRNet: the configurable NR engine (level L2): slot schedule, fading, UL/DL, step_rx,
                     #        several cells (step_cells: association, handover, same-slot UL/DL interference)
    radio.py         # RadioMC (per-link large-scale gain, model = NRConfig.channel), CellAssociation (attach, A3/TTT)
    channels/        # channel models behind RadioMC (docs/channels.md)
      fields.py      #   plane-wave random fields (legacy band or exponential ACF)
      tr38901.py     #   TR 38.901 path loss, LOS probability, shadow-fading sigma, O2I (tables, hand-checkable)
      models.py      #   TR38901Channel (spatially consistent LOS state, O2I), RadioMapChannel
      radio_map.py   #   RadioMap: [C,H,W] gain map file format and bilinear sampling
      blockage.py    #   robot bodies as spheres on the robot-gNB segment
      doppler.py     #   per-robot AR(1) fading correlation for the NR engine
    phy.py           # 3GPP MCS/TBS tables, EESM effective SINR, BLER tables (Sionna, or local 5G-LENA)
    mac.py           # MacLink: per-slot MAC of one direction: multi-process HARQ, PF per RBG (one scheduler per
                     #          cell), link adaptation, handover of a robot's MAC state
    mac_ul.py        # uplink hooks: SR/BSR, proactive grants, power split / whole-band PSD, PHR cap
    mac_dl.py        # downlink hooks: delayed quantized CQI, K1 HARQ feedback
    queues.py        # fixed-shape frame FIFOs on a per-robot byte stream, RLC in-order delivery, reset helpers
    traffic.py       # Requests (message class per robot per step), TrafficModel / TrafficGen (generators in the L2 step)
    edge.py          # EdgeLoop: edge compute stage (FIFO / PS servers per env) and return path over any engine
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
    net_module.py    # NetModule over make_engine(level, E, R, device, NRConfig, backend): reset(env_ids),
                     #   submit(t, TrafficRequest(send, tag)), step(t, poses_end, cur_tag) -> dict with freshness
                     #   (last_cap, aoi_s) and tag_delivered; MessageHistory; IsaacNetCfg (Isaac-only settings); NetConfig is a deprecated alias
    radio.py         # IsaacRadio: poses -> SNR with per-env parameters (DR), several gNBs, LOS blockage
    mixins.py        # NetEnvMixin for DirectRLEnv (net_setup / net_step / net_reset / net_obs)
    mdp/             # randomize_network EventTerm (network domain randomization)
    netmodule.py     # compatibility re-exports (the demo's registry engine was retired)
  mjx/               # MuJoCo Playground / MJX (JAX) layer, the second simulator backend
    net_module.py    # NetModuleMJX: the isaac NetModule (pure torch) called from jitted, vmapped JAX through
                     #   jax.experimental.buffer_callback: zero-copy DLPack views on XLA's CUDA stream, one host
                     #   sync per step for partial resets; replay() drives a torch NetModule with recorded inputs
  bridges/           # validation only, never in the training loop
    ns3_lockstep/    # TCP / Unix-socket / ns3-ai shared-memory lockstep co-simulation with ns-3 5G-LENA
    ns3_pool/        # one ns-3 process per env (the CPU co-simulation baseline)
    ns3_offline/     # trace-driven, open-loop replay; fixed-SNR 5G-LENA replay of the NR engine
    ns3/             # C++ ns-3 programs (our own, written against the ns-3 APIs) and build scripts
  examples/          # fleet_task.py (pure torch task), isaac_fleet_env.py (Isaac Lab demo env),
                     # mjx_fleet_env.py (the same task as a MuJoCo Playground MjxEnv)
  tools/             # Sionna table export, local 5G-LENA table extraction, BLER curve CSVs,
                     # fit_levels.py: fits TR / GE / QA / NN from L2 or L2-legacy rollouts
benchmarks/isaac/    # fleet env throughput (bench.py), PPO smoke (train_ppo.py), grid / scale / train .ps1 drivers
benchmarks/mjx/      # MJX fleet env throughput (bench.py, run_grid.sh), Brax PPO smoke (train_ppo.py)
scripts/windows/     # Isaac Sim 6.1 + Isaac Lab 3.0 install (01..04), cartpole check (05), SYSTEM-task helpers
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

- **One engine API.** `make_engine(level, E, R, device, config, backend)` builds every level. `L2` is the configurable NR engine (NRNet, merged from nrconfig). The prototype NetSlot stays, frozen, as `L2-legacy`: the earlier prototype experiments were produced with it, and its `graph` backend is bitwise verified against its reference. `L0` to `L1` keep their prototype implementations.
- **NR engine clock.** NRNet simulates continuous physical time with one global slot clock. `NREngine` keeps that clock and gives every env an episode clock (`clock = T - epoch`, zeroed by `reset(env_ids)`), and converts capture steps in its outputs. HARQ, SR and CQI timers of a reset env are cleared, so the reset is exact. An explicit `t` must equal the engine clock, because the NR engine cannot jump in time.
- **One config.** `NRConfig` carries the NR fields, the cells block merged from multicell, and the application fields including `msg_sizes`. The prototype levels read only the application fields and refuse a config that disagrees with their compiled constants (frame buffer 16, timeout 20 steps, 100 ms step). `L1` also requires the legacy single cell, because its radio is fixed. `L2-legacy` routes to NetSlot for the legacy single cell and to NetSlotMC for any other cell layout or noise model.
- **Multi-cell (legacy).** NetSlotMC is a NetSlot subclass and lives in `core/proto/netsim_mc.py`, next to the engine it extends, rather than in the NR engine's `mac_ul.py` as the multicell README proposed. Its radio and association (`RadioMC`, `CellAssociation`) live in `core/radio.py`, shared with the NR engine. At C = 1 with the fixed noise floor and power control off (the NRConfig defaults) it is bitwise equal to NetSlot (`tests/test_multicell.py`). Uplink fractional power control (P0 = −88 dBm per subband, α = 1) is on by default whenever `n_cells > 1`.
- **Multi-cell NR engine.** NRNet takes `n_cells` = C up to 7 and ports NetSlotMC's design onto the NR MAC. At C = 1 none of the multi-cell code runs (every hook is `None`), and a frozen copy of the single-cell engine (971fc12) in `tests/nr_frozen/` checks that outputs and every state tensor stay bitwise equal.
  - *Input and state.* The input is the large-scale gain of every robot-cell link, either `pathgain_db` [E,R,C] or poses through the engine's `RadioMC`. The per-env state gains a cell dimension: fading [E,R,C,S,2] (shared by UL and DL, since TDD is reciprocal), N+I measured per gNB [E,C,S] (UL) and per robot [E,R,S] (DL), and the association. Every tensor has a fixed shape, and `reset(env_ids)` covers all of them.
  - *Schedulers.* The MAC state stays per robot, and each robot belongs to one cell. `MacLink.member` [E,C,R] runs retransmission admission and PF per RBG once per cell over the same carrier, batched as a masked max, and `sched_ok` blocks robots inside a handover interruption.
  - *Interference.* It goes through `MacLink.sinr_hook` and is computed in the same slot: in UL, from the robots the other cells scheduled on the same RBG, at their transmit PSD after the power split and fractional power control, with the fading of that robot-gNB link; in DL, from every other gNB that transmits on the RBG, at `gnb_tx_dbm` spread over the carrier. Link adaptation (the scheduler estimate, MCS, PHR cap and DL CQI) uses the N+I measured in the previous data slot of that direction (EWMA `li_alpha`), and decoding uses the actual N+I. A user hook set with `set_sinr_hook` runs after the engine's own. As in NetSlotMC, interference needs `noise_model="thermal"`, because the fixed floor already includes it.
  - *Handover.* `CellAssociation` counts TTT and interruption in NR slots (`slot_ms`), and the handover fires at its exact slot inside the step. `MacLink.handover` resets OLLA, the PF average and the CSI; the UL also drops a pending SR and sends the BSR with the handover-complete message. HARQ processes with undecoded bytes restart their combining at the target (RLC AM). Under RLC UM, and always with `ho_rlc="flush"`, they are lost instead, and `flush` also drops the queued frames.
  - *Outputs.* `step` outputs `serving_cell` [E,R], and `sinr_db` becomes the serving-link SINR against the gNB's latest N+I estimate. `step_rx(pathgain, ul_interf_dbm_prb=...)` still takes an external interference trace at C = 1.
- **Licensing.** Only the Sionna-derived BLER tables (Apache-2.0) ship. The 5G-LENA tables are GPL-derived: `python -m isaaclab_net.tools.extract_lena_tables <your nr checkout>` builds them outside the source tree (`~/.cache/isaaclab_net/` or `$ISAACLAB_NET_LENA_TABLES`), and `.gitignore` blocks them. The C++ bridge programs are our own code written against ns-3 APIs; no ns-3 or 5G-LENA source is vendored.
- **Surrogate and bound levels.** `TR`, `GE`, `QA`, `NN`, `ORACLE` and `NOCOMM` live in `core/levels/`, ported from the earlier prototype baselines and NetDelay modes onto per-env clocks and partial resets. Each is written once with graph-safe ops (no `nonzero`, no host sync in `submit` / `step`); `reference` runs it eagerly and `graph` captures it, and the FIFO, enqueue and end-of-step bookkeeping reuse the prototype's graph-safe bodies, so every level has the prototype's message semantics (F = 16, 2 s timeout, same dict outputs). Per-robot and per-env draws come from the global RNG in fixed positions, and reset draws (the replayed trace, the initial GE state) from the engine generator, so a partial reset leaves other envs bitwise unaffected. `NN` evaluates its MLP on every robot at every submit to keep shapes fixed. The parameter formats are those of the legacy baseline fitter, whose fit files therefore load directly.
- **Fitted parameters stay outside the repository.** `python -m isaaclab_net.tools.fit_levels` rolls out `L2` or `L2-legacy` on the example fleet task under a behavior policy, logs every frame through the public API (so any level can be the source), fits all four surrogates into one file under `~/.cache/isaaclab_net/levels/` (or `$ISAACLAB_NET_LEVELS_DIR`), and refuses a path inside the source tree. `make_engine(level, ..., params=<file>)` loads it with `weights_only=True` and checks the message sizes it was fitted with.
- **Isaac layer on the engine API.** `isaac.NetModule` builds its engine with `make_engine(level, E, R, device, config, backend)` from one `NRConfig`, so every level and backend is available in Isaac Lab (L0, L0DR, L05/L05Q with fitted params, L1 and L2-legacy on reference / eager / graph / compile / triton, the NR engine L2 on reference). The engines keep the per-env clocks and exact partial resets; the module adds only what an Isaac task needs: the Isaac radio (per-env radio parameters for DR, several gNBs, LOS blockage, SNR averaged over poses interpolated across the step), the per-message tag (`tag` [E,R] maps onto the engine's `det`/`hid`, and `cur_tag` gives `tag_delivered`), and the freshness outputs. The demo's `NetConfig` stays as a thin alias (`rung="L2"` means `L2-legacy`). The demo's registry engine (`netmodule.py`) was retired: its per-env MAC parameters (`bg_load`, per-env L0 lognormal) are not carried over; `L0DR` covers randomized delay. Multi-cell configs use the engine's radio (`radio="engine"`).
- **Isaac adapter fixes (from the demo).** MessageHistory starts from `seen_cap = -1`, so the first capture of an episode is delivered; the fast backends enqueue and reset without host syncs; frames are captured at the start-of-step pose and `step` takes the end-of-step poses, which also start the next step's interpolation; per-message tags as above. `net_step` sits in `_get_dones`, before `_reset_idx`.
- **Second backend: MJX.** `mjx.NetModuleMJX` reuses the Isaac layer's `NetModule` unchanged (it has no Isaac imports), so every level and backend is available in MuJoCo Playground. The torch engine is called from inside the jitted, vmapped env step through `jax.experimental.buffer_callback` (`vmap_method="broadcast_all"`, one call per step with the whole batch): the callback wraps XLA's buffers with `torch.from_dlpack` (zero copy) and runs torch on XLA's stream through `torch.cuda.ExternalStream`, so no stream sync or host copy is needed. Resets arrive as a per-env flag (the env passes `data.time == 0`, since Playground's autoreset restores the cached first `mjx.Data` and keeps `info`), and the callback turns them into indices with one host sync per step. The fast backends are warmed up (graph capture) at construction, outside any XLA program. Pinned to JAX 0.9.2 because Brax 0.14.2 PPO fails on JAX 0.10.
- **Isaac on Windows.** The lab box runs Isaac Sim 6.1 / Isaac Lab 3.0 natively on Windows (Python 3.12, torch 2.12 cu130, triton-windows 3.8). CUDA works there only for SYSTEM while nobody is logged on at the console, so GPU jobs run as one-shot SYSTEM scheduled tasks (`scripts/windows/systask.ps1`, `wait.ps1`).

## Module status

| Module | Status | Tests |
|:---|:---|:---|
| `core/proto` (L0 ... L1, L2-legacy, fast backends) | frozen; bitwise-verified | `test_equivalence_cpu`, `test_gpu`, `tests/scripts/test_equiv.py`, `test_reset.py`, `test_regress.py` |
| `core/nr_engine`, `mac*`, `phy`, `queues` (L2) | merged; multi-cell (UL + DL interference, power control, handover); `reference`, `graph` (bitwise) and `triton` (one cell) backends in `nr_fast.py` / `nr_triton.py` | `test_nr_phy`, `test_nr_harq`, `test_nr_compat`, `test_engine_api`, `test_nr_multicell` |
| `core/engine` (`make_engine`, `NREngine`) | new | `test_engine_api`, `test_package` |
| `core/levels` (TR, GE, QA, NN, ORACLE, NOCOMM) | new; reference and graph backends, graph bitwise equal to reference | `test_levels` |
| `core/config` (`NRConfig`, presets) | merged (nrconfig + multicell cells block) | `test_nr_phy`, `test_multicell` |
| `core/radio`, `core/proto/netsim_mc` (multi-cell) | merged; shared by L2 and L2-legacy | `test_multicell`, `test_nr_multicell` |
| `isaac/` | rebuilt on `make_engine` + `NRConfig`; validated on Windows Isaac Lab 3.0 (2026-09-29) | `test_isaac_layer` (module == reference engine bitwise at every level through a partial reset, CPU and GPU; graph bitwise with injected draws; triton reset invariants; radio, MessageHistory, mixin); `test_isaac_env` (`isaac`) |
| `core/edge` (`EdgeLoop`, `EdgeConfig`) | new; layered on the step dict of every level, engines untouched; optional CUDA-graph capture of the edge stage | `test_edge` |
| `mjx/`, `examples/mjx_fleet_env.py` | new; validated on the lab box (WSL2, JAX 0.9.2, MJX 3.14 warp, Playground 0.2.0, 2026-09-29); Brax PPO runs end to end | `tests/mjx` (`mjx`: buffer_callback zero copy on XLA's stream; in-env network == direct torch replay bitwise through staggered Playground autoresets for L2-legacy graph / triton / eager-vs-reference, L0 graph, L1 eager-vs-reference; resets hit exactly the restarted envs), not in CI |
| `benchmarks/mjx/` | throughput grid (off / L0 / L2-legacy triton, E 256–4096, R 16–32) and PPO smoke, under 0–98% shared-GPU contention | run by hand (`run_grid.sh`, `train_ppo.py`) |
| `examples/fleet_task.py` | stable | `test_env_cpu`, `test_gpu` |
| `examples/edge_control.py` | new; edge-offloaded tracking with hold / zero stale-action handling | run by hand |
| `examples/isaac_fleet_env.py` | stable demo env (needs Isaac Lab 3.0); PPO trains end to end | `test_isaac_env` (`isaac`: in-env network == reference replay bitwise through DirectRLEnv partial resets; every level and backend steps), not in CI |
| `benchmarks/isaac/` | fleet env throughput and PPO smoke; the published scale numbers were taken under 98–99% GPU contention | run by hand (`run_scale.ps1`, `run_train.ps1`) |
| `bridges/` | merged; ported to the per-env-clock NetBase; full resets only; lockstep and pool smoke-tested against the lab ns-3 builds | import tests; `tests/bridges/` need an ns-3 build |
| `tools/` | merged; `fit_levels` new | `test_nr_phy` (LENA path checks), `test_levels` (smoke fit) |

## Follow-ups
1. NR engine fast backends: `triton` for several cells (per-cell schedulers and same-slot interference in the fused kernel), and a tiled kernel for R above about 128 (the kernel keeps one env's robots in one program).
2. Isaac: uncontended rerun of the scale sweep (`benchmarks/isaac/run_scale.ps1`); faster scene startup (`clone_in_fabric`, one multi-instance asset per env instead of R rigid objects); a per-robot parameter-shared policy wrapper (`[E·R, obs]`) for R = 128; a downlink NetModule gating commands; the Warp mesh LOS kernel for R ≥ 64.
3. Surrogate fits from the NR engine with a non-default frame buffer or timeout: the surrogate levels share the prototype constants (F = 16, 20-step timeout, 100 ms step), so the fit tool refuses such configurations for now.
4. Isaac scale runs with the NR engine's `graph` / `triton` backends (`NetModule("L2", ..., backend=...)` works; not yet measured inside Isaac Sim).
5. MJX: a masked reset in the core (`reset_mask(mask [E])`, draws for all envs) to remove the one host sync per step; one CUDA graph for the whole NetModule step (the MJX step with the network is bound by NetModule's host-side launches, 5–17 ms, while MJX alone takes 2.5–10 ms); a neutral home for `NetModule` / `IsaacRadio` / `MessageHistory` (e.g. `isaaclab_net.module`, re-exported by `isaac/`). The `mjx` extra and pytest marker are in `pyproject.toml`.
