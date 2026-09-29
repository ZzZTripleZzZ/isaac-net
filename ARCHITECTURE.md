# Architecture: `isaaclab_net`

The goal is a pip-installable network extension for Isaac Lab. A backend-agnostic core (pure PyTorch/Triton, no Isaac imports) sits under a thin Isaac Lab layer, so the same core can also serve ManiSkill, MuJoCo Playground or plain PyTorch envs.

## Target layout

```
isaaclab_net/
  core/
    config.py        # NRConfig: numerology, bandwidth/PRBs, RBG size, TDD pattern, K1/K2, SR period,
                     #           HARQ processes, noise model, cell layout
    radio.py         # path loss, shadowing, fading, cell layout, association and handover
    phy.py           # 3GPP MCS/TBS tables, EESM/MIESM effective SINR, BLER tables
    mac_ul.py        # uplink: PF / wideband PF / RR, SR/BSR, multi-process HARQ, OLLA, power headroom
    mac_dl.py        # downlink: per-robot queues, DL scheduler, CQI-based link adaptation
    queues.py        # fixed-shape message FIFOs, RLC in-order delivery, timeouts
    engine.py        # NetEngine: fidelity levels behind one API; backends eager / graph / triton
    traffic.py       # message types: frames, periodic telemetry, control packets
  isaac/
    net_module.py    # NetModule: reset(env_ids), submit(...), step(poses) -> deliveries, delays, AoI, SINR
    mixins.py        # DirectRLEnv mixin wiring the hooks (_get_dones / _reset_idx / _get_observations)
    mdp/             # observation terms, event terms (network domain randomization), reward helpers
  bridges/           # validation only, never in the training loop
    ns3_lockstep/    # ZMQ/TCP and shared-memory lockstep co-simulation with ns-3 5G-LENA
    ns3_pool/        # one ns-3 process per env (the CPU co-simulation baseline)
    ns3_offline/     # trace-driven, open-loop replay
  examples/          # demo tasks
  tests/             # equivalence (graph == reference), MCS/TBS unit tests, partial-reset tests
  benchmarks/        # scaling over envs x robots, speed vs ns-3
```

## Interface contract (every component honors it)
- All state is fixed-shape tensors with leading dims `[E, R]` (plus a cell dim where needed). There are no Python loops over envs, robots or cells.
- Every state tensor supports partial `reset(env_ids)`, because Isaac Lab resets subsets of envs.
- Engine API: `reset(env_ids)`, `submit(t, requests)` for new messages per robot, and `step(t, poses)`. `step` returns a dict with delivered masks, per-message delay, newest delivered timestamp (age of information), queue state, SINR/RSRP and serving cell. The prototype's `add_frames` / `step` is the precursor of this API.
- One shared `NRConfig` dataclass configures every module.
- Backends: `graph` is bitwise equal to the eager reference and is meant for scientific runs; `triton` is for scale. Any new backend must pass `tests/` equivalence against the eager reference.

## Fidelity levels
`L0` i.i.d. delay/loss · `L0DR` randomized delay · `L05` / `L05Q` lookup tables fitted offline from `L2` · `L1` fluid slot model · `L2` slot-level MAC/PHY.
All levels expose the same API, so a task can switch fidelity with one config field.
