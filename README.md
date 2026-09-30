<div align="center">

<h1>isaaclab-net</h1>

<p><b>GPU-batched 5G network simulation for massively parallel robot learning: thousands of Isaac Lab environments, tens to hundreds of robots per cell, one GPU, network state stepped in lockstep with physics.</b></p>

<p>
  <a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white"></a>
  <a href="https://pytorch.org/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-CUDA-EE4C2C?style=flat-square&logo=pytorch&logoColor=white"></a>
  <a href="https://triton-lang.org/"><img alt="Triton" src="https://img.shields.io/badge/kernels-Triton-2F5C9E?style=flat-square"></a>
  <a href="https://isaac-sim.github.io/IsaacLab/"><img alt="Isaac Lab" src="https://img.shields.io/badge/Isaac%20Lab-3.0%20(in%20progress)-76B900?style=flat-square&logo=nvidia&logoColor=white"></a>
  <a href="https://github.com/ZzZTripleZzZ/isaaclab-net/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/ZzZTripleZzZ/isaaclab-net/actions/workflows/ci.yml/badge.svg"></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/License-BSD--3--Clause-yellow?style=flat-square&logo=opensourceinitiative&logoColor=white"></a>
  <img alt="Status" src="https://img.shields.io/badge/status-prototype-B7791F?style=flat-square">
</p>

</div>

```mermaid
flowchart LR
    subgraph SIM["Isaac Lab (E parallel envs)"]
        P["physics step<br/>robot poses [E, R, 3]"]
        POL["policy<br/>actions + messages to send"]
    end
    subgraph NET["isaaclab-net NetEngine (one GPU, all envs at once)"]
        RAD["radio<br/>path loss, shadowing, fading"]
        PHY["PHY abstraction<br/>MCS, BLER"]
        MAC["MAC per UL slot<br/>SR/BSR, PF scheduler, HARQ, OLLA"]
        Q["message queues<br/>FIFO, timeouts"]
    end
    P -- poses --> RAD
    POL -- messages --> Q
    RAD --> PHY --> MAC --> Q
    Q -- "delivered, delay, AoI, SINR" --> OBS["observations<br/>and rewards"]
    OBS --> POL
```

Parallel robot learning runs thousands of environments on one GPU, but the network between robots and the edge is usually reduced to a fixed or random delay, if it is modeled at all. Packet-level simulators such as ns-3 capture scheduling, retransmissions and contention, but they run one scenario at a time on a CPU, far from the throughput an RL loop needs. `isaaclab-net` closes that gap. Every piece of network state, from each robot's channel and HARQ process to its queued messages, is a fixed-shape tensor with leading dimensions `[envs, robots]`. The engine advances all environments' uplinks slot by slot on the GPU, in lockstep with the physics. A policy therefore trains against queues that build up when the team transmits together, links that degrade as robots move, and retransmissions that stretch delay tails.

**Status.** Early prototype, now packaged as `isaaclab_net`. The prototype levels and their fast backends are tested for bitwise equivalence, the configurable NR engine (3GPP MCS/TBS and BLER tables, multiple HARQ processes, downlink) and the multi-cell uplink are merged behind one engine factory, and the Isaac Lab layer and the ns-3 bridges are in the package. The Isaac Lab demo is still in development. [ARCHITECTURE.md](ARCHITECTURE.md) lists the status of every module.

## Install

```bash
git clone git@github.com:ZzZTripleZzZ/isaaclab-net.git && cd isaaclab-net
uv venv --python 3.11 && source .venv/bin/activate
uv pip install torch                                # CUDA build of PyTorch; Triton ships with it on Linux
uv pip install -e ".[dev]"                          # the isaaclab_net package, plus pytest and ruff
```

Linux with an NVIDIA GPU. The reference engines also run on a CPU. Scripts in `prototype/` still work: they are thin shims over the package.

## Put a network in your environment

```python
import torch
from isaaclab_net import NRConfig, Requests, make_engine

E, R, dev = 256, 16, torch.device("cuda")          # 256 envs, 16 robots each
net = make_engine("L2-legacy", E, R, dev, backend="graph")    # slot-level uplink, CUDA-graph backend
pos = torch.rand(E, R, 2, device=dev) * 150         # robot positions from your simulator ([E,R,3] also works)
last = torch.full((E, R), -1, dtype=torch.long, device=dev)

for _ in range(300):                                # one control step = 100 ms = 40 uplink slots
    send = (torch.rand(E, R, device=dev) < 0.3).long()   # per robot: 0 nothing, 1 small frame, 2 large frame
    net.submit(None, Requests(send))                # None = each env's own clock net.clock [E]
    out = net.step(None, pos)                       # positions go through the engine's radio; an SNR [E,R] also works
    last = torch.maximum(last, out["newest"])       # capture step of the newest frame delivered, -1 if none
    aoi = out["t"][:, None] + 1 - last              # age of the freshest delivered frame, in control steps
    queued = out["queue_len"]                       # frames still waiting per robot
    pos = (pos + 0.3 * torch.randn_like(pos)).clamp(0, 150)
    done = torch.nonzero(torch.rand(E, device=dev) < 0.005).squeeze(-1)   # envs whose episode ended
    net.reset(done)                                 # partial reset: queues, MAC, fading, radio and clock of these envs
    last[done] = -1
```

`make_engine(level, E, R, device, config, backend)` builds every fidelity level, and every engine has the same API. `step` also returns, per message slot, the `delivered` and `timed_out` masks, the `delay` in control steps and the `cap`/`cls` of each message, plus `queue_bytes`, `sinr_db` and, if `Requests(send, det, hid)` carried an application tag, `det_env`. `reset(env_ids)` takes an index tensor, a list or a bool mask and leaves every other env bit-for-bit unaffected. The earlier calls `add_frames(t, send, det, hid, snr)` and `step(t, snr, hid) -> (newest, det_env)` still work. `aoi` and `queued` go straight into observations. The example task in [`isaaclab_net/examples/fleet_task.py`](isaaclab_net/examples/fleet_task.py) uses the application tag to mark frames that captured a hazard. [`isaaclab_net/isaac/mixins.py`](isaaclab_net/isaac/mixins.py) wires a network into an Isaac Lab `DirectRLEnv` with four hook calls.

## Configure the network

One `NRConfig` dataclass configures every module: numerology, carrier and TDD pattern, MAC timing, HARQ and RLC, the PHY tables, the radio and cell layout, and the application fields (frame buffer, timeout, message sizes). The configurable NR engine is level `L2`:

```python
from isaaclab_net import NRConfig, make_engine
from isaaclab_net.core import lena_validation, multicell, netslot_compat, oai_like, srsran_like

cfg = NRConfig(mu=1, bandwidth_mhz=20, tdd_pattern="DDDSU", n_harq=16, mcs_table=2, dl=True)
net = make_engine("L2", E, R, dev, cfg)             # 51 PRB in 13 RBGs, 16 HARQ processes, EESM, uplink + downlink
net.add_dl_frames(None, torch.full((E, R), 3000.0, device=dev))  # downlink bytes per robot; see out["dl_newest"]
net = make_engine("L2", E, R, dev, netslot_compat())                     # closest to the legacy NetSlot
net = make_engine("L2-legacy", E, R, dev, multicell(3))                  # 3 cells with interference and handover
```

Presets: `netslot_compat()` (the legacy L2 geometry and timing with the 3GPP PHY), `lena_like()` and `lena_validation()` (the ns-3 5G-LENA reference scenario), `srsran_like()` and `oai_like()` (latency fitted to public srsRAN and OAI measurements), and `multicell(n)` (hexagonal cells at 100 m spacing, thermal noise, uplink fractional power control on). Uplink power control is on by default whenever `n_cells > 1`: without it, full-power robots next to their own gNB dominate the interference, and three cells carry less than one. The NR engine runs on the `reference` backend, and multi-cell configurations run on `L2-legacy` until the NR engine gets its multi-cell MAC.

**PHY tables and licensing.** The BLER tables shipped in `isaaclab_net/core/data/` are exported from Sionna SYS 2.2.0 (Apache-2.0, license file alongside). The 5G-LENA tables used by `bler_source="lena"` (the `lena_like` presets) are GPL-2.0 data and are never shipped or committed. Generate them from your own 5G-LENA checkout; the script asks for its location if you omit it:

```bash
git clone https://gitlab.com/cttc-lena/nr.git ~/src/nr
python -m isaaclab_net.tools.extract_lena_tables ~/src/nr   # writes ~/.cache/isaaclab_net/lena_eesm_tables.npz
```

`ISAACLAB_NET_LENA_TABLES` points the engine to another location. Keep the generated file out of any redistribution.

## What the engine models

| Layer | Configurable NR engine (`L2`) | Legacy slot-level model (`L2-legacy`) |
|:---|:---|:---|
| Radio | path loss, spatially correlated shadowing per env and cell, correlated Rayleigh fading per subband | the same, one cell at the arena corner |
| Frame structure | numerology 0 to 2, any bandwidth (38.101 N_RB), any TDD pattern and special slot, RBGs per 38.214 | TDD `DDDSU` at 30 kHz SCS, 40 uplink slots per 100 ms, 5 subbands of 10 PRBs |
| Access | periodic SR, grant delay, BSR, optional proactive grants | scheduling request, grant delay, buffer status reports |
| Scheduling | proportional fair per RBG (subband or wideband metric), retransmissions first | proportional fair over subbands, power split with a headroom cap |
| Link | 3GPP MCS tables and exact TBS, EESM, BLER-target link adaptation, OLLA, MCS caps | OLLA, one transport block per robot per slot, logistic BLER on effective SINR |
| Retransmission | multiple HARQ processes, chase or IR combining, RLC AM retry or UM loss | HARQ with a chase-combining gain and a retransmission limit |
| Downlink | per-robot gNB queues, delayed and quantized CQI, K1 feedback | none |
| Cells | one cell (multi-cell is next) | 1 to 7 cells, same-slot interference, fractional power control, A3 handover |
| Application | per-robot FIFO, in-order completion, timeout or PDCP discard | per-robot FIFO of frames, in-order completion, 2 s application timeout |

## Fidelity levels

Every level exposes the same API, so a task switches fidelity by changing one argument of `make_engine`.

| Level | Model | Typical use |
|:---|:---|:---|
| `L0` | i.i.d. lognormal delay and loss | the usual randomized-delay baseline |
| `L0DR` | `L0` with per-episode randomized delay and loss | domain randomization |
| `L05`, `L05Q` | lookup tables fitted offline from `L2` rollouts | cheap state-conditioned delay |
| `L1` | fluid slot model with equal PRB shares and FIFO queues | contention without MAC detail |
| `L2` | configurable NR MAC and PHY (table above) | the fidelity model |
| `L2-legacy` | the prototype slot-level MAC and PHY, frozen; multi-cell capable | reproducing the kill-test results, and speed at scale |
| `TR` | trace replay: each env replays one recorded `L2` env-episode, open loop | the replayed-trace baseline |
| `GE` | 3-state Markov-modulated delay and loss, one chain per env | the Gilbert–Elliott-style baseline |
| `QA` | analytic processor-sharing queue per control step, FIFO service, SR delay | contention without slot simulation |
| `NN` | learned stateful surrogate: MLP drop probability and delay quantiles from send-time features | the learned-surrogate baseline |
| `ORACLE` | every message delivered at capture, delay 0, never lost | upper bound on what any network gives a task |
| `NOCOMM` | no message ever delivered | lower bound: the task without communication |

`TR`, `GE`, `QA` and `NN` are fitted from `L2` or `L2-legacy` rollouts of the example fleet task. The fit writes one parameter file outside the repository, and every engine loads it:

```bash
python -m isaaclab_net.tools.fit_levels --source L2-legacy --task T1 --backend graph   # ~/.cache/isaaclab_net/levels/L2-legacy_T1.pt
```

```python
net = make_engine("NN", E, R, dev, params="~/.cache/isaaclab_net/levels/L2-legacy_T1.pt", backend="graph")
```

`ORACLE` and `NOCOMM` are value-of-information bounds for task design. Run a task under both first: a task in which network fidelity can matter must show a large gap between its `ORACLE` and `NOCOMM` returns. If the gap is small, the policy gains little from what the network delivers, and the task cannot tell fidelity levels apart.

## Backends and speed

Every prototype level (`L0` to `L1`, `L2-legacy`) has a readable eager reference in `isaaclab_net/core/proto/netsim.py` and graph-safe fast versions in `isaaclab_net/core/proto/netsim_fast.py`. `graph` records the same operations once as a CUDA graph and is **bitwise identical** to the reference at every level (per-message outputs, every queue and MAC state, with random partial resets). `triton` (`L1`, `L2-legacy`) runs all 40 slots of a control step in one fused kernel and matches the reference to rounding: from an identical state every finish time agrees, and over long runs aggregate delivery and delay agree to three or four significant digits. `compile` (torch.compile + CUDA graph) also agrees to rounding. The NR engine (`L2`) and the multi-cell engine have only the `reference` backend so far. The surrogate and bound levels (`TR` to `NOCOMM`) are written once with graph-safe ops, so their `reference` backend runs the same operations as `graph`, which is bitwise identical to it. On a shared GPU the NR uplink costs about 2.6–3.9 times the legacy reference per step, and the multi-cell engine about 1.2 times.

Network step time (`submit` + `step`, dict outputs) in ms, on an RTX 4090 that other jobs kept 98–99% busy, so absolute numbers are pessimistic:

| level | 256 × 16 reference | graph | triton | 4,096 × 100 reference | graph | triton |
|:---|---:|---:|---:|---:|---:|---:|
| `L0`, `L0DR` | 10.0–10.3 | 2.1 | | 19.2–19.5 | 45 | |
| `L05`, `L05Q` | 13.5–14.3 | 2.1 | | 21–23 | 46 | |
| `L1` | 118 | 7.3 | 2.1 | 458 | 474 | 48 |
| `L2-legacy` | 769 | 32 | 2.1 | 1,102 | 566 | 71 |

At small sizes every fast backend sits at a ~2 ms floor set by the busy GPU. At 4,096 × 100 the fixed-shape graph versions of the delay levels are memory-bound and slower than the eager reference, which only touches the new and finished frames. Resetting 1% of envs every step adds 0.3–2 ms at 256 × 16. `tests/scripts/test_equiv.py`, `test_reset.py` and `benchmarks/bench.py` reproduce these results, and `pytest -m gpu` runs the equivalence checks as tests.

## Validation against ns-3

The reference simulator is ns-3.48 with 5G-LENA NR v5.1, used unmodified except for a documented crash guard that does not change behavior. A 186-run sweep over 1 to 64 UEs, two frame sizes and 13–160% offered load is complete. Replaying 153 of its runs in the NR engine with the same per-UE link budgets (`lena_validation()`, `python -m isaaclab_net.bridges.ns3_offline.lena_replay`) matches the drop rate to a mean absolute difference of 0.029. The remaining light-load delay offset of about 20 ms is closed by an SR-to-grant delay of 40 slots, which is inferred from the replay and not yet measured in 5G-LENA. The comparison of delay distributions, throughput, HARQ statistics and PRB use will be added here. Co-simulation bridges to ns-3 (lockstep, process pool and offline replay) are in `isaaclab_net/bridges/`, with their C++ ns-3 programs in `isaaclab_net/bridges/ns3/`. They are for validation only and are never needed for training. They need a local ns-3 + 5G-LENA build, which is not part of this package.

## Roadmap

- [x] Slot-level uplink engine and lower fidelity levels
- [x] `graph` backend, bitwise equal to the reference, and `triton` backend for scale
- [x] Package layout `isaaclab_net/` (core, isaac, bridges, examples) per [ARCHITECTURE.md](ARCHITECTURE.md)
- [x] Partial resets per env, per-env clocks and the `submit` / `step` dict API, at every level and backend
- [x] Configurable NR: numerology, TDD patterns, 3GPP MCS/TBS and BLER tables, multiple HARQ processes, downlink
- [ ] `graph` / `triton` backends for the NR engine
- [x] Multi-cell interference and handover (legacy L2)
- [ ] Multi-cell MAC in the NR engine
- [ ] Isaac Lab 3.0 integration, demo tasks and scaling benchmarks (in development)
- [ ] Validation against ns-3 5G-LENA and public measurement traces
- [x] Test suite (CPU tests, GPU equivalence tests) and CI
- [x] Fitted surrogate levels (trace replay, Markov-modulated, analytic queue, learned) and ORACLE / NOCOMM bounds

## Repository layout

```
isaaclab-net/
├── isaaclab_net/
│   ├── core/                     # backend-agnostic engines, no simulator imports
│   │   ├── config.py             #   NRConfig, the one config dataclass, and its presets
│   │   ├── engine.py             #   make_engine(level, ...) and NREngine, the contract API of every level
│   │   ├── nr_engine.py          #   configurable NR engine (L2): slot schedule, fading, UL/DL, partial reset
│   │   ├── phy.py  queues.py     #   3GPP MCS/TBS/BLER/EESM; fixed-shape frame FIFOs on a byte stream
│   │   ├── mac.py mac_ul.py mac_dl.py   # per-slot MAC: multi-HARQ, PF, link adaptation; UL and DL hooks
│   │   ├── radio.py  traffic.py  #   per-link radio, cell association and handover; Requests
│   │   ├── data/                 #   Sionna BLER tables (Apache-2.0)
│   │   ├── proto/                #   prototype levels L0 ... L1 and L2-legacy: reference, fast backends,
│   │   │                         #   Triton kernels, and the multi-cell NetSlotMC
│   │   └── levels/               #   fitted surrogates TR, GE, QA, NN and the ORACLE / NOCOMM bounds
│   ├── isaac/                    # Isaac Lab layer: NetModule, DirectRLEnv mixin, mdp terms, skeletons
│   ├── bridges/                  # ns-3 co-simulation (lockstep, process pool, offline replay), validation only
│   │   └── ns3/                  #   the C++ ns-3 programs and their build scripts
│   ├── examples/                 # fleet_task.py (pure torch), isaac_fleet_env.py (Isaac Lab demo env)
│   └── tools/                    # PHY table export, local 5G-LENA table extraction, surrogate fits (fit_levels)
├── tests/                        # pytest suite; scripts/ (equivalence scripts), bridges/ (need ns-3)
├── benchmarks/                   # engine, NR, multi-cell, Isaac and ns-3 scaling benchmarks
├── prototype/                    # compatibility shims for the old module paths
├── scripts/                      # GPU test runner, Windows install scripts of the Isaac demo
├── ARCHITECTURE.md               # package layout, interface contract and module status
├── CONTRIBUTING.md
└── LICENSE
```

## Contributing

The repository is private while the first paper is in preparation. Collaborators should start from [CONTRIBUTING.md](CONTRIBUTING.md) and the open roadmap items. Changes to the prototype levels go into the eager reference first, and the `graph` backend must stay bitwise equal to it. New MAC and PHY modelling goes into the NR engine.

## Citation

A paper describing the engine is in preparation. A citation and an arXiv link will be added on release.

## License

BSD-3-Clause, see [LICENSE](LICENSE). The shipped Sionna tables are Apache-2.0 (`isaaclab_net/core/data/LICENSE-sionna-Apache-2.0`). No ns-3 or 5G-LENA code or data is included.
