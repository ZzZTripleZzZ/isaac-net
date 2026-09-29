# isaaclab-net

A GPU-batched 5G/cellular network module for massively parallel robot learning in NVIDIA Isaac Lab.

Robot RL in Isaac Lab runs thousands of environments in parallel on one GPU, but communication is usually modeled as a fixed or random delay, if at all. Packet-level network simulators such as ns-3 are accurate but run one scenario at a time on the CPU. `isaaclab-net` steps a slot-level 5G uplink model for thousands of environments, each with tens to hundreds of robots sharing a cell, in lockstep with the physics, entirely on the GPU.

**Status: early prototype.** The engine runs and is tested. The Isaac Lab integration is a skeleton, and the package layout in [ARCHITECTURE.md](ARCHITECTURE.md) is the next milestone.

## What is in `prototype/`

| File | What it is |
|---|---|
| `netsim.py` | Reference engine (eager PyTorch). Fidelity levels from i.i.d. delay (`L0`) to a slot-level uplink (`L2`): TDD, subband PF scheduling, SR/BSR, fading, link adaptation, BLER, HARQ, FIFO frame queues with timeouts. |
| `fast/netsim_fast.py` | Fast `L2` backends. `graph` is a CUDA-graph version, bitwise identical to the reference. `triton` is a fused kernel for scale. |
| `fast/triton_slot.py` | The Triton kernel: all UL slots of a control step in one launch, one env per program. |
| `fast/test_equiv.py` | Equivalence test: fast backends vs the reference under identical random draws. |
| `fast/bench.py`, `fast/bench_env.py` | Network-only and full-env benchmarks over envs × robots. |
| `env.py` | Example task: a kinematic multi-robot fleet that offloads hazard detection over a shared uplink. |
| `isaac/netmodule.py` | Backend-agnostic `NetModule` API with partial `reset(env_ids)` (skeleton). |
| `isaac/isaac_env_skeleton.py` | How the module hooks into an Isaac Lab `DirectRLEnv` (not yet run). |
| `isaac/maniskill_adapter_skeleton.py` | The same module in ManiSkill3, as a second backend (not yet run). |

## Quick start

Requirements: an NVIDIA GPU, Python 3.11, PyTorch ≥ 2.4 with CUDA, and Triton (bundled with PyTorch on Linux) for the `triton` backend.

```bash
cd prototype
python fast/test_equiv.py --help     # equivalence of fast backends vs the reference
python fast/bench.py --help          # ms per control step over E x R
```

Minimal use:

```python
import torch
from netsim import Radio
from fast.netsim_fast import make_net_fast

E, R, dev = 256, 16, torch.device("cuda")
net = make_net_fast("L2", E, R, dev, sizes=(4000.0, 30000.0), backend="graph")
radio = Radio(E, dev)
pos = torch.rand(E, R, 2, device=dev) * 150
snr = radio.snr_db(pos)
send = torch.randint(0, 3, (E, R), device=dev)          # 0 none, 1 small frame, 2 large frame
det = torch.zeros(E, R, dtype=torch.bool, device=dev)
hid = torch.zeros(E, dtype=torch.long, device=dev)
net.add_frames(0, send, det, hid, snr)
newest_delivered, _ = net.step(0, snr, hid)             # per-robot newest delivered capture step
```

## Early numbers (RTX 4090, GPU shared with other jobs)

| envs × robots | reference (ms/step) | graph (ms/step) | triton (ms/step) |
|---|---|---|---|
| 256 × 16 | 1,478 | 54.6 | 2.0 |
| 4096 × 100 | 1,586 | 818 | 60 |

The GPU was 95–99% busy with other jobs during these runs. On a quieter GPU, 256 × 16 took 13.4 ms (graph) and 0.43 ms (triton). One control step is 100 ms of simulated time, which is 40 uplink slots.

## Roadmap

See [ARCHITECTURE.md](ARCHITECTURE.md). Near-term milestones:
1. Restructure `prototype/` into the `isaaclab_net` package (core / isaac / bridges / examples).
2. Configurable NR (numerology, TDD pattern, 3GPP MCS/TBS and BLER tables, multiple HARQ processes, downlink).
3. Multi-cell with interference and handover.
4. Isaac Lab 3.0 integration and demo tasks at scale.
5. Validation against ns-3 5G-LENA through co-simulation bridges.

## License

BSD-3-Clause, see [LICENSE](LICENSE).
