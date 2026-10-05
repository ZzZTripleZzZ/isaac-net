<div align="center">

<img src="https://raw.githubusercontent.com/ZzZTripleZzZ/isaac-net/main/docs/img/banner.jpg" alt="isaac-net: GPU-batched 5G simulation for massively parallel robot learning" width="100%">

<p><b>GPU-batched 5G and Wi-Fi network simulation for massively parallel robot learning: thousands of Isaac Lab environments, tens to hundreds of robots per cell, one GPU, network state stepped in lockstep with physics.</b></p>

<p>
  <a href="https://arxiv.org/abs/2610.02370"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-2610.02370-B31B1B?style=flat-square&logo=arxiv&logoColor=white"></a>
  <a href="https://pypi.org/project/isaac-net/"><img alt="PyPI" src="https://img.shields.io/pypi/v/isaac-net?style=flat-square&color=4B5563"></a>
  <a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB?style=flat-square&logo=python&logoColor=white"></a>
  <a href="https://pytorch.org/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-CUDA-EE4C2C?style=flat-square&logo=pytorch&logoColor=white"></a>
  <a href="https://triton-lang.org/"><img alt="Triton" src="https://img.shields.io/badge/kernels-Triton-2F5C9E?style=flat-square"></a>
  <a href="https://isaac-sim.github.io/IsaacLab/"><img alt="Isaac Lab" src="https://img.shields.io/badge/Isaac%20Lab-3.0-76B900?style=flat-square&logo=nvidia&logoColor=white"></a>
  <a href="https://github.com/google-deepmind/mujoco_playground"><img alt="MuJoCo Playground" src="https://img.shields.io/badge/MuJoCo%20Playground-MJX-1F6FEB?style=flat-square"></a>
  <a href="https://github.com/ZzZTripleZzZ/isaac-net/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/ZzZTripleZzZ/isaac-net/actions/workflows/ci.yml/badge.svg"></a>
  <a href="https://isaacnet.zifanzhang.com"><img alt="Website" src="https://img.shields.io/badge/website-isaacnet.zifanzhang.com-0B7285?style=flat-square&logo=cloudflare&logoColor=white"></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/License-BSD--3--Clause-yellow?style=flat-square&logo=opensourceinitiative&logoColor=white"></a>
  <img alt="Status" src="https://img.shields.io/badge/status-research%20prototype-B7791F?style=flat-square">
</p>

<p>
  <a href="https://arxiv.org/abs/2610.02370"><b>Paper (arXiv)</b></a> &nbsp;·&nbsp;
  <a href="https://isaacnet.zifanzhang.com"><b>Project website</b></a> &nbsp;·&nbsp;
  <a href="docs/README.md"><b>Documentation</b></a> &nbsp;·&nbsp;
  <a href="docs/tutorials/index.md"><b>Tutorials</b></a> &nbsp;·&nbsp;
  <a href="docs/benchmark-suite.md"><b>Benchmark suite</b></a> &nbsp;·&nbsp;
  <a href="CHANGELOG.md"><b>Changelog</b></a>
</p>

</div>

<p align="center">
  <img src="https://raw.githubusercontent.com/ZzZTripleZzZ/isaac-net/main/docs/img/overview.png" alt="One control step of isaac-net: Isaac Lab environments submit message classes and robot poses, the GPU network engine runs K uplink slots of NR MAC against PHY tables, and per-robot deliveries, delays, AoI and SNR return as observations" width="100%">
</p>

*One control step. Isaac Lab (left) submits a message class and the robot poses for every environment. The engine (right) keeps queues, radio state and the NR MAC in `[envs, robots, ...]` tensors, runs the K uplink slots of the step, and returns per-robot deliveries, delays, AoI, queue lengths and SNR. The strip shows how D physics substeps and K uplink slots share one control step.*

Parallel robot learning runs thousands of environments on one GPU, but the network between robots and the edge is usually reduced to a fixed or random delay, if it is modeled at all. Packet-level simulators such as ns-3 capture scheduling, retransmissions and contention, but they run one scenario at a time on a CPU, far from the throughput an RL loop needs. `isaac-net` closes that gap. Every piece of network state, from each robot's channel and HARQ process to its queued messages, is a fixed-shape tensor with leading dimensions `[envs, robots]`. The engine advances all environments' uplinks slot by slot on the GPU, in lockstep with the physics. A policy therefore trains against queues that build up when the team transmits together, links that degrade as robots move, and retransmissions that stretch delay tails.

**Status.** Research prototype, version 0.1.0, packaged as `isaac_net`. One factory builds every fidelity level behind one API: the configurable NR engine with multiple cells, the frozen legacy slot model, a Wi-Fi level, cheaper fluid and delay levels, fitted surrogates and two bounds. Around them sit selectable channel models and radio maps baked from USD scenes, traffic generators, an edge-computing loop, background users, a radio energy model, adaptive fidelity per env and multi-GPU sharding. The Isaac Lab layer and a MuJoCo Playground / MJX backend run all of it, the benchmark suite defines four network-aware multi-robot tasks, and ns-3 5G-LENA and OAI 5G bridges check the engine against a packet-level simulator and a real protocol stack. All speed and scale numbers come from an uncontended campaign on an idle GPU, and the 5G-LENA scheduler and grant mechanisms are engine switches with no fitted parameter. [docs/STATUS.md](https://github.com/ZzZTripleZzZ/isaac-net/blob/main/docs/STATUS.md) has the details and the open items, and [CHANGELOG.md](https://github.com/ZzZTripleZzZ/isaac-net/blob/main/CHANGELOG.md) lists what 0.1.0 contains.

**Platforms.** The `isaac_net` package itself is plain Python + PyTorch and runs on both Linux and Windows 11 with an NVIDIA GPU (the reference engines also run on a CPU). The Isaac Lab integration was developed and tested natively on **Windows 11**; Linux is supported through the recipe in [docs/isaac-lab-linux.md](https://github.com/ZzZTripleZzZ/isaac-net/blob/main/docs/isaac-lab-linux.md). Commands below are given for both: `bash` blocks are Linux, `powershell` blocks are Windows (Windows PowerShell 5.1 or PowerShell 7). Paths written as `~/.cache/isaac_net/...` mean `$HOME\.cache\isaac_net\...` (i.e. `C:\Users\<you>\.cache\isaac_net\...`) on Windows.

## Install

This installs the standalone package (no Isaac Lab). To use it inside Isaac Lab, follow the [Isaac Lab quick start](#isaac-lab-quick-start) instead, which installs it into Isaac's own environment.

**Linux (bash):**

```bash
git clone git@github.com:ZzZTripleZzZ/isaac-net.git && cd isaac-net
uv venv --python 3.11 && source .venv/bin/activate
uv pip install torch                                # CUDA build of PyTorch; Triton ships with it on Linux
uv pip install isaac-net                            # the released package from PyPI
# or, from a clone, for development:
uv pip install -e ".[dev]"                          # the isaac_net package, plus pytest, ruff and build
```

**Windows 11 (PowerShell):**

```powershell
git clone git@github.com:ZzZTripleZzZ/isaac-net.git
cd isaac-net
uv venv --python 3.11
.venv\Scripts\Activate.ps1                          # if blocked: Set-ExecutionPolicy -Scope Process Bypass
uv pip install torch --index-url https://download.pytorch.org/whl/cu130   # the PyPI torch wheel is CPU-only on Windows
uv pip install triton-windows                       # optional: community Triton build, only for the triton backend
uv pip install isaac-net                            # the released package from PyPI
# or, from a clone, for development:
uv pip install -e ".[dev]"                          # the isaac_net package, plus pytest, ruff and build
```

Linux with an NVIDIA GPU is the main target for the standalone package, and Python 3.10 to 3.12 with PyTorch 2.7 or later is supported. Every reference engine also runs on a CPU (`pip install torch --index-url https://download.pytorch.org/whl/cpu`), which is enough for the CPU test suite. Releases are on PyPI as [`isaac-net`](https://pypi.org/project/isaac-net/): `pip install isaac-net` (add `[dev]` for the test tools). Scripts in `prototype/` still work, as thin shims over the package.

| Extra | Adds | For |
|:---|:---|:---|
| `dev` | pytest, ruff, build | tests, lint, building the wheel |
| `docs` | mkdocs-material, mkdocstrings, mkdocs-jupyter | the docs site (`mkdocs build --strict`) |
| `mjx` | JAX 0.9.2 (CUDA 12), MuJoCo Playground 0.2.0, Brax 0.14.2 | the MuJoCo Playground / MJX backend ([docs/backends-mjx.md](https://github.com/ZzZTripleZzZ/isaac-net/blob/main/docs/backends-mjx.md)) |
| `isaac` | nothing from pip | the Isaac Lab layer. Install Isaac Sim and Isaac Lab separately, on [Windows](https://github.com/ZzZTripleZzZ/isaac-net/blob/main/docs/isaac-lab.md) or [Linux / HPC](https://github.com/ZzZTripleZzZ/isaac-net/blob/main/docs/isaac-lab-linux.md) |
| `ns3` | pybind11 | the ns-3 bridges ([docs/bridges.md](https://github.com/ZzZTripleZzZ/isaac-net/blob/main/docs/bridges.md)). ns-3.48 and 5G-LENA v5.1 are built locally |
| `oai` | pyarrow | the OAI 5G rfsim bridge and measurement tools ([docs/bridges-oai.md](https://github.com/ZzZTripleZzZ/isaac-net/blob/main/docs/bridges-oai.md)). OAI runs from its Docker images |
| `sionna` | Sionna RT, Sionna 2.2.0, usd-core | radio maps from USD scenes and the PHY table export ([docs/scene-radio-map.md](https://github.com/ZzZTripleZzZ/isaac-net/blob/main/docs/scene-radio-map.md)) |
| `wifi` | nothing | level `WIFI` needs only the core dependencies |
| `all` | every extra above except `isaac` | everything pip can install on Linux without Isaac |

Three console scripts come with the package: `isaac-net-bench` (the benchmark suite), `isaac-net-bake` (bake a radio map from a USD scene) and `isaac-net-measure` (probe, ingest and calibrate for gNB measurement campaigns).

## Put a network in your environment

```python
import torch
from isaac_net import NRConfig, Requests, make_engine

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

`make_engine(level, E, R, device, config, backend)` builds every fidelity level, and every engine has the same API. `step` also returns, per message slot, the `delivered` and `timed_out` masks, the `delay` in control steps and the `cap`/`cls` of each message, plus `queue_bytes`, `sinr_db` and, if `Requests(send, det, hid)` carried an application tag, `det_env`. `reset(env_ids)` takes an index tensor, a list or a bool mask and leaves every other env bit-for-bit unaffected. The earlier calls `add_frames(t, send, det, hid, snr)` and `step(t, snr, hid) -> (newest, det_env)` still work. `aoi` and `queued` go straight into observations. The example task in [`isaac_net/examples/fleet_task.py`](https://github.com/ZzZTripleZzZ/isaac-net/blob/main/isaac_net/examples/fleet_task.py) uses the application tag to mark frames that captured a hazard. [`isaac_net/isaac/mixins.py`](https://github.com/ZzZTripleZzZ/isaac-net/blob/main/isaac_net/isaac/mixins.py) wires a network into an Isaac Lab `DirectRLEnv` with four hook calls (see [Isaac Lab quick start](#isaac-lab-quick-start)).

## Isaac Lab quick start

This section is written for **Windows 11** (PowerShell), where the integration was developed and tested natively with an RTX 4090 (driver 617.14; the CUDA 13.0 build of PyTorch needs 580.88 or newer). For **Linux**, including clusters without root or with a glibc older than 2.35, install Isaac Lab with [docs/isaac-lab-linux.md](https://github.com/ZzZTripleZzZ/isaac-net/blob/main/docs/isaac-lab-linux.md) (kit-less Isaac Lab 3.0 on Newton or OV PhysX, where `ISAAC_NET_PHYSICS=newton` or `ovphysx` selects the fleet env's physics backend); the test, benchmark and training commands after that are given in bash below.

| Component | Version |
|:---|:---|
| Isaac Sim | 6.1.0.0 (pip wheels from pypi.nvidia.com) |
| Isaac Lab | 3.0 (`release/3.0.0`, package `isaaclab` 25.0.0; Isaac Sim 5.1 and older are not supported) |
| Python | 3.12 (uv venv) |
| PyTorch | 2.12.0+cu130, torchvision 0.27.0 |
| Triton | `triton-windows` 3.8.0.post29 (community build, only for the `triton` backend) |
| RL library | `rsl-rl-lib` 5.5.1 (installed by `isaaclab.bat -i`) |

**Install Isaac Sim and Isaac Lab (Windows).** The scripts in `scripts/windows/` follow the Isaac Lab 3.0 page "Python environment with Isaac Sim" (Windows, uv) and keep everything under `C:\isaac5g`. Run them from an Administrator PowerShell in the repository folder:

```powershell
New-Item -ItemType Directory -Force C:\isaac5g | Out-Null
Copy-Item scripts\windows\env.ps1 C:\isaac5g\env.ps1      # venv, uv, caches and EULA flag for every later step
powershell -File scripts\windows\01_bootstrap.ps1          # long paths on, uv and portable git in C:\isaac5g\tools
powershell -File scripts\windows\02_install_isaacsim.ps1   # venv, isaacsim[all,extscache]==6.1.0.0, torch 2.12 cu130, Isaac Lab clone
powershell -File scripts\windows\03_install_isaaclab.ps1   # isaaclab.bat -i: Isaac Lab and its RL libraries
powershell -File scripts\windows\04_triton_windows.ps1     # triton-windows, for the triton backend
```

`env.ps1` sets `OMNI_KIT_ACCEPT_EULA=YES`, which accepts the NVIDIA Omniverse EULA; read it before you run the scripts. It also moves the per-user caches (Kit, Triton, uv, temp) into `C:\isaac5g\home`. The download is about 40 GB and the install takes about 15 minutes.

**Add the package and run the tests, a benchmark and a short training run.** Windows (PowerShell):

```powershell
. C:\isaac5g\env.ps1                                       # activates the Isaac venv
cd C:\isaac5g\isaac-net                                 # this repository
uv pip install --no-deps -e .                              # --no-deps keeps Isaac's CUDA build of torch
uv pip install pytest
python -m pytest -m isaac tests\test_isaac_env.py
python benchmarks\isaac\bench.py --num_envs 256 --num_robots 16 --level L2-legacy --backend triton --steps 100
python benchmarks\isaac\train_ppo.py --num_envs 256 --num_robots 16 --level L2-legacy --backend triton --iters 5
```

Linux (bash), with the Isaac Lab venv from [docs/isaac-lab-linux.md](https://github.com/ZzZTripleZzZ/isaac-net/blob/main/docs/isaac-lab-linux.md) activated:

```bash
cd isaac-net                                               # this repository
uv pip install --no-deps -e .                              # --no-deps keeps Isaac's CUDA build of torch
uv pip install pytest
python -m pytest -m isaac tests/test_isaac_env.py
python benchmarks/isaac/bench.py --num_envs 256 --num_robots 16 --level L2-legacy --backend triton --steps 100
python benchmarks/isaac/train_ppo.py --num_envs 256 --num_robots 16 --level L2-legacy --backend triton --iters 5
```

Isaac Lab 3.0 runs headless by default. The Isaac tests launch one Isaac Sim process per case and take about 1 minute each. On a machine where nobody is logged on at the console, CUDA is available only to jobs that run as SYSTEM: `scripts\windows\systask.ps1` runs a script as a one-shot SYSTEM task, and `scripts\windows\wait.ps1` waits for it and removes the task.

**Add the network to your DirectRLEnv.** `NetEnvMixin` wires a `NetModule` into four hooks. The network is configured by the same `NRConfig` that `make_engine` takes, and a small `IsaacNetCfg` adds the Isaac-side settings: where the poses come from, the network rate, blockage, domain randomization and the observation features. Frames are captured at the start-of-step pose, and the network step takes the end-of-step poses:

```python
from isaaclab.envs import DirectRLEnv
from isaac_net import NRConfig
from isaac_net.isaac import IsaacNetCfg, NetEnvMixin

NR = NRConfig(msg_sizes=(4000.0, 30000.0))                  # the network: one config, as for make_engine
ISAAC = IsaacNetCfg(pose_asset="robots",                     # poses from the scene's "robots" collection
                    obs_features=("aoi", "sinr", "queue_len", "delay_history"),
                    dr_ranges={"noise_dbm": (-95.0, -85.0), "shadow_sigma_db": (3.0, 9.0)})
# env cfg: observation_space = R * (task features + ISAAC.obs_dim(NR))

class MyFleetEnv(NetEnvMixin, DirectRLEnv):
    def _setup_scene(self):
        ...                                                  # self.scene["robots"]: a RigidObjectCollection of R robots
        self.net_setup("L2-legacy", R, NR, "triton", isaac=ISAAC)
    def _pre_physics_step(self, actions):
        ...                                                  # read the start-of-step pose, decide what to send
        self.send = choose_messages(actions)                 # [E,R] long: 0 nothing, 1 small, 2 large message
    def _get_dones(self):
        out = self.net_step(None, self.send)                 # end-of-step poses from "robots"; newest_cap, aoi_s, ...
        ...
    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids); ...; self.net_reset(env_ids)   # also redraws the dr_ranges of env_ids
    def _get_observations(self):
        net = self.net_obs()                                 # [E,R,ISAAC.obs_dim(NR)] normalized network features
        ...
```

`net_setup` takes any level of `make_engine` (`"off"` for an ideal link), an `NRConfig`, a backend and an `IsaacNetCfg`. The observation features are chosen from the delivered mask and the delay of each message slot, age of information, queue length and bytes, SINR and RSRP, the serving cell, a last-delivery flag, the delays of the last k delivered messages, and a blockage flag, all with one normalization. The domain-randomization ranges cover the radio (transmit power, noise floor, path loss, shadowing sigma, blockage loss), the gNB placement, and the delay and loss of L0 and L0DR, and `dr_support(level)` tells which of them a level honors. `net_decimation` and `net_substeps` run the network slower or faster than the env step. `net_step(pos, send, tag, cur_tag)` also carries a per-message tag, such as the id of the hazard a frame captured, and returns `tag_delivered` per env. The fields, the feature table and the randomization table are in [docs/isaac-lab.md](https://github.com/ZzZTripleZzZ/isaac-net/blob/main/docs/isaac-lab.md#configuring-the-network). The earlier `NetConfig` is a deprecated alias. [`isaac_fleet_env.py`](https://github.com/ZzZTripleZzZ/isaac-net/blob/main/isaac_net/examples/isaac_fleet_env.py) is the complete example: E envs × R robots in a 150 m arena, with hazards that the whole fleet learns about only when a detection frame is delivered.

**Scale.** These numbers come from the fleet env with random actions, which saturate the uplink from 16 robots per env, measured on an idle RTX 4090 (0% utilization before every run; each process reports its median of 3 windows). The NR engine runs `ul_v2l`, the configuration validated against 5G-LENA except for the buffer-report grant pipeline, which the fused kernel lacks (median delay error −3.5% / −5.7% / −0.8% at light / moderate / saturated load). Network off and NR are the mean of 3 processes with the range in parentheses, `L2-legacy` is a single process from the first campaign (conditions in [docs/performance.md](https://github.com/ZzZTripleZzZ/isaac-net/blob/main/docs/performance.md#isaac-lab-scale)):

| Envs × robots | Robots | Network off (control steps/s) | L2-legacy `triton` (1 process) | NR `L2` `triton`, `ul_v2l` (uplink) | Network per step, isolated (legacy / NR) |
|---:|---:|---:|---:|---:|---:|
| 2,048 × 128 | 262,144 | 5.38 (5.04–5.61) | 4.08 | 4.11 (3.82–4.34) | 11 / 50 ms |
| 4,096 × 128 | 524,288 | 2.95 (2.84–3.07) | 2.39 | 2.44 (2.37–2.52) | 23 / 100 ms |
| 8,192 × 128 | 1,048,576 | 1.50 (1.37–1.57) | 1.51 | 1.24 (1.23–1.24) | 45 / 199 ms |

<p align="center">
  <img src="https://raw.githubusercontent.com/ZzZTripleZzZ/isaac-net/main/docs/img/teaser_blender.png" alt="Rendered fleet scene: many arenas, each with its own gNB mast and robot fleet, with an inset comparing robot-steps per second for GPU physics alone, physics with the per-slot 5G uplink, and ns-3 co-simulation" width="85%">
</p>

One control step is 0.1 s of simulated time. At about one million robots the network runs in the loop at 1.59 million (legacy) and 1.30 million (NR) robot-steps per second, in at most 20 GiB of device memory. The Isaac step is bound by host work. From 524k robots up the validated NR uplink lengthens the step by 40–86% of its isolated cost (about 30% of its GPU work overlaps with host work on average), and the network-off rate varies by 8–14% between processes, so single-run on / off differences of a few percent are not meaningful. Use `graph` for bitwise-reference runs and `triton` for scale. Startup grows about linearly with the number of robots (PhysX cloning), 17–26 minutes at one million robots. End-to-end PPO (rsl_rl, 1,024 × 16, L2-legacy `triton`) ran 30 iterations in 241 s on the earlier shared GPU.

**A second backend: MuJoCo Playground / MJX.** The same `NetModule` runs inside jitted, vmapped JAX code: [`isaac_net.mjx.NetModuleMJX`](https://github.com/ZzZTripleZzZ/isaac-net/blob/main/isaac_net/mjx/net_module.py) hands the MJX poses to the torch engine through `jax.experimental.buffer_callback` with zero-copy DLPack views on XLA's own CUDA stream, and [`mjx_fleet_env.py`](https://github.com/ZzZTripleZzZ/isaac-net/blob/main/isaac_net/examples/mjx_fleet_env.py) is the fleet task as a Playground env that Brax PPO trains. The in-env network is bitwise equal to a direct torch replay of the same poses on `graph`, `triton` and the reference engine; versions, costs and limits are in [docs/backends-mjx.md](https://github.com/ZzZTripleZzZ/isaac-net/blob/main/docs/backends-mjx.md).

## Configure the network

One `NRConfig` dataclass configures every module: numerology, carrier and TDD pattern, MAC timing, HARQ and RLC, the PHY tables, the radio and cell layout, and the application fields (frame buffer, timeout, message sizes). The configurable NR engine is level `L2`:

```python
from isaac_net import NRConfig, make_engine
from isaac_net.core import lena_validation, multicell, netslot_compat, oai_like, srsran_like

cfg = NRConfig(mu=1, bandwidth_mhz=20, tdd_pattern="DDDSU", n_harq=16, mcs_table=2, dl=True)
net = make_engine("L2", E, R, dev, cfg)             # 51 PRB in 13 RBGs, 16 HARQ processes, EESM, uplink + downlink
net.add_dl_frames(None, torch.full((E, R), 3000.0, device=dev))  # downlink bytes per robot; see out["dl_newest"]
net = make_engine("L2", E, R, dev, netslot_compat())                     # closest to the legacy NetSlot
net = make_engine("L2", E, R, dev, multicell(3, dl=True))               # 3 cells: UL + DL interference, handover
out = net.step(None, pos)                           # several cells take poses (or pathgain_db=[E,R,C]); out["serving_cell"]
```

Presets: `netslot_compat()` (the legacy L2 geometry and timing with the 3GPP PHY), `lena_like()` and `lena_validation()` (the ns-3 5G-LENA reference scenario), `srsran_like()` and `oai_like()` (latency fitted to public srsRAN and OAI measurements), and `multicell(n)` (hexagonal cells at 100 m spacing, thermal noise, uplink fractional power control on). Uplink power control is on by default whenever `n_cells > 1`: without it, full-power robots next to their own gNB dominate the interference, and three cells carry less than one. Multi-cell configurations run on `L2` (per-cell schedulers and HARQ, uplink and downlink interference) and on `L2-legacy` (NetSlotMC, uplink only). The NR engine runs on the `reference`, `graph` (bitwise equal to it) and `triton` (single cell, equal to rounding) backends.

**PHY tables and licensing.** The BLER tables shipped in `isaac_net/core/data/` are exported from Sionna SYS 2.2.0 (Apache-2.0, license file alongside). The 5G-LENA tables used by `bler_source="lena"` (the `lena_like` presets) are GPL-2.0 data and are never shipped or committed. Generate them from your own 5G-LENA checkout; the script asks for its location if you omit it:

```bash
git clone https://gitlab.com/cttc-lena/nr.git ~/src/nr
python -m isaac_net.tools.extract_lena_tables ~/src/nr   # writes ~/.cache/isaac_net/lena_eesm_tables.npz
```

```powershell
git clone https://gitlab.com/cttc-lena/nr.git $HOME\src\nr
python -m isaac_net.tools.extract_lena_tables $HOME\src\nr   # writes $HOME\.cache\isaac_net\lena_eesm_tables.npz
```

`ISAAC_NET_LENA_TABLES` (`export` in bash, `$env:ISAAC_NET_LENA_TABLES = "..."` in PowerShell) points the engine to another location. Keep the generated file out of any redistribution.

## What the engine models

The three simulating levels share the radio, the traffic and the application layer, and differ in how they model access to the channel:

| Layer | Configurable NR engine (`L2`) | Legacy slot-level model (`L2-legacy`) | Wi-Fi (`WIFI`) |
|:---|:---|:---|:---|
| Radio | selectable channel ([docs/channels.md](https://github.com/ZzZTripleZzZ/isaac-net/blob/main/docs/channels.md)): log-distance with correlated and white shadowing, TR 38.901 RMa / UMa / UMi / InH / InF path loss with a spatially consistent LOS state and O2I, or a precomputed radio map, for example baked from a USD scene ([docs/scene-radio-map.md](https://github.com/ZzZTripleZzZ/isaac-net/blob/main/docs/scene-radio-map.md)); optional robot-body blockage; correlated Rayleigh fading per subband and link, with per-robot Doppler | the same large-scale models (legacy fading), one cell at the arena corner by default | the same large-scale models between robots and access points, RSSI association with hysteresis and roaming interruption, a pairwise sensing matrix for hidden nodes |
| Frame structure | numerology 0 to 2, any bandwidth (38.101 N_RB), any TDD pattern and special slot, RBGs per 38.214 | TDD `DDDSU` at 30 kHz SCS, 40 uplink slots per 100 ms, 5 subbands of 10 PRBs | 802.11ax / ac / a PPDUs at 20 to 160 MHz, 1 to 4 spatial streams, the contention model re-solved every sub-step (1 ms by default) |
| Access | periodic SR, grant delay, BSR, optional proactive grants | scheduling request, grant delay, buffer status reports | DCF or EDCA contention (AIFS, CW per access class) as a Bianchi-style mean-field fixed point per sub-step, optional RTS/CTS |
| Scheduling | proportional fair per RBG (subband or wideband metric), max C/I or round robin, retransmissions first | proportional fair over subbands, power split with a headroom cap | random service order among the stations that win the channel |
| Link | 3GPP MCS tables and exact TBS, EESM, BLER-target link adaptation, OLLA, MCS caps | OLLA, one transport block per robot per slot, logistic BLER on effective SINR | SNR-threshold rate adaptation, A-MPDU aggregation up to the Block Ack window and the PPDU time limit |
| Retransmission | multiple HARQ processes, chase or IR combining, RLC AM retry or UM loss | HARQ with a chase-combining gain and a retransmission limit | collisions and a retry limit |
| Downlink | per-robot gNB queues, delayed and quantized CQI, K1 feedback | none | none |
| Cells | 1 to 7 cells, a PF scheduler and HARQ per cell, same-slot UL and DL interference, fractional UL power control, A3 handover with interruption | 1 to 7 cells, same-slot UL interference, fractional power control, A3 handover | several APs on shared or separate channels, co-channel contention domains |
| Application | per-robot FIFO, in-order completion, timeout or PDCP discard | per-robot FIFO of frames, in-order completion, 2 s application timeout | per-robot FIFO, in-order completion, application timeout |
| Traffic | policy messages plus periodic (sub-step periods), Markov on/off bursty, video I/P and event-triggered generators per robot, with arrival offsets in slots, tags, priority and deadline fields ([Traffic models](https://github.com/ZzZTripleZzZ/isaac-net/blob/main/docs/configurability.md#traffic-models)) | policy messages, one per robot per step | policy messages, one per robot per step |

Stages on top of the levels are set through `NRConfig` as well, keep the engine API, and add their own keys to the step dict:

| Stage | What it adds | Levels | Configure |
|:---|:---|:---|:---|
| Edge loop | edge servers per env (FIFO or processor sharing, deterministic or exponential service per message class, bounded queue, deadlines) and the return path to the robot: instant, delay from SINR, or a real NR downlink message. Outputs the capture step, age and latency of each robot's newest action, split into uplink, edge and return delays | every level | `NRConfig(edge=EdgeConfig(...))`, [configurability.md](https://github.com/ZzZTripleZzZ/isaac-net/blob/main/docs/configurability.md#edge-computing-loop) |
| Background users | UEs the policy does not control, with their own placement, mobility and traffic. In the NR engine they are extra rows that compete in PF, hold HARQ, interfere and hand over. On `L1` and `L2-legacy` they reduce the capacity through their offered load | `L2`, `L2-legacy`, `L1` | `NRConfig(background=BackgroundConfig(...))`, [background-energy-sharding.md](https://github.com/ZzZTripleZzZ/isaac-net/blob/main/docs/background-energy-sharding.md) |
| Radio energy | joules per robot from transmit power, circuit, receive and idle power, and a battery with a low-battery flag. Exact per slot on `L2`, an airtime estimate elsewhere | every level | `NRConfig(energy=EnergyConfig(...))` |
| Adaptive fidelity | a cheap and an expensive level side by side, chosen per env: a static mix, switching on a load indicator with queue handoff, or a curriculum over training iterations | cheap `L0` to `L1`, expensive `L1`, `L2-legacy` or `L2` | `make_adaptive(E, R, dev, FidelityConfig(...))`, [adaptive-fidelity.md](https://github.com/ZzZTripleZzZ/isaac-net/blob/main/docs/adaptive-fidelity.md) |
| Sharding | the envs split over several GPUs behind the API of one engine; under `rng="engine"` two shards are bitwise equal to one engine on the prototype levels, `L2-legacy`, `L2` and `WIFI` | every level (`L2` with traffic models or background users is not shard-invariant) | `ShardedEngine(level, E, R, devices, config, backend)` |

Without these fields nothing changes. [`examples/edge_control.py`](https://github.com/ZzZTripleZzZ/isaac-net/blob/main/isaac_net/examples/edge_control.py) is edge-offloaded tracking with a hold or zero rule for stale actions, and [`examples/traffic_models.py`](https://github.com/ZzZTripleZzZ/isaac-net/blob/main/isaac_net/examples/traffic_models.py) shows the traffic generators. Level `WIFI` takes its settings from `NRConfig(wifi=WifiConfig(...))` and is described, with its validation and what it leaves out, in [docs/wifi.md](https://github.com/ZzZTripleZzZ/isaac-net/blob/main/docs/wifi.md).

## Fidelity levels

Every level exposes the same API, so a task switches fidelity by changing one argument of `make_engine`.

| Level | Model | Typical use |
|:---|:---|:---|
| `L0` | i.i.d. lognormal delay and loss | the usual randomized-delay baseline |
| `L0DR` | `L0` with per-episode randomized delay and loss | domain randomization |
| `L05`, `L05Q` | lookup tables fitted offline from `L2` rollouts | cheap state-conditioned delay |
| `L1` | fluid slot model with equal PRB shares and FIFO queues | contention without MAC detail |
| `L2` | configurable NR MAC and PHY, 1 to 7 cells (table above) | the fidelity model |
| `L2-legacy` | the prototype slot-level MAC and PHY, frozen; multi-cell capable | reproducing the earlier prototype experiments, and speed at scale |
| `WIFI` | mean-field 802.11 DCF / EDCA uplink with 802.11ax / ac / a rates, several APs | fleets on Wi-Fi instead of private 5G |
| `TR` | trace replay: each env replays one recorded `L2` env-episode, open loop | the replayed-trace baseline |
| `GE` | 3-state Markov-modulated delay and loss, one chain per env | the Gilbert–Elliott-style baseline |
| `QA` | analytic processor-sharing queue per control step, FIFO service, SR delay | contention without slot simulation |
| `NN` | learned stateful surrogate: MLP drop probability and delay quantiles from send-time features | the learned-surrogate baseline |
| `ORACLE` | every message delivered at capture, delay 0, never lost | upper bound on what any network gives a task |
| `NOCOMM` | no message ever delivered | lower bound: the task without communication |
| `L1D`, `QAD` | differentiable relaxations of `L1` and `QA` in `core/diff` (`DiffFluid`), exact at temperature 0; not built by `make_engine` | gradients of delay, delivery, AoI and energy with respect to send probability, message size, transmit power and position ([docs/differentiable.md](https://github.com/ZzZTripleZzZ/isaac-net/blob/main/docs/differentiable.md)) |
| neural proxy | an MLP fitted to per-robot KPIs of `L2-legacy` rollouts (`core/diff/proxy.py`), a recipe | differentiable stand-in for `L2-legacy` |

`TR`, `GE`, `QA` and `NN` are fitted from `L2` or `L2-legacy` rollouts of the example fleet task. The fit writes one parameter file outside the repository, and every engine loads it:

```bash
python -m isaac_net.tools.fit_levels --source L2-legacy --task T1 --backend graph   # ~/.cache/isaac_net/levels/L2-legacy_T1.pt
```

```powershell
python -m isaac_net.tools.fit_levels --source L2-legacy --task T1 --backend graph   # $HOME\.cache\isaac_net\levels\L2-legacy_T1.pt
```

```python
net = make_engine("NN", E, R, dev, params="~/.cache/isaac_net/levels/L2-legacy_T1.pt", backend="graph")
```

The `~` in `params` is expanded by Python, so the same string works on Linux and Windows.

`ORACLE` and `NOCOMM` are value-of-information bounds for task design. Run a task under both first: a task in which network fidelity can matter must show a large gap between its `ORACLE` and `NOCOMM` returns. If the gap is small, the policy gains little from what the network delivers, and the task cannot tell fidelity levels apart.

`AdaptiveEngine` (`core/adaptive.py`) mixes two of these levels per env behind the same API, for example `L1` for most envs and `L2-legacy` for the envs whose cell is congested. With the switching threshold at 0 or infinity it is bitwise equal to the expensive or the cheap level alone.

## Backends and speed

Every prototype level (`L0` to `L1`, `L2-legacy`) has a readable eager reference in `isaac_net/core/proto/netsim.py` and graph-safe fast versions in `isaac_net/core/proto/netsim_fast.py`. `graph` records the same operations once as a CUDA graph and is **bitwise identical** to the reference at every level (per-message outputs, every queue and MAC state, with random partial resets). `triton` (`L1`, `L2-legacy`) runs all 40 slots of a control step in one fused kernel and matches the reference to rounding: from an identical state every finish time agrees, and over long runs aggregate delivery and delay agree to three or four significant digits. `compile` (torch.compile + CUDA graph) also agrees to rounding. The NR engine (`L2`) has `graph` (one or several cells, bitwise identical to its reference, including with random partial resets) and `triton` (one cell, one fused kernel per control step, equal to rounding); both need its engine RNG (`rng="engine"`, the default). The legacy multi-cell engine has only the `reference` backend. The surrogate and bound levels (`TR` to `NOCOMM`) are written once with graph-safe ops, so their `reference` backend runs the same operations as `graph`, which is bitwise identical to it. On an idle GPU the NR uplink costs about 3.5 times the legacy reference per step and 4–6 times the legacy `triton` kernel.

Network step time (`submit` + `step`, dict outputs) in ms on an idle RTX 4090, median of 3 processes ([docs/performance.md](https://github.com/ZzZTripleZzZ/isaac-net/blob/main/docs/performance.md) has every level, backend and size, the memory and the spreads):

| level | 256 × 16 reference | graph | triton | 4,096 × 100 reference | graph | triton |
|:---|---:|---:|---:|---:|---:|---:|
| `L0`, `L0DR` | 1.2–1.4 | 0.36 | | 4.7 | 10.6 | |
| `L05`, `L05Q` | 1.5–1.6 | 0.40 | | 5.0–5.1 | 10.9 | |
| `L1` | 15.1 | 1.9 | 0.36 | 107 | 108 | 11.1 |
| `L2-legacy` | 94 | 9.7 | 0.46 | 191 | 135 | 17.7 |
| `L2` (NR, uplink, `NRConfig()`) | 330 | 50 | 1.7 | 2,310 | 2,228 | 67.8 |
| `L2` (NR, uplink, v2 minus BSR) | | | 2.0 | | | 94.4 |
| `L2` (NR, uplink + downlink, `NRConfig()`) | 1,540 | 240 | 5.7 | 11,467 | 11,084 | 295 |

`NRConfig()` is the engine's default configuration and is not validated against 5G-LENA: its replay of the 5G-LENA sweep puts the median delay 30–76% low. "v2 minus BSR" is `lena_validation_v2()` without the SR / BSR grant pipeline, the closest validated configuration the fused kernel accepts (median delay error −3.5% / −5.7% / −0.8% at light / moderate / saturated load, [docs/fidelity-vs-lena.md](https://github.com/ZzZTripleZzZ/isaac-net/blob/main/docs/fidelity-vs-lena.md#scale-configurations)). At 4,096 × 100 the fixed-shape graph versions of the delay levels are memory-bound and slower than the eager reference, which only touches the new and finished frames, and the NR `graph` backend costs as much as its reference. `triton` is the scale path. `tests/scripts/test_equiv.py`, `test_reset.py` and `benchmarks/bench.py` reproduce these results, and `pytest -m gpu` runs the equivalence checks as tests.

## Validation

The engine is checked in four independent ways. Each check is a tool in the package or in `benchmarks/`, and each page lists its setup, its numbers and where the model still differs.

- **Backend equivalence.** Every fast backend is tested against the readable reference of its level: `graph` bitwise (per-message outputs, every queue and MAC state, through random partial resets), `triton` and `compile` to rounding. The same holds for the NR engine, the surrogates, the Wi-Fi level, the edge stage, adaptive fidelity and sharding, and the Isaac Lab and MJX layers replay bitwise against a direct engine run. `pytest -m gpu` runs these tests ([docs/performance.md](https://github.com/ZzZTripleZzZ/isaac-net/blob/main/docs/performance.md), [tests/README.md](https://github.com/ZzZTripleZzZ/isaac-net/blob/main/tests/README.md)).
- **ns-3 5G-LENA.** ns-3.48 with 5G-LENA v5.1 is the packet-level reference. A 186-run sweep over 1 to 64 UEs, two frame sizes and 13–160% offered load is replayed in the NR engine with the same per-UE link budgets and offered traffic (`lena_validation()`, `python -m isaac_net.bridges.ns3_offline.lena_replay`). The formal comparison reports delay quantiles, KS and Wasserstein distances, drops, goodput, HARQ and PRB use per run, with a fit and hold-out split for the one fitted parameter. Over the 153 runs of the no-fading arm the median p95-delay error is −7.8% and the median drop-rate difference −0.63 pp, and the engine is optimistic in loaded cells. A follow-up traced that gap to 5G-LENA's scheduler sharing, grant pipeline and RLC timing, and these mechanisms are being folded into the engine ([docs/validation-5g-lena.md](https://github.com/ZzZTripleZzZ/isaac-net/blob/main/docs/validation-5g-lena.md), [docs/fidelity-vs-lena.md](https://github.com/ZzZTripleZzZ/isaac-net/blob/main/docs/fidelity-vs-lena.md), [docs/fidelity-load-gap.md](https://github.com/ZzZTripleZzZ/isaac-net/blob/main/docs/fidelity-load-gap.md)). Co-simulation bridges (lockstep, process pool, offline replay) run a task against ns-3 directly ([docs/bridges.md](https://github.com/ZzZTripleZzZ/isaac-net/blob/main/docs/bridges.md)).
- **OAI 5G rfsim.** OpenAirInterface's gNB, nr-UE and core network, connected through the RF simulator, run behind a lockstep bridge with 1 to 10 UEs. The measured uplink access, HARQ and contention are compared with the engine's presets, and the fitted `oai_rfsim` preset matches the stock stack's single-UE small-frame delays to a median Wasserstein-1 distance of 2.5 ms ([docs/bridges-oai.md](https://github.com/ZzZTripleZzZ/isaac-net/blob/main/docs/bridges-oai.md)).
- **Public data and real cells.** Uplink latency fits to srsRAN and OAI measurements, a contention check on ColO-RAN and channel fits on POWDER drive tests produced the `srsran_like` and `oai_like` presets ([docs/calibration-public-data.md](https://github.com/ZzZTripleZzZ/isaac-net/blob/main/docs/calibration-public-data.md)). The measurement protocol and its tools (`isaac-net-measure`) are ready for a lab gNB and POWDER, for the layers public data cannot reach ([docs/measurement-protocol.md](https://github.com/ZzZTripleZzZ/isaac-net/blob/main/docs/measurement-protocol.md)).

The Wi-Fi level is validated separately against Bianchi's model, an exact slot-level CSMA/CA simulator and ns-3's 802.11ax model ([docs/wifi.md](https://github.com/ZzZTripleZzZ/isaac-net/blob/main/docs/wifi.md#validation)). All bridges and measurement tools are for validation only and are never needed for training. The ns-3 bridges need a local ns-3 + 5G-LENA build, which is not part of this package, and binaries built from them are GPL-covered ([docs/licensing.md](https://github.com/ZzZTripleZzZ/isaac-net/blob/main/docs/licensing.md)).

## Benchmark suite

`isaac_net.bench` gives network-aware multi-robot learning a common set of tasks, metrics and baselines. Four tasks run E envs of R robots on one GPU and talk to the network only through the `NetModule` of the Isaac Lab layer: `fleet_alert` (detection frames warn the fleet of hazards), `coop_map` (map patches keep an edge map fresh), `coverage_nav` (navigation under a remote supervisor that stops robots it has not heard from) and `edge_control` (tracking with an edge-offloaded controller). Every task has `default`, `light` and `background` variants and runs on any level and backend, and every run writes one versioned JSON result file.

```bash
isaac-net-bench list                                         # tasks, variants, levels, presets, baselines
isaac-net-bench run --task coop_map --level L2-legacy --backend triton \
    --baselines random,heuristic,ppo_mlp --seeds 0,1,2 --out results/
isaac-net-bench report results/                              # mean ± 95% CI over seeds, as a Markdown table
isaac-net-bench calibrate --task all --level L2-legacy --backend triton   # offered vs delivered load
```

`python -m isaac_net.bench` is the same command. The suite ships random, heuristic and PPO (MLP and GRU) baselines and sanity numbers that show the pipeline works, not tuned results. The task API, the metrics, the result format and how to add a task or submit a baseline are in [docs/benchmark-suite.md](https://github.com/ZzZTripleZzZ/isaac-net/blob/main/docs/benchmark-suite.md).

## Roadmap

- [x] Slot-level uplink engine and lower fidelity levels
- [x] `graph` backend, bitwise equal to the reference, and `triton` backend for scale
- [x] Package layout `isaac_net/` (core, isaac, mjx, bridges, bench, examples, tools) per [ARCHITECTURE.md](https://github.com/ZzZTripleZzZ/isaac-net/blob/main/ARCHITECTURE.md)
- [x] Partial resets per env, per-env clocks and the `submit` / `step` dict API, at every level and backend
- [x] Engine-owned random streams at every level
- [x] Configurable NR: numerology, TDD patterns, 3GPP MCS/TBS and BLER tables, multiple HARQ processes, downlink, schedulers
- [x] `graph` backend for the NR engine (one or several cells) and `triton` for one cell
- [x] Multi-cell interference and handover, in the NR engine and the legacy engine
- [x] Selectable channel models (TR 38.901, radio maps, blockage, per-robot Doppler) and radio maps baked from USD scenes
- [x] Traffic models, edge-computing loop, background users, radio energy model
- [x] Wi-Fi level (802.11 DCF / EDCA)
- [x] Uncontended Isaac Lab scaling benchmarks on an idle GPU (see docs/performance.md)
- [x] Validation against ns-3 5G-LENA (formal study) and public measurement traces; OAI rfsim as a real-stack check
- [x] Fitted surrogate levels (trace replay, Markov-modulated, analytic queue, learned) and ORACLE / NOCOMM bounds
- [x] Adaptive and mixed fidelity per env, differentiable fluid models, multi-GPU sharding
- [x] Isaac Lab 3.0 integration on the engine API: DirectRLEnv mixin, network domain randomization, fleet and warehouse demo envs, PPO; Windows install and Linux / HPC recipe
- [x] Second backend: MuJoCo Playground / MJX (JAX) through a zero-copy `buffer_callback`, fleet env, Brax PPO
- [x] Benchmark suite: four tasks, metrics, baselines, result format
- [x] Validation against ns-3 5G-LENA, OAI 5G rfsim and public measurement data
- [x] Test suite (CPU tests, GPU equivalence tests), CI, docs site, 0.1.0 packaging
- [ ] Uncontended speed and scale benchmarks (in flight)
- [ ] Load-gap mechanisms of 5G-LENA as NR engine switches (in flight)
- [ ] `triton` for the multi-cell NR engine and a tiled kernel for large R
- [ ] Lab gNB and POWDER measurement campaign; multi-cell calibration against a reference
- [ ] Isaac Sim (Kit) on Linux clusters once a final Isaac Lab 3.0 container exists
- [x] Public release and PyPI (`pip install isaac-net`, 0.1.0)

## Repository layout

```
isaac-net/
├── isaac_net/
│   ├── core/                     # backend-agnostic engines, no simulator imports
│   │   ├── config.py             #   NRConfig, the one config dataclass, and its presets
│   │   ├── engine.py             #   make_engine(level, ...) and NREngine, the contract API of every level
│   │   ├── nr_engine.py          #   configurable NR engine (L2): slot schedule, fading, UL/DL, cells, partial reset
│   │   ├── nr_fast.py nr_triton.py nr_rng.py   # NR graph and triton backends, engine-owned NR random streams
│   │   ├── phy.py  queues.py     #   3GPP MCS/TBS/BLER/EESM; fixed-shape frame FIFOs on a byte stream
│   │   ├── mac.py mac_ul.py mac_dl.py   # per-slot MAC: multi-HARQ, schedulers, link adaptation; UL and DL hooks
│   │   ├── radio.py  traffic.py  #   per-link radio, cell association and handover; Requests, TrafficModel
│   │   ├── channels/             #   log-distance fields, TR 38.901, radio maps, blockage, per-robot Doppler
│   │   ├── edge.py               #   EdgeLoop: edge servers and the return path on top of any engine
│   │   ├── background.py energy.py slot_tap.py  # background users; radio energy and battery; read-only slot tap
│   │   ├── adaptive.py           #   AdaptiveEngine: cheap and expensive level per env, curriculum
│   │   ├── sharded.py            #   ShardedEngine: envs split over several GPUs
│   │   ├── nr_loadfix.py         #   prototype of the 5G-LENA load-gap mechanisms (NR engine subclasses)
│   │   ├── wifi/                 #   level WIFI: 802.11 rates, mean-field DCF / EDCA, engine, event simulator
│   │   ├── diff/                 #   differentiable L1D / QAD, reference rollouts, neural-proxy recipe
│   │   ├── levels/               #   fitted surrogates TR, GE, QA, NN and the ORACLE / NOCOMM bounds
│   │   ├── proto/                #   prototype levels L0 ... L1 and L2-legacy: reference, fast backends,
│   │   │                         #   Triton kernels, counter RNG, and the multi-cell NetSlotMC
│   │   └── data/                 #   Sionna BLER tables (Apache-2.0), synthetic radio map
│   ├── isaac/                    # Isaac Lab layer: NetModule, IsaacNetCfg, DirectRLEnv mixin, mdp terms, scene maps
│   ├── mjx/                      # MuJoCo Playground / MJX layer: NetModuleMJX (JAX buffer_callback)
│   ├── bench/                    # benchmark suite: task API, four tasks, metrics, baselines, runner, CLI
│   ├── bridges/                  # validation only
│   │   ├── ns3_lockstep/ ns3_pool/ ns3_offline/   # ns-3 co-simulation and offline replay
│   │   ├── ns3/                  #   the C++ ns-3 programs and their build scripts
│   │   └── oai/                  #   OAI 5G rfsim bridge: stack, agents, virtual clock, deploy/
│   ├── examples/                 # fleet_task.py (pure torch), edge_control.py, traffic_models.py,
│   │                             # isaac_fleet_env.py, isaac_warehouse_env.py (Isaac Lab), mjx_fleet_env.py (MJX)
│   └── tools/                    # PHY table export, local 5G-LENA table extraction, surrogate fits (fit_levels),
│       ├── scene/                #   USD export, Sionna RT bake (isaac-net-bake), synthetic scenes
│       └── measure/              #   gNB log parsers, UDP probe, calibration (isaac-net-measure)
├── tests/                        # pytest suite; scripts/ (equivalence scripts), mjx/, bridges/ (need ns-3 or OAI)
├── benchmarks/                   # engine, NR, multi-cell, sharding, adaptive, differentiable, fidelity, Isaac,
│                                 # MJX, ns-3 and OAI benchmarks and campaigns, with their small result CSVs
├── tutorials/                    # tutorial scripts behind the notebooks of the docs site
├── docs/                         # docs site sources (mkdocs.yml): concepts, tutorials, API reference, project notes
├── prototype/                    # compatibility shims for the old module paths
├── scripts/                      # GPU test runner; windows/: Isaac install and SYSTEM-task helpers; hazel/: Slurm templates
├── ARCHITECTURE.md               # package layout, interface contract and module status
├── CHANGELOG.md  RELEASE.md      # release notes; how to cut a release
├── CONTRIBUTING.md
└── LICENSE
```

## Documentation

The project website is [isaacnet.zifanzhang.com](https://isaacnet.zifanzhang.com), with an overview, the validation results, the scale numbers and a quick start. Collaborator documentation lives in [`docs/`](https://github.com/ZzZTripleZzZ/isaac-net/blob/main/docs/README.md) and builds into a site with `mkdocs build --strict` (extra `docs`): [concepts](https://github.com/ZzZTripleZzZ/isaac-net/blob/main/docs/concepts.md), five [tutorials](https://github.com/ZzZTripleZzZ/isaac-net/blob/main/docs/tutorials/index.md), the [API reference](https://github.com/ZzZTripleZzZ/isaac-net/blob/main/docs/reference/index.md), the [benchmark suite](https://github.com/ZzZTripleZzZ/isaac-net/blob/main/docs/benchmark-suite.md) and [licensing](https://github.com/ZzZTripleZzZ/isaac-net/blob/main/docs/licensing.md). The project notes cover the [project status](https://github.com/ZzZTripleZzZ/isaac-net/blob/main/docs/STATUS.md) with open items and starter tasks, [configurability](https://github.com/ZzZTripleZzZ/isaac-net/blob/main/docs/configurability.md) with the feature matrix against 5G-LENA, Sionna SYS and Simu5G, [channel models](https://github.com/ZzZTripleZzZ/isaac-net/blob/main/docs/channels.md), [radio maps from USD scenes](https://github.com/ZzZTripleZzZ/isaac-net/blob/main/docs/scene-radio-map.md), [multi-cell networks](https://github.com/ZzZTripleZzZ/isaac-net/blob/main/docs/multicell.md), [Wi-Fi](https://github.com/ZzZTripleZzZ/isaac-net/blob/main/docs/wifi.md), [background users, energy and sharding](https://github.com/ZzZTripleZzZ/isaac-net/blob/main/docs/background-energy-sharding.md), [adaptive fidelity](https://github.com/ZzZTripleZzZ/isaac-net/blob/main/docs/adaptive-fidelity.md), [differentiable models](https://github.com/ZzZTripleZzZ/isaac-net/blob/main/docs/differentiable.md), [performance](https://github.com/ZzZTripleZzZ/isaac-net/blob/main/docs/performance.md) with the backend equivalence methodology, the [Isaac Lab integration](https://github.com/ZzZTripleZzZ/isaac-net/blob/main/docs/isaac-lab.md) and its [Linux / HPC recipe](https://github.com/ZzZTripleZzZ/isaac-net/blob/main/docs/isaac-lab-linux.md), the [MuJoCo Playground / MJX backend](https://github.com/ZzZTripleZzZ/isaac-net/blob/main/docs/backends-mjx.md), the [5G-LENA validation](https://github.com/ZzZTripleZzZ/isaac-net/blob/main/docs/validation-5g-lena.md) with the [formal comparison](https://github.com/ZzZTripleZzZ/isaac-net/blob/main/docs/fidelity-vs-lena.md) and the [load-gap study](https://github.com/ZzZTripleZzZ/isaac-net/blob/main/docs/fidelity-load-gap.md), the [ns-3 bridges](https://github.com/ZzZTripleZzZ/isaac-net/blob/main/docs/bridges.md), the [OAI rfsim bridge](https://github.com/ZzZTripleZzZ/isaac-net/blob/main/docs/bridges-oai.md), the [public-data calibration](https://github.com/ZzZTripleZzZ/isaac-net/blob/main/docs/calibration-public-data.md) and the [real-network measurement protocol](https://github.com/ZzZTripleZzZ/isaac-net/blob/main/docs/measurement-protocol.md).

## Contributing

Contributions are welcome. Collaborators should start from [CONTRIBUTING.md](https://github.com/ZzZTripleZzZ/isaac-net/blob/main/CONTRIBUTING.md), the open roadmap items and the starter tasks in [docs/STATUS.md](https://github.com/ZzZTripleZzZ/isaac-net/blob/main/docs/STATUS.md). Changes to the prototype levels go into the eager reference first, and the `graph` backend must stay bitwise equal to it. New MAC and PHY modelling goes into the NR engine. [RELEASE.md](https://github.com/ZzZTripleZzZ/isaac-net/blob/main/RELEASE.md) describes how a release is cut.

## Citation

The paper describing Isaac-Net is on arXiv: [Network-in-the-Loop at Scale: GPU-Batched 5G Simulation for Massively Parallel Robot Learning](https://arxiv.org/abs/2610.02370) (arXiv:2610.02370, [PDF](https://arxiv.org/pdf/2610.02370)). Zifan Zhang and Mingzhe Han contributed equally.

```bibtex
@article{zhang2026isaacnet,
  title   = {Network-in-the-Loop at Scale: GPU-Batched 5G Simulation for Massively Parallel Robot Learning},
  author  = {Zhang, Zifan and Han, Mingzhe and Athreya, Kannan and Liu, Yuchen},
  journal = {arXiv preprint arXiv:2610.02370},
  year    = {2026},
  note    = {Zifan Zhang and Mingzhe Han contributed equally}
}
```

## License

BSD-3-Clause, see [LICENSE](https://github.com/ZzZTripleZzZ/isaac-net/blob/main/LICENSE). The shipped Sionna tables are Apache-2.0 (`isaac_net/core/data/LICENSE-sionna-Apache-2.0`). No ns-3 or 5G-LENA code or data and no OAI configuration files are included. [docs/licensing.md](https://github.com/ZzZTripleZzZ/isaac-net/blob/main/docs/licensing.md) covers the locally generated 5G-LENA tables, the GPL status of binaries built from the ns-3 bridge programs, Isaac Sim and the Omniverse EULA, and the licenses of the public datasets used for calibration.
