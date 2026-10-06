# Cookbook

Ten short recipes for tasks that come up in the first week. Each one is a complete script: copy it into a file and run it. They use tiny sizes and pick the device themselves, so every recipe runs on a CPU in seconds (the whole page in about a minute). On a GPU, raise `E` and `R` and switch the backend as noted under each recipe. [Choosing a configuration](choosing.md) explains the choices behind them, and the [FAQ](faq.md) answers the questions they raise.

| # | Recipe | Needs |
|:---|:---|:---|
| 1 | [Network delay as an observation in a Direct env](#1-network-delay-as-an-observation-in-a-direct-env) | CPU; the real env needs Isaac Lab 3.0 |
| 2 | [Domain-randomize the radio per episode](#2-domain-randomize-the-radio-per-episode) | CPU |
| 3 | [Record KPIs to Parquet and plot a delay CDF](#3-record-kpis-to-parquet-and-plot-a-delay-cdf) | CPU, `pandas`, `pyarrow`, `matplotlib` |
| 4 | [Compare two fidelity levels on the same seeds](#4-compare-two-fidelity-levels-on-the-same-seeds) | CPU |
| 5 | [The 5G-LENA-validated configuration and the scale configuration](#5-the-5g-lena-validated-configuration-and-the-scale-configuration) | CPU; the validated run needs the locally generated 5G-LENA tables, `triton` needs a GPU |
| 6 | [Three cells with handover and radio link failure](#6-three-cells-with-handover-and-radio-link-failure) | CPU (`reference` or `graph`; `triton` refuses several cells) |
| 7 | [Obstacles: line of sight from a synthetic radio map](#7-obstacles-line-of-sight-from-a-synthetic-radio-map) | CPU |
| 8 | [QoS classes for commands and video](#8-qos-classes-for-commands-and-video) | CPU |
| 9 | [Shard the envs over two GPUs](#9-shard-the-envs-over-two-gpus) | two GPUs; runs on two CPU shards here |
| 10 | [Check a config before a long run](#10-check-a-config-before-a-long-run) | CPU |

## 1. Network delay as an observation in a Direct env

`NetEnvMixin` adds `net_setup`, `net_step`, `net_reset` and `net_obs` to an Isaac Lab `DirectRLEnv`, and `IsaacNetCfg.obs_features` picks what `net_obs` returns. The stand-in class below needs only `num_envs`, `device` and `episode_length_buf`, so the same hooks run without Isaac Lab. In a real task, write `class MyEnv(NetEnvMixin, DirectRLEnv)` and call the hooks from `_setup_scene`, `_get_dones`, `_reset_idx` and `_get_observations` ([Isaac Lab](isaac-lab.md#configuring-the-network)).

```python
import torch
from isaac_net import NRConfig
from isaac_net.isaac import IsaacNetCfg, NetEnvMixin

E, R = 8, 4
NR = NRConfig(msg_sizes=(4000.0, 30000.0))
ISAAC = IsaacNetCfg(obs_features=("aoi", "msg_delay", "delay_history"), obs_history=4)

class StandIn(NetEnvMixin):                     # in Isaac Lab: class MyEnv(NetEnvMixin, DirectRLEnv)
    num_envs, device = E, "cpu"
    episode_length_buf = torch.ones(E, dtype=torch.long)

env = StandIn()
env.net_setup("L2-legacy", R, NR, "reference", isaac=ISAAC, seed=0)   # "triton" on a GPU
for _ in range(10):
    env.net_step(torch.rand(E, R, 3) * 60.0, torch.randint(0, 3, (E, R)))   # end-of-step poses, sends
obs = env.net_obs()                             # [E, R, ISAAC.obs_dim(NR)], each feature scaled to [0, 1]
print(obs.shape, ISAAC.obs_dims(NR))
```

`msg_delay` gives the delay of each delivered message slot (16 values per robot), and `delay_history` the delays of the last 4 delivered messages, both divided by the observation time scale (5 s by default). Size the env's `observation_space` with `ISAAC.obs_dim(NR)`.

## 2. Domain-randomize the radio per episode

`IsaacNetCfg.dr_ranges` redraws per-env radio parameters at every reset of that env, uniformly in `[lo, hi]`, and leaves the other envs alone. `NetModule.dr_support` tells which keys the level honors: the radio keys need a level that reads the SNR, such as `L1`, `L2-legacy` or `L2`.

```python
import torch
from isaac_net import NRConfig
from isaac_net.isaac import IsaacNetCfg, NetModule, TrafficRequest

E, R = 8, 4
isaac = IsaacNetCfg(dr_ranges={"noise_dbm": (-95.0, -85.0), "pl_exp": (2.8, 4.0), "shadow_sigma_db": (3.0, 9.0)})
net = NetModule("L2-legacy", E, R, "cpu", NRConfig(), "reference", isaac=isaac, seed=0)
print({k: ok for k, (ok, _) in net.dr_support.items()})

done = torch.tensor([1, 5])                     # envs whose episode ended
net.reset(done)                                 # redraws dr_ranges for envs 1 and 5 only
print("path-loss exponent per env:", [round(x, 2) for x in net.radio.params()["pl_exp"].tolist()])
net.submit(None, TrafficRequest(send=torch.ones(E, R, dtype=torch.long)))
out = net.step(None, torch.rand(E, R, 3) * 60.0)
print("SINR of env 1 (dB):", out["sinr_db"][1].round(decimals=1).tolist())
```

Inside Isaac Lab the mixin calls `net_reset`, which calls `NetModule.reset`, so the same `dr_ranges` apply without extra code. The event term `isaac_net.isaac.mdp.randomize_network` does the same draw from an Isaac Lab `EventTermCfg` ([Tutorial 04](tutorials/04_isaac_lab_integration.ipynb)).

## 3. Record KPIs to Parquet and plot a delay CDF

The step dict carries per-message outcomes, and `counters()` on the NR engine carries cumulative MAC counters (transport blocks, HARQ retransmissions, PRBs used). One row per delivered message and one row per step go into two tables. Parquet needs `pandas` and `pyarrow` (`pip install pandas pyarrow matplotlib`).

```python
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, pandas as pd, torch
from isaac_net import NRConfig, Requests, make_engine

E, R, T = 16, 8, 40
cfg = NRConfig()
net = make_engine("L2", E, R, "cpu", cfg, seed=0)          # "triton" on a GPU
pos, msgs, steps = torch.rand(E, R, 2) * 120.0, [], []
for t in range(T):
    net.submit(None, Requests((torch.rand(E, R) < 0.3).long() * torch.randint(1, 3, (E, R))))
    out = net.step(None, pos)
    m = out["delivered"]
    msgs.append(pd.DataFrame({"step": t, "env": m.nonzero()[:, 0].numpy(), "cls": out["cls"][m].numpy(),
                              "delay_ms": (out["delay"][m] * cfg.control_step_ms).numpy()}))
    ul = net.counters()["ul"]
    steps.append({"step": t, "delivered": int(m.sum()), "queued": int(out["queue_len"].sum()),
                  "tb_fail_ratio": ul["tb_fail"] / max(ul["tb_new"] + ul["tb_retx"], 1.0),
                  "prb_use": ul["prb_used"] / max(ul["prb_avail"], 1.0)})
msgs, steps = pd.concat(msgs), pd.DataFrame(steps)
msgs.to_parquet("messages.parquet"); steps.to_parquet("steps.parquet")
d = np.sort(msgs["delay_ms"].to_numpy())
plt.step(d, np.arange(1, len(d) + 1) / len(d), where="post")
plt.xlabel("delay (ms)"); plt.ylabel("CDF"); plt.savefig("delay_cdf.png", dpi=120)
print(steps.tail(3))
```

The counters are cumulative since the last full reset, so the ratios above are running ratios. The delays come in control steps and are converted with `control_step_ms`. On a GPU, call `.cpu()` before `.numpy()`, and log every k steps if the host copy shows up in your step time.

## 4. Compare two fidelity levels on the same seeds

Every level has the same API, so a comparison only needs the same traffic, positions and seeds for each. Draw the inputs once from a fixed generator, and pass `seed=` to `make_engine`, which keys every engine draw by (seed, env id, episode).

```python
import torch
from isaac_net import NRConfig, Requests, make_engine

E, R, T = 8, 8, 60
g = torch.Generator().manual_seed(1)
sends = (torch.rand(T, E, R, generator=g) < 0.4).long() * torch.randint(1, 3, (T, E, R), generator=g)
pos = torch.rand(E, R, 2, generator=g) * 120.0

def run(level, seed):
    net = make_engine(level, E, R, "cpu", NRConfig(), seed=seed)    # "triton" for both levels on a GPU
    delays = []
    for t in range(T):
        net.submit(None, Requests(sends[t]))
        out = net.step(None, pos)
        delays.append(out["delay"][out["delivered"]] * 100.0)      # ms at the default 100 ms step
    d = torch.cat(delays)
    return len(d), d.quantile(torch.tensor([0.5, 0.95])).round().tolist()

for seed in (0, 1, 2):
    print(seed, "L1", run("L1", seed), "| L2-legacy", run("L2-legacy", seed))
```

Report the spread over seeds next to the difference between levels, and treat a difference smaller than that spread as no difference. `L1` shares every slot equally among the robots with data and has no HARQ, link adaptation or scheduling requests, so the two levels can disagree in either direction: in this small run `L1` delivers more messages but with a similar or higher median delay.

## 5. The 5G-LENA-validated configuration and the scale configuration

`NRConfig()` is not validated: its replay of the 5G-LENA sweep puts the median delay 29–76% low. `lena_validation_v2()` is the validated configuration, and "v2 minus BSR" replaces its SR / BSR grant pipeline with the lumped 40-slot SR-to-grant delay, the closest validated configuration that the fused `triton` kernel accepts ([fidelity-vs-lena.md](fidelity-vs-lena.md#scale-configurations)).

```python
import dataclasses, os, torch
from isaac_net import NRConfig, Requests, make_engine
from isaac_net.core import lena_tables_path, lena_validation_v2
from isaac_net.core.nr_fast import NRTritonEngine

v2 = lena_validation_v2()
scale = lena_validation_v2(ul_grant_model="lumped", sr_grant_delay_slots=40, frame_buffer=16)  # "v2 minus BSR"
diff = {f.name: (getattr(v2, f.name), getattr(scale, f.name)) for f in dataclasses.fields(NRConfig)
        if getattr(v2, f.name) != getattr(scale, f.name)}
print("v2 vs scale:", diff)
print("triton refuses v2:", [r[0] for r in NRTritonEngine.refusals(v2)])
print("triton refuses scale:", NRTritonEngine.refusals(scale))
have_tables = os.path.exists(lena_tables_path())
if not have_tables:   # GPL tables absent: run the same MAC on the shipped Sionna tables, NOT the validated config
    v2, scale = (c.with_(bler_source="pdsch", tbs_mode="38214", harq_combining="cc") for c in (v2, scale))
for name, cfg in (("v2", v2), ("scale", scale)):
    net = make_engine("L2", 4, 8, "cpu", cfg.with_(msg_sizes=(4000.0, 30000.0)), seed=0)  # scale: "triton" on a GPU
    for _ in range(10):
        net.submit(None, Requests(torch.randint(0, 3, (4, 8))))
        out = net.step(None, snr_db=torch.full((4, 8), 15.0))
    print(name, "validated tables" if have_tables else "Sionna stand-in", int(out["queue_len"].sum()), "queued")
```

The two configurations differ only in how a robot gets its first uplink grant and in the buffer depth. The grant pipeline mostly shapes the delay tail at light and moderate load: without it the p95 delay error is −8.9% and −14.5% against −2.5% and −6.1% for v2. The 5G-LENA BLER tables are GPL-2.0 data and are never shipped, so generate them once with `python -m isaac_net.tools.extract_lena_tables <your 5G-LENA checkout>` before you report numbers from either configuration.

## 6. Three cells with handover and radio link failure

`multicell(3)` places three gNBs 100 m apart with interference, uplink power control and A3 handover. `rlf=True` adds radio link failure and re-establishment after TS 38.331 on level `L2`. A multi-cell engine needs robot positions (not an SNR), and it reports the serving cell of every robot.

```python
import torch
from isaac_net import Requests, make_engine
from isaac_net.core import multicell

E, R = 4, 8
net = make_engine("L2", E, R, "cpu", multicell(3, rlf=True), seed=0)   # "graph" on a GPU; triton refuses n_cells > 1
pos = torch.rand(E, R, 2) * 150.0
vel = torch.randn(E, R, 2) * 2.0                                      # m per control step
for _ in range(40):
    net.submit(None, Requests(torch.ones(E, R, dtype=torch.long)))
    out = net.step(None, pos)
    pos = (pos + vel).clamp(0.0, 150.0)
print("serving cell of env 0:", out["serving_cell"][0].tolist())
print("robots in RLF now:", int(out["rlf"].sum()))
print("handovers per robot (env 0):", net.assoc.n_ho[0].tolist())
print("RLF counters:", net.counters()["rlf"])
```

`net.assoc.n_ho` and `net.assoc.n_rlf` count handovers and failures per robot and episode. In this small arena every robot has a usable cell, so the RLF counters stay at zero. They move when robots spend steps below `rlf_qout_db`, for example at cell edges with obstacles. The timers and thresholds (`rlf_qout_db`, `t310_ms`, `reest_delay_ms`, and the A3 fields) are in [Multi-cell networks](multicell.md#radio-link-failure).

## 7. Obstacles: line of sight from a synthetic radio map

`make_synthetic_radio_map --obstacles` writes a 40 × 24 m warehouse hall with four rows of racks, two gNBs on the short walls, and the per-cell line-of-sight probability `los_prob` and the obstacle height map `obstacle_z`. With `los_source="map"` the engine draws each link's LOS state from `los_prob`, and with `"raycast"` it marches the link over `obstacle_z`. No USD scene or Sionna is needed.

```python
import subprocess, sys, torch
from isaac_net import NRConfig, Requests, make_engine
from isaac_net.tools.make_synthetic_radio_map import OBSTACLE_GNBS

subprocess.run([sys.executable, "-m", "isaac_net.tools.make_synthetic_radio_map", "--obstacles", "hall.npz"], check=True)
cfg = NRConfig(channel="radio_map", radio_map_path="hall.npz", n_cells=2, cell_layout="custom",
               cell_positions_m=tuple(g[:2] for g in OBSTACLE_GNBS), los_source="raycast", los_raycast_samples=64)
E, R = 4, 16
net = make_engine("L2", E, R, "cpu", cfg, seed=0)
pos = torch.rand(E, R, 2) * torch.tensor([40.0, 24.0])   # robots anywhere in the hall
net.submit(None, Requests(torch.ones(E, R, dtype=torch.long)))
out = net.step(None, pos)
print("LOS share of the serving links:", out["los"].float().mean().item())
print("mean SINR, LOS vs NLOS (dB):", out["sinr_db"][out["los"]].mean().item(), out["sinr_db"][~out["los"]].mean().item())
```

On a ray-traced radio map the LOS state adds no path loss, because the map already holds it. The state then drives the outputs, the Rician K-factor and the observation feature `los` ([Obstacles and NLOS](obstacles.md)). `los_raycast_samples=64` resolves the 1 m racks exactly on these 40 m links, while 32 samples step over a few of them.

## 8. QoS classes for commands and video

`scheduler="qos"` weights proportional-fair scheduling by a 3GPP priority level per message class, after 5G-LENA's QoS scheduler, and a finite packet delay budget makes a class delay-critical. Here class 0 carries 200-byte commands with a 20 ms budget and class 1 carries 30 kB video frames.

```python
import torch
from isaac_net import NRConfig, Requests, make_engine

E, R, T = 4, 16, 30
cfg = NRConfig(msg_sizes=(200.0, 30000.0), qos_priority=(10, 70), qos_pdb_ms=(20.0, float("inf")))
for sched in ("pf", "qos"):
    net = make_engine("L2", E, R, "cpu", cfg.with_(scheduler=sched), seed=0)   # every L2 backend runs "qos"
    d = {0: [], 1: []}
    for t in range(T):
        send = torch.full((E, R), 1 if t % 3 == 0 else 2)       # a command every third step, video otherwise
        net.submit(None, Requests(send), priority=send - 1)     # class 0 = commands, class 1 = video
        out = net.step(None, snr_db=torch.full((E, R), 12.0))
        for c in (0, 1):
            d[c].append(out["delay"][out["delivered"] & (out["priority"] == c)] * 100.0)
    p50 = [round(torch.cat(d[c]).median().item()) for c in (0, 1)]
    print(f"{sched}: commands delivered {len(torch.cat(d[0]))}, median delay (ms) commands / video {p50}")
```

The video saturates the uplink, so under `"pf"` each command waits in its robot's queue behind the video frames submitted before it (median about 800 ms in this run). Under `"qos"` the scheduler moves the command ahead of the unsent video once per step and ranks robots by class weight, which cuts the command median to under 200 ms and delivers about twice as many commands ([QoS scheduling](configurability.md#qos-scheduling)).

## 9. Shard the envs over two GPUs

`ShardedEngine` splits the envs over several devices behind the API of one engine. Inputs are sliced per shard and outputs come back on the first device. With the default engine-owned random streams, two shards are bitwise equal to one engine of the same total size on the same device type (except `L2` with traffic models or background users).

```python
import torch
from isaac_net import NRConfig, Requests, make_engine
from isaac_net.core import ShardedEngine

devs = ["cuda:0", "cuda:1"] if torch.cuda.device_count() >= 2 else ["cpu", "cpu"]
E, R = 16, 8
shard = ShardedEngine("L2-legacy", E, R, devs, NRConfig(), "reference", seed=0)   # "triton" on GPUs
one = make_engine("L2-legacy", E, R, devs[0], NRConfig(), "reference", seed=0)
pos = torch.rand(E, R, 2, device=devs[0]) * 120.0
for _ in range(20):
    send = Requests((torch.rand(E, R, device=devs[0]) < 0.4).long())
    a, b = shard.submit(None, send), one.submit(None, send)
    oa, ob = shard.step(None, pos), one.step(None, pos)
print("shard invariant:", shard.shard_invariant, "| same delivered:", torch.equal(oa["delivered"], ob["delivered"]))
shard.reset(torch.tensor([3, 12]))             # global env ids; each shard resets its own
```

The shards run one after another from one Python thread, and since the engine steps have no host syncs, the kernels on different GPUs overlap. On a CPU the float outputs of `L1` and `L2-legacy` may differ from one engine in the last bit, because CPU math kernels can round differently for a different tensor shape ([Background users, energy and sharding](background-energy-sharding.md)).

## 10. Check a config before a long run

Every level reads only some `NRConfig` fields. `unused_fields(level)` lists the fields you set away from their defaults that the level would ignore, and `make_engine(..., strict=True)` raises instead of ignoring them. For `triton`, `NRTritonEngine.refusals(cfg)` lists what the fused kernel does not implement, before you allocate anything.

```python
import torch
from isaac_net import NRConfig, make_engine
from isaac_net.core.nr_fast import NRTritonEngine

cfg = NRConfig(mcs_table=2, n_harq=8, pathloss_exp=3.2, l0_loss=0.05)
for level in ("L0", "L1", "L2-legacy", "L2"):
    print(f"{level:9s} ignores", cfg.unused_fields(level))
try:
    make_engine("L2-legacy", 4, 4, "cpu", cfg, strict=True)
except ValueError as err:
    print("strict:", err)
print("triton refuses:", NRTritonEngine.refusals(cfg.with_(n_cells=3, cell_layout="hex")))
print(cfg.summary())
```

Run with `strict=True` in every experiment script: a typo-free field that a level silently ignores is the most common reason two runs that "should differ" do not. `summary()` prints the derived quantities (PRBs, RBG size, slots per control step) in one line for the run log.
