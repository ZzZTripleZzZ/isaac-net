# Lockstep ns-3 / 5G-LENA bridges for FleetEnv and Isaac

> **In isaac_net.** The Python side is `isaac_net.bridges.ns3_lockstep` (`py/ns3bridge/` below; `from
> isaac_net.bridges.ns3_lockstep.lockstep_net import Ns3Net`), the C++ server and build scripts are this
> directory (`cpp/` and `scripts/` below), and the checks are `tests/bridges/lockstep/`. Ns3Net follows the
> package's NetBase: per-env clocks with full resets only (`reset(env_ids)` raises). Paths below are the lab-box
> layout; `NS3BRIDGE_ROOT` (build root with `bin/` and the ns-3.48 copy) and `NS3_TOOLCHAIN_ENV` override them.
> The C++ program is our own code, written against the ns-3 / 5G-LENA APIs; no ns-3 or 5G-LENA source is included.

Two co-simulation bridges between a robot simulator and ns-3.48 + 5G-LENA NR v5.1, advancing ns-3
exactly one 100 ms control step per call. Both speak the same framed binary protocol and run the same
ns-3 program (`cpp/netslot-bridge.cc`); they differ only in transport.

| Variant | Transport | Where the robot side may run | Server flag |
|---|---|---|---|
| 1. TCP lockstep (ns3-gym style) | TCP, one connection per ns-3 process | anywhere, incl. Windows -> WSL2 via localhost forwarding | `--bridge=tcp:PORT` |
| 1b. Unix-socket lockstep | AF_UNIX stream socket | same WSL instance | `--bridge=unix:PATH` |
| 2. ns3-ai shared memory | ns3-ai msg-interface (Boost managed shm, spin semaphores) | same WSL instance | `--bridge=shm:NAME` (binary `netslot-bridge-ai`) |

The scenario is the ns3ref `netslot-ref` configuration (single gNB at (0,0,10 m), 20 MHz / 50 PRB,
TDD DDDSU, mu = 1, 40 + 35 log10(d) path loss + shadowing, 3GPP UMi NLOS fading, UL OFDMA PF, HARQ,
EESM IR error model, RLC UM with 2 s PDCP discard). Code is copied verbatim, so a bridge run
reproduces a standalone netslot-ref run bit for bit (tests/test_correctness.py).

## Layout

```
cpp/netslot-bridge.cc        ns-3 server (scenario + lockstep loop + 3 transports)
cpp/pyshm/ns3ai_bridge_py.cc pybind11 module: Python side of the ns3-ai shm transport
py/ns3bridge/protocol.py     wire format (numpy)
py/ns3bridge/transport.py    StreamConn (tcp/unix), ShmConn (ns3-ai)
py/ns3bridge/core.py         Ns3Lockstep: spawns/drives E envs, one step per call
py/ns3bridge/lockstep_net.py Ns3Net(NetBase): drop-in network for FleetEnv / train.py / evaluate()
py/ns3bridge/netmodule_ns3.py Ns3NetModule: isaac/net_module.py API (reset / submit / step(t, poses, cur_tag) -> dict)
py/win_client.py             stdlib-only Windows client + round-trip benchmark
scripts/build_bridge.sh      build netslot-bridge (add "ai" for netslot-bridge-ai)
scripts/build_pyshm.sh       build the ns3-ai Python module for the i5g interpreter
scripts/serve_wsl.sh         start E TCP servers in WSL for an external (Windows) client
scripts/ai_compat_build.sh   ns3-ai-as-contrib-module compatibility build (separate tree)
tests/                       correctness, multicell, closed loop, NetModule API, speed
```

Lab box: `/home/zzhang66/experiments/bridge_lockstep` (this directory plus `ns-3.48/`, a `cp -a` of
ns3ref's tree including `build/`, `bin/`, `results/`, `logs/`).

## Build (WSL, lab box)

```bash
cd /home/zzhang66/experiments/bridge_lockstep
bash scripts/build_bridge.sh        # -> bin/netslot-bridge      (about 15 s)
bash scripts/build_bridge.sh ai     # -> bin/netslot-bridge-ai   (needs ns3-ai-src/ and /usr/include/boost)
bash scripts/build_pyshm.sh         # -> bin/ns3ai_bridge_py.cpython-311-x86_64-linux-gnu.so
```

The server compiles against the copied `ns-3.48/build/include` and links the copied
`ns-3.48/build/lib/libns3.48-*-optimized.so` with the same flags ns-3's CMake uses, so nothing in
ns-3 is reconfigured or rebuilt. The toolchain is ns3ref's conda env, used read-only
(`envrc.sh`). The forwarding headers in the copied `build/include/ns3` were repointed from the
ns3ref source tree to the copy. At run time set
`LD_LIBRARY_PATH=$PWD/ns-3.48/build/lib:/home/zzhang66/experiments/ns3ref/env/lib` (the Python
launcher does this).

ns3-ai: `ns3-ai-src/` is `hust-diangroup/ns3-ai` main at b8c9858 (2025-01-23, after v1.2.0). Only its
header-only msg-interface is used by the bridge. See the report for the full contrib-module build.

## Run

FleetEnv / train.py / evaluate() (WSL, i5g env): construct the net and pass it to FleetEnv.

```python
import sys; sys.path.insert(0, "/home/zzhang66/experiments/bridge_lockstep/py")
from ns3bridge.lockstep_net import Ns3Net
net = Ns3Net(E, R, device, TASK_SIZES["T1"], mode="procs", transport="tcp")   # or "unix" / "shm"
env = FleetEnv(E, R, net, device)      # no edits to env.py: Ns3Net binds to the calling FleetEnv
...
net.close()
```

- `mode="procs"`: one ns-3 process per env (parallel). `mode="single"`: one process, E independent
  cells. `mode="groups", envs_per_proc=k`: E/k processes with k cells each.
- `shadow="env"` (default): each step Ns3Net sends per-UE shadowing derived from FleetEnv's `snr_db`,
  so ns-3's large-scale SNR equals FleetEnv's `Radio` exactly. `shadow="ns3"` keeps ns-3's frozen
  per-UE draw.
- FleetEnv.reset() -> NetBase.reset() -> the next step sends RESET with the new episode's poses:
  ns-3 tears the world down and rebuilds it in-process (Simulator::Destroy, RNG stream counter and
  IPv4 address pool reset), with run number base + episode * E + env.
- Suggested merge into `netfactory.py` (not applied, repo root is read-only for me):

```python
    if rung == "NS3":
        sys.path.insert(0, "/home/zzhang66/experiments/bridge_lockstep/py")
        from ns3bridge.lockstep_net import Ns3Net
        return Ns3Net(E, R, device, sizes, mode=os.environ.get("NS3_MODE", "procs"),
                      transport=os.environ.get("NS3_TRANSPORT", "tcp"))
```

Isaac (NetModule API):

```python
from ns3bridge.netmodule_ns3 import Ns3NetModule, NetConfig, TrafficRequest
net = Ns3NetModule(NetConfig(num_envs=E, num_robots=R, device="cuda", msg_sizes=(4000., 30000.)),
                   transport="tcp", mode="procs")                 # in WSL: spawns the servers
# Windows side (servers already started in WSL with scripts/serve_wsl.sh E R 57100):
net = Ns3NetModule(cfg, transport="tcp", spawn=False, endpoints=[f"tcp:{57100+e}" for e in range(E)])
net.submit(None, TrafficRequest(send[E, R], tag[E, R]))
out = net.step(None, poses_local[E, R, 3], cur_tag[E], blocked=blocked[E, R, 1])     # dict, as NetModule.step
```

NetConfig and TrafficRequest are the isaac layer's classes (`isaac_net.isaac.net_module`), re-exported. Poses are
env-local END-of-step poses; ns-3 moves each UE linearly from its previous commanded end pose in 4 sub-steps (the
counterpart of NetModule's `pose_chunks`). `blocked` adds 20 dB for that step. `out["ns3"]` carries the raw per-UE
ns-3 statistics. The earlier form `net.step(poses, TrafficRequest(send), blocked=...)` still works and returns a
`NetOutput` with attribute access (`out.ns3`).

Windows without numpy/torch: `py/win_client.py` (stdlib only) implements the client (`Ns3Client`)
and a round-trip benchmark:

```
WSL:     bash scripts/serve_wsl.sh 4 16 57100
Windows: C:\bridge_lockstep\py\python.exe C:\bridge_lockstep\client\win_client.py --ports 57100,57101,57102,57103
```

## Message schema (version 1)

All messages: header `<u32 magic=0x3142534E ("NSB1"), u32 type, u32 payload_len>` + payload,
little endian, packed. N = nEnv * nUe UEs of the process, UE index i = env * nUe + ue.
Times are ns-3 simulation seconds. Control step k covers [t0 + k * step, t0 + (k+1) * step), with
t0 = `--appStart` (0.5 s, after the ideal attach) and step = `--periodMs` (100 ms).

| type | direction | payload |
|---|---|---|
| 1 HELLO | server -> client, on connect and after RESET | `u32 version, u32 nEnv, u32 nUe, u32 run, f64 t0_s, f64 step_s` |
| 2 STEP | client -> server | `i32 t (client step, echoed), u32 flags, u32 nFrames`, `f32 pos[N][3]` (env-local x, y, z; NaN x = keep, NaN z = `--ueHeight`), if flags & 1: `f32 shadow_db[N]` (added to path loss; NaN = keep), `nFrames x {u16 env, u16 ue, u32 fid, u32 bytes}` |
| 3 RESULT | server -> client | `i32 t, u32 N, u32 nDone, u32 k (steps since build), f64 t0_s (start of this step), f64 wall_run_s (inside Simulator::Run), f64 wall_io_s (rest of the server's step handling)`, then `f32 sinr_db[N], f32 rsrp_dbm[N], f32 rlc_bytes[N], f32 mcs[N], u32 n_tb[N], u32 n_retx[N], u32 n_corrupt[N], u32 n_tb_lost[N], u32 ok_bytes[N]`, then `nDone x {u16 env, u16 ue, u32 fid, f64 t_done_s}` |
| 4 RESET | client -> server | `u32 run, u32 flags` (1: pos follows, 2: shadow follows), `f32 pos[N][3]`, `f32 shadow_db[N]`; the server rebuilds the scenario and answers HELLO |
| 5 CLOSE | client -> server | empty; the server exits |

STEP flags: 1 = shadow present, 2 = positions are end-of-step waypoints (4 linear sub-steps).
Frames in a STEP are handed to the UE's UDP socket at the step's first instant t0 + k * step, as
ceil(bytes / 1400) packets carrying an 18-byte frame header. A frame is done when its last packet
reaches the remote host. Frames with a lost segment (RLC UM after 4 HARQ attempts, PDCP discard)
never complete; the client applies its own 2 s timeout (NetBase semantics). Per-UE stats are over
this step's UL transport blocks at the gNB (`RxPacketTraceGnb`, de-duplicated): mean SINR (linear
average, in dB; NaN when the UE sent nothing), mean MCS, TB count, retransmissions (rv > 0),
corrupted TBs, TBs lost after the 4th transmission, bytes of correct TBs. `rsrp_dbm` is the UE's
last DL RSRP report and `rlc_bytes` the RLC buffer at the UE's last MAC SR/BSR event.

Lockstep timing: after each step the server stops 1 ns before the next step boundary, so the frames
of step k+1, which the client sends only after it has seen the result of step k, are released by
the sender's periodic tick event at exactly t0 + (k+1) * step, in the same event order as in the
standalone netslot-ref.
