# OAI 5G rfsim bridge (full-stack validation)

The ns-3 bridges ([bridges.md](bridges.md)) check the GPU engine against another simulator. This bridge checks it against real protocol software: OpenAirInterface (OAI) 5G with its core network, gNB and nr-UE stacks, connected through OAI's RF simulator (rfsim) instead of radios. Every packet goes through the UE's SDAP, PDCP, RLC and MAC, SR and BSR procedures, the gNB's scheduler, HARQ and link adaptation, and the core's UPF. The radio channel, however, is simulated, and no base-station hardware is involved.

Like the ns-3 bridges it is a **validation tool and is never part of the training loop**. It runs in lockstep with a control loop at 10 Hz and one UE per robot, so it serves a handful of robots (1–10 UEs were tested), for example one environment of up to ten robots.

The code is in `isaaclab_net/bridges/oai/`, the measurement scripts are in `benchmarks/oai/`, and the tables below are in `benchmarks/oai/results/`.

## Summary

- **The stack runs on the lab box.** OAI 2026.w39 and CN5G v2.2.1 run from the official Docker images in WSL2 with Docker Engine 26.1, after three environment fixes (memlock limit, no real-time threads, network ordering; see [Setup](#setup)). The rfsim cell matches the engine's `lena_match` frame: 20 MHz (51 PRB) at 30 kHz, TDD DDDSU with a (10, 2, 2) special slot. Up to 10 UEs attach (the limit of the test subscriber database).
- **rfsim is not real time.** Virtual (air) time ran 0.35–2.0× as fast as wall time on the shared host, depending on the load of other jobs. The bridge therefore takes its clock from the gNB's per-slot T-tracer ticks and reports every delay in virtual time. Wall-clock medians of the same frames were up to 3.4× the virtual ones when rfsim ran slow, and 0.6× when it ran fast.
- **Uplink access is scheduling-request driven and slow.** With the stock MAC configuration, a 100-byte packet sent at 10 Hz takes 21.2 ms at the median (p95 27.7 ms). The trace splits it into 14.1 ms from send to the first PUSCH carrying data (SR period 5 ms, then a 6-slot K2), a second transport block 5–7.5 ms later because the SR grant holds only 63–76 bytes, and about 1 ms from the last PUSCH to the sink. With a grant every TDD period (`ulsch_max_frame_inactivity = 0`) the median falls to 3.6 ms.
- **HARQ.** Retransmissions follow the failed PUSCH after 10 slots (5 ms, two TDD periods). OAI's link adaptation keeps the first-transmission BLER at 6–10%, inside its 5–15% target band, by lowering the MCS as the uplink is attenuated, until the link breaks at 35 dB.
- **Contention.** With 1–10 UEs at 0.32 Mbit/s each, the median delay stays at 23–27 ms and Jain's index at 1.0, while the p99 grows from 61 ms to 483 ms. At 2.4 Mbit/s per UE the cell saturates from two UEs on (p95 0.6 s at N = 2, 1.3 s at N = 4), far below the 14 Mbit/s the PHY could carry at MCS 28.
- **Against the engine.** The fitted `oai_rfsim` preset (`benchmarks/oai/presets/oai_rfsim.json`) matches the stock stack's single-UE small-frame delays to a median Wasserstein-1 distance of 2.5 ms (p50 error +1.2 ms), against 13.3 ms for `oai_like` and 13.2 ms for `lena_match`, which are both about 12 ms too fast because they have no SR-driven access of this length. `oai_like` instead matches OAI run with a grant every TDD period (W1 0.8–1.0 ms for 100-byte frames). The measured SR-to-data pipeline, about 12–14 ms, is shorter than the 20 ms stand-in that the 5G-LENA study fitted but of the same kind. No preset reproduces the multi-UE tails or the saturation at 4.8 Mbit/s.

## Setup

### Versions

| Component | Version |
|:---|:---|
| OAI RAN (gNB, nr-UE) | `oaisoftwarealliance/oai-gnb:2026.w39`, `oai-nr-ue:2026.w39` (commit `29d5fa7`, 2026-09-25) |
| OAI CN5G | `oai-amf`, `oai-smf`, `oai-upf` `v2.2.1`, `trf-gen-cn5g:latest`, `mysql:9.6` (the "mini" deployment without NRF of OAI's CI, `ci-scripts/yaml_files/5g_rfsimulator`) |
| Traffic sidecars | `python:3.11-slim` |
| Host | lab box, WSL2 Ubuntu 20.04, kernel 6.18.33 (WSL), Docker Engine 26.1.3 (native in WSL, not Docker Desktop), Docker Compose v5.5.1 (standalone binary), 32 cores shared with other jobs |
| T-tracer | `textlog` built from the same OAI checkout (`common/utils/T/tracer`, `make textlog`) |

Image digests as pulled on 2026-09-29: `oai-gnb` `sha256:42f82744c5bd`, `oai-nr-ue` `sha256:cb0ee4dce9e5`, `oai-amf` `sha256:c6dee19b65b2`, `oai-smf` `sha256:c1f1452edfc8`, `oai-upf` `sha256:03985036c247`, `trf-gen-cn5g` `sha256:3f9ab2e401d2`, `mysql` `sha256:c5df04bee1a4`, `python:3.11-slim` `sha256:e41613d42d48` (first 12 hex digits). The OAI configuration files are not copied into this repository: `deploy/make_configs.py` reads them from a checkout at tag `2026.w39` and applies the overrides of a profile.

### Recipe

```bash
# on a Linux host with Docker (here: WSL2 on the lab box)
git clone --depth 1 --branch 2026.w39 https://github.com/OPENAIRINTERFACE/openairinterface5g.git oai-src
(cd oai-src/common/utils/T/tracer && make textlog)
docker pull oaisoftwarealliance/oai-gnb:2026.w39     # and oai-nr-ue, oai-amf/smf/upf:v2.2.1, trf-gen-cn5g, mysql:9.6

python isaaclab_net/bridges/oai/deploy/make_configs.py --oai oai-src --out run --profile lena_match --n-ue 2 \
    --gnb-extra="--T_stdout 2 --T_nowait"          # T-tracer on (the bridge's clock), no wait for a tracer
(cd run && docker compose up -d)                   # UEs attach in about 40 s; `docker ps` shows them healthy

export OAI_TEXTLOG=$PWD/oai-src/common/utils/T/tracer/textlog OAI_T_MESSAGES=$PWD/oai-src/common/utils/T/T_messages.txt
python - <<'EOF'
from isaaclab_net.bridges.oai import DockerOaiStack, OaiBridge
from isaaclab_net.bridges.oai.net import OaiNet
stack = DockerOaiStack(n_ue=2)                     # agents, telnet channel control, T-tracer clock
bridge = OaiBridge(stack, step_dt=0.1, pacing="virtual", log_dir="logs/ep0")
net = OaiNet(1, 2, "cpu", (4000.0, 30000.0), bridge)   # a NetBase: submit / step like any engine
EOF
```

`make_oai_netmodule(bridge, num_robots, config=...)` returns the Isaac Lab `NetModule` (reset / submit / step dict / obs) with its engine replaced by an `OaiNet`, so an Isaac task can swap the GPU engine for the real stack. `FakeStack(n_ue, delay_ms, jitter_ms, loss, rate_bps)` stands in for OAI with local processes and a UDP relay; the tests (`tests/bridges/oai/`, run with `python -m pytest tests/bridges/oai`) use it.

MAC options go through `--macrlc KEY=VALUE`, for example `ulsch_max_frame_inactivity=0` (a grant every TDD period), `ul_min_mcs=9 ul_max_mcs=9` (fixed MCS) or `pusch_TargetSNRx10=300`.

### Environment fixes

Three things failed with the stock compose file on this host, and the deployment here works around them:

1. **Memory locking.** OAI calls `mlockall()`, and Docker's default memlock limit of 64 MB then makes later allocations fail: the gNB died with `std::bad_alloc` and the telnet server with `pthread_create: Resource temporarily unavailable`. The services set `ulimits: memlock: -1`.
2. **Real-time threads.** With `CAP_SYS_NICE`, OAI creates `SCHED_FIFO` threads, which failed with `EAGAIN` in this kernel's containers. The deployment drops the capability, so every thread runs at normal priority. This is also kinder to other jobs on a shared host.
3. **Interface order.** `interface_name` for the UPF's two networks needs Docker Engine 28.1. Network priorities give the same `eth0` (N3) / `eth1` (N6) order.

For the 51-PRB cell the nr-UE also needs `--ssb 234` (the SSB's first subcarrier above point A), which the gNB prints in its start-up log.

### The rfsim cell

| | `lena_match` profile (used for every measurement) | engine `lena_match` preset | `oai_default` profile (OAI's CI cell) |
|:---|:---|:---|:---|
| Band, carrier | n78, 3609.12 MHz | 3.5 GHz | n78, 3319.68 MHz |
| Numerology, bandwidth | μ = 1, 51 PRB (20 MHz) | μ = 1, 50 PRB | μ = 1, 106 PRB (40 MHz) |
| TDD | 2.5 ms, DDDSU, special (10 DL, 2 guard, 2 UL) | DDDSU, (10, 2, 2) | 5 ms, 7 D + mixed (6, 4, 4) + 2 U |
| K1 / K2 floor | `min_rxtxtime = 6`: K2 = 6 slots | `k2 = 2` | 6 |
| SR period | 10 slots (5 ms), offset at the first UL-capable slot (set in OAI's code, not configurable) | 5 slots | 10 slots |
| UL MCS table, range | 64QAM, 0–28 | 64QAM | same |
| UL BLER control | MCS lowered above 15%, raised below 5% BLER | OLLA off, `bler_target` 0.1 | same |
| HARQ | 16 processes, 4 transmissions | 16, 4 | same |
| Channel | rfsim AWGN, per-UE attenuation | fading off in the comparison | same |
| Core | OAI CN5G: AMF, SMF, UPF, data network behind the UPF | not modeled | same |

`min_rxtxtime = 6` is what OAI's own rfsim CI uses; its documentation notes that the rfsim UE cannot meet shorter processing times.

## How the bridge works

**Traffic.** A sidecar container shares each UE's network namespace and runs `agent.py` (standard library only). On the bridge's command it sends a frame as UDP datagrams of at most 1400 bytes from the UE's PDU session (`oaitun_ue1`) to a sink in the data network behind the UPF. Another sidecar shares the data network host's namespace and receives them with kernel timestamps (`SO_TIMESTAMPNS`). Every datagram carries the probe header of `tools/measure/probe.py`, so the logs are probe logs and go through `tools/measure/owd.py`, `ingest.py` and `calibrate.py` unchanged. The UE index rides in the header's flow field, because the UPF rewrites source addresses. All containers share the host clock, so the clock offset is exactly 0.

**Virtual clock.** A `textlog` process reads the gNB's `GNB_PHY_UL_TICK` T-tracer event (frame and slot of every received slot, with the gNB's wall time), and `VClock` interpolates between the ticks to map any wall timestamp to virtual time. `pacing="virtual"` makes a control step last 100 ms of virtual time, which keeps the offered load right in air time at any rfsim speed; `pacing="wall"` keeps Isaac's wall-clock 10 Hz instead. The logs are written twice, in wall time and mapped to virtual time (`*_vt.csv`), and the mapping for the logs is recomputed at the end from the complete tick record. Online delays are mapped when a frame completes, at most a few milliseconds past the newest tick.

**Moving UEs.** Each UE's uplink and downlink attenuation is set at run time through the telnet servers (`channelmod modify <model> ploss <dB>`: the gNB's model `rfsimu_channel_ue<k>` for the uplink of the k-th UE that attached, each UE's `rfsimu_channel_enB0` for its downlink). `OaiNet(..., snr_ref_db=S)` maps the SNR of the caller's radio to an attenuation `clip(S − SNR, 0, 60)` and updates it when it changes by 1 dB or more.

**Resets.** rfsim cannot rewind. A reset restarts the bookkeeping of the reset environments, and their frames still in flight are ignored when they arrive, as in the ns-3 bridge's single-process mode, so partial resets are allowed.

**Outputs.** `OaiNet` is a `NetBase`: its step dict has the same keys as every engine's (`delivered`, `delay` in control steps, `newest`, `timed_out`, `queue_len`, ...), and `bridge.steps` records how late each step ended and the rfsim speed.

## What rfsim can and cannot emulate

| Can | Cannot |
|:---|:---|
| The full protocol stacks: NAS attach and PDU session, SDAP/PDCP/RLC, MAC (SR, BSR, grants, PF scheduling, HARQ, BLER-driven MCS), PHY encoding and decoding on the sampled signal, GTP-U through the UPF | Real time: virtual time runs at whatever speed the host allows (0.35–2.0× here); no radio front-end timing, no late-slot or underflow behavior |
| Per-UE uplink and downlink attenuation and noise at run time, and channel models (AWGN, Rayleigh, TDL, Ricean factor) per link | A physical link budget: the attenuation scales fixed-point samples, and from about 20 dB on decoding fails in a way the reported SNR does not explain, apparently through quantization rather than the configured noise (see [HARQ](#c-harq-at-fixed-attenuations)) |
| Several UEs sharing one cell (10 attach here) | UE power control: the gNB's closed loop runs, but the rfsim UE's sample amplitude does not follow it, so the reported PUSCH SNR in `nrMAC_stats` stays at the 20 dB target while decoding degrades |
| Traces: `nrMAC_stats.log` counters and the T-tracer's per-grant and per-PUSCH events | Moving antennas, Doppler from motion, and interference from other cells (one gNB) |

Two pitfalls cost time here and are worth knowing:

- **`ploss` is a gain.** rfsim scales received samples by 10^(ploss/20) (`radio/rfsimulator/apply_channelmod.c`: "path_loss_dB should contain the total path gain"). An attenuation of L dB is `ploss = −L`. Positive values amplify the signal until the int16 samples clip, and the UE loses the link while the gNB still reports a good SNR. `DockerOaiStack.set_pathloss` takes an attenuation and sends its negative.
- **`rfsimu vtime` over telnet crashed the gNB.** Polling the rfsim sample clock at 50 Hz aborted the 2026.w39 gNB ("buffer overflow detected") after about 20 queries, apparently because the command runs in a worker thread that prints into the telnet server's shared buffer. The clock therefore comes from the T-tracer, and telnet is used only for the occasional attenuation change.

## Measurements

All runs: the `lena_match` profile, one gNB, 20 s of virtual time per run (2 s tail), UEs at 0 dB attenuation unless stated, virtual-time delays, first datagram sent to last datagram received. Three MAC configurations:

- **stock**: OAI's defaults, including `ulsch_max_frame_inactivity = 10` (a grant after 10 frames without uplink, i.e. every 100 ms);
- **sr**: `ulsch_max_frame_inactivity = 1000`, so access is by scheduling request only;
- **pp**: `ulsch_max_frame_inactivity = 0`, a grant every TDD period (2.5 ms).

`benchmarks/oai/campaign.py` runs the grid, `analyze.py` writes the tables. Each run directory has a `manifest.json` in the format of [measurement-protocol.md](measurement-protocol.md), so the campaign also goes through `ingest.py` and `calibrate.py`.

### (a) Uplink one-way delay against frame size and rate

Frame delay in ms (`owd_grid.csv`), one UE, constant bit rate unless stated:

| Config | Frame | Rate | p5 | p50 | p95 | p99 | Wall-clock p50 | rfsim speed |
|:---|---:|:---|---:|---:|---:|---:|---:|---:|
| stock | 100 B | 10 Hz | 15.3 | 21.2 | 27.7 | 29.4 | 55.2 | 0.37 |
| stock | 100 B | 50 Hz | 3.0 | 15.5 | 20.0 | 24.9 | 22.0 | 0.63 |
| stock | 100 B | 100 Hz | 3.5 | 12.1 | 19.7 | 24.9 | 23.5 | 0.43 |
| stock | 100 B | Poisson 20 Hz | 5.5 | 14.6 | 21.1 | 23.9 | 35.0 | 0.41 |
| stock | 1000 B | 10 Hz | 19.2 | 23.2 | 29.2 | 32.2 | 42.7 | 0.55 |
| stock | 1000 B | 50 Hz | 7.9 | 18.9 | 27.2 | 29.2 | 26.1 | 0.66 |
| stock | 1000 B | 100 Hz | 5.1 | 14.9 | 28.5 | 50.7 | 21.7 | 0.62 |
| stock | 4000 B | 10 Hz | 18.8 | 23.7 | 31.5 | 54.1 | 45.0 | 0.52 |
| stock | 4000 B | 50 Hz | 4.9 | 19.7 | 31.8 | 153.3 | 38.2 | 0.45 |
| stock | 30000 B | 5 Hz | 34.9 | 39.1 | 96.2 | 237.2 | 131.9 | 0.35 |
| stock | 30000 B | 10 Hz | 34.5 | 39.0 | 251.3 | 436.0 | 103.3 | 0.42 |
| stock | robot mix | 4 kB 10 Hz + 100 B 50 Hz + 30 kB 5 Hz | 3.5 | 16.0 | 40.9 | 138.2 | 40.0 | 0.38 |
| sr | 100 B | 10 Hz | 16.7 | 21.1 | 26.4 | 28.3 | 31.4 | 0.68 |
| sr | 100 B | 50 Hz | 4.0 | 14.9 | 19.5 | 22.3 | 21.6 | 0.62 |
| sr | 4000 B | 10 Hz | 19.1 | 23.2 | 27.2 | 29.4 | 43.3 | 0.56 |
| sr | 30000 B | 5 Hz | 34.7 | 38.9 | 44.5 | 47.8 | 48.7 | 0.87 |
| pp | 100 B | 10 Hz | 1.3 | 3.6 | 6.9 | 8.5 | 5.1 | 0.66 |
| pp | 100 B | 50 Hz | 2.4 | 4.5 | 7.0 | 8.3 | 4.1 | 1.08 |
| pp | 4000 B | 10 Hz | 22.8 | 24.7 | 26.7 | 27.8 | 15.2 | 1.61 |
| pp | 30000 B | 5 Hz | 165.0 | 167.3 | 169.0 | 169.4 | 100.4 | 1.66 |

Every frame of these runs was delivered. The stock and SR-only configurations agree within 0.5 ms at the median, so the 100 ms inactivity grant plays no role for traffic at 5 Hz or faster: access is by scheduling request. Faster small-frame traffic has lower delays, probably because a frame then often finds the UE still granted for the previous one (p5 of 3 ms). A grant every period removes the SR step for small frames (3.6 ms). For 4 kB it does not help (24.7 ms), because the 480-byte periodic grant carries only the start of the frame and the rest waits for the BSR round trip. The 30 kB row at `pp`, a nearly constant 167 ms, is a second effect of the periodic grants, perhaps because grants sized for an empty buffer keep taking the UE's scheduling opportunities from the grants sized by the BSR. We did not investigate this further. The 30 kB tails of the stock configuration (p95 96–251 ms) are absent from the SR-only run at 5 Hz (p95 44.5 ms); we did not trace their cause.

The wall-clock column shows why the virtual clock matters: when rfsim ran at a third of real time, the wall-clock median of 100-byte frames was 55 ms against 21 ms of air time.

### (b) The SR → grant → data pipeline

For the 100-byte frames at 10 Hz, the T-tracer places every grant (`GNB_MAC_UL`, at its DCI slot) and every decoded PUSCH (`GNB_MAC_UL_PDU_WITH_DATA`, with the logical-channel bytes of `GNB_MAC_LCID_UL`) at the air time of its slot on the same virtual axis as the probe timestamps (`pipeline.csv`, medians with quartiles):

| Component | stock | sr | pp |
|:---|:---|:---|:---|
| Send → first UL grant (DCI slot) | 11.1 ms (10.0–12.6) | 11.2 ms (9.8–13.1) | 1.3 ms (0.6–1.9) |
| Send → first PUSCH with data | 14.1 ms (13.0–15.6) | 14.2 ms (12.8–16.1) | 3.1 ms (2.1–4.3) |
| DCI → PUSCH (K2) | 6 slots (every frame) | 6 slots | not paired (unused grants are not traced) |
| First → last PUSCH with data | 7.5 ms (5.0–7.5) | 5.0 ms (5.0–7.5) | 0 (one TB) |
| Last PUSCH → sink | 1.1 ms (0.5–1.8) | 0.8 ms (0.4–1.5) | 0.4 ms (0.2–0.9) |
| Total | 21.2 ms (19.5–23.8) | 21.1 ms (18.8–23.0) | 3.6 ms (2.6–5.3) |
| PUSCHs with data per frame | 2 (2–3) | 2 (2–3) | 1 |
| TB size of the first data PUSCH | 76 B (63–76) | 76 B (63–88) | 480 B |

So the SR-driven access costs about 14 ms before the first byte is on the air: on average 2.5 ms to the next SR opportunity (period 5 ms), about 8.5 ms from there to the grant's DCI slot (UE and gNB processing, and the wait for a DL slot whose K2 of 6 lands on the UL slot), and K2 = 6 slots (3 ms). The SR grant is 5 PRBs at MCS 5–6, 63–76 bytes, too small for the 128-byte IP packet with its PDCP and RLC headers, so RLC segments it and the rest follows one or two TDD periods later. The last hop from PUSCH decoding through the UPF to the sink takes about 1 ms.

### (c) HARQ at fixed attenuations

One UE, 1000-byte frames at 50 Hz, uplink attenuated (downlink at 0 dB so the UE keeps its sync), the gNB's T-tracer with the failed-CRC inference of `tools/measure/oai.py` (a PUSCH with a power-control record but no decoded PDU in the same slot is a failed CRC). `harq.csv`:

| Attenuation | T-tracer PUSCH SNR | UL MCS (median) | First-tx BLER | Needed a 3rd tx | Residual loss | HARQ RTT | Frame p50 / p95 |
|---:|---:|---:|---:|---:|---:|---:|:---|
| 0 dB | 51 dB | 28 | 0 | 0 | 0 | | 19.5 / 27.1 ms |
| 10 | 41 | 28 | 0 | 0 | 0 | | 20.4 / 26.8 |
| 15 | 36 | 28 | 0 | 0 | 0 | | 21.0 / 27.5 |
| 20 | 31 | 2 | 6.8% | 0 | 0 | 15 slots | 22.0 / 36.7 |
| 23 | 28 | 5 | 8.0% | 0 | 0 | 10 | 21.2 / 34.7 |
| 26 | 25 | 5 | 9.1% | 0 | 0 | 15 | 22.2 / 35.7 |
| 29 | 22 | 4 | 8.3% | 0 | 0 | 10 | 21.6 / 36.7 |
| 31 | 20 | 2 | 9.7% | 1.1% | 0 | 10 | 24.5 / 41.9 |
| 33 | 18 | 0 | 6.4% | 5.7% | 0 | 10 | 39.9 / 123.5 |
| 35 | 16 | 0 | 85% | 85% | (link lost) | 5 | 21% delivered |

With the MCS fixed at 9 (100-byte frames at 20 Hz): no failure at 15 dB, then **every** first transmission fails from 20 dB to 30 dB while the second succeeds (a 3rd transmission is needed in 0.1–66% of TBs), 60% of frames are lost at 31 dB and the link is gone at 32 dB. The retransmission always follows the failed PUSCH after 10 slots (1,270–2,270 inferred retransmissions per run).

Three conclusions. The HARQ round trip is 10 slots (5 ms), two TDD periods (15 slots when the retransmission misses the next grant opportunity). OAI's BLER controller holds the first-transmission BLER inside its 5–15% band by cutting the MCS by up to 26 steps, and loses nothing until the link fails. The step at 15–20 dB is not an AWGN waterfall: the T-tracer SNR still reads 31 dB at 20 dB of attenuation, where MCS 9 should never fail, and chase combining then always succeeds. The attenuation acts on the fixed-point sample path, and apparently the receiver's quantization, not the configured noise (−50 dB), sets the failure point. rfsim therefore exercises HARQ and link-adaptation mechanics, but its BLER-against-SNR curve should not be used to calibrate the engine's BLER tables.

### (d) UE-count sweep

Every UE sends one frame per 100 ms: 4000 bytes (light, 0.32 Mbit/s) or 30000 bytes (heavy, 2.4 Mbit/s), with random phases (`contention.csv`, per-UE medians over the UEs of a run):

| UEs | Light: p50 / p95 / p99 | Heavy: p50 / p95 / p99 | rfsim speed (light, heavy) |
|---:|:---|:---|:---|
| 1 | 25.1 / 32.6 / 60.9 ms | 39.5 / 188 / 369 ms | 1.9, 2.0 |
| 2 | 26.5 / 38.6 / 77.1 | 53.0 / 602 / 729 | 1.3, 1.5 |
| 3 | 24.5 / 32.2 / 177.6 | 50.7 / 1,075 / 1,178 | 1.1, 1.0 |
| 4 | 22.9 / 48.8 / 259.1 | 78.0 / 1,347 / 1,428 | 0.9, 0.8 |
| 6 | 24.3 / 64.3 / 299.4 | not run | 0.51 |
| 8 | 24.3 / 287.4 / 446.8 | not run | 0.05 |
| 10 | 25.3 / 326.9 / 482.7 | not run | 0.05 |

All frames were delivered in every run, and Jain's index of per-UE goodput was 1.00. Light load does not raise the median with N, but the p99 grows from 61 ms at one UE to 259 ms at four and 483 ms at ten, and the p95 jumps between six and eight UEs. With eight and ten UEs the gNB logs hundreds of PDCCH allocation failures per UE (`CCE fail` in `nrMAC_stats`): a 51-PRB carrier has room for few DCIs per slot, which probably delays their grants. Those two runs also ran at 0.05× real time on a host with a load average near 60, so their tails are the least certain numbers on this page. Heavy load saturates the uplink from two UEs on: 4.8 Mbit/s of offered load builds queues of 0.6 s, although a 51-PRB DDDSU uplink at MCS 28 carries about 14 Mbit/s. This suggests that the per-frame SR and BSR cycle, rather than the PHY rate, limits the capacity here; we did not trace the scheduler to confirm it.

## Against the NR engine

Every latency and UE-count run was replayed through the NR engine (level `L2`, reference backend, fading off) with its measured arrival times and sizes (`compare_engine.py`, `tools/measure/replay.py`, 4 replicas, first 10 s of each run, SNR 30 dB so that the engine, like OAI at 0 dB of attenuation, runs at a high MCS). Three presets:

- `oai_like()`: fitted to public OAI one-way delays, with a proactive grant every TDD period, a 2.25 ms processing offset, an SR period of 40 slots, an SR-to-PUSCH delay of 50 slots, a HARQ round trip of 20 slots, a 0.5% BLER target and the UL MCS capped at 15;
- `lena_match` (`lena_like()`; the Sionna PDSCH tables and 38.214 TBS stand in for the 5G-LENA tables, which are not generated on the lab box): SR period 5 slots, SR-to-PUSCH `gnb_proc_slots + k2` = 3 slots, K2 = 2, HARQ round trip 3 slots, no processing offset;
- `oai_rfsim`: fitted here (below).

### Which preset fields the measurements confirm or contradict

| Field | `oai_like` | `lena_match` | OAI rfsim (this page) | Verdict |
|:---|:---|:---|:---|:---|
| `proactive_grant` | `per_period` | `off` | stock: `off` in effect (a grant only after 100 ms without uplink, which the engine cannot express, and irrelevant at ≥ 5 Hz); `per_period` only with `ulsch_max_frame_inactivity = 0` | `oai_like` describes OAI with periodic grants, not the stock 2026.w39 configuration |
| `sr_period_slots` | 40 | 5 | 10 (5 ms), fixed by OAI for this TDD pattern | contradicts both |
| `sr_grant_delay_slots` (SR to first PUSCH) | 50 | 3 | about 23 slots (11.6 ms from the SR opportunity to the first data PUSCH); 28 in the engine fit, where the engine's first grant carries no data | contradicts both; the 40-slot (20 ms) stand-in of `lena_validation()` is the right kind of correction, 1.4–1.7× longer than OAI |
| `k2` | 2 | 2 | 6 (`min_rxtxtime = 6`, every grant) | contradicts both for rfsim; a hardware UE may allow 2 |
| `ul_harq_rtt_slots` | 20 | 3 | 10 (2,801 retransmissions) | contradicts both |
| `proc_offset_ms` | 2.25 | 0 | 0.4–1.1 ms from the last PUSCH to the sink; 0 in the fit | `oai_like`'s 2.25 ms is mostly grant timing absorbed into an offset |
| `bler_target` | 0.005 | 0.1 (OLLA off) | first-transmission BLER 5.5–9.7% under OAI's 5–15% controller | contradicts `oai_like` for rfsim |
| `ul_mcs_max` | 15 | none | 28 in use up to 15 dB of attenuation | contradicts the cap for rfsim (the public-data cap reflects radio hardware) |
| `max_harq_tx`, `n_harq` | 4, 16 | 4, 16 | 4 transmissions, 16 processes | confirms |
| TDD, bandwidth | DDDSU, 20 MHz | DDDSU, 50 PRB | DDDSU (10, 2, 2), 51 PRB | configured to match |
| First grant after an SR | 1 byte (then a BSR round trip) | same | 5 PRBs, 63–76 bytes at MCS 5–6, then a BSR-sized grant | the engine's small first grant has the right structure; OAI's already carries about 56 bytes |

### The fitted `oai_rfsim` preset

`python -m isaaclab_net.tools.measure.calibrate <campaign> --engine-fit` on the stock-MAC runs, with the configuration facts from the manifests (numerology, bandwidth, TDD, SR period, K2, MCS table, HARQ), the HARQ round trip from the traced retransmissions and a replay fit of `sr_grant_delay_slots` over 16–32 slots with proactive grants off:

| `sr_grant_delay_slots` | 16 | 20 | 24 | **28** | 32 |
|:---|---:|---:|---:|---:|---:|
| fitted offset d0 (ms) | 4.9 | 2.3 | 0.5 | **0** | 0 |
| fit W1 (ms), 11 runs | 9.7 | 9.1 | 8.6 | **8.4** | 9.1 |
| held-out W1 (ms), Poisson and robot runs | 5.4 | 4.9 | 4.6 | **5.2** | 6.3 |

The fit W1 is dominated by the two 30 kB runs (19 and 41 ms); on the small-frame runs it is 1.7–3.1 ms. The preset file is `oai_like()` with `sr_period_slots = 10`, `sr_grant_delay_slots = 28`, `k2 = 6`, `ul_harq_rtt_slots = 10`, `proactive_grant = "off"`, `proc_offset_ms = 0`, `bler_target = 0.1`, `ul_mcs_max = 28` and `fading = False`, with the provenance of every field. Two things were set by hand: the HARQ round trip comes from the HARQ runs with a working link (the automatic value, 5 slots, was dominated by the run at 35 dB where the link failed), and the link-level knobs are left out, because the fitted values (a link-adaptation offset of 24 dB, a BLER shift of −12 dB and a slope of 0.1 per dB, all at the edges of their grids) describe rfsim's fixed-point decoding, not a radio link.

```python
from isaaclab_net.tools.measure.preset import load_preset
cfg = load_preset("benchmarks/oai/presets/oai_rfsim.json")
```

### Replay results

Median over the runs of each group (`engine_compare.csv`):

| Runs | `oai_like` W1 / KS / p50 error | `lena_match` | `oai_rfsim` |
|:---|:---|:---|:---|
| stock, 1 UE, 100 B–4 kB (8 runs) | 13.3 ms / 0.89 / −12.1 ms | 13.2 / 0.86 / −11.4 | **2.5 / 0.32 / +1.2** |
| stock, held out (Poisson 100 B, robot mix) | 11.6 / 0.74 / −11.3 | 12.0 / 0.79 / −9.3 | **5.5 / 0.38 / +3.7** |
| SR only (4 runs) | 12.4 / 0.95 / −13.0 | 14.7 / 1.00 / −14.9 | **1.8 / 0.34 / −0.5** |
| grant every period (4 runs) | **8.5 / 0.71 / −8.3** (100 B: 0.8–1.0 ms) | 9.3 / 0.89 / −7.1 | 13.9 / 0.89 / +6.3 |
| stock, 30 kB (2 runs) | 30.6 / 0.50 / −0.8 | 47.8 / 1.00 / −20.8 | 31.4 / 0.57 / −4.1 |
| stock, 1–10 UEs, light (7 runs) | 30.5 / 0.99 / −15.8 | 30.6 / 1.00 / −16.4 | **18.7 / 0.48 / −2.5** |
| stock, 1–4 UEs, heavy (4 runs) | 237 / 0.55 / +268 | 213 / 0.99 / −26.8 | 191 / 0.55 / −3.6 |

The fitted preset gets the medians right wherever access dominates, including the multi-UE light runs (p50 within 2.5 ms), but not the tails: OAI's p95 grows from 42 ms at one UE to 466 ms at ten, while the engine's stays near 25 ms. The engine has no PDCCH capacity limit (OAI logs hundreds of CCE allocation failures per UE at eight and ten UEs) and grants an SR at a fixed delay whatever the load. At heavy load all presets miss OAI's saturation, which the PHY capacity does not explain; `oai_like`, with its MCS cap of 15, even saturates earlier than OAI (38.8% delivered at four UEs). With a grant every period `oai_like` is the right preset for small frames, as it was fitted to data from such a configuration, but it underestimates 4 kB and 30 kB frames there (by 16 ms and 126 ms at the median), whose periodic 480-byte grants are too small.

### Closed loop

`benchmarks/oai/closed_loop.py` drives two robots random-walking in a 60 m arena through `OaiNet` for 150 steps at 10 Hz of virtual time (4000-byte frames with probability 0.5 per step, attenuation 45 dB minus the legacy radio's SNR, capped at 30 dB), and the same traffic through the engine with the `oai_rfsim` preset. OAI delivered all 145 frames at a median of 53 ms (p95 91 ms) and a mean age of information of 2.11 steps; the engine delivered all 145 at 30 ms (p95 40 ms) and 2.10 steps. The difference comes mostly from the attenuation mapping, which drives OAI's link adaptation down to low MCS while the engine's own link model sees the same SNR as a good link; this run checks the closed loop, not the fidelity. The rfsim clock ran at 1.03× wall time, and the bridge ended its steps 1.7 ms (median) and 15 ms (p99) after their virtual deadlines.


## Caveats

- **Shared host.** The load average of the lab box ranged from 8 to 60 during the campaign, and rfsim's speed with it (0.35–2.0). Virtual-time delays are protocol delays and do not depend on that speed as long as the stacks keep their slot timing, but at very low speed (8 and 10 UEs) the traffic agents' own scheduling adds jitter in wall time, which the mapping stretches by 1/speed.
- **The clock mapping.** Ticks are sampled every 10 slots (5 ms of virtual time) with the gNB's wall timestamps, which carry the gNB's processing jitter. A frame's delay is exact to about one slot for the logs (mapped after the run) and to a few milliseconds online.
- **Where the time stamps are taken.** Send times are taken by the agent just before `sendto` into the UE's TUN interface, receive times by the sink's kernel. The UE's own TUN read and SDAP queueing are inside the delay, as they would be for an application on a real UE.
- **The SR period and K2 are OAI's, not free parameters.** OAI derives the SR period from the TDD pattern (10 slots here) and uses K2 ≥ `min_rxtxtime`. The rfsim UE needs `min_rxtxtime = 6`; a hardware UE could run with 2, which would shorten every grant-to-data step by 2 ms.
- **One cell, AWGN.** No fading, no interference, no mobility beyond attenuation steps.
- **Not an over-the-air measurement.** The numbers describe OAI's protocol behavior on a simulated channel, not a radio link.

## Reproduce

```bash
# lab box, conda env with torch; OAI_TEXTLOG and OAI_T_MESSAGES as above
python benchmarks/oai/campaign.py --oai oai-src --work campaign --phase all     # about 45 min at speed 1
python benchmarks/oai/campaign.py --oai oai-src --work campaign --phase dmany   # 6, 8 and 10 UEs
python benchmarks/oai/analyze.py --work campaign --out benchmarks/oai/results
python -m isaaclab_net.tools.measure.calibrate campaign/campaign_default --out calib --engine-fit --max-s 10 \
    --grid '{"proactive_grant": ["off"], "sr_grant_delay_slots": [16, 20, 24, 28, 32]}'
python benchmarks/oai/compare_engine.py --work campaign --preset benchmarks/oai/presets/oai_rfsim.json
python benchmarks/oai/closed_loop.py --steps 100 --robots 2 --preset benchmarks/oai/presets/oai_rfsim.json
```

If OAI cannot run on a machine (no Docker, or a kernel without SCTP), `FakeStack` exercises the whole bridge without it, and any Linux host with Docker, or a CloudLab / POWDER node, runs the deployment as described.
