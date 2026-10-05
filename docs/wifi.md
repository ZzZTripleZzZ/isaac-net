# Wi-Fi (802.11 DCF / EDCA)

Warehouse and factory AMR fleets often run on Wi-Fi rather than private 5G. Level `WIFI` models the 802.11 uplink of such a fleet with the same API as every other level, so a task switches between 5G and Wi-Fi with one argument:

```python
from isaac_net.core import NRConfig, Requests, make_engine
from isaac_net.core.wifi import WifiConfig

cfg = NRConfig(channel="tr38901_inf_sh",                      # any channel model of docs/channels.md
               wifi=WifiConfig(standard="ax", bandwidth_mhz=40, carrier_ghz=5.2,
                               ap_positions_m=((20, 20), (60, 20), (40, 60)), n_channels=2))
net = make_engine("WIFI", E, R, "cuda", cfg, backend="graph")
net.submit(None, Requests(send))          # one message per robot per step, as for any level
out = net.step(None, poses)               # poses [E,R,2|3]; or SNR [E,R] with one AP; or rx_dbm=[E,R,A]
```

`step` returns the dict of every level (`delivered`, `timed_out`, `delay`, `newest`, `queue_len`, `sinr_db`, ...) plus `serving_cell` (the AP), `wifi_mcs`, `wifi_rate_mbps`, `wifi_access_ms` (mean channel-access time), `wifi_p_fail` (failure probability of an attempt) and `wifi_busy` (channel busy fraction as the robot senses it). Messages, the frame buffer, the application timeout and partial resets behave exactly as at the other levels, because the level reuses their FIFO and bookkeeping (`levels/base.LevelNet`). The code is in `isaac_net/core/wifi/`.

The model is a slot-synchronous, fixed-shape approximation of 802.11 channel access. It is not a packet-level 802.11 simulator, and the section [What the model drops](#what-the-model-drops) lists what it leaves out. Its errors against an exact event-driven CSMA/CA simulator, Bianchi's model and ns-3 are in [Validation](#validation).

## The model

### Time base

A 9 µs backoff slot is far too fine for a 100 ms control step at thousands of envs. The level therefore cuts each control step into sub-steps of `substep_ms` (default 1 ms, 100 per step) and, in every sub-step and every env, solves a mean-field contention model for the robots that have data. The sub-step loop has a fixed trip count and every tensor has a fixed shape, so the `graph` backend captures the whole sub-step loop of a control step in one CUDA graph.

### Contention: a Bianchi-style fixed point per sub-step

Every contending station *i* (a robot with queued bytes, a usable MCS and no roaming interruption) attempts in a slot with probability τ<sub>i</sub>. With conditional failure probability *p*<sub>i</sub>, contention windows *W*<sub>k</sub> = min(2<sup>k</sup> *W*<sub>0</sub>, *W*<sub>max</sub>) (*W* = CW + 1) and at most *L* = `max_tx` attempts per frame, a renewal argument over one frame gives

τ(*p*) = Σ<sub>k<L</sub> *p*<sup>k</sup> / Σ<sub>k<L</sub> *p*<sup>k</sup> (*W*<sub>k</sub> + 1) / 2,

which is Bianchi's expression [Bianchi, IEEE JSAC 18(3), 2000] when *L* → ∞ (a test checks the identity). The stations are coupled through

*p*<sub>i</sub> = 1 − (1 − *p*<sup>sense</sup><sub>i</sub>)(1 − *p*<sup>hidden</sup><sub>i</sub>)(1 − FER),  *p*<sup>sense</sup><sub>i</sub> = 1 − Π<sub>j ∈ V(i), j ≠ i</sub> (1 − τ<sub>j</sub>),

where V(*i*) is the set of stations *i* senses and FER is `frame_error_rate`. The fixed point is solved per env by damped iteration (`fp_iters` = 6 iterations, damping 0.6), warm-started from the previous sub-step. From a cold start, 20 iterations reach 10<sup>−7</sup> relative error on τ for 1–100 stations. The expected duration of a generic slot seen by station *i* is

*T*<sub>slot,i</sub> = σ *P*<sub>idle</sub> + *P*<sub>idle</sub> Σ<sub>j</sub> τ<sub>j</sub> / (1 − τ<sub>j</sub>) *T*<sub>s,j</sub> + *P*<sub>coll</sub> *T̄*<sub>c</sub>

over *j* ∈ V(*i*), with *T*<sub>s</sub> and *T*<sub>c</sub> the channel time of a success and of a collision of each station's own access. Station *i* then completes μ<sub>i</sub> = τ<sub>i</sub> (1 − *p*<sub>i</sub>) / *T*<sub>slot,i</sub> successful accesses per microsecond. So the access rate of every robot depends on the number of contenders, their frame sizes and PHY rates, RTS/CTS, and each one's access category.

**EDCA.** Each robot contends with the access category (AC) of its head-of-line message: `access_category` for all, or `class_ac` per message class. The AC sets CWmin, CWmax, AIFSN and the TXOP limit (the 802.11 defaults for a non-AP station: BK 15/1023/7, BE 15/1023/3, VI 7/15/2 with 3.008 ms, VO 3/7/2 with 1.504 ms, and DCF 15/1023/2). A station whose AIFSN exceeds the smallest one it senses by *d* counts down only after *d* idle slots that follow a busy period. The model sorts slots into zones (idle slots since the last busy period). In zone *z* only stations with *d* ≤ *z* count down, the zone of a generic slot follows the stationary distribution of that chain, and *p*<sub>i</sub>, *T*<sub>slot</sub> and μ<sub>i</sub> are averaged over the zones. With one AIFS this reduces to the plain model above.

**Hidden nodes** (`hidden_nodes=True`). V(*i*) comes from a pairwise sensing matrix [E, R, R]: robot *j* is sensed by *i* if its power at *i* is at least `cca_dbm` (−82 dBm), from a robot-robot log-distance path loss (free space at 1 m, exponent `sta_pl_exp`), or from a matrix the caller passes as `step(..., sense=)`, for example from ray tracing. A robot *j* that *i* does not sense but that reaches *i*'s AP corrupts *i*'s frame if it starts during the vulnerable window: *p*<sup>hidden</sup><sub>i</sub> = 1 − exp(−Σ<sub>j</sub> (attempt rate of *j*) (*T*<sub>v,i</sub> + *T*<sub>v,j</sub>)), with *T*<sub>v</sub> the PPDU, or only the RTS with RTS/CTS.

**Several APs.** Each robot associates with the AP of largest RSSI after a reset and re-associates when another AP is stronger by `roam_hyst_db`, after which it has no access for `roam_ms`. APs are spread over `n_channels` non-overlapping channels (or `ap_channels`). APs on the same channel share the medium: all their robots form one collision domain, or, with hidden nodes, sense each other through the matrix. `bg_stations` adds saturated non-robot clients per AP (other traffic on the same BSS) that contend like the robots.

### PHY: rates, rate adaptation and channel times

The robot's SNR at its AP comes from the channel models of [channels.md](channels.md) (`RadioMC` over the AP positions, with the Wi-Fi carrier and `sta_tx_dbm`) and the thermal noise over the channel bandwidth plus `ap_nf_db`. TR 38.901 InF at 5–6 GHz, log-distance and radio maps all work. Rate adaptation is ideal and per step: the highest MCS whose SNR threshold (plus `ra_margin_db`) the SNR meets. With `log_distance`, the 1 m loss is the free-space value at the Wi-Fi carrier (46.8 dB at 5.2 GHz) unless `pl_const_db` is set away from its NR default. With `rng="engine"` (the default) the radio's random draws (shadowing fields, LOS state, O2I) are keyed by (seed, env id, episode), so an env's channel does not depend on E, on other envs' resets, or on the `ShardedEngine` shard that holds it. `rng="global"` draws them from the engine generator.

| | 802.11ax (HE SU) | 802.11ac (VHT) | 802.11a |
|:---|:---|:---|:---|
| MCS | 0–11, BPSK 1/2 ... 1024-QAM 5/6 | 0–9 (invalid combinations left out, e.g. no MCS 9 at 20 MHz for 1, 2, 4 streams) | 6–54 Mb/s |
| data subcarriers (20/40/80/160 MHz) | 234 / 468 / 980 / 1960 | 52 / 108 / 234 / 468 | 48 |
| symbol | 12.8 µs + GI (0.8, 1.6, 3.2) | 3.2 µs + GI (0.8, 0.4) | 4 µs |
| 20 MHz, 1 stream | 8.6 ... 143.4 Mb/s | 6.5 ... 78 Mb/s | 6 ... 54 Mb/s |

The SNR thresholds are the standard's minimum receiver sensitivities at 20 MHz (−82, −79, −77, −74, −70, −66, −65, −64, −59, −57, −54, −52 dBm for MCS 0–11) minus the noise these sensitivities assume (thermal noise over 20 MHz plus a 10 dB noise figure, −91 dBm). This gives 9 dB for MCS 0 and 39 dB for HE MCS 11, independent of the bandwidth. The sensitivities include the standard's implementation margin, so the thresholds are on the conservative side. VHT-MCS / stream / bandwidth combinations that IEEE 802.11-2016 §21.5 marks as not valid are left out, and `wifi_mcs` is the MCS number.

One channel access carries *B* = min(queued bytes, cap) application bytes. The cap is the A-MPDU length limit (`max_ampdu_bytes`, default 65 535 B on the air), the Block Ack window (64 MPDUs) and the PPDU time limit (5.484 ms, or the TXOP limit of VI / VO). Each 1472 B MSDU costs 70 B on the air (UDP/IP, LLC/SNAP, QoS MAC header, FCS, A-MPDU delimiter). The PPDU lasts preamble + symbols (HE SU preamble 36 µs + 8 µs per HE-LTF, VHT 36 µs + 4 µs per VHT-LTF, non-HT 20 µs). A success costs *T*<sub>s</sub> = AIFS + PPDU + SIFS + Block Ack (plus RTS + SIFS + CTS + SIFS with RTS/CTS). A collision costs AIFS + PPDU with basic access (`collision_time="difs"`, the default, as in Bianchi's model; `"ack"` adds SIFS + ACK as a conservative stand-in for ACK timeout and EIFS) and AIFS + RTS + SIFS + CTS with RTS/CTS. Control frames go at `ctrl_rate_mbps` (24 Mb/s). `max_ampdu_bytes=0` disables aggregation: one MSDU per access, normal ACK.

### From access rates to messages

In each sub-step, the number of successful accesses of a robot is Poisson with mean μ<sub>i</sub> × sub-step (`access_noise="poisson"`, the default). The Poisson count is drawn over a fixed support of λ<sub>max</sub> + 6 √λ<sub>max</sub> + 1, with λ<sub>max</sub> = sub-step / *T*<sub>s,min</sub>, so the cap does not bind in practice. Competing Poisson clocks with state-dependent rates serve the contenders in a uniformly random order, which is what contention does to robots that all submit at the start of a control step. With `access_noise="mean"` the count is deterministic (a renewal process at rate μ<sub>i</sub>), which gives smooth means but serves synchronized robots in lockstep. The accesses serve min(queued, accesses × *B*) bytes FIFO from the robot's queue, and a message finishes at the (expected) time of the access that carries its last byte. A message is lost with the probability that one of its accesses exhausted `max_tx` attempts. That probability is *p̄*<sup>max_tx</sup> per access, where *p̄* is the geometric mean of the robot's failure probability over the sub-steps since it became backlogged. The retries spread over that period, and the later retries meet the lower contention at the end of a burst. A lost message is never delivered and leaves at the application timeout, as at the other levels.

## What the model drops

- **Slot-level correlations.** The mean-field decoupling treats each station's attempts as independent given the others' rates. This holds well for DCF / BE / BK windows (errors of about 1–2 % in saturation throughput). It breaks down for VO windows (CW 3–7) under heavy saturated contention, where the real collision probability is lower than the model's (Table C).
- **Synchronized arrivals.** A robot whose backoff counter reached zero sends a newly arrived frame at once (immediate access). Several such robots that receive messages in the same instant collide on the first slot. The model has no memory of counters, so it misses this first-slot collision burst.
- **Channel access details.** The model has no capture effect, no EIFS asymmetry, no propagation delay, no TXOP bursts of several PPDUs (a TXOP is one long PPDU), no fragmentation, no beacons, management frames or power save, and no NAV beyond RTS/CTS in the hidden-node term.
- **PHY.** Rate adaptation is ideal and per step, from the large-scale SNR (no Minstrel dynamics, no fast fading). Errors beyond collisions are a constant `frame_error_rate` per attempt, not a PER curve. The PPDU formula uses the BCC service and tail bits for every standard (HE's LDPC and packet extension are left out).
- **802.11ax features.** There is no UL OFDMA / trigger-based access, no MU-MIMO, no BSS coloring or spatial reuse, and no target wake time. Robots contend with single-user EDCA.
- **Traffic direction and queues.** The level models the uplink only. Downlink traffic of the APs can be approximated with `bg_stations`. Each robot has one FIFO, not one queue per AC, so a BK message ahead of a VO message delays it.
- **Hidden nodes and several BSSs** use the standard vulnerable-window approximation and are not validated against a packet-level model here (the event-driven simulator is single-BSS without hidden nodes).

## Validity range

- **Stations.** Validated for 1 to 50 contending stations per channel (Tables B, D, F). Goodput in saturation is within 2.5 % of the event-driven simulator and within 3.1 % of ns-3.
- **Access categories.** DCF, BE and BK (CWmin 15) are accurate. VI (CW 7–15) lies in between: −5 % goodput at 10 saturated stations in a spot check that is not in the tables. For VO with more than two saturated stations the model underestimates goodput (−9 % at 5, −41 % at 10), and in VO/BE mixes it overstates the small BE share (Table C). Robots that send occasional VO messages are fine. A fleet that saturates the channel with VO traffic is not.
- **Load.** Below saturation, message delays are within 18 % in the mean and 31 % at the 95th percentile (Table D). In overload, goodput is within 10 % but delays can be off by 40 %, because the application timeout amplifies small capacity errors.
- **Sub-step.** Keep `substep_ms` at 1 ms for delay studies. 2 ms is fine for training (+8 % mean delay), and 5 ms and above bias delays by 13–25 % in the mean and 40–60 % at the tail (Table E). The control step must be a whole number of sub-steps.
- **Bursts.** The Poisson draw has a fixed support with a 6-sigma margin over the largest possible mean (`poisson_cap`), so the number of accesses per robot and sub-step is not capped in practice.
- **Radio.** The SNR is the large-scale gain of the channel model per control step. Robots below the MCS 0 threshold (9 dB by default) cannot send, and their messages time out.
- **Not validated here.** Hidden nodes, several co-channel BSSs, roaming and background stations follow standard approximations. The event-driven simulator has no hidden nodes and only one BSS, so these parts are checked for behaviour (more collisions with hidden nodes, fewer with RTS/CTS, more contention on shared channels) but not for accuracy.

## Validation

All numbers below come from `python -m isaac_net.core.wifi.validate` (lab box, 8 CPU workers). The event-driven simulator (`core/wifi/eventsim.py`) is the reference for Tables B to E. It keeps the exact slot-level semantics of 802.11 DCF / EDCA in one BSS: per-station AIFS and backoff counters that freeze while the medium is busy, collisions of stations that reach zero in the same slot, binary exponential backoff with the retry limit, post-backoff and immediate access, and the same channel times per access as the model. A frame lost to FER occupies the medium for *T*<sub>s</sub> in both models.

**Pending regeneration.** Tables B, D and E below predate the fixes of the Poisson access cap and of the FER channel time in the event simulator. `validate.py` now also has FER = 0.2 rows for Table B (one MSDU with basic access, 30 000 B A-MPDUs with RTS/CTS), and the three tables are to be regenerated from it.

### A. Bianchi's saturation throughput

Bianchi's model with his Table I parameters (FHSS, 1 Mbit/s, 8184-bit payload, no retry limit), solved exactly by bisection. The tensor solver reaches the same root, and the event-driven simulator (200 s of channel time per point) is within 1.1 % of the analytic curves. This checks both the solver and the simulator.

| access | W | m | n | S (Bianchi) | S (tensor solver) | S (event sim) | event vs Bianchi |
|:---|---:|---:|---:|---:|---:|---:|---:|
| basic | 32 | 3 | 5 | 0.8097 | 0.8097 | 0.8043 | -0.7 % |
| basic | 32 | 3 | 10 | 0.7532 | 0.7532 | 0.7492 | -0.5 % |
| basic | 32 | 3 | 20 | 0.6788 | 0.6788 | 0.6805 | +0.3 % |
| basic | 32 | 3 | 50 | 0.5529 | 0.5529 | 0.5588 | +1.1 % |
| rts | 32 | 3 | 5 | 0.8342 | 0.8342 | 0.8303 | -0.5 % |
| rts | 32 | 3 | 10 | 0.8371 | 0.8371 | 0.8326 | -0.5 % |
| rts | 32 | 3 | 20 | 0.8356 | 0.8356 | 0.8308 | -0.6 % |
| rts | 32 | 3 | 50 | 0.8270 | 0.8270 | 0.8218 | -0.6 % |
| basic | 128 | 3 | 5 | 0.8250 | 0.8250 | 0.8213 | -0.4 % |
| basic | 128 | 3 | 10 | 0.8263 | 0.8263 | 0.8221 | -0.5 % |
| basic | 128 | 3 | 20 | 0.7981 | 0.7981 | 0.7923 | -0.7 % |
| basic | 128 | 3 | 50 | 0.7252 | 0.7252 | 0.7209 | -0.6 % |

### B. 802.11ax saturation: mean-field vs event-driven

802.11ax, 20 MHz, MCS 7 (86 Mb/s), AC_BE, `max_tx` = 7, every station saturated. The rows vary the bytes per access (1472 = one MSDU without aggregation, 8000 and 30 000 = A-MPDUs) and the number of stations *n*. The table gives the aggregate goodput (Mb/s), the conditional failure probability of an attempt *p*, and the share of bytes dropped at the retry limit. Goodput is within 2.4 % everywhere and *p* within 0.02. Table B includes FER = 0.2 rows once regenerated (see the note above).

| bytes / access | n | goodput model | goodput event | error | p model | p event | retry drops model | retry drops event |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1472 | 1 | 33.83 | 33.87 | -0.1 % | 0.000 | 0.000 | 0.0000 | 0.0000 |
| 1472 | 2 | 35.37 | 34.75 | +1.8 % | 0.105 | 0.112 | 0.0000 | 0.0000 |
| 1472 | 5 | 34.15 | 33.60 | +1.6 % | 0.272 | 0.262 | 0.0001 | 0.0002 |
| 1472 | 10 | 32.13 | 31.67 | +1.5 % | 0.389 | 0.375 | 0.0014 | 0.0017 |
| 1472 | 20 | 29.68 | 29.53 | +0.5 % | 0.496 | 0.475 | 0.0074 | 0.0080 |
| 1472 | 50 | 25.53 | 25.65 | -0.5 % | 0.634 | 0.614 | 0.0413 | 0.0394 |
| 8000 | 1 | 64.56 | 64.58 | -0.0 % | 0.000 | 0.000 | 0.0000 | 0.0000 |
| 8000 | 2 | 63.15 | 62.71 | +0.7 % | 0.105 | 0.109 | 0.0000 | 0.0000 |
| 8000 | 5 | 58.03 | 58.16 | -0.2 % | 0.272 | 0.260 | 0.0001 | 0.0001 |
| 8000 | 10 | 53.40 | 53.54 | -0.3 % | 0.389 | 0.378 | 0.0014 | 0.0017 |
| 8000 | 20 | 48.46 | 48.90 | -0.9 % | 0.496 | 0.480 | 0.0074 | 0.0088 |
| 8000 | 50 | 40.78 | 41.80 | -2.4 % | 0.634 | 0.616 | 0.0413 | 0.0354 |
| 30000 | 1 | 76.76 | 76.75 | +0.0 % | 0.000 | 0.000 | 0.0000 | 0.0000 |
| 30000 | 2 | 73.32 | 73.27 | +0.1 % | 0.105 | 0.104 | 0.0000 | 0.0000 |
| 30000 | 5 | 66.27 | 66.89 | -0.9 % | 0.272 | 0.256 | 0.0001 | 0.0000 |
| 30000 | 10 | 60.54 | 60.77 | -0.4 % | 0.389 | 0.380 | 0.0014 | 0.0020 |
| 30000 | 20 | 54.63 | 55.39 | -1.4 % | 0.496 | 0.483 | 0.0074 | 0.0069 |
| 30000 | 50 | 45.65 | 46.63 | -2.1 % | 0.634 | 0.627 | 0.0413 | 0.0329 |

### C. EDCA: AC_VO and AC_BE together

Same PHY, one MSDU per access, saturated. The goodput per AC is in Mb/s. BE-only rows are as accurate as Table B. VO-only rows show where the mean-field decoupling fails: with VO's tiny windows (CW 3–7) and several saturated VO stations, the real collision probability is much lower than the model's, and the model underestimates VO goodput by 9 % (5 stations) and 41 % (10 stations). In the mixes, the model keeps the priority order and the total within 10 %, but it gives the starved BE stations two to four times their small true share (below 3 Mb/s in every row).

| VO stations | BE stations | VO model | VO event | VO error | BE model | BE event | total error |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 5 | 0.00 | 0.00 | n/a | 34.15 | 33.49 | +2.0 % |
| 0 | 10 | 0.00 | 0.00 | n/a | 32.13 | 31.78 | +1.1 % |
| 5 | 0 | 23.39 | 25.67 | -8.9 % | 0.00 | 0.00 | -8.9 % |
| 10 | 0 | 10.59 | 18.08 | -41.4 % | 0.00 | 0.00 | -41.4 % |
| 2 | 5 | 32.13 | 32.98 | -2.6 % | 1.94 | 0.89 | +0.6 % |
| 2 | 10 | 30.33 | 31.94 | -5.1 % | 2.95 | 1.44 | -0.3 % |
| 5 | 5 | 23.06 | 25.50 | -9.6 % | 0.17 | 0.05 | -9.1 % |
| 5 | 20 | 22.30 | 25.31 | -11.9 % | 0.50 | 0.13 | -10.4 % |

### D. Level WIFI vs event-driven: message delays

The full level (1 ms sub-steps, Poisson accesses, FIFO, 2 s timeout) against the event-driven simulator with the same traffic. Every robot submits one message per 100 ms control step, all at the start of the step as in an RL env, at 30 dB SNR (MCS 7), for 60 steps. The engine runs 64 envs, the simulator 8 independent runs, and messages captured in steps 2 to 39 are compared. Delays are in ms.

| robots | message (B) | offered (Mb/s) | mean level | mean event | error | median level | median event | p95 level | p95 event | p95 error | delivered level | delivered event |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 4000 | 1.6 | 2.13 | 2.33 | -8.6 % | 1.50 | 2.21 | 5.25 | 4.00 | +31.2 % | 1.000 | 1.000 |
| 10 | 4000 | 3.2 | 3.98 | 3.98 | -0.1 % | 3.50 | 3.81 | 8.50 | 7.02 | +21.0 % | 0.999 | 1.000 |
| 20 | 4000 | 6.4 | 7.89 | 8.12 | -2.8 % | 7.50 | 8.38 | 16.33 | 14.61 | +11.8 % | 0.996 | 1.000 |
| 40 | 4000 | 12.8 | 16.61 | 19.08 | -13.0 % | 16.50 | 20.07 | 32.50 | 32.31 | +0.6 % | 0.984 | 1.000 |
| 5 | 30000 | 12.0 | 10.72 | 13.07 | -18.0 % | 9.50 | 12.33 | 25.50 | 21.71 | +17.5 % | 1.000 | 1.000 |
| 10 | 30000 | 24.0 | 20.79 | 22.56 | -7.8 % | 19.50 | 21.51 | 45.50 | 39.86 | +14.1 % | 0.999 | 1.000 |
| 20 | 30000 | 48.0 | 47.28 | 46.52 | +1.6 % | 43.50 | 48.86 | 93.50 | 82.73 | +13.0 % | 0.996 | 1.000 |
| 40 | 30000 | 96.0 | 1240.44 | 893.85 | +38.8 % | 1344.50 | 836.18 | 1968.50 | 1907.75 | +3.2 % | 0.628 | 0.693 |

Below saturation, the mean delay is within 18 % (−18 % to +2 %) and the 95th percentile within 31 % (always above the simulator's: the Poisson access count spreads delays more than real backoff does). The level underestimates the mean for a few robots with synchronized arrivals because it misses the first-slot collision of robots that all find their counters at zero (see [What the model drops](#what-the-model-drops)). It loses up to 1.6 % of messages at the retry limit while the simulator loses none. When all robots submit at once, the simulator's retries meet a quickly shrinking set of contenders, and the level's geometric-mean failure probability still overstates that. In overload (40 robots × 30 kB, 1.5 times the capacity), goodput is 9 % lower than the simulator's and the mean delay 39 % higher. There the queues run into the 2 s timeout, which amplifies the small capacity difference.

### E. Sub-step length and access noise

Two Table D scenarios. With Poisson accesses, 0.5 ms and 1 ms sub-steps agree (mean within 5 %), 2 ms adds about 8 % to the mean delay, and 5 ms adds 13–25 % to the mean and 40–60 % to the 95th percentile. `access_noise="mean"` serves synchronized robots in lockstep, so all their messages finish together and the mean delay is 73–89 % too high whatever the sub-step. Use it only for deterministic, smooth access rates (for example in unit tests), not for delays. The last column is the engine's wall time for the whole scenario (64 envs, 60 steps, CPU reference backend, shared machine).

| sub-step (ms) | access noise | robots | message (B) | mean level | mean event | error | p95 level | p95 event | p95 error | engine time (s) |
|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.5 | poisson | 10 | 4000 | 3.81 | 3.98 | -4.4 % | 8.25 | 7.02 | +17.5 % | 43.9 |
| 0.5 | poisson | 20 | 30000 | 47.43 | 46.52 | +1.9 % | 94.75 | 82.73 | +14.5 % | 41.1 |
| 0.5 | mean | 10 | 4000 | 6.88 | 3.98 | +72.6 % | 6.88 | 7.02 | -2.1 % | 20.2 |
| 0.5 | mean | 20 | 30000 | 87.86 | 46.52 | +88.9 % | 87.86 | 82.73 | +6.2 % | 24.7 |
| 1.0 | poisson | 10 | 4000 | 3.98 | 3.98 | -0.1 % | 8.50 | 7.02 | +21.0 % | 9.8 |
| 1.0 | poisson | 20 | 30000 | 47.28 | 46.52 | +1.6 % | 93.50 | 82.73 | +13.0 % | 13.2 |
| 1.0 | mean | 10 | 4000 | 6.88 | 3.98 | +72.6 % | 6.88 | 7.02 | -2.1 % | 9.7 |
| 1.0 | mean | 20 | 30000 | 87.86 | 46.52 | +88.9 % | 87.86 | 82.73 | +6.2 % | 14.5 |
| 2.0 | poisson | 10 | 4000 | 4.31 | 3.98 | +8.3 % | 9.00 | 7.02 | +28.2 % | 4.9 |
| 2.0 | poisson | 20 | 30000 | 50.18 | 46.52 | +7.9 % | 99.00 | 82.73 | +19.7 % | 7.9 |
| 2.0 | mean | 10 | 4000 | 6.88 | 3.98 | +72.6 % | 6.88 | 7.02 | -2.1 % | 4.7 |
| 2.0 | mean | 20 | 30000 | 87.86 | 46.52 | +88.9 % | 87.86 | 82.73 | +6.2 % | 7.3 |
| 5.0 | poisson | 10 | 4000 | 5.00 | 3.98 | +25.5 % | 11.25 | 7.02 | +60.2 % | 2.1 |
| 5.0 | poisson | 20 | 30000 | 52.61 | 46.52 | +13.1 % | 117.50 | 82.73 | +42.0 % | 3.4 |
| 5.0 | mean | 10 | 4000 | 6.88 | 3.98 | +72.6 % | 6.88 | 7.02 | -2.1 % | 2.0 |
| 5.0 | mean | 20 | 30000 | 87.86 | 46.52 | +88.9 % | 87.86 | 82.73 | +6.2 % | 3.6 |

### F. ns-3 reference points

ns-3.48's own `src/wifi/examples/wifi-bianchi.cc`, built unmodified from a copy of the lab's ns-3.48 tree: 802.11ax HeMcs7, 20 MHz, 800 ns GI, 1500 B packets, no A-MPDU, saturated, the example's retry limit of 65 535, 10 s per point and one trial. At *n* = 50 the example's infrastructure run stopped with "Not all stations got traffic!" (association), so that point uses its ad-hoc ring; at *n* = 10 the two modes differ by 0.4 %. The model uses the same payload and MAC overheads (1500 B + 38 B), `max_tx` = 1000 and the default `collision_time="difs"`. The ns-3 example also prints its own Bianchi reference, and its runs at 5–20 stations match it within 1.1 % with the DIFS collision model. Runs with A-MPDU (`maxMpdus=16`) reported 74–88 Mb/s, more than the 86 Mb/s PHY rate at 50 stations. We therefore do not trust the example's byte count for aggregated goodput and leave those runs out. Goodput is in Mb/s.

| n | ns-3 mode | ns-3 | model | error | event sim | error | model, `collision_time="ack"` | error |
|---:|:---|---:|---:|---:|---:|---:|---:|---:|
| 5 | infra | 34.86 | 34.80 | -0.2 % | 34.06 | -2.3 % | 34.03 | -2.4 % |
| 10 | infra | 33.18 | 32.84 | -1.0 % | 32.37 | -2.4 % | 31.73 | -4.4 % |
| 20 | infra | 31.27 | 30.63 | -2.0 % | 30.36 | -2.9 % | 29.27 | -6.4 % |
| 35 | infra | 29.36 | 28.69 | -2.3 % | 28.63 | -2.5 % | 27.16 | -7.5 % |
| 50 | adhoc | 28.23 | 27.34 | -3.1 % | 27.31 | -3.3 % | 25.72 | -8.9 % |

The model is within 3.1 % of ns-3 from 5 to 50 stations. The conservative `collision_time="ack"`, which charges every collision one extra ACK exchange, drifts to −9 % at 50 stations, which is why `"difs"` is the default.

### Performance

The level is launch-bound: every sub-step runs one to two hundred small tensor ops (the fixed point with six iterations, the channel times, the Poisson draw and the FIFO), and a control step has 100 sub-steps at the default 1 ms. The table gives control steps per second (env-steps per second in parentheses) of the `graph` backend on the lab RTX 4090, all robots sending every step, measured while other jobs kept the GPU at 97–99 % utilization, so the numbers are lower bounds. The `reference` backend ran at 0.3–1.8 steps per second on the same runs.

| sub-step | configuration | E × R = 256 × 16 | 1024 × 16 | 4096 × 16 | 1024 × 32 |
|:---|:---|---:|---:|---:|---:|
| 1 ms | one AP | 6.7 (1.7 k) | 3.1 (3.1 k) | 1.1 (4.7 k) | 2.2 (2.3 k) |
| 2 ms | one AP | 13.0 (3.3 k) | 9.1 (9.3 k) | 3.3 (13.6 k) | 5.6 (5.8 k) |
| 2 ms | two APs, hidden nodes | 7.4 (1.9 k) | 6.5 (6.7 k) | 2.8 (11.5 k) | 5.2 (5.3 k) |
| 5 ms | one AP | 32.3 (8.3 k) | 14.0 (14.3 k) | 5.9 (24.1 k) | 10.0 (10.2 k) |

For training at scale, `substep_ms=2.0` roughly triples the speed for about 8 % more mean delay (Table E). A fused Triton kernel for the sub-step loop, like the prototype's `triton` backend, is the natural next step.

## Tests

`tests/test_wifi.py` checks the PHY tables and access timing against hand values, the solver against the exact scalar fixed point and the τ(*p*) identity, Bianchi's curves (regression values), the event-driven simulator against Bianchi's model and the mean-field model against the simulator in saturation and with EDCA, the model against the recorded ns-3 points (Table F), and the level WIFI against the simulator on periodic messages (the bounds of Table D). It also checks the engine API: dict outputs, clocks, conservation (accepted = delivered + timed out + queued) and exact timeouts under overload with retry-limit losses, and partial resets (index and mask) that leave other envs bitwise unaffected with SNR input, poses, and hidden nodes with several APs and background stations. Further checks cover determinism, delay and failure probability monotone in the number of robots, message size, background load and SNR, out-of-range robots, association, co-channel sharing and roaming, hidden nodes with and without RTS/CTS and a user sensing matrix, and EDCA classes in the engine. On a GPU (`gpu` marker), the `graph` backend is bitwise equal to the reference through a partial reset.

`python -m isaac_net.core.wifi.validate` regenerates Tables A to F (about 10 minutes on 8 CPU cores; `--quick` for a short run). Table F uses the recorded ns-3 values in `validate.NS3_WIFI_BIANCHI`, so it runs without ns-3.
