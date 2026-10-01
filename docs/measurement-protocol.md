# Real-network measurement protocol

This page is a runbook for one afternoon of measurements on a private 5G cell (srsRAN Project or OpenAirInterface, up to 4 UEs) and on POWDER (up to 2 UEs). The data it produces calibrates and validates the three layers of the NR engine (`L2`) that public data could not reach ([calibration-public-data.md](calibration-public-data.md), "What public data could not validate", items 1–4 and 7): the uplink latency structure (SR, grants, HARQ timing, processing offset), link adaptation and HARQ statistics against SINR, and multi-UE contention. The parsers, the probe tool and the calibration script are in `isaac_net/tools/measure/`, and the calibration writes an `NRConfig` preset file that `make_engine` loads directly.

The channel layers (items 5 and 6: indoor path loss, shadowing, fading against speed) need a spatial survey and are outside this afternoon. An optional add-on is sketched at the end.

## At a glance

| | Experiment | Runs | Time | Calibrates (`NRConfig` fields) |
|:---|:---|---:|---:|:---|
| 0 | Setup, clock sync, idle capture, capacity probe | 3 | 45 min | `tdd_pattern`, `special_split`, `mu`, `bandwidth_mhz`, `sr_period_slots`, `k2`, `mcs_table` (recorded, not fitted); `proactive_grant` (idle capture) |
| a | Single-UE uplink latency against frame size and send rate | 14 | 30 min | `sr_grant_delay_slots`, `proactive_grant`, `ul_harq_rtt_slots`, `proc_offset_ms`; checks `bler_target` through the delay tail |
| b | UE-count sweep 1–4 at fixed per-UE load | 8 | 25 min | capacity scale η (knob), checks `pf_metric`, `pf_window`, `ul_mcs_max`, queueing |
| c | MCS, BLER and HARQ at fixed SINR positions | 7 + 9 | 40 min | `bler_target`, `ul_mcs_max`, `max_harq_tx`, `harq_fail`; knobs `la_offset_db`, `bler_sinr_shift_db`, BLER slope |
| d | Robot-like traffic replay | 4–6 | 15 min | nothing new: held-out validation of the preset from a–c |
| | Second stack, reduced grid (a4 + c3 + d1) | 8 | 60 min | the same fields for the other stack |

The full grid on one stack takes about 2.5 hours including restarts, so an afternoon covers the full grid on the stack the lab gNB normally runs plus a reduced grid on the other. POWDER is a separate session with the same scripts (see [POWDER](#powder)).

## Equipment and topology

**Lab cell.** One gNB host with the radio (for example a USRP B210 or X310), the core (Open5GS or OAI CN5G) on the same host, and 1–4 UEs. Commercial modems (Quectel RM520N-GL or similar, on USB) are preferred over software UEs because a software UE shares the gNB host's CPU and has its own scheduling artifacts. The UEs and the gNB are cabled through a splitter/combiner with programmable step attenuators (conducted RF), or placed at fixed spots in the room when no attenuators are available.

**Probe hosts.** The probe sender runs on the host the UE modem is attached to, bound to the modem's PDU-session address. The probe receiver runs on the core host, on the data-network side of the UPF (the `ogstun` or `oaitun` interface address). The best arrangement puts every modem on the gNB/core host itself, so sender and receiver share one clock (see [Clock synchronization](#clock-synchronization)). This is also how the public Zenodo data was taken.

**Directory layout.** One directory per run, with a `manifest.json` (format below) and the raw files it lists. Everything stays on the measurement machine or the lab box; only the unified tables and the calibration outputs (a few MB) come back.

```
campaign-2026-10-02-srsran/
  a-0100B-010Hz/  manifest.json  tx_ue1.csv  rx_ue1.csv  gnb_mac.pcap  metrics.jsonl  gnb.log
  c-att30/        manifest.json  gnb.log  metrics.jsonl  gnb_mac.pcap
  ...
```

## Step 0: configure the gNB (both stacks)

Use one configuration for the whole afternoon per stack, and record it in every manifest. The values below match the engine's default frame so the fitted preset can be compared with `srsran_like()` and `oai_like()`: μ = 1 (30 kHz), 20 MHz (51 PRB), TDD DDDSU with a (10, 2, 2) special slot, 64QAM MCS table. If the lab normally runs another pattern, keep it and record it: the calibration takes the pattern from the manifest.

### srsRAN Project (YAML)

The option names below were read from `du_high_config_cli11_schema.cpp`, `configs/low_latency.yml` and `configs/debug.yml` of srsRAN_Project `main` (latest release `release_25_10`). Section nesting has moved between releases, so run `gnb --help` or check the generated config (`gnb -c base.yml --dump-config` is not guaranteed on every release) before the afternoon.

```yaml
cell_cfg:
  channel_bandwidth_MHz: 20
  common_scs: 30
  tdd_ul_dl_cfg: {dl_ul_tx_period: 5, nof_dl_slots: 3, nof_dl_symbols: 10, nof_ul_slots: 1, nof_ul_symbols: 2}
  pucch: {sr_period_ms: 20}                  # the value srsran_like() inferred; record it
  pusch: {min_k2: 2, mcs_table: qam64, olla_target_bler: 0.01}
  mac_cell_group:
    bsr_cfg: {periodic_bsr_timer: 10}
metrics:
  enable_json: true
  layers: {enable_sched: true}
  periodicity: {du_report_period: 1000}      # ms
remote_control: {enabled: true, bind_addr: 127.0.0.1, port: 8001}   # 25.x: JSON metrics go to WebSocket subscribers
pcap: {mac_enable: true, mac_type: udp, mac_filename: /tmp/gnb_mac.pcap}
log: {filename: /tmp/gnb.log, all_level: warning, phy_level: warning, hex_max_size: 0}
```

For experiment c only, set `log: phy_level: info` (one line per decoded PUSCH with CRC, SINR, TBS, PRBs and modulation) and, for the fixed-MCS sub-grid, `pusch: {min_ue_mcs: M, max_ue_mcs: M}`. The HARQ options (`max_nof_harq_retxs`, `nof_harqs`, `harq_retx_timeout`) exist in the schema; keep their defaults and record them.

**Capturing the metrics JSON.** In releases 25.x the scheduler report is pushed to clients of the remote-control WebSocket after a `metrics_subscribe` command. Any WebSocket client works, for example `websocat ws://127.0.0.1:8001`, then send `{"cmd": "metrics_subscribe"}` and redirect the output to `metrics.jsonl` (one JSON object per line). Releases before 24.10 sent JSON over UDP (`metrics: enable_json_metrics: true, addr, port`), captured with `socat -u UDP-RECV:55555 - > metrics.jsonl`. The parser accepts both layouts.

### OpenAirInterface (gNB `.conf`)

Parameter names were read from `openair2/GNB_APP/MACRLC_nr_paramdef.h` and `gnb_paramdef.h` (develop branch) and from the POWDER OpenTwin profile template.

```
gNBs = ({ ...
  min_rxtxtime = 2;                                  # K1/K2 floor; POWDER template uses 6
  servingCellConfigCommon = ({
    dl_subcarrierSpacing = 1; ul_subcarrierSpacing = 1;
    dl_carrierBandwidth = 51; ul_carrierBandwidth = 51;
    dl_UL_TransmissionPeriodicity = 5;               # 2.5 ms = 5 slots at mu 1
    nrofDownlinkSlots = 3; nrofDownlinkSymbols = 10;
    nrofUplinkSlots = 1;   nrofUplinkSymbols = 2;
  ... }); });
MACRLCs = ({ ...
  ulsch_max_frame_inactivity = 10;                   # frames without UL before a proactive grant; record it
  ul_bler_target_upper = 0.15; ul_bler_target_lower = 0.05;
  ul_max_mcs = 28; ul_harq_round_max = 4;
  pusch_TargetSNRx10 = 200;
}); 
```

OAI has no SR-period option in the configuration file (the SR resource is set inside the RRC code), so `sr_period_slots` stays a fitted quantity for OAI. Its proactive grants are controlled by `ulsch_max_frame_inactivity`, which is why `oai_like()` fits `proactive_grant = "per_period"`.

**Logs to enable.**

- `nrMAC_stats.log` is rewritten about once per second in the gNB's working directory. Snapshot it with a timestamp marker, which the parser needs: `while sleep 1; do echo "### t=$(date +%s.%N)"; cat nrMAC_stats.log; done > macstats.log`.
- The MAC pcap: run the gNB with `--opt.type pcap --opt.path /tmp/oai_mac.pcap` (UDP-framed `mac-nr` in IPv4 packets to port 9999; `--opt.type wireshark` sends the same packets to a socket that `tcpdump -i lo -w oai_mac.pcap udp port 9999` can capture). Unlike srsRAN, OAI records the slot, not only the subframe.
- The T-tracer for per-TB records: run the gNB with `--T_stdout 2` (T-tracer on, normal console output kept; with the default `T_nowait = 0` the gNB waits at start-up until a tracer connects, `--T_nowait` skips the wait) and, on the same host, `common/utils/T/tracer/textlog -d common/utils/T/T_messages.txt -raw-time -on GNB_MAC_UL -on GNB_MAC_PUSCH_POWER_CONTROL -on GNB_MAC_UL_PDU_WITH_DATA > ttrace.txt`. `-raw-time` adds epoch seconds, without it the parser needs the run's local midnight (`ttrace_day_epoch` in the manifest).

### Settings that must be the same for both stacks

- **Closed-loop uplink power control.** Attenuating a conducted path does not lower the uplink SINR while the UE still has power headroom, because the gNB's closed loop raises the UE's power to its target (OAI `pusch_TargetSNRx10`). For experiment c, either use attenuation large enough that the UE runs at maximum power (power headroom `phr_db` near 0 in the unified tables), or set the target far above the reachable SNR (OAI: `pusch_TargetSNRx10 = 300`) so the UE always transmits at maximum power. The calibration uses the gNB-reported SINR, not the attenuator setting, for every fit.
- **Keep UEs connected.** Idle-mode transitions add hundreds of milliseconds. Disable modem power saving, and keep a 1 Hz keep-alive ping on each UE between runs.
- **CPU.** Set the gNB host to the performance governor and pin the gNB as its documentation recommends. Late slots and underflows change latency; count them in the gNB log for every run and redo a run that has any.

## Clock synchronization

One-way delay needs the sender and receiver clocks to agree to well below the quantity measured (the stacks' medians are 4–16 ms, the slot is 0.5 ms). In order of preference:

1. **Same host.** Attach the modem to the core host and put its network interface in its own network namespace (`ip netns add ue1; ip link set wwan0 netns ue1`, then bring up the PDU session inside it), so the probe sender in `ip netns exec ue1` cannot short-circuit through the local routing table. Sender and receiver then read the same `CLOCK_REALTIME` and the offset is exactly 0 (`"method": "same_host"`). Whether a given modem driver's interface can move into a namespace, and whether its connection manager still works there, must be checked on the day.
2. **PTP on a wired LAN** between the UE hosts and the core host (`ptp4l` + `phc2sys`, NICs with hardware timestamping): offsets of a few microseconds. Record `ptp4l`'s rms offset.
3. **chrony against a LAN server** (the core host as server, `local stratum 10`, `makestep 0.1 3`): typically tens of microseconds on an idle LAN. Record `chronyc tracking` ("System time") before and after every run. NTP to an internet pool is not good enough.

**Offset check (always).** Before experiment a and after experiment d, run the probe over the wired LAN in both directions (10 s each, 100 B at 100 Hz). On a symmetric wired path, half the difference of the two median OWDs is the clock offset (receiver minus sender); write it into the manifest as `clock.offset_ms` and the half-range of the before/after values as `clock.offset_bound_ms`. The OWD summary of every run also counts negative delays, which should be zero.

## Probes

```bash
# receiver on the core host, data-network side of the UPF
python -m isaac_net.tools.measure.probe recv --port 5201 --duration 75 --out rx_ue1.csv
# sender on the UE side, bound to the PDU-session address
python -m isaac_net.tools.measure.probe send --dst 10.45.0.1 --bind 10.45.0.2 --profile cbr \
    --size 1000 --rate 50 --duration 60 --out tx_ue1.csv
```

Frames larger than `--mtu` (default 1400 bytes) are split into back-to-back datagrams, and the frame delay (first datagram sent to last received) is what the engine's frame delay measures. The receiver uses kernel receive timestamps when available. If a UE host cannot run Python (a phone, a robot controller), capture both ends with `tcpdump` instead and use `owd.from_pcaps`, which matches packets by the probe header.

## Experiment grid

Every run is 60 s of traffic plus 15 s of margin unless stated. Start the gNB-side captures (pcap, metrics, T-tracer, snapshots) before the receiver, the receiver before the sender, and stop them in reverse order. Write the manifest right after each run, including the RNTI of each UE read from the gNB log or metrics, because RNTIs change on every attach.

### 0. Idle capture and capacity probe (10 min)

- **Idle capture** (60 s, UEs attached, no traffic, MAC pcap on): uplink PDUs without data reveal proactive grants and their period. Tag the run `"experiment": "idle"`.
- **Capacity probe** (30 s per UE count 1 and 4): `iperf3 -u -b 100M` uplink from each UE. The cell's uplink goodput sets the heavy load of experiment b.

### a. Single-UE uplink latency (14 runs, 30 min)

One UE at the best SINR position (UE at maximum MCS). Frame sizes and rates:

| Frame bytes | Send rates (Hz) | Offered load |
|---:|:---|:---|
| 100 | 10, 50, 100 | ≤ 80 kbit/s |
| 1000 | 10, 50, 100 | ≤ 0.8 Mbit/s |
| 4000 | 10, 50 | ≤ 1.6 Mbit/s |
| 30000 | 5, 10 | ≤ 2.4 Mbit/s |
| 100, Poisson | 50 | held out (`"holdout": true`) |

Repeat the first run (100 B at 10 Hz) at the end to check for drift. Logs: MAC pcap, metrics JSON (srsRAN) or macstats snapshots (OAI). Keep the PHY log at warning level: per-PUSCH logging on the real-time path can itself delay slots.

**What it calibrates.** Small frames at low rate isolate the grant path. With SR-based grants, the delay median sits near half the SR period plus the SR-to-grant delay (srsRAN's 12–16 ms public median was explained by a 20 ms SR cycle), and with proactive grants it sits near half a TDD period plus K2 (OAI's 3–4 ms). Large frames add the TB count, hence the capacity per UL slot. srsRAN's metrics report `avg_sr_to_pusch_delay` directly, which gives `sr_grant_delay_slots` without a fit. The HARQ round trip `ul_harq_rtt_slots` comes from the slot gap between transmissions of one HARQ process (PHY log of experiment c). `proc_offset_ms` (the public-data d0) and the SR / proactive-grant knobs are fitted by replaying the measured arrivals through the engine and minimizing the Wasserstein-1 distance of the frame-delay distributions, with the Poisson run and experiment d held out.

**Data size.** Probe logs 0.1–1 MB per run. The srsRAN MAC pcap holds every UL and DL TB: at DDDSU about 400 UL and 1,600 DL slots per second, 10–50 MB per minute under load. Metrics JSON a few kB per second.

**Pitfalls.** Clock offset (above). The SFN wraps every 10.24 s, and the parsers unwrap it with the capture timestamps. srsRAN's MAC pcap stores the subframe, not the slot, and its timestamps are taken when the pcap writer thread writes the record, so use the pcap for counts and grant spacing, not for sub-millisecond timing. Keep other traffic off the UE (modem firmware updates, NTP, DNS) by pointing the modem's default route away or filtering by the probe port.

### b. UE-count sweep (8 runs, 25 min)

N = 1, 2, 3, 4 UEs, all at the same SINR position, each sending the same traffic, at two load levels:

- **light:** 4000-byte frames at 10 Hz per UE (0.32 Mbit/s), the engine's small message class;
- **heavy:** a per-UE CBR rate of 0.4 × the one-UE capacity from step 0, so the cell saturates at N ≥ 3.

Stagger the sender start times by a random 0–100 ms so frames do not arrive in lockstep, and record each UE's SINR (it should agree within 2 dB; if not, move a UE or trim its attenuator). Logs as in a.

**What it calibrates.** Per-UE goodput, Jain fairness and frame-delay quantiles against N anchor the PF sharing and queueing at small N (item 1 of the public-data gaps). In saturated runs, the ratio of cell goodput to the 38.214 TBS capacity at the measured MCS gives the capacity scale η, the lab counterpart of ColO-RAN's η ≈ 0.8. The engine replay of these runs with the fitted preset is a validation, not a fit.

**Pitfalls.** Attach order changes RNTIs. UEs at slightly different SINRs get different MCS and therefore unequal PF shares, which is not unfairness. Commercial modems may throttle when hot. With 4 UEs on one host, check that the host's USB bus does not saturate before the radio does.

### c. MCS, BLER and HARQ at fixed SINR positions (40 min)

**OLLA sweep (7 runs).** One UE, saturating uplink (`iperf3 -u` at 1.2 × capacity, so every UL slot carries a TB), at 7 attenuation steps from the best position to about 3 dB above the point where the UE detaches (for example 0, 10, 20, 30, 35, 40, 45 dB beyond the maximum-power point). 60 s each gives about 24,000 TBs per position. Logs: srsRAN PHY log at info level plus metrics JSON; OAI T-tracer, macstats snapshots and MAC pcap.

**Fixed-MCS sub-grid (9 runs, 30 s each).** MCS 5, 10 and 15 (srsRAN `min_ue_mcs = max_ue_mcs`, OAI `ul_min_mcs = ul_max_mcs`, gNB restart per MCS) at three attenuations around each MCS's waterfall. This gives BLER against SINR per MCS without link adaptation, the direct test of the engine's BLER tables and the logistic slope of 1.5 dB⁻¹ that public data could not check.

**What it calibrates.** First-transmission BLER at each position (the OLLA operating point, which becomes `bler_target`), the MCS the stack chooses against SINR (the link-adaptation offset against the engine's thresholds, `la_offset_db`), the distribution of transmissions per TB and the residual loss after the last round (`max_harq_tx`, `harq_fail`), the highest usable MCS (`ul_mcs_max`), and, from per-TB CRC with SINR, the SINR shift that makes the engine's BLER model match (`bler_sinr_shift_db`) and a logistic slope. The per-TB MCS is recovered from TBS, PRBs and symbols by exact TS 38.214 TBS inversion when the log line has no MCS.

**Data size.** PHY info logging is one line per PUSCH, about 400 lines (80 kB) per second at DDDSU. T-tracer text about 1,200 lines per second for the three events.

**Pitfalls.** Uplink power control (see Step 0). The gNB's SINR estimate is biased at low SINR and saturates at high SINR (check where the reported SINR stops rising with less attenuation, and exclude those positions from the BLER fit). Logging at info level must not cause late slots; check the gNB log. OAI's `ulsch_rounds` counters are cumulative since attach and the parser differences consecutive snapshots, so a re-attach during a run restarts the difference.

### d. Robot-traffic replay (4–6 runs, 15 min)

`probe send --profile robot`: flow 1 is a 4000-byte state frame at 10 Hz, flow 2 a 100-byte control packet at 50 Hz, flow 3 a 30000-byte camera frame at 5 Hz (the engine's two message classes plus control), each with a random phase. Runs: N = 1 and 2 UEs (4 on the lab cell) at the best position, and N = 1 at an edge position. 120 s each.

**What it calibrates.** Nothing new. These runs are held out from every fit and score the final preset (W1 and KS of frame delay per flow in `params_latency_*.json`). They are also the first over-5G trace of robot-like traffic (gap 7).

### Reduced grid for the second stack (60 min)

Experiment a at 100 B / 10 Hz, 100 B / 100 Hz, 4000 B / 10 Hz and 30000 B / 5 Hz; experiment c at 3 positions (best, middle, edge); experiment d with one UE. This fixes the stack's latency structure and operating BLER, which is what distinguishes `srsran_like()` from `oai_like()`.

## POWDER

The OpenTwin POWDER profile (`NICELabExp,opentwin-matrix`, source in the OpenTwin repository under `code/testbed/powder/`) already brings up an OAI gNB on `x310-1` with two Quectel RM520N-GL UEs on `nuc27` and `nuc22` through the JFW attenuator matrix (crossbar C). Two UEs is a hard limit of that crossbar. It is a conducted RF setup: describe results as controlled RF measurements on real radio hardware, not over-the-air.

- **Stack.** OAI, with the configuration keys above applied to the profile's `etc/oai/gnb.template.conf` (the profile unpacks `/local/repository` at every boot, so changes go into the profile and are republished). srsRAN on the same matrix needs a separate profile (the dmaas `srs-rf-matrix` profile is referenced in the OpenTwin runbook; its current contents were not checked).
- **Attenuators.** Downlink and uplink are separately addressable: for `gnb1–ue1` the downlink id is 49 and the uplink id 18, for `gnb1–ue2` 113 and 82 (`bin/atten <id> <dB>`, 0–95 dB on top of about 30 dB insertion loss). Experiment c can therefore sweep the uplink alone, which avoids changing the UE's downlink CQI at the same time.
- **Clock.** UEs and gNB are on different hosts. Run `bin/sync-clocks.sh` (chrony against one experiment host) before measuring, and do the wired offset check.
- **Grid.** Experiments a (full), b (N = 1, 2 only), c (uplink attenuator only, 7 positions) and d (N = 1, 2) fit in about 2.5 hours after the profile is up. Allocate the matrix early: it is first-come, first-served without an approved reservation.

## Run manifest

```json
{
  "run_id": "a-0100B-010Hz",
  "experiment": "a",
  "stack": "srsran",
  "site": "lab",
  "n_ue": 1,
  "holdout": false,
  "gnb": {"mu": 1, "bandwidth_mhz": 20, "tdd_pattern": "DDDSU", "special_split": [10, 2, 2],
          "sr_period_ms": 20, "min_k2": 2, "mcs_table": "qam64", "olla_target_bler": 0.01,
          "ul_mcs_max": 28, "max_harq_tx": 4, "metrics_period_ms": 1000},
  "ues": [{"ue": "ue1", "rnti": "4601"}],
  "traffic": {"profile": "cbr", "size": 100, "rate_hz": 10},
  "radio": {"atten_db": 0, "sinr_setpoint_db": null, "fading": false},
  "clock": {"method": "same_host", "offset_ms": 0.0, "offset_bound_ms": 0.0},
  "files": {"srsran_pcap": "gnb_mac.pcap", "srsran_metrics": "metrics.jsonl",
            "probes": {"ue1": {"tx_csv": "tx_ue1.csv", "rx_csv": "rx_ue1.csv", "src": "10.45.0.2"}}}
}
```

`experiment` is `a`, `b`, `c`, `d` or `idle`. Instead of `tdd_pattern`, the `gnb` block may give the stack's own TDD fields as `"tdd": {"period_slots": 5, "nof_dl_slots": 3, "nof_dl_symbols": 10, "nof_ul_slots": 1, "nof_ul_symbols": 2}` (OAI: `"periodicity_idx": 5` instead of `period_slots`). OAI runs give `ul_bler_target_upper` / `ul_bler_target_lower` instead of `olla_target_bler`, and `ulsch_max_frame_inactivity`. File keys: `srsran_metrics`, `srsran_pcap`, `srsran_log`, `oai_macstats`, `oai_pcap`, `oai_ttrace`, and `probes` (per UE either `tx_csv` + `rx_csv` + optional `src`, or `tx_pcap` + `rx_pcap` + optional `port`; a per-UE `offset_ms` overrides the run's).

## Unified schema

`python -m isaac_net.tools.measure.ingest CAMPAIGN` parses every run and writes four tables to `CAMPAIGN/unified/` as CSV (and Parquet when `pyarrow` is installed), plus `runs.json` with the manifests. Missing values are -1 (integers), empty (floats, read back as NaN) and "" (strings). The column list with units is in `isaac_net/tools/measure/schema.py`.

| Table | One row per | Sources | Key columns |
|:---|:---|:---|:---|
| `sched` | scheduled or decoded TB | srsRAN MAC pcap, PHY and scheduler log; OAI MAC pcap, T-tracer | `sfn`, `slot`, `slot_abs` (unwrapped), `slot_exact`, `rnti`, `ue`, `dir`, `event` (sched / rx / pc), `harq_id`, `newtx`, `rv`, `nrtx`, `mcs`, `qm`, `n_prb`, `n_sym`, `tbs_bytes`, `crc`, `sinr_db`, `data_bytes`, `bsr_idx`, `bsr_bytes` |
| `ue_period` | UE, direction and report period | srsRAN metrics JSON; OAI `nrMAC_stats` snapshots | `n_ok`, `n_nok`, `tx_r0`–`tx_r3` (OAI HARQ rounds), `n_fail`, `mcs`, `snr_db`, `bler`, `brate_bps`, `bsr_bytes`, `cqi`, `phr_db`, `sr_to_pusch_avg_ms`, `crc_delay_avg_ms`, `harq_delay_avg_ms` |
| `owd` | probe packet | probe logs or two-sided pcaps | `flow`, `seq`, `frame_id`, `frag`, `t_tx_s`, `t_rx_s`, `owd_ms`, `lost` |
| `frames` | application frame | derived from `owd` | `frame_bytes`, `n_frag`, `n_rx`, `delay_ms` (first sent to last received), `complete` |

Which source fills which column differs by stack: srsRAN's info-level PHY log has CRC, SINR, TBS, PRBs and modulation per PUSCH but no MCS index (recovered from the TBS), its MAC pcap has the HARQ id and the MAC PDU (data bytes, BSR) of decoded uplink TBs only, and its metrics JSON has per-period OK/NOK counts, mean MCS and SNR, and the SR-to-PUSCH delay. OAI's T-tracer has MCS, TBS, PRBs and SNR per scheduled PUSCH (`GNB_MAC_PUSCH_POWER_CONTROL`) and the decoded PDUs (`GNB_MAC_UL_PDU_WITH_DATA`), but no failed-CRC event among the events used here, so OAI's BLER comes from the `ulsch_rounds` counters.

## Calibration

```bash
python -m isaac_net.tools.measure.calibrate CAMPAIGN --out CAMPAIGN/calib --engine-fit
```

Per stack it writes `params_latency_<stack>.json`, `params_link_<stack>.json`, `params_contention_<stack>.json` (the same three families as the public-data calibration) and `preset_<stack>.json`. The preset starts from `srsran_like()` or `oai_like()` and overrides the fields the afternoon determined, and every override carries its provenance (configuration, direct measurement, or engine replay fit with its W1 and KS). Quantities without an `NRConfig` field (η, the link-adaptation offset, the BLER SINR shift and slope) go into `calibration_knobs`, as η ≈ 0.8 did for the public data.

```python
from isaac_net import make_engine
from isaac_net.tools.measure.preset import load_preset
cfg = load_preset("CAMPAIGN/calib/preset_srsran.json")          # an NRConfig
net = make_engine("L2", 256, 4, "cuda", cfg)
```

The engine replay steps the NR engine on the CPU, one TDD period per control step, with the measured arrival times and sizes. It costs about 6 ms per step on the lab box, so replaying 20 s of each of ten runs for a grid of four candidates takes about 35 minutes. `--max-s` and `--grid '{"proactive_grant": ["off", "per_period"], "sr_grant_delay_slots": [5, 10, 20]}'` bound it. Without `--engine-fit` the script still writes everything except the fitted `proc_offset_ms` and SR / proactive-grant choice.

## Optional add-on: channel survey

If time remains, a grid of 20–30 positions in a robot-like space with the UE static for 20 s each (gNB-reported SINR and RSRP, UE-reported RSRP) gives an indoor path-loss exponent, shadowing σ and decorrelation distance for `pathloss_exp`, `shadow_sigma_db` and `shadow_dcorr_m` (gap 5). A cart moved at 0.5, 1 and 2 m/s along one line with per-slot SINR from the PHY log gives the fading correlation against speed (gap 6). Neither is parsed by the calibration script yet.
