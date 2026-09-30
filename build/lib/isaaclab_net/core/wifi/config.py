"""WifiConfig: the settings of the Wi-Fi level (make_engine("WIFI", ...)), carried in NRConfig.wifi.

The application fields (control step, frame buffer, timeout, message sizes), the randomness fields (seed, rng) and
the channel model (channel, tr38901_scenario, shadowing, blockage, radio map) stay in NRConfig and mean the same as
for the 5G levels. WifiConfig adds the 802.11 PHY and MAC, the access points and the model knobs.

The default EDCA parameter set of IEEE 802.11 for a non-AP STA (aCWmin = 15, aCWmax = 1023 for the OFDM PHYs):

    AC     CWmin  CWmax  AIFSN  TXOP limit
    BK     15     1023   7      0
    BE     15     1023   3      0
    VI     7      15     2      3.008 ms
    VO     3      7      2      1.504 ms
    DCF    15     1023   2      0          (legacy DCF: DIFS = SIFS + 2 slots)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

# (CWmin, CWmax, AIFSN, TXOP limit in us); index = access category id
AC_NAMES = ("BK", "BE", "VI", "VO", "DCF")
EDCA = {"BK": (15, 1023, 7, 0.0), "BE": (15, 1023, 3, 0.0), "VI": (7, 15, 2, 3008.0), "VO": (3, 7, 2, 1504.0),
        "DCF": (15, 1023, 2, 0.0)}
STANDARDS = ("ax", "ac", "a")
BANDWIDTHS = {"ax": (20, 40, 80, 160), "ac": (20, 40, 80, 160), "a": (20,)}


@dataclass
class WifiConfig:
    """802.11 uplink of a robot fleet: PHY, EDCA, access points, and the mean-field model's knobs.

    PHY and rate adaptation
      standard          "ax" (HE SU PPDU), "ac" (VHT) or "a" (non-HT OFDM, 20 MHz only)
      bandwidth_mhz     20, 40, 80 or 160 (ax / ac)
      n_ss              spatial streams (1-4); rates scale with it, the SNR thresholds do not
      gi_us             guard interval: ax 0.8 / 1.6 / 3.2, ac 0.8 / 0.4; ignored for "a"
      carrier_ghz       5.x or 6.x GHz; used by the channel model and the robot-robot path loss
      ra_margin_db      rate adaptation picks the highest MCS whose SNR threshold + margin <= SNR
      ap_nf_db          AP receiver noise figure; noise = -174 dBm/Hz + 10 log10(bandwidth) + NF
      sta_tx_dbm        robot transmit power
    MAC
      access_category   EDCA category of every robot ("BK", "BE", "VI", "VO" or "DCF")
      class_ac          optional AC per message class (a tuple over NRConfig.msg_sizes); a robot contends with
                        the AC of its head-of-line message (one FIFO per robot, not one queue per AC)
      rts_cts           RTS/CTS before every data PPDU
      max_tx            transmission attempts per frame (first + retries); the frame is dropped after that
      max_ampdu_bytes   A-MPDU length limit on the air; 0 = no aggregation (one MSDU per access, normal ACK)
      ba_window         Block Ack window in MPDUs (caps the MPDUs of one A-MPDU)
      msdu_payload_bytes, msdu_overhead_bytes  a message is carried in MSDUs of this payload; each costs this many
                        extra bytes on the air (UDP/IP 28 + LLC/SNAP 8 + QoS MAC header 26 + FCS 4 + delimiter 4)
      slot_us, sifs_us  9 and 16 us (OFDM PHYs at 5 / 6 GHz)
      ctrl_rate_mbps    non-HT rate of ACK / Block Ack / RTS / CTS
      frame_error_rate  residual error probability of each attempt (fading, interference), on top of collisions
      collision_time    channel time of a basic-access collision: "difs" = AIFS + PPDU (Bianchi's model; ns-3's
                        wifi module agrees with it within 3 %, docs/wifi.md Table F), or "ack" = AIFS + PPDU + SIFS +
                        ACK (the stations wait about one ACK: a conservative stand-in for ACK timeout and EIFS)
    Access points and channels
      ap_positions_m    one (x, y) per AP, env-local metres; None = NRConfig's cell positions (gnb_xy())
      ap_height_m       AP antenna height; None = NRConfig.gnb_height_m (or the scenario default for tr38901)
      n_channels        non-overlapping channels the APs are spread over (round robin) when ap_channels is None
      ap_channels       explicit channel index per AP. APs on the same channel share one medium: their robots
                        contend together (one collision domain per channel)
      roam_hyst_db      a robot re-associates when another AP is stronger by this much (RSSI, large-scale)
      roam_ms           interruption after a re-association (no channel access)
      bg_stations       saturated non-robot stations per AP (other clients on the same BSS)
      bg_frame_bytes, bg_mcs, bg_ac  their MSDU size (no aggregation), MCS and access category
    Hidden nodes
      hidden_nodes      model carrier sensing between robots (pairwise [E,R,R]); off = every robot on a channel
                        senses every other one on it
      cca_dbm           preamble-detection threshold: robot j is sensed by robot i if its power at i >= this
      sta_pl_exp, sta_pl_1m_db  robot-robot path loss 10 n log10(d) + PL(1 m) (None = free space at 1 m); no
                        shadowing. step(..., sense=) takes a user sensing matrix instead (e.g. from ray tracing)
    Model
      substep_ms        the mean-field model is re-solved every sub-step (control step / substep_ms per step)
      fp_iters          fixed-point iterations per sub-step (warm-started from the previous sub-step)
      fp_damping        damping of the fixed-point update (1 = undamped)
      access_noise      "poisson": the successful channel accesses of a robot in a sub-step are Poisson with the
                        mean-field rate (random service order, as contention gives); "mean": the expected number
                        (fluid, deterministic, finish times interpolated)
    """
    # ---- PHY and rate adaptation ----
    standard: str = "ax"
    bandwidth_mhz: int = 20
    n_ss: int = 1
    gi_us: float = 0.8
    carrier_ghz: float = 5.2
    ra_margin_db: float = 0.0
    ap_nf_db: float = 7.0
    sta_tx_dbm: float = 20.0
    # ---- MAC ----
    access_category: str = "BE"
    class_ac: tuple | None = None
    rts_cts: bool = False
    max_tx: int = 7
    max_ampdu_bytes: int = 65535
    ba_window: int = 64
    msdu_payload_bytes: int = 1472
    msdu_overhead_bytes: int = 70
    slot_us: float = 9.0
    sifs_us: float = 16.0
    ctrl_rate_mbps: float = 24.0
    frame_error_rate: float = 0.0
    collision_time: str = "difs"
    # ---- access points and channels ----
    ap_positions_m: tuple | None = None
    ap_height_m: float | None = None
    n_channels: int = 1
    ap_channels: tuple | None = None
    roam_hyst_db: float = 6.0
    roam_ms: float = 0.0
    bg_stations: int = 0
    bg_frame_bytes: int = 1500
    bg_mcs: int = 4
    bg_ac: str = "BE"
    # ---- hidden nodes ----
    hidden_nodes: bool = False
    cca_dbm: float = -82.0
    sta_pl_exp: float = 3.0
    sta_pl_1m_db: float | None = None
    # ---- model ----
    substep_ms: float = 1.0
    fp_iters: int = 6
    fp_damping: float = 0.6
    access_noise: str = "poisson"

    def __post_init__(self):
        assert self.standard in STANDARDS, f"standard must be one of {STANDARDS}"
        assert self.bandwidth_mhz in BANDWIDTHS[self.standard], (
            f"{self.standard}: bandwidth_mhz in {BANDWIDTHS[self.standard]}")
        assert 1 <= self.n_ss <= 4 and (self.standard != "a" or self.n_ss == 1)
        if self.standard == "ax":
            assert self.gi_us in (0.8, 1.6, 3.2), "ax: gi_us 0.8, 1.6 or 3.2"
        elif self.standard == "ac":
            assert self.gi_us in (0.4, 0.8), "ac: gi_us 0.4 or 0.8"
        assert self.access_category in EDCA and self.bg_ac in EDCA
        if self.class_ac is not None:
            self.class_ac = tuple(self.class_ac)
            assert all(a in EDCA for a in self.class_ac), f"class_ac entries in {tuple(EDCA)}"
        assert self.max_tx >= 1 and self.max_ampdu_bytes >= 0 and self.ba_window >= 1
        assert self.msdu_payload_bytes > 0 and self.msdu_overhead_bytes >= 0
        assert self.slot_us > 0 and self.sifs_us > 0 and self.ctrl_rate_mbps > 0
        assert 0.0 <= self.frame_error_rate < 1.0 and self.collision_time in ("ack", "difs")
        assert self.n_channels >= 1 and self.bg_stations >= 0 and self.bg_frame_bytes > 0 and self.bg_mcs >= 0
        assert self.roam_ms >= 0 and self.substep_ms > 0 and self.fp_iters >= 1 and 0 < self.fp_damping <= 1
        assert self.access_noise in ("poisson", "mean")
        if self.ap_positions_m is not None:
            self.ap_positions_m = tuple(tuple(float(v) for v in p) for p in self.ap_positions_m)
            assert len(self.ap_positions_m) >= 1
        if self.ap_channels is not None:
            self.ap_channels = tuple(int(c) for c in self.ap_channels)
            if self.ap_positions_m is not None:
                assert len(self.ap_channels) == len(self.ap_positions_m), "one channel per AP"

    def with_(self, **kw):
        return replace(self, **kw)

    def ap_xy(self, nr_cfg=None):
        """AP positions: ap_positions_m, else the NRConfig cell positions (default one AP at the origin)."""
        if self.ap_positions_m is not None:
            return [tuple(p) for p in self.ap_positions_m]
        return nr_cfg.gnb_xy() if nr_cfg is not None else [(0.0, 0.0)]

    def channels(self, n_ap):
        """Channel index of every AP."""
        if self.ap_channels is not None:
            assert len(self.ap_channels) == n_ap, "one channel per AP"
            return list(self.ap_channels)
        return [a % self.n_channels for a in range(n_ap)]

    def substeps(self, control_step_ms):
        n = control_step_ms / self.substep_ms
        if abs(n - round(n)) > 1e-9 or round(n) < 1:
            raise ValueError(f"control_step_ms={control_step_ms} is not a whole number of substep_ms="
                             f"{self.substep_ms}")
        return int(round(n))

    def noise_dbm(self):
        """Thermal noise over the channel bandwidth at the AP."""
        return -174.0 + 10 * math.log10(self.bandwidth_mhz * 1e6) + self.ap_nf_db

    def sta_pl_1m(self):
        if self.sta_pl_1m_db is not None:
            return float(self.sta_pl_1m_db)
        return 20 * math.log10(4 * math.pi * self.carrier_ghz * 1e9 / 299_792_458.0)
