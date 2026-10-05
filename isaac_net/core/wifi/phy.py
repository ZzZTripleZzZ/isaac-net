"""802.11 PHY abstraction of the Wi-Fi level: MCS rates, SNR thresholds for rate adaptation, PPDU durations and
the channel time of one access.

Rates. Data rate = N_SD * bits per subcarrier * coding rate * N_SS / T_sym, with N_SD the data subcarriers and T_sym
the OFDM symbol including the guard interval:
    HE  (ax)  N_SD = 234 / 468 / 980 / 1960 at 20 / 40 / 80 / 160 MHz, T_sym = 12.8 us + GI (0.8, 1.6, 3.2 us)
    VHT (ac)  N_SD = 52 / 108 / 234 / 468,                           T_sym = 3.2 us + GI (0.8 or 0.4 us)
    non-HT (a) 48 data subcarriers, T_sym = 4 us, 6 ... 54 Mb/s
so HE 20 MHz, 1 SS, 0.8 us GI gives 8.6 ... 143.4 Mb/s for MCS 0 ... 11, and VHT 20 MHz gives 6.5 ... 78 Mb/s for
MCS 0 ... 8. The VHT-MCS / N_SS / bandwidth combinations that IEEE Std 802.11-2016 Sec. 21.5 (Tables 21-30 to
21-61) marks as not valid are left out: MCS 9 at 20 MHz for 1, 2, 4, 5, 7, 8 streams, MCS 6 at 80 MHz for 3 and 7
streams, MCS 9 at 80 MHz for 6 streams, and MCS 9 at 160 MHz for 3 streams (n_ss <= 4 here, so 20 MHz MCS 9 with
1, 2, 4 streams, 80 MHz MCS 6 with 3 and 160 MHz MCS 9 with 3 apply). HE (802.11ax) has no such exclusions.
mcs_numbers() gives the MCS number of every entry of the table.

SNR thresholds. The receiver minimum input sensitivity of the standard at 20 MHz (HT / VHT / HE MCS 0 ... 11:
-82, -79, -77, -74, -70, -66, -65, -64, -59, -57, -54, -52 dBm; non-HT 6 ... 54 Mb/s: -82, -81, -79, -77, -74,
-70, -66, -65 dBm) minus the noise these sensitivities assume (-174 dBm/Hz over 20 MHz + a 10 dB noise figure =
-91 dBm). Both scale by 3 dB per bandwidth doubling, so the SNR threshold does not depend on the bandwidth. The
sensitivities also carry the standard's implementation margin, so the thresholds are conservative: MCS 0 needs
9 dB and HE MCS 11 needs 39 dB. WifiConfig.ra_margin_db shifts them (negative = more aggressive).

PPDU duration = preamble + T_sym * ceil((16 + 8 L + 6) / N_DBPS) for a PSDU of L bytes (service and tail bits as
in the BCC PHYs; HE's LDPC and packet extension are left out). Preambles: non-HT 20 us; VHT 20 + 8 + 4 + 4 N_LTF + 4
(L-STF/LTF/SIG, VHT-SIG-A, VHT-STF, VHT-LTFs, VHT-SIG-B); HE SU 20 + 4 + 8 + 4 + 8 N_LTF (RL-SIG, HE-SIG-A, HE-STF,
2x HE-LTFs of 8 us), N_LTF = 1, 2, 4, 4 for 1-4 streams. Control frames (ACK 14 B, compressed Block Ack 32 B,
RTS 20 B, CTS 14 B) go at ctrl_rate_mbps in non-HT PPDUs.
"""
from __future__ import annotations

import math

import torch

MOD = [(1, 1 / 2), (2, 1 / 2), (2, 3 / 4), (4, 1 / 2), (4, 3 / 4), (6, 2 / 3), (6, 3 / 4), (6, 5 / 6), (8, 3 / 4),
       (8, 5 / 6), (10, 3 / 4), (10, 5 / 6)]
SENS_20MHZ = [-82, -79, -77, -74, -70, -66, -65, -64, -59, -57, -54, -52]
LEGACY_RATES = [6, 9, 12, 18, 24, 36, 48, 54]
LEGACY_SENS = [-82, -81, -79, -77, -74, -70, -66, -65]
SENS_NOISE_DBM = -174.0 + 10 * math.log10(20e6) + 10.0        # -91 dBm: noise assumed by the sensitivity tables
N_SD = {"ax": {20: 234, 40: 468, 80: 980, 160: 1960}, "ac": {20: 52, 40: 108, 80: 234, 160: 468}}
N_LTF = {1: 1, 2: 2, 3: 4, 4: 4}
ACK_BYTES, BA_BYTES, RTS_BYTES, CTS_BYTES = 14, 32, 20, 14
PPDU_MAX_US = 5484.0                                          # aPPDUMaxTime (HT / VHT / HE)


def symbol_us(std, gi_us):
    return {"ax": 12.8 + gi_us, "ac": 3.2 + gi_us, "a": 4.0}[std]


def preamble_us(std, n_ss=1):
    if std == "a":
        return 20.0
    if std == "ac":
        return 36.0 + 4.0 * N_LTF[n_ss]
    return 36.0 + 8.0 * N_LTF[n_ss]


# (bandwidth MHz, VHT-MCS, N_SS) marked "not valid" in IEEE Std 802.11-2016 Tables 21-30 to 21-61
VHT_INVALID = frozenset({(20, 9, n) for n in (1, 2, 4, 5, 7, 8)} | {(80, 6, 3), (80, 6, 7), (80, 9, 6), (160, 9, 3)})


def mcs_numbers(std, bandwidth_mhz=20, n_ss=1):
    """MCS number of every entry of mcs_table (the index into the non-HT rate list for "a")."""
    if std == "a":
        return list(range(len(LEGACY_RATES)))
    n_mcs = 12 if std == "ax" else 10
    return [m for m in range(n_mcs) if not (std == "ac" and (bandwidth_mhz, m, n_ss) in VHT_INVALID)]


def mcs_table(std, bandwidth_mhz=20, n_ss=1, gi_us=0.8):
    """(rates in Mb/s, SNR thresholds in dB) of the valid MCS of this PHY, lowest first (mcs_numbers gives their
    MCS numbers)."""
    if std == "a":
        return [float(r) for r in LEGACY_RATES], [s - SENS_NOISE_DBM for s in LEGACY_SENS]
    tsym = symbol_us(std, gi_us)
    nsd = N_SD[std][bandwidth_mhz]
    rates, thr = [], []
    for m in mcs_numbers(std, bandwidth_mhz, n_ss):
        bits, code = MOD[m]
        rates.append(nsd * bits * code * n_ss / tsym)
        thr.append(SENS_20MHZ[m] - SENS_NOISE_DBM)
    return rates, thr


def ppdu_us(nbytes, rate_mbps, std, gi_us=0.8, n_ss=1):
    """Duration of a PPDU carrying nbytes (tensor or float) at rate_mbps (same shape or scalar)."""
    tsym = symbol_us(std, gi_us)
    ndbps = rate_mbps * tsym
    bits = 22.0 + 8.0 * nbytes
    if torch.is_tensor(bits) or torch.is_tensor(ndbps):
        nsym = torch.ceil(bits / ndbps)
    else:
        nsym = math.ceil(bits / ndbps)
    return preamble_us(std, n_ss) + tsym * nsym


def ctrl_us(nbytes, ctrl_rate_mbps):
    """Non-HT control frame duration (20 us preamble, 4 us symbols)."""
    return 20.0 + 4.0 * math.ceil((22 + 8 * nbytes) / (ctrl_rate_mbps * 4.0))


class AccessTiming:
    """Channel time of one access of B application bytes at a PHY rate (all tensors broadcast).

    Ts = AIFS + [RTS + SIFS + CTS + SIFS] + PPDU(B) + SIFS + ACK   (success, AIFS = SIFS + AIFSN * slot)
    Tc = AIFS + PPDU(B)                                            (collision, basic access, collision_time="difs")
    Tc = AIFS + PPDU(B) + SIFS + ACK                               (collision_time="ack": the stations wait about one
                                                                    ACK, a stand-in for ACK timeout and EIFS)
    Tc = AIFS + RTS + SIFS + CTS                                   (collision, RTS/CTS)
    Tv = PPDU(B) (basic) or RTS (RTS/CTS): the part of the exchange a hidden station can corrupt.
    """

    def __init__(self, wc):
        self.wc = wc
        self.ovh = wc.msdu_overhead_bytes / wc.msdu_payload_bytes
        agg = wc.max_ampdu_bytes > 0
        self.t_ack = ctrl_us(BA_BYTES if agg else ACK_BYTES, wc.ctrl_rate_mbps)
        self.t_rts = ctrl_us(RTS_BYTES, wc.ctrl_rate_mbps)
        self.t_cts = ctrl_us(CTS_BYTES, wc.ctrl_rate_mbps)
        self.pre = preamble_us(wc.standard, wc.n_ss)

    def air_bytes(self, B):
        return B * (1.0 + self.ovh)

    def cap_bytes(self, rate_mbps, txop_us):
        """Application bytes one access can carry: the A-MPDU length limit, the Block Ack window, and the PPDU
        time limit (aPPDUMaxTime, or the TXOP limit when it is set)."""
        wc = self.wc
        if wc.max_ampdu_bytes == 0:
            return torch.full_like(rate_mbps, float(wc.msdu_payload_bytes))
        tmax = torch.where(txop_us > 0, torch.clamp(txop_us, max=PPDU_MAX_US), torch.full_like(txop_us, PPDU_MAX_US))
        by_time = (tmax - self.pre) * rate_mbps / 8.0 / (1.0 + self.ovh)
        lim = min(wc.max_ampdu_bytes / (1.0 + self.ovh), wc.ba_window * wc.msdu_payload_bytes)
        return torch.clamp(by_time, max=lim).clamp(min=float(wc.msdu_payload_bytes))

    def times(self, B, rate_mbps, aifsn):
        """(Ts, Tc, Tv) in us for B application bytes per access."""
        wc = self.wc
        aifs = wc.sifs_us + aifsn * wc.slot_us
        tdata = ppdu_us(self.air_bytes(B), rate_mbps, wc.standard, wc.gi_us, wc.n_ss)
        if wc.rts_cts:
            hs = self.t_rts + wc.sifs_us + self.t_cts + wc.sifs_us
            ts = aifs + hs + tdata + wc.sifs_us + self.t_ack
            tc = aifs + self.t_rts + wc.sifs_us + self.t_cts
            tv = tdata * 0.0 + self.t_rts
        else:
            ts = aifs + tdata + wc.sifs_us + self.t_ack
            tc = ts if wc.collision_time == "ack" else aifs + tdata
            tv = tdata
        return ts, tc, tv
