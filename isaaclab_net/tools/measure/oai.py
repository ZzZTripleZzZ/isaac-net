"""OpenAirInterface gNB outputs -> unified schema.

Format evidence (openairinterface5g develop, GitHub mirror, read 2026-09-29):

- ``nrMAC_stats.log``: openair2/LAYER2/NR_MAC_gNB/main.c ``nrmac_stats_thread`` rewrites this file in the gNB's
  working directory about once per second; ``dump_mac_stats`` prints, per UE,
  ``UE RNTI 1d47 CU-UE-ID 1 in-sync PH 52 dB PCMAX 24 dBm, average RSRP -44 (16 meas), average SINR 25.3 (16 meas)``,
  an optional ``UE 1d47: CSI [CQI 15 RI 1 PMI (0,0)]`` line,
  ``UE 1d47: dlsch_rounds 2541/3/0/0, dlsch_errors 0, pucch0_DTX 0 (SNR 30.0+0.0) RSSI -50.0, BLER 0.00000 MCS (1) 27
  (Qm 8) CCE fail 0``,
  ``UE 1d47: ulsch_rounds 7766/12/0/0, ulsch_errors 0, ulsch_DTX 0, BLER 0.00000 MCS (0) 9 (Qm 4 deltaMCS 0) NPRB 5
  SNR 30.0 (+0.0) RSSI -60.0 CCE fail 0`` and ``UE 1d47: LCID 1,4, goodput DL 0.10 UL 0.20 Mbps``.
  ``*_rounds a/b/c/d`` count transmissions at HARQ round 0, 1, 2, 3 since attach (cumulative), ``*_errors`` TBs
  lost after the last round. Older releases print "UE 1d47: MAC: TX ... RX ... bytes" instead of goodput;
  the regular expressions below accept both and ignore unknown fields. Because the file is overwritten, a
  campaign snapshots it with a timestamp marker (``### t=<epoch s>`` lines, see the protocol); the same blocks
  in the gNB stdout log (lines prefixed by ``[NR_MAC]``) also parse.
- T-tracer text: ``common/utils/T/tracer/textlog -d T_messages.txt -on GNB_MAC_UL ...`` prints one line per
  event, ``HH:MM:SS.nnnnnnnnn: EVENT_NAME arg value arg value ...`` (tracer/logger/textlog.c; with
  ``-raw-time`` a ``[epoch_s]`` follows the clock). Events used (T_messages.txt): GNB_MAC_UL (rnti, frame,
  slot, mcs, tbs: UL scheduler decision), GNB_MAC_PUSCH_POWER_CONTROL (rnti, frame, slot, snrx10, phr, tpc,
  tb_size, txpower_calc, rbSize, mcs, rssi), GNB_MAC_UL_PDU_WITH_DATA (gNB_ID, CC_id, rnti, frame, slot,
  harq_pid, data: decoded UL PDU), GNB_MAC_DL (rnti, frame, slot, mcs, tbs). The RNTI prints in decimal.
"""
from __future__ import annotations

import math
import re

from .schema import new_row, ue_label, unwrap_slots

_T_MARK = re.compile(r"^###\s*t=([\d.]+)")
_UE_HDR = re.compile(r"UE RNTI ([0-9a-fA-F]{4})\b(.*)$")
_ROUNDS = re.compile(r"UE ([0-9a-fA-F]{4}): (dl|ul)sch_rounds ([\d/]+),(.*)$")
_ERRORS = re.compile(r"(?:dl|ul)sch_errors (\d+)")
_CSI = re.compile(r"UE ([0-9a-fA-F]{4}):.*CQI (\d+)")
_GOODPUT = re.compile(r"UE ([0-9a-fA-F]{4}):.*goodput DL\s+([\d.]+) UL\s+([\d.]+) Mbps")
_FRAME = re.compile(r"Frame\.Slot (\d+)\.(\d+)")


def _f(pat, s):
    m = re.search(pat, s)
    return float(m.group(1)) if m else float("nan")


def parse_macstats(path, run_id="", rnti_map=None):
    """Snapshots of nrMAC_stats (``### t=`` markers) or a gNB stdout log -> ue_period rows, one per UE per
    direction per snapshot, with the cumulative round/error counters differenced against the previous snapshot
    of that UE (the first snapshot of a UE only seeds the difference; counters that go backwards, i.e. a UE
    re-attached under the same RNTI, restart the difference)."""
    snaps, t, cur = [], float("nan"), None
    with open(path, errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            m = _T_MARK.match(line)
            if m:
                t = float(m.group(1))
                cur = None
                continue
            body = line.split("] ", 1)[1] if line.startswith("[") and "] " in line else line
            body = body.strip()
            h = _UE_HDR.search(body)
            if h:
                rnti = int(h.group(1), 16)
                cur = {"t": t, "rnti": rnti, "ph": _f(r"PH (-?\d+) dB", h.group(2)),
                       "sinr": _f(r"average SINR (-?[\d.]+)", h.group(2))}
                snaps.append(cur)
                continue
            r = _ROUNDS.search(body)
            if r and cur is not None and int(r.group(1), 16) == cur["rnti"]:
                d = r.group(2).upper()
                rounds = [int(x) for x in r.group(3).split("/")]
                rest = r.group(4)
                e = _ERRORS.search(rest)
                cur[d] = {"rounds": rounds, "errors": int(e.group(1)) if e else 0, "bler": _f(r"BLER ([\d.]+)", rest),
                          "mcs": _f(r"MCS \(\d+\) (\d+)", rest), "nprb": _f(r"NPRB (\d+)", rest),
                          "snr": _f(r"(?<!\()SNR (-?[\d.]+)", rest)}
                continue
            c = _CSI.search(body)
            if c and cur is not None and "CQI" in body:
                cur["cqi"] = float(c.group(2))
            g = _GOODPUT.search(body)
            if g and cur is not None:
                cur["gp_dl"], cur["gp_ul"] = float(g.group(2)) * 1e6, float(g.group(3)) * 1e6
    rows, prev = [], {}
    for s in snaps:
        for d in ("UL", "DL"):
            if d not in s:
                continue
            x = s[d]
            key = (s["rnti"], d)
            p = prev.get(key)
            prev[key] = (s["t"], x)
            if p is None:
                continue
            pt, px = p
            dr = [a - b for a, b in zip(x["rounds"], px["rounds"])]
            de = x["errors"] - px["errors"]
            if any(v < 0 for v in dr) or de < 0:
                continue
            dr += [-1] * (4 - len(dr))
            n_first, n_fail = dr[0], de
            rows.append(new_row(
                "ue_period", run_id=run_id, stack="oai", source="oai_macstats", t_s=s["t"],
                period_s=s["t"] - pt if not (math.isnan(s["t"]) or math.isnan(pt)) else float("nan"),
                rnti=s["rnti"], ue=ue_label(s["rnti"], rnti_map), dir=d, tx_r0=dr[0], tx_r1=dr[1], tx_r2=dr[2],
                tx_r3=dr[3], n_fail=n_fail, n_ok=n_first - n_fail, n_nok=sum(v for v in dr[1:] if v > 0) + n_fail,
                mcs=x["mcs"], n_prb=x["nprb"], bler=x["bler"],
                snr_db=x["snr"] if d == "UL" else float("nan"),
                brate_bps=s.get("gp_ul" if d == "UL" else "gp_dl", float("nan")),
                cqi=s.get("cqi", float("nan")) if d == "DL" else float("nan"),
                phr_db=s["ph"] if d == "UL" else float("nan")))
    return rows


_TLINE = re.compile(r"^(\d\d):(\d\d):(\d\d)\.(\d+)(?:\s*\[(\d+)\])?:\s+(\w+)\s*(.*)$")


def _args(s):
    toks = re.sub(r"\{buffer size:(\d+)[^}]*\}", r"\1", s).split()
    return {toks[i]: toks[i + 1] for i in range(0, len(toks) - 1, 2)}


def parse_ttracer(path, mu, run_id="", rnti_map=None, day_epoch=None):
    """T-tracer textlog output -> sched rows.

    GNB_MAC_UL -> event "sched" (mcs, tbs); GNB_MAC_PUSCH_POWER_CONTROL -> event "pc" (mcs, tbs, PRBs, SNR);
    GNB_MAC_UL_PDU_WITH_DATA -> event "rx" (decoded PDU: harq_pid, size, crc = 1); GNB_MAC_DL -> DL "sched".
    t_s = day_epoch + time of day (textlog prints local time of day only; pass the local midnight of the run
    as day_epoch, or use -raw-time, whose epoch seconds are used when present)."""
    rows, stamp = [], []
    with open(path, errors="replace") as f:
        for line in f:
            m = _TLINE.match(line.strip())
            if not m:
                continue
            hh, mm, ss, frac, raw, ev, rest = m.groups()
            tod = int(hh) * 3600 + int(mm) * 60 + int(ss) + float("0." + frac)
            if raw is not None:
                t = int(raw) + float("0." + frac)
            elif day_epoch is not None:
                t = day_epoch + tod
            else:
                t = tod
            a = _args(rest)
            gi = lambda k: int(a[k]) if k in a else -1
            if "rnti" not in a or "frame" not in a or "slot" not in a:
                continue
            rnti = gi("rnti")
            base = dict(run_id=run_id, stack="oai", source="oai_ttrace", t_s=t, sfn=gi("frame"), slot=gi("slot"),
                        slot_exact=1, rnti=rnti, ue=ue_label(rnti, rnti_map))
            if ev == "GNB_MAC_UL":
                row = new_row("sched", **base, event="sched", dir="UL", mcs=gi("mcs"),
                              tbs_bytes=gi("tbs"))
            elif ev == "GNB_MAC_DL":
                row = new_row("sched", **base, event="sched", dir="DL", mcs=gi("mcs"), tbs_bytes=gi("tbs"))
            elif ev == "GNB_MAC_PUSCH_POWER_CONTROL":
                row = new_row("sched", **base, event="pc", dir="UL", mcs=gi("mcs"), tbs_bytes=gi("tb_size"),
                              n_prb=gi("rbSize"),
                              sinr_db=gi("snrx10") / 10 if "snrx10" in a else float("nan"))
            elif ev == "GNB_MAC_UL_PDU_WITH_DATA":
                row = new_row("sched", **base, event="rx", dir="UL", harq_id=gi("harq_pid"), crc=1,
                              tbs_bytes=gi("data"))
            else:
                continue
            rows.append(row)
            stamp.append((row["sfn"], row["slot"], t))
    sabs = unwrap_slots([s[:2] for s in stamp], mu, [s[2] for s in stamp] if stamp else None)
    for r, sa in zip(rows, sabs):
        r["slot_abs"] = sa
    return rows
