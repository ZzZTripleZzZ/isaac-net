"""srsRAN Project gNB outputs -> unified schema.

Sources and the format evidence behind each parser (srsRAN_Project main, read 2026-09-29; release_25_10 is the
latest tag):

- Metrics JSON (``metrics: enable_json: true`` plus ``metrics: layers: enable_sched: true``; in 25.x the JSON is
  pushed to WebSocket subscribers of the remote-control server, ``remote_control: enabled: true``). The scheduler
  report is ``{"timestamp": "YYYY-MM-DDTHH:MM:SS.mmm" (UTC), "cells": [{"cell_metrics": {...}, "ue_list":
  [{...}]}]}`` with per-UE keys rnti, cqi, dl_mcs, dl_brate, dl_nof_ok, dl_nof_nok, dl_bs, pusch_snr_db,
  ul_mcs, ul_brate, ul_nof_ok, ul_nof_nok, bsr, last_phr, avg_sr_to_pusch_delay, avg_crc_delay,
  avg_pusch_harq_delay (apps/helpers/metrics/json_generators/du_high/scheduler.cpp). Releases before 24.10 wrote
  ``{"timestamp": <epoch float>, "ue_list": [{"ue_container": {...}}]}``; both are accepted, one JSON object per
  line (the capture client in docs/measurement-protocol.md writes that).
- MAC pcap (``pcap: mac_enable: true, mac_type: udp|dlt``): see macnr.py. Only the subframe is recorded, and the
  pcap timestamp is taken when the pcap backend writes the record (asynchronous), not at the air interface.
- PHY log at info level (``log: phy_level: info``): one line per decoded PUSCH, e.g.
  ``2026-09-29T12:00:00.123456 [PHY     ] [I] [  512.7] PUSCH: rnti=0x4601 h_id=0 prb=[0, 51) symb=[0, 14)
  mod=QPSK rv=0 tbs=1289 crc=OK iter=1.0 sinr=22.5dB ...`` (lib/phy/upper/channel_processors/pusch/
  logging_pusch_processor_decorator.h + formatters.h; the srslog context field is [sfn.slot]). The exact field
  set differs between releases and log levels, so the parser reads key=value tokens and ignores unknown ones.
- Scheduler log at debug level (``log: mac_level: debug``): per-slot "Slot decisions" blocks with
  ``- UE PUSCH: ue=0 c-rnti=0x4601 h_id=0 rb=[0..51) symb=[0..14) tbs=1289 rv=0 nrtx=0 nof_layers=1 olla=-0.3``
  lines (lib/scheduler/logging/scheduler_result_logger.cpp), and at info level ``UL: ue=0 rnti=0x4601 h_id=0 ...
  newtx=true rv=0 tbs=1289`` entries.
"""
from __future__ import annotations

import datetime as _dt
import json
import math
import re

from . import macnr
from . import pcap as P
from .schema import new_row, ue_label, unwrap_slots

MOD_QM = {"BPSK": 1, "PI/2-BPSK": 1, "pi/2-BPSK": 1, "QPSK": 2, "QAM16": 4, "16QAM": 4, "QAM64": 6, "64QAM": 6,
          "QAM256": 8, "256QAM": 8}


def _ts(v):
    if v is None:
        return float("nan")
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).rstrip("Z")
    try:
        d = _dt.datetime.fromisoformat(s)
    except ValueError:
        return float("nan")
    if d.tzinfo is None:
        d = d.replace(tzinfo=_dt.timezone.utc)
    return d.timestamp()


def _num(d, k):
    v = d.get(k)
    if v is None:
        return float("nan")
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _iter_json_objects(text):
    """JSON objects from a file holding one object per line, or a JSON array, or concatenated objects."""
    text = text.strip()
    if not text:
        return
    if text[0] == "[":
        yield from json.loads(text)
        return
    dec, i = json.JSONDecoder(), 0
    while i < len(text):
        while i < len(text) and text[i] in " \r\n\t,":
            i += 1
        if i >= len(text):
            break
        obj, i = dec.raw_decode(text, i)
        yield obj


def _ue_lists(obj):
    """(timestamp, [ue dict]) pairs of one metrics object, new ("cells") or old ("ue_container") layout."""
    ts = _ts(obj.get("timestamp"))
    if "cells" in obj:
        for c in obj["cells"] or []:
            yield ts, list(c.get("ue_list") or [])
    elif "ue_list" in obj:
        yield ts, [u.get("ue_container", u) for u in obj["ue_list"] or []]
    elif "cell_list" in obj:                                   # some 23.x/24.x builds
        for c in obj["cell_list"] or []:
            cc = c.get("cell_container", c)
            yield ts, [u.get("ue_container", u) for u in cc.get("ue_list") or []]


def parse_metrics_json(path, run_id="", rnti_map=None, period_s=None):
    """srsRAN scheduler metrics JSON -> ue_period rows (one UL and one DL row per UE per report).

    period_s: the report period (``metrics: periodicity: du_report_period`` in ms / 1000); if None, it is
    taken as the median spacing of the report timestamps."""
    with open(path) as f:
        objs = list(_iter_json_objects(f.read()))
    reports = [(ts, ues) for o in objs if isinstance(o, dict) for ts, ues in _ue_lists(o)]
    if period_s is None:
        ts = sorted({t for t, _ in reports if not math.isnan(t)})
        gaps = sorted(b - a for a, b in zip(ts, ts[1:]) if b > a)
        period_s = gaps[len(gaps) // 2] if gaps else float("nan")
    rows = []
    for ts, ues in reports:
        for u in ues:
            rnti = int(u.get("rnti", -1))
            base = dict(run_id=run_id, stack="srsran", source="srsran_metrics", t_s=ts, period_s=period_s,
                        rnti=rnti, ue=ue_label(rnti, rnti_map), cqi=_num(u, "cqi"))
            rows.append(new_row("ue_period", **base, dir="UL", n_ok=int(u.get("ul_nof_ok", -1)),
                                n_nok=int(u.get("ul_nof_nok", -1)), mcs=_num(u, "ul_mcs"),
                                snr_db=_num(u, "pusch_snr_db"), brate_bps=_num(u, "ul_brate"),
                                bsr_bytes=_num(u, "bsr"), phr_db=_num(u, "last_phr"),
                                sr_to_pusch_avg_ms=_num(u, "avg_sr_to_pusch_delay"),
                                sr_to_pusch_max_ms=_num(u, "max_sr_to_pusch_delay"),
                                crc_delay_avg_ms=_num(u, "avg_crc_delay"),
                                harq_delay_avg_ms=_num(u, "avg_pusch_harq_delay")))
            rows.append(new_row("ue_period", **base, dir="DL", n_ok=int(u.get("dl_nof_ok", -1)),
                                n_nok=int(u.get("dl_nof_nok", -1)), mcs=_num(u, "dl_mcs"),
                                snr_db=_num(u, "pucch_snr_db"), brate_bps=_num(u, "dl_brate"),
                                bsr_bytes=_num(u, "dl_bs"),
                                harq_delay_avg_ms=_num(u, "avg_pucch_harq_delay")))
    return rows


def parse_mac_pcap(path, mu, run_id="", rnti_map=None, stack="srsran", c_rnti_only=True):
    """MAC-NR pcap (srsRAN DLT 252, or OAI / any UDP-framed capture) -> sched rows, event "rx" for UL (the gNB
    writes UL PDUs it decoded) and "sched" for DL. Rows keep the pcap order; slot_abs unwraps SFN using the
    pcap timestamps."""
    raw = []
    for t, lt, fr in P.iter_pcap(path):
        info, pdu = macnr.frame_to_macnr(lt, fr)
        if info is None or "sfn" not in info:
            continue
        if c_rnti_only and info.get("rnti_type") != macnr.C_RNTI:
            continue
        raw.append((t, info, pdu))
    spf = 2 ** mu
    sfn_slot = [(i["sfn"], i["slot"] if i.get("slot_exact") else i["subframe"] * spf) for _, i, _ in raw]
    sabs = unwrap_slots(sfn_slot, mu, [t for t, _, _ in raw])
    rows = []
    for (t, info, pdu), (sfn, slot), sa in zip(raw, sfn_slot, sabs):
        ul = info["direction"] == 0
        w = macnr.walk_ul_pdu(pdu) if ul else None
        rnti = info.get("rnti", -1)
        rows.append(new_row(
            "sched", run_id=run_id, stack=stack, source=f"{stack}_pcap", event="rx" if ul else "sched", t_s=t,
            sfn=sfn, slot=slot, slot_abs=sa, slot_exact=info.get("slot_exact", 0), rnti=rnti,
            ue=ue_label(rnti, rnti_map), dir="UL" if ul else "DL", harq_id=info.get("harq_id", -1),
            tbs_bytes=len(pdu), crc=1 if ul else -1,
            data_bytes=w["data_bytes"] if w else -1, bsr_idx=w["bsr_idx"] if w else -1,
            bsr_bytes=w["bsr_bytes"] if w else float("nan")))
    return rows


_PREFIX = re.compile(r"^(?P<ts>\d{4}-\d\d-\d\dT[\d:.]+)\s+\[(?P<ch>[^\]]*)\]\s+\[(?P<lvl>\w)\]\s+"
                     r"(?:\[\s*(?P<sfn>\d+)\.(?P<slot>\d+)\]\s+)?(?P<msg>.*)$")
_KV = re.compile(r"([A-Za-z_][\w\-\[\]]*)=(\[[^\]\)]*[\]\)]|\S+)")


def _range_len(v):
    m = re.match(r"\[\s*(\d+)\s*(?:,|\.\.)\s*(\d+)\s*[\)\]]", v)
    return (int(m.group(2)) - int(m.group(1)), int(m.group(1))) if m else (-1, -1)


def _db(v):
    m = re.match(r"([+-]?[\d.]+)", v)
    return float(m.group(1)) if m else float("nan")


def _int(v, base=10):
    try:
        return int(v, 0) if v.startswith("0x") else int(v, base)
    except ValueError:
        return -1


def parse_phy_log(path, mu, run_id="", rnti_map=None, mcs_table=1):
    """srsRAN gNB log -> sched rows: "PUSCH:" PHY lines (event rx, with CRC and SINR) and scheduler PUSCH
    decisions ("- UE PUSCH:" debug lines and "UL:" info entries, event sched). The MCS index is recovered from
    mod + tcr when the line has tcr (verbose), else left missing."""
    from isaaclab_net.core.phy import MCS_TABLES
    mcs_tab = MCS_TABLES[mcs_table]
    rows, stamp = [], []
    cur = None                                        # (ts, sfn, slot) of the current Slot decisions block
    with open(path, errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            m = _PREFIX.match(line)
            if m:
                ts = _ts(m.group("ts"))
                sfn = int(m.group("sfn")) if m.group("sfn") else -1
                slot = int(m.group("slot")) if m.group("slot") else -1
                msg = m.group("msg")
                cur = (ts, sfn, slot) if "Slot decisions" in msg else None
            elif cur is not None and line.lstrip().startswith("- UE PUSCH:"):
                ts, sfn, slot = cur
                msg = line.strip()
            else:
                continue
            if msg.startswith("PUSCH:"):
                kv = dict(_KV.findall(msg))
                rnti = _int(kv.get("rnti", kv.get("c-rnti", "-1")))
                nprb, _ = _range_len(kv.get("prb", ""))
                nsym, _ = _range_len(kv.get("symb", ""))
                qm = MOD_QM.get(kv.get("mod", ""), -1)
                mcs = -1
                if "tcr" in kv and qm > 0:
                    tcr = float(kv["tcr"]) * (1024 if float(kv["tcr"]) < 1 else 1)
                    cand = [(abs(r - tcr), i) for i, (q, r) in enumerate(mcs_tab) if q == qm]
                    mcs = min(cand)[1] if cand else -1
                sinr = float("nan")
                for k in ("sinr", "sinr_eq[sel]", "sinr_ch_est[sel]", "sinr_evm[sel]", "snr"):
                    if k in kv:
                        sinr = _db(kv[k])
                        break
                crc = {"OK": 1, "KO": 0}.get(kv.get("crc", ""), -1)
                nd = kv.get("new_data")
                rows.append(new_row(
                    "sched", run_id=run_id, stack="srsran", source="srsran_phylog", event="rx", t_s=ts, sfn=sfn,
                    slot=slot, slot_exact=1 if slot >= 0 else 0, rnti=rnti, ue=ue_label(rnti, rnti_map), dir="UL",
                    harq_id=_int(kv.get("h_id", "-1")), rv=_int(kv.get("rv", "-1")), mcs=mcs, qm=qm, n_prb=nprb,
                    n_sym=nsym, tbs_bytes=_int(kv.get("tbs", "-1")), crc=crc, sinr_db=sinr,
                    newtx={"true": 1, "false": 0}.get(nd, -1) if nd else -1))
                stamp.append((sfn, slot, ts))
            elif msg.startswith("- UE PUSCH:") or re.search(r"(^|[\s,:])UL: ue=", msg):
                parts = [msg] if msg.startswith("- UE PUSCH:") else re.split(r"(?=UL: ue=)", msg)[1:]
                for p in parts:
                    kv = dict(_KV.findall(p))
                    rnti = _int(kv.get("c-rnti", kv.get("rnti", "-1")))
                    nprb, _ = _range_len(kv.get("rb", ""))
                    nsym, _ = _range_len(kv.get("symb", ""))
                    nt = kv.get("newtx")
                    nrtx = _int(kv.get("nrtx", "-1"))
                    newtx = {"true": 1, "false": 0}.get(nt, -1) if nt else (1 if nrtx == 0 else 0 if nrtx > 0 else -1)
                    rows.append(new_row(
                        "sched", run_id=run_id, stack="srsran", source="srsran_schedlog", event="sched", t_s=ts,
                        sfn=sfn, slot=slot, slot_exact=1 if slot >= 0 else 0, rnti=rnti,
                        ue=ue_label(rnti, rnti_map), dir="UL", harq_id=_int(kv.get("h_id", "-1").rstrip(",")),
                        rv=_int(kv.get("rv", "-1").rstrip(",")), nrtx=nrtx, newtx=newtx, n_prb=nprb, n_sym=nsym,
                        tbs_bytes=_int(kv.get("tbs", "-1").rstrip(","))))
                    stamp.append((sfn, slot, ts))
    ok = [i for i, (s, sl, _) in enumerate(stamp) if s >= 0 and sl >= 0]
    sabs = unwrap_slots([stamp[i][:2] for i in ok], mu, [stamp[i][2] for i in ok])
    for i, sa in zip(ok, sabs):
        rows[i]["slot_abs"] = sa
    return rows
