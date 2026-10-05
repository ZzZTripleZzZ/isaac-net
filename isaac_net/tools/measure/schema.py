"""Unified measurement schema: the tables every parser in this package writes and calibrate.py reads.

Four tables, one row type each. Every column has a fixed dtype and a missing-value sentinel, so a table from
srsRAN and one from OAI can be concatenated without guessing:

- ``sched``: one row per scheduled or decoded transport block (TB), from MAC pcaps, PHY/scheduler logs or the
  OAI T-tracer. Time is the NR slot clock (``sfn``, ``slot``, ``slot_abs``) plus a wall clock ``t_s`` when the
  source has one.
- ``ue_period``: one row per UE per reporting period, from the srsRAN metrics JSON or the OAI ``nrMAC_stats``
  dumps. Counters are per period (the OAI parser differences its cumulative counters).
- ``owd``: one row per UDP probe packet (one-way delay, loss).
- ``frames``: one row per application frame (a frame may span several probe packets); ``delay_ms`` is from the
  first packet sent to the last packet received, which is what the engine's frame delay measures.

Runs are described by a ``manifest.json`` next to the tables (see docs/measurement-protocol.md, "Run manifest").

Missing values: -1 for integer columns, NaN for float columns, "" for strings. CSV is always written; Parquet is
written in addition when pyarrow is importable (it is not a dependency of the package).
"""
from __future__ import annotations

import csv
import math
import os

INT, FLOAT, STR = "int", "float", "str"
MISSING = {INT: -1, FLOAT: float("nan"), STR: ""}

# name -> [(column, dtype, unit / meaning)]
TABLES = {
    "sched": [
        ("run_id", STR, "run identifier from the manifest"),
        ("stack", STR, "srsran | oai"),
        ("source", STR, "parser that produced the row: srsran_pcap, srsran_phylog, srsran_schedlog, oai_pcap, "
                        "oai_ttrace"),
        ("event", STR, "sched (scheduler decision), rx (decoded or CRC-checked PUSCH/PDSCH), pc (OAI PUSCH power "
                       "control record)"),
        ("t_s", FLOAT, "wall clock, s since the Unix epoch (source's own clock; NaN if the source has none)"),
        ("sfn", INT, "system frame number 0..1023"),
        ("slot", INT, "slot index within the frame (0..10*2^mu-1)"),
        ("slot_abs", INT, "slot count unwrapped across SFN wraps, 0 at the first row of the run"),
        ("slot_exact", INT, "1 if the source gives the slot, 0 if only the subframe (srsRAN MAC pcap)"),
        ("rnti", INT, "C-RNTI"),
        ("ue", STR, "UE label from the manifest's rnti map, else the RNTI in hex"),
        ("dir", STR, "UL | DL"),
        ("harq_id", INT, "HARQ process"),
        ("newtx", INT, "1 new data, 0 retransmission, -1 unknown"),
        ("rv", INT, "redundancy version"),
        ("nrtx", INT, "retransmission count of this TB (0 = first transmission)"),
        ("mcs", INT, "MCS index (TS 38.214 table of the run)"),
        ("qm", INT, "modulation order"),
        ("n_prb", INT, "allocated PRBs"),
        ("n_sym", INT, "allocated OFDM symbols"),
        ("tbs_bytes", INT, "transport block size, bytes"),
        ("crc", INT, "1 CRC OK, 0 CRC KO, -1 unknown"),
        ("sinr_db", FLOAT, "per-TB SINR or SNR estimate of the gNB, dB"),
        ("data_bytes", INT, "bytes of logical-channel SDUs (LCID 1..32) in the MAC PDU"),
        ("bsr_idx", INT, "buffer-size index of the last BSR MAC CE in the PDU"),
        ("bsr_bytes", FLOAT, "upper bound of that index in bytes (short BSR only), NaN otherwise"),
    ],
    "ue_period": [
        ("run_id", STR, ""),
        ("stack", STR, ""),
        ("source", STR, "srsran_metrics | oai_macstats"),
        ("t_s", FLOAT, "end of the period, s since the Unix epoch"),
        ("period_s", FLOAT, "period length, s"),
        ("rnti", INT, ""),
        ("ue", STR, ""),
        ("dir", STR, "UL | DL"),
        ("n_ok", INT, "TBs with CRC OK / ACK in the period"),
        ("n_nok", INT, "TBs with CRC KO / NACK in the period"),
        ("tx_r0", INT, "transmissions at HARQ round 0 (first transmissions), OAI only"),
        ("tx_r1", INT, "transmissions at round 1"),
        ("tx_r2", INT, "transmissions at round 2"),
        ("tx_r3", INT, "transmissions at round 3"),
        ("n_fail", INT, "TBs lost after the last HARQ round (OAI ulsch_errors / dlsch_errors)"),
        ("mcs", FLOAT, "MCS (srsRAN: period mean; OAI: the BLER-controller MCS at the dump)"),
        ("n_prb", FLOAT, "PRBs of the last allocation (OAI UL only)"),
        ("snr_db", FLOAT, "PUSCH SNR (UL) or PUCCH SNR (DL rows of OAI), dB"),
        ("bler", FLOAT, "the stack's own BLER estimate (OAI EWMA), NaN if none"),
        ("brate_bps", FLOAT, "MAC bit rate, bit/s"),
        ("bsr_bytes", FLOAT, "last reported UL buffer (srsRAN bsr) or DL buffer (dl_bs), bytes"),
        ("cqi", FLOAT, "wideband CQI"),
        ("phr_db", FLOAT, "last power headroom, dB"),
        ("sr_to_pusch_avg_ms", FLOAT, "srsRAN: mean SR-to-PUSCH delay in the period"),
        ("sr_to_pusch_max_ms", FLOAT, "srsRAN: max SR-to-PUSCH delay"),
        ("crc_delay_avg_ms", FLOAT, "srsRAN: mean PUSCH-to-CRC-indication delay"),
        ("harq_delay_avg_ms", FLOAT, "srsRAN: mean PUSCH HARQ delay"),
    ],
    "owd": [
        ("run_id", STR, ""),
        ("ue", STR, "probe sender label"),
        ("flow", INT, "flow id (0 = single flow; robot profile: 1 state, 2 control, 3 camera)"),
        ("seq", INT, "packet sequence number within the flow"),
        ("frame_id", INT, "application frame of the packet"),
        ("frag", INT, "fragment index within the frame"),
        ("n_frag", INT, "fragments in the frame"),
        ("frame_bytes", INT, "application frame size, bytes"),
        ("pkt_bytes", INT, "UDP payload bytes of this packet"),
        ("t_tx_s", FLOAT, "send time, s since the Unix epoch (sender clock)"),
        ("t_rx_s", FLOAT, "receive time (receiver clock), NaN if lost"),
        ("owd_ms", FLOAT, "t_rx - t_tx - clock offset correction, ms"),
        ("lost", INT, "1 if never received"),
    ],
    "frames": [
        ("run_id", STR, ""),
        ("ue", STR, ""),
        ("flow", INT, ""),
        ("frame_id", INT, ""),
        ("frame_bytes", INT, ""),
        ("n_frag", INT, ""),
        ("n_rx", INT, "fragments received"),
        ("t_tx_first_s", FLOAT, "first fragment sent"),
        ("t_rx_last_s", FLOAT, "last fragment received, NaN if incomplete"),
        ("delay_ms", FLOAT, "t_rx_last - t_tx_first (offset-corrected), NaN if incomplete"),
        ("complete", INT, "1 if every fragment arrived"),
    ],
}


def columns(table):
    return [c for c, _, _ in TABLES[table]]


def dtypes(table):
    return {c: t for c, t, _ in TABLES[table]}


def new_row(table, **kw):
    """A row dict with every column of `table`, missing values filled in; unknown keys raise."""
    dt = dtypes(table)
    bad = set(kw) - set(dt)
    if bad:
        raise KeyError(f"unknown {table} columns: {sorted(bad)}")
    row = {c: MISSING[t] for c, t in dt.items()}
    row.update(kw)
    return row


def _coerce(v, t):
    if t == STR:
        return "" if v is None else str(v)
    if v is None or v == "" or (isinstance(v, float) and math.isnan(v)):
        return MISSING[t]
    if t == INT:
        return int(v)
    return float(v)


def normalize(rows, table):
    """Coerce every value of every row to its column dtype (missing -> sentinel)."""
    dt = dtypes(table)
    out = []
    for r in rows:
        bad = set(r) - set(dt)
        if bad:
            raise KeyError(f"unknown {table} columns: {sorted(bad)}")
        out.append({c: _coerce(r.get(c), t) for c, t in dt.items()})
    return out


def write_table(rows, table, path, parquet=None):
    """Write rows to `path` (.csv). parquet=None: also write .parquet if pyarrow is importable; True: require it."""
    rows = normalize(rows, table)
    cols = columns(table)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: ("" if isinstance(r[c], float) and math.isnan(r[c]) else r[c]) for c in cols})
    if parquet is not False:
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            if parquet:
                raise
            return path
        tmap = {INT: pa.int64(), FLOAT: pa.float64(), STR: pa.string()}
        dt = dtypes(table)
        arrs = {c: pa.array([r[c] for r in rows], type=tmap[dt[c]]) for c in cols}
        pq.write_table(pa.table(arrs), os.path.splitext(path)[0] + ".parquet")
    return path


def read_table(path, table):
    """Read a CSV (or .parquet) written by write_table back into typed row dicts."""
    if path.endswith(".parquet"):
        import pyarrow.parquet as pq
        return normalize(pq.read_table(path).to_pylist(), table)
    dt = dtypes(table)
    with open(path, newline="") as f:
        rd = csv.DictReader(f)
        missing = set(dt) - set(rd.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: not a {table} table, missing columns {sorted(missing)}")
        return [{c: _coerce(r[c], t) for c, t in dt.items()} for r in rd]


def column(rows, name):
    return [r[name] for r in rows]


# ---------------------------------------------------------------- slot clock
def slots_per_frame(mu):
    return 10 * 2 ** mu


def unwrap_slots(sfn_slot, mu, t_s=None, realtime=True):
    """Absolute slot counts for a sequence of (sfn, slot) in time order, unwrapping the 1024-frame SFN cycle
    (10.24 s of slot time). The first entry is slot 0.

    t_s must advance at the slot rate. On a real radio the wall clock does, and with t_s a gap longer than the SFN
    cycle adds the right number of whole cycles. Under a simulated radio (OAI rfsim, srsRAN ZMQ) the slot clock runs
    slower or faster than the wall clock, so wall times would add cycles that never happened: pass realtime=False
    there, or pass the simulator's virtual time as t_s (it advances at the slot rate). With realtime=False or
    without t_s, only a backward jump (by more than half a cycle) counts, as exactly one wrap; a gap of more than
    one whole cycle between consecutive rows cannot be seen then."""
    spf = slots_per_frame(mu)
    cyc = 1024 * spf
    if not realtime:
        t_s = None
    out, base, prev = [], 0, None
    for i, (sfn, slot) in enumerate(sfn_slot):
        raw = sfn * spf + slot
        if prev is not None:
            d = raw - prev[0]
            if t_s is not None and not math.isnan(t_s[i]) and not math.isnan(prev[1]):
                # whole cycles implied by the wall clock
                dt_slots = (t_s[i] - prev[1]) * 1000 * 2 ** mu
                k = round((dt_slots - d) / cyc)
                base += k * cyc
            elif d < -cyc // 2:
                base += cyc
        out.append(base + raw)
        prev = (raw, t_s[i] if t_s is not None else float("nan"))
    if out:
        o0 = out[0]
        out = [x - o0 for x in out]
    return out


def ue_label(rnti, rnti_map=None):
    if rnti_map:
        for k in (rnti, f"{rnti:04x}", f"0x{rnti:04x}", str(rnti)):
            if k in rnti_map:
                return str(rnti_map[k])
    return f"{rnti:04x}" if rnti is not None and rnti >= 0 else ""
