"""Parsers of isaac_net.tools.measure on small synthetic fixtures (tests/fixtures/measure/ and pcaps built here
byte for byte in the layouts srsRAN and OAI write). No real gNB logs are needed."""
import math
import os
import struct

import pytest

from isaac_net.tools.measure import ingest, macnr, oai, owd, probe, srsran
from isaac_net.tools.measure import pcap as P
from isaac_net.tools.measure.schema import TABLES, columns, read_table, unwrap_slots, write_table

FX = os.path.join(os.path.dirname(__file__), "fixtures", "measure")
fx = lambda name: os.path.join(FX, name)


# ------------------------------------------------------------------ schema
def test_schema_roundtrip(tmp_path):
    rows = srsran.parse_phy_log(fx("srsran_gnb.log"), mu=1, run_id="r")
    p = write_table(rows, "sched", str(tmp_path / "sched.csv"), parquet=False)
    back = read_table(p, "sched")
    assert len(back) == len(rows)
    for a, b in zip(rows, back):
        for c in columns("sched"):
            assert (a[c] == b[c]) or (isinstance(a[c], float) and math.isnan(a[c]) and math.isnan(b[c])), c
    with pytest.raises(KeyError):
        write_table([{"nope": 1}], "sched", str(tmp_path / "x.csv"), parquet=False)
    assert set(TABLES) == {"sched", "ue_period", "owd", "frames"}


def test_unwrap_slots_sfn_wrap():
    mu, spf = 1, 20
    seq = [(1022, 0), (1023, 19), (0, 0), (0, 5)]
    assert unwrap_slots(seq, mu) == [0, 39, 40, 45]
    # a 10.24 s gap is one whole SFN cycle; only the wall clock can tell
    t = [0.0, 10.24 + 0.0025]
    assert unwrap_slots([(5, 0), (5, 5)], mu, t) == [0, 1024 * spf + 5]
    assert unwrap_slots([(5, 0), (5, 5)], mu) == [0, 5]


def test_tdd_pattern_helpers():
    assert ingest.tdd_pattern(5, 3, 10, 1, 2) == ("DDDSU", (10, 2, 2))
    assert ingest.tdd_pattern(10, 6, 8, 3, 0) == ("DDDDDDSUUU", (8, 6, 0))
    # OAI: dl_UL_TransmissionPeriodicity 6 (5 ms) at mu 1, 7 DL slots + 6 DL symbols, 2 UL slots + 4 UL symbols
    per = int(ingest.TDD_PERIOD_MS[6] * 2)
    assert ingest.tdd_pattern(per, 7, 6, 2, 4) == ("DDDDDDDSUU", (6, 4, 4))
    with pytest.raises(ValueError):
        ingest.tdd_pattern(10, 3, 4, 3, 2)


# ------------------------------------------------------------------ srsRAN
def test_srsran_metrics_json_new_and_old():
    rows = srsran.parse_metrics_json(fx("srsran_metrics.jsonl"), run_id="r", rnti_map={17921: "ue1"})
    ul = [r for r in rows if r["dir"] == "UL"]
    assert len(rows) == 6 and len(ul) == 3                       # 2 UEs, then 1, then an empty cell
    a = ul[0]
    assert a["ue"] == "ue1" and a["rnti"] == 17921 and a["n_ok"] == 400 and a["n_nok"] == 4
    assert a["mcs"] == 20 and a["snr_db"] == 22.5 and a["bsr_bytes"] == 1200 and a["phr_db"] == 30
    assert a["sr_to_pusch_avg_ms"] == 10.2 and a["crc_delay_avg_ms"] == 1.5 and a["harq_delay_avg_ms"] == 9.0
    assert ul[1]["ue"] == "4602" and ul[1]["sr_to_pusch_avg_ms"] == 9.8
    assert a["period_s"] == pytest.approx(1.0)
    assert a["t_s"] == pytest.approx(1790683201.0)              # 2026-09-29T12:00:01Z
    dl = [r for r in rows if r["dir"] == "DL"]
    assert dl[0]["n_ok"] == 100 and dl[0]["bsr_bytes"] == 0 and dl[0]["cqi"] == 15
    old = srsran.parse_metrics_json(fx("srsran_metrics_old.jsonl"), period_s=0.5)
    assert len(old) == 2 and old[0]["n_ok"] == 300 and old[0]["t_s"] == 1790000000.5 and old[0]["period_s"] == 0.5


def test_srsran_phy_and_sched_log():
    rows = srsran.parse_phy_log(fx("srsran_gnb.log"), mu=1, run_id="r")
    rx = [r for r in rows if r["event"] == "rx"]
    sc = [r for r in rows if r["event"] == "sched"]
    assert len(rx) == 4 and len(sc) == 2
    r0 = rx[0]
    assert (r0["rnti"], r0["harq_id"], r0["rv"], r0["crc"], r0["n_prb"], r0["n_sym"], r0["qm"], r0["tbs_bytes"]) == \
        (0x4601, 0, 0, 0, 51, 14, 2, 736)
    assert r0["sinr_db"] == 3.1 and (r0["sfn"], r0["slot"]) == (100, 7)
    assert [r["crc"] for r in rx] == [0, 1, 1, 1]
    assert rx[1]["slot_abs"] - rx[0]["slot_abs"] == 18           # 101.5 - 100.7 at 20 slots per frame
    d = sc[0]
    assert (d["rnti"], d["harq_id"], d["n_prb"], d["tbs_bytes"], d["nrtx"], d["newtx"], d["slot"]) == \
        (0x4601, 0, 51, 736, 0, 1, 3)
    i = sc[1]
    assert (i["rnti"], i["n_prb"], i["newtx"], i["tbs_bytes"], i["sfn"]) == (0x4602, 20, 1, 928, 102)


def _srs_pdu(data_len, bsr_idx):
    sdu = bytes([0x04, data_len]) + b"\xaa" * data_len             # LCID 4, 8-bit L
    bsr = bytes([macnr.UL_SHORT_BSR, (1 << 5) | bsr_idx])            # short BSR, LCG 1
    return sdu + bsr + bytes([macnr.UL_PADDING]) + b"\x00" * 5


@pytest.mark.parametrize("mode", ["udp", "dlt"])
def test_srsran_mac_pcap(tmp_path, mode):
    recs = []
    t = 1790000000.0
    for k, (sfn, sf) in enumerate([(1023, 9), (0, 0), (0, 1)]):
        ctx = macnr.build_context(0, 0x4601, k, sfn, subframe=sf)
        recs.append((t + k * 1e-3, macnr.srsran_record(ctx, _srs_pdu(100, 10 + k), mode)))
    dl = macnr.build_context(1, 0x4601, 5, 0, subframe=2)
    recs.append((t + 3e-3, macnr.srsran_record(dl, b"\x3f" * 50, mode)))
    path = str(tmp_path / "gnb_mac.pcap")
    P.write_pcap(path, P.DLT_EXPORTED_PDU, recs)
    rows = srsran.parse_mac_pcap(path, mu=1, run_id="r", rnti_map={"4601": "ue1"})
    assert len(rows) == 4
    ul = [r for r in rows if r["dir"] == "UL"]
    assert [r["harq_id"] for r in ul] == [0, 1, 2]
    assert all(r["data_bytes"] == 100 and r["ue"] == "ue1" and r["slot_exact"] == 0 for r in ul)
    assert [r["bsr_idx"] for r in ul] == [10, 11, 12] and ul[0]["bsr_bytes"] == 198.0
    assert [r["slot_abs"] for r in rows] == [0, 2, 4, 6]        # subframes 9 -> 0 -> 1 -> 2 across the wrap
    assert rows[-1]["dir"] == "DL" and rows[-1]["tbs_bytes"] == 50 and rows[-1]["data_bytes"] == -1


def test_oai_mac_pcap_ipv4_udp(tmp_path):
    ctx = macnr.build_context(0, 0x1D47, 3, 512, slot=7)
    pdu = bytes([0x44, 0x01, 0x00]) + b"\x55" * 256 + bytes([macnr.UL_PADDING])   # LCID 4, 16-bit L = 256
    pkt = P.ipv4_udp("127.0.0.1", "127.0.0.1", 9999, 9999, macnr.SIG + ctx + pdu)
    path = str(tmp_path / "oai.pcap")
    P.write_pcap(path, P.DLT_IPV4, [(1.0, pkt), (1.1, P.ipv4_udp("1.1.1.1", "2.2.2.2", 5, 6, b"not mac"))])
    rows = srsran.parse_mac_pcap(path, mu=1, stack="oai")
    assert len(rows) == 1
    r = rows[0]
    assert (r["source"], r["sfn"], r["slot"], r["slot_exact"], r["harq_id"], r["data_bytes"]) == \
        ("oai_pcap", 512, 7, 1, 3, 256)


def test_ul_pdu_walk_malformed():
    w = macnr.walk_ul_pdu(bytes([0x04, 200, 1, 2, 3]))            # L = 200 but only 3 bytes follow
    assert not w["ok"]
    w = macnr.walk_ul_pdu(bytes([58, 0x46, 0x01, 0x05, 2, 9, 9, macnr.UL_PADDING]))   # C-RNTI CE + LCID 5
    assert w["ok"] and w["data_bytes"] == 2 and w["lcids"] == [58, 5, 63]


def test_pcapng_reader(tmp_path):
    # SHB + IDB (Ethernet, ns resolution) + one EPB carrying IPv4/UDP
    def blk(t, body):
        body += b"\x00" * ((4 - len(body) % 4) % 4)
        n = 12 + len(body)
        return struct.pack("<II", t, n) + body + struct.pack("<I", n)
    shb = blk(0x0A0D0D0A, struct.pack("<IHHq", 0x1A2B3C4D, 1, 0, -1))
    idb = blk(1, struct.pack("<HHI", P.DLT_EN10MB, 0, 65535) + struct.pack("<HHB3x", 9, 1, 9) + b"\x00" * 4)
    eth = b"\x00" * 12 + b"\x08\x00" + P.ipv4_udp("10.0.0.1", "10.0.0.2", 1, 5201, probe.pack(0, 7, 3, 0, 1, 40, 5, 40))
    ts = 1_790_000_000_123_456_789
    epb = blk(6, struct.pack("<IIIII", 0, ts >> 32, ts & 0xFFFFFFFF, len(eth), len(eth)) + eth)
    path = tmp_path / "x.pcapng"
    path.write_bytes(shb + idb + epb)
    (t, lt, fr), = list(P.iter_pcap(str(path)))
    assert lt == P.DLT_EN10MB and t == pytest.approx(1790000000.123456789)
    h = probe.unpack(P.udp_payload(lt, fr)[4])
    assert h["seq"] == 7 and h["frame_id"] == 3


# ------------------------------------------------------------------ OAI
def test_oai_macstats_snapshots():
    rows = oai.parse_macstats(fx("oai_nrMAC_stats.log"), run_id="r", rnti_map={"1d47": "ue1"})
    ul = [r for r in rows if r["dir"] == "UL" and r["ue"] == "ue1"]
    assert len(ul) == 2
    a, b = ul
    assert (a["tx_r0"], a["tx_r1"], a["tx_r2"], a["tx_r3"], a["n_fail"]) == (500, 10, 1, 0, 0)
    assert (b["tx_r0"], b["tx_r1"], b["tx_r2"], b["tx_r3"], b["n_fail"]) == (500, 15, 1, 1, 1)
    assert a["n_ok"] == 500 and b["n_nok"] == 15 + 1 + 1 + 1
    assert a["mcs"] == 20 and b["mcs"] == 19 and a["snr_db"] == 22.5 and a["n_prb"] == 51 and a["bler"] == 0.02
    assert a["period_s"] == pytest.approx(1.0) and a["brate_bps"] == pytest.approx(2.2e6) and a["phr_db"] == 50
    # the second UE's counters went backwards (re-attach): no row for that difference
    assert not [r for r in rows if r["rnti"] == 0xA0B1]
    dl = [r for r in rows if r["dir"] == "DL" and r["rnti"] == 0x1D47]
    assert dl[0]["tx_r0"] == 100 and dl[0]["tx_r1"] == 1


def test_oai_stdout_older_format():
    rows = oai.parse_macstats(fx("oai_gnb_stdout.log"))
    ul = [r for r in rows if r["dir"] == "UL"]
    assert len(ul) == 1 and (ul[0]["tx_r0"], ul[0]["tx_r1"], ul[0]["tx_r2"]) == (100, 3, 1)
    assert math.isnan(ul[0]["period_s"])


def test_oai_ttracer_text():
    rows = oai.parse_ttracer(fx("oai_ttrace.txt"), mu=1, run_id="r", day_epoch=1790640000)
    ev = [(r["event"], r["dir"]) for r in rows]
    assert ev == [("sched", "UL"), ("pc", "UL"), ("rx", "UL"), ("sched", "DL"), ("sched", "UL"), ("sched", "UL")]
    pc = rows[1]
    assert (pc["rnti"], pc["mcs"], pc["tbs_bytes"], pc["n_prb"], pc["sinr_db"]) == (7495, 9, 1313, 51, 22.5)
    rx = rows[2]
    assert rx["harq_id"] == 3 and rx["tbs_bytes"] == 1313 and rx["crc"] == 1 and rx["ue"] == "1d47"
    assert rows[0]["t_s"] == pytest.approx(1790640000 + 12 * 3600 + 0.0002)
    assert rows[5]["slot_abs"] - rows[4]["slot_abs"] == 1          # 1023.19 -> 0.0 is one slot
    assert rows[4]["slot_abs"] - rows[0]["slot_abs"] == 1023 * 20 + 19 - (100 * 20 + 7)


# ------------------------------------------------------------------ probes and OWD
def test_probe_pack_and_schedule():
    b = probe.pack(3, 9, 2, 1, 3, 30000, 123456789, 1400)
    h = probe.unpack(b)
    assert len(b) == 1400 and (h["flow"], h["seq"], h["frame_id"], h["frag"], h["n_frag"], h["frame_bytes"],
                                h["t_tx_ns"]) == (3, 9, 2, 1, 3, 30000, 123456789)
    assert probe.unpack(b"xxxx" + b[4:]) is None
    assert probe.fragments(30000, 1400) == [1400] * 21 + [600]
    assert probe.fragments(10, 1400) == [probe.HDR.size]
    ev = probe.schedule("robot", 1.0, seed=1)
    cnt = {f: sum(1 for e in ev if e[1] == f) for f in (1, 2, 3)}
    assert cnt == {1: 10, 2: 50, 3: 5}
    assert len(probe.schedule("cbr", 2.0, rate=100)) == 200


def test_owd_from_probe_logs():
    rows, frames, s = owd.from_probe_logs(fx("probe_rx.csv"), fx("probe_tx.csv"), run_id="r", ue="ue1",
                                          offset_ms=1.0, src="10.45.0.2")
    assert len(rows) == 6 and s["lost"] == 1 and rows[1]["lost"] == 1
    assert rows[0]["owd_ms"] == pytest.approx(4.0, abs=1e-9)                # 5 ms minus the 1 ms clock offset
    fr = {f["frame_id"]: f for f in frames}
    assert fr[1]["complete"] == 0 and math.isnan(fr[1]["delay_ms"])
    assert fr[2]["complete"] == 1 and fr[2]["n_rx"] == 3 and fr[2]["delay_ms"] == pytest.approx(12.0, abs=1e-3)
    # without the sender log: embedded send times, no loss visible, and the other source is kept
    rows2, _, s2 = owd.from_probe_logs(fx("probe_rx.csv"))
    assert len(rows2) == 5 and s2["lost"] == 0                   # duplicate (flow, seq) from 10.45.0.9 ignored


def test_owd_from_pcaps(tmp_path):
    tx, rx = [], []
    for k in range(5):
        b = probe.pack(0, k, k, 0, 1, 200, 0, 200)
        tx.append((100.0 + k * 0.01, P.ipv4_udp("10.45.0.2", "10.45.0.1", 40000, 5201, b)))
        if k != 3:
            rx.append((100.0 + k * 0.01 + 0.006, P.ipv4_udp("10.45.0.2", "10.45.0.1", 40000, 5201, b)))
    P.write_pcap(str(tmp_path / "tx.pcap"), P.DLT_RAW, tx)
    P.write_pcap(str(tmp_path / "rx.pcap"), P.DLT_IPV4, rx)
    rows, frames, s = owd.from_pcaps(str(tmp_path / "tx.pcap"), str(tmp_path / "rx.pcap"), port=5201)
    assert s["lost"] == 1 and s["p50_ms"] == pytest.approx(6.0, abs=1e-3)
    assert sum(f["complete"] for f in frames) == 4


def test_ingest_run(tmp_path):
    import json
    import shutil
    rd = tmp_path / "run1"
    rd.mkdir()
    for n in ("srsran_gnb.log", "srsran_metrics.jsonl", "probe_tx.csv", "probe_rx.csv"):
        shutil.copy(fx(n), rd / n)
    m = {"run_id": "run1", "experiment": "a", "stack": "srsran",
         "gnb": {"mu": 1, "tdd_pattern": "DDDSU", "metrics_period_ms": 1000},
         "ues": [{"ue": "ue1", "rnti": "4601"}], "clock": {"method": "same_host", "offset_ms": 0.0},
         "files": {"srsran_log": "srsran_gnb.log", "srsran_metrics": "srsran_metrics.jsonl",
                   "probes": {"ue1": {"tx_csv": "probe_tx.csv", "rx_csv": "probe_rx.csv", "src": "10.45.0.2"}}}}
    (rd / "manifest.json").write_text(json.dumps(m))
    tabs, summ = ingest.ingest_run(str(rd))
    assert len(tabs["sched"]) == 6 and len(tabs["ue_period"]) == 6 and len(tabs["owd"]) == 6
    assert {r["ue"] for r in tabs["sched"] if r["rnti"] == 0x4601} == {"ue1"}
    assert summ["ue1"]["lost"] == 1
    assert os.path.exists(rd / "unified" / "frames.csv")
