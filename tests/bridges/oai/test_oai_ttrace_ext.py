"""T-tracer parser extensions used by the OAI bridge campaign: SDU bytes from GNB_MAC_LCID_UL and failed-CRC
inference from GNB_MAC_PUSCH_POWER_CONTROL without a decoded PDU (tools/measure/oai.py, infer_crc)."""
from isaac_net.tools.measure import oai
from isaac_net.tools.measure.calibrate import harq_chains

TRACE = """turning ON GNB_MAC_LCID_UL
01:00:00.000100000 [1790740000]: GNB_MAC_UL rnti 100 frame 10 slot 3 mcs 5 tbs 200
01:00:00.001000000 [1790740000]: GNB_MAC_PUSCH_POWER_CONTROL rnti 100 frame 10 slot 9 snrx10 50 phr 40 tpc 0 tb_size 200 txpower_calc 0 rbSize 10 mcs 5 rssi 800
01:00:00.002000000 [1790740000]: GNB_MAC_PUSCH_POWER_CONTROL rnti 100 frame 11 slot 9 snrx10 60 phr 40 tpc 0 tb_size 200 txpower_calc 0 rbSize 10 mcs 5 rssi 800
01:00:00.002000000 [1790740000]: GNB_MAC_UL_PDU_WITH_DATA gNB_ID 0 CC_id 0 rnti 100 frame 11 slot 9 harq_pid 4 data {buffer size:200}
01:00:00.002000000 [1790740000]: GNB_MAC_LCID_UL rnti 100 frame 11 slot 9 lcid 4 data_size 1200
01:00:00.003000000 [1790740000]: GNB_MAC_PUSCH_POWER_CONTROL rnti 100 frame 12 slot 4 snrx10 70 phr 40 tpc 0 tb_size 26 txpower_calc 0 rbSize 5 mcs 1 rssi 800
01:00:00.003000000 [1790740000]: GNB_MAC_UL_PDU_WITH_DATA gNB_ID 0 CC_id 0 rnti 100 frame 12 slot 4 harq_pid 5 data {buffer size:26}
"""


def test_lcid_bytes_and_crc_inference(tmp_path):
    p = tmp_path / "t.txt"
    p.write_text(TRACE)
    rows = oai.parse_ttracer(str(p), mu=1)
    rx = [r for r in rows if r["event"] == "rx"]
    assert [r["data_bytes"] for r in rx] == [150, 0]              # 1200 bits of LCID 4; a padding-only PDU
    assert all(r["mcs"] == -1 for r in rx)                        # without infer_crc: rows as before
    rows = oai.parse_ttracer(str(p), mu=1, infer_crc=True)
    rx = sorted((r for r in rows if r["event"] == "rx"), key=lambda r: r["slot_abs"])
    assert [(r["crc"], r["harq_id"]) for r in rx] == [(0, 4), (1, 4), (1, 5)]
    assert rx[0]["sinr_db"] == 5.0 and rx[1]["mcs"] == 5 and rx[1]["sinr_db"] == 6.0
    ch = harq_chains(rows)
    by = {c["tbs_bytes"]: c for c in ch}
    assert by[200]["ntx"] == 2 and by[200]["success"] and by[200]["gaps"] == [20]    # HARQ RTT 20 slots
    assert by[26]["ntx"] == 1
