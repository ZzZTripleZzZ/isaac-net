"""Calibration hooks end to end on a synthetic campaign with known ground truth: config facts, latency components,
the BLER-model SINR shift, contention statistics, the engine replay fit and the preset round trip."""
import json
import math
import os
import random

import pytest
import torch

from isaaclab_net import NRConfig
from isaaclab_net.core.phy import MCS_TABLES, PHY, tbs_38214
from isaaclab_net.tools.measure import calibrate, preset, replay

GNB = {"mu": 1, "bandwidth_mhz": 20, "tdd_pattern": "DDDSU", "special_split": [10, 2, 2], "sr_period_ms": 20,
       "min_k2": 2, "olla_target_bler": 0.01, "mcs_table": 1, "metrics_period_ms": 1000}
SHIFT_TRUE = -2.0          # the synthetic stack decodes 2 dB worse than the engine's AWGN tables
RTT = 8                    # slots between HARQ transmissions
T0 = 1790000000.0


def _probe_csvs(rd, ue, frames, delay, rng, t0=T0):
    """frames: [(t, bytes)]; delay(rng) -> ms or None (lost)."""
    tx = ["flow,seq,frame_id,frag,n_frag,frame_bytes,pkt_bytes,t_tx_ns"]
    rx = ["src,flow,seq,frame_id,frag,n_frag,frame_bytes,pkt_bytes,t_tx_ns,t_rx_ns"]
    for i, (t, nb) in enumerate(frames):
        tt = int((t0 + t) * 1e9)
        tx.append(f"0,{i},{i},0,1,{nb},{nb},{tt}")
        d = delay(rng)
        if d is not None:
            rx.append(f"{ue},0,{i},{i},0,1,{nb},{nb},{tt},{tt + int(d * 1e6)}")
    (rd / f"tx_{ue}.csv").write_text("\n".join(tx) + "\n")
    (rd / f"rx_{ue}.csv").write_text("\n".join(rx) + "\n")
    return {"tx_csv": f"tx_{ue}.csv", "rx_csv": f"rx_{ue}.csv", "src": ue}


def _write_run(root, rid, exp, files, **extra):
    rd = root / rid
    rd.mkdir(exist_ok=True)
    m = {"run_id": rid, "experiment": exp, "stack": "srsran", "site": "lab", "gnb": GNB,
         "clock": {"method": "same_host", "offset_ms": 0.0}, "files": files}
    m.update(extra)
    (rd / "manifest.json").write_text(json.dumps(m))
    return rd


def _metrics(rd, sr_ms, snr, mcs, n=3):
    lines = []
    for k in range(n):
        ts = f"2026-09-29T12:00:{k + 1:02d}.000"
        ue = {"rnti": 17921, "ul_mcs": mcs, "pusch_snr_db": snr, "ul_nof_ok": 100, "ul_nof_nok": 1,
              "ul_brate": 1e6, "bsr": 0, "avg_sr_to_pusch_delay": sr_ms, "max_sr_to_pusch_delay": 2 * sr_ms}
        lines.append(json.dumps({"timestamp": ts, "cells": [{"cell_metrics": {"pci": 1}, "ue_list": [ue]}]}))
    (rd / "metrics.jsonl").write_text("\n".join(lines) + "\n")


def _phy_log(rd, sinr, n_tb, rng, phy, thr):
    """PUSCH PHY log lines of n_tb TBs at one SINR: MCS = engine choice 1 dB below, CRC drawn from the engine BLER
    at sinr + SHIFT_TRUE, retransmissions RTT slots later succeed with probability 0.9, up to 4 transmissions."""
    m = max(k for k in range(len(thr)) if thr[k] <= sinr - 1)
    q, c = MCS_TABLES[1][m]
    tbs = int(tbs_38214(torch.tensor(float(q)), torch.tensor(c / 1024), torch.tensor(51.0), torch.tensor(14.0), 12)) // 8
    p1 = float(phy.tb_error_prob(torch.tensor([m]), torch.tensor([sinr + SHIFT_TRUE]),
                                 torch.tensor([float(tbs * 8)]))[0])
    mod = {2: "QPSK", 4: "QAM16", 6: "QAM64"}[q]
    lines, slot = [], 0
    for i in range(n_tb):
        h = i % 8
        ok = rng.random() >= p1
        seq = [ok]
        while not seq[-1] and len(seq) < 4:
            seq.append(rng.random() < 0.9)
        for k, crc in enumerate(seq):
            s = slot + k * RTT
            sfn, sl = (s // 20) % 1024, s % 20
            ts = f"2026-09-29T12:00:{(s * 0.0005) % 60:09.6f}"
            lines.append(f"{ts} [PHY     ] [I] [{sfn:5d}.{sl}] PUSCH: rnti=0x4601 h_id={h} prb=[0, 51) symb=[0, 14) "
                         f"mod={mod} rv={[0, 2, 3, 1][k]} tbs={tbs} crc={'OK' if crc else 'KO'} iter=1.0 "
                         f"sinr={sinr + rng.gauss(0, 0.3):.1f}dB t=100us")
        slot += 40
    (rd / "gnb.log").write_text("\n".join(lines) + "\n")
    return m, p1


@pytest.fixture(scope="module")
def campaign(tmp_path_factory):
    root = tmp_path_factory.mktemp("campaign")
    rng = random.Random(3)
    phy = PHY("ul", 1, "cpu", bler_target=0.01)
    thr = phy.thr_ref.tolist()
    lat = lambda r: 6.0 + r.expovariate(1 / 3.0)
    # experiment a: two latency runs (one held out) and experiment d: robot-like mix
    for rid, rate, size, hold in (("a-100B-50Hz", 50, 100, False), ("a-1000B-20Hz", 20, 1000, True)):
        rd = root / rid
        rd.mkdir()
        fr = [(k / rate, size) for k in range(int(3 * rate))]
        pr = _probe_csvs(rd, "10.45.0.2", fr, lat, rng)
        _metrics(rd, 10.0, 25.0, 20)
        _write_run(root, rid, "a", {"srsran_metrics": "metrics.jsonl", "probes": {"ue1": pr}},
                   traffic={"profile": "cbr", "size": size, "rate_hz": rate}, holdout=hold,
                   ues=[{"ue": "ue1", "rnti": "4601"}])
    # experiment c: three SINR positions
    truth = {}
    for s in (6.0, 12.0, 18.0):
        rid = f"c-sinr{int(s)}"
        rd = root / rid
        rd.mkdir()
        truth[rid] = _phy_log(rd, s, 400, rng, phy, thr)
        _write_run(root, rid, "c", {"srsran_log": "gnb.log"}, radio={"sinr_setpoint_db": s})
    # experiment b: 1 and 2 UEs at 2 Mbit/s each
    for n in (1, 2):
        rid = f"b-n{n}"
        rd = root / rid
        rd.mkdir()
        probes = {}
        for u in range(n):
            fr = [(k / 100 + u * 1e-3, 2500) for k in range(300)]
            probes[f"ue{u + 1}"] = _probe_csvs(rd, f"10.45.0.{u + 2}", fr, lat, rng)
        _write_run(root, rid, "b", {"probes": probes}, n_ue=n)
    return root, truth


def test_calibrate_without_engine(campaign, tmp_path):
    root, truth = campaign
    out = calibrate.calibrate(str(root), str(tmp_path / "out"), engine_fit=False, log=lambda *a: None)
    link = json.load(open(out["params_link_srsran"]))
    assert link["mcs_inferred_from_tbs"] >= 1200
    pos = {p["run_id"]: p for p in link["positions"]}
    for rid, (m, p1) in truth.items():
        assert pos[rid]["mcs"] == m
        assert abs(pos[rid]["bler_first_tx"] - p1) < 0.08
    assert abs(link["bler_sinr_shift_db"] - SHIFT_TRUE) <= 1.0
    lat = json.load(open(out["params_latency_srsran"]))
    assert lat["components"]["sr_grant_delay_slots"] == 20              # 10 ms at mu = 1
    assert lat["components"]["ul_harq_rtt_slots"] == RTT
    r = {x["run_id"]: x for x in lat["runs"]}
    assert r["a-100B-50Hz"]["delay_ms"]["p50"] == pytest.approx(6.0 + 3.0 * math.log(2), abs=0.8)
    cont = json.load(open(out["params_contention_srsran"]))
    per_n = {x["n_ue"]: x for x in cont["per_n"]}
    assert set(per_n) == {1, 2} and per_n[2]["jain"] > 0.99
    assert per_n[1]["per_ue"]["ue1"]["goodput_bps"] == pytest.approx(2e6, rel=0.02)
    cfg = preset.load_preset(out["preset_srsran"])
    assert isinstance(cfg, NRConfig)
    assert (cfg.sr_period_slots, cfg.k2, cfg.bler_target, cfg.sr_grant_delay_slots, cfg.ul_harq_rtt_slots) == \
        (40, 2, 0.01, 20, RTT)
    assert cfg.tdd_pattern == "DDDSU" and cfg.special_split == (10, 2, 2) and cfg.proc_offset_ms == 2.5
    p = json.load(open(out["preset_srsran"]))
    assert p["base"] == "srsran_like" and "bler_sinr_shift_db" in p["calibration_knobs"]
    assert "proc_offset_ms" in p["provenance"]
    assert preset.load_preset(out["preset_srsran"], pf_window=10.0).pf_window == 10.0


@pytest.mark.slow
def test_calibrate_engine_fit(campaign, tmp_path):
    root, _ = campaign
    out = calibrate.calibrate(str(root), str(tmp_path / "out"), engine_fit=True, reparse=False,
                              grid={"proactive_grant": ["off", "per_period"]}, replicas=1, max_s=1.0,
                              log=lambda *a: None)
    lat = json.load(open(out["params_latency_srsran"]))
    fit = lat["engine_fit"]
    assert len(fit["grid"]) == 2 and fit["holdout_runs"] == ["a-1000B-20Hz"]
    b = fit["best"]
    assert b["d0_ms"] >= 0 and b["fit_w1_ms"] == min(g["fit_w1_ms"] for g in fit["grid"])
    cfg = preset.load_preset(out["preset_srsran"])
    assert cfg.proc_offset_ms == pytest.approx(round(b["d0_ms"], 3))
    assert cfg.proactive_grant == b["params"]["proactive_grant"]


def test_replay_waits_and_ordering():
    cfg = NRConfig(mu=1, tdd_pattern="DDDSU", fading=False, sr_period_slots=5, proactive_grant="off")
    arr = [(0.0001, 0, 100.0), (0.0003, 0, 100.0), (0.1, 0, 100.0), (0.2, 1, 2000.0)]
    d = replay.replay(cfg, arr, 2, 25.0, replicas=2)
    assert d.shape == (2, 4) and (d > 0).all()
    # two packets merged into one frame: the earlier one waited longer
    assert (d[:, 0] > d[:, 1]).all()
    w1, ks, d0 = replay.w1_shift([5, 6, 7, 8], [1, 2, 3, 4])
    assert d0 == pytest.approx(4.0) and w1 == pytest.approx(0.0) and ks == pytest.approx(0.0)


def test_preset_rejects_unknown_fields(tmp_path):
    p = tmp_path / "p.json"
    p.write_text(json.dumps({"base": "oai_like", "nrconfig": {"not_a_field": 1}}))
    with pytest.raises(ValueError):
        preset.load_preset(str(p))
    p.write_text(json.dumps({"base": "oai_like", "nrconfig": {"special_split": [8, 4, 2]}}))
    assert preset.load_preset(str(p)).special_split == (8, 4, 2)
    assert os.path.exists(str(p))


def test_harq_chains_ignore_decoded_only_sources():
    from isaaclab_net.tools.measure.schema import new_row
    base = dict(run_id="r", ue="ue1", dir="UL", event="rx", harq_id=0)
    rows = [new_row("sched", **base, source="srsran_phylog", slot_abs=0, rv=0, crc=0, mcs=5),
            new_row("sched", **base, source="srsran_phylog", slot_abs=8, rv=2, crc=1, mcs=5),
            new_row("sched", **base, source="srsran_pcap", slot_abs=8, crc=1)]
    ch = calibrate.harq_chains(rows)
    assert len(ch) == 1 and ch[0]["ntx"] == 2 and ch[0]["success"] and ch[0]["gaps"] == [8]
