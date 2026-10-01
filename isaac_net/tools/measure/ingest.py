"""Run manifests and ingestion: raw gNB / probe files of one run -> the four unified tables.

A campaign is a directory of run directories, each with a ``manifest.json`` (format in
docs/measurement-protocol.md, "Run manifest"). ``ingest_run`` converts one run into ``<run>/unified/*.csv``;
``ingest_campaign`` does every run and concatenates the tables into ``<campaign>/unified/*.csv`` plus
``<campaign>/unified/runs.json`` (the manifests), which is what calibrate.py reads.

  python -m isaac_net.tools.measure.ingest CAMPAIGN_DIR
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from . import oai, owd, srsran
from .schema import TABLES, read_table, write_table

TDD_PERIOD_MS = {0: 0.5, 1: 0.625, 2: 1.0, 3: 1.25, 4: 2.0, 5: 2.5, 6: 5.0, 7: 10.0}   # 38.331 enum index


def tdd_pattern(period_slots, nof_dl_slots, nof_dl_symbols, nof_ul_slots, nof_ul_symbols):
    """NRConfig (tdd_pattern, special_split) of a single-pattern TDD-UL-DL-ConfigCommon.

    srsRAN: ``cell_cfg.tdd_ul_dl_cfg`` gives dl_ul_tx_period (slots), nof_dl_slots, nof_dl_symbols, nof_ul_slots,
    nof_ul_symbols. OAI: period_slots = TDD_PERIOD_MS[dl_UL_TransmissionPeriodicity] * 2^mu, and the nrof* fields.
    The partial DL symbols sit in the slot after the full DL slots and the partial UL symbols in the slot before
    the full UL slots (38.213 Sec. 11.1); NRConfig has one special slot per period, so both must fall in the same
    slot (period = DL + UL + 1), or there must be no partial symbols (period = DL + UL)."""
    n_s = period_slots - nof_dl_slots - nof_ul_slots
    if n_s == 0 and nof_dl_symbols == 0 and nof_ul_symbols == 0:
        return "D" * nof_dl_slots + "U" * nof_ul_slots, (10, 2, 2)
    if n_s != 1:
        raise ValueError(f"TDD pattern with {n_s} flexible slots cannot be expressed as one special slot per period")
    guard = 14 - nof_dl_symbols - nof_ul_symbols
    if guard < 0:
        raise ValueError("more than 14 symbols in the special slot")
    return "D" * nof_dl_slots + "S" + "U" * nof_ul_slots, (nof_dl_symbols, guard, nof_ul_symbols)


def load_manifest(run_dir):
    with open(os.path.join(run_dir, "manifest.json")) as f:
        m = json.load(f)
    m.setdefault("run_id", os.path.basename(os.path.normpath(run_dir)))
    return m


def _rnti_map(m):
    out = {}
    for u in m.get("ues", []):
        if u.get("rnti") not in (None, ""):
            r = u["rnti"]
            r = int(r, 16) if isinstance(r, str) else int(r)
            out[r] = u["ue"]
    return out


def ingest_run(run_dir, write=True):
    """Parse every file the manifest lists. Returns {table: rows} and, with write, stores <run>/unified/."""
    m = load_manifest(run_dir)
    rid, g, files = m["run_id"], m.get("gnb", {}), m.get("files", {})
    mu, rmap = int(g.get("mu", 1)), _rnti_map(m)
    p = lambda k: os.path.join(run_dir, files[k])
    tabs = {t: [] for t in TABLES}
    if "srsran_metrics" in files:
        per = g.get("metrics_period_ms")
        tabs["ue_period"] += srsran.parse_metrics_json(p("srsran_metrics"), rid, rmap,
                                                       per / 1000 if per else None)
    if "srsran_pcap" in files:
        tabs["sched"] += srsran.parse_mac_pcap(p("srsran_pcap"), mu, rid, rmap, "srsran")
    if "srsran_log" in files:
        tabs["sched"] += srsran.parse_phy_log(p("srsran_log"), mu, rid, rmap, int(g.get("mcs_table", 1)))
    if "oai_macstats" in files:
        tabs["ue_period"] += oai.parse_macstats(p("oai_macstats"), rid, rmap)
    if "oai_pcap" in files:
        tabs["sched"] += srsran.parse_mac_pcap(p("oai_pcap"), mu, rid, rmap, "oai")
    if "oai_ttrace" in files:
        tabs["sched"] += oai.parse_ttracer(p("oai_ttrace"), mu, rid, rmap, m.get("ttrace_day_epoch"),
                                           infer_crc=bool(m.get("ttrace_infer_crc", False)))
    off = float(m.get("clock", {}).get("offset_ms", 0.0))
    summaries = {}
    for ue, spec in (files.get("probes") or {}).items():
        o = float(spec.get("offset_ms", off))
        if "rx_csv" in spec:
            rows, frames, s = owd.from_probe_logs(os.path.join(run_dir, spec["rx_csv"]),
                                                  os.path.join(run_dir, spec["tx_csv"]) if spec.get("tx_csv") else None,
                                                  rid, ue, o, spec.get("src"))
        else:
            rows, frames, s = owd.from_pcaps(os.path.join(run_dir, spec["tx_pcap"]),
                                             os.path.join(run_dir, spec["rx_pcap"]), rid, ue, o, spec.get("port"))
        tabs["owd"] += rows
        tabs["frames"] += frames
        summaries[ue] = s
    if write:
        d = os.path.join(run_dir, "unified")
        for t, rows in tabs.items():
            write_table(rows, t, os.path.join(d, f"{t}.csv"))
        with open(os.path.join(d, "owd_summary.json"), "w") as f:
            json.dump(summaries, f, indent=1)
    return tabs, summaries


def find_runs(campaign_dir):
    return sorted(os.path.join(campaign_dir, d) for d in os.listdir(campaign_dir)
                  if os.path.isfile(os.path.join(campaign_dir, d, "manifest.json")))


def ingest_campaign(campaign_dir, reparse=True):
    """Ingest every run and write the concatenated tables to <campaign>/unified/."""
    allt = {t: [] for t in TABLES}
    manifests = []
    for rd in find_runs(campaign_dir):
        if reparse or not os.path.isdir(os.path.join(rd, "unified")):
            tabs, s = ingest_run(rd)
        else:
            tabs = {t: read_table(os.path.join(rd, "unified", f"{t}.csv"), t) for t in TABLES}
        for t in TABLES:
            allt[t] += tabs[t]
        manifests.append(load_manifest(rd))
    d = os.path.join(campaign_dir, "unified")
    for t, rows in allt.items():
        write_table(rows, t, os.path.join(d, f"{t}.csv"))
    with open(os.path.join(d, "runs.json"), "w") as f:
        json.dump(manifests, f, indent=1)
    return allt, manifests


def load_campaign(campaign_dir):
    d = os.path.join(campaign_dir, "unified")
    tabs = {t: read_table(os.path.join(d, f"{t}.csv"), t) for t in TABLES}
    with open(os.path.join(d, "runs.json")) as f:
        return tabs, json.load(f)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("campaign", help="directory of run directories with manifest.json")
    ap.add_argument("--run", help="ingest only this run directory")
    a = ap.parse_args(argv)
    if a.run:
        tabs, s = ingest_run(a.run)
    else:
        tabs, _ = ingest_campaign(a.campaign)
    print({t: len(r) for t, r in tabs.items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
