"""NR engine (level L2, reference backend) against OAI rfsim on the same traffic.

Every latency and UE-count run of the campaign is replayed through the engine with tools/measure/replay.py: the
measured frame arrival times and sizes (virtual time) go into NRNet, one TDD period per control step, and the
engine's frame delays are compared with OAI's. Presets: oai_like(), lena_match (lena_like(); with the Sionna PDSCH
tables and 38.214 TBS when the local 5G-LENA tables are missing) and, if given, the fitted oai_rfsim preset. All
run with fading off (rfsim's channel is AWGN) at a high SNR (OAI reached MCS 28 at 0 dB attenuation).

    python benchmarks/oai/compare_engine.py --work ~/oai_rfsim/campaign --preset benchmarks/oai/presets/oai_rfsim.json
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))

from isaac_net.core import config as C  # noqa: E402
from isaac_net.tools.measure import ingest  # noqa: E402
from isaac_net.tools.measure.preset import load_preset  # noqa: E402
from isaac_net.tools.measure.replay import ks, replay  # noqa: E402
from isaac_net.tools.measure.schema import read_table  # noqa: E402


def lena_match_cfg():
    cfg = C.lena_match(fading=False)
    try:
        from isaac_net.core.phy import PHY
        PHY("ul", cfg.mcs_table, "cpu", bler_target=cfg.bler_target, source=cfg.bler_source)
        return cfg, "lena_match"
    except Exception:  # noqa: BLE001 - LENA tables are generated locally, often absent
        return C.lena_match(fading=False, bler_source="pdsch", tbs_mode="38214", harq_combining="cc"), \
            "lena_match (pdsch tables)"


def w1(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if not a.size or not b.size:
        return float("nan")
    qs = (np.arange(200) + 0.5) / 200
    return float(np.mean(np.abs(np.quantile(a, qs) - np.quantile(b, qs))))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", required=True)
    ap.add_argument("--preset", default=None, help="fitted oai_rfsim preset file")
    ap.add_argument("--out", default=os.path.join(HERE, "results", "engine_compare.csv"))
    ap.add_argument("--snr-db", type=float, default=30.0)
    ap.add_argument("--max-s", type=float, default=15.0)
    ap.add_argument("--replicas", type=int, default=4)
    ap.add_argument("--configs", default="default,sr,pp", help="MAC configs (campaign_<name>) to replay")
    ap.add_argument("--presets", default="oai_like,lena_match,oai_rfsim", help="which presets to replay")
    a = ap.parse_args(argv)
    want = a.presets.split(",")
    lm, lm_name = lena_match_cfg()
    presets = [("oai_like", C.oai_like(fading=False)), (lm_name, lm)]
    presets = [p for p in presets if p[0].split(" ")[0] in want]
    if a.preset and "oai_rfsim" in want:
        presets.append(("oai_rfsim", load_preset(a.preset, fading=False)))
    rows = []
    for cname in a.configs.split(","):
        camp = os.path.join(a.work, f"campaign_{cname}")
        if not os.path.isdir(camp):
            continue
        for rd in ingest.find_runs(camp):
            m = ingest.load_manifest(rd)
            if m["experiment"] not in ("a", "b", "d"):
                continue
            fr = read_table(os.path.join(rd, "unified", "frames.csv"), "frames") \
                if os.path.exists(os.path.join(rd, "unified", "frames.csv")) else ingest.ingest_run(rd)[0]["frames"]
            if not fr:
                continue
            ues = sorted({f["ue"] for f in fr})
            t0 = min(f["t_tx_first_s"] for f in fr)
            arr = [(f["t_tx_first_s"] - t0, ues.index(f["ue"]), f["frame_bytes"]) for f in fr]
            keep = [i for i, x in enumerate(arr) if x[0] <= a.max_s]
            meas = np.array([fr[i]["delay_ms"] for i in keep], float)
            tr = m.get("traffic", {})
            base = {"mac_config": cname, "run_id": m["run_id"], "n_ue": len(ues), "size_b": tr.get("size"),
                    "rate_hz": tr.get("rate_hz"), "frames": len(keep),
                    "oai_p50_ms": round(float(np.nanpercentile(meas, 50)), 2),
                    "oai_p95_ms": round(float(np.nanpercentile(meas, 95)), 2),
                    "oai_delivered": round(float(np.mean(np.isfinite(meas))), 4)}
            for pname, cfg in presets:
                sim = replay(cfg, arr, len(ues), a.snr_db, replicas=a.replicas, max_s=a.max_s)[:, keep]
                s = sim.ravel()
                rows.append({**base, "preset": pname,
                             "engine_p50_ms": round(float(np.nanpercentile(s, 50)), 2),
                             "engine_p95_ms": round(float(np.nanpercentile(s, 95)), 2),
                             "engine_delivered": round(float(np.mean(np.isfinite(s))), 4),
                             "w1_ms": round(w1(meas, s), 2), "ks": round(ks(meas, s), 3)})
                print(rows[-1], flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {a.out} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
