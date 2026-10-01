"""Closed-loop check of the OAI bridge: robots random-walking in a 60 m arena whose uplink is carried by OAI rfsim
through OaiNet (NetBase submit / step dicts), next to the same traffic on the NR engine with a given preset.

    python benchmarks/oai/closed_loop.py --steps 100 --robots 2 --preset benchmarks/oai/presets/oai_rfsim.json

Each robot sends a 4000-byte frame with probability --p per 100 ms step. OaiNet paces the steps in rfsim virtual
time and moves each UE with an attenuation clip(--snr-ref - SNR, 0, --max-atten) from the legacy radio's SNR.
Prints delivery, delay quantiles and the mean age of information of both runs as JSON.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))

from isaac_net.bridges.oai import DockerOaiStack, OaiBridge  # noqa: E402
from isaac_net.bridges.oai.bridge import summarize_steps  # noqa: E402
from isaac_net.bridges.oai.net import OaiNet  # noqa: E402
from isaac_net.core.engine import make_engine  # noqa: E402
from isaac_net.core.proto.netsim import Radio  # noqa: E402
from isaac_net.tools.measure.preset import load_preset  # noqa: E402

SIZES = (4000.0, 30000.0)


def episode(net, R, steps, p, seed, L=60.0):
    """R robots random-walking in an L x L arena with the gNB at the corner (the legacy Radio), each sending a
    4000-byte frame with probability p per step; delivery, delay and age of information from the step dicts."""
    g = torch.Generator().manual_seed(seed)
    radio = Radio(1, "cpu", generator=torch.Generator().manual_seed(seed))
    pos = torch.rand(1, R, 2, generator=g) * L
    net.reset()
    delays, sent, last, aoi = [], 0, torch.zeros(1, R), []
    for k in range(steps):
        send = (torch.rand(1, R, generator=g) < p).long()
        sent += int(send.sum())
        net.submit(None, send)
        pos = (pos + (torch.rand(1, R, 2, generator=g) * 2 - 1) * 0.3).clamp(0, L)
        o = net.step(None, radio.snr_db(pos))
        delays += (o["delay"][o["delivered"]] * 100.0).tolist()
        last = torch.maximum(last, o["newest"].float())
        aoi.append(float((k + 1 - last).mean()))
    d = np.asarray(delays)
    return {"frames_sent": sent, "delivered": len(d), "delay_p50_ms": float(np.percentile(d, 50)) if len(d) else None,
            "delay_p95_ms": float(np.percentile(d, 95)) if len(d) else None, "aoi_mean_steps": float(np.mean(aoi))}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--robots", type=int, default=2)
    ap.add_argument("--p", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--snr-ref", type=float, default=45.0)
    ap.add_argument("--max-atten", type=float, default=30.0)
    ap.add_argument("--preset", default=None)
    ap.add_argument("--log-dir", default=None)
    a = ap.parse_args(argv)
    out = {}
    stack = DockerOaiStack(n_ue=a.robots)
    br = OaiBridge(stack, step_dt=0.1, pacing="virtual", log_dir=a.log_dir)
    net = OaiNet(1, a.robots, "cpu", SIZES, br, snr_ref_db=a.snr_ref, ploss_max_db=a.max_atten)
    t0 = time.time()
    out["oai_rfsim"] = episode(net, a.robots, a.steps, a.p, a.seed)
    out["oai_rfsim"].update(wall_s=time.time() - t0, **summarize_steps(br.steps))
    for k in range(a.robots):
        stack.set_pathloss(k, 0.0)
    net.close()
    stack.close()
    cfg = load_preset(a.preset) if a.preset else None
    if cfg is not None:
        eng = make_engine("L2", 1, a.robots, "cpu", cfg.with_(msg_sizes=SIZES))
        out["engine_" + os.path.splitext(os.path.basename(a.preset))[0]] = episode(eng, a.robots, a.steps, a.p, a.seed)
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
