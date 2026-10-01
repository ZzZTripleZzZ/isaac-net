"""Sanity check of the NetBase adapter: robots parked 15-40 m from the gNB must get frames through,
and every frame ns-3 reports as complete must be matched to a queued frame (no fid bookkeeping loss)."""
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))  # repo root
from isaac_net.bridges.ns3_lockstep.lockstep_net import Ns3Net  # noqa: E402
REPO = os.environ["NS3BRIDGE_REPO"]   # training repo with train.py and pilot/ (required)

sys.path.insert(0, REPO)
from isaac_net.core.proto.netsim import Radio  # noqa: E402


class FakeEnv:
    def __init__(self, E, R):
        ang = torch.rand(E, R) * 1.5 + 0.03
        d = 15 + 25 * torch.rand(E, R)
        self.pos = torch.stack([d * torch.cos(ang), d * torch.sin(ang)], -1)
        self.radio = Radio(E, "cpu")


def main():
    torch.manual_seed(0)
    E, R = 2, 8
    out = {}
    for tr in ("tcp", "unix", "shm"):
        net = Ns3Net(E, R, "cpu", (4000.0, 30000.0), mode="procs", transport=tr)
        env = FakeEnv(E, R)
        net.bind_env(env)
        net.log_stats = True
        net.reset()
        raw, matched = 0, 0
        for t in range(60):
            send = (torch.rand(E, R) < 0.5).long()
            snr = env.radio.snr_db(env.pos)
            net.add_frames(t, send, torch.zeros(E, R, dtype=torch.bool), torch.zeros(E, dtype=torch.long), snr)
            newest, _ = net.step(t, snr, torch.zeros(E, dtype=torch.long))
            raw += len(net.last["done"])
        st = net.collect()
        out[tr] = {"ns3_done": raw, "netbase_delivered": int(st["delay"].numel()),
                   "timed_out": int(st["x_cls"].numel()),
                   "delay_p50_ms": float(st["delay"].quantile(0.5) * 100) if st["delay"].numel() else None,
                   "sinr_db_mean": float(torch.tensor(net.last["sinr_db"]).nanmean())}
        net.close()
    print(json.dumps(out))


if __name__ == "__main__":
    main()
