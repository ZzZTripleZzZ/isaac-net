"""Wall time per 100 ms control step: legacy NetSlot vs NRNet presets, E in {16, 64, 128, 256}, R=16
(merged from nrconfig/tests/bench.py). The lena_like presets need the local 5G-LENA tables (README).
usage: python benchmarks/bench_nr.py out.json"""
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # without pip install -e .
from isaac_net.core.config import NRConfig, lena_like, netslot_compat  # noqa: E402
from isaac_net.core.nr_engine import NRNet  # noqa: E402
from isaac_net.core.proto.netsim import NetSlot  # noqa: E402

dev = "cuda"
R = 16
SIZES = (4000.0, 30000.0)
ENGINES = {
    "NetSlot(legacy)": lambda E: NetSlot(E, R, dev, SIZES),
    "NR compat (1 HARQ, 5 SB)": lambda E: NRNet(E, R, dev, SIZES, netslot_compat()),
    "NR lena_like (16 HARQ, EESM, WB PF)": lambda E: NRNet(E, R, dev, SIZES, lena_like()),
    "NR default mu1 20MHz (13 SB, EESM)": lambda E: NRNet(E, R, dev, SIZES, NRConfig()),
    "NR lena_like + DL": lambda E: NRNet(E, R, dev, SIZES, lena_like(dl=True)),
}
out = {}
for E in (16, 64, 128, 256):
    for name, make in ENGINES.items():
        torch.manual_seed(0)
        net = make(E)
        snr = torch.rand(E, R, device=dev) * 25
        z = torch.zeros(E, dtype=torch.long, device=dev)
        zb = torch.zeros(E, R, dtype=torch.bool, device=dev)
        times = []
        for t in range(15):
            send = (torch.rand(E, R, device=dev) < 0.5).long()
            net.add_frames(t, send, zb, z, snr)
            if isinstance(net, NRNet) and net.dl is not None:
                net.add_dl_frames(t, send.float() * 4000)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            net.step(t, snr, z)
            torch.cuda.synchronize()
            if t >= 3:
                times.append(time.perf_counter() - t0)
        ms = 1000 * sum(times) / len(times)
        out.setdefault(name, {})[E] = ms
        print(f"E={E:4d} {name:38s} {ms:8.1f} ms/step", flush=True)
json.dump(out, open(sys.argv[1] if len(sys.argv) > 1 else "bench.json", "w"), indent=1)
