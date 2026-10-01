"""ms per control step: NetSlot (+Radio) vs NetSlotMC (+RadioMC) at C in {1 fixed-NI, 3, 7}.
Each timed step = radio evaluation + add_frames + 40 UL slots. Moderate load (ps 0.5, pl 0.1).

usage: python timing_mc.py > timing.jsonl
"""
import json
import time

import torch

from common import MC, Traffic, SIZES
from isaac_net.core.proto.netsim import NetSlot, Radio
from isaac_net.core.proto.netsim_mc import NetSlotMC
from isaac_net.core.radio import RadioMC


def engines(E, R, dev):
    yield "NetSlot", None
    yield "MC_C1_fixedNI", MC(n_cells=1, cell_layout="custom", cell_positions_m=((0.0, 0.0),), noise_model="fixed")
    yield "MC_C3_hex", MC(n_cells=3)
    yield "MC_C7_hex", MC(n_cells=7)


def bench(name, cfg, E, R, dev="cuda", warm=15, n=50):
    torch.manual_seed(0)
    if cfg is None:
        net = NetSlot(E, R, dev, SIZES); radio = Radio(E, dev)
        def one(t, wl):
            send, det, hid, pos = wl.inputs()
            snr = radio.snr_db(pos)
            net.add_frames(t, send, det, hid, snr)
            net.step(t, snr, hid)
    else:
        net = NetSlotMC(E, R, dev, SIZES, cfg); radio = RadioMC(cfg, E, dev)
        def one(t, wl):
            send, det, hid, pos = wl.inputs()
            rx = radio.rx_dbm(pos)
            net.add_frames(t, send, det, hid, net.serving_snr_db(rx))
            net.step(t, rx, hid)
    wl = Traffic(E, R, dev, 1, 0.5, 0.1)
    for t in range(warm):
        one(t, wl)
    torch.cuda.synchronize()
    ts = []
    for t in range(warm, warm + n):
        a = time.perf_counter(); one(t, wl); torch.cuda.synchronize(); ts.append(time.perf_counter() - a)
    ts = torch.tensor(ts) * 1e3
    return {"engine": name, "E": E, "R": R, "ms_per_step_median": round(float(ts.median()), 1),
            "ms_per_step_mean": round(float(ts.mean()), 1), "ms_per_step_min": round(float(ts.min()), 1)}


if __name__ == "__main__":
    for E in (64, 256):
        for R in (16, 32):
            for name, cfg in engines(E, R, "cuda"):
                print(json.dumps(bench(name, cfg, E, R)), flush=True)
