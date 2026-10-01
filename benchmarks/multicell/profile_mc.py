"""Contention-robust cost: CUDA kernels launched and summed kernel time per control step
(torch.profiler over 5 steps after warm-up). Wall-clock on a shared GPU/CPU is launch-bound and noisy.

usage: python profile_mc.py > profile.jsonl
"""
import json

import torch
from torch.profiler import profile, ProfilerActivity

from common import Traffic, SIZES
from timing_mc import engines
from isaac_net.core.proto.netsim import NetSlot, Radio
from isaac_net.core.proto.netsim_mc import NetSlotMC
from isaac_net.core.radio import RadioMC


def prof(name, cfg, E, R, dev="cuda", warm=10, n=5):
    torch.manual_seed(0)
    if cfg is None:
        net, radio = NetSlot(E, R, dev, SIZES), Radio(E, dev)
        def one(t, wl):
            send, det, hid, pos = wl.inputs(); snr = radio.snr_db(pos)
            net.add_frames(t, send, det, hid, snr); net.step(t, snr, hid)
    else:
        net, radio = NetSlotMC(E, R, dev, SIZES, cfg), RadioMC(cfg, E, dev)
        def one(t, wl):
            send, det, hid, pos = wl.inputs(); rx = radio.rx_dbm(pos)
            net.add_frames(t, send, det, hid, net.serving_snr_db(rx)); net.step(t, rx, hid)
    wl = Traffic(E, R, dev, 1, 0.5, 0.1)
    for t in range(warm):
        one(t, wl)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        for t in range(warm, warm + n):
            one(t, wl)
        torch.cuda.synchronize()
    ev = [e for e in p.events() if e.device_type == torch.autograd.DeviceType.CUDA]
    kus = sum(e.device_time for e in ev) if ev and hasattr(ev[0], "device_time") else sum(e.cuda_time for e in ev)
    return {"engine": name, "E": E, "R": R, "kernels_per_step": round(len(ev) / n),
            "kernel_ms_per_step": round(kus / 1e3 / n, 2)}


if __name__ == "__main__":
    for E in (64, 256):
        for R in (16, 32):
            for name, cfg in engines(E, R, "cuda"):
                print(json.dumps(prof(name, cfg, E, R)), flush=True)
