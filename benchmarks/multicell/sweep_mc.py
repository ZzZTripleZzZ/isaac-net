"""Sanity sweep: SINR, delay, drops and capacity vs cell count and load (R = 32 robots per env).

usage: python sweep_mc.py --E 64 --steps 150 > sweep.jsonl
"""
import argparse
import json

import torch

from common import MC, Traffic, SIZES
from isaac_net.core.proto.netsim import TIMEOUT
from isaac_net.core.proto.netsim_mc import NetSlotMC
from isaac_net.core.radio import RadioMC

LOADS = {"L1.2k": (0.3, 0.0), "L4k": (1.0, 0.0), "L8.4k": (0.6, 0.2), "L15k": (0.0, 0.5), "L30k": (0.0, 1.0)}


def configs():
    base = dict(cell_layout="hex", cell_isd_m=60.0, cell_center_m=(75.0, 75.0))
    pc = dict(ul_pc_p0_dbm=-88.0, ul_pc_alpha=1.0)   # full path-loss compensation, about 15 dB SNR per subband
    yield "C1_fixedNI_corner", MC(n_cells=1, cell_layout="custom", cell_positions_m=((0.0, 0.0),), noise_model="fixed")
    yield "C1_center", MC(n_cells=1, **base)
    yield "C3_hex_noICI", MC(n_cells=3, ul_interference=False, **base)
    yield "C3_hex", MC(n_cells=3, **base)
    yield "C7_hex", MC(n_cells=7, **base)
    yield "C3_hex_ewma", MC(n_cells=3, li_alpha=0.05, **base)     # LA on a slow N+I average, not the last slot
    yield "C1_center_PC", MC(n_cells=1, **base, **pc)
    yield "C3_hex_PC", MC(n_cells=3, **base, **pc)
    yield "C7_hex_PC", MC(n_cells=7, **base, **pc)


def q(x, p):
    return round(float(torch.quantile(x, p)), 2) if x.numel() else None


def run(name, cfg, load, E, R, steps, seed, dev="cuda"):
    torch.manual_seed(seed)
    net = NetSlotMC(E, R, dev, SIZES, cfg)
    radio = RadioMC(cfg, E, dev)
    net.log_stats = True
    net.log_cap_max = steps - TIMEOUT - 1
    net.log_sinr = True
    wl = Traffic(E, R, dev, seed + 1, *LOADS[load])
    samples = []
    for t in range(steps):
        send, det, hid, pos = wl.inputs()
        rx = radio.rx_dbm(pos)
        snr = net.serving_snr_db(rx)
        net.add_frames(t, send, det, hid, snr)
        net.step(t, rx, hid)
        if net.sinr_log:
            x = torch.cat(net.sinr_log)
            net.sinr_log.clear()
            if t >= 10:
                keep = torch.rand(x.shape[0], device=dev) < 0.2
                samples.append(x[keep].cpu())
    st = net.collect()
    s = torch.cat(samples)
    sinr, snr_, nsb, geo = s[:, 0], s[:, 1], s[:, 2], s[:, 3]
    dly = st["delay"] * 100.0
    n_dl, n_to = dly.numel(), st["x_cls"].numel()
    bytes_dl = float(torch.tensor(SIZES)[st["d_cls"] - 1].sum()) if n_dl else 0.0
    logged_steps = net.log_cap_max + 1
    off = E * R * logged_steps * (4000 * LOADS[load][0] + 30000 * LOADS[load][1])
    edge, centre = geo < 3.0, geo > 10.0
    iot = 10 * torch.log10(1 + net.ioN_sum / max(net.n_slots, 1)).mean() if net.C > 1 else torch.tensor(0.0)
    return {"cfg": name, "C": net.C, "load": load, "E": E, "R": R, "steps": steps,
            "offered_kB_per_env_step": round(off / E / logged_steps / 1e3, 1),
            "delivered_kB_per_env_step": round(bytes_dl / E / logged_steps / 1e3, 1),
            "delay_ms_p50": q(dly, 0.5), "delay_ms_p95": q(dly, 0.95),
            "drop_frac": round(n_to / max(n_dl + n_to, 1), 4), "overflow": st["overflow"],
            "sinr_p10": q(sinr, 0.1), "sinr_p50": q(sinr, 0.5), "sinr_p90": q(sinr, 0.9),
            "snr_minus_sinr_p50": q(snr_ - sinr, 0.5),
            "sinr_p50_edge_geo<3dB": q(sinr[edge], 0.5), "sinr_p50_centre_geo>10dB": q(sinr[centre], 0.5),
            "frac_TB_edge": round(float(edge.float().mean()), 3),
            "mean_subbands_per_TB": round(float(nsb.mean()), 2),
            "mean_IoT_dB": round(float(iot), 2),
            "HO_per_robot_per_min": round(float(net.assoc.n_ho.float().mean()) / (steps * 0.1 / 60), 3)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--E", type=int, default=64)
    ap.add_argument("--R", type=int, default=32)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cfgs", default="", help="comma-separated subset of config names")
    ap.add_argument("--loads", default="", help="comma-separated subset of load names")
    a = ap.parse_args()
    cf = set(a.cfgs.split(",")) if a.cfgs else None
    ld = a.loads.split(",") if a.loads else list(LOADS)
    for name, cfg in configs():
        if cf is not None and name not in cf:
            continue
        for load in ld:
            print(json.dumps(run(name, cfg, load, a.E, a.R, a.steps, a.seed)), flush=True)
