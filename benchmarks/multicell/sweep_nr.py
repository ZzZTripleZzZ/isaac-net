"""Multi-cell sanity sweep of the NR engine (level L2), with the workload and metrics of sweep_mc.py, so the same
table can be produced for NetSlotMC (--engine mc) and compared.

usage: python sweep_nr.py --E 32 --R 32 --steps 150 [--engine nr|mc] [--isd 60] [--dl] > sweep_nr.jsonl
"""
import argparse
import json

import torch

from common import MC, SIZES, Traffic
from isaaclab_net.core.nr_engine import NRNet
from isaaclab_net.core.proto.netsim_mc import NetSlotMC
from isaaclab_net.core.radio import RadioMC

LOADS = {"L1.2k": (0.3, 0.0), "L4k": (1.0, 0.0), "L8.4k": (0.6, 0.2), "L15k": (0.0, 0.5), "L30k": (0.0, 1.0)}


def configs(isd, dl):
    base = dict(cell_layout="hex", cell_isd_m=isd, cell_center_m=(75.0, 75.0), dl=dl)
    pc = dict(ul_pc_p0_dbm=-88.0, ul_pc_alpha=1.0)
    yield "C1_center", MC(n_cells=1, **base)
    yield "C3_hex_noICI", MC(n_cells=3, ul_interference=False, dl_interference=False, **base)
    yield "C3_hex", MC(n_cells=3, **base)
    yield "C7_hex", MC(n_cells=7, **base)
    yield "C1_center_PC", MC(n_cells=1, **base, **pc)
    yield "C3_hex_PC", MC(n_cells=3, **base, **pc)
    yield "C7_hex_PC", MC(n_cells=7, **base, **pc)


def q(x, p):
    return round(float(torch.quantile(x, p)), 2) if x.numel() else None


def run(name, cfg, load, E, R, steps, seed, engine="nr", dev="cuda"):
    torch.manual_seed(seed)
    radio = RadioMC(cfg, E, dev, generator=torch.Generator(device=dev).manual_seed(seed + 7))
    if engine == "nr":
        net = NRNet(E, R, dev, SIZES, cfg, generator=torch.Generator(device=dev).manual_seed(seed + 9))
    else:
        net = NetSlotMC(E, R, dev, SIZES, cfg, seed=seed + 9)
    C = cfg.n_cells
    net.log_stats = True
    net.log_cap_max = steps - cfg.timeout_steps - 1
    net.log_sinr = True
    wl = Traffic(E, R, dev, seed + 1, *LOADS[load])
    samples, dl_samples = [], []
    for t in range(steps):
        send, det, hid, pos = wl.inputs()
        pg = radio.pathgain_db(pos)
        if engine == "mc":
            rx = pg + cfg.ue_tx_dbm
            net.add_frames(t, send, det, hid, net.serving_snr_db(rx))
            net.step(t, rx, hid)
            logs = [x for x in net.sinr_log]
        else:
            if C > 1:
                snr = net.serving_sinr_db() if t else pg[..., 0] + cfg.ue_tx_dbm - cfg.subband_noise_dbm
                net.add_frames(t, send, det, hid, snr)
                if cfg.dl:
                    net.add_dl_frames(t, net.sizes[(send - 1).clamp(min=0)] * (send > 0))
                net.step_cells(t, pg, hid)
            else:
                net.add_frames(t, send, det, hid, pg[..., 0] + cfg.ue_tx_dbm - cfg.subband_noise_dbm)
                if cfg.dl:
                    net.add_dl_frames(t, net.sizes[(send - 1).clamp(min=0)] * (send > 0))
                net.step_rx(t, pg[..., 0], hid)
            logs = [x for d, x in net.sinr_log if d == "ul"] if C > 1 else []
            if C > 1 and t >= 10:
                dl_samples += [x for d, x in net.sinr_log if d == "dl"]
        if engine == "mc" or C > 1:
            net.sinr_log.clear()
        if logs and t >= 10:
            x = torch.cat([v.to(dev) for v in logs])
            keep = torch.rand(x.shape[0], device=dev) < 0.2
            samples.append(x[keep].cpu())
    st = net.collect()
    s = torch.cat(samples) if samples else torch.zeros(0, 4)
    sinr, snr_, nsb, geo = s[:, 0], s[:, 1], s[:, 2], s[:, 3]
    dly = st["delay"] * cfg.control_step_ms
    n_dl, n_to = dly.numel(), st["x_cls"].numel()
    bytes_dl = float(torch.tensor(SIZES)[st["d_cls"].long() - 1].sum()) if n_dl else 0.0
    logged_steps = net.log_cap_max + 1
    off = E * R * logged_steps * (4000 * LOADS[load][0] + 30000 * LOADS[load][1])
    edge, centre = geo < 3.0, geo > 10.0
    if C == 1:
        iot = 0.0
    elif engine == "mc":
        iot = float(10 * torch.log10(1 + net.ioN_sum / max(net.n_slots, 1)).mean())
    else:
        iot = float(net.iot_db("ul").mean())
    n_ho = net.assoc.n_ho.float().mean() if C > 1 else torch.tensor(0.0)
    res = {"engine": engine, "cfg": name, "C": C, "isd": cfg.cell_isd_m, "load": load, "E": E, "R": R, "steps": steps,
           "offered_kB_per_env_step": round(off / E / logged_steps / 1e3, 1),
           "delivered_kB_per_env_step": round(bytes_dl / E / logged_steps / 1e3, 1),
           "delay_ms_p50": q(dly, 0.5), "delay_ms_p95": q(dly, 0.95),
           "drop_frac": round(n_to / max(n_dl + n_to, 1), 4), "overflow": st["overflow"],
           "sinr_p10": q(sinr, 0.1), "sinr_p50": q(sinr, 0.5), "sinr_p90": q(sinr, 0.9),
           "snr_minus_sinr_p50": q(snr_ - sinr, 0.5),
           "sinr_p50_edge_geo<3dB": q(sinr[edge], 0.5), "sinr_p50_centre_geo>10dB": q(sinr[centre], 0.5),
           "mean_subbands_per_TB": round(float(nsb.mean()), 2) if nsb.numel() else None,
           "mean_IoT_dB": round(iot, 2),
           "HO_per_robot_per_min": round(float(n_ho) / (steps * cfg.control_step_ms / 1e3 / 60), 3)}
    if engine == "nr" and cfg.dl and C > 1:
        d = torch.cat(dl_samples) if dl_samples else torch.zeros(0, 4)
        res.update({"dl_sinr_p50": q(d[:, 0], 0.5), "dl_snr_minus_sinr_p50": q(d[:, 1] - d[:, 0], 0.5),
                    "dl_IoT_dB": round(float(net.iot_db("dl").mean()), 2),
                    "dl_delay_ms_p50": q(st["dl_delay"] * cfg.control_step_ms, 0.5)})
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--E", type=int, default=32)
    ap.add_argument("--R", type=int, default=32)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--engine", default="nr", choices=["nr", "mc"])
    ap.add_argument("--isd", type=float, default=60.0)
    ap.add_argument("--dl", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cfgs", default="", help="comma-separated subset of config names")
    ap.add_argument("--loads", default="", help="comma-separated subset of load names")
    a = ap.parse_args()
    cf = set(a.cfgs.split(",")) if a.cfgs else None
    ld = a.loads.split(",") if a.loads else list(LOADS)
    for name, cfg in configs(a.isd, a.dl):
        if cf is not None and name not in cf:
            continue
        for load in ld:
            print(json.dumps(run(name, cfg, load, a.E, a.R, a.steps, a.seed, a.engine, a.device)), flush=True)
