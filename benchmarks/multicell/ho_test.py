"""Handover sanity: robot 0 drives along y = 75 m from x = 0 to x = 150 m at 0.3 m per step through
three cells on a line (x = 25, 75, 125 m; ideal boundaries at x = 50 and 100 m). The other robots are
static with moderate traffic. Robot 0 sends a 30 kB frame every step.

usage: python ho_test.py --E 64 --R 16 > ho.jsonl
"""
import argparse
import json

import torch

from common import MC, Traffic, SIZES
from isaaclab_net.core.proto.netsim import UL_PER_STEP
from isaaclab_net.core.proto.netsim_mc import NetSlotMC
from isaaclab_net.core.radio import RadioMC

LINE = dict(n_cells=3, cell_layout="custom", cell_positions_m=((25.0, 75.0), (75.0, 75.0), (125.0, 75.0)))


def variants():
    yield "default_hys3_ttt300ms_int40ms", MC(**LINE)
    yield "hys0_ttt0_int40ms", MC(**LINE, a3_hyst_db=0.0, a3_ttt_ms=0.0)
    yield "hys3_ttt300ms_int40ms_flush", MC(**LINE, ho_rlc="flush")
    yield "hys3_ttt300ms_int0", MC(**LINE, ho_interruption_ms=0.0)


def run(name, cfg, E, R, seed, dev="cuda", steps=500):
    torch.manual_seed(seed)
    net = NetSlotMC(E, R, dev, SIZES, cfg)
    radio = RadioMC(cfg, E, dev)
    net.slot_trace = 0
    net.log_stats = True
    wl = Traffic(E, R, dev, seed + 1, 0.4, 0.0, speed=0.0)
    xs = []
    for t in range(steps):
        send, det, hid, pos = wl.inputs()
        pos = pos.clone()
        x0 = 0.3 * t
        pos[:, 0, 0], pos[:, 0, 1] = x0, 75.0
        send = send.clone(); send[:, 0] = 2
        rx = radio.rx_dbm(pos)
        net.add_frames(t, send, det, hid, net.serving_snr_db(rx))
        net.step(t, rx, hid)
        xs.append(x0)
    tr = torch.stack(net.trace, 1).cpu()        # [E, slots, 4]: served, serv, in_ho, queue
    served, serv, inho = tr[..., 0], tr[..., 1].long(), tr[..., 2] > 0
    xslot = torch.tensor(xs).repeat_interleave(UL_PER_STEP)
    change = serv[:, 1:] != serv[:, :-1]
    n_ho = change.sum(1)
    queue = tr[..., 3]
    ho_x, pingpong, gaps, served_in_ho = [], 0, [], int((served[inho] > 0).sum())
    first_srv = []      # backlogged at the HO slot: time from HO to first service at the target cell
    for e in range(E):
        idx = change[e].nonzero().flatten() + 1
        cells = [int(serv[e, i]) for i in idx]
        prev = [int(serv[e, i - 1]) for i in idx]
        for j, i in enumerate(idx):
            ho_x.append(float(xslot[i]))
            if j > 0 and cells[j] == prev[j - 1] and (i - idx[j - 1]) <= 10 * UL_PER_STEP:
                pingpong += 1
            s_e = served[e]
            if queue[e, i] > 0:
                nxt = (s_e[i:] > 0).nonzero().flatten()
                if nxt.numel():
                    first_srv.append(float(nxt[0] + 1) * 2.5)
            before = (s_e[:i] > 0).nonzero().flatten()
            after = (s_e[i:] > 0).nonzero().flatten()
            if before.numel() and after.numel():
                gaps.append(float(i + after[0] - before[-1]) * 2.5)
    # baseline service gap distribution away from HOs (robot 0 served slots)
    gap_all = []
    for e in range(E):
        sidx = (served[e] > 0).nonzero().flatten()
        if sidx.numel() > 1:
            gap_all.append((sidx[1:] - sidx[:-1]).float() * 2.5)
    gap_all = torch.cat(gap_all)
    hx = torch.tensor(ho_x)
    g = torch.tensor(gaps)
    st = net.collect()
    q = lambda v, p: round(float(torch.quantile(v, p)), 1) if v.numel() else None
    return {"variant": name, "E": E, "R": R,
            "HO_per_traversal_mean": round(float(n_ho.float().mean()), 2),
            "HO_per_traversal_min_max": [int(n_ho.min()), int(n_ho.max())],
            "frac_envs_exactly_2_HO": round(float((n_ho == 2).float().mean()), 3),
            "pingpong_HOs_within_1s": pingpong,
            "HO_x_near_50_p10_p50_p90": [q(hx[hx < 75], .1), q(hx[hx < 75], .5), q(hx[hx < 75], .9)],
            "HO_x_near_100_p10_p50_p90": [q(hx[hx >= 75], .1), q(hx[hx >= 75], .5), q(hx[hx >= 75], .9)],
            "configured_interruption_ms": cfg.ho_int_slots * 2.5,
            "slots_served_during_interruption": served_in_ho,
            "service_gap_at_HO_ms_p50_p90_max": [q(g, .5), q(g, .9), round(float(g.max()), 1) if g.numel() else None],
            "backlogged_HOs": len(first_srv),
            "HO_to_first_service_ms_p10_p50_p90": [q(torch.tensor(first_srv), .1), q(torch.tensor(first_srv), .5),
                                                   q(torch.tensor(first_srv), .9)],
            "service_gap_other_ms_p50_p99": [q(gap_all, .5), q(gap_all, .99)],
            "robot0_served_kB_per_step": round(float(served.sum(1).mean()) / steps / 1e3, 2),
            "flushed_frames": int(net.n_flushed.sum()),
            "all_robots_drop_frac": round(st["x_cls"].numel() / max(st["x_cls"].numel() + st["delay"].numel(), 1), 4)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--E", type=int, default=64)
    ap.add_argument("--R", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=500)
    a = ap.parse_args()
    for name, cfg in variants():
        print(json.dumps(run(name, cfg, a.E, a.R, a.seed, steps=a.steps)), flush=True)
