"""Warehouse radio-map demo: the fleet task in Isaac Lab's warehouse, channel baked from the warehouse USD vs the
log-distance default.

    python benchmarks/isaac/warehouse_map_demo.py --num_envs 8 --num_robots 8 --steps 300 --out warehouse_demo.json

Builds WarehouseFleetEnv (isaac_net/examples/isaac_warehouse_env.py) with IsaacNetCfg.scene_map set, so
net_setup exports env_0's warehouse and bakes the radio map with Sionna RT (or loads it from the cache when the stage
did not change). Sionna RT must be importable, or $ISAAC_NET_SIONNA_PYTHON must name an interpreter that has it
($ISAAC_NET_SIONNA_VARIANT picks the Mitsuba variant, e.g. cuda). Then it runs the same scripted fleet (drive to
goal, random sends, same seeds) under three channels that differ in nothing else:

    map           channel="radio_map" from the bake
    logdist       the NRConfig default log-distance channel (40 + 35 log10 d, 6 dB shadowing), same cells
    map_blockage  the map plus the robot-body blockage add-on (moving robots are not in the static map)

and writes the delivered KPIs per arm: message delivery ratio, delay, age of information, SINR, share of hazard
time the fleet knew the hazard.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(conflict_handler="resolve")
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--num_robots", type=int, default=8)
parser.add_argument("--level", default="L2-legacy", help="L2-legacy (multi-cell NetSlotMC) or L2 (NR engine)")
parser.add_argument("--backend", default="reference")
parser.add_argument("--steps", type=int, default=300)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--arms", default="map,logdist,map_blockage")
parser.add_argument("--send_probs", default="0.6,0.3,0.1", help="probabilities of no / small / large message")
parser.add_argument("--noise", default="thermal", help="NRConfig.noise_model: thermal (cells interfere) or fixed")
parser.add_argument("--ue_tx_dbm", type=float, default=23.0, help="robot transmit power (lower it to leave the top MCS)")
parser.add_argument("--usd", default=None, help="warehouse USD (default: Simple_Warehouse/warehouse_multiple_shelves)")
parser.add_argument("--out", default="warehouse_demo.json")
add_launcher_args(parser)
_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _root not in sys.path:
    sys.path.insert(0, _root)
args = parser.parse_args()


def run_arm(env, steps, seed, R):
    import torch
    E, dev = env.num_envs, env.device
    acc = dict(sent=0.0, sent_large=0.0, delivered=0.0, timed_out=0.0, aoi=0.0, n=0, hz=0.0, known=0.0, goals=0.0,
               moved=0.0, cell0=0.0, imbalance=0.0)
    probs = torch.tensor([float(x) for x in args.send_probs.split(",")], device=dev)
    delays, sinrs = [], []
    with torch.inference_mode():            # reset inside too: the env's buffers become inference tensors
        torch.manual_seed(seed)
        env.reset()
        g = torch.Generator(device=dev)
        g.manual_seed(seed + 1)
        for _ in range(steps):
            pos = env._pos_radio()
            d = env.goal - pos
            v = d / d.norm(dim=-1, keepdim=True).clamp(min=1e-3)
            v = (v + 0.3 * torch.randn(v.shape, device=dev, generator=g)).clamp(-1, 1)
            ch = torch.multinomial(probs.expand(E * R, 3), 1, generator=g).view(E, R).float() - 1.0   # -1/0/1
            goals_before = env.ep_stats["goals"].clone()
            env.step(torch.cat([v, ch[..., None]], -1).reshape(E, -1))
            acc["goals"] += float(torch.where(env.tt > 0, env.ep_stats["goals"] - goals_before, 0.0).sum() * R)
            acc["moved"] += float((env._pos_radio() - pos).norm(dim=-1).mean())
            c0 = (env.net_out["serving"] == 0).float()
            acc["cell0"] += float(c0.mean())
            acc["imbalance"] += float((2 * c0.sum(-1) - R).abs().mean() / R)      # |n0 - n1| / R per env
            o = env.net_out
            send = env._send
            acc["sent"] += float((send > 0).sum())
            acc["sent_large"] += float((send == 2).sum())
            acc["delivered"] += float(o["msg_delivered"].sum())
            acc["timed_out"] += float(o["timed_out"].sum())
            acc["aoi"] += float(o["aoi_s"].mean())
            acc["hz"] += float(env.h_on.sum())
            acc["known"] += float((env.known & env.h_on).sum())
            acc["n"] += 1
            dl = o["delay_s"][o["msg_delivered"]]
            delays.append(dl.float().cpu())
            sinrs.append(o["sinr_db"].float().flatten().cpu())
    dly = torch.cat(delays)
    s = torch.cat(sinrs)
    return dict(msgs_sent=acc["sent"], delivery_ratio=acc["delivered"] / max(acc["sent"], 1),
                timeout_ratio=acc["timed_out"] / max(acc["sent"], 1),
                delay_mean_ms=float(dly.mean() * 1e3) if dly.numel() else None,
                delay_p95_ms=float(dly.quantile(0.95) * 1e3) if dly.numel() else None,
                aoi_mean_s=acc["aoi"] / acc["n"], sinr_mean_db=float(s.mean()), sinr_p5_db=float(s.quantile(0.05)),
                sinr_p50_db=float(s.quantile(0.5)), sinr_below_0db=float((s < 0).float().mean()),
                hazard_known_frac=acc["known"] / max(acc["hz"], 1), goals_per_robot=acc["goals"] / (E * R),
                speed_mps=acc["moved"] / acc["n"] / env.step_dt, serving_cell0_frac=acc["cell0"] / acc["n"],
                load_imbalance=acc["imbalance"] / acc["n"])


def main():

    from isaac_net.examples.isaac_warehouse_env import (WAREHOUSE_USD, WarehouseFleetEnv, make_warehouse_cfg,
                                                          warehouse_isaac_cfg, warehouse_net_config)
    from isaac_net.isaac.scene_map import resolve_scene_map

    E, R = args.num_envs, args.num_robots
    cfg = make_warehouse_cfg(E, R, args.level, device=getattr(args, "device", None) or "cuda:0",
                             backend=args.backend, isaac=warehouse_isaac_cfg(scene_map=True),
                             nr=warehouse_net_config(0.1, noise_model=args.noise, ue_tx_dbm=args.ue_tx_dbm), usd_path=args.usd or WAREHOUSE_USD)
    cfg.net_seed = args.seed
    rec = dict(E=E, R=R, level=args.level, backend=args.backend, steps=args.steps, usd=cfg.usd_path,
               send_probs=args.send_probs, noise=args.noise, ue_tx_dbm=args.ue_tx_dbm)
    with launch_simulation(cfg, args):
        t0 = time.time()
        env = WarehouseFleetEnv(cfg)
        rec["startup_s"] = time.time() - t0
        info = dict(resolve_scene_map.last_info or {})
        rec["scene_map"] = {k: info.get(k) for k in ("path", "key", "scene_hash", "export_s", "bake_s", "cached",
                                                      "coverage", "summary")}
        rec["occupancy_free_frac"] = env.occ_free_frac
        nr_map, isaac = env.net.config, env.net.isaac
        from isaac_net.core.channels.radio_map import RadioMap
        m = RadioMap.load(nr_map.radio_map_path)
        g = m.gain.view(m.C, m.H, m.W)
        rec["map"] = dict(C=m.C, H=m.H, W=m.W, bounds=m.bounds, gain_min_db=float(g.min()), gain_max_db=float(g.max()),
                          gain_mean_db=float(g.mean()))
        if m.los_prob is not None:
            rec["map"]["los_mean"] = [float(x) for x in m.los_prob.float().mean(1)]
        arms = dict(map=nr_map, logdist=nr_map.with_(channel="log_distance", radio_map_path=None),
                    map_blockage=nr_map.with_(blockage=True))
        rec["arms"] = {}
        for name in args.arms.split(","):
            env.net_setup(args.level, R, arms[name], args.backend, isaac=isaac, seed=args.seed)
            t1 = time.time()
            rec["arms"][name] = run_arm(env, args.steps, args.seed, R)
            rec["arms"][name]["wall_s"] = time.time() - t1
            print(name, json.dumps(rec["arms"][name]), flush=True)
        print("RESULT " + json.dumps(rec), flush=True)
        with open(args.out, "w") as f:
            json.dump(rec, f, indent=1)
        env.close()


if __name__ == "__main__":
    main()
