"""Inside Isaac Lab: build a registered isaac_net task with gym.make, overlays on, and step it with random actions.

usage: python tests/scripts/isaac_tasks_check.py --task Isaac-NetFleet-Direct-v0 --num_envs 8 --steps 30
The overlays are inert in a headless run, so after the steps the script also builds NetMarkers with force=True and
draws one frame through VisualizationMarkers (prototype creation and the visualize call; without a visualizer
Isaac Lab drops the instances). Prints one line "CHECK {json}". Run by tests/test_isaac_ux.py (marker isaac).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(conflict_handler="resolve")
parser.add_argument("--task", default="Isaac-NetFleet-Direct-v0")
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--steps", type=int, default=30)
add_launcher_args(parser)
_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repo root
if _root not in sys.path:
    sys.path.insert(0, _root)
args = parser.parse_args()


def main():
    import gymnasium as gym
    import torch

    from isaac_net.examples.isaac_markers_demo import load_task_cfg, network_of
    from isaac_net.isaac import NetMarkers, NetMarkersCfg

    cfg = load_task_cfg(args.task, args.num_envs, getattr(args, "device", None) or "cuda:0", NetMarkersCfg())
    with launch_simulation(cfg, args):
        env = gym.make(args.task, cfg=cfg)
        u = env.unwrapped
        obs, _ = env.reset()
        A = env.action_space.shape[-1]
        finite = True
        with torch.inference_mode():
            for _ in range(args.steps):
                obs, rew, term, trunc, info = env.step(2 * torch.rand(u.num_envs, A, device=u.device) - 1)
                finite = finite and bool(torch.isfinite(obs["policy"]).all()) and bool(torch.isfinite(rew).all())
            net, markers, host = network_of(u)
            out = host.net_out
            forced = NetMarkers(u, net, NetMarkersCfg(prim_path="/Visuals/IsaacNetCheck"), force=True)
            drew = forced.update(host.net_poses(), out, force=True)
        res = dict(task=args.task, obs_shape=list(obs["policy"].shape), obs_finite=finite, action_dim=int(A),
                   markers=markers is not None, markers_active=bool(markers is not None and markers.active),
                   forced_draw=bool(drew), marker_instances=forced.last_frame.num_markers if drew else 0,
                   mean_aoi_s=float(out["aoi_s"].float().mean()) if out is not None else float("nan"),
                   net_steps=int(getattr(host, "steps", args.steps)))
        print("CHECK " + json.dumps(res), flush=True)
        env.close()


if __name__ == "__main__":
    main()
