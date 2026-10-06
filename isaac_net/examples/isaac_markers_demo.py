"""Run a registered isaac_net task with the viewport overlays on, random actions, for a fixed wall-clock time.

    python -m isaac_net.examples.isaac_markers_demo --task Isaac-NetFleet-Direct-v0 --num_envs 4 --seconds 30 --viz kit
    python -m isaac_net.examples.isaac_markers_demo --task Isaac-NetFleet-Manager-v0 --num_envs 4 --viz kit

`--viz kit` opens the Kit viewport (Isaac Lab 3.0 is headless by default; without a visualizer the overlays are
inert and the script only steps the env). The camera looks at env 0. Every second it prints one line with the env
steps so far and the overlay instances drawn. Use it for the screenshot in docs/isaac-lab.md
(docs/img/isaac_markers.png): links coloured by SINR, gNB mast and coverage disc, AoI bars.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(conflict_handler="resolve")
parser.add_argument("--task", default="Isaac-NetFleet-Direct-v0")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--seconds", type=float, default=30.0)
parser.add_argument("--update_every", type=int, default=1, help="overlay redraw interval in env steps")
parser.add_argument("--line_backend", default="markers", choices=["markers", "debug_draw"])
add_launcher_args(parser)


def load_task_cfg(task: str, num_envs: int, device: str, markers):
    """The registered env cfg of `task` with num_envs, device and overlays set (before the app launches)."""
    import isaac_net.isaac.tasks  # noqa: F401
    from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

    cfg = load_cfg_from_registry(task, "env_cfg_entry_point")
    cfg.scene.num_envs = num_envs
    cfg.sim.device = device
    if hasattr(cfg, "isaac_net"):                 # manager-based: NetManagerCfg on the env cfg
        cfg.isaac_net.markers = markers
    else:
        cfg.net_markers = markers
    try:                                          # camera over env 0 (its arena is centred on the env origin)
        from isaaclab.visualizers import VisualizerCfg
        cfg.sim.default_visualizer_cfg = VisualizerCfg(eye=(90.0, 90.0, 70.0), lookat=(0.0, 0.0, 0.0))
    except (ImportError, TypeError):
        pass
    return cfg


def network_of(env):
    """(NetModule, NetMarkers or None, pose reader) of a Direct (NetEnvMixin) or manager-based (env.isaac_net) env."""
    host = env.isaac_net if hasattr(env, "isaac_net") else env
    return host.net, host.net_markers, host


def main():
    args = parser.parse_args()
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if root not in sys.path:
        sys.path.insert(0, root)
    import gymnasium as gym
    import torch

    from isaac_net.isaac import NetMarkersCfg

    cfg = load_task_cfg(args.task, args.num_envs, getattr(args, "device", None) or "cuda:0",
                        NetMarkersCfg(update_every=args.update_every, line_backend=args.line_backend))
    with launch_simulation(cfg, args):
        env = gym.make(args.task, cfg=cfg)
        u = env.unwrapped
        env.reset()
        A = env.action_space.shape[-1]
        net, markers, _ = network_of(u)
        print(f"[isaac_net] overlays {'active' if markers is not None and markers.active else 'inert (headless)'}",
              flush=True)
        t0 = last = time.time()
        steps = 0
        with torch.inference_mode():
            while time.time() - t0 < args.seconds:
                env.step(2 * torch.rand(u.num_envs, A, device=u.device) - 1)
                steps += 1
                if time.time() - last >= 1.0:
                    last = time.time()
                    n = markers.last_frame.num_markers if markers is not None and markers.last_frame else 0
                    print(f"[isaac_net] t={last - t0:5.1f}s steps={steps} overlay_instances={n}", flush=True)
        env.close()


if __name__ == "__main__":
    main()
