"""Inside Isaac Lab: the fleet env's network (NetModule on a fast backend, fed by PhysX poses) equals a NetModule on
the reference engine replayed with the recorded inputs, bitwise, through partial resets made by DirectRLEnv itself.

usage: python tests/scripts/isaac_fleet_check.py --level L2-legacy --backend eager --num_envs 8 --num_robots 4
Prints one line "CHECK {json}"; exit code 0 if every recorded step matched. Run by tests/test_isaac_env.py.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(conflict_handler="resolve")
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--num_robots", type=int, default=4)
parser.add_argument("--level", default="L2-legacy")
parser.add_argument("--backend", default="eager")
parser.add_argument("--steps", type=int, default=60)
add_launcher_args(parser)
args = parser.parse_args()
_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _root not in sys.path:
    sys.path.insert(0, _root)

KEYS = ("newest_cap", "last_cap", "aoi_s", "queue_len", "queue_bytes", "delivered", "tag_delivered", "sinr_db",
        "msg_delivered", "timed_out", "cap", "t")


def main():
    import torch

    from isaaclab_net.examples.isaac_fleet_env import NetFleetEnv, make_cfg
    from isaaclab_net.isaac import NetModule, TrafficRequest

    cfg = make_cfg(args.num_envs, args.num_robots, args.level, device="cuda:0", backend=args.backend)
    cfg.net_seed = 1234
    with launch_simulation(cfg, args):
        env = NetFleetEnv(cfg)
        net = env.net
        log = []
        o_submit, o_step, o_reset = net.submit, net.step, net.reset

        def submit(t, req):
            log.append(("submit", req.send.clone(), None if req.tag is None else req.tag.clone(),
                        torch.cuda.get_rng_state()))
            return o_submit(t, req)

        def step(t, poses, cur_tag=None, blocked_fn=None):
            out = o_step(t, poses, cur_tag=cur_tag, blocked_fn=blocked_fn)
            log.append(("step", poses.clone(), None if cur_tag is None else cur_tag.clone(),
                        {k: out[k].clone() for k in KEYS if k in out}))
            return out

        def reset(env_ids=None):
            log.append(("reset", None if env_ids is None else torch.as_tensor(env_ids, device=env.device).clone()))
            return o_reset(env_ids)

        net.submit, net.step, net.reset = submit, step, reset
        env.reset()
        E, A = env.num_envs, cfg.action_space
        n_resets = 0
        with torch.inference_mode():
            for k in range(args.steps):
                if k in (17, 38):      # let DirectRLEnv time out a subset: a real partial reset in its own flow
                    ids = torch.arange(k % 3, E, 3, device=env.device)
                    env.episode_length_buf[ids] = env.max_episode_length - 1
                    n_resets += 1
                env.step(2 * torch.rand(E, A, device=env.device) - 1)
        # replay on the reference engine, same seed, same inputs, same RNG stream
        rep = NetModule(args.level, E, args.num_robots, env.device, net.config, "reference",
                        pose_chunks=net.pose_chunks, gnb_pos=net.gnb.tolist(), seed=net.seed)
        worst, steps, partial = {}, 0, 0
        with torch.inference_mode():
            for ev in log:
                if ev[0] == "reset":
                    rep.reset(ev[1])
                    partial += int(ev[1] is not None and ev[1].numel() < E)
                elif ev[0] == "submit":
                    torch.cuda.set_rng_state(ev[3])
                    rep.submit(None, TrafficRequest(ev[1], ev[2]))
                else:
                    o = rep.step(None, ev[1], cur_tag=ev[2])
                    for key, v in ev[3].items():
                        a, b = v.double(), o[key].double()
                        both_nan = torch.isnan(a) & torch.isnan(b)
                        d = torch.where(both_nan, torch.zeros_like(a), (a - b).abs())
                        worst[key] = max(worst.get(key, 0.0), float(d.max()) if d.numel() else 0.0)
                    steps += 1
        delivered = float(sum(ev[3]["delivered"].float().mean() for ev in log if ev[0] == "step") / max(steps, 1))
        res = dict(level=args.level, backend=args.backend, E=E, R=args.num_robots, steps=steps,
                   partial_resets=partial, forced=n_resets, delivered_frac=delivered, worst=worst,
                   ok=bool(steps > 0 and partial >= 2 and all(v == 0.0 for v in worst.values())))
        print("CHECK " + json.dumps(res), flush=True)
        env.close()
    sys.exit(0 if res["ok"] else 1)


if __name__ == "__main__":
    main()
