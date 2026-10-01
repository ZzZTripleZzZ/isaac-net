"""Command-line options for the network of the Isaac fleet env (bench, training and check scripts).

No Isaac imports, so the options can be added before the simulation app is launched:

    add_net_args(parser)                          # --obs, --env_decimation, --net_decimation, --net_substeps, --dr
    cfg = make_cfg(E, R, level, device, backend, isaac=isaac_cfg_from_args(args))
    cfg.decimation = cfg.sim.render_interval = args.env_decimation
"""
from __future__ import annotations

from isaac_net.isaac.config import OBS_FEATURES, IsaacNetCfg

# network domain randomization used by --dr: radio parameters and cell placement (honored at L05 ... L2, NN, QA)
DR_DEMO = {"noise_dbm": (-95.0, -85.0), "shadow_sigma_db": (3.0, 9.0), "pl_exp": (3.2, 3.8),
           "gnb_offset_m": (-10.0, 10.0)}


def add_net_args(parser):
    parser.add_argument("--obs", default=None,
                        help="network observation features, comma-separated, or 'all' "
                             f"(default queue_len,aoi,sinr); one of {', '.join(OBS_FEATURES)}")
    parser.add_argument("--env_decimation", type=int, default=5, help="physics steps per env step (dt = 1/50 s)")
    parser.add_argument("--net_decimation", type=int, default=1, help="env steps per network step")
    parser.add_argument("--net_substeps", type=int, default=1, help="network steps per env step")
    parser.add_argument("--dr", action="store_true", help="randomize noise, shadowing, path loss and gNB placement "
                        "per env at every reset")
    return parser


def isaac_cfg_from_args(args, **kw) -> IsaacNetCfg:
    from isaac_net.examples.isaac_fleet_env import FLEET_OBS, fleet_isaac_cfg   # imports Isaac Lab
    obs = FLEET_OBS if not args.obs else (OBS_FEATURES if args.obs == "all" else tuple(args.obs.split(",")))
    return fleet_isaac_cfg(obs_features=obs, net_decimation=args.net_decimation, net_substeps=args.net_substeps,
                           dr_ranges=dict(DR_DEMO) if args.dr else {}, **kw)
