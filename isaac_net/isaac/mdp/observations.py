"""ObservationTerm functions for the manager-based workflow: func(env, asset_cfg=None) -> [E, R * k] (no Isaac imports).

    from isaaclab.managers import ObservationTermCfg as ObsTerm
    from isaac_net.isaac import mdp as net_mdp
    aoi = ObsTerm(func=net_mdp.net_aoi)                 # [E, R]
    acc = ObsTerm(func=net_mdp.net_access_state)        # [E, 5 R]

Every term reads the last network step of env.isaac_net (isaac/runtime.py) and uses the one normalization of
isaac/obs.py: times divided by the time scale (IsaacNetCfg.obs_time_scale_s, default 50 control steps) and clamped to
[0, 1], queue length by the frame buffer, dB by 40, flags and one-hot codes as 0 / 1. An env that has not had a
network step since its reset observes zeros, as with NetModule.obs(). Per-robot values are laid out robot-major:
column r * k + j is feature j of robot r. asset_cfg (a SceneEntityCfg of the robots asset) selects robots through its
body_ids when they are a list; with the default slice(None) all robots are observed.

    term              k   value
    net_aoi           1   age of information / time scale
    net_delivered     1   at least one message of the robot was delivered this step
    net_delay         1   delay of the newest message delivered this step / time scale (0 if none)
    net_queue         1   queued messages / frame buffer
    net_sinr          1   SINR in dB / 40
    net_los           1   line of sight to the serving gNB (1) or blocked (0)
    net_access_state  5   one-hot idle, RACH, connected, DRX-dormant, then RLF; without the access model
                          (NRConfig rach / drx at level L2) every robot is connected
"""
from __future__ import annotations

import torch

from ..config import DB_SCALE
from ..runtime import get_runtime

ACCESS_DIM = 5
TERM_WIDTH = {"net_aoi": 1, "net_delivered": 1, "net_delay": 1, "net_queue": 1, "net_sinr": 1, "net_los": 1,
              "net_access_state": ACCESS_DIM}


def _robots(asset_cfg):
    ids = getattr(asset_cfg, "body_ids", None) if asset_cfg is not None else None
    return list(ids) if isinstance(ids, (list, tuple)) else None


def feature(out: dict | None, name: str, E: int, R: int, F: int, time_scale: float, fresh: torch.Tensor | None,
            device) -> torch.Tensor:
    """[E, R, k] feature `name` (TERM_WIDTH) from a NetModule.step dict; zeros if out is None, and for envs whose
    `fresh` flag is False (no network step since their reset)."""
    k = TERM_WIDTH[name]
    if out is None:
        return torch.zeros(E, R, k, device=device)
    if name == "net_aoi":
        x = (out["aoi_s"] / time_scale).clamp(0.0, 1.0)[..., None]
    elif name == "net_delivered":
        x = out["delivered"].float()[..., None]
    elif name == "net_delay":
        dlv = out["msg_delivered"]
        cap = torch.where(dlv, out["cap"], torch.full_like(out["cap"], -(2 ** 62)))
        d = torch.nan_to_num(out["delay_s"], nan=0.0).gather(-1, cap.argmax(-1, keepdim=True))
        x = torch.where(dlv.any(-1, keepdim=True), (d / time_scale).clamp(0.0, 1.0), torch.zeros_like(d))
    elif name == "net_queue":
        x = (out["queue_len"].float() / F)[..., None]
    elif name == "net_sinr":
        x = (out["sinr_db"] / DB_SCALE)[..., None]
    elif name == "net_los":
        x = out["los"].float()[..., None]
    elif name == "net_access_state":
        st = out.get("access_state")
        if st is None:
            st = torch.full(out["delivered"].shape, 2, dtype=torch.long, device=out["delivered"].device)
        oh = torch.nn.functional.one_hot(st.long().clamp(0, 3), 4).float()
        rlf = out["rlf"].float()[..., None] if "rlf" in out else torch.zeros_like(oh[..., :1])
        x = torch.cat([oh, rlf], -1)
    else:
        raise KeyError(name)
    x = x.float()
    if fresh is not None:
        x = x * fresh.view(-1, 1, 1).to(x.dtype)
    return x


def term(env, name: str, asset_cfg=None) -> torch.Tensor:
    """[E, R * k] (robots selected by asset_cfg.body_ids) of feature `name` for env.isaac_net."""
    rt = get_runtime(env)
    net = rt.net
    E, R = rt.num_envs, rt.R
    if net is None:
        x = torch.zeros(E, R, TERM_WIDTH[name], device=rt.device)
    else:
        ts = net.isaac.time_scale_s(net.config)
        x = feature(rt.out, name, E, R, net.F, ts, rt.fresh, net.dev)
    ids = _robots(asset_cfg)
    if ids is not None:
        x = x[:, ids]
    return x.reshape(E, -1)


def net_aoi(env, asset_cfg=None) -> torch.Tensor:
    return term(env, "net_aoi", asset_cfg)


def net_delivered(env, asset_cfg=None) -> torch.Tensor:
    return term(env, "net_delivered", asset_cfg)


def net_delay(env, asset_cfg=None) -> torch.Tensor:
    return term(env, "net_delay", asset_cfg)


def net_queue(env, asset_cfg=None) -> torch.Tensor:
    return term(env, "net_queue", asset_cfg)


def net_sinr(env, asset_cfg=None) -> torch.Tensor:
    return term(env, "net_sinr", asset_cfg)


def net_los(env, asset_cfg=None) -> torch.Tensor:
    return term(env, "net_los", asset_cfg)


def net_access_state(env, asset_cfg=None) -> torch.Tensor:
    return term(env, "net_access_state", asset_cfg)


OBS_TERMS = {"net_aoi": net_aoi, "net_delivered": net_delivered, "net_delay": net_delay, "net_queue": net_queue,
             "net_sinr": net_sinr, "net_los": net_los, "net_access_state": net_access_state}
