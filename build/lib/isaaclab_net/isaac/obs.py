"""Per-robot network observation with feature selection (see isaac/config.py for the features and their scaling).

    obs = NetObs(("aoi", "sinr", "delay_history"), E, R, config, n_cells=1, history=4, device="cuda")
    obs.update(out)        # after every NetModule.step (NetModule does this itself)
    obs.reset(env_ids)     # zero the features and the delay history of env_ids (NetModule.reset does this)
    x = obs.get()          # [E, R, obs.dim]

One normalization for every feature: times (AoI, delays) divided by the time scale and clamped to [0, 1], queue
length by the frame buffer, queue bytes by frame_buffer * max(msg_sizes), dB quantities by 40 (RSRP relative to the
config's nominal noise floor), flags and one-hot codes as 0 / 1. An env that has not stepped since its reset
observes zeros.
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch

from ..core.config import NRConfig
from ..core.proto.netsim import env_index, fill_rows
from .config import DB_SCALE, TIME_SCALE_STEPS, feature_dims


class NetObs:
    def __init__(self, features: Sequence[str], num_envs: int, num_robots: int, config: Optional[NRConfig] = None,
                 n_cells: int = 1, history: int = 4, time_scale_s: Optional[float] = None, device="cpu"):
        cfg = config or NRConfig()
        self.dims = feature_dims(features, cfg.frame_buffer, n_cells, history)
        self.features = tuple(self.dims)
        self.dim = sum(self.dims.values())
        self.E, self.R, self.F, self.G, self.K = num_envs, num_robots, cfg.frame_buffer, n_cells, history
        self.dev = torch.device(device)
        self.time_scale = float(time_scale_s if time_scale_s is not None
                                else TIME_SCALE_STEPS * cfg.control_step_ms / 1000.0)
        self.q_bytes = float(cfg.frame_buffer * max(cfg.msg_sizes))
        self.noise_ref = float(cfg.subband_noise_dbm)
        self.x = torch.zeros(num_envs, num_robots, self.dim, device=self.dev)
        self.hist = torch.zeros(num_envs, num_robots, history, device=self.dev)

    def _t(self, s: torch.Tensor) -> torch.Tensor:
        return (s / self.time_scale).clamp(0.0, 1.0)

    def reset(self, env_ids=None):
        ids = env_index(env_ids, self.E, self.dev)
        fill_rows(self.x, ids, 0.0)
        fill_rows(self.hist, ids, 0.0)

    def update(self, out: dict) -> torch.Tensor:
        """Features of one NetModule.step output dict; also advances the delay history."""
        dlv = out["msg_delivered"]
        delay = torch.nan_to_num(out["delay_s"], nan=0.0)
        if "delay_history" in self.dims:
            # the newest delivered message of the robot this step (largest capture step among delivered slots)
            cap = torch.where(dlv, out["cap"], torch.full_like(out["cap"], -(2 ** 62)))
            idx = cap.argmax(-1, keepdim=True)
            newest = self._t(delay.gather(-1, idx))                                  # [E,R,1]
            any_d = dlv.any(-1, keepdim=True)
            shifted = torch.cat([newest, self.hist[..., :-1]], -1)
            self.hist = torch.where(any_d, shifted, self.hist)
        cols = []
        for f in self.features:
            if f == "delivered_mask":
                cols.append(dlv.float())
            elif f == "msg_delay":
                cols.append(torch.where(dlv, self._t(delay), torch.zeros_like(delay)))
            elif f == "aoi":
                cols.append(self._t(out["aoi_s"])[..., None])
            elif f == "queue_len":
                cols.append((out["queue_len"].float() / self.F)[..., None])
            elif f == "queue_bytes":
                cols.append((out["queue_bytes"].float() / self.q_bytes)[..., None])
            elif f == "sinr":
                cols.append((out["sinr_db"] / DB_SCALE)[..., None])
            elif f == "rsrp":
                cols.append(((out["rsrp_dbm"] - self.noise_ref) / DB_SCALE)[..., None])
            elif f == "serving_cell":
                cols.append(torch.nn.functional.one_hot(out["serving"].long().clamp(0, self.G - 1), self.G).float())
            elif f == "last_delivered":
                cols.append(out["delivered"].float()[..., None])
            elif f == "delay_history":
                cols.append(self.hist)
            elif f == "blocked":
                cols.append(out["blocked"].float()[..., None])
        self.x = torch.cat(cols, -1) if cols else self.x
        return self.x

    def get(self) -> torch.Tensor:
        return self.x
