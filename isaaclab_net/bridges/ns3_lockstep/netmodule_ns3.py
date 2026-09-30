"""Ns3NetModule: the isaaclab_net.isaac.netmodule NetModule API backed by the ns-3 lockstep bridge.

    net = Ns3NetModule(NetConfig(num_envs=E, num_robots=R, msg_sizes=(4000., 30000.)),
                       transport="tcp", mode="procs")
    net.reset(env_ids)                                  # per-env reset (procs mode: in-process rebuild)
    out = net.step(poses_local[E,R,3], TrafficRequest(send[E,R]), blocked=None)
    out.delivered, out.newest_cap, out.delay_s, out.aoi_s, out.queue_bytes, out.sinr_db, out.ns3

Semantics follow NetModule.step: poses are env-local and taken at the END of the control step; the
messages in req are captured at the start of the step (env clock t). ns-3 moves each UE linearly from
its previous pose to the new one in 4 sub-steps (STEP_INTERP_POS), the ns-3 counterpart of
NetModule's pose_chunks. `blocked` [E,R] or [E,R,1] adds cfg blockage loss (20 dB) to the UE's
shadowing for this step. Works on the Windows side too (transport="tcp", spawn=False,
endpoints=[...]) when numpy and torch are available; win_client.py is the stdlib-only fallback.

Per-env reset: mode="procs" rebuilds only that env's ns-3 process (RNG run advances). In mode="single"
a partial reset is logical only: the env's bookkeeping is cleared and frames of the old episode that
are still in the ns-3 RLC queues are ignored when they complete (they still occupy air time).
"""
import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from . import protocol as P
from .core import Ns3Lockstep

try:
    from ...isaac.netmodule import NetConfig, NetOutput, TrafficRequest  # noqa: F401
except Exception:  # standalone copies of the three dataclasses
    @dataclass
    class NetConfig:
        num_envs: int
        num_robots: int
        device: str = "cpu"
        step_dt: float = 0.1
        frame_depth: int = 16
        msg_sizes: tuple = (1500.0, 12000.0)
        timeout_steps: int = 20

    @dataclass
    class TrafficRequest:
        send: torch.Tensor
        bytes: Optional[torch.Tensor] = None

    @dataclass
    class NetOutput:
        delivered: torch.Tensor
        newest_cap: torch.Tensor
        last_cap: torch.Tensor
        delay_s: torch.Tensor
        aoi_s: torch.Tensor
        dropped: torch.Tensor
        queue_bytes: torch.Tensor
        queue_len: torch.Tensor
        sinr_db: torch.Tensor
        blocked: torch.Tensor
        serving: torch.Tensor

BLOCKAGE_DB = 20.0


class Ns3NetModule:
    def __init__(self, cfg, transport="tcp", mode="procs", run=1, ns3_args=None, shadow_sigma_db=6.0,
                 spawn=True, endpoints=None, host="127.0.0.1", envs_per_proc=None, seed=0):
        self.cfg = cfg
        self.E, self.R, self.F = cfg.num_envs, cfg.num_robots, cfg.frame_depth
        self.dev = torch.device(cfg.device)
        self.sizes = np.asarray(cfg.msg_sizes, np.float64)
        self.core = Ns3Lockstep(self.E, self.R, mode=mode, transport=transport, run=run, ns3_args=ns3_args,
                                spawn=spawn, endpoints=endpoints, host=host, envs_per_proc=envs_per_proc)
        self.rng = np.random.default_rng(seed)
        self.sigma = shadow_sigma_db
        E, R, F = self.E, self.R, self.F
        self.t = np.zeros(E, np.int64)
        self.cap = np.full((E, R, F), -1, np.int64)
        self.cls = np.zeros((E, R, F), np.int64)
        self.rem = np.zeros((E, R, F))
        self.fid = np.zeros((E, R, F), np.int64)
        self.next_fid = np.zeros((E, R), np.int64)
        self.last_cap = np.zeros((E, R), np.int64)
        self.sh = self.rng.normal(0, self.sigma, (E, R)).astype(np.float32)
        self.sinr_last = np.full((E, R), np.nan, np.float32)
        self._pending_reset = np.ones(E, bool)
        self._warned = False

    # ------------------------------------------------------------------ API
    def _ids(self, env_ids):
        if env_ids is None:
            return np.arange(self.E)
        if isinstance(env_ids, torch.Tensor):
            env_ids = env_ids.cpu().numpy()
        env_ids = np.asarray(env_ids)
        return np.nonzero(env_ids)[0] if env_ids.dtype == bool else env_ids.astype(np.int64)

    def reset(self, env_ids=None):
        ids = self._ids(env_ids)
        if ids.size == 0:
            return
        self.t[ids] = 0
        self.cap[ids] = -1
        self.rem[ids] = 0
        self.last_cap[ids] = 0
        self.sh[ids] = self.rng.normal(0, self.sigma, (len(ids), self.R))
        self.sinr_last[ids] = np.nan
        self._pending_reset[ids] = True      # ns-3 rebuild happens at the next step, with the new poses

    def queued(self):
        return torch.as_tensor((self.cap >= 0).sum(-1), device=self.dev)

    def _flush_resets(self, pos):
        need = self._pending_reset
        if not need.any():
            return
        epp = self.core.epp
        groups = [g for g in range(self.core.G) if need[g * epp:(g + 1) * epp].all()]
        partial = [g for g in range(self.core.G) if need[g * epp:(g + 1) * epp].any() and g not in groups]
        if partial and not self._warned:
            warnings.warn("partial reset inside a multi-env ns-3 process is logical only")
            self._warned = True
        if groups:
            self.core.reset(groups=groups, pos=pos, shadow=self.sh)
        self._pending_reset[:] = False

    def step(self, pos, req, blocked=None):
        E, R, F = self.E, self.R, self.F
        pos = pos.detach().float().cpu().numpy() if isinstance(pos, torch.Tensor) else np.asarray(pos, np.float32)
        self._flush_resets(pos)
        # enqueue (capture time = env clock t)
        send = req.send.detach().cpu().numpy() if isinstance(req.send, torch.Tensor) else np.asarray(req.send)
        count = (self.cap >= 0).sum(-1)
        new = (send > 0) & (count < F)
        overflow = ((send > 0) & ~new).astype(np.int64)
        e, r = np.nonzero(new)
        i = count[e, r]
        c = send[e, r]
        by = (req.bytes.detach().cpu().numpy()[e, r] if req.bytes is not None else self.sizes[c - 1])
        f = self.next_fid[e, r]
        self.next_fid[e, r] += 1
        self.cap[e, r, i], self.cls[e, r, i], self.rem[e, r, i], self.fid[e, r, i] = self.t[e], c, by, f
        fr = np.zeros(len(e), P.FRAME_IN)
        fr["env"], fr["ue"], fr["fid"], fr["bytes"] = e, r, f, np.round(by)
        sh = self.sh
        blk = None
        if blocked is not None:
            blk = blocked.detach().cpu().numpy() if isinstance(blocked, torch.Tensor) else np.asarray(blocked)
            blk = blk.reshape(E, R, -1)[..., 0].astype(bool)
            sh = sh + BLOCKAGE_DB * blk
        res = self.core.step(pos, fr, shadow=sh, interp=True)
        # map completions to FIFO slots
        fin = np.full((E, R, F), np.inf)
        for de, du, df, frac in zip(res["done"]["env"], res["done"]["ue"], res["done"]["fid"], res["done"]["frac"]):
            s = np.nonzero((self.fid[de, du] == df) & (self.cap[de, du] >= 0))[0]
            if len(s):
                fin[de, du, s[0]] = self.t[de] + frac
        tf1 = self.t + 1
        delivered_f = (self.cap >= 0) & np.isfinite(fin)
        newest = np.where(delivered_f, self.cap, -1).max(-1)
        self.last_cap = np.maximum(self.last_cap, newest)
        delay = np.where(delivered_f, (fin - self.cap) * self.cfg.step_dt, np.nan)
        timed = (self.cap >= 0) & ~delivered_f & ((tf1[:, None, None] - self.cap) >= self.cfg.timeout_steps)
        gone = delivered_f | timed
        self.cap[gone], self.rem[gone] = -1, 0
        order = np.argsort((self.cap < 0) * F + np.arange(F), -1, kind="stable")
        for n in ("cap", "cls", "rem", "fid"):
            setattr(self, n, np.take_along_axis(getattr(self, n), order, -1))
        self.t = tf1
        sinr = res["sinr_db"]
        self.sinr_last = np.where(np.isfinite(sinr), sinr, self.sinr_last)
        T = lambda x, dt=None: torch.as_tensor(x if dt is None else x.astype(dt), device=self.dev)
        out = NetOutput(
            delivered=T(delivered_f.any(-1)),
            newest_cap=T(newest),
            last_cap=T(self.last_cap.copy()),
            delay_s=T(delay, np.float32),
            aoi_s=T((self.t[:, None] - self.last_cap) * self.cfg.step_dt, np.float32),
            dropped=T(timed.sum(-1) + overflow),
            queue_bytes=T(self.rem.sum(-1), np.float32),
            queue_len=T((self.cap >= 0).sum(-1)),
            sinr_db=T(self.sinr_last.copy()),
            blocked=T(blk if blk is not None else np.zeros((E, R), bool)),
            serving=T(np.zeros((E, R), np.int64)),
        )
        out.ns3 = {k: res[k] for k in P.RESULT_F32 + P.RESULT_U32} | {"timing": res["timing"]}
        return out

    def close(self):
        self.core.close()
