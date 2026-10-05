"""Ns3NetModule: the isaac_net.isaac NetModule API (isaac/net_module.py) backed by the ns-3 lockstep bridge.

    net = Ns3NetModule(NetConfig(num_envs=E, num_robots=R, device="cpu", msg_sizes=(4000., 30000.)),
                       transport="tcp", mode="procs")
    net.reset(env_ids)                                     # per-env reset (procs mode: in-process rebuild)
    net.submit(None, TrafficRequest(send[E,R], tag[E,R]))  # messages captured at the START of the step
    out = net.step(None, poses_end[E,R,3], cur_tag[E], blocked=None)   # -> dict, as NetModule.step
    out["delivered"], out["newest_cap"], out["last_cap"], out["aoi_s"], out["delay_s"], out["sinr_db"], out["ns3"]

NetConfig and TrafficRequest are the isaac layer's own classes (re-exported here); every NetConfig field left at
None takes its NRConfig default (frame_buffer, timeout_steps, control_step_ms, msg_sizes). Semantics follow
NetModule.step: t is ignored (per-env clocks), poses are env-local and taken at the END of the control step, and
the messages of the last submit are captured at the start of the step (env clock t). ns-3 moves each UE linearly
from its previous commanded pose to the new one in 4 sub-steps (STEP_INTERP_POS), the ns-3 counterpart of
NetModule's pose_chunks. The returned dict has NetModule's keys (rsrp_dbm is the ns-3 DL RSRP, serving is 0,
tag_delivered when cur_tag is given) plus `dropped` [E,R] (timed out or FIFO overflow this step) and `ns3` (the raw
per-UE ns-3 statistics and timing). `blocked` [E,R] or [E,R,1] (or blocked_fn(poses) -> [E,R,G], first gNB used)
adds BLOCKAGE_DB (20 dB) to the UE's shadowing for this step. Works on the Windows side too (transport="tcp",
spawn=False, endpoints=[...]) when numpy and torch are available; win_client.py is the stdlib-only fallback.

The earlier call pattern step(poses_end, TrafficRequest, blocked=None) is still accepted: it submits the request,
steps, and returns a NetOutput (this module's dataclass, attribute access) with the same fields as the dict.

Per-env reset: mode="procs" rebuilds only that env's ns-3 process (RNG run advances). In mode="single"
a partial reset is logical only: the env's bookkeeping is cleared and frames of the old episode that
are still in the ns-3 RLC queues are ignored when they complete (they still occupy air time).
"""
import warnings
from dataclasses import dataclass, field

import numpy as np
import torch

from ...isaac.net_module import NetConfig, TrafficRequest  # noqa: F401  (re-exported for callers)
from . import protocol as P
from .core import Ns3Lockstep

BLOCKAGE_DB = 20.0


@dataclass
class NetOutput:
    """Legacy return type of step(poses, TrafficRequest): the keys of the step dict as attributes."""
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
    ns3: dict = field(default_factory=dict)


def _resolve(cfg):
    """(E, R, device, frame_buffer, timeout_steps, step_dt, msg_sizes) from an isaac NetConfig."""
    kw = cfg.to_kwargs()
    nr = kw["config"]
    return (int(kw["num_envs"]), int(kw["num_robots"]), kw["device"], int(nr.frame_buffer), int(nr.timeout_steps),
            nr.control_step_ms / 1000.0, tuple(float(s) for s in nr.msg_sizes))


def _np(x, dtype=None):
    x = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
    return x if dtype is None else x.astype(dtype)


class Ns3NetModule:
    """The Isaac NetModule call pattern over the ns-3 lockstep bridge (see the module docstring).

    cfg: an isaac NetConfig (num_envs, num_robots, device, msg_sizes, frame_depth, timeout_steps, step_dt).
    reset(env_ids) marks envs for a rebuild at the next step; submit(t, TrafficRequest) queues this step's
    messages; step(t, poses_end [E,R,3], cur_tag, blocked=...) returns the NetModule step dict with the raw
    per-UE ns-3 statistics in out["ns3"]. close() stops the ns-3 processes.
    """
    def __init__(self, cfg, transport="tcp", mode="procs", run=1, ns3_args=None, shadow_sigma_db=6.0,
                 spawn=True, endpoints=None, host="127.0.0.1", envs_per_proc=None, seed=0):
        self.cfg = cfg
        self.E, self.R, device, self.F, self.timeout, self.step_dt, sizes = _resolve(cfg)
        self.dev = torch.device(device)
        self.sizes = np.asarray(sizes, np.float64)
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
        self.tag = np.full((E, R, F), -1, np.int64)
        self.next_fid = np.zeros((E, R), np.int64)
        self.last_cap = np.zeros((E, R), np.int64)
        self.sh = self.rng.normal(0, self.sigma, (E, R)).astype(np.float32)
        self.sinr_last = np.full((E, R), np.nan, np.float32)
        self.rsrp_last = np.full((E, R), np.nan, np.float32)
        self._pending_reset = np.ones(E, bool)
        self._req = None
        self._warned = False

    @property
    def clock(self):
        """[E] per-env episode clock (control steps since that env's reset)."""
        return torch.as_tensor(self.t.copy(), device=self.dev)

    # ------------------------------------------------------------------ API
    def _ids(self, env_ids):
        if env_ids is None:
            return np.arange(self.E)
        env_ids = _np(env_ids)
        return np.nonzero(env_ids)[0] if env_ids.dtype == bool else env_ids.astype(np.int64).reshape(-1)

    def reset(self, env_ids=None):
        ids = self._ids(env_ids)
        if ids.size == 0:
            return
        self.t[ids] = 0
        self.cap[ids] = -1
        self.rem[ids] = 0
        self.tag[ids] = -1
        self.last_cap[ids] = 0
        self.sh[ids] = self.rng.normal(0, self.sigma, (len(ids), self.R))
        self.sinr_last[ids] = np.nan
        self.rsrp_last[ids] = np.nan
        self._pending_reset[ids] = True      # ns-3 rebuild happens at the next step, with the new poses

    def queued(self):
        return torch.as_tensor((self.cap >= 0).sum(-1), device=self.dev)

    def submit(self, t, req):
        """Messages captured at the start of this control step (t is ignored: per-env clocks). Call before step.
        Returns accepted [E,R] bool (send > 0 and a free FIFO slot)."""
        self._req = req
        send = _np(req.send)
        return torch.as_tensor((send > 0) & ((self.cap >= 0).sum(-1) < self.F), device=self.dev)

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

    def step(self, t, poses=None, cur_tag=None, blocked_fn=None, *, blocked=None):
        """Advance every env by one control step (see the module docstring). Legacy form: step(poses, req,
        blocked=None) -> NetOutput."""
        if poses is not None and hasattr(poses, "send"):          # step(poses_end, TrafficRequest, blocked)
            if cur_tag is not None and blocked is None:
                blocked = cur_tag
            self.submit(None, poses)
            out = self._step(t, None, blocked, None)
            return NetOutput(**{k: out[k] for k in NetOutput.__dataclass_fields__})
        if poses is None:
            raise TypeError("step(t, poses_end, cur_tag=None, blocked_fn=None, *, blocked=None)")
        return self._step(poses, cur_tag, blocked, blocked_fn)

    def _step(self, poses, cur_tag, blocked, blocked_fn):
        E, R, F = self.E, self.R, self.F
        pos = _np(poses, np.float32)
        self._flush_resets(pos)
        req, self._req = self._req, None
        # enqueue (capture time = env clock t)
        send = np.zeros((E, R), np.int64) if req is None else _np(req.send, np.int64)
        tag_in = getattr(req, "tag", None) if req is not None else None
        count = (self.cap >= 0).sum(-1)
        new = (send > 0) & (count < F)
        overflow = ((send > 0) & ~new).astype(np.int64)
        e, r = np.nonzero(new)
        i = count[e, r]
        c = send[e, r]
        nbytes = getattr(req, "bytes", None) if req is not None else None   # optional per-message sizes
        by = _np(nbytes, np.float64)[e, r] if nbytes is not None else self.sizes[c - 1]
        f = self.next_fid[e, r]
        self.next_fid[e, r] += 1
        self.cap[e, r, i], self.cls[e, r, i], self.rem[e, r, i], self.fid[e, r, i] = self.t[e], c, by, f
        self.tag[e, r, i] = _np(tag_in, np.int64)[e, r] if tag_in is not None else -1
        fr = np.zeros(len(e), P.FRAME_IN)
        fr["env"], fr["ue"], fr["fid"], fr["bytes"] = e, r, f, np.round(by)
        sh = self.sh
        blk = None
        if blocked is None and blocked_fn is not None:
            blocked = blocked_fn(poses)
        if blocked is not None:
            blk = _np(blocked).reshape(E, R, -1)[..., 0].astype(bool)
            sh = sh + BLOCKAGE_DB * blk
        res = self.core.step(pos, fr, shadow=sh, interp=True)
        # map completions to FIFO slots
        fin = np.full((E, R, F), np.inf)
        for de, du, df, frac in zip(res["done"]["env"], res["done"]["ue"], res["done"]["fid"], res["done"]["frac"]):
            s = np.nonzero((self.fid[de, du] == df) & (self.cap[de, du] >= 0))[0]
            if len(s):
                fin[de, du, s[0]] = self.t[de] + frac
        t_step = self.t.copy()
        tf1 = self.t + 1
        cap_pre, cls_pre = self.cap.copy(), self.cls.copy()
        delivered_f = (self.cap >= 0) & np.isfinite(fin)
        newest = np.where(delivered_f, self.cap, -1).max(-1)
        self.last_cap = np.maximum(self.last_cap, newest)
        delay = np.where(delivered_f, (fin - self.cap) * self.step_dt, np.nan)
        timed = (self.cap >= 0) & ~delivered_f & ((tf1[:, None, None] - self.cap) >= self.timeout)
        tag_hit = None
        if cur_tag is not None:
            ct = _np(cur_tag, np.int64).reshape(E)
            tag_hit = (delivered_f & (self.tag == ct[:, None, None])).reshape(E, -1).any(-1) & (ct >= 0)
        gone = delivered_f | timed
        self.cap[gone], self.rem[gone], self.tag[gone] = -1, 0, -1
        order = np.argsort((self.cap < 0) * F + np.arange(F), -1, kind="stable")
        for n in ("cap", "cls", "rem", "fid", "tag"):
            setattr(self, n, np.take_along_axis(getattr(self, n), order, -1))
        self.t = tf1
        sinr, rsrp = res["sinr_db"], res["rsrp_dbm"]
        self.sinr_last = np.where(np.isfinite(sinr), sinr, self.sinr_last)
        self.rsrp_last = np.where(np.isfinite(rsrp), rsrp, self.rsrp_last)
        T = lambda x, dt=None: torch.as_tensor(x if dt is None else x.astype(dt), device=self.dev)  # noqa: E731
        out = dict(
            delivered=T(delivered_f.any(-1)),
            newest_cap=T(newest),
            last_cap=T(self.last_cap.copy()),
            aoi_s=T((self.t[:, None] - self.last_cap) * self.step_dt, np.float32),
            queue_len=T((self.cap >= 0).sum(-1)),
            queue_bytes=T(self.rem.sum(-1), np.float32),
            sinr_db=T(self.sinr_last.copy()),
            rsrp_dbm=T(self.rsrp_last.copy()),
            serving=T(np.zeros((E, R), np.int64)),
            blocked=T(blk if blk is not None else np.zeros((E, R), bool)),
            msg_delivered=T(delivered_f),
            timed_out=T(timed),
            cap=T(cap_pre),
            cls=T(cls_pre),
            delay_s=T(delay, np.float32),
            t=T(t_step),
            dropped=T(timed.sum(-1) + overflow),
        )
        if tag_hit is not None:
            out["tag_delivered"] = T(tag_hit)
        out["ns3"] = {k: res[k] for k in P.RESULT_F32 + P.RESULT_U32} | {"timing": res["timing"]}
        return out

    def close(self):
        self.core.close()
