"""Isaac-side network module: one object per env batch, contract API of ARCHITECTURE.md.

Snapshot of isaac/demo (2026-09-29, still in development there); imports rewired to the package layout.

    net = NetModule(NetConfig(num_envs=E, num_robots=R, rung="L2", backend="graph"))
    net.reset(env_ids)                                   # partial reset, any subset
    net.submit(t, TrafficRequest(send[E,R], tag[E,R]))   # messages captured at the start of step t
    out = net.step(t, poses_local[E,R,3], cur_tag[E])    # advance [t, t+1) -> dict

`t` is the GLOBAL control-step index (e.g. DirectRLEnv.common_step_counter). Outputs are converted to each
env's own episode clock, so an env that reset at global step t0 sees capture times 0, 1, ... .

Engines
  L2 + backend in {"eager","graph","compile","triton"}: core.proto.netsim_fast.NetSlotFast (the fast engine,
     `graph` is bitwise equal to netsim.NetSlot under injected noise; see fast/ report). Partial resets are
     written in place into its persistent buffers so captured CUDA graphs stay valid. Its clock is global;
     every comparison inside it is relative (t+1-cap, g-sr_t, g>=wait), so neutral reset values
     (cap=-1, sr_t=-1, wait=0) make a partial reset exact.
  L0 / L1 / L2 + backend "ref": isaac.netmodule.NetModule, the per-env-clock registry port
     (L2 bit-exact to netsim.NetSlot, tests/test_port_equiv.py). Sync-free but launch-bound, eager only.

Radio (both engines): log-distance path loss + spatially correlated shadowing field per env (reset per env)
+ optional LOS blockage; SNR is evaluated on `pose_chunks` poses interpolated across the control step and
averaged in dB, which is what the fast engine consumes as its per-step SNR.
"""
from __future__ import annotations

import dataclasses
import math
from typing import Optional

import torch

from . import netmodule as _ref
from .netmodule import EnvIds, NetConfig as _RefCfg, ParamRanges, TrafficRequest  # noqa: F401

FAST_BACKENDS = ("eager", "graph", "compile", "triton")
NetConfig = _RefCfg          # one config dataclass; `backend` selects the engine


class _RadioField:
    """Per-env radio state for the fast engine (the ref engine carries its own copy of the same model)."""

    def __init__(self, cfg: NetConfig, ranges: ParamRanges):
        # reuse the ref engine's registry machinery with an L0 FIFO of depth 1 (tiny), only for radio + params
        rc = dataclasses.replace(cfg, rung="L0", frame_depth=1, backend="ref")
        self._m = _ref.NetModule(rc, ranges)

    def reset(self, ids):
        self._m.reset(ids)

    def snr_db(self, pos, blocked=None):
        return self._m.snr_db(pos, blocked)

    def params(self):
        return self._m


class NetModule:
    def __init__(self, cfg: NetConfig, ranges: Optional[ParamRanges] = None):
        self.cfg = cfg
        self.E, self.R = cfg.num_envs, cfg.num_robots
        self.dev = torch.device(cfg.device)
        self.fast = cfg.rung == "L2" and cfg.backend in FAST_BACKENDS
        ranges = ranges or ParamRanges()
        if self.fast:
            from ..core.proto import netsim_fast as nf
            assert cfg.slots_per_step == nf.UL_PER_STEP and cfg.num_subbands == nf.S
            assert cfg.frame_depth == nf.F and cfg.timeout_steps == nf.TIMEOUT, "fast engine uses netsim constants"
            self.eng = nf.NetSlotFast(self.E, self.R, cfg.device, tuple(cfg.msg_sizes), backend=cfg.backend)
            self.radio = _RadioField(cfg, ranges)
            self._S = nf.S
            self.t0 = torch.zeros(self.E, dtype=torch.long, device=self.dev)       # global step of episode start
            self.last_cap_g = torch.zeros(self.E, self.R, dtype=torch.long, device=self.dev)
            self._prev = torch.zeros(self.E, self.R, 3, device=self.dev)
            self._prev_valid = torch.zeros(self.E, dtype=torch.bool, device=self.dev)
            self._snr = torch.zeros(self.E, self.R, device=self.dev)
            self._t_next = 0
        else:
            assert cfg.backend == "ref", f"backend {cfg.backend} only exists for L2"
            self.eng = _ref.NetModule(cfg, ranges)
        self._req: Optional[TrafficRequest] = None

    # ------------------------------------------------------------------ parameters (DR)
    @property
    def _pm(self):
        return self.radio.params() if self.fast else self.eng

    def set_params(self, env_ids: EnvIds = None, **values):
        if self.fast:
            bad = set(values) - {"p_tx_dbm", "noise_dbm", "pl_const_db", "pl_exp", "shadow_sigma_db", "blockage_db"}
            if bad:
                raise KeyError(f"fast L2 engine does not model {sorted(bad)}")
        self._pm.set_params(env_ids, **values)

    def sample_params(self, env_ids: EnvIds = None, ranges: Optional[dict] = None):
        if ranges is None:
            ranges = vars(self._pm.ranges)
        for k, (lo, hi) in ranges.items():
            ids = self._pm._ids(env_ids)
            u = torch.rand(ids.numel(), device=self.dev)
            self.set_params(ids, **{k: lo + (hi - lo) * u})

    @property
    def gnb(self):
        return self._pm.gnb

    # ------------------------------------------------------------------ contract API
    def reset(self, env_ids: EnvIds = None):
        """Partial reset of every state tensor for env_ids (index tensor, bool mask, list or None = all)."""
        if not self.fast:
            self.eng.reset(env_ids)
            return
        ids = self._pm._ids(env_ids)
        if ids.numel() == 0:
            return
        e, n, R, S = self.eng, ids.numel(), self.R, self._S
        # in place, so CUDA-graph pointers stay valid
        e.cap[ids] = -1; e.cls[ids] = 0; e.det[ids] = False; e.hid[ids] = -1
        e.rem[ids] = 0.0; e.dlv[ids] = float("inf")
        e.f_nact[ids] = 0; e.f_snr[ids] = 0.0; e.f_own[ids] = 0
        e.bsr[ids] = 0.0; e.sr_t[ids] = -1; e.avg[ids] = 100.0; e.olla[ids] = 0.0
        e.wait[ids] = 0; e.hcnt[ids] = 0.0
        e.h[ids] = torch.randn(n, R, S, 2, device=self.dev) / math.sqrt(2)
        self.radio.reset(ids)
        self.t0[ids] = self._t_next
        self.last_cap_g[ids] = self._t_next          # "state at reset is known" (same choice as ref engine)
        self._prev_valid[ids] = False

    def submit(self, t: Optional[int], req: TrafficRequest):
        """Messages captured at the start of global step t. send [E,R] in {0,1,..}, tag [E,R] (-1 = none).

        The fast engine carries one label per env and step (hid[E]); tags within an env must agree
        (true for hazard ids). Call before step(t, ...).
        """
        self._req = req
        if self.fast:
            t = self._t_next if t is None else int(t)
            self._t_next = t
            tag = req.tag if req.tag is not None else torch.full_like(req.send, -1)
            self.eng.add_frames(t, req.send, tag >= 0, tag.max(-1).values.clamp(min=0), self._snr)

    def step(self, t: Optional[int], poses: torch.Tensor, cur_tag: Optional[torch.Tensor] = None,
             blocked_fn=None) -> dict:
        """Advance [t, t+1). poses [E,R,3] env-local at the END of the step. Returns a dict of [E,R]/[E] tensors."""
        req = self._req if self._req is not None else TrafficRequest(
            send=torch.zeros(self.E, self.R, dtype=torch.long, device=self.dev))
        self._req = None
        if not self.fast:
            o = self.eng.step(poses, req, blocked_fn=blocked_fn)
            out = dict(delivered=o.delivered, newest_cap=o.newest_cap, last_cap=o.last_cap, aoi_s=o.aoi_s,
                       queue_len=o.queue_len, queue_bytes=o.queue_bytes, sinr_db=o.sinr_db, serving=o.serving,
                       dropped=o.dropped, delay_s=o.delay_s, blocked=o.blocked)
            if cur_tag is not None:
                out["tag_delivered"] = (o.delivered_tag == cur_tag[:, None, None]).flatten(1).any(-1)
            return out
        t = self._t_next if t is None else int(t)
        # radio: SNR averaged (dB) over pose chunks interpolated between the previous and current poses
        prev = torch.where(self._prev_valid[:, None, None], self._prev, poses)
        C = self.cfg.pose_chunks
        snr = torch.zeros(self.E, self.R, device=self.dev)
        serving = None
        for c in range(C):
            p = prev + ((c + 0.5) / C) * (poses - prev)
            blk = blocked_fn(p) if blocked_fn is not None else None
            s_g, serving = self.radio.snr_db(p, blk).max(-1)
            snr = snr + s_g / C
        self._snr.copy_(snr)
        hid = cur_tag.clamp(min=0) if cur_tag is not None else torch.zeros(self.E, dtype=torch.long, device=self.dev)
        newest_g, det_env = self.eng.step(t, snr, hid)
        self._prev.copy_(poses)
        self._prev_valid.fill_(True)
        self.last_cap_g = torch.maximum(self.last_cap_g, newest_g)
        self._t_next = t + 1
        rel = lambda x: torch.where(x >= 0, x - self.t0[:, None], torch.full_like(x, -1))
        out = dict(
            delivered=newest_g >= 0,
            newest_cap=rel(newest_g),
            last_cap=rel(self.last_cap_g),
            aoi_s=(t + 1 - self.last_cap_g).float() * self.cfg.step_dt,
            queue_len=self.eng.queued(),
            queue_bytes=self.eng.rem.sum(-1),
            sinr_db=snr,
            serving=serving,
        )
        if cur_tag is not None:
            out["tag_delivered"] = det_env & (cur_tag >= 0)
        return out


def net_features(out: dict, step_dt: float, depth: int = 16) -> torch.Tensor:
    """Compact per-robot network observation [E,R,4]: AoI, SNR, queued frames, delivered-this-step."""
    return torch.stack([
        (out["aoi_s"] / step_dt).clamp(max=50) / 50,
        out["sinr_db"] / 40,
        out["queue_len"].float() / depth,
        out["delivered"].float(),
    ], -1)
