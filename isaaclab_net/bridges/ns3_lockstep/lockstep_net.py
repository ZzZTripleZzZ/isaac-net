"""Ns3Net: a NetBase implementation (drop-in for FleetEnv / train.py / evaluate) backed by ns-3 5G-LENA.

The frame table (cap, cls, det, hid, ... [E,R,F]), timeouts, overflow, stats and outputs are NetBase's
own code; only _transmit is replaced: the frames enqueued this step are sent to ns-3 as application
frames, ns-3 runs [t, t+1) and returns which frames completed and when.

Positions: FleetEnv does not pass poses to the network, so Ns3Net reads env.pos from the FleetEnv that
calls it (auto-bound on the first add_frames through the caller's frame, or explicitly via
bind_env(env)). With shadow="env" (default) the per-UE shadowing sent to ns-3 is derived from the
snr_db FleetEnv passes, so ns-3's large-scale SNR equals FleetEnv's Radio (same field, same path loss).
"""
import sys

import numpy as np
import torch

from ...core.proto.netsim import F, NetBase

from . import protocol as P
from .core import Ns3Lockstep, shadow_from_snr


class Ns3Net(NetBase):
    """ns-3 cannot rewind a subset of its envs here: reset() is full-only (reset(env_ids) raises), and every env
    shares one clock, so t is taken from env 0."""
    FIELDS = NetBase.FIELDS + ["fid"]
    INIT = {**NetBase.INIT, "fid": 0}
    DTYPE = {**NetBase.DTYPE, "fid": torch.long}

    def __init__(self, E, R, device, sizes, mode="procs", transport="tcp", shadow="env", run=1,
                 ns3_args=None, envs_per_proc=None, spawn=True, endpoints=None, host="127.0.0.1", **core_kw):
        self.core = None
        self.env = None
        self.shadow_mode = shadow
        self._pending = []
        self._need_reset = False
        self.last = None
        super().__init__(E, R, device, sizes)
        self.core = Ns3Lockstep(E, R, mode=mode, transport=transport, envs_per_proc=envs_per_proc, run=run,
                                ns3_args=ns3_args, spawn=spawn, endpoints=endpoints, host=host, **core_kw)
        self._need_reset = False      # the fresh processes already hold a built scenario

    # ------------------------------------------------------------------ binding
    def bind_env(self, env):
        self.env = env
        return self

    def add_frames(self, t, send, det, hid, snr_db):
        if self.env is None:
            caller = sys._getframe(1).f_locals.get("self")
            if caller is not None and hasattr(caller, "pos") and hasattr(caller, "radio"):
                self.env = caller
        super().add_frames(t, send, det, hid, snr_db)

    # ------------------------------------------------------------------ NetBase hooks
    def _reset_state(self, ids):
        if ids is not None:
            raise NotImplementedError("Ns3Net supports full resets only (reset() / reset(None))")
        self.fid = torch.zeros((self.E, self.R, F), dtype=torch.long, device=self.dev)
        self.next_fid = np.zeros((self.E, self.R), np.int64)
        self._pending = []
        self._need_reset = True       # rebuild ns-3 lazily at the next step, with the episode's poses

    def _on_arrival(self, t, e, r, i, draws):
        ec, rc = e.cpu().numpy(), r.cpu().numpy()
        f = self.next_fid[ec, rc].copy()
        self.next_fid[ec, rc] += 1
        self.fid[e, r, i] = torch.as_tensor(f, device=self.dev)
        by = self.sizes[self.cls[e, r, i] - 1].cpu().numpy()
        fr = np.zeros(len(ec), P.FRAME_IN)
        fr["env"], fr["ue"], fr["fid"], fr["bytes"] = ec, rc, f, np.round(by)
        self._pending.append(fr)

    def _poses(self):
        if self.env is not None:
            return self.env.pos.detach().float().cpu().numpy()
        return np.full((self.E, self.R, 2), np.nan, np.float32)

    def _transmit(self, t, snr_db):
        t = int(t[0]) if torch.is_tensor(t) else int(t)       # one shared clock (full resets only)
        pos = self._poses()
        sh = None
        if self.shadow_mode == "env" and np.isfinite(pos).all():
            sh = shadow_from_snr(pos, snr_db.detach().float().cpu().numpy()).astype(np.float32)
        if self._need_reset:
            self.core.reset(pos=pos, shadow=sh)
            self._need_reset = False
        frames = np.concatenate(self._pending) if self._pending else np.zeros(0, P.FRAME_IN)
        self._pending = []
        res = self.core.step(pos, frames, shadow=sh)
        self.last = res
        fin = np.full((self.E, self.R, F), np.inf, np.float32)
        d = res["done"]
        if len(d):
            cap = self.cap.cpu().numpy()
            fid = self.fid.cpu().numpy()
            for e, u, f, frac in zip(d["env"], d["ue"], d["fid"], d["frac"]):
                slot = np.nonzero((fid[e, u] == f) & (cap[e, u] >= 0))[0]
                if len(slot):
                    fin[e, u, slot[0]] = np.float32(t + frac)
        return torch.as_tensor(fin, device=self.dev)

    def close(self):
        if self.core is not None:
            self.core.close()
            self.core = None


def make_net_ns3(rung, E, R, device, sizes, params=None, **kw):
    """netfactory-compatible constructor: rung is ignored (the ns-3 bridge is its own rung, 'NS3')."""
    return Ns3Net(E, R, device, sizes, **kw)
