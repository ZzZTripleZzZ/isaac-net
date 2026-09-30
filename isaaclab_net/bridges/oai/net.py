"""NetBase and Isaac NetModule front ends of the OAI bridge (validation only, never in the training loop).

    stack = DockerOaiStack(n_ue=2)                      # or FakeStack(n_ue=2, delay_ms=8)
    bridge = OaiBridge(stack, step_dt=0.1, pacing="virtual", log_dir="runs/ep0")
    net = OaiNet(1, 2, "cpu", (4000.0, 30000.0), bridge)                  # NetBase: submit / step dicts
    mod = make_oai_netmodule(bridge, num_robots=2, config=NRConfig())      # isaac NetModule API

OaiNet keeps NetBase's own frame table (capture step, class, FIFO of F frames, the timeout and every output) and
replaces only the transmission: a frame enters the network the moment it is submitted (robot r of env e sends from
UE e*R + r), and step() returns the frames whose last datagram reached the sink during the control step, with
fin = capture step + delay / step_dt (delay in rfsim virtual time). Resets are logical, as in the ns-3 bridge's
single-process mode: rfsim cannot rewind, so the reset envs' bookkeeping restarts and their frames still in flight
are ignored when they arrive (they still use air time). Partial resets are therefore allowed.

Moving robots: with ``snr_ref_db`` set, the SNR the caller passes to step() moves each UE through the rfsim channel
path loss, ploss = clip(snr_ref_db - snr, 0, ploss_max_db), on both links, updated when it changes by at least
``ploss_step_db``. snr_ref_db is the SNR the gNB reports at zero path loss (about 51 dB in the lena_match profile,
docs/bridges-oai.md); the engine's SNR convention differs (snr_ref_prbs), so this is a monotone mapping, not a
calibrated link budget.
"""
from __future__ import annotations

import numpy as np
import torch

from ...core.proto.netsim import NetBase


class OaiNet(NetBase):
    FIELDS = NetBase.FIELDS + ["fid"]
    INIT = {**NetBase.INIT, "fid": -1}
    DTYPE = {**NetBase.DTYPE, "fid": torch.long}

    def __init__(self, E, R, device, sizes, bridge, fb=16, timeout=20, snr_ref_db=None, ploss_step_db=1.0,
                 ploss_max_db=60.0, seed=0):
        if E * R > bridge.stack.n_ue:
            raise ValueError(f"{E} envs x {R} robots need {E * R} UEs; the stack has {bridge.stack.n_ue}")
        self.bridge = bridge
        self.episode = np.zeros(E, np.int64)
        self.snr_ref_db, self.ploss_step, self.ploss_max = snr_ref_db, ploss_step_db, ploss_max_db
        self._ploss = np.full(E * R, np.nan)
        self.last = None
        super().__init__(E, R, device, sizes, seed=seed, fb=fb, timeout=timeout)

    def _reset_state(self, ids):
        ids = np.arange(self.E) if ids is None else ids.detach().cpu().numpy()
        self.episode[ids] += 1
        if self.bridge.t0 is None or len(ids) == self.E:
            self.bridge.reset()

    def _on_arrival(self, t, e, r, i, draws):
        ec, rc, ic = e.tolist(), r.tolist(), i.tolist()
        by = self.sizes[self.cls[e, r, i] - 1].round().long().tolist()
        frames = [(ee * self.R + rr, (ee, int(self.episode[ee]), rr, ii), nb)
                  for ee, rr, ii, nb in zip(ec, rc, ic, by)]
        fids = self.bridge.submit(frames)
        for ue, key, _ in frames:
            ee, _, rr, ii = key
            self.fid[ee, rr, ii] = fids[(ue, key)]

    def _move(self, snr_db):
        if self.snr_ref_db is None:
            return
        s = snr_db.detach().float().cpu().numpy().reshape(-1)
        pl = np.clip(self.snr_ref_db - s, 0.0, self.ploss_max)
        for k in np.nonzero(~(np.abs(pl - self._ploss) < self.ploss_step))[0]:
            self.bridge.set_pathloss(int(k), float(pl[k]))
            self._ploss[k] = pl[k]

    def _transmit(self, t, snr_db):
        self._move(snr_db)
        res = self.bridge.step()
        self.last = res
        fin = torch.full_like(self.rem, float("inf"))
        if res["done"]:
            cap, fid = self.cap.cpu(), self.fid.cpu()
            dt = self.bridge.dt
            hit = [], [], [], []
            for ue, (e, ep, r, _), delay, _, f in res["done"]:
                if ep != self.episode[e]:
                    continue                                   # frame of an episode that was reset
                slot = ((fid[e, r] == f) & (cap[e, r] >= 0)).nonzero()
                if slot.numel() == 0:
                    continue                                   # timed out on the NetBase side meanwhile
                i = int(slot[0, 0])
                for lst, v in zip(hit, (e, r, i, float(cap[e, r, i]) + delay / dt)):
                    lst.append(v)
            if hit[0]:
                d = self.dev
                idx = tuple(torch.tensor(x, device=d) for x in hit[:3])
                fin[idx] = torch.tensor(hit[3], dtype=fin.dtype, device=d)
                self.rem[idx] = 0.0
        return fin

    def close(self):
        self.bridge.close()


def make_oai_netmodule(bridge, num_robots, num_envs=1, device="cpu", config=None, snr_ref_db=None, **isaac_kw):
    """The Isaac NetModule (isaaclab_net.isaac.NetModule: reset / submit / step dict / obs) with its engine replaced
    by an OaiNet over `bridge`. The module's own radio (IsaacRadio) turns poses into SNR, which moves the UEs when
    snr_ref_db is set. config: the NRConfig whose application fields (message sizes, frame buffer, timeout, control
    step) the module and OaiNet use."""
    from ...core.config import NRConfig
    from ...isaac.net_module import NetModule
    cfg = config if config is not None else NRConfig()
    if abs(cfg.control_step_ms / 1000.0 - bridge.dt) > 1e-9:
        raise ValueError(f"config.control_step_ms {cfg.control_step_ms} != bridge step {bridge.dt * 1000} ms")
    # The module is built on the ORACLE level, whose engine is then replaced. ORACLE accepts only the prototype's
    # application constants, so it is built with them and the module takes the real ones afterwards.
    base = cfg.with_(control_step_ms=100.0, frame_buffer=16, timeout_steps=20)
    mod = NetModule("ORACLE", num_envs, num_robots, device, base, **isaac_kw)
    mod.config, mod.F, mod.step_dt = cfg, cfg.frame_buffer, cfg.control_step_ms / 1000.0
    mod.eng = OaiNet(num_envs, num_robots, device, tuple(cfg.msg_sizes), bridge, fb=cfg.frame_buffer,
                     timeout=cfg.timeout_steps, snr_ref_db=snr_ref_db)
    return mod
