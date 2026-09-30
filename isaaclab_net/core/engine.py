"""One engine API for every fidelity level: make_engine(level, E, R, device, config, backend).

Levels
  "L0", "L0DR", "L05", "L05Q", "L1"   prototype levels (proto/netsim.py and proto/netsim_fast.py)
  "L2"                                configurable NR engine (nr_engine.NRNet behind NREngine): NRConfig
                                      numerology / TDD / 3GPP MCS-TBS-BLER / multi-HARQ / optional downlink
  "L2-legacy"                         the prototype slot-level NetSlot, frozen: the earlier prototype experiments ran
                                      with it and its graph backend is bitwise equal to its reference. With a
                                      multi-cell or thermal-noise config it runs NetSlotMC (proto/netsim_mc.py).
  "TR", "GE", "QA", "NN"              surrogates fitted from L2 / L2-legacy rollouts (levels/surrogates.py): trace
                                      replay, Markov-modulated delay/loss, analytic queue, learned surrogate.
                                      params = a fit file path or dict from isaaclab_net.tools.fit_levels.
  "ORACLE", "NOCOMM"                  value-of-information bounds (levels/bounds.py): instant lossless delivery,
                                      and nothing delivered. Not network models: they bracket what any level
                                      can give a task.

Backends
  "reference"                          the readable eager engine of the level (every level; the only NR backend)
  "eager", "graph", "compile", "triton" prototype fast backends (graph = bitwise equal to reference; triton for
                                      L1 / L2-legacy only). A graph/triton backend for the NR engine is a follow-up.
                                      The surrogate and bound levels have "reference" (= "eager") and "graph".

Every engine returned here has the contract API of ARCHITECTURE.md:
  reset(env_ids=None)                 partial reset: None, index tensor, list or bool mask [E]
  submit(t, requests, snr_db=None)    enqueue new messages (Requests or a send tensor [E,R]); returns accepted [E,R]
  step(t, poses_or_snr) -> dict       advance [t, t+1): delivered / timed_out / cap / cls / delay [E,R,F],
                                      newest [E,R], det_env [E], queue_len / queue_bytes / sinr_db [E,R], t [E]
  clock [E]                           per-env episode clock; pass t=None to use it (recommended)
plus the legacy calls add_frames(t, send, det, hid, snr_db) and step(t, snr_db, hid) -> (newest, det_env).
"""
from __future__ import annotations

import math

import torch

from .channels import install_per_robot_fading, rho_per_ms_from_speed
from .config import NRConfig
from .levels import BOUND_LEVELS, SURROGATE_LEVELS, make_level
from .nr_engine import NRNet
from .proto import netsim as _proto
from .radio import RadioMC
from .traffic import Requests

PROTO_LEVELS = ("L0", "L0DR", "L05", "L05Q", "L1")
SIM_LEVELS = PROTO_LEVELS + ("L2", "L2-legacy")
LEVELS = SIM_LEVELS + SURROGATE_LEVELS + BOUND_LEVELS
FAST_BACKENDS = ("eager", "graph", "compile", "triton")
BACKENDS = ("reference",) + FAST_BACKENDS


def _check_proto_config(level, cfg: NRConfig):
    """The prototype levels are compiled around fixed constants; refuse a config that asks for something else."""
    want = {"frame_buffer": _proto.F, "timeout_steps": _proto.TIMEOUT, "control_step_ms": 100.0}
    bad = {k: (getattr(cfg, k), v) for k, v in want.items() if getattr(cfg, k) != v}
    if bad:
        raise ValueError(f"level {level} has fixed {', '.join(f'{k}={v[1]}' for k, v in bad.items())}; "
                         f"the config asks for {', '.join(f'{k}={v[0]}' for k, v in bad.items())}. "
                         "Use level 'L2' for a configurable engine.")
    if level != "L2-legacy" and cfg.n_cells != 1:
        raise ValueError(f"level {level} is single-cell; multi-cell runs on 'L2' or 'L2-legacy'")
    if level in ("L1", "QA") and not cfg.is_legacy_cell():
        raise ValueError(f"level {level} has the fixed legacy radio (one gNB at the origin, -90 dBm noise floor); "
                         "use the default cell settings, or level 'L2' / 'L2-legacy'")


def _level_params(level, cfg: NRConfig, params):
    """Parameters of the prototype delay levels and L1 from the config when the caller passes none."""
    if params is not None:
        return params
    if level == "L0":
        return {"mu": math.log(cfg.l0_delay_median_steps), "sig": cfg.l0_delay_log_sigma, "p": cfg.l0_loss}
    if level == "L0DR":
        return {"median_steps": cfg.dr_delay_median_steps, "log_sigma": cfg.dr_delay_log_sigma, "loss": cfg.dr_loss}
    if level == "L1":
        return {"eta": cfg.l1_eta}
    return None


def make_engine(level, E, R, device="cpu", config: NRConfig | None = None, backend="reference", *, sizes=None,
                params=None, seed=None, inject=False, strict=False):
    """Build the network engine of fidelity `level` for E envs x R robots.

    config: NRConfig shared by every module (default NRConfig()); the prototype levels read only its application
      fields (frame_buffer, timeout_steps, control_step_ms, msg_sizes) and check that they match their constants.
    sizes: override of config.msg_sizes. params: fitted parameters of L0 ({"mu", "sig", "p"}) and L05 / L05Q
      ({"q", "pdrop"}); for TR / GE / QA / NN a fit file path, a fit-file dict or the level's own dict (see
      levels.load_level_params). seed: engine generator (reset draws); stepping draws come from the global
      torch RNG.
    inject: fast backends only, take the per-slot random draws from set_noise(...) (equivalence tests).
    strict: raise if the config sets fields away from their defaults that this level ignores
      (config.unused_fields(level)); by default they are ignored silently.
    L0, L0DR and L1 without params take them from the config (l0_*, dr_*, l1_eta; defaults = earlier behavior).
    """
    if level not in LEVELS:
        raise ValueError(f"unknown level {level!r}; one of {LEVELS}")
    if backend in ("orig", "ref"):
        backend = "reference"
    if backend not in BACKENDS:
        raise ValueError(f"unknown backend {backend!r}; one of {BACKENDS}")
    cfg = config if config is not None else NRConfig()
    if sizes is not None:
        cfg = cfg.with_(msg_sizes=tuple(float(s) for s in sizes))
    sizes = tuple(cfg.msg_sizes)
    if strict and cfg.unused_fields(level):
        raise ValueError(f"level {level} ignores these config fields: {', '.join(cfg.unused_fields(level))} "
                         "(see NRConfig.unused_fields and docs/configurability.md)")
    params = _level_params(level, cfg, params)
    if level == "L2":
        if backend != "reference":
            raise NotImplementedError(f"backend {backend!r} is not available for the NR engine yet (follow-up); use "
                                      "backend='reference', or level='L2-legacy' for the graph / triton backends")
        return NREngine(E, R, device, cfg, seed=seed)
    _check_proto_config(level, cfg)
    if level in SURROGATE_LEVELS + BOUND_LEVELS:
        if backend not in ("reference", "eager", "graph"):
            raise NotImplementedError(f"level {level} has the backends 'reference' and 'graph', not {backend!r}")
        net = make_level(level, E, R, device, sizes, params, backend=backend, inject=inject, seed=seed)
        net.config = cfg
        return net
    if level == "L2-legacy" and not cfg.is_legacy_cell():
        if backend != "reference":
            raise NotImplementedError("the multi-cell legacy engine (NetSlotMC) has only the reference backend")
        from .proto.netsim_mc import NetSlotMC
        net = NetSlotMC(E, R, device, sizes, cfg, seed=seed)
    else:
        rung = "L2" if level == "L2-legacy" else level
        if backend == "reference":
            net = _proto.make_net(rung, E, R, device, sizes, params, seed=seed)
        else:
            from .proto.netsim_fast import NetFast
            net = NetFast(rung, E, R, device, sizes, params=params, backend=backend, inject=inject, seed=seed)
    net.level, net.config = level, cfg
    return net


class NREngine:
    """Contract API over NRNet (level "L2").

    NRNet simulates continuous physical time with one global control-step clock. NREngine keeps that clock in
    `self.T` and gives every env an episode clock `clock[e] = T - epoch[e]` that reset(env_ids) zeroes, so its
    outputs use the same per-env clock as the prototype levels (capture steps, newest, t). HARQ, SR and CQI
    timers stay in global slots; reset(env_ids) clears them for those envs, which makes the reset exact.

    t arguments: None uses the engine clock (recommended). An int or an [E] tensor must equal the engine clock;
    the NR engine cannot jump in time. After a partial reset an int can no longer match every env: pass None.
    """

    level = "L2"

    def __init__(self, E, R, device, cfg: NRConfig, seed=None):
        self.E, self.R, self.dev, self.config = E, R, torch.device(device), cfg
        if seed is None:
            seed = int(torch.randint(0, 2 ** 62, ()).item())
        self.gen = torch.Generator(device=self.dev)
        self.gen.manual_seed(seed)
        self.net = NRNet(E, R, self.dev, cfg.msg_sizes, cfg, generator=self.gen)
        self.per_robot_doppler = cfg.fading_doppler == "per_robot"
        if self.per_robot_doppler:
            install_per_robot_fading(self.net)
        self.F = cfg.frame_buffer
        self.radio = None
        self.T = 0
        self.epoch = torch.zeros(E, dtype=torch.long, device=self.dev)
        self._uniform_epoch = 0                 # host copy of the epoch while no partial reset happened
        self._last_snr = torch.zeros(E, R, device=self.dev)
        self._last_hid = torch.zeros(E, dtype=torch.long, device=self.dev)

    # ------------------------------------------------------------------ passthroughs
    def __getattr__(self, name):          # ul, dl, stats, counters(), cap, ... of the wrapped NRNet
        if name == "net":
            raise AttributeError(name)
        return getattr(self.net, name)

    @property
    def log_stats(self):
        return self.net.log_stats

    @log_stats.setter
    def log_stats(self, v):
        self.net.log_stats = v

    @property
    def clock(self):
        return self.T - self.epoch

    def queued(self):
        return self.net.queued()

    def collect(self):
        return self.net.collect()

    def clear_stats(self):
        self.net.clear_stats()

    def attach_radio(self, radio):
        """Use an external RadioMC (C = 1) for step(t, poses); reset(env_ids) then resets its rows too."""
        self.radio = radio

    def set_sinr_hook(self, fn, direction="ul"):
        """fn(g, dir, won [E,R,S], n_prb [E,R], sinr [E,R,S]) -> sinr [E,R,S] is called between allocation and
        decoding in every data slot of that direction (won: RBGs of the robots that transmit). With several cells
        the engine's own inter-cell interference runs first and fn gets its result."""
        self.net.set_sinr_hook(fn, direction)

    # ------------------------------------------------------------------ time
    def _now(self, t):
        if t is None:
            return self.T
        if isinstance(t, torch.Tensor):
            if t.dim() == 0:
                t = int(t)
            else:
                if not torch.equal(t.to(self.dev, torch.long), self.clock):
                    raise ValueError("t must equal the engine clock net.clock (the NR engine cannot jump in time); "
                                     "pass t=None")
                return self.T
        t = int(t)
        if self._uniform_epoch is None:
            raise ValueError("after a partial reset the envs have different clocks: pass t=None")
        if t + self._uniform_epoch != self.T:
            raise ValueError(f"t={t} but the engine clock is {self.T - self._uniform_epoch}: the NR engine cannot "
                             "jump in time; pass t=None")
        return self.T

    def _rel(self, x):
        """Global capture steps -> env clock (-1 stays -1)."""
        shape = (-1,) + (1,) * (x.dim() - 1)
        return torch.where(x >= 0, x - self.epoch.view(shape), torch.full_like(x, -1))

    # ------------------------------------------------------------------ contract API
    def reset(self, env_ids=None):
        """Re-initialize env_ids (None = all): queues, HARQ, SR/BSR, OLLA, PF, CSI, fading, radio and clock."""
        ids = _proto.env_index(env_ids, self.E, self.dev)
        if ids is None:
            self.net.reset(None)
            self.T = 0
            self.epoch.zero_()
            self._uniform_epoch = 0
            self._last_snr.zero_()
            self._last_hid.zero_()
            if self.radio is not None:
                self.radio.reset(None)
            return
        if ids.numel() == 0:
            return
        self.net.reset(ids)
        self.epoch.index_fill_(0, ids, self.T)
        self._uniform_epoch = None
        self._last_snr.index_fill_(0, ids, 0.0)
        self._last_hid.index_fill_(0, ids, 0)
        if self.radio is not None:
            self.radio.reset(ids)

    def submit(self, t, requests, snr_db=None):
        """Enqueue new messages at capture time t (None = engine clock). requests: Requests or send [E,R].
        snr_db [E,R] is recorded as a frame feature (default: the SNR of the previous step). Returns accepted."""
        T = self._now(t)
        req = requests if isinstance(requests, Requests) else Requests(send=requests)
        det = req.det if req.det is not None else torch.zeros_like(req.send, dtype=torch.bool)
        hid = req.hid if req.hid is not None else torch.zeros(self.E, dtype=torch.long, device=self.dev)
        self._last_hid = hid
        return self.net.add_frames(T, req.send, det, hid, self._last_snr if snr_db is None else snr_db)

    def add_frames(self, t, send, det, hid, snr_db):
        """Legacy wrapper (NetSlot API)."""
        self.submit(t, Requests(send, det, hid), snr_db)

    def add_dl_frames(self, t, nbytes, cls=None):
        """Downlink messages of nbytes [E,R] (0 = none) at t (needs config.dl)."""
        self.net.add_dl_frames(self._now(t), nbytes, cls)

    def _ul_input(self, x, vel=None):
        """SNR [E,R] dB (full UE power over snr_ref_prbs PRBs), or poses [E,R,2|3] -> (pathgain or None, snr)."""
        if x.dim() == 3:
            return self._pathgain(x, vel)[..., 0]
        return None

    def _pathgain(self, pos, vel=None):
        if self.radio is None:
            self.radio = RadioMC(self.config, self.E, self.dev, generator=self.gen, R=self.R)
        pg = self.radio.pathgain_db(pos)
        if self.per_robot_doppler:
            speed = self.radio.observe_motion(pos, vel)
            self.net.fading_rho_ms = rho_per_ms_from_speed(speed, self.config.carrier_ghz)
        return pg

    def step(self, t, x=None, cur_hid=None, *, snr_db=None, dl_snr_db=None, pathgain_db=None, vel=None):
        """Advance [t, t+1). x: SNR [E,R] in dB, or poses [E,R,2|3] (through the engine's radio); snr_db=
        takes a per-subband SNR [E,R,S]. With several cells (config.n_cells > 1) x must be poses, or pass
        pathgain_db= [E,R,C] (large-scale gain of every robot-cell link, dB). vel [E,R,2|3] (m/s): robot velocities
        for config.fading_doppler="per_robot" (default: from consecutive poses). Legacy form step(t, x, cur_hid)
        returns (newest, det_env). Without cur_hid it returns the dict of the module docstring plus, for this
        engine:
          dropped [E,R,F] bool   lost under RLC UM (harq_fail="drop"), or on a handover with ho_rlc="flush", and
                                 resolved this step
          serving_cell [E,R]     serving cell (0 with one cell)
          dl_newest, dl_queue_len [E,R]  when config.dl
        With several cells sinr_db is the serving-link SINR against the gNB's latest N+I estimate.
        """
        T = self._now(t)
        legacy = cur_hid is not None
        hid = cur_hid if legacy else self._last_hid
        if self.net.C > 1:
            if pathgain_db is None:
                if x is None or x.dim() != 3:
                    raise ValueError("several cells: pass poses [E,R,2|3] or pathgain_db=[E,R,C]")
                pathgain_db = self._pathgain(x, vel)
            out = self.net.step_cells(T, pathgain_db, hid, full=True)
            snr = self.net.serving_sinr_db()
        elif pathgain_db is not None:
            raise ValueError("pathgain_db= needs config.n_cells > 1; use x (poses or SNR) with one cell")
        elif snr_db is None:
            pg = self._ul_input(x, vel)
            if pg is not None:
                out = self.net.step_rx(T, pg, hid, full=True)
                c = self.config
                snr = pg + c.ue_tx_dbm - c.subband_noise_dbm
            else:
                out = self.net.step(T, x, hid, dl_snr_db, full=True)
                snr = x
        else:
            out = self.net.step(T, snr_db, hid, dl_snr_db, full=True)
            snr = snr_db if snr_db.dim() == 2 else snr_db.mean(-1)
        self._last_snr = snr
        newest = self._rel(out["newest"])
        self.T = T + 1
        if legacy:
            return newest, out["det_env"]
        out["newest"] = newest
        out["cap"] = self._rel(out["cap"])
        if "dl_newest" in out:
            out["dl_newest"] = self._rel(out["dl_newest"])
        out["sinr_db"] = snr
        if "serving_cell" not in out:
            out["serving_cell"] = torch.zeros(self.E, self.R, dtype=torch.long, device=self.dev)
        out["t"] = (T - self.epoch).clone()
        return out
