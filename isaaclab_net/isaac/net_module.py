"""Isaac-side network module: one object per env batch, built on core.make_engine and NRConfig.

    net = NetModule("L2-legacy", E, R, "cuda", NRConfig(msg_sizes=(4000.0, 30000.0)), backend="triton")
    net.reset(env_ids)                                   # partial reset, any subset of envs
    net.submit(None, TrafficRequest(send[E,R], tag[E,R]))   # messages captured at the START of this control step
    out = net.step(None, poses_end[E,R,3], cur_tag[E])   # advance one control step -> dict

Levels and backends are those of make_engine(level, E, R, device, config, backend): L0, L0DR, L05 / L05Q (with
fitted params), L1 and L2-legacy on "reference" / "eager" / "graph" / "compile" (and "triton" for L1 and
L2-legacy), and L2 (the configurable NR engine) on "reference". The engine keeps a per-env episode clock, so t
arguments are accepted for compatibility with the demo signature and ignored: every env is always at its own
clock, and outputs are in that clock (an env that reset sees capture steps 0, 1, ...).

What this module adds to the engine
  radio      poses -> SNR through IsaacRadio: per-env parameters for network domain randomization, several gNBs,
             line-of-sight blockage, and the SNR averaged in dB over `pose_chunks` poses interpolated between the
             previous and the current end-of-step pose. radio="engine" passes the end-of-step poses to the
             engine's own radio instead (required for multi-cell configs, n_cells > 1).
  tag        TrafficRequest.tag [E,R] is an opaque per-message label (-1 = none), e.g. the id of the hazard a
             detection frame captured. step(..., cur_tag[E]) returns tag_delivered [E]: a message tagged with the
             env's current tag was delivered this step. The engine stores one label per env and step, so tags
             that are >= 0 must agree within an env in one submit (true for hazard ids).
  freshness  last_cap and aoi_s: the newest capture ever delivered this episode (the state at reset counts as
             known, capture 0) and the age of information at the end of the step.

Adapter fixes carried over from isaac/demo (2026-09-29)
  * first report: MessageHistory starts from seen_cap = -1, so the first delivered capture (env clock 0) of
    every episode reaches the receiver (it was dropped with seen_cap = 0).
  * no host syncs in the hot path: enqueue, step, tag handling and reset use device ops only on the fast
    backends (the reference backends keep their readable nonzero-based enqueue).
  * pose stamping: frames are captured at the start-of-step pose, and step() takes the END-of-step poses,
    which become the start of the next step's interpolation. Stamping the post-physics pose as capture step t
    made AoI and delays optimistic by one control step.
  * per-message tag, above.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import torch

from ..core.config import NRConfig
from ..core.engine import LEVELS, make_engine
from ..core.proto.netsim import env_index, fill_rows
from ..core.traffic import Requests
from .radio import RADIO_PARAMS, IsaacRadio, ParamRanges

FAST_BACKENDS = ("eager", "graph", "compile", "triton")
BACKENDS = ("reference",) + FAST_BACKENDS
L0_DEFAULT_PARAMS = {"mu": math.log(0.05), "sig": 0.5, "p": 0.0}   # L0: lognormal delay in control steps, loss


@dataclass
class TrafficRequest:
    """Messages each robot enqueues this control step.

    send: [E,R] long, 0 = nothing, c >= 1 = one message of traffic class c (size config.msg_sizes[c-1]).
    tag:  optional [E,R] long per-message label (-1 = none), e.g. the hazard id if the frame detects it.
    """
    send: torch.Tensor
    tag: Optional[torch.Tensor] = None


class NetModule:
    """Network in the loop of a batch of E envs x R robots (see the module docstring)."""

    def __init__(self, level="L2-legacy", num_envs: int = None, num_robots: int = None, device="cuda",
                 config: Optional[NRConfig] = None, backend: str = "reference", *, pose_chunks: int = 4,
                 gnb_pos: Optional[Sequence[Sequence[float]]] = None, radio: str = "isaac",
                 ranges: Optional[ParamRanges] = None, params: Optional[dict] = None, seed: Optional[int] = None,
                 inject: bool = False):
        if isinstance(level, NetConfig):              # NetModule(NetConfig(...), ranges): the demo's signature
            kw = level.to_kwargs()
            if isinstance(num_envs, ParamRanges):
                kw["ranges"] = num_envs
            self.__init__(**kw)
            return
        if level not in LEVELS:
            raise ValueError(f"unknown level {level!r}; one of {LEVELS}")
        if backend in ("ref", "orig"):
            backend = "reference"
        cfg = config if config is not None else NRConfig()
        self.level, self.backend, self.config = level, backend, cfg
        self.E, self.R, self.dev = int(num_envs), int(num_robots), torch.device(device)
        self.F = cfg.frame_buffer
        self.step_dt = cfg.control_step_ms / 1000.0
        self.pose_chunks = int(pose_chunks)
        if level == "L0" and params is None:
            params = L0_DEFAULT_PARAMS
        if radio == "isaac" and cfg.n_cells > 1:
            raise ValueError("a multi-cell config needs the engine's radio (cells, interference, handover): "
                             "pass radio='engine'")
        if seed is None:
            seed = int(torch.randint(0, 2 ** 62, ()).item())
        self.seed = seed
        self.eng = make_engine(level, self.E, self.R, self.dev, cfg, backend, params=params, seed=seed,
                               inject=inject)
        self.radio_mode = radio
        if radio == "isaac":
            if gnb_pos is None:
                gnb_pos = [(x, y, 0.0) for x, y in cfg.gnb_xy()]
            self.radio = IsaacRadio(self.E, self.dev, gnb_pos, ranges or ParamRanges.from_config(cfg),
                                    shadow_modes=cfg.shadow_modes, seed=seed + 1)
        elif radio == "engine":
            self.radio = None
        else:
            raise ValueError(f"radio must be 'isaac' or 'engine', got {radio!r}")
        L = torch.long
        self.last_cap = torch.zeros(self.E, self.R, dtype=L, device=self.dev)   # 0: the state at reset is known
        self._prev = torch.zeros(self.E, self.R, 3, device=self.dev)            # end-of-step pose of the last step
        self._prev_valid = torch.zeros(self.E, dtype=torch.bool, device=self.dev)
        self._zero_hid = torch.zeros(self.E, dtype=L, device=self.dev)
        self._submitted = False

    # ------------------------------------------------------------------ properties
    @property
    def clock(self) -> torch.Tensor:
        """[E] per-env episode clock (control steps since that env's reset)."""
        return self.eng.clock

    @property
    def gnb(self) -> torch.Tensor:
        return self.radio.gnb if self.radio is not None else None

    # ------------------------------------------------------------------ parameters (network DR)
    def set_params(self, env_ids=None, **values):
        """Per-env radio parameters (RADIO_PARAMS) of env_ids. Never touched by reset()."""
        if self.radio is None:
            raise RuntimeError("radio='engine': per-env radio parameters live in NRConfig")
        self.radio.set_params(env_ids, **values)

    def sample_params(self, env_ids=None, ranges: Optional[dict] = None):
        """Uniform draw of radio parameters within ranges ({name: (lo, hi)}, default: the module's ParamRanges)."""
        if self.radio is None:
            raise RuntimeError("radio='engine': per-env radio parameters live in NRConfig")
        self.radio.sample_params(env_ids, ranges)

    # ------------------------------------------------------------------ contract API
    def reset(self, env_ids=None):
        """Partial reset of env_ids (index tensor, bool mask, list or None = all): queues, MAC, fading, clock,
        the shadowing field and the freshness state. Radio parameters stay. Other envs are bitwise unaffected."""
        ids = env_index(env_ids, self.E, self.dev)
        if ids is not None and ids.numel() == 0:
            return
        self.eng.reset(ids)
        if self.radio is not None:
            self.radio.reset(ids)
        fill_rows(self.last_cap, ids, 0)
        fill_rows(self._prev_valid, ids, False)

    def submit(self, t, req: TrafficRequest):
        """Messages captured at the start of this control step (t is ignored: per-env clocks). Call before step."""
        tag = req.tag
        if tag is None:
            det, hid = None, None
        else:
            det = tag >= 0
            hid = tag.max(-1).values.clamp(min=0)
        self._submitted = True
        return self.eng.submit(None, Requests(req.send, det, hid))

    def _snr(self, poses, blocked_fn):
        """SNR [E,R] averaged in dB over pose chunks between the previous and the current end-of-step pose."""
        prev = torch.where(self._prev_valid[:, None, None], self._prev, poses)
        C = self.pose_chunks
        snr, serving, blk_last = 0.0, None, None
        for c in range(C):
            p = prev + ((c + 0.5) / C) * (poses - prev)
            blk = blocked_fn(p) if blocked_fn is not None else None
            s, serving = self.radio.snr_db(p, blk).max(-1)
            snr = snr + s / C
            if blk is not None:
                blk_last = blk.gather(-1, serving[..., None]).squeeze(-1)
        return snr, serving, blk_last

    def step(self, t, poses: torch.Tensor, cur_tag: Optional[torch.Tensor] = None, blocked_fn=None) -> dict:
        """Advance every env by one control step (t is ignored: per-env clocks).

        poses [E,R,3] (or [E,R,2]) env-local positions at the END of this control step.
        cur_tag [E] long: the env's current tag (-1 = none) for tag_delivered.
        blocked_fn(poses [E,R,3]) -> [E,R,G] bool line-of-sight blockage, evaluated per pose chunk.
        Returns a dict:
          delivered [E,R] bool      at least one message of the robot was delivered this step
          newest_cap [E,R] long     newest capture step delivered this step (-1 if none), env clock
          last_cap [E,R] long       newest capture step delivered this episode (0 = the state at reset)
          aoi_s [E,R] float         age of that information at the end of the step, in seconds
          queue_len, queue_bytes    FIFO state after the step
          sinr_db [E,R], serving [E,R], blocked [E,R]   radio of this step
          tag_delivered [E] bool    if cur_tag is given
          msg_delivered, timed_out [E,R,F] bool, cap, cls [E,R,F] long, delay_s [E,R,F] float (NaN if not
          delivered), t [E] long    per message slot as queued before the step, from the engine
        """
        if not self._submitted:
            self.submit(None, TrafficRequest(torch.zeros(self.E, self.R, dtype=torch.long, device=self.dev)))
        self._submitted = False
        if poses.shape[-1] == 2:
            poses = torch.cat([poses, torch.zeros_like(poses[..., :1])], -1)
        # det_env of the engine compares queued messages with the hid of the step; use the env's current tag
        self.eng._last_hid.copy_(cur_tag.clamp(min=0) if cur_tag is not None else self._zero_hid)
        blocked = None
        if self.radio is not None:
            snr, serving, blocked = self._snr(poses, blocked_fn)
            o = self.eng.step(None, snr)
        else:
            o = self.eng.step(None, poses)
            serving = o.get("serving_cell", torch.zeros(self.E, self.R, dtype=torch.long, device=self.dev))
        self._prev.copy_(poses)
        self._prev_valid.fill_(True)
        newest = o["newest"]
        self.last_cap = torch.maximum(self.last_cap, newest)
        out = dict(
            delivered=newest >= 0,
            newest_cap=newest,
            last_cap=self.last_cap.clone(),
            aoi_s=(o["t"][:, None] + 1 - self.last_cap).float() * self.step_dt,
            queue_len=o["queue_len"],
            queue_bytes=o["queue_bytes"],
            sinr_db=o["sinr_db"],
            serving=serving,
            blocked=blocked if blocked is not None else torch.zeros_like(newest, dtype=torch.bool),
            msg_delivered=o["delivered"],
            timed_out=o["timed_out"],
            cap=o["cap"],
            cls=o["cls"],
            delay_s=o["delay"] * self.step_dt,
            t=o["t"],
        )
        if cur_tag is not None:
            out["tag_delivered"] = o["det_env"] & (cur_tag >= 0)
        return out

    def queued(self) -> torch.Tensor:
        return self.eng.queued()


# --------------------------------------------------------------------------------------------------------------
# Compatibility: the demo's NetConfig
# --------------------------------------------------------------------------------------------------------------
@dataclass
class NetConfig:
    """Thin compatibility alias for the demo's configuration. New code passes level, sizes and NRConfig to
    NetModule directly. rung "L2" of the demo is the slot-level NetSlot, which is level "L2-legacy" here;
    backend "ref" is "reference". to_kwargs() maps the fields onto NetModule(level, E, R, device, config, ...)."""
    num_envs: int
    num_robots: int
    device: str = "cuda"
    rung: str = "L1"
    backend: str = "ref"
    step_dt: float = 0.1
    slot_dt: float = 0.0025
    pose_chunks: int = 4
    frame_depth: int = 16
    msg_sizes: Sequence[float] = (1500.0, 12000.0)
    timeout_steps: int = 20
    gnb_pos: Sequence[Sequence[float]] = ((0.0, 0.0, 6.0),)
    config: Optional[NRConfig] = field(default=None)

    def to_kwargs(self) -> dict:
        if abs(self.slot_dt - 0.0025) > 1e-12:
            raise ValueError("slot_dt is set by the engine (NRConfig numerology and TDD pattern)")
        level = {"L2": "L2-legacy"}.get(self.rung, self.rung)
        cfg = (self.config or NRConfig()).with_(msg_sizes=tuple(float(s) for s in self.msg_sizes),
                                                frame_buffer=self.frame_depth, timeout_steps=self.timeout_steps,
                                                control_step_ms=self.step_dt * 1000.0)
        return dict(level=level, num_envs=self.num_envs, num_robots=self.num_robots, device=self.device, config=cfg,
                    backend=self.backend, pose_chunks=self.pose_chunks, gnb_pos=self.gnb_pos)


# --------------------------------------------------------------------------------------------------------------
# What the receiver sees
# --------------------------------------------------------------------------------------------------------------
class MessageHistory:
    """Ring buffer of per-robot payloads indexed by env-clock capture step (the receiver's delayed view).

    push(t_env, data) stores data [E,R,D] captured at each env's own clock (the start-of-step pose). After
    net.step, update(out["newest_cap"]) replaces the receiver's copy for robots whose newest delivered capture is
    fresher than what it holds; others keep the last payload (hold-last on loss). history_len must exceed
    config.timeout_steps so that every deliverable capture is still stored.
    """

    def __init__(self, num_envs: int, num_robots: int, dim: int, history_len: int, device="cuda"):
        self.H, self.dev = history_len, torch.device(device)
        self.hist = torch.zeros(history_len, num_envs, num_robots, dim, device=self.dev)
        self.seen = torch.zeros(num_envs, num_robots, dim, device=self.dev)
        # -1 = "only the reset state". With 0 the strict test newest_cap > seen_cap dropped the first
        # delivered capture (env clock 0) of every episode (isaac/demo fix, 2026-09-29).
        self.seen_cap = torch.full((num_envs, num_robots), -1, dtype=torch.long, device=self.dev)
        self._arE = torch.arange(num_envs, device=self.dev)

    def push(self, t_env: torch.Tensor, data: torch.Tensor):
        self.hist[t_env % self.H, self._arE] = data

    def update(self, newest_cap: torch.Tensor) -> torch.Tensor:
        fresher = newest_cap > self.seen_cap
        idx = newest_cap.clamp(min=0) % self.H
        R = newest_cap.shape[1]
        got = self.hist[idx, self._arE[:, None], torch.arange(R, device=self.dev)[None, :]]
        self.seen = torch.where(fresher[..., None], got, self.seen)
        self.seen_cap = torch.where(fresher, newest_cap, self.seen_cap)
        return self.seen

    def reset(self, env_ids: torch.Tensor, init: torch.Tensor):
        """init [n,R,D]: the receiver is told the true state at reset (last_cap = 0 in NetModule)."""
        self.hist[:, env_ids] = init[None]
        self.seen[env_ids] = init
        self.seen_cap[env_ids] = -1


def net_features(out: dict, step_dt: float, depth: int = 16) -> torch.Tensor:
    """Compact per-robot network observation [E,R,4]: AoI, SNR, queued frames, delivered-this-step."""
    return torch.stack([
        (out["aoi_s"] / step_dt).clamp(max=50) / 50,
        out["sinr_db"] / 40,
        out["queue_len"].float() / depth,
        out["delivered"].float(),
    ], -1)


__all__ = ["NetModule", "NetConfig", "TrafficRequest", "ParamRanges", "MessageHistory", "net_features",
           "FAST_BACKENDS", "BACKENDS", "RADIO_PARAMS", "L0_DEFAULT_PARAMS"]
