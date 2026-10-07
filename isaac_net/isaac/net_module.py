"""Isaac-side network module: one object per env batch, built on core.make_engine and NRConfig.

    net = NetModule("L2-legacy", E, R, "cuda", NRConfig(msg_sizes=(4000.0, 30000.0)), backend="triton",
                    isaac=IsaacNetCfg(obs_features=("aoi", "sinr", "delay_history")))
    net.reset(env_ids)                                   # partial reset, any subset of envs
    net.submit(None, TrafficRequest(send[E,R], tag[E,R]))   # messages captured at the START of this control step
    out = net.step(None, poses_end[E,R,3], cur_tag[E])   # advance one control step -> dict
    x = net.obs()                                        # [E,R,net.obs_dim] selected, normalized features

One configuration: the NRConfig is the object make_engine takes (message sizes, frame buffer, timeout, control step,
radio and cell layout, MAC and PHY, the L0 / L0DR delay parameters); level and backend are the make_engine
arguments; IsaacNetCfg (isaac/config.py) holds only the Isaac-specific settings (pose source, multi-rate, blockage,
domain randomization, observation selection). The keyword arguments pose_chunks, gnb_pos and radio are accepted as
shortcuts for the IsaacNetCfg fields of the same name.

Levels and backends are those of make_engine(level, E, R, device, config, backend): L0, L0DR, L05 / L05Q (with
fitted params), L1 and L2-legacy on "reference" / "eager" / "graph" / "compile" (and "triton" for L1 and
L2-legacy), L2 (the configurable NR engine) on "reference", the fitted surrogates TR / GE / QA / NN (params = a
fit file) and the ORACLE / NOCOMM bounds. The engine keeps a per-env episode clock, so t
arguments are accepted for compatibility with the demo signature and ignored: every env is always at its own
clock, and outputs are in that clock (an env that reset sees capture steps 0, 1, ...).

What this module adds to the engine
  radio      poses -> SNR through IsaacRadio: per-env parameters for network domain randomization, several gNBs,
             line-of-sight blockage, and the SNR averaged in dB over `pose_chunks` poses interpolated between the
             previous and the current end-of-step pose. radio="engine" passes the end-of-step poses to the
             engine's own radio instead (required for multi-cell configs, n_cells > 1). With the Isaac radio the
             engine only sees the SNR, so the levels with the fixed legacy radio (L1, L2-legacy, QA) get the
             legacy cell fields and stay on their fast backends whatever radio fields the NRConfig sets; the Isaac
             radio uses the NRConfig's values.
  DR         IsaacNetCfg.dr_ranges, redrawn at reset or at an interval (dr_mode); dr_support() says which keys the
             level honors, and unhonored keys warn (or raise with dr_strict).
  tag        TrafficRequest.tag [E,R] is an opaque per-message label (-1 = none), e.g. the id of the hazard a
             detection frame captured. step(..., cur_tag[E]) returns tag_delivered [E]: a message tagged with the
             env's current tag was delivered this step. The engine stores one label per env and step, so tags
             that are >= 0 must agree within an env in one submit (true for hazard ids).
  freshness  last_cap and aoi_s: the newest capture ever delivered this episode (the state at reset counts as
             known, capture 0) and the age of information at the end of the step.
  obs        NetObs (isaac/obs.py): the selected observation features, updated by step and zeroed by reset.

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

import warnings
from dataclasses import dataclass, fields, replace
from typing import Optional, Sequence

import torch

from ..core import checkpoint as _ckpt
from ..core.checkpoint import StateDictMixin
from ..core.config import NRConfig
from ..core.engine import LEVELS, make_engine
from ..core.proto.netsim import env_index, fill_rows
from ..core.traffic import Requests
from .config import IsaacNetCfg, dr_support
from .obs import NetObs
from .radio import RADIO_PARAMS, IsaacRadio, ParamRanges, segment_sphere_blocked

FAST_BACKENDS = ("eager", "graph", "compile", "triton")
BACKENDS = ("reference",) + FAST_BACKENDS
# levels compiled around the fixed legacy radio: with the Isaac radio they get these NRConfig fields at default
_LEGACY_RADIO_LEVELS = ("L1", "L2-legacy", "QA")
_PASSTHROUGH = ("access_state", "access_sleep_frac", "rach_attempts", "rlf")
_LEGACY_CELL_FIELDS = ("cell_layout", "cell_positions_m", "noise_model", "ni_fixed_dbm", "ue_tx_dbm", "pl_const_db",
                       "pathloss_exp", "shadow_sigma_db", "shadow_modes")


@dataclass
class TrafficRequest:
    """Messages each robot enqueues this control step.

    send: [E,R] long, 0 = nothing, c >= 1 = one message of traffic class c (size config.msg_sizes[c-1]).
    tag:  optional [E,R] long per-message label (-1 = none), e.g. the hazard id if the frame detects it.
    """
    send: torch.Tensor
    tag: Optional[torch.Tensor] = None


class NetModule(StateDictMixin):
    """Network in the loop of a batch of E envs x R robots (see the module docstring).

    Checkpoints (docs/checkpoint.md): state_dict() / load_state_dict(sd) hold the engine's state and the module's own
    (the Isaac radio with the per-env parameters drawn by domain randomization and its generator, last_cap, the
    previous end-of-step poses, the DR timers, the observation features and their delay history); save(path) and
    load(path) write and read a file, optionally with the multi-rate state of the env that hosts the module
    (host=: held outputs, pending messages, decimation tick; see find_network)."""

    def __init__(self, level="L2-legacy", num_envs: int = None, num_robots: int = None, device="cuda",
                 config: Optional[NRConfig] = None, backend: str = "reference", *, isaac: Optional[IsaacNetCfg] = None,
                 ranges: Optional[ParamRanges] = None, params: Optional[dict] = None, seed: Optional[int] = None,
                 inject: bool = False, strict: bool = False, **isaac_overrides):
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
        isc = isaac if isaac is not None else IsaacNetCfg()
        if isaac_overrides:
            known = {f.name for f in fields(IsaacNetCfg)}
            bad = sorted(set(isaac_overrides) - known)
            if bad:
                raise TypeError(f"unknown NetModule arguments {bad}; IsaacNetCfg fields are {sorted(known)}")
            isc = replace(isc, **isaac_overrides)
        self.level, self.backend, self.isaac = level, backend, isc
        self.E, self.R, self.dev = int(num_envs), int(num_robots), torch.device(device)
        # the delay keys of dr_ranges are NRConfig fields (dr_* for L0DR, l0_* for L0)
        cfg = isc.engine_config(cfg, level) if isc.dr_mode != "off" else cfg
        self.config = cfg
        self.F = cfg.frame_buffer
        self.step_dt = cfg.control_step_ms / 1000.0
        self.pose_chunks = int(isc.pose_chunks)
        if isc.radio == "isaac" and cfg.n_cells > 1:
            raise ValueError("a multi-cell config needs the engine's radio (cells, interference, handover): "
                             "pass radio='engine'")
        ignored = cfg.unused_fields(level)
        if strict and ignored:
            raise ValueError(f"level {level} ignores these NRConfig fields: {', '.join(ignored)}")
        # which randomization keys this level honors
        self.dr_support = dr_support(level, cfg, isc, keys=list(isc.dr_ranges))
        off = {k: why for k, (ok, why) in self.dr_support.items() if not ok}
        if off and isc.dr_mode != "off":
            msg = f"level {level} does not honor these dr_ranges keys: " + "; ".join(f"{k} ({w})" for k, w in off.items())
            if isc.dr_strict:
                raise ValueError(msg)
            warnings.warn(msg, stacklevel=2)
        self._dr_radio = {k: v for k, v in isc.radio_ranges().items() if self.dr_support.get(k, (False,))[0]}
        if seed is None:
            seed = int(torch.randint(0, 2 ** 62, ()).item())
        self.seed = seed
        eng_cfg = cfg
        if isc.radio == "isaac" and level in _LEGACY_RADIO_LEVELS and not cfg.is_legacy_cell():
            ref = NRConfig()
            eng_cfg = cfg.with_(**{f: getattr(ref, f) for f in _LEGACY_CELL_FIELDS})
        self.eng = make_engine(level, self.E, self.R, self.dev, eng_cfg, backend, params=params, seed=seed,
                               inject=inject)
        self.radio_mode = isc.radio
        if isc.radio == "isaac":
            self.radio = IsaacRadio(self.E, self.dev, isc.gnb_positions(cfg),
                                    ranges or ParamRanges.from_config(cfg, blockage_db=isc.blockage_db),
                                    shadow_modes=cfg.shadow_modes, seed=seed + 1)
            self.G = self.radio.G
        else:
            self.radio = None
            self.G = cfg.n_cells
        L = torch.long
        self.last_cap = torch.zeros(self.E, self.R, dtype=L, device=self.dev)   # 0: the state at reset is known
        self._prev = torch.zeros(self.E, self.R, 3, device=self.dev)            # end-of-step pose of the last step
        self._prev_valid = torch.zeros(self.E, dtype=torch.bool, device=self.dev)
        self._zero_hid = torch.zeros(self.E, dtype=L, device=self.dev)
        self._submitted = False
        self._eye = torch.eye(self.R, dtype=torch.bool, device=self.dev)[None].expand(self.E, -1, -1)
        self._dr_timer = torch.zeros(self.E, dtype=L, device=self.dev)
        self.obs_features = NetObs(isc.obs_features, self.E, self.R, cfg, n_cells=self.G, history=isc.obs_history,
                                   time_scale_s=isc.time_scale_s(cfg), device=self.dev)
        self.obs_dim = self.obs_features.dim
        if self._dr_radio and isc.dr_mode != "off":
            self.randomize(None)                     # every env starts from its own draw

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
        """Per-env radio parameters (RADIO_PARAMS, gnb_offset_m) of env_ids. Never touched by reset()."""
        if self.radio is None:
            raise RuntimeError("radio='engine': per-env radio parameters live in NRConfig")
        self.radio.set_params(env_ids, **values)

    def sample_params(self, env_ids=None, ranges: Optional[dict] = None):
        """Uniform draw of radio parameters within ranges ({name: (lo, hi)}, default: the module's ParamRanges)."""
        if self.radio is None:
            raise RuntimeError("radio='engine': per-env radio parameters live in NRConfig")
        self.radio.sample_params(env_ids, ranges)

    def randomize(self, env_ids=None):
        """Redraw the honored radio keys of IsaacNetCfg.dr_ranges for env_ids (the delay keys are drawn by the
        engine itself at every reset). Called by reset() with dr_mode="reset" and by step() with "interval"."""
        if self._dr_radio and self.radio is not None:
            self.radio.sample_params(env_ids, self._dr_radio)

    def _dr_due(self):
        """dr_mode="interval": redraw the envs whose timer ran out and restart their timer (no host sync)."""
        lo, hi = self.isaac.dr_interval_steps
        self._dr_timer.sub_(1)
        due = self._dr_timer <= 0
        g = self.radio.gen
        for k, (a, b) in self._dr_radio.items():
            shape = (self.E, self.radio.G, 2) if k == "gnb_offset_m" else (self.E,)
            new = a + (b - a) * torch.rand(shape, device=self.dev, generator=g)
            cur = getattr(self.radio, k)
            m = due.view(-1, *([1] * (cur.dim() - 1)))
            cur.copy_(torch.where(m, new, cur))
        nxt = torch.randint(int(lo), int(hi) + 1, (self.E,), device=self.dev, generator=g)
        self._dr_timer.copy_(torch.where(due, nxt, self._dr_timer))

    # ------------------------------------------------------------------ contract API
    def reset(self, env_ids=None):
        """Partial reset of env_ids (index tensor, bool mask, list or None = all): queues, MAC, fading, clock,
        the shadowing field, the freshness state and the observation features. Radio parameters stay, unless
        IsaacNetCfg.dr_mode is "reset", which redraws them. Other envs are bitwise unaffected."""
        ids = env_index(env_ids, self.E, self.dev)
        if ids is not None and ids.numel() == 0:
            return
        if self.isaac.dr_mode == "reset":
            self.randomize(ids)
        self.eng.reset(ids)
        if self.radio is not None:
            self.radio.reset(ids)
        fill_rows(self.last_cap, ids, 0)
        fill_rows(self._prev_valid, ids, False)
        fill_rows(self._dr_timer, ids, 0)
        self.obs_features.reset(ids)

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

    def _blocked(self, p, blocked_fn):
        """[E,R,G] line-of-sight blockage at poses p, or None."""
        if not self.isaac.blockage:
            return None
        blk = blocked_fn(p) if blocked_fn is not None else None
        if self.isaac.robot_blockers and self.R > 1:
            E, R, G = self.E, self.R, self.radio.G
            a = self.radio.gnb_env[:, None].expand(E, R, G, 3)
            b = p[:, :, None, :].expand(E, R, G, 3)
            rb = segment_sphere_blocked(a, b, p, self.isaac.blocker_radius_m, ignore=self._eye)
            blk = rb if blk is None else (blk | rb)
        return blk

    def _engine_blocked_fn(self, blocked_fn):
        """radio="engine": the blocked_fn becomes the engine radio's LOS callback (NRConfig.los_source="callback")."""
        if self.config.los_source == "callback" and hasattr(self.eng, "set_los_callback"):
            if getattr(self, "_los_fn", None) is not blocked_fn:
                self.eng.set_los_callback(blocked_fn)
                self._los_fn = blocked_fn
        elif not getattr(self, "_warned_fn", False):
            self._warned_fn = True
            warnings.warn("radio='engine' ignores blocked_fn unless NRConfig(los_source='callback') at level L2 "
                          "(see docs/obstacles.md)", stacklevel=3)

    def _snr(self, poses, blocked_fn):
        """SNR [E,R] averaged in dB over pose chunks between the previous and the current end-of-step pose."""
        prev = torch.where(self._prev_valid[:, None, None], self._prev, poses)
        C = self.pose_chunks
        snr, serving, blk_last = 0.0, None, None
        for c in range(C):
            p = prev + ((c + 0.5) / C) * (poses - prev)
            blk = self._blocked(p, blocked_fn)
            s, serving = self.radio.snr_db(p, blk).max(-1)
            snr = snr + s / C
            if blk is not None:
                blk_last = blk.gather(-1, serving[..., None]).squeeze(-1)
        return snr, serving, blk_last

    def step(self, t, poses: torch.Tensor, cur_tag: Optional[torch.Tensor] = None, blocked_fn=None,
             blockers: Optional[torch.Tensor] = None) -> dict:
        """Advance every env by one control step (t is ignored: per-env clocks).

        poses [E,R,3] (or [E,R,2]) env-local positions at the END of this control step.
        cur_tag [E] long: the env's current tag (-1 = none) for tag_delivered.
        blocked_fn(poses [E,R,3]) -> [E,R,G] bool line-of-sight blockage, evaluated per pose chunk (the per-env gNB
        positions are net.radio.gnb_env [E,G,3]). Ignored with IsaacNetCfg.blockage = False. With radio="engine"
        it becomes the engine radio's LOS source when NRConfig.los_source="callback" (multi-cell scene blockage;
        the poses are in the radio frame, the gNBs at the config's cells), else it is ignored with a warning.
        blockers [E,M,3] (x, y, class): extra dynamic blockers (humans, vehicles) as TR 38.901 model-B screens, for
        radio="engine" at level L2 with NRConfig(blockage=True, blockage_model="screen").
        Returns a dict:
          delivered [E,R] bool      at least one message of the robot was delivered this step
          newest_cap [E,R] long     newest capture step delivered this step (-1 if none), env clock
          last_cap [E,R] long       newest capture step delivered this episode (0 = the state at reset)
          aoi_s [E,R] float         age of that information at the end of the step, in seconds
          queue_len, queue_bytes    FIFO state after the step
          sinr_db [E,R], rsrp_dbm [E,R], serving [E,R], blocked [E,R], los [E,R]   radio of this step (rsrp =
                                    SINR + the env's noise floor: the received power when there is no
                                    interference; los = the engine radio's LOS state of the serving link, or
                                    ~blocked when the radio has none)
          tag_delivered [E] bool    if cur_tag is given
          access_state [E,R] long, access_sleep_frac [E,R], rach_attempts [E,R], rlf [E,R] bool
                                    passed through from the NR engine when it reports them (core/access.py:
                                    NRConfig rach / drx; core/radio.py: rlf with several cells)
          msg_delivered, timed_out [E,R,F] bool, cap, cls [E,R,F] long, delay_s [E,R,F] float (NaN if not
          delivered), t [E] long    per message slot as queued before the step, from the engine
        """
        if not self._submitted:
            self.submit(None, TrafficRequest(torch.zeros(self.E, self.R, dtype=torch.long, device=self.dev)))
        self._submitted = False
        if poses.shape[-1] == 2:
            poses = torch.cat([poses, torch.zeros_like(poses[..., :1])], -1)
        if self.isaac.dr_mode == "interval" and self._dr_radio:
            self._dr_due()
        # det_env of the engine compares queued messages with the hid of the step; use the env's current tag
        self.eng._last_hid.copy_(cur_tag.clamp(min=0) if cur_tag is not None else self._zero_hid)
        blocked = None
        if self.radio is not None:
            if blockers is not None:
                raise ValueError("blockers= needs radio='engine' (TR 38.901 screens in the engine radio); with the "
                                 "Isaac radio pass a blocked_fn")
            snr, serving, blocked = self._snr(poses, blocked_fn)
            o = self.eng.step(None, snr)
            noise = self.radio.noise_dbm[:, None]
        else:
            if blocked_fn is not None:
                self._engine_blocked_fn(blocked_fn)
            kw = {}
            if blockers is not None:
                if self.level != "L2":
                    raise ValueError("blockers= needs level 'L2' (the NR engine's radio)")
                kw["blockers"] = blockers
            o = self.eng.step(None, poses, **kw)
            serving = o.get("serving_cell", torch.zeros(self.E, self.R, dtype=torch.long, device=self.dev))
            noise = self.config.subband_noise_dbm
            blocked = o.get("blocked")
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
            rsrp_dbm=o["sinr_db"] + noise,
            serving=serving,
            blocked=blocked if blocked is not None else torch.zeros_like(newest, dtype=torch.bool),
            los=o["los"] if "los" in o else (~blocked if blocked is not None
                                             else torch.ones_like(newest, dtype=torch.bool)),
            msg_delivered=o["delivered"],
            timed_out=o["timed_out"],
            cap=o["cap"],
            cls=o["cls"],
            delay_s=o["delay"] * self.step_dt,
            t=o["t"],
        )
        if cur_tag is not None:
            out["tag_delivered"] = o["det_env"] & (cur_tag >= 0)
        for k in _PASSTHROUGH:                       # access state machine and RLF of the NR engine (level L2)
            if k in o:
                out[k] = o[k]
        self.obs_features.update(out)
        return out

    def obs(self) -> torch.Tensor:
        """[E,R,obs_dim] the selected observation features of the last step (zeros after a reset)."""
        return self.obs_features.get()

    # ------------------------------------------------------------------ checkpoints (core/checkpoint.py)
    def save(self, path, extra=None, host=None):
        """Write the module's state (engine included) to path; host (an env with NetEnvMixin, or a NetRuntime) adds
        its multi-rate state under "host.*" keys. Returns path."""
        sd = self.state_dict()
        if host is not None:
            sd.update({f"host.{k}": v for k, v in host_state(host).items()})
        return _ckpt.save(self, path, extra=extra, state=sd)

    def load(self, path, strict: bool = True, host=None, global_rng: bool = False):
        """Restore a file written by save() (same level, E, R and config; checked). host: restore the env's
        multi-rate state too, if the file has it. Returns the file's `extra`."""
        ck = _ckpt.read(path)
        if not isinstance(ck, dict) or "state" not in ck:
            raise ValueError(f"{path} is not an isaac_net checkpoint (no 'state')")
        _ckpt.check(self, ck, strict=strict)
        sd = {k: v for k, v in ck["state"].items() if not k.startswith("host.")}
        hs = {k[5:]: v for k, v in ck["state"].items() if k.startswith("host.")}
        self.load_state_dict(sd, strict=strict)
        if host is not None and hs:
            load_host_state(host, hs, self.dev)
        if global_rng:
            _ckpt.set_global_rng(ck.get("global_rng"))
        return ck.get("extra")

    def queued(self) -> torch.Tensor:
        return self.eng.queued()


# --------------------------------------------------------------------------------------------------------------
# Checkpoints of a hosted module (NetEnvMixin / NetRuntime) and the env lookup used by the rsl_rl hook
# --------------------------------------------------------------------------------------------------------------
# Multi-rate and traffic state that NetEnvMixin (net_out: the held outputs, _pend_send / _pend_tag: messages pending
# for the next network step, _net_tick: the decimation tick) and NetRuntime (send, last_send, fresh, steps) keep on
# their host, outside the NetModule.
HOST_STATE = ("net_out", "_pend_send", "_pend_tag", "_net_tick", "send", "last_send", "fresh", "steps")


def host_state(host) -> dict:
    """{name: tensor | int | dict of tensors} of the HOST_STATE attributes the host has (flat keys for dicts)."""
    out = {}
    d = getattr(host, "__dict__", {})
    for n in HOST_STATE:
        if n not in d:
            continue
        v = d[n]
        if isinstance(v, dict):
            out[n] = "dict"
            for k, x in v.items():
                if torch.is_tensor(x):
                    out[f"{n}[{k}]"] = x.detach().clone()
        elif torch.is_tensor(v):
            out[n] = v.detach().clone()
        elif v is None or isinstance(v, (bool, int, float)):
            out[n] = v
    return out


def load_host_state(host, hs: dict, device=None):
    """Inverse of host_state: tensors in place when the host has a tensor of that shape, else assigned."""
    for n in HOST_STATE:
        if n not in hs:
            continue
        v = hs[n]
        if isinstance(v, str) and v == "dict":
            pre = n + "["
            setattr(host, n, {k[len(pre):-1]: x.to(device) if device is not None else x
                              for k, x in hs.items() if k.startswith(pre) and k.endswith("]")})
        elif torch.is_tensor(v):
            cur = getattr(host, n, None)
            if torch.is_tensor(cur) and cur.shape == v.shape and cur.dtype == v.dtype:
                cur.copy_(v.to(cur.device))
            else:
                setattr(host, n, v.to(device) if device is not None else v.clone())
        else:
            setattr(host, n, v)


def find_network(env):
    """(host, NetModule) of an env (any gymnasium / rsl_rl wrapper of it), or (None, None): a manager-based env's
    env.isaac_net (NetRuntime, isaac/runtime.py), or a Direct env with NetEnvMixin (env.net)."""
    seen, e = set(), env
    while e is not None and id(e) not in seen:
        seen.add(id(e))
        d = getattr(e, "__dict__", {})
        rt = d.get("isaac_net")
        if rt is not None and isinstance(getattr(rt, "net", None), NetModule):
            return rt, rt.net
        if isinstance(d.get("net"), NetModule):
            return e, d["net"]
        e = d.get("unwrapped") or d.get("env") or getattr(e, "unwrapped", None)
    return None, None


def save_env_network(env, path, extra=None):
    """Save the network of an env (find_network) with its host's multi-rate state; returns path, or None if the env
    has no isaac_net network."""
    host, net = find_network(env)
    if net is None:
        return None
    return net.save(path, extra=extra, host=host)


def load_env_network(env, path, strict: bool = True):
    """Restore a file written by save_env_network into the env's network; returns the file's extra (None if the env
    has no isaac_net network)."""
    host, net = find_network(env)
    if net is None:
        return None
    return net.load(path, strict=strict, host=host)


# --------------------------------------------------------------------------------------------------------------
# Compatibility: the demo's NetConfig
# --------------------------------------------------------------------------------------------------------------
@dataclass
class NetConfig:
    """Deprecated alias of the demo's configuration; it has no defaults of its own. Pass level, an NRConfig and an
    IsaacNetCfg to NetModule instead. Every field left at None comes from `config` (an NRConfig, default
    NRConfig()) or from IsaacNetCfg: rung None = "L2-legacy" ("L2" of the demo is the slot-level NetSlot, i.e.
    level "L2-legacy"), backend None = "reference" ("ref" also means "reference"), step_dt = control_step_ms,
    frame_depth = frame_buffer, msg_sizes and timeout_steps from the NRConfig, pose_chunks and gnb_pos from
    IsaacNetCfg. to_kwargs() maps the fields onto NetModule(level, E, R, device, config, ...)."""
    num_envs: int
    num_robots: int
    device: str = "cuda"
    rung: Optional[str] = None
    backend: Optional[str] = None
    step_dt: Optional[float] = None
    slot_dt: Optional[float] = None
    pose_chunks: Optional[int] = None
    frame_depth: Optional[int] = None
    msg_sizes: Optional[Sequence[float]] = None
    timeout_steps: Optional[int] = None
    gnb_pos: Optional[Sequence[Sequence[float]]] = None
    config: Optional[NRConfig] = None
    isaac: Optional[IsaacNetCfg] = None

    def __post_init__(self):
        warnings.warn("isaac_net.isaac.NetConfig is deprecated: pass level, an NRConfig and an IsaacNetCfg to "
                      "NetModule (see docs/isaac-lab.md)", DeprecationWarning, stacklevel=3)

    def to_kwargs(self) -> dict:
        if self.slot_dt is not None and abs(self.slot_dt - 0.0025) > 1e-12:
            raise ValueError("slot_dt is set by the engine (NRConfig numerology and TDD pattern)")
        level = {"L2": "L2-legacy", None: "L2-legacy"}.get(self.rung, self.rung)
        over = {}
        if self.msg_sizes is not None:
            over["msg_sizes"] = tuple(float(s) for s in self.msg_sizes)
        if self.frame_depth is not None:
            over["frame_buffer"] = self.frame_depth
        if self.timeout_steps is not None:
            over["timeout_steps"] = self.timeout_steps
        if self.step_dt is not None:
            over["control_step_ms"] = self.step_dt * 1000.0
        cfg = (self.config or NRConfig()).with_(**over)
        isc = self.isaac or IsaacNetCfg()
        if self.pose_chunks is not None:
            isc = isc.with_(pose_chunks=self.pose_chunks)
        if self.gnb_pos is not None:
            isc = isc.with_(gnb_pos=tuple(tuple(p) for p in self.gnb_pos))
        return dict(level=level, num_envs=self.num_envs, num_robots=self.num_robots, device=self.device, config=cfg,
                    backend=self.backend or "reference", isaac=isc)


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
    """Deprecated: the default observation [E,R,4] (AoI, SNR, queued frames, delivered this step) with the one
    normalization of isaac/obs.py. Use NetModule.obs() with IsaacNetCfg.obs_features instead."""
    from .config import DB_SCALE, TIME_SCALE_STEPS
    return torch.stack([
        (out["aoi_s"] / (TIME_SCALE_STEPS * step_dt)).clamp(0.0, 1.0),
        out["sinr_db"] / DB_SCALE,
        out["queue_len"].float() / depth,
        out["delivered"].float(),
    ], -1)


__all__ = ["NetModule", "NetConfig", "IsaacNetCfg", "TrafficRequest", "ParamRanges", "MessageHistory",
           "net_features", "FAST_BACKENDS", "BACKENDS", "RADIO_PARAMS"]
