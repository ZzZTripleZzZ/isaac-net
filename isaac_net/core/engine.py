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
                                      params = a fit file path or dict from isaac_net.tools.fit_levels.
  "ORACLE", "NOCOMM"                  value-of-information bounds (levels/bounds.py): instant lossless delivery,
                                      and nothing delivered. Not network models: they bracket what any level
                                      can give a task.

Backends
  "reference"                          the readable eager engine of the level (every level)
  "eager", "graph", "compile", "triton" prototype fast backends (graph = bitwise equal to reference; triton for
                                      L1 / L2-legacy only).
                                      The surrogate and bound levels have "reference" (= "eager") and "graph".
  L2 (NR engine)                      "reference" (= "eager"), "graph" (CUDA graphs of the reference step, bitwise
                                      equal to it) and "triton" (fused UL slot kernel, equal to rounding); the fast
                                      ones need CUDA and rng="engine" (the default). See nr_fast.py.

Every engine returned here has the contract API of ARCHITECTURE.md:
  reset(env_ids=None)                 partial reset: None, index tensor, list or bool mask [E]
  submit(t, requests, snr_db=None)    enqueue new messages (Requests or a send tensor [E,R]); returns accepted [E,R]
  step(t, poses_or_snr) -> dict       advance [t, t+1): delivered / timed_out / cap / cls / delay [E,R,F],
                                      newest [E,R], det_env [E], queue_len / queue_bytes / sinr_db [E,R], t [E]
  clock [E]                           per-env episode clock; pass t=None to use it (recommended)
plus the legacy calls add_frames(t, send, det, hid, snr_db) and step(t, snr_db, hid) -> (newest, det_env).

Traffic models (NRConfig.traffic, traffic.TrafficModel) run inside NREngine.step on level "L2"; every other level
raises if the config asks for them (see _check_traffic).
"""
from __future__ import annotations

import math

import torch

from .channels import install_per_robot_fading, rho_per_ms_from_speed
from .config import NRConfig
from .levels import BOUND_LEVELS, SURROGATE_LEVELS, fit_app, make_level
from .nr_engine import NRNet
from .proto import netsim as _proto
from .radio import RadioMC
from .queues import env_mask, onehot
from .traffic import Requests, TrafficGen, generates

PROTO_LEVELS = ("L0", "L0DR", "L05", "L05Q", "L1")
SIM_LEVELS = PROTO_LEVELS + ("L2", "L2-legacy")
LEVELS = SIM_LEVELS + SURROGATE_LEVELS + BOUND_LEVELS
WIFI_LEVELS = ("WIFI",)                  # 802.11 uplink (core/wifi, docs/wifi.md)
FAST_BACKENDS = ("eager", "graph", "compile", "triton")
BACKENDS = ("reference",) + FAST_BACKENDS


def _proto_app(cfg: NRConfig):
    """Application constants and RNG mode of the prototype, surrogate and bound levels (constructor kwargs)."""
    return {"fb": cfg.frame_buffer, "timeout": cfg.timeout_steps, "ul_per_step": cfg.proto_slots_per_step,
            "rng": cfg.rng}


def _check_proto_config(level, cfg: NRConfig):
    """What the prototype levels cannot honor: multi-cell (except L2-legacy) and non-legacy radio (L1, QA)."""
    cfg.proto_slots_per_step          # raises if the control step is not a whole number of UL slots
    if level != "L2-legacy" and cfg.n_cells != 1:
        raise ValueError(f"level {level} is single-cell; multi-cell runs on 'L2' or 'L2-legacy'")
    if level in ("L1", "QA") and not cfg.is_legacy_cell():
        raise ValueError(f"level {level} has the fixed legacy radio (one gNB at the origin, -90 dBm noise floor); "
                         "use the default cell settings, or level 'L2' / 'L2-legacy'")


def _check_traffic(level, cfg: NRConfig):
    """Traffic models run inside the NR engine step. Every other level would silently drop them, so refuse."""
    if generates(cfg.traffic):
        names = ", ".join(f"{m.kind}()" for m in cfg.traffic if m.generates)
        raise ValueError(f"level {level} ignores NRConfig.traffic ({names}): traffic models run inside the NR "
                         "engine step. Use level 'L2', or generate the messages yourself and pass them to submit(). "
                         "policy() alone is accepted by every level.")


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


def _edge_wrapped(factory):
    """make_engine returns EdgeLoop(engine) when the config sets `edge` (NRConfig(edge=EdgeConfig(...)))."""
    import functools

    @functools.wraps(factory)
    def make(*args, **kw):
        net = factory(*args, **kw)
        cfg = getattr(net, "config", None)
        if cfg is not None and getattr(cfg, "edge", None) is not None:
            from .edge import EdgeLoop
            return EdgeLoop(net, cfg.edge)
        return net
    return make


def _bgenergy_wrapped(factory):
    """make_engine adds BackgroundLoop (NRConfig.background, innermost, below EdgeLoop) and EnergyLoop
    (NRConfig.energy, outermost) when the config sets them; see core/background.py and core/energy.py."""
    import functools
    import inspect
    sig = inspect.signature(factory)

    @functools.wraps(factory)
    def make(*args, **kw):
        a = sig.bind(*args, **kw)
        a.apply_defaults()
        p = dict(a.arguments)
        cfg = p["config"] if p["config"] is not None else NRConfig()
        bg, en = getattr(cfg, "background", None), getattr(cfg, "energy", None)
        if bg is None and en is None:
            return factory(*args, **kw)
        if p["sizes"] is not None:
            cfg = cfg.with_(msg_sizes=tuple(float(s) for s in p["sizes"]))
        seed = p["seed"] if p["seed"] is not None else cfg.seed
        if seed is None:
            seed = int(torch.randint(0, 2 ** 62, ()).item())
        rest = {k: p[k] for k in ("sizes", "params", "inject", "strict")}
        if bg is not None and bg.n_background > 0:
            from .background import BackgroundLoop
            if cfg.edge is not None and cfg.edge.return_path == "nr_dl":
                raise ValueError("EdgeConfig(return_path='nr_dl') cannot be combined with background users")
            inner = BackgroundLoop.build(p["level"], p["E"], p["R"], p["device"], cfg.with_(energy=None),
                                         p["backend"], factory, seed=seed, **rest)
            if cfg.edge is not None:
                from .edge import EdgeLoop
                inner = EdgeLoop(inner, cfg.edge)
        else:
            inner = factory(p["level"], p["E"], p["R"], p["device"], cfg.with_(background=None, energy=None),
                            p["backend"], seed=seed, **rest)
        if en is None:
            return inner
        from .energy import EnergyLoop
        return EnergyLoop(inner, en, seed=seed, config=cfg)
    return make


@_bgenergy_wrapped
@_edge_wrapped
def make_engine(level, E, R, device="cpu", config: NRConfig | None = None, backend="reference", *, sizes=None,
                params=None, seed=None, inject=False, strict=False):
    """Build the network engine of fidelity `level` for E envs x R robots.

    config: NRConfig shared by every module (default NRConfig()); the prototype levels read only its application
      fields (frame_buffer, timeout_steps, control_step_ms -> UL slots per step, msg_sizes) and rng / seed.
    sizes: override of config.msg_sizes. params: fitted parameters of L0 ({"mu", "sig", "p"}, or
      {"q", "p"} for an i.i.d. delay from an empirical marginal: q = delay quantiles or sorted sample in control
      steps, drawn by inverted CDF) and L05 / L05Q
      ({"q", "pdrop"}); for TR / GE / QA / NN a fit file path, a fit-file dict or the level's own dict (see
      levels.load_level_params). seed: overrides config.seed. With config.rng = "engine" (default) every draw of
      the prototype, surrogate and bound levels comes from the engine's streams seeded by it (proto/rng.py);
      with "global", reset draws use the engine generator and stepping draws the global torch RNG. The NR engine
      (L2) keys its slot draws and, with "engine", its radio (RadioMC) draws by (seed, env id, episode) as well;
      only traffic models (TrafficGen) still use one generator per engine.
    inject: fast backends only, take the per-slot random draws from set_noise(...) (equivalence tests).
    strict: raise if the config sets fields away from their defaults that this level ignores
      (config.unused_fields(level)); by default they are ignored silently.
    L0, L0DR and L1 without params take them from the config (l0_*, dr_*, l1_eta; defaults = earlier behavior).
    config.edge (an EdgeConfig) wraps the engine in core.edge.EdgeLoop: same API, plus the edge-loop step keys.
    """
    if level not in LEVELS + WIFI_LEVELS:
        raise ValueError(f"unknown level {level!r}; one of {LEVELS + WIFI_LEVELS}")
    if backend in ("orig", "ref"):
        backend = "reference"
    if backend not in BACKENDS:
        raise ValueError(f"unknown backend {backend!r}; one of {BACKENDS}")
    cfg = config if config is not None else NRConfig()
    if sizes is not None:
        cfg = cfg.with_(msg_sizes=tuple(float(s) for s in sizes))
    sizes = tuple(cfg.msg_sizes)
    if level != "L2" and (cfg.rach or cfg.drx):
        raise ValueError(f"level {level} ignores NRConfig.rach / drx: the access state machine (core/access.py) "
                         "gates the NR engine's MAC; use level 'L2'")
    if level in WIFI_LEVELS:
        from .wifi.engine import make_wifi_level
        return make_wifi_level(E, R, device, cfg, backend, seed=seed, inject=inject, strict=strict)
    if strict and cfg.unused_fields(level):
        raise ValueError(f"level {level} ignores these config fields: {', '.join(cfg.unused_fields(level))} "
                         "(see NRConfig.unused_fields and docs/configurability.md)")
    if level != "L2":
        _check_traffic(level, cfg)
    params = _level_params(level, cfg, params)
    if seed is None:
        seed = cfg.seed
    if level == "L2":
        seed = seed if seed is not None else cfg.seed
        if backend in ("reference", "eager"):
            return NREngine(E, R, device, cfg, seed=seed)
        if backend == "graph":
            from .nr_fast import NRGraphEngine
            return NRGraphEngine(E, R, device, cfg, seed=seed)
        if backend == "triton":
            if cfg.rach or cfg.drx:
                raise ValueError("NRConfig.rach / drx block scheduling through MacLink.sched_ok, which the fused triton "
                                 "kernel does not read; use backend='graph' (bitwise equal to the reference) or "
                                 "'reference'")
            from .nr_fast import NRTritonEngine
            return NRTritonEngine(E, R, device, cfg, seed=seed)
        raise NotImplementedError(f"backend {backend!r} is not available for the NR engine: 'reference' (= 'eager'), "
                                  "'graph' (bitwise equal to the reference) or 'triton'")
    _check_proto_config(level, cfg)
    if level in SURROGATE_LEVELS + BOUND_LEVELS:
        if backend not in ("reference", "eager", "graph"):
            raise NotImplementedError(f"level {level} has the backends 'reference' and 'graph', not {backend!r}")
        net = make_level(level, E, R, device, sizes, params, backend=backend, inject=inject, seed=seed,
                         app=fit_app(cfg), **_proto_app(cfg))
        net.config = cfg
        return net
    if level == "L2-legacy" and not cfg.is_legacy_cell():
        if backend != "reference":
            raise NotImplementedError("the multi-cell legacy engine (NetSlotMC) has only the reference backend")
        from .proto.netsim_mc import NetSlotMC
        net = NetSlotMC(E, R, device, sizes, cfg, seed=seed, **_proto_app(cfg))
    else:
        rung = "L2" if level == "L2-legacy" else level
        if backend == "reference":
            net = _proto.make_net(rung, E, R, device, sizes, params, seed=seed, **_proto_app(cfg))
        else:
            from .proto.netsim_fast import NetFast
            net = NetFast(rung, E, R, device, sizes, params=params, backend=backend, inject=inject, seed=seed,
                          **_proto_app(cfg))
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
        self.seed = seed
        self.net = NRNet(E, R, self.dev, cfg.msg_sizes, cfg, generator=self.gen, seed=seed)
        self.rng = self.net.rng                # engine RNG (None with rng="global"); sharded.set_env_offset finds it
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
        # traffic models (NRConfig.traffic): own generator seeded from the engine seed, so neither the policy's
        # sampling nor the network's draws shift the traffic, and the traffic does not shift the network
        self.traffic = None
        self._extras = False
        self._gate = None
        if generates(cfg.traffic):
            tseed = (int(seed) * 6364136223846793005 + 1442695040888963407) % 2 ** 62
            self.traffic = TrafficGen(cfg.traffic, E, R, self.dev, cfg.control_step_ms, cfg.slots_per_step,
                                      seed=tseed)
            self._enable_extras()
        # access state machine (NRConfig.rach / drx, core/access.py): hooks on this instance's NRNet; None = off, and
        # the engine then runs exactly the ops it ran before the feature
        self.access = None
        if cfg.rach or cfg.drx:
            from .access import AccessStage
            self.access = AccessStage(self)

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

    def counters(self):
        """NRNet.counters() ({"ul": ..., "dl": ...}) plus, with rach / drx, "access": preamble transmissions,
        collisions, successes, failed procedures and RRC releases since the last full reset."""
        res = self.net.counters()
        if self.access is not None:
            res["access"] = self.access.counters()
        return res

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

    # ------------------------------------------------------------------ traffic models and message extras
    def _enable_extras(self):
        """Per-message arrival offset, tag, priority and deadline in the UL queue, and the hooks that gate the
        stream by arrival slot. Only called when traffic models or submit extras are used, so an engine without
        them runs exactly the ops it ran before. The hooks wrap four UlMac methods on this instance
        (sr_step / slot: open the stream up to the arrivals of the current slot; end_step: open it fully and
        snapshot the extras before compaction; handover: a flush spares messages that have not arrived yet);
        mac.py and nr_engine.py are unchanged."""
        if self._extras:
            return
        self._extras = True
        ul = self.net.ul
        ul.q.enable_extras()
        sr0, slot0, end0 = ul.sr_step, ul.slot, ul.end_step

        def sr_step(g):
            self._open_gate(g)
            return sr0(g)

        def slot(g, *a, **k):
            self._open_gate(g)
            return slot0(g, *a, **k)

        def end_step(t, timeout):
            self._close_gate()
            q = ul.q
            self._snap = {"cap": q.cap.clone(), "fin": q.fin.clone(), "off": q.off.clone(), "tag": q.tag.clone(),
                          "prio": q.prio.clone(), "dline": q.dline.clone(), "bytes": (q.end - q.start).clone()}
            return end0(t, timeout)

        ho0 = ul.handover

        def handover(ho, flush=False):
            # ho_rlc="flush" / RLC UM drop every queued frame not yet completed; a message that arrives in a
            # later slot of this step is not queued yet at the handover, so it must survive it
            if self._gate is None:
                return ho0(ho, flush)
            q = ul.q
            pending = (q.cap >= 0) & (q.start >= q.enq[..., None]) & ~q.lost
            r = ho0(ho, flush)
            ul.ctr["lost_frames"] -= (pending & q.lost).sum()
            q.lost = q.lost & ~pending
            return r

        ul.sr_step, ul.slot, ul.end_step, ul.handover = sr_step, slot, end_step, handover
        self._snap = None
        self.traffic_stats = {k: torch.zeros((), dtype=torch.long, device=self.dev)
                              for k in ("generated", "generated_bytes", "accepted", "accepted_bytes", "refused")}

    def _open_gate(self, g):
        """Stream bytes of messages arriving at slot rel = g - g0 or earlier become visible to SR/BSR and the MAC."""
        if self._gate is None:
            return
        ends, slots, base = self._gate
        rel = g - self._g0
        vis = torch.where(slots <= rel, ends, base[..., None]).max(-1).values
        self.net.ul.q.enq = torch.maximum(base, vis)

    def _close_gate(self):
        if self._gate is not None:
            self.net.ul.q.enq = self._full_enq
            self._gate = None
            self._gate_seen = True

    def _enqueue(self, T, want, nbytes, slot, det, tag, prio, dline, snr):
        """NRNet.add_frames for arbitrary byte sizes and the extras; returns (accepted [E,R], accepted bytes)."""
        net, N = self.net, self.config.slots_per_step
        q = net.ul.q
        count = q.count()
        adm = net._admit(net.ul, T, want)
        if net.log_stats:
            net.stats["overflow"] += int((adm & (count >= q.F)).sum())
            net.stats["discarded"] += int((want & ~adm).sum())
        size = net.air_bytes(nbytes.float())
        nact = (q.cap >= 0).any(-1).sum(-1)
        before = q.enq
        acc, i, oh = q.add(T, adm, size)
        if net.log_stats:
            net.refused_env += (want & ~acc).sum(-1)
        put = lambda name, v: setattr(q, name, torch.where(oh, v[..., None].to(getattr(q, name).dtype), getattr(q, name)))
        put("cls", torch.zeros_like(slot))
        put("det", det)
        put("hid", self._last_hid[:, None].expand(-1, self.R))
        put("f_nact", nact[:, None].expand(-1, self.R))
        put("f_snr", snr)
        put("f_own", i)
        put("off", slot.double() / N)
        put("tag", tag)
        put("prio", prio)
        put("dline", dline)
        return acc, q.enq - before

    def _inject(self, T, triggers):
        """Generate this step's messages, enqueue them in arrival order and close the stream gate."""
        arr = self.traffic.step(self.clock, triggers)
        q = self.net.ul.q
        base = q.enq
        ends = []
        n_acc = torch.zeros(self.E, self.R, dtype=torch.long, device=self.dev)
        st = self.traffic_stats
        for m in range(arr.valid.shape[-1]):
            v = arr.valid[..., m]
            acc, _ = self._enqueue(T, v, arr.nbytes[..., m], arr.slot[..., m], arr.det[..., m] & v,
                                    arr.tag[..., m], arr.prio[..., m], arr.dline[..., m], self._last_snr)
            ends.append(q.enq)
            n_acc = n_acc + acc.long()
            st["generated"] += v.sum()
            st["generated_bytes"] += (arr.nbytes[..., m].round().long() * v).sum()
            st["accepted"] += acc.sum()
            st["accepted_bytes"] += (arr.nbytes[..., m].round().long() * acc).sum()
            st["refused"] += (v & ~acc).sum()
        self._full_enq = q.enq
        if ends:
            never = torch.full_like(arr.slot, self.config.slots_per_step)      # empty entries never open the gate
            self._gate = (torch.stack(ends, -1), torch.where(arr.valid, arr.slot, never), base)
            q.enq = base
            self._g0 = T * self.config.slots_per_step
        return arr, n_acc

    def _outputs_extras(self, out, delivered, timed, dropped):
        """Arrival-corrected delays and the per-message extras from the snapshot taken before compaction."""
        sn, c = self._snap, self.config
        valid = sn["cap"] >= 0
        delay = (sn["fin"] - sn["cap"].double() - sn["off"]).float()
        out["delay"] = torch.where(delivered, delay, torch.full_like(delay, float("nan")))
        cap_rel = sn["cap"] - self.epoch.view(-1, 1, 1)
        out["arrival"] = torch.where(valid, cap_rel.double() + sn["off"], torch.full_like(sn["off"], float("nan")))
        out["arrival_slot"] = torch.where(valid, (sn["off"] * c.slots_per_step).round().long(),
                                          torch.full_like(sn["cap"], -1))
        out["tag"] = torch.where(valid, sn["tag"], torch.zeros_like(sn["tag"]))
        out["priority"] = torch.where(valid, sn["prio"], torch.zeros_like(sn["prio"]))
        out["bytes"] = torch.where(valid, sn["bytes"], torch.zeros_like(sn["bytes"]))
        late = delivered & (delay.double() * c.control_step_ms > sn["dline"])
        out["deadline_miss"] = late | ((timed | dropped) & torch.isfinite(sn["dline"]))
        net = self.net
        if net.log_stats and net.stats["delay"]:
            dk = delivered & (sn["cap"] <= net.log_cap_max)
            net.stats["delay"][-1] = delay[dk].cpu()

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
        self._gate = None
        if self.traffic is not None and (ids is None or ids.numel() > 0):
            self.traffic.reset(None if ids is None else env_mask(self.E, ids, self.dev))
        if ids is None:
            self.net.reset(None)
            self.T = 0
            self.epoch.zero_()
            self._uniform_epoch = 0
            self._last_snr = torch.zeros_like(self._last_snr)     # may alias the caller's tensors: replace
            self._last_hid = torch.zeros_like(self._last_hid)
            if self.radio is not None:
                self.radio.reset(None)
            if self.access is not None:
                self.access.reset(None)
            return
        if ids.numel() == 0:
            return
        self.net.reset(ids)
        self.epoch.index_fill_(0, ids, self.T)
        self._uniform_epoch = None
        self._last_snr = self._last_snr.clone().index_fill_(0, ids, 0.0)
        self._last_hid = self._last_hid.clone().index_fill_(0, ids, 0)
        if self.radio is not None:
            self.radio.reset(ids)
        if self.access is not None:
            self.access.reset(ids)

    def submit(self, t, requests, snr_db=None, *, tag=None, priority=None, deadline_ms=None):
        """Enqueue new messages at capture time t (None = engine clock). requests: Requests or send [E,R].
        snr_db [E,R] is recorded as a frame feature (default: the SNR of the previous step). Returns accepted.
        tag / priority ([E,R] long or int) and deadline_ms ([E,R] float or float) are optional per-message extras
        that the queue carries and step() reports (tag, priority, deadline_miss)."""
        T = self._now(t)
        req = requests if isinstance(requests, Requests) else Requests(send=requests)
        det = req.det if req.det is not None else torch.zeros_like(req.send, dtype=torch.bool)
        hid = req.hid if req.hid is not None else torch.zeros(self.E, dtype=torch.long, device=self.dev)
        self._last_hid = hid.clone()        # a copy: the caller may reuse its buffer (as the fast backends copy)
        extras = tag is not None or priority is not None or deadline_ms is not None
        if extras:
            self._enable_extras()
        q = self.net.ul.q
        cnt = q.count() if extras else None
        acc = self.net.add_frames(T, req.send, det, hid, self._last_snr if snr_db is None else snr_db)
        if extras:
            oh = onehot(cnt.clamp(max=q.F - 1), q.F) & acc[..., None]
            for name, v in (("tag", tag), ("prio", priority), ("dline", deadline_ms)):
                if v is not None:
                    cur = getattr(q, name)
                    v = torch.as_tensor(v, dtype=cur.dtype, device=self.dev).expand(self.E, self.R)
                    setattr(q, name, torch.where(oh, v[..., None], cur))
        return acc

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
            # rng="engine": the radio draws from the engine's counter RNG keyed by (seed, env id, episode), so an env's
            # shadowing / LOS / O2I draws depend neither on E nor on other envs' resets; rng="global": self.gen
            self.radio = RadioMC(self.config, self.E, self.dev, generator=self.gen, R=self.R, rng=self.rng)
        pg = self.radio.pathgain_db(pos)
        if self.per_robot_doppler:
            speed = self.radio.observe_motion(pos, vel)
            self.net.fading_rho_ms = rho_per_ms_from_speed(speed, self.config.carrier_ghz)
        return pg

    def step(self, t, x=None, cur_hid=None, *, snr_db=None, dl_snr_db=None, pathgain_db=None, vel=None,
             triggers=None):
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
        With traffic models (NRConfig.traffic) the step first generates this step's messages; triggers= feeds the
        event models (a mask [E,R] / [E], or {name: mask}). With traffic models or submit extras it also returns,
        per frame [E,R,F]: arrival (env clock incl. the in-step offset), arrival_slot, tag, priority, bytes (on the
        air), deadline_miss; delay is then measured from the arrival slot. gen_accepted / gen_bytes [E,R] count
        the generated messages accepted this step.
        """
        T = self._now(t)
        legacy = cur_hid is not None
        hid = cur_hid if legacy else self._last_hid
        if self.net.C > 1 and pathgain_db is None and (x is None or x.dim() != 3):
            raise ValueError("several cells: pass poses [E,R,2|3] or pathgain_db=[E,R,C]")
        if self.net.C == 1 and pathgain_db is not None:
            raise ValueError("pathgain_db= needs config.n_cells > 1; use x (poses or SNR) with one cell")
        arr = None
        if self.traffic is not None:
            self._gate_seen = False
            n0 = self.net.ul.q.enq
            arr, gen_acc = self._inject(T, triggers)
            gen_bytes = self._full_enq - n0
            if arr.valid.shape[-1] == 0:
                self._gate_seen = True
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
        self._last_snr = snr.clone()        # a copy: snr may be the caller's x / snr_db buffer
        if self.traffic is not None and not getattr(self, "_gate_seen", True):
            raise RuntimeError("the traffic gate hooks on net.ul were not reached: NRNet no longer calls "
                               "ul.sr_step / ul.slot / ul.end_step; traffic models need an update")
        if self._extras and self._snap is not None:
            self._outputs_extras(out, out["delivered"], out["timed_out"], out["dropped"])
            self._snap = None
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
        if arr is not None:
            out["gen_accepted"] = gen_acc
            out["gen_bytes"] = gen_bytes
        return out
