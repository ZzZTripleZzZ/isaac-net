"""Edge-computing loop on top of any engine: edge processing of delivered uplink messages and the return path.

    from isaac_net.core import EdgeConfig, EdgeLoop, NRConfig, make_engine
    net = make_engine("L2-legacy", E, R, dev, NRConfig(edge=EdgeConfig(service_ms=20.0)))   # wrapped automatically
    net = EdgeLoop(make_engine("L1", E, R, dev), EdgeConfig(return_path="delay"))           # or wrap by hand

EdgeLoop keeps the engine API (reset / submit / step / clock, attribute passthrough) and adds keys to the step dict.
The engine decides when a message reaches the edge: arrival = cap + delay of every message in `delivered`, in the
env's clock (control steps). EdgeLoop then runs, per env:

1. Edge compute stage. One server pool per env shared by its R robots: `servers_per_env` servers, FIFO
   (non-preemptive, one message per server) or processor sharing (each of n messages in service gets
   min(1, servers / n) of a server). Service times per message are deterministic or exponential with a mean per
   message class. The edge holds at most `queue_cap + servers_per_env` messages; an arrival that finds it full is
   dropped, and so is a message that has not started (FIFO) or finished (PS) by its deadline (from capture).
   The stage is an exact continuous-time event simulation: every loop iteration advances each env to its next
   event (arrival, completion, deadline, or the end of the control step), vectorized over envs with fixed
   shapes. The loop runs `max_events_per_step` iterations; an env with more events stops early and continues
   exactly where it stopped at the next step (`edge_lag`), so times stay exact and only their reporting is late.
2. Return path. The newest result of each robot in a step becomes a command: "instant" (arrives at completion),
   "delay" (fixed + uniform jitter + cmd_bytes over a rate from the robot's SINR), or "nr_dl" (a real downlink
   message of cmd_bytes through the NR engine's DL scheduler, reference backend only; the command is enqueued at
   the next control-step boundary, since the engine takes new messages per step).
3. Loop accounting. Each robot keeps the newest command it received (highest capture step): its capture step,
   arrival time, age, and the uplink / edge / return delays of that action.

Step dict keys added (per step; times in control steps of the env clock, NaN = none):
  edge_done [E,R] long            results completed this step
  edge_done_cap, edge_done_time   capture step (-1) and completion time of the newest result completed this step
  edge_dropped [E,R] long         messages dropped at the edge this step (= edge_dropped_full + edge_dropped_deadline)
  edge_queue_len [E] long         messages at the edge after the step (in service, waiting, and not yet admitted)
  edge_in_service [E] long        messages in service after the step
  edge_lag [E] bool               the env ran out of its event budget (it catches up next step)
  act_new [E,R] bool              a newer action (command) reached the robot this step
  act_cap [E,R] long              capture step of the robot's newest action (-1 = none yet)
  act_time [E,R] float            its arrival time at the robot
  act_age [E,R] float             t + 1 - act_cap: age of the action the robot holds at the end of the step (NaN
                                  before its first action)
  act_latency [E,R] float         act_time - act_cap: capture -> uplink -> edge -> return of that action
  act_ul_delay, act_edge_delay, act_ret_delay [E,R] float   the three stages of act_latency
  cmd_dropped [E,R] long          commands lost this step (replaced in flight, DL loss or DL queue full)
Cumulative per-robot counters: counters() -> arrived, completed, dropped_full, dropped_deadline, at_edge, with
arrived == completed + dropped_full + dropped_deadline + at_edge (conservation).

Graph safety: every tensor has a fixed shape, per-env state is updated in place, the step has no host syncs
when it runs inside a CUDA-graph capture, and randomness comes from the default device generator.
EdgeLoop(..., graph=True) captures the edge stage (not the engine) in a CUDA graph; it needs a CUDA device and
return_path "instant" or "delay". Partial reset(env_ids) clears those envs only.
"""
from __future__ import annotations

import math

import torch

from .config import EdgeConfig, NRConfig
from .queues import env_mask

INF = float("inf")
EPS = 1e-9                                  # control steps; completion tolerance of the event loop
F64 = torch.float64


def _capturing(dev):
    return dev.type == "cuda" and torch.cuda.is_current_stream_capturing()


class EdgeLoop:
    """Wrap `engine` (any make_engine level) with an edge server and a return path. See the module docstring."""

    OUT_KEYS = ("edge_done", "edge_done_cap", "edge_done_time", "edge_dropped", "edge_dropped_full",
                "edge_dropped_deadline", "edge_queue_len", "edge_in_service", "edge_lag", "act_new", "act_cap",
                "act_time", "act_age", "act_latency", "act_ul_delay", "act_edge_delay", "act_ret_delay",
                "cmd_dropped")

    def __init__(self, engine, cfg: EdgeConfig | None = None, *, graph=False):
        self.engine = engine
        ncfg = getattr(engine, "config", None) or NRConfig()
        self.ncfg = ncfg
        self.cfg = cfg = cfg if cfg is not None else (ncfg.edge or EdgeConfig())
        self.E, self.R = E, R = engine.E, engine.R
        self.dev = d = torch.device(engine.dev)
        self.F = F = ncfg.frame_buffer
        self.step_ms = ncfg.control_step_ms
        self.c = cfg.servers_per_env
        self.J = J = cfg.queue_cap + cfg.servers_per_env
        self.B = B = R * F
        self.D = D = cfg.ret_inflight
        self.K = cfg.max_events_per_step or (2 * R + 2 * self.c + 8)
        self.graph = graph
        if graph and d.type != "cuda":
            raise ValueError("EdgeLoop(graph=True) needs a CUDA device")
        if graph and cfg.return_path == "nr_dl":
            raise ValueError("return_path='nr_dl' runs through the NR engine (reference backend only): graph=False")
        svc = cfg.service_table(len(ncfg.msg_sizes))
        self.svc = torch.tensor([svc[0]] + list(svc), dtype=F64, device=d) / self.step_ms   # index = class
        self.deadline = INF if cfg.deadline_ms is None else cfg.deadline_ms / self.step_ms
        self.jidx = torch.arange(J, device=d)
        self.robot_of = torch.arange(R, device=d)[:, None].expand(R, F).reshape(-1)          # [R*F]
        if cfg.return_path == "delay":
            bw_hz = ncfg.nprb * 12 * ncfg.scs_khz * 1e3
            share = cfg.ret_share if cfg.ret_share is not None else 1.0 / R
            self.ret_rate_hz = cfg.ret_rate_eta * share * bw_hz
        z = lambda shape, dt, v: torch.full(shape, v, dtype=dt, device=d)
        # edge jobs [E,J] (compacted, in arrival order), ingress [E,B] (sorted by arrival), env time [E]
        self.state = {
            "jv": z((E, J), torch.bool, False), "jarr": z((E, J), F64, 0.0), "jcap": z((E, J), torch.long, -1),
            "jrob": z((E, J), torch.long, 0), "jrem": z((E, J), F64, 0.0), "jdl": z((E, J), F64, INF),
            "jst": z((E, J), torch.bool, False),
            "iv": z((E, B), torch.bool, False), "iarr": z((E, B), F64, INF), "icap": z((E, B), torch.long, -1),
            "irob": z((E, B), torch.long, 0), "isvc": z((E, B), F64, 0.0), "idl": z((E, B), F64, INF),
            "now": z((E,), F64, 0.0),
            # commands in flight [E,R,D]
            "scap": z((E, R, D), torch.long, -1), "sarr": z((E, R, D), F64, INF), "sdone": z((E, R, D), F64, 0.0),
            "sin": z((E, R, D), F64, 0.0),
            # the robot's newest action
            "acap": z((E, R), torch.long, -1), "atime": z((E, R), F64, math.nan), "ain": z((E, R), F64, math.nan),
            "adone": z((E, R), F64, math.nan),
            # cumulative counters
            "n_arr": z((E, R), torch.long, 0), "n_done": z((E, R), torch.long, 0),
            "n_full": z((E, R), torch.long, 0), "n_dl": z((E, R), torch.long, 0),
        }
        self.INIT = {k: v.flatten()[0].item() if v.numel() else 0 for k, v in self.state.items()}
        self._pending_dl = None
        self._dl_obs = None
        self._graph = None
        if cfg.return_path == "nr_dl":
            self._install_dl_hook()

    # ------------------------------------------------------------------ passthroughs
    def __getattr__(self, name):
        if name in ("engine", "state"):
            raise AttributeError(name)
        return getattr(self.engine, name)

    @property
    def clock(self):
        return self.engine.clock

    @property
    def config(self):
        return self.engine.config

    def submit(self, t, requests, snr_db=None, **kw):
        return self.engine.submit(t, requests, snr_db, **kw)

    def add_frames(self, t, send, det, hid, snr_db):
        return self.engine.add_frames(t, send, det, hid, snr_db)

    def queued(self):
        return self.engine.queued()

    # ------------------------------------------------------------------ reset
    def reset(self, env_ids=None):
        """Partial reset of the engine and of the edge state of env_ids (None = all). In place."""
        self.engine.reset(env_ids)
        self.reset_edge(env_ids)

    def reset_edge(self, env_ids=None):
        """Clear the edge and return-path state of env_ids only (the engine is not touched)."""
        m = env_mask(self.E, env_ids, self.dev)
        for k, v in self.state.items():
            mm = m.view(-1, *([1] * (v.dim() - 1)))
            v.copy_(torch.where(mm, torch.full_like(v, self.INIT[k]), v))
        if self._pending_dl is not None:
            nb, cls = self._pending_dl
            self._pending_dl = (torch.where(m[:, None], torch.zeros_like(nb), nb), cls)

    def counters(self):
        """Cumulative per-robot counts [E,R] since the env's last reset (conservation:
        arrived == completed + dropped_full + dropped_deadline + at_edge)."""
        s = self.state
        at = self._scatter(s["jrob"], s["jv"]) + self._scatter(s["irob"], s["iv"])
        return {"arrived": s["n_arr"].clone(), "completed": s["n_done"].clone(), "dropped_full": s["n_full"].clone(),
                "dropped_deadline": s["n_dl"].clone(), "at_edge": at}

    # ------------------------------------------------------------------ step
    def step(self, t, x=None, cur_hid=None, **kw):
        """engine.step(t, x, ...) plus the edge loop. The legacy form (cur_hid given) is passed through unchanged."""
        if self._pending_dl is not None:
            nb, cls = self._pending_dl
            self._pending_dl = None
            self.engine.add_dl_frames(None, nb, cls)
        if cur_hid is not None:
            return self.engine.step(t, x, cur_hid, **kw)
        out = self.engine.step(t, x, **kw)
        out.update(self.process(out))
        return out

    def process(self, out):
        """Edge loop of one control step from the engine's step dict; returns the added keys."""
        delay = out["delay"]
        if "arrival" in out:        # traffic models: delay counts from the in-step arrival, not the capture step
            delay = delay + (out["arrival"] - out["cap"].double()).nan_to_num(0.0).to(delay.dtype)
        ins = (out["delivered"], out["cap"], out["cls"], delay, out["t"],
               out.get("sinr_db", torch.zeros(self.E, self.R, device=self.dev)))
        if self.cfg.return_path == "nr_dl":
            self._dl_lost = torch.zeros(self.E, self.R, dtype=torch.long, device=self.dev)
            self._apply_dl()
        if not self.graph:
            res = self._core(*ins)
        else:
            if self._graph is None:
                self._capture(ins)
            for s, v in zip(self._static_in, ins):
                s.copy_(v)
            self._graph.replay()
            res = {k: v.clone() for k, v in self._static_out.items()}
        if self.cfg.return_path == "nr_dl":
            self._submit_dl(res.pop("_new"), res.pop("_newcap"))
            res["cmd_dropped"] = res["cmd_dropped"] + self._dl_lost + self._dl_refused.long()
        else:
            res.pop("_new"), res.pop("_newcap")
        return res

    # ------------------------------------------------------------------ graph
    def _capture(self, ins):
        self._static_in = [v.clone() for v in ins]
        snap = {k: v.clone() for k, v in self.state.items()}
        s = torch.cuda.Stream(self.dev)
        s.wait_stream(torch.cuda.current_stream(self.dev))
        with torch.cuda.stream(s):
            self._core(*self._static_in)                  # warm-up (allocator, lazy init); state restored below
        torch.cuda.current_stream(self.dev).wait_stream(s)
        for k, v in self.state.items():
            v.copy_(snap[k])
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            self._static_out = self._core(*self._static_in)
        self._graph = g

    # ------------------------------------------------------------------ helpers
    def _scatter(self, rob, mask):
        """Per-robot count [E,R] of mask [E,N] at robot index rob [E,N]."""
        idx = torch.where(mask, rob, torch.full_like(rob, self.R))
        z = torch.zeros(self.E, self.R + 1, dtype=torch.long, device=self.dev)
        return z.scatter_add(1, idx, torch.ones_like(idx))[:, :self.R]

    # ------------------------------------------------------------------ the edge step (captured in graph mode)
    def _core(self, delivered, cap, cls, delay, t, sinr_db):
        E, R, J, B, D, c, dev = self.E, self.R, self.J, self.B, self.D, self.c, self.dev
        cfg, s = self.cfg, self.state
        fifo = cfg.discipline == "fifo"
        T1 = t.to(F64) + 1.0
        # ---- 1. new arrivals -> ingress (sorted by arrival time)
        m = delivered.reshape(E, -1)
        capn = cap.reshape(E, -1)
        arr = torch.where(m, capn.to(F64) + delay.reshape(E, -1).to(F64), torch.full_like(capn, 0, dtype=F64))
        clsn = cls.reshape(E, -1).clamp(0, self.svc.numel() - 1)
        svc = self.svc[clsn]
        if cfg.service_dist == "exponential":
            u = torch.rand(E, B, dtype=F64, device=dev)
            svc = svc * -torch.log1p(-u)
        rob = self.robot_of.expand(E, -1)
        n_arr = s["n_arr"] + self._scatter(rob, m)
        cat = lambda a, b: torch.cat([a, b], 1)
        iv = cat(s["iv"], m)
        key = torch.where(iv, cat(s["iarr"], arr), torch.full((E, 2 * B), INF, dtype=F64, device=dev))
        order = torch.sort(key, dim=1, stable=True).indices
        g = lambda a, b: cat(a, b).gather(1, order)
        iv, iarr, icap = iv.gather(1, order), g(s["iarr"], arr), g(s["icap"], capn)
        irob, isvc = g(s["irob"], rob), g(s["isvc"], svc)
        idl = g(s["idl"], capn.to(F64) + self.deadline)
        n_full = s["n_full"] + self._scatter(irob[:, B:], iv[:, B:])       # ingress overflow (more than R*F queued)
        iv, iarr, icap, irob, isvc, idl = (x[:, :B] for x in (iv, iarr, icap, irob, isvc, idl))
        iarr = torch.where(iv, iarr, torch.full_like(iarr, INF))
        jv, jarr, jcap, jrob, jrem, jdl, jst = (s[k] for k in ("jv", "jarr", "jcap", "jrob", "jrem", "jdl", "jst"))
        now, n_done, n_dl = s["now"], s["n_done"], s["n_dl"]
        # per-step newest result per robot
        res_cap = torch.full((E, R), -1, dtype=torch.long, device=dev)
        res_time = torch.zeros(E, R, dtype=F64, device=dev)
        res_in = torch.zeros(E, R, dtype=F64, device=dev)
        idx = self.jidx
        inf_e = torch.full((E,), INF, dtype=F64, device=dev)
        # ---- 2. event loop
        for k in range(self.K):
            n = jv.sum(-1)
            if fifo:
                busy = jv & (idx < c)
                rate = busy.to(F64)
                jst = jst | busy
                wait = jv & ~jst
            else:
                rate = jv.to(F64) * (c / n.clamp(min=1).to(F64)).clamp(max=1.0)[:, None]
                busy = jv
                wait = jv
            tc = now + torch.where(busy, jrem / torch.where(busy, rate, torch.ones_like(rate)),
                                   torch.full_like(jrem, INF)).amin(-1)
            ta = torch.where(iv[:, 0], iarr[:, 0], inf_e)
            td = torch.where(wait, jdl, torch.full_like(jdl, INF)).amin(-1)
            ev = torch.maximum(torch.minimum(torch.minimum(tc, ta), torch.minimum(td, T1)), now)
            jrem = jrem - rate * (ev - now)[:, None]
            now = ev
            # completions
            comp = busy & (jrem <= EPS)
            crob = torch.where(comp, jrob, torch.full_like(jrob, R))
            best = torch.full((E, R + 1), -1, dtype=torch.long, device=dev).scatter_reduce(
                1, crob, torch.where(comp, jcap, torch.full_like(jcap, -1)), "amax")
            sel = comp & (jcap == best.gather(1, crob))
            srob = torch.where(sel, jrob, torch.full_like(jrob, R))
            bin_ = torch.zeros(E, R + 1, dtype=F64, device=dev).scatter(1, srob, jarr)[:, :R]
            best = best[:, :R]
            upd = best > res_cap
            res_cap = torch.where(upd, best, res_cap)
            res_time = torch.where(upd, now[:, None].expand(E, R), res_time)
            res_in = torch.where(upd, bin_, res_in)
            n_done = n_done + self._scatter(jrob, comp)
            # deadline drops
            dd = wait & ~comp & (jdl <= now[:, None])
            n_dl = n_dl + self._scatter(jrob, dd)
            jv = jv & ~comp & ~dd
            # compaction (stable: arrival order is kept)
            order = ((~jv).long() * J + idx).argsort(-1)
            jv, jarr, jcap, jrob, jrem, jdl, jst = (x.gather(1, order) for x in (jv, jarr, jcap, jrob, jrem, jdl, jst))
            # one arrival
            aev = iv[:, 0] & (ta <= now)
            late = aev & (idl[:, 0] <= now)                     # arrives past its deadline: dropped at once
            n2 = jv.sum(-1)
            adm = aev & ~late & (n2 < J)
            fd = aev & ~late & ~adm
            n_full = n_full + self._scatter(irob[:, :1], fd[:, None])
            n_dl = n_dl + self._scatter(irob[:, :1], late[:, None])
            oh = (idx == n2[:, None]) & adm[:, None]
            jv = jv | oh
            jarr = torch.where(oh, iarr[:, :1], jarr)
            jcap = torch.where(oh, icap[:, :1], jcap)
            jrob = torch.where(oh, irob[:, :1], jrob)
            jrem = torch.where(oh, isvc[:, :1], jrem)
            jdl = torch.where(oh, idl[:, :1], jdl)
            jst = jst & ~oh
            pop = aev[:, None]
            sh = lambda x, v: torch.where(pop, torch.cat([x[:, 1:], torch.full_like(x[:, :1], v)], 1), x)
            iv, iarr, icap = sh(iv, False), sh(iarr, INF), sh(icap, -1)
            irob, isvc, idl = sh(irob, 0), sh(isvc, 0.0), sh(idl, INF)
            if k % 4 == 3 and not _capturing(dev):
                if bool(((now >= T1) & ~(iv[:, 0] & (iarr[:, 0] <= T1))).all()):
                    break
        lag = (now < T1) | (iv[:, 0] & (iarr[:, 0] <= T1))
        # ---- 3. return path: the newest result of each robot becomes a command
        new = res_cap >= 0
        rp = cfg.return_path
        if rp == "instant":
            carr = res_time
        elif rp == "delay":
            extra = torch.full((E, R), cfg.ret_fixed_ms, dtype=F64, device=dev)
            if cfg.ret_jitter_ms > 0:
                extra = extra + cfg.ret_jitter_ms * torch.rand(E, R, dtype=F64, device=dev)
            snr = 10.0 ** ((sinr_db.to(F64) + cfg.ret_snr_offset_db).clamp(min=-20.0) / 10.0)
            rate_bps = self.ret_rate_hz * torch.log2(1.0 + snr)
            tx_ms = cfg.cmd_bytes * 8.0 / rate_bps * 1e3
            carr = res_time + (extra + tx_ms) / self.step_ms
        else:
            carr = torch.full((E, R), INF, dtype=F64, device=dev)
        scap, sarr, sdone, sin_ = s["scap"], s["sarr"], s["sdone"], s["sin"]
        free = scap < 0
        slot = torch.where(free, torch.full_like(scap, -1), scap).argmin(-1)     # a free slot, else the oldest
        replaced = new & ~free.any(-1)
        soh = (torch.arange(D, device=dev) == slot[..., None]) & new[..., None]
        scap = torch.where(soh, res_cap[..., None], scap)
        sarr = torch.where(soh, carr[..., None], sarr)
        sdone = torch.where(soh, res_time[..., None], sdone)
        sin_ = torch.where(soh, res_in[..., None], sin_)
        # ---- 4. delivery to the robot
        dm = (scap >= 0) & (sarr <= T1[:, None, None])
        dcap = torch.where(dm, scap, torch.full_like(scap, -1))
        bcap, bslot = dcap.max(-1)
        take = lambda x: x.gather(-1, bslot[..., None]).squeeze(-1)
        acap = s["acap"]
        anew = bcap > acap
        acap = torch.where(anew, bcap, acap)
        atime = torch.where(anew, take(sarr), s["atime"])
        ain = torch.where(anew, take(sin_), s["ain"])
        adone = torch.where(anew, take(sdone), s["adone"])
        scap = torch.where(dm, torch.full_like(scap, -1), scap)
        sarr = torch.where(dm, torch.full_like(sarr, INF), sarr)
        # ---- write state in place
        upd_state = dict(jv=jv, jarr=jarr, jcap=jcap, jrob=jrob, jrem=jrem, jdl=jdl, jst=jst, iv=iv, iarr=iarr,
                         icap=icap, irob=irob, isvc=isvc, idl=idl, now=now, scap=scap, sarr=sarr, sdone=sdone,
                         sin=sin_, acap=acap, atime=atime, ain=ain, adone=adone, n_arr=n_arr, n_done=n_done,
                         n_full=n_full, n_dl=n_dl)
        dropped_full = n_full - s["n_full"]
        dropped_dl = n_dl - s["n_dl"]
        done = n_done - s["n_done"]
        for kk, v in upd_state.items():
            s[kk].copy_(v)
        f32 = lambda x: x.to(torch.float32)
        has = acap >= 0
        nan = lambda x: torch.where(has, f32(x), torch.full_like(f32(x), math.nan))
        capf = acap.to(F64)
        at_edge = jv.sum(-1) + iv.sum(-1)
        return {
            "edge_done": done, "edge_done_cap": res_cap,
            "edge_done_time": torch.where(new, f32(res_time), torch.full((E, R), math.nan, device=dev)),
            "edge_dropped": dropped_full + dropped_dl, "edge_dropped_full": dropped_full,
            "edge_dropped_deadline": dropped_dl, "edge_queue_len": at_edge,
            "edge_in_service": (jv & (idx < c)).sum(-1) if fifo else jv.sum(-1), "edge_lag": lag,
            "act_new": anew, "act_cap": acap.clone(), "act_time": nan(atime),
            "act_age": nan(T1[:, None] - capf), "act_latency": nan(atime - capf),
            "act_ul_delay": nan(ain - capf), "act_edge_delay": nan(adone - ain), "act_ret_delay": nan(atime - adone),
            "cmd_dropped": replaced.long(), "_new": new, "_newcap": res_cap,
        }

    # ------------------------------------------------------------------ return path through the NR downlink
    def _install_dl_hook(self):
        from .engine import NREngine
        eng = self.engine
        if not isinstance(eng, NREngine) or eng.net.dl is None:
            raise ValueError("return_path='nr_dl' needs the NR engine (level 'L2') with NRConfig(dl=True)")
        backend = getattr(eng, "backend", "reference")
        if backend != "reference":
            # the hook below is Python: the graph and triton backends run it only while capturing, so every later
            # step would replay without observing the DL frames and no command would ever arrive
            raise ValueError(f"return_path='nr_dl' needs the reference backend of the NR engine, not {backend!r}: "
                             "the DL observation hook does not run inside a replayed CUDA graph")
        link = eng.net.dl
        orig = link.end_step

        def end_step(t, timeout):
            dd, dt_, dr = orig(t, timeout)
            self._dl_obs = (dd.clone(), (dt_ | dr).clone(), link.q.fin.clone(), link.q.cls.clone())
            return dd, dt_, dr

        link.end_step = end_step            # instance attribute: observes the DL frames before compaction

    def _apply_dl(self):
        """Commands whose DL message was delivered (or lost) during this engine step."""
        if self._dl_obs is None:
            return
        dd, lost, fin, cls = self._dl_obs
        self._dl_obs = None
        s = self.state
        tag = cls - 1                                                      # [E,R,Fd] capture step of the result
        fin_env = fin - self.engine.epoch.to(F64)[:, None, None]
        scap = s["scap"]
        match = (tag[:, :, None, :] == scap[..., None]) & (scap[..., None] >= 0)      # [E,R,D,Fd]
        got = match & dd[:, :, None, :]
        arr = torch.where(got, fin_env[:, :, None, :], torch.full_like(fin_env[:, :, None, :], INF)).amin(-1)
        s["sarr"].copy_(torch.where(got.any(-1), arr, s["sarr"]))
        gone = (match & lost[:, :, None, :]).any(-1) & ~got.any(-1)
        self._dl_lost = gone.sum(-1)
        s["scap"].copy_(torch.where(gone, torch.full_like(scap, -1), scap))
        s["sarr"].copy_(torch.where(gone, torch.full_like(s["sarr"], INF), s["sarr"]))

    def _submit_dl(self, new, cap):
        """Queue this step's new commands as DL messages of cmd_bytes, sent at the next step boundary. A command
        the DL queue cannot take is dropped (cmd_dropped)."""
        s = self.state
        room = self.engine.net.dl.q.count() < self.engine.net.dl.q.F
        acc = new & room
        refused = new & ~room
        if refused.any():
            s["scap"].copy_(torch.where(refused[..., None] & (s["scap"] == cap[..., None]),
                                        torch.full_like(s["scap"], -1), s["scap"]))
        nb = torch.where(acc, torch.full_like(cap, self.cfg.cmd_bytes, dtype=torch.float32),
                         torch.zeros_like(cap, dtype=torch.float32))
        self._pending_dl = (nb, torch.where(acc, cap + 1, torch.zeros_like(cap)))
        self._dl_refused = refused
