"""Slot-level trace of selected robots of the NR engine (level "L2", reference backend), read-only.

    from isaac_net.core import NRConfig, make_engine
    from isaac_net.core.trace import SlotTrace
    net = make_engine("L2", E, R, "cpu", NRConfig(dl=True))
    trace = SlotTrace.attach(net, env=0, robots=[0, 3])
    ...                                            # submit / step as usual
    trace.to_frame()                               # one row per event (pandas DataFrame, or a list of dicts)
    trace.to_frame("samples")                      # one row per (data slot, robot): SINR, MCS, queue bytes
    trace.summary()                                # per robot: delay mean / p95, retransmissions per TB, ...
    trace.save("run.parquet")                      # Parquet with pyarrow, CSV otherwise (docs/trace.md)

What it reads, and where (nothing here changes the engine's state or draws a random number):

  MacLink.slot_hook   after decoding in every slot() call (one per data slot, one per mini-slot occasion): the TB
                      decisions (new / retransmission, HARQ process, transmission number, MCS, TBS, RBGs, PRBs, the
                      stream byte range of the TB) and the decode result (ACK / NACK, HARQ exhaustion)
  SlotTap observer    the SINR the MAC decodes with, per RBG, after the inter-cell interference and any user hook
                      (core/slot_tap.py chains it with the energy and background taps), and the SR / TPC state
                      between the slot's grant step (_pre_slot) and its decode
  ul.sr_step          SR transmissions (instance wrapper, as NREngine's traffic gate wraps it)
  dl.cqi_report       DL CQI reports (the gNB's per-RBG SINR estimate after the report)
  link.end_step       per control step: every queued frame (arrival, byte range) and the delivered / timed-out /
                      dropped masks before compaction, the same masks and fin times the step dict is built from
  ul.reset            partial and full resets (episode boundaries)

Every hook gathers the traced rows (a fixed index list, built once) into one small float64 tensor and appends it to
a list on the engine's device; no hook calls .item() or copies to the host. The records are copied to the host in one
transfer every `flush_steps` control steps (and when a table is read) and turned into events there. Cost per
control step: about 25 gathers of K rows per data slot plus one stack, independent of E and R.

Backends: the reference (= eager) backend only. The graph backend replays a captured CUDA graph, so Python hooks run
once at capture and never again, and the triton backend runs every slot of a step in one fused kernel with no
per-slot hook; both are refused. The reference backend is bitwise equal to graph, so a trace taken on it explains
what graph computes with the same config and seed.
"""
from __future__ import annotations

import csv
import json
import math
import os
from collections import defaultdict

import torch

from .queues import env_mask

# event types (column "event"); the order breaks ties at equal time
EVENTS = ("reset", "arrival", "sr", "sr_grant", "cqi", "grant", "tb_new", "tb_retx", "ack", "nack", "rlc_retx",
          "harq_drop", "tpc", "handover", "rlf", "rlf_end", "access", "rach_preamble", "delivered", "timeout",
          "dropped")
_ORDER = {e: i for i, e in enumerate(EVENTS)}
EVENT_COLS = ("run", "episode", "step", "g", "t_ms", "env", "robot", "dir", "event", "frame", "fidx", "pid", "ntx",
              "mcs", "n_rbg", "n_prb", "bytes", "tbs", "lo", "hi", "delay_ms", "value")
SAMPLE_COLS = ("run", "episode", "step", "g", "t_ms", "env", "robot", "dir", "tx", "sinr_db", "sinr_wb_db", "mcs",
               "n_rbg", "queue_bytes", "cell", "access")
ACCESS_NAMES = ("idle", "rach", "connected", "dormant")

# columns of a slot record (one row per traced robot)
_S = ("tx_new", "tx_rx", "ok", "exh", "p_tx", "ntx", "mcs", "tbs", "n_sb", "n_prb", "lo", "hi", "sinr_tx", "sinr_wb",
      "qbytes", "sr_grant", "tpc_iss", "tpc_cmd", "serv", "rlf", "acc")
_SI = {n: i for i, n in enumerate(_S)}
# rows of an end-of-step record [K, len(_F), F]
_F = ("cap", "start", "end", "fin", "off", "delivered", "timed", "dropped")
_FI = {n: i for i, n in enumerate(_F)}
_NAN = float("nan")


def _optional(name):
    try:
        return __import__(name)
    except ImportError:
        return None


class TraceTable:
    """Events and per-slot samples of a trace as host rows, with the timing constants needed to read them. SlotTrace
    is one; SlotTrace.load returns one from files written by save()."""

    def __init__(self, events=None, samples=None, meta=None):
        self._ev = list(events or [])
        self._sm = list(samples or [])
        self.meta = dict(meta or {})

    # ---------------------------------------------------------------- tables
    def _sync(self):
        pass

    def events(self):
        """Event rows (dicts with the keys EVENT_COLS), sorted by (run, time, event order)."""
        self._sync()
        return [{k: e[k] for k in EVENT_COLS} for e in self._ev]

    def samples(self):
        self._sync()
        return list(self._sm)

    def to_frame(self, kind="events"):
        """kind "events" or "samples" -> pandas DataFrame (columns EVENT_COLS / SAMPLE_COLS), or a list of dicts
        when pandas is not installed."""
        rows = self.events() if kind == "events" else self.samples()
        cols = EVENT_COLS if kind == "events" else SAMPLE_COLS
        pd = _optional("pandas")
        if pd is None:
            return rows
        return pd.DataFrame(rows, columns=list(cols))

    # ---------------------------------------------------------------- files
    def save(self, path, samples=True):
        """Write the events to path (".parquet" with pyarrow, else CSV; a ".parquet" path without pyarrow is written
        as CSV next to it), the samples to <stem>.samples.<ext> and the timing constants to <stem>.meta.json.
        Returns the list of files written."""
        stem, ext = os.path.splitext(str(path))
        ext = ext.lower()
        pd = _optional("pandas")
        if ext == ".parquet" and (pd is None or _optional("pyarrow") is None):
            ext = ".csv"
        if ext not in (".csv", ".parquet"):
            stem, ext = str(path), ".csv"
        out = []
        for kind, suffix in (("events", ""), ("samples", ".samples")):
            if kind == "samples" and not samples:
                continue
            p = f"{stem}{suffix}{ext}"
            if ext == ".parquet":
                self.to_frame(kind).to_parquet(p, index=False)
            else:
                cols = EVENT_COLS if kind == "events" else SAMPLE_COLS
                rows = self.events() if kind == "events" else self.samples()
                with open(p, "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(cols))
                    w.writeheader()
                    for r in rows:
                        w.writerow({k: ("" if (isinstance(v, float) and math.isnan(v)) else v) for k, v in r.items()})
            out.append(p)
        mp = f"{stem}.meta.json"
        with open(mp, "w") as f:
            json.dump(self.meta, f, indent=1)
        out.append(mp)
        return out

    @staticmethod
    def load(path):
        """A TraceTable from the files save() wrote (path = the events file)."""
        stem, ext = os.path.splitext(str(path))
        meta = {}
        if os.path.exists(f"{stem}.meta.json"):
            with open(f"{stem}.meta.json") as f:
                meta = json.load(f)

        def read(p, cols):
            if not os.path.exists(p):
                return []
            if ext.lower() == ".parquet":
                import pandas as pd
                return pd.read_parquet(p).to_dict("records")
            with open(p, newline="") as f:
                return [{k: _parse(k, v) for k, v in r.items()} for r in csv.DictReader(f)]
        return TraceTable(read(path, EVENT_COLS), read(f"{stem}.samples{ext}", SAMPLE_COLS), meta)

    # ---------------------------------------------------------------- frames and summary
    def frames(self, direction="ul"):
        """One dict per traced frame: id, env, robot, arrival / end time (ms), status (delivered / timeout / dropped /
        queued), delay_ms, byte range, and the TB transmissions that carried its bytes (list of event rows)."""
        ev = self.events()
        fr = {}
        for e in ev:
            if e["dir"] != direction or e["frame"] < 0:
                continue
            key = (e["run"], e["env"], e["robot"], e["frame"])
            if e["event"] == "arrival":
                fr[key] = {"frame": e["frame"], "run": e["run"], "episode": e["episode"], "env": e["env"],
                           "robot": e["robot"], "fidx": e["fidx"], "arrival_ms": e["t_ms"], "bytes": e["bytes"],
                           "lo": e["lo"], "hi": e["hi"], "status": "queued", "end_ms": _NAN, "delay_ms": _NAN,
                           "tbs": []}
            elif key in fr and e["event"] in ("delivered", "timeout", "dropped"):
                f = fr[key]
                f["status"], f["end_ms"], f["delay_ms"] = e["event"], e["t_ms"], e["delay_ms"]
                f["lo"], f["hi"] = e["lo"], e["hi"]          # the layout at resolution (a qos reorder may move it)
        by = defaultdict(list)
        for e in ev:
            if e["dir"] == direction and e["event"] in ("tb_new", "tb_retx") and e["hi"] > e["lo"]:
                by[(e["run"], e["episode"], e["env"], e["robot"])].append(e)
        for f in fr.values():
            f["tbs"] = [e for e in by[(f["run"], f["episode"], f["env"], f["robot"])]
                        if e["lo"] < f["hi"] and e["hi"] > f["lo"] and e["t_ms"] >= f["arrival_ms"] - 1e-9]
        return sorted(fr.values(), key=lambda f: (f["run"], f["arrival_ms"], f["frame"]))

    def summary(self, direction="ul"):
        """Per traced robot: frames, delivered / timeouts / drops, delay mean / p50 / p95 / max (ms), TBs new and
        retransmitted, retransmissions per TB, NACK ratio, HARQ exhaustions, grants per frame, TBs per delivered
        frame, SRs, mean SINR and MCS of the transmissions, handovers, RLFs, and the time spent IDLE / in RACH /
        DRX-dormant (sampled at the robot's data slots, scaled to the traced span). DataFrame, or a list of dicts."""
        ev, sm = self.events(), self.samples()
        fr = self.frames(direction)
        slot_ms = self.meta.get("slot_ms", 0.5)
        robots = sorted({(r["env"], r["robot"]) for r in sm + ev})
        rows = []
        for e_, r_ in robots:
            E_ = [x for x in ev if x["env"] == e_ and x["robot"] == r_]
            D_ = [x for x in E_ if x["dir"] == direction]
            S_ = [x for x in sm if x["env"] == e_ and x["robot"] == r_ and x["dir"] == direction]
            F_ = [f for f in fr if f["env"] == e_ and f["robot"] == r_]
            n = lambda k: sum(1 for x in D_ if x["event"] == k)        # noqa: E731
            dl = sorted(f["delay_ms"] for f in F_ if f["status"] == "delivered")
            tx = [x for x in S_ if x["tx"] > 0]
            new, retx, ack, nack = n("tb_new"), n("tb_retx"), n("ack"), n("nack")
            span = (max(x["t_ms"] for x in S_) - min(x["t_ms"] for x in S_) + slot_ms) if S_ else 0.0
            acc = [x["access"] for x in S_ if x["access"] >= 0]
            share = lambda s: (sum(1 for a in acc if a == s) / len(acc)) if acc else 0.0   # noqa: E731
            rows.append({
                "env": e_, "robot": r_, "dir": direction, "frames": len(F_), "delivered": len(dl),
                "timeouts": sum(f["status"] == "timeout" for f in F_), "dropped": sum(f["status"] == "dropped" for f in F_),
                "delay_mean_ms": (sum(dl) / len(dl)) if dl else _NAN, "delay_p50_ms": _q(dl, 0.5),
                "delay_p95_ms": _q(dl, 0.95), "delay_max_ms": dl[-1] if dl else _NAN,
                "tb_new": new, "tb_retx": retx, "retx_per_tb": retx / new if new else _NAN,
                "nack_ratio": nack / (ack + nack) if ack + nack else _NAN,
                "harq_exhausted": n("rlc_retx") + n("harq_drop"),
                "grants_per_frame": n("grant") / len(F_) if F_ else _NAN,
                "tbs_per_frame": (sum(len(f["tbs"]) for f in F_ if f["status"] == "delivered") / len(dl)) if dl else _NAN,
                "sr": n("sr"), "sinr_tx_mean_db": (sum(x["sinr_db"] for x in tx) / len(tx)) if tx else _NAN,
                "mcs_mean": (sum(x["mcs"] for x in tx) / len(tx)) if tx else _NAN,
                "handovers": sum(1 for x in E_ if x["event"] == "handover"),
                "rlf": sum(1 for x in E_ if x["event"] == "rlf"),
                "idle_ms": share(0) * span, "rach_ms": share(1) * span, "dormant_ms": share(3) * span,
                "traced_ms": span,
            })
        pd = _optional("pandas")
        return rows if pd is None else pd.DataFrame(rows)


def _q(xs, p):
    """Quantile p of a sorted list (linear interpolation, as numpy's default); NaN when empty."""
    if not xs:
        return _NAN
    k = (len(xs) - 1) * p
    lo, hi = int(math.floor(k)), int(math.ceil(k))
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


_INT_COLS = {"run", "episode", "step", "g", "env", "robot", "frame", "fidx", "pid", "ntx", "mcs", "n_rbg", "bytes",
             "tbs", "lo", "hi", "tx", "cell", "access"}


def _parse(k, v):
    if k in ("dir", "event"):
        return v
    if v == "":
        return _NAN
    x = float(v)
    return int(x) if k in _INT_COLS and x.is_integer() else x


class SlotTrace(TraceTable):
    """Slot-level trace of selected (env, robot) pairs of an L2 engine. Build it with SlotTrace.attach."""

    @classmethod
    def attach(cls, engine, env=0, robots=None, *, pairs=None, max_events=1_000_000, flush_steps=32):
        """Trace robots `robots` (default: all) of env `env` (an int or a list of envs), or the explicit (env, robot)
        `pairs`, of `engine` (an L2 engine from make_engine, or any wrapper of one: background, edge, energy).
        Robot indices are the engine's rows, so with background users rows R... are the ghosts. Recording stops
        (and `truncated` is set) once max_events events or samples are stored."""
        from .energy import find_nr_engine
        nr = find_nr_engine(engine)
        if nr is None:
            raise ValueError("SlotTrace needs the configurable NR engine: make_engine('L2', ...) (or a wrapper of one);"
                             f" got {type(engine).__name__}")
        backend = getattr(type(nr), "backend", "reference")
        if backend == "triton":
            raise ValueError("SlotTrace cannot trace the triton backend: its fused kernel runs every slot of a control "
                             "step in one launch, so there is no per-slot Python hook to read the MAC from. Build the "
                             "engine with backend='reference' (same model; graph is bitwise equal to it).")
        if backend != "reference":
            raise ValueError(f"SlotTrace needs backend='reference', not {backend!r}: the graph backend replays a "
                             "captured CUDA graph, so the trace hooks would run only once, at capture. The reference "
                             "backend is bitwise equal to graph with the same config and seed.")
        if pairs is None:
            envs = [env] if isinstance(env, int) else list(env)
            rows = list(range(nr.R)) if robots is None else ([robots] if isinstance(robots, int) else list(robots))
            pairs = [(int(e), int(r)) for e in envs for r in rows]
        pairs = [(int(e), int(r)) for e, r in pairs]
        for e, r in pairs:
            if not (0 <= e < nr.E and 0 <= r < nr.R):
                raise IndexError(f"(env, robot) = ({e}, {r}) outside [0, {nr.E}) x [0, {nr.R})")
        if len(set(pairs)) != len(pairs) or not pairs:
            raise ValueError("pairs must be distinct and non-empty")
        t = cls.__new__(cls)
        t._init(nr, pairs, int(max_events), int(flush_steps))
        return t

    # ---------------------------------------------------------------- construction
    def _init(self, nr, pairs, max_events, flush_steps):
        TraceTable.__init__(self)
        self.nr, self.net = nr, nr.net
        net, cfg = self.net, nr.config
        self.pairs = pairs
        self.K = len(pairs)
        d = net.dev
        self._ie = torch.tensor([p[0] for p in pairs], dtype=torch.long, device=d)
        self._ir = torch.tensor([p[1] for p in pairs], dtype=torch.long, device=d)
        self.max_events = max_events
        self.flush_steps = max(1, flush_steps)
        self.truncated = False
        self.N = cfg.slots_per_step
        self.slot_ms = cfg.slot_ms
        self.step_ms = cfg.control_step_ms
        self.meta = {"slot_ms": self.slot_ms, "step_ms": self.step_ms, "slots_per_step": self.N,
                     "n_harq": cfg.n_harq, "n_cells": net.C, "dl": net.dl is not None, "pairs": pairs,
                     "harq_fail": cfg.harq_fail}
        self._pending = []                 # device records in call order
        self._steps_pending = 0
        self._mid = {}
        self._active = True
        # host state of the event derivation
        self.run = 0
        self._episode = [0] * self.K
        self._frames = {}                  # (k, dir, run, episode, frame key) -> frame id
        self._frame_rng = {}               # frame id -> (lo, hi)
        self._next_fid = 0
        self._last = [{} for _ in range(self.K)]   # last serving cell / rlf / access state per traced robot
        self._seq = 0
        ul = net.ul
        self._sr_last = self._rows(ul.sr_t).clone()
        self._wrapped = []
        from .slot_tap import SlotTap
        self.tap = SlotTap.of(nr)
        self.tap.add_observer(self._on_sinr)
        for link in (ul, net.dl):
            if link is None:
                continue
            self._chain_slot_hook(link)
            self._wrap(link, "end_step", self._end_step_wrapper(link))
        self._wrap(ul, "sr_step", self._sr_wrapper(ul))
        self._wrap(ul, "reset", self._reset_wrapper(ul))
        if net.dl is not None:
            self._wrap(net.dl, "cqi_report", self._cqi_wrapper(net.dl))

    def _wrap(self, obj, name, make):
        orig = getattr(obj, name)
        fn = make(orig)
        setattr(obj, name, fn)
        self._wrapped.append((obj, name, orig, fn))

    def _chain_slot_hook(self, link):
        prev = link.slot_hook

        def hook(lk, info):
            if prev is not None:
                prev(lk, info)
            self._on_slot(lk, info)
        link.slot_hook = hook
        self._wrapped.append((link, "slot_hook", prev, hook))

    def detach(self):
        """Remove the hooks (where nothing was chained on top of them since) and stop recording."""
        self._sync()
        self._active = False
        self.tap.remove_observer(self._on_sinr)
        for obj, name, orig, fn in reversed(self._wrapped):
            if obj.__dict__.get(name) is fn:
                setattr(obj, name, orig)
        self._wrapped = []

    # ---------------------------------------------------------------- device side (hooks)
    def _rows(self, x):
        return x[self._ie, self._ir]

    def _T(self):
        return int(self.nr.T)

    def _on_sinr(self, link, g, won, n_prb, sinr):
        if not self._active:
            return
        w = self._rows(won).float()
        a = self._rows(sinr)
        n = w.sum(-1)
        nan = torch.full_like(n, _NAN)
        wb = a.mean(-1)
        if link.dir == "ul":       # a UL SINR depends on the PSD of the robot's own grant: none without a grant
            wb = torch.where(n > 0, wb, nan)
        mid = {"sinr_tx": torch.where(n > 0, (a * w).sum(-1) / n.clamp(min=1.0), nan), "sinr_wb": wb}
        if link.dir == "ul":
            mid["sr_t"] = self._rows(link.sr_t)
            if getattr(link, "_tpc", False):
                mid["tpc_at"] = self._rows(link.tpc_at)
        self._mid[link.dir] = mid

    def _ack_rows(self, link):
        """ack_ptr of the traced rows (MacLink.ack_ptr on K rows)."""
        hs, hl, hh = self._rows(link.h_state), self._rows(link.h_lo), self._rows(link.h_hi)
        fl, sent = self._rows(link.floor), self._rows(link.sent)
        own = (hs == 1) & (hh > fl[..., None])
        lo = torch.where(own, torch.maximum(hl, fl[..., None]), torch.full_like(hl, 2 ** 62))
        return torch.minimum(sent, lo.min(-1).values)

    def _state_rows(self, g):
        """(serving cell, RLF active, access state) of the traced rows at slot g."""
        net = self.net
        z = torch.zeros(self.K, dtype=torch.float64, device=net.dev)
        serv = self._rows(net.assoc.serv).double() if net.C > 1 else z
        rlf = self._rows(net.assoc.rlf_active).double() if (net.C > 1 and net.assoc.rlf) else z
        acc = self.nr.__dict__.get("access")
        if acc is None:
            st = z - 1.0
        else:
            st = acc.st
            if acc.rach:
                st = torch.where((st == 1) & (acc.conn_at <= g), torch.full_like(st, 2), st)
            if acc.drx:
                st = torch.where((st == 2) & ~acc._awake(g), torch.full_like(st, 3), st)
            st = self._rows(st).double()
        return serv, rlf, st

    def _on_slot(self, link, info):
        if not self._active:
            return
        R_ = self._rows
        g = info["g"]
        mid = self._mid.pop(link.dir, None)
        nan = torch.full((self.K,), _NAN, dtype=torch.float64, device=self.net.dev)
        zero = torch.zeros(self.K, dtype=torch.float64, device=self.net.dev)
        sr_grant, tpc_iss, tpc_cmd = zero, zero, zero
        if link.dir == "ul":
            if mid is not None and "sr_t" in mid:
                sr_grant = ((self._sr_last >= 0) & (mid["sr_t"] < 0)).double()
            self._sr_last = R_(link.sr_t)
            if getattr(link, "_tpc", False) and mid is not None and "tpc_at" in mid:
                at = R_(link.tpc_at)
                tpc_iss = ((at != mid["tpc_at"]) & (at >= 0)).double()
                tpc_cmd = R_(link.tpc_cmd).double()
        serv, rlf, acc = self._state_rows(g)
        cols = [R_(info[k]) for k in ("tx_new", "tx_rx", "ok", "exh", "p_tx", "ntx", "mcs", "tbs", "n_sb", "n_prb",
                                      "lo", "hi")]
        cols += [nan if mid is None else mid["sinr_tx"], nan if mid is None else mid["sinr_wb"],
                 (R_(link.q.enq) - self._ack_rows(link)).clamp(min=0), sr_grant, tpc_iss, tpc_cmd, serv, rlf, acc]
        rec = torch.stack([c.to(torch.float64) for c in cols], -1)
        self._pending.append(("slot", link.dir, int(g), self._T(), rec))

    def _sr_wrapper(self, ul):
        def make(orig):
            def sr_step(g):
                r = orig(g)
                if self._active:
                    s = self._rows(ul.sr_t)
                    sent = ((s == g) & (self._sr_last < 0)).double()
                    self._sr_last = s
                    self._pending.append(("sr", "ul", int(g), self._T(), sent[:, None]))
                return r
            return sr_step
        return make

    def _cqi_wrapper(self, dl):
        def make(orig):
            def cqi_report(sinr_ref, gain_now):
                r = orig(sinr_ref, gain_now)
                if self._active:
                    est = self._rows(sinr_ref + dl.csi).mean(-1).double()
                    g = self.net.last_g if self.nr.config.fading else None
                    self._pending.append(("cqi", "dl", g, self._T(), est[:, None]))
                return r
            return cqi_report
        return make

    def _reset_wrapper(self, ul):
        def make(orig):
            def reset(env_ids=None):
                m = None
                if self._active:
                    m = env_mask(self.nr.E, env_ids, self.net.dev)[self._ie].double()
                r = orig(env_ids)
                if self._active:
                    self._sr_last = self._rows(ul.sr_t)
                    self._pending.append(("reset", "ul", None, self._T(), m[:, None], env_ids is None))
                return r
            return reset
        return make

    def _end_step_wrapper(self, link):
        def make(orig):
            def end_step(t, timeout):
                res = orig(t, timeout)
                if self._active:
                    q = link.q
                    off = getattr(q, "off", None)
                    rows = [q.cap, q.start, q.end, q.fin, torch.zeros_like(q.fin) if off is None else off] + list(res)
                    rec = torch.stack([self._rows(x).to(torch.float64) for x in rows], 1)      # [K, 8, F]
                    T = int(t)
                    self._pending.append(("end", link.dir, None, T, rec))
                    if link.dir == "ul":
                        acc = self.nr.__dict__.get("access")
                        if acc is not None and acc.rach:
                            self._pending.append(("rach", "ul", None, T, self._rows(acc.ra_step).double()[:, None]))
                        self._steps_pending += 1
                        self.tap.install()        # a user set_sinr_hook since the last step: chain the observer again
                        if self._steps_pending >= self.flush_steps:
                            self._sync()
                return res
            return end_step
        return make

    # ---------------------------------------------------------------- host side
    def _sync(self):
        """Copy the pending device records to the host (one transfer) and turn them into events and samples."""
        if not self._pending:
            return
        pend, self._pending, self._steps_pending = self._pending, [], 0
        tens = [p[4] for p in pend]
        flat = torch.cat([x.reshape(-1) for x in tens]).cpu().tolist()
        pos = 0
        next_g = None
        # CQI records without a slot (fading off): the slot of the next record of the same step
        gs = [p[2] for p in pend]
        for i in range(len(pend) - 1, -1, -1):
            if gs[i] is None and pend[i][0] == "cqi":
                gs[i] = next_g if next_g is not None else pend[i][3] * self.N
            elif gs[i] is not None:
                next_g = gs[i]
        for i, p in enumerate(pend):
            n = tens[i].numel()
            vals = flat[pos:pos + n]
            pos += n
            if self.truncated:
                continue
            kind, d, T = p[0], p[1], p[3]
            shp = tuple(tens[i].shape)
            if kind == "slot":
                self._host_slot(d, gs[i], T, [vals[k * shp[1]:(k + 1) * shp[1]] for k in range(self.K)])
            elif kind == "end":
                F = shp[2]
                self._host_end(d, T, [[vals[(k * shp[1] + j) * F:(k * shp[1] + j + 1) * F] for j in range(shp[1])]
                                      for k in range(self.K)])
            elif kind == "sr":
                for k in range(self.K):
                    if vals[k] > 0:
                        self._emit(k, "sr", "ul", gs[i], T)
            elif kind == "cqi":
                for k in range(self.K):
                    self._emit(k, "cqi", "dl", gs[i], T, value=vals[k])
            elif kind == "rach":
                for k in range(self.K):
                    if vals[k] > 0:
                        self._emit(k, "rach_preamble", "ul", None, T, value=vals[k], t_ms=(T + 1) * self.step_ms)
            elif kind == "reset":
                if p[5]:
                    self.run += 1
                    self._episode = [0] * self.K
                for k in range(self.K):
                    if vals[k] > 0:
                        if not p[5]:
                            self._episode[k] += 1
                        self._last[k] = {}
                        self._emit(k, "reset", "ul", None, T, t_ms=T * self.step_ms)
            if len(self._ev) + len(self._sm) > self.max_events:
                self.truncated = True
                self._active = False
        self._ev.sort(key=lambda e: (e["run"], e["t_ms"], e["_seq"]))
        self._sm.sort(key=lambda s: (s["run"], s["t_ms"]))

    def _emit(self, k, event, d, g, T, t_ms=None, **kw):
        e, r = self.pairs[k]
        row = {"run": self.run, "episode": self._episode[k], "step": T, "g": -1 if g is None else int(g),
               "t_ms": (g * self.slot_ms if t_ms is None else t_ms), "env": e, "robot": r, "dir": d, "event": event,
               "frame": -1, "fidx": -1, "pid": -1, "ntx": -1, "mcs": -1, "n_rbg": -1, "n_prb": _NAN, "bytes": -1,
               "tbs": -1, "lo": -1, "hi": -1, "delay_ms": _NAN, "value": _NAN}
        row.update(kw)
        self._seq += 1
        row["_seq"] = (_ORDER[event], self._seq)
        self._ev.append(row)

    def _host_slot(self, d, g, T, rows):
        for k, v in enumerate(rows):
            S = dict(zip(_S, v))
            tx = 1 if S["tx_new"] > 0 else (2 if S["tx_rx"] > 0 else 0)
            e, r = self.pairs[k]
            self._sm.append({"run": self.run, "episode": self._episode[k], "step": T, "g": g, "t_ms": g * self.slot_ms,
                             "env": e, "robot": r, "dir": d, "tx": tx,
                             "sinr_db": S["sinr_tx"] if tx else S["sinr_wb"], "sinr_wb_db": S["sinr_wb"],
                             "mcs": int(S["mcs"]) if tx else -1, "n_rbg": int(S["n_sb"]) if tx else 0,
                             "queue_bytes": int(S["qbytes"]), "cell": int(S["serv"]), "access": int(S["acc"])})
            if S["sr_grant"] > 0:
                self._emit(k, "sr_grant", d, g, T)
            if tx:
                pid, ntx, mcs = int(S["p_tx"]), int(S["ntx"]), int(S["mcs"])
                lo, hi = int(S["lo"]), int(S["hi"])
                self._emit(k, "grant", d, g, T, pid=pid, ntx=ntx, mcs=mcs, n_rbg=int(S["n_sb"]), n_prb=S["n_prb"])
                self._emit(k, "tb_new" if tx == 1 else "tb_retx", d, g, T, pid=pid, ntx=ntx, mcs=mcs,
                           n_rbg=int(S["n_sb"]), n_prb=S["n_prb"], tbs=int(S["tbs"]), bytes=hi - lo, lo=lo, hi=hi,
                           value=S["sinr_tx"])
                self._emit(k, "ack" if S["ok"] > 0 else "nack", d, g, T, pid=pid, ntx=ntx, mcs=mcs, lo=lo, hi=hi)
                if S["exh"] > 0:
                    self._emit(k, "rlc_retx" if self.meta["harq_fail"] == "rlc_am" else "harq_drop", d, g, T, pid=pid,
                               ntx=ntx, lo=lo, hi=hi)
            if S["tpc_iss"] > 0:
                self._emit(k, "tpc", d, g, T, value=S["tpc_cmd"])
            last = self._last[k]
            for name, val in (("cell", int(S["serv"])), ("rlf", int(S["rlf"])), ("access", int(S["acc"]))):
                prev = last.get(name)
                last[name] = val
                if prev is None or prev == val:
                    continue
                if name == "cell":
                    self._emit(k, "handover", d, g, T, value=val)
                elif name == "rlf":
                    self._emit(k, "rlf" if val else "rlf_end", d, g, T)
                else:
                    self._emit(k, "access", d, g, T, value=val)

    def _host_end(self, d, T, rows):
        sm = self.step_ms
        for k, v in enumerate(rows):
            Fv = dict(zip(_F, v))
            seen = defaultdict(int)
            for f in range(len(Fv["cap"])):
                cap = Fv["cap"][f]
                if cap < 0:
                    continue
                cap, off, lo, hi = int(cap), Fv["off"][f], int(Fv["start"][f]), int(Fv["end"][f])
                sig = (cap, round(off * self.N), hi - lo)
                key = (k, d, self.run, self._episode[k], sig, seen[sig])
                seen[sig] += 1
                fid = self._frames.get(key)
                if fid is None:
                    fid = self._next_fid
                    self._next_fid += 1
                    self._frames[key] = fid
                    self._emit(k, "arrival", d, None, cap, t_ms=(cap + off) * sm, frame=fid, fidx=f, bytes=hi - lo,
                               lo=lo, hi=hi)
                for name in ("delivered", "timed", "dropped"):
                    if Fv[name][f] > 0:
                        fin = Fv["fin"][f]
                        delay = (fin - cap - off) * sm if name == "delivered" else _NAN
                        t_ms = fin * sm if name == "delivered" else (T + 1) * sm
                        self._emit(k, "timeout" if name == "timed" else name, d, None, T, t_ms=t_ms, frame=fid, fidx=f,
                                   bytes=hi - lo, lo=lo, hi=hi, delay_ms=delay)
                        break
