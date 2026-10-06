"""Per-step KPI recorder on top of any engine: accumulate on the device, write Parquet (or CSV) at flush time.

    from isaac_net import make_engine, record
    net = record(make_engine("L2", E, R, dev, cfg), "runs/l2", every=10, label="L2 default")
    for _ in range(T):
        net.submit(None, send)
        out = net.step(None, pos)          # the engine's step dict, unchanged
    net.close()                            # flushes what is left; also a context manager
    # later: isaac_net.viz.cdf.plot_delay_cdf(["runs/l2"]) or isaac_net.viz.report.from_records(...)

RecorderLoop wraps an engine (any make_engine level, or EdgeLoop / BackgroundLoop / EnergyLoop around one) like
EnergyLoop does: same API, same step dict. It is an explicit wrapper (no NRConfig field), so make_engine's default
wrapper chain is unchanged. Put it outermost to see every key the inner wrappers add (energy_*, bg_*).

What it does per control step: reads the step dict and the engine's device counters, adds into fixed-shape device
accumulators and never syncs with the host. Every `every` steps it closes a window: one row per env goes into a
device buffer of `flush_every` windows. Only a flush (buffer full, flush() or close()) copies to the host and writes.
It draws no random numbers and writes into no engine tensor, so the engine's outputs are bitwise those of the bare
engine (tests/test_record.py).

Tables (out_dir/<table>/part-NNNNN.parquet, or out_dir/<table>.csv without pyarrow or with format="csv"), plus
out_dir/meta.json (schema, sizes, step length, columns). Every table has the constant columns label and level.

  steps       one row per (window, env). step = number of control steps since the recorder was made, at the window's
              end; episode = the env's resets since recording began (a reset before its first step does not count);
              t = the env's clock at the end; steps = window length. Sums over the window: sent (accepted
              messages), offered_bytes (accepted bytes, plus traffic-model bytes on L2), delivered, delivered_bytes,
              lost (timed out or dropped), harq_tx / harq_retx (UL transport blocks, all / retransmissions), energy_j
              (summed over robots). delay_mean_ms, delay_p50_ms, delay_p95_ms over the messages delivered in the
              window (p50 / p95 from a log-spaced histogram, about 6% bin width). aoi_mean_s, aoi_p95_s: age of
              information per robot and step (the freshest delivered capture; the state at reset counts as known).
              Means over the window's steps: queue_bytes, queue_len (summed over robots), sinr_mean_db (over robots),
              access_idle / access_rach / access_connected / access_dormant (share of robots in each access state),
              los_frac, blocked_frac (serving link), bg_util (background PRB share, mean over cells). prb_util,
              dl_prb_util: PRB-slots granted / available on L2 (counters of the NR MAC). harq_bler: first-transmission
              BLER. battery_frac: mean over robots at the window end. A column the engine does not produce is NaN.
  cells       one row per (window, env, cell) with per_cell: robots served at the end, delivered, prb_util (the
              robots' PRB-slots in that cell over the cell's available PRB-slots; L2 only), bg_util, bg_n.
  robots      one row per (window, env, robot) for the envs in raw_envs: position x, y (when step gets poses),
              sinr_db, serving_cell, queue_bytes, queue_len, access_state, los, blocked, battery_frac at the end;
              delivered, lost, delay_mean_ms, energy_j over the window; aoi_s at the end.
  delay_hist, aoi_hist   sparse per-env histograms over each flush period: step (the flush's last step), env, bin,
              lo_ms, hi_ms, count. They give exact-to-the-bin delay and AoI CDFs (isaac_net.viz.cdf).

TensorBoard (tensorboard="logdir", needs the tensorboard package) and Weights & Biases (wandb=True for the active
run, or a dict of wandb.init arguments) get the env-mean of LOG_COLUMNS for every window whose end step is a
multiple of log_every, written at flush time.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, field

import numpy as np
import torch

SCHEMA = "isaac-net-record/1"

# log-spaced histogram (the bins of isaac_net.bench.metrics): bin 0 = [0, 0.1 ms), then NB - 1 bins up to 60 s
NB = 240
LO_MS, HI_MS = 0.1, 60_000.0
_LOG_LO, _LOG_HI = math.log10(LO_MS), math.log10(HI_MS)

STEP_COLUMNS = (
    "step", "env", "episode", "t", "steps",
    "sent", "offered_bytes", "delivered", "delivered_bytes", "lost",
    "delay_mean_ms", "delay_p50_ms", "delay_p95_ms", "aoi_mean_s", "aoi_p95_s",
    "queue_bytes", "queue_len", "sinr_mean_db",
    "prb_util", "dl_prb_util", "harq_tx", "harq_retx", "harq_bler",
    "energy_j", "battery_frac",
    "access_idle", "access_rach", "access_connected", "access_dormant",
    "los_frac", "blocked_frac", "bg_util")
CELL_COLUMNS = ("step", "env", "cell", "robots", "delivered", "prb_util", "bg_util", "bg_n")
ROBOT_COLUMNS = ("step", "env", "robot", "x", "y", "sinr_db", "serving_cell", "queue_bytes", "queue_len",
                 "access_state", "los", "blocked", "battery_frac", "delivered", "lost", "delay_mean_ms", "energy_j",
                 "aoi_s")
HIST_COLUMNS = ("step", "env", "bin", "lo_ms", "hi_ms", "count")
INT_COLUMNS = {"step", "env", "episode", "t", "steps", "cell", "robot", "robots", "bin", "count", "bg_n",
               "serving_cell", "access_state"}
LOG_COLUMNS = ("delivered", "lost", "delay_mean_ms", "delay_p50_ms", "delay_p95_ms", "aoi_p95_s", "queue_bytes",
               "prb_util", "harq_bler", "energy_j")
TABLES = ("steps", "cells", "robots", "delay_hist", "aoi_hist")


@dataclass
class RecordConfig:
    """What RecorderLoop records and where (core/record.py).

    out_dir      directory of the tables and meta.json (created)
    every        control steps per row (window length)
    flush_every  windows kept on the device before a host copy and write
    format       "auto" (Parquet with pyarrow, else CSV), "parquet" or "csv"
    per_cell     write the cells table
    raw_envs     env ids whose robots get a row per window in the robots table
    hist         write the delay_hist / aoi_hist tables
    label        free-form name of the run (a column of every table; used by isaac_net.viz to group runs)
    level        the level name for the level column; None = the wrapped engine's class name
    tensorboard  log directory for torch.utils.tensorboard, or None
    wandb        False, True (log to the active wandb run, init one if none) or a dict of wandb.init kwargs
    log_every    scalar logging cadence in control steps (a window is logged when its end step is a multiple)
    """
    out_dir: str = "records"
    every: int = 1
    flush_every: int = 256
    format: str = "auto"
    per_cell: bool = True
    raw_envs: tuple = ()
    hist: bool = True
    label: str = ""
    level: str | None = None
    tensorboard: str | None = None
    wandb: object = False
    log_every: int = 1
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        if int(self.every) < 1 or int(self.flush_every) < 1 or int(self.log_every) < 1:
            raise ValueError("every, flush_every and log_every must be >= 1")
        if self.format not in ("auto", "parquet", "csv"):
            raise ValueError(f"format must be 'auto', 'parquet' or 'csv', not {self.format!r}")
        self.raw_envs = tuple(int(e) for e in self.raw_envs)


def _bins(ms: torch.Tensor) -> torch.Tensor:
    d = ms.double().clamp(min=0.0)
    k = ((torch.log10(d.clamp(min=LO_MS)) - _LOG_LO) / (_LOG_HI - _LOG_LO) * (NB - 1)).floor().long() + 1
    return torch.where(d < LO_MS, torch.zeros_like(k), k).clamp(0, NB - 1)


def bin_edges_ms() -> np.ndarray:
    """[NB + 1] histogram edges in ms (0, then log-spaced 0.1 ms ... 60 s)."""
    return np.concatenate([[0.0], np.logspace(_LOG_LO, _LOG_HI, NB)])


def _hist_quantile(h: torch.Tensor, q: float, edges: torch.Tensor) -> torch.Tensor:
    """Quantile q of each row of h [E, NB], NaN for empty rows (geometric interpolation inside a log bin)."""
    h = h.double()
    tot = h.sum(-1, keepdim=True)
    c = h.cumsum(-1)
    target = q * tot
    k = torch.searchsorted(c.contiguous(), target.contiguous()).clamp(max=NB - 1)
    below = torch.where(k > 0, c.gather(-1, (k - 1).clamp(min=0)), torch.zeros_like(target))
    frac = ((target - below) / h.gather(-1, k).clamp(min=1e-12)).clamp(0.0, 1.0)
    lo, hi = edges[k], edges[k + 1]
    val = torch.where(k == 0, lo + frac * (hi - lo), lo * (hi / lo.clamp(min=1e-12)) ** frac)
    return torch.where(tot > 0, val, torch.full_like(val, math.nan)).squeeze(-1)


def _find_nr(engine):
    from .energy import find_nr_engine
    return find_nr_engine(engine)


class _Writer:
    """Appends column dicts to out_dir/<name>/part-NNNNN.parquet or out_dir/<name>.csv."""

    def __init__(self, out_dir, name, columns, fmt):
        self.dir, self.name, self.columns, self.fmt = out_dir, name, tuple(columns), fmt
        self.part = 0

    def write(self, cols: dict):
        n = len(next(iter(cols.values())))
        if n == 0:
            return
        if self.fmt == "parquet":
            import pyarrow as pa
            import pyarrow.parquet as pq
            d = os.path.join(self.dir, self.name)
            os.makedirs(d, exist_ok=True)
            tab = pa.table({k: cols[k] for k in self.columns})
            pq.write_table(tab, os.path.join(d, f"part-{self.part:05d}.parquet"))
        else:
            import csv
            path = os.path.join(self.dir, self.name + ".csv")
            new = not os.path.exists(path) or self.part == 0
            with open(path, "w" if new else "a", newline="") as f:
                w = csv.writer(f)
                if new:
                    w.writerow(self.columns)
                arrs = [cols[k] for k in self.columns]
                for i in range(n):
                    w.writerow([_cell(a[i]) for a in arrs])
        self.part += 1


def _cell(v):
    if isinstance(v, (float, np.floating)):
        return "" if not math.isfinite(v) else repr(float(v))
    return v if isinstance(v, str) else int(v)


def _columns(names, mat: np.ndarray, const: dict) -> dict:
    """Column dict from a [n, len(names)] float64 matrix, ints cast, plus constant string columns."""
    out = {}
    for j, k in enumerate(names):
        v = mat[:, j]
        out[k] = np.nan_to_num(v, nan=-1).astype(np.int64) if k in INT_COLUMNS else v.astype(np.float64)
    n = mat.shape[0]
    for k, v in const.items():
        out[k] = np.array([v] * n, dtype=object)
    return out


class RecorderLoop:
    """Wrap `engine` with the per-step KPI recorder (module docstring). RecorderLoop(engine, RecordConfig(...)) or
    RecorderLoop(engine, out_dir=..., every=...)."""

    def __init__(self, engine, cfg: RecordConfig | None = None, **kw):
        cfg = cfg if cfg is not None else RecordConfig()
        if kw:
            cfg = RecordConfig(**{**asdict(cfg), **kw})
        self.engine, self.cfg = engine, cfg
        self.E, self.R = E, R = engine.E, engine.R
        self.dev = d = torch.device(getattr(engine, "dev", "cpu"))
        ncfg = engine.config
        self.step_ms = float(ncfg.control_step_ms)
        self.C = C = int(getattr(ncfg, "n_cells", 1) or 1)
        self.sizes = torch.tensor((0.0,) + tuple(float(s) for s in ncfg.msg_sizes), dtype=torch.float64, device=d)
        self.edges = torch.as_tensor(bin_edges_ms(), dtype=torch.float64, device=d)
        self.level = cfg.level or type(_find_nr(engine) or engine).__name__.replace("NREngine", "L2")
        fmt = cfg.format
        if fmt == "auto":
            try:
                import pyarrow  # noqa: F401
                import pyarrow.parquet  # noqa: F401
                fmt = "parquet"
            except ImportError:
                fmt = "csv"
        self.fmt = fmt
        os.makedirs(cfg.out_dir, exist_ok=True)
        self._writers = {"steps": _Writer(cfg.out_dir, "steps", STEP_COLUMNS + ("label", "level"), fmt),
                         "cells": _Writer(cfg.out_dir, "cells", CELL_COLUMNS + ("label", "level"), fmt),
                         "robots": _Writer(cfg.out_dir, "robots", ROBOT_COLUMNS + ("label", "level"), fmt),
                         "delay_hist": _Writer(cfg.out_dir, "delay_hist", HIST_COLUMNS + ("label", "level"), fmt),
                         "aoi_hist": _Writer(cfg.out_dir, "aoi_hist", HIST_COLUMNS + ("label", "level"), fmt)}
        # NR MAC counters (device tensors; read without a host sync) and the per-robot slot tap for per-cell PRBs
        nr = _find_nr(engine)
        net = getattr(nr, "__dict__", {}).get("net") if nr is not None else None
        self._links = {k: getattr(net, k, None) for k in ("ul", "dl")} if net is not None else {}
        self.tap = None
        if nr is not None and cfg.per_cell and self._links.get("ul") is not None:
            from .slot_tap import SlotTap
            self.tap = SlotTap.of(nr)
        raw = torch.tensor(cfg.raw_envs, dtype=torch.long, device=d)
        if raw.numel() and (int(raw.min()) < 0 or int(raw.max()) >= E):
            raise ValueError(f"raw_envs must be env ids in [0, {E})")
        self._raw = raw
        z = lambda *s, dt=torch.float64: torch.zeros(*s, dtype=dt, device=d)   # noqa: E731
        self.last_cap = z(E, R, dt=torch.long)
        self.episode = z(E, dt=torch.long)
        self._stepped = torch.zeros(E, dtype=torch.bool, device=d)        # env stepped since its last reset
        self._t = z(E, dt=torch.long)
        self._acc = {k: z(E) for k in STEP_COLUMNS}
        self._cnt = {k: z(E) for k in STEP_COLUMNS}         # steps that contributed to a mean column
        self._hd, self._ha = z(E, NB, dt=torch.long), z(E, NB, dt=torch.long)       # window histograms
        self._fd, self._fa = z(E, NB, dt=torch.long), z(E, NB, dt=torch.long)       # flush-period histograms
        self._cell = {k: z(E, C) for k in CELL_COLUMNS[3:]}
        self._cell_n = {k: z(E, C) for k in CELL_COLUMNS[3:]}
        nr_ = max(raw.numel(), 1)
        self._rob = {k: z(nr_, R) for k in ROBOT_COLUMNS[3:]}
        self._rob_n = {k: z(nr_, R) for k in ROBOT_COLUMNS[3:]}
        self._ctr_snap = self._ctr_now()
        self._ctr_acc = {k: torch.zeros_like(v) for k, v in self._ctr_snap.items()}
        F = int(cfg.flush_every)
        self._buf = z(F, E, len(STEP_COLUMNS))
        self._bufc = z(F, E, C, len(CELL_COLUMNS))
        self._bufr = z(F, nr_, R, len(ROBOT_COLUMNS))
        self._win_steps = []                     # host ints: end step of each buffered window
        self.n_steps = 0                         # control steps recorded (host counter)
        self._in_window = 0
        self._closed = False
        self._tb = self._wb = None
        if cfg.tensorboard:
            from torch.utils.tensorboard import SummaryWriter        # needs the tensorboard package
            self._tb = SummaryWriter(cfg.tensorboard)
        if cfg.wandb:
            import wandb
            if isinstance(cfg.wandb, dict):
                self._wb = wandb.init(**cfg.wandb)
            else:
                self._wb = wandb.run if wandb.run is not None else wandb.init()
        self._write_meta()

    # ------------------------------------------------------------------ passthroughs
    def __getattr__(self, name):
        if name in ("engine", "cfg", "tap"):
            raise AttributeError(name)
        return getattr(self.engine, name)

    @property
    def clock(self):
        return self.engine.clock

    @property
    def config(self):
        return self.engine.config

    def queued(self):
        return self.engine.queued()

    def set_sinr_hook(self, fn, direction="ul"):
        self.engine.set_sinr_hook(fn, direction)
        if self.tap is not None:
            self.tap.install()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    # ------------------------------------------------------------------ engine API
    def reset(self, env_ids=None):
        if env_ids is None:              # a full reset zeroes the MAC counters: bank the window's part first
            now = self._ctr_now()
            for k in now:
                self._ctr_acc[k] += now[k] - self._ctr_snap[k]
        self.engine.reset(env_ids)
        from .queues import env_mask
        m = env_mask(self.E, env_ids, self.dev)
        self.last_cap.masked_fill_(m[:, None], 0)
        self.episode.add_((m & self._stepped).long())          # a reset before the first step starts no episode
        self._stepped.masked_fill_(m, False)
        if env_ids is None:
            self._ctr_snap = self._ctr_now()

    def submit(self, t, requests, snr_db=None, **kw):
        acc = self.engine.submit(t, requests, snr_db, **kw)
        send = getattr(requests, "send", requests)
        if torch.is_tensor(acc) and torch.is_tensor(send):
            a = acc.bool()
            cls = send.long().clamp(0, self.sizes.numel() - 1)
            self._acc["sent"] += a.sum(-1).double()
            self._acc["offered_bytes"] += (self.sizes[cls] * a).sum(-1)
        return acc

    def add_frames(self, t, send, det, hid, snr_db):
        from .traffic import Requests
        self.submit(t, Requests(send, det, hid), snr_db)

    def step(self, t, x=None, cur_hid=None, **kw):
        if cur_hid is not None:
            raise NotImplementedError("RecorderLoop needs the dict form step(t, x): the legacy form "
                                      "step(t, x, cur_hid) returns only (newest, det_env)")
        if self.tap is not None:
            self.tap.begin()
        out = self.engine.step(t, x, **kw)
        self._observe(out, x)
        return out

    # ------------------------------------------------------------------ accumulation (device only, no host sync)
    def _ctr_now(self):
        res = {}
        for name, link in self._links.items():
            if link is None:
                continue
            res[f"{name}_prb"] = link.prb_used_env.double().clone()
            res[f"{name}_avail"] = link.ctr["prb_avail"].double().reshape(1).clone()
            res[f"{name}_rvtx"] = link.rv_tx.double().clone()
            res[f"{name}_rvfail"] = link.rv_fail.double().clone()
        return res

    def _mean(self, k, v):
        self._acc[k] += v
        self._cnt[k] += 1.0

    def _observe(self, out, x):
        E, R = self.E, self.R
        a = self._acc
        dlv = out["delivered"][:, :R].bool()
        lost = out["timed_out"][:, :R].bool()
        if "dropped" in out:
            lost = lost | out["dropped"][:, :R].bool()
        a["delivered"] += dlv.sum((-1, -2)).double()
        a["lost"] += lost.sum((-1, -2)).double()
        if "bytes" in out:
            fb = out["bytes"][:, :R].double()
        else:
            fb = self.sizes[out["cls"][:, :R].long().clamp(0, self.sizes.numel() - 1)]
        a["delivered_bytes"] += (fb * dlv).sum((-1, -2))
        if "gen_bytes" in out:
            a["offered_bytes"] += out["gen_bytes"][:, :R].double().sum(-1)
        d_ms = torch.where(dlv, torch.nan_to_num(out["delay"][:, :R].double(), nan=0.0),
                           torch.zeros((), dtype=torch.float64, device=self.dev)) * self.step_ms
        a["delay_mean_ms"] += d_ms.sum((-1, -2))
        b = torch.where(dlv, _bins(d_ms), torch.full_like(dlv, NB, dtype=torch.long)).reshape(E, -1)
        h = torch.zeros(E, NB + 1, dtype=torch.long, device=self.dev)
        h.scatter_add_(1, b, torch.ones_like(b))
        self._hd += h[:, :NB]
        # age of information: freshest delivered capture (state at reset = capture 0)
        newest = out["newest"][:, :R].long()
        self.last_cap = torch.maximum(self.last_cap, newest)
        tt = out["t"].long()
        self._t = tt.clone()
        aoi_ms = (tt[:, None] + 1 - self.last_cap).double() * self.step_ms
        self._mean("aoi_mean_s", aoi_ms.mean(-1) * 1e-3)
        ha = torch.zeros(E, NB, dtype=torch.long, device=self.dev)
        ha.scatter_add_(1, _bins(aoi_ms), torch.ones(E, R, dtype=torch.long, device=self.dev))
        self._ha += ha
        self._mean("queue_bytes", out["queue_bytes"][:, :R].double().sum(-1))
        self._mean("queue_len", out["queue_len"][:, :R].double().sum(-1))
        self._mean("sinr_mean_db", out["sinr_db"][:, :R].double().mean(-1))
        if "energy_j" in out:
            a["energy_j"] += out["energy_j"][:, :R].double().sum(-1)
            self._cnt["energy_j"] += 1.0
        if "access_state" in out:
            st = out["access_state"][:, :R]
            for i, k in enumerate(("access_idle", "access_rach", "access_connected", "access_dormant")):
                self._mean(k, (st == i).double().mean(-1))
        for k, col in (("los", "los_frac"), ("blocked", "blocked_frac")):
            if k in out:
                self._mean(col, out[k][:, :R].double().mean(-1))
        if "bg_util" in out:
            self._mean("bg_util", out["bg_util"].double().mean(-1))
        if "battery_frac" in out:
            self._battery = out["battery_frac"][:, :R].clone()
        serv = out.get("serving_cell")
        serv = serv[:, :R].long().clamp(0, self.C - 1) if serv is not None else None
        if self.cfg.per_cell:
            self._observe_cells(out, serv, dlv)
        if self._raw.numel():
            self._observe_robots(out, x, serv, dlv, lost, d_ms, aoi_ms)
        self._stepped.fill_(True)
        self.n_steps += 1
        self._in_window += 1
        if self._in_window >= self.cfg.every:
            self._close_window()

    def _observe_cells(self, out, serv, dlv):
        c, n = self._cell, self._cell_n
        E, C = self.E, self.C
        s = serv if serv is not None else torch.zeros(E, self.R, dtype=torch.long, device=self.dev)
        per = torch.zeros(E, C, dtype=torch.float64, device=self.dev)
        c["delivered"] += per.scatter_add(1, s, dlv.sum(-1).double())
        if self.tap is not None:
            c["prb_util"] += per.scatter_add(1, s, self.tap.ul_prb[:, :self.R].double())
            n["prb_util"] += 1.0
        if "bg_util" in out:
            c["bg_util"] += out["bg_util"].double()
            n["bg_util"] += 1.0
        self._serv_last = s
        self._bg_n_last = out.get("bg_n")

    def _observe_robots(self, out, x, serv, dlv, lost, d_ms, aoi_ms):
        idx = self._raw
        r, n = self._rob, self._rob_n
        g = lambda v: v[idx][:, :self.R].double()     # noqa: E731
        r["delivered"] += g(dlv.sum(-1))
        r["lost"] += g(lost.sum(-1))
        r["delay_mean_ms"] += g(d_ms.sum(-1))
        if "energy_j" in out:
            r["energy_j"] += g(out["energy_j"])
            n["energy_j"] += 1.0
        last = {"sinr_db": out["sinr_db"], "queue_bytes": out["queue_bytes"], "queue_len": out["queue_len"],
                "aoi_s": aoi_ms * 1e-3}
        if serv is not None:
            last["serving_cell"] = serv
        for k in ("access_state", "los", "blocked", "battery_frac"):
            if k in out:
                last[k] = out[k]
        if torch.is_tensor(x) and x.dim() == 3:
            last["x"], last["y"] = x[..., 0], x[..., 1]
        for k, v in last.items():
            r[k] = g(v)
            n[k] = torch.ones_like(n[k])

    def _close_window(self):
        E = self.E
        a, cnt = self._acc, self._cnt
        nan = torch.full((E,), math.nan, dtype=torch.float64, device=self.dev)
        ratio = lambda num, den: torch.where(den > 0, num / den.clamp(min=1e-12), nan)   # noqa: E731
        now = self._ctr_now()
        diff = {k: self._ctr_acc[k] + now[k] - self._ctr_snap[k] for k in now}
        self._ctr_snap = now
        for v in self._ctr_acc.values():
            v.zero_()
        cols = {"step": torch.full((E,), float(self.n_steps), dtype=torch.float64, device=self.dev),
                "env": torch.arange(E, dtype=torch.float64, device=self.dev),
                "episode": self.episode.double(), "t": self._t.double(),
                "steps": torch.full((E,), float(self._in_window), dtype=torch.float64, device=self.dev)}
        for k in ("sent", "offered_bytes", "delivered", "delivered_bytes", "lost"):
            cols[k] = a[k].clone()
        cols["delay_mean_ms"] = ratio(a["delay_mean_ms"], a["delivered"])
        cols["delay_p50_ms"] = _hist_quantile(self._hd, 0.5, self.edges)
        cols["delay_p95_ms"] = _hist_quantile(self._hd, 0.95, self.edges)
        cols["aoi_p95_s"] = _hist_quantile(self._ha, 0.95, self.edges) * 1e-3
        for k in ("aoi_mean_s", "queue_bytes", "queue_len", "sinr_mean_db", "access_idle", "access_rach",
                  "access_connected", "access_dormant", "los_frac", "blocked_frac", "bg_util"):
            cols[k] = ratio(a[k], cnt[k])
        cols["energy_j"] = torch.where(cnt["energy_j"] > 0, a["energy_j"], nan)
        bat = getattr(self, "_battery", None)
        cols["battery_frac"] = bat.double().mean(-1) if bat is not None else nan
        for name, col in (("ul", "prb_util"), ("dl", "dl_prb_util")):
            if f"{name}_prb" in diff:
                cols[col] = ratio(diff[f"{name}_prb"], (diff[f"{name}_avail"] / E).expand(E))
            else:
                cols[col] = nan
        if "ul_rvtx" in diff:
            tx, fail = diff["ul_rvtx"], diff["ul_rvfail"]
            cols["harq_tx"] = tx.sum(-1)
            cols["harq_retx"] = tx[:, 2:].sum(-1)
            cols["harq_bler"] = ratio(fail[:, 1], tx[:, 1])
        else:
            cols["harq_tx"] = cols["harq_retx"] = cols["harq_bler"] = nan
        w = len(self._win_steps)
        self._buf[w].copy_(torch.stack([cols[k] for k in STEP_COLUMNS], -1))
        self._fd += self._hd
        self._fa += self._ha
        if self.cfg.per_cell:
            self._close_cells(w, diff)
        if self._raw.numel():
            self._close_robots(w)
        self._win_steps.append(self.n_steps)
        for v in list(a.values()) + list(cnt.values()) + [self._hd, self._ha]:
            v.zero_()
        self._in_window = 0
        if len(self._win_steps) >= self.cfg.flush_every:
            self.flush()

    def _close_cells(self, w, diff):
        E, C = self.E, self.C
        c, n = self._cell, self._cell_n
        nan = torch.full((E, C), math.nan, dtype=torch.float64, device=self.dev)
        f = lambda v: torch.full((E, C), float(v), dtype=torch.float64, device=self.dev)    # noqa: E731
        serv = getattr(self, "_serv_last", None)
        robots = (torch.zeros(E, C, dtype=torch.float64, device=self.dev)
                  .scatter_add(1, serv, torch.ones_like(serv, dtype=torch.float64)) if serv is not None else nan)
        if self.tap is not None and "ul_avail" in diff:
            avail = (diff["ul_avail"] / (E * C)).expand(E, C)
            prb = torch.where(avail > 0, c["prb_util"] / avail.clamp(min=1e-12), nan)
        else:
            prb = nan
        bg = torch.where(n["bg_util"] > 0, c["bg_util"] / n["bg_util"].clamp(min=1.0), nan)
        bgn = self._bg_n_last.double() if getattr(self, "_bg_n_last", None) is not None else nan
        cols = {"step": f(self.n_steps), "env": torch.arange(E, dtype=torch.float64, device=self.dev)[:, None]
                .expand(E, C), "cell": torch.arange(C, dtype=torch.float64, device=self.dev)[None].expand(E, C),
                "robots": robots, "delivered": c["delivered"].clone(), "prb_util": prb, "bg_util": bg, "bg_n": bgn}
        self._bufc[w].copy_(torch.stack([cols[k] for k in CELL_COLUMNS], -1))
        for v in list(c.values()) + list(n.values()):
            v.zero_()

    def _close_robots(self, w):
        idx = self._raw
        r, n = self._rob, self._rob_n
        nr_, R = idx.numel(), self.R
        nan = torch.full((nr_, R), math.nan, dtype=torch.float64, device=self.dev)
        cols = {"step": torch.full((nr_, R), float(self.n_steps), dtype=torch.float64, device=self.dev),
                "env": idx.double()[:, None].expand(nr_, R),
                "robot": torch.arange(R, dtype=torch.float64, device=self.dev)[None].expand(nr_, R)}
        for k in ROBOT_COLUMNS[3:]:
            if k in ("delivered", "lost"):
                cols[k] = r[k].clone()
            elif k == "delay_mean_ms":
                cols[k] = torch.where(r["delivered"] > 0, r[k] / r["delivered"].clamp(min=1.0), nan)
            else:
                cols[k] = torch.where(n[k] > 0, r[k], nan)
        self._bufr[w, :nr_].copy_(torch.stack([cols[k] for k in ROBOT_COLUMNS], -1))
        for k in ("delivered", "lost", "delay_mean_ms", "energy_j"):
            r[k].zero_()
            n[k].zero_()

    # ------------------------------------------------------------------ host side
    def _const(self):
        return {"label": self.cfg.label, "level": self.level}

    def flush(self):
        """Copy the buffered windows to the host and write them (one host sync). Called automatically."""
        nw = len(self._win_steps)
        if nw == 0:
            return
        E, C = self.E, self.C
        const = self._const()
        steps = self._buf[:nw].cpu().numpy().reshape(nw * E, -1)
        self._writers["steps"].write(_columns(STEP_COLUMNS, steps, const))
        if self.cfg.per_cell:
            cells = self._bufc[:nw].cpu().numpy().reshape(nw * E * C, -1)
            self._writers["cells"].write(_columns(CELL_COLUMNS, cells, const))
        if self._raw.numel():
            rob = self._bufr[:nw, :self._raw.numel()].cpu().numpy().reshape(-1, len(ROBOT_COLUMNS))
            self._writers["robots"].write(_columns(ROBOT_COLUMNS, rob, const))
        if self.cfg.hist:
            edges = bin_edges_ms()
            for name, h in (("delay_hist", self._fd), ("aoi_hist", self._fa)):
                hh = h.cpu().numpy()
                env, b = np.nonzero(hh)
                mat = np.stack([np.full(env.size, float(self._win_steps[-1])), env.astype(float), b.astype(float),
                                edges[b], edges[b + 1], hh[env, b].astype(float)], -1).reshape(-1, len(HIST_COLUMNS))
                self._writers[name].write(_columns(HIST_COLUMNS, mat, const))
                h.zero_()
        self._log(steps, nw)
        self._win_steps = []

    def _log(self, steps: np.ndarray, nw: int):
        if self._tb is None and self._wb is None:
            return
        E = self.E
        cols = {k: STEP_COLUMNS.index(k) for k in LOG_COLUMNS}
        mat = steps.reshape(nw, E, -1)
        for w, s in enumerate(self._win_steps):
            if s % self.cfg.log_every:
                continue
            vals = {}
            for k, j in cols.items():
                v = mat[w, :, j]
                v = v[np.isfinite(v)]
                if v.size:
                    vals[f"net/{k}"] = float(v.mean())
            if self._tb is not None:
                for k, v in vals.items():
                    self._tb.add_scalar(k, v, s)
            if self._wb is not None:
                self._wb.log(vals, step=s)

    def _write_meta(self):
        c = self.cfg
        ncfg = self.engine.config
        try:
            summary = ncfg.summary()
        except Exception:
            summary = None
        meta = {"schema": SCHEMA, "label": c.label, "level": self.level, "E": self.E, "R": self.R, "C": self.C,
                "control_step_ms": self.step_ms, "every": c.every, "flush_every": c.flush_every, "format": self.fmt,
                "raw_envs": list(c.raw_envs), "per_cell": c.per_cell, "hist": c.hist,
                "msg_sizes": [float(s) for s in ncfg.msg_sizes], "config": summary,
                "tables": {"steps": list(STEP_COLUMNS), "cells": list(CELL_COLUMNS), "robots": list(ROBOT_COLUMNS),
                           "delay_hist": list(HIST_COLUMNS), "aoi_hist": list(HIST_COLUMNS)},
                "hist_edges_ms": bin_edges_ms().tolist(), **(c.meta or {})}
        with open(os.path.join(c.out_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=1, default=str)

    def close(self):
        """Flush and close the loggers. The partial window at the end is written too (its steps column says how
        many control steps it covers)."""
        if self._closed:
            return
        if self._in_window:
            self._close_window()
        self.flush()
        if self._tb is not None:
            self._tb.flush()
            self._tb.close()
        self._closed = True


def record(engine, out_dir="records", **kw) -> RecorderLoop:
    """RecorderLoop(engine, RecordConfig(out_dir=out_dir, **kw)): wrap an engine with the KPI recorder."""
    return RecorderLoop(engine, RecordConfig(out_dir=out_dir, **kw))


def read_meta(path) -> dict:
    with open(os.path.join(path, "meta.json")) as f:
        return json.load(f)


def read_records(path, table="steps"):
    """One table of a recorder directory as a pandas DataFrame (needs pandas; Parquet also needs pyarrow). Returns
    an empty DataFrame with the table's columns when the table was not written."""
    import pandas as pd
    if table not in TABLES:
        raise ValueError(f"table must be one of {TABLES}")
    d = os.path.join(path, table)
    if os.path.isdir(d):
        parts = sorted(f for f in os.listdir(d) if f.endswith(".parquet"))
        if parts:
            return pd.concat([pd.read_parquet(os.path.join(d, f)) for f in parts], ignore_index=True)
    csv = d + ".csv"
    if os.path.exists(csv):
        return pd.read_csv(csv, keep_default_na=True)
    cols = {"steps": STEP_COLUMNS, "cells": CELL_COLUMNS, "robots": ROBOT_COLUMNS}.get(table, HIST_COLUMNS)
    return pd.DataFrame(columns=list(cols) + ["label", "level"])
