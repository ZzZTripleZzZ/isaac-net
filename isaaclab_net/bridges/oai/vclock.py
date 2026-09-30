"""The rfsim virtual clock and its mapping from wall time.

OAI's RF simulator is not paced by a sample clock: the gNB and the UEs exchange samples as fast as they compute
them, so virtual (air) time runs faster or slower than wall time with the load of the host. On the shared lab box it
ran between 0.45x and 1.9x wall time (docs/bridges-oai.md). Everything the protocol stack does (SR periods, K2,
HARQ timing, the TDD pattern) happens in virtual time, while packets enter the UE's TUN interface and leave the UPF in
wall time. A VClock collects (wall time, virtual time) pairs and maps any wall timestamp to virtual time by
piecewise-linear interpolation, so one-way delays can be reported in virtual time, which is what a real-time radio
would show.

Sources:
    TTraceClock   the gNB's T-tracer: GNB_PHY_UL_TICK (frame, slot) of every slot with the gNB's wall timestamp,
                  read through OAI's ``textlog`` tool; every ``every``-th slot is kept. This is the default: exact
                  slot counts, no load on the gNB beyond the T-tracer itself. The same textlog process can record
                  other T events (the MAC scheduling trace) to a file.
    VClock(src)   polls ``src.vtime()`` (OaiTelnet: ``rfsimu vtime``). Not recommended: in OAI 2026.w39 the gNB
                  aborted ("buffer overflow detected") after about 20 queries at 50 Hz, apparently because the
                  command runs in a worker thread that prints into the telnet server's shared buffer.
    VClock(None)  identity (the fake stack; wall time = virtual time).
The tick timestamps are taken when the gNB processes a slot, a roughly constant number of slots before or after
the slot's air time; a constant offset cancels in every delay.

Virtual timestamps are anchored to the wall clock at the first sample, ``v_epoch(w) = w0 + (V(w) - V(w0))``, so they
look like ordinary epoch times and the measurement tools (probe logs, owd.py, ingest) take them unchanged.
"""
from __future__ import annotations

import os
import re
import subprocess
import threading
import time

import numpy as np


def map_wall_to_virtual(wall_ns, w, v, w0=None, v0=None):
    """Wall epoch ns -> virtual epoch ns for clock samples w [ns], v [s], anchored at (w0, v0) (default: the first
    sample). Outside the sampled range the nearest segment's slope is extended."""
    wall_ns = np.asarray(wall_ns, np.int64)
    w0 = int(w[0]) if w0 is None else int(w0)
    v0 = float(v[0]) if v0 is None else float(v0)
    x = (wall_ns - w0).astype(float)
    xs = (w - w0).astype(float)
    out = np.interp(x, xs, v)
    lo, hi = x < xs[0], x > xs[-1]
    if lo.any():
        out[lo] = v[0] + (x[lo] - xs[0]) * (v[1] - v[0]) / max(xs[1] - xs[0], 1.0)
    if hi.any():
        k = max(0, len(xs) - 10)                     # slope over the last few samples (ticks are jittery)
        out[hi] = v[-1] + (x[hi] - xs[-1]) * (v[-1] - v[k]) / max(xs[-1] - xs[k], 1.0)
    return (w0 + (out - v0) * 1e9).astype(np.int64)


class VClock:
    """(wall, virtual) samples and the mapping between them. source: None (identity) or an object with vtime() ->
    (virtual s, wall ns, round trip ns), polled every period_s in a background thread."""

    def __init__(self, source=None, period_s=0.02, keep=200_000):
        self.src = source
        self.period = period_s
        self.keep = keep
        self.w, self.v, self.rtt = [], [], []
        self.w0 = self.v0 = None
        self.lock = threading.Lock()
        self._stop = threading.Event()
        self._thr = None
        if source is not None:
            self.poll()
            self._thr = threading.Thread(target=self._loop, daemon=True)
            self._thr.start()

    @property
    def identity(self):
        return self.src is None and self.w0 is None and not isinstance(self, TTraceClock)

    def add(self, w_ns, v_s, rtt_ns=0):
        with self.lock:
            if self.v and v_s < self.v[-1]:            # the clock restarted (gNB restart): start over
                self.w, self.v, self.rtt = [], [], []
                self.w0 = self.v0 = None
            if self.w0 is None:
                self.w0, self.v0 = int(w_ns), float(v_s)
            self.w.append(int(w_ns))
            self.v.append(float(v_s))
            self.rtt.append(int(rtt_ns))
            if len(self.w) > self.keep:
                n = len(self.w) - self.keep
                del self.w[:n], self.v[:n], self.rtt[:n]

    def poll(self):
        v, w, rtt = self.src.vtime()
        self.add(w, v, rtt)

    def _loop(self):
        while not self._stop.wait(self.period):
            try:
                self.poll()
            except (OSError, TimeoutError, ConnectionError):
                continue

    def samples(self):
        with self.lock:
            return np.asarray(self.w, np.int64), np.asarray(self.v, float), np.asarray(self.rtt, np.int64)

    def ready(self, n=2):
        return self.identity or len(self.w) >= n

    def wait_ready(self, timeout=10.0, n=20):
        t_end = time.monotonic() + timeout
        while not self.ready(n):
            if time.monotonic() > t_end:
                raise TimeoutError("no virtual clock samples")
            time.sleep(0.01)

    def to_virtual_ns(self, wall_ns):
        """Wall epoch ns -> virtual epoch ns (anchored at the first sample)."""
        wall_ns = np.asarray(wall_ns, np.int64)
        if self.identity:
            return wall_ns.copy()
        w, v, _ = self.samples()
        if len(w) < 2:
            return wall_ns.copy()
        return map_wall_to_virtual(wall_ns, w, v, self.w0, self.v0)

    def now_virtual_s(self):
        """Current virtual time as epoch seconds (wall time for the identity clock)."""
        return float(self.to_virtual_ns(np.array([time.time_ns()]))[0]) / 1e9

    def speed(self, window_s=5.0):
        """Virtual seconds per wall second over the last window_s of samples (1.0 for the identity clock)."""
        if self.identity:
            return 1.0
        w, v, _ = self.samples()
        if len(w) < 2:
            return float("nan")
        i = min(int(np.searchsorted(w, w[-1] - int(window_s * 1e9))), len(w) - 2)
        return float((v[-1] - v[i]) / ((w[-1] - w[i]) / 1e9))

    def wait_virtual(self, target_s, poll_s=0.002):
        """Sleep until the virtual epoch time reaches target_s; returns the virtual time at wake-up."""
        while True:
            now = self.now_virtual_s()
            if now >= target_s:
                return now
            sp = self.speed(1.0)
            sp = sp if np.isfinite(sp) and sp > 0 else 1.0
            time.sleep(max(poll_s, min(0.05, (target_s - now) / sp * 0.8)))

    def save(self, path):
        w, v, rtt = self.samples()
        with open(path, "w") as f:
            f.write("wall_ns,vtime_s,rtt_ns\n")
            for a, b, c in zip(w, v, rtt):
                f.write(f"{a},{b:.9f},{c}\n")

    def close(self):
        self._stop.set()
        if self._thr is not None:
            self._thr.join(timeout=1.0)


_TL = re.compile(r"^(\d\d):(\d\d):(\d\d)\.(\d+)\s*\[(\d+)\]:\s+(\w+)\s*(.*)$")
_TICK = re.compile(r"frame (\d+) slot (\d+)")


class TTraceClock(VClock):
    """Virtual clock from the gNB's T-tracer slot ticks (see the module docstring).

    textlog: path of OAI's common/utils/T/tracer/textlog (built from the checkout: ``make textlog``); t_db: its
    T_messages.txt. mu: numerology (slot = 1 ms / 2^mu). every: keep one tick in `every` slots (20 = 10 ms at mu 1).
    extra_events / log_path: other T events to record, written verbatim (textlog format with -raw-time) to log_path.
    The gNB must run with ``--T_stdout 2 --T_nowait`` (T-tracer on, console kept, no wait for a tracer)."""

    def __init__(self, textlog, t_db, host="192.168.71.140", port=2021, mu=1, every=20, extra_events=(),
                 log_path=None, keep=200_000):
        super().__init__(None, keep=keep)
        self.slot_s = 1e-3 / 2 ** mu
        self.spf = 10 * 2 ** mu
        self.every = every
        argv = [textlog, "-d", t_db, "-ip", host, "-p", str(port), "-raw-time", "-on", "GNB_PHY_UL_TICK"]
        for ev in extra_events:
            argv += ["-on", ev]
        self.log = open(log_path, "w") if log_path else None
        self.proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self._base, self._prev, self._n = 0, None, 0
        self._thr = threading.Thread(target=self._reader, daemon=True)
        self._thr.start()

    def _reader(self):
        cyc = 1024 * self.spf
        for line in self.proc.stdout:
            m = _TL.match(line)
            if not m:
                continue
            ev = m.group(6)
            if ev != "GNB_PHY_UL_TICK":
                if self.log:
                    self.log.write(line)
                continue
            t = _TICK.search(m.group(7))
            if not t:
                continue
            raw = int(t.group(1)) * self.spf + int(t.group(2))
            if self._prev is not None and raw < self._prev - cyc // 2:
                self._base += cyc
            self._prev = raw
            self._n += 1
            if self._n % self.every:
                continue
            w = int(m.group(5)) * 1_000_000_000 + int(m.group(4).ljust(9, "0")[:9])
            self.add(w, (self._base + raw) * self.slot_s)

    def close(self):
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self._thr.join(timeout=2.0)
        if self.log:
            self.log.close()
            self.log = None


def load_vtime_csv(path):
    """(wall_ns, vtime_s) arrays from VClock.save()."""
    a = np.genfromtxt(path, delimiter=",", names=True, dtype=None)
    return a["wall_ns"].astype(np.int64), a["vtime_s"].astype(float)


def default_textlog():
    """$OAI_TEXTLOG and $OAI_T_MESSAGES, if set and present, else (None, None)."""
    tl, db = os.environ.get("OAI_TEXTLOG"), os.environ.get("OAI_T_MESSAGES")
    if tl and db and os.path.exists(tl) and os.path.exists(db):
        return tl, db
    return None, None
