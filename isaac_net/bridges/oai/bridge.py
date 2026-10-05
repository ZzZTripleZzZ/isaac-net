"""OaiBridge: wall-clock lockstep between a control loop and a real protocol stack (OAI rfsim, or the fake stack).

Per control step the caller hands over the frames its robots send (robot k = UE k), the bridge sends them at once
as UDP datagrams from the UE's PDU session to the sink behind the core (one probe header per datagram, frames larger
than the MTU split into back-to-back datagrams), then waits for the end of the step and collects what the sink
received. A frame is delivered when its last fragment arrives; its delay is from the first fragment sent to the last
received, the frame delay the engine reports.

Pacing (``pacing``):
    "virtual"  a step lasts step_dt of rfsim virtual time (the stack's VClock). This keeps the offered load right
               in air time whatever the speed of the simulation, and is the default for OAI rfsim.
    "wall"     a step lasts step_dt of wall time (the Isaac 10 Hz pacing).
    "none"     no waiting: step() collects whatever has arrived (for offline drivers that pace themselves).
Delays are reported in virtual time (the stack's clock); the raw wall-clock values are kept in the logs.

Logs (log_dir): probe-format CSVs, so tools/measure/owd.py and ingest.py read them directly:
    tx_ue<k>.csv, rx.csv          wall clock (sender and receiver on one host: offset 0)
    tx_ue<k>_vt.csv, rx_vt.csv    the same with every timestamp mapped to virtual time, written at close() from
                                  the complete clock record (online mappings extrapolate past the newest sample)
    vtime.csv                     the clock samples (wall ns, virtual s, query round trip)
Online delays (step()["done"]) map both ends when the frame completes: the send time is interpolated, the receive
time can lie a few ms past the newest clock sample and is extrapolated at the recent speed.
The UE index travels in the probe header's flow field (flow = 1 + k): the UPF rewrites source addresses.
"""
from __future__ import annotations

import csv
import json
import os
import socket
import time

import numpy as np

from ...tools.measure.probe import RX_COLS, TX_COLS


class _Agent:
    def __init__(self, host, port, timeout=5.0):
        self.sock = socket.create_connection((host, port), timeout=timeout)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.f = self.sock.makefile("rwb", buffering=0)

    def call(self, **m):
        self.f.write((json.dumps(m) + "\n").encode())
        r = json.loads(self.f.readline())
        if not r.get("ok"):
            raise RuntimeError(f"agent error: {r.get('err')}")
        return r

    def close(self):
        self.sock.close()


class OaiBridge:
    """Frame-level lockstep over a Stack (see the module docstring).

    submit(frames): frames = iterable of (ue, key, nbytes); returns {(ue, key): frame id}. step(): waits for the end of the
    control step and returns a dict with the frames completed during it:
        done      list of (ue, key, delay_s, t_rx_virtual_s, frame id)   delay in virtual seconds
        n_rx      datagrams received this step, late_s: how late the step ended (s, >= 0), speed: virtual / wall
    reset(): new episode (clock restart for pacing; frames still in flight are forgotten).
    """

    def __init__(self, stack, step_dt=0.1, pacing="virtual", mtu=1400, log_dir=None, frame_timeout_s=10.0):
        if pacing not in ("virtual", "wall", "none"):
            raise ValueError(pacing)
        self.stack, self.dt, self.pacing, self.mtu = stack, float(step_dt), pacing, int(mtu)
        self.frame_timeout_s = frame_timeout_s
        self.vc = stack.vclock
        self.ues = [_Agent(*c) for c in stack.ue_ctl]
        self.sink = _Agent(*stack.sink_ctl)
        self.bind = []
        for a in self.ues:
            r = a.call(op="config", dst=stack.sink_dst[0], port=stack.sink_dst[1], bind=stack.ue_bind_ip,
                       bind_if=stack.ue_bind_if, mtu=self.mtu)
            self.bind.append(r.get("bind", ""))
        self.sink.call(op="drain")
        self.log_dir = log_dir
        self._logs = None
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            self._open_logs()
        self.next_fid = [0] * stack.n_ue
        self.reg = {}                    # (ue, fid) -> [key, n_frag, t_tx_first_ns (wall), got frags set, t_last_ns]
        self.k = 0
        self.t0 = None
        self.steps = []                  # per-step timing: (k, late_s, speed)

    # ---------------------------------------------------------------- logs
    def _open_logs(self):
        d = self.log_dir
        self._logs = {}
        for k in range(self.stack.n_ue):
            f = open(os.path.join(d, f"tx_ue{k + 1}.csv"), "w", newline="")
            w = csv.writer(f)
            w.writerow(TX_COLS)
            self._logs[("tx", k)] = (f, w)
        f = open(os.path.join(d, "rx.csv"), "w", newline="")
        w = csv.writer(f)
        w.writerow(RX_COLS)
        self._logs[("rx",)] = (f, w)

    def _write_virtual_logs(self):
        """tx_ue<k>_vt.csv and rx_vt.csv from the wall logs and the complete clock record. Every timestamp column is
        mapped (rx.csv: both t_tx_ns, from the probe header, and t_rx_ns), so OWDs from the _vt files alone are in
        virtual time. A negative t_tx_ns (local send error, see agent.py) stays negative: |t| is mapped."""
        d = self.log_dir
        jobs = [(f"tx_ue{k + 1}", TX_COLS, ("t_tx_ns",)) for k in range(self.stack.n_ue)] + \
               [("rx", RX_COLS, ("t_tx_ns", "t_rx_ns"))]
        for name, cols, tcols in jobs:
            with open(os.path.join(d, f"{name}.csv"), newline="") as f:
                rows = list(csv.DictReader(f))
            for tcol in tcols:
                raw = [int(r[tcol]) for r in rows]
                tv = self.vc.to_virtual_ns([abs(x) for x in raw]) if rows else []
                for r, x, v in zip(rows, raw, tv):
                    r[tcol] = -int(v) if x < 0 else int(v)
            with open(os.path.join(d, f"{name}_vt.csv"), "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(cols)
                for r in rows:
                    w.writerow([r[c] for c in cols])

    def _now(self):
        return self.vc.now_virtual_s() if self.pacing == "virtual" else time.time()

    # ---------------------------------------------------------------- API
    def reset(self):
        self.reg.clear()
        self.sink.call(op="drain")
        self.k = 0
        self.t0 = self._now()

    def set_pathloss(self, ue, ul_db, dl_db=None):
        self.stack.set_pathloss(ue, ul_db, dl_db)

    def submit(self, frames):
        if self.t0 is None:
            self.reset()
        by_ue = {}
        for ue, key, nb in frames:
            fid = self.next_fid[ue]
            self.next_fid[ue] += 1
            by_ue.setdefault(ue, []).append((fid, key, int(nb)))
        fids = {}
        for ue, fl in by_ue.items():
            r = self.ues[ue].call(op="send", frames=[[1 + ue, fid, nb] for fid, _, nb in fl])
            tx = r["tx"]
            first = {}
            for row in tx:
                first.setdefault(row[2], row)
            for fid, key, nb in fl:
                row = first[fid]
                self.reg[(ue, fid)] = [key, row[4], abs(row[7]), set(), 0]
                fids[(ue, key)] = fid
            if self._logs:
                self._log_tx(ue, tx)
        return fids

    def _log_tx(self, ue, tx):
        _, w = self._logs[("tx", ue)]
        for r in tx:
            w.writerow(r)

    def _pace(self):
        target = self.t0 + (self.k + 1) * self.dt
        if self.pacing == "virtual":
            now = self.vc.wait_virtual(target)
        elif self.pacing == "wall":
            while True:
                now = time.time()
                if now >= target:
                    break
                time.sleep(min(0.05, target - now))
        else:
            now = target
        return max(0.0, now - target)

    def step(self):
        if self.t0 is None:
            self.reset()
        late = self._pace()
        rx = self.sink.call(op="drain")["rx"]
        done = []
        if rx:
            if self._logs:
                _, w = self._logs[("rx",)]
                for r in rx:
                    w.writerow(r)
            for src, flow, seq, fid, frag, n_frag, fb, pb, t_tx, t_rx in rx:
                ue = flow - 1
                ent = self.reg.get((ue, fid))
                if ent is None:
                    continue                          # frame of an earlier episode, or a duplicate
                ent[3].add(frag)
                ent[4] = max(ent[4], t_rx)
                if len(ent[3]) == ent[1]:
                    del self.reg[(ue, fid)]
                    tv = self.vc.to_virtual_ns([ent[2], ent[4]])
                    done.append((ue, ent[0], (tv[1] - tv[0]) / 1e9, tv[1] / 1e9, fid))
        # forget frames that can no longer complete
        if self.reg:
            cut = time.time_ns() - int(self.frame_timeout_s * 1e9)
            for key in [k for k, e in self.reg.items() if e[2] < cut]:
                del self.reg[key]
        sp = self.vc.speed(1.0)
        self.steps.append((self.k, late, sp))
        self.k += 1
        return {"done": done, "n_rx": len(rx), "late_s": late, "speed": sp}

    def in_flight(self):
        return len(self.reg)

    def close(self):
        if self._logs:
            for f, _ in self._logs.values():
                f.close()
            self._logs = None
            if not self.vc.identity:
                self.vc.save(os.path.join(self.log_dir, "vtime.csv"))
            self._write_virtual_logs()
        for a in (*self.ues, self.sink):
            a.close()


def summarize_steps(steps):
    a = np.asarray([(s[1], s[2]) for s in steps], float) if steps else np.zeros((0, 2))
    if not len(a):
        return {}
    return {"steps": len(a), "late_p50_ms": float(np.median(a[:, 0]) * 1e3),
            "late_p99_ms": float(np.quantile(a[:, 0], 0.99) * 1e3), "speed_mean": float(np.nanmean(a[:, 1]))}
