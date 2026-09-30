"""Pool of ns-3 / 5G-LENA worker processes (netslot-bridge), one per env, R UEs per cell.

Transport: one TCP connection per worker (localhost). The same class works from WSL (workers
are spawned directly) and from Windows (workers are spawned through wsl.exe and reached via
WSL2 localhost forwarding; pass launcher="wsl").

Per control step the pool scatters one line per worker and gathers one line per worker, so the
step is a barrier: step time = slowest worker + protocol overhead.
"""
from __future__ import annotations

import os
import random
import socket
import subprocess
import time
from dataclasses import dataclass, field

ROOT = os.environ.get("BRIDGE_ROOT", "/home/zzhang66/experiments/bridge_parallel")
BIN = f"{ROOT}/bin/netslot-bridge"
WSL_DISTRO = "Ubuntu-20.04"

# netslot-ref defaults that NetSlot uses; override through Ns3Pool(extra_args=[...])
DEFAULT_ARGS = ["--macTraces=0", "--pktLog=0", "--flowmon=0", "--trafficTime=0", "--ueUeFilter=1"]


@dataclass
class StepReply:
    t: int
    wall_us: float
    done: list = field(default_factory=list)   # (ue, fid, gen_s, last_s)


class Ns3Worker:
    def __init__(self, n_ue, port, init, extra_args=(), launcher="local", seed_run=1, log=None):
        self.n_ue, self.port = n_ue, port
        args = [f"--nUe={n_ue}", f"--io=tcp:{port}"]
        if init is not None:          # None: keep netslot-ref's own placement (used by the tests)
            args.append("--init=" + ",".join(f"{x:.4f}:{y:.4f}:{l:.4f}" for x, y, l in init))
        args += [f"--run={seed_run}", "--outDir=/tmp"] + DEFAULT_ARGS + list(extra_args)
        cmd = [BIN] + args
        if launcher == "wsl":     # from Windows: spawn inside WSL, reach it via localhost forwarding
            cmd = ["wsl", "-d", WSL_DISTRO, "--"] + cmd + ["--bindAll=1"]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                     stderr=log or subprocess.DEVNULL)
        self.sock = None
        self.f = None

    def connect(self, timeout=120.0):
        t_end = time.time() + timeout
        while True:
            try:
                self.sock = socket.create_connection(("127.0.0.1", self.port), timeout=timeout)
                break
            except OSError:
                if self.proc.poll() is not None:
                    raise RuntimeError(f"worker on port {self.port} exited rc={self.proc.returncode}")
                if time.time() > t_end:
                    raise
                time.sleep(0.02)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock.settimeout(None)
        self.f = self.sock.makefile("rwb", buffering=1 << 16)

    def hello(self):
        h = self.f.readline().split()
        assert h and h[0] == b"H", h
        self.app_start, self.period, self.setup_wall = float(h[2]), float(h[3]), float(h[4])

    def send(self, line: bytes):
        self.f.write(line)
        self.f.flush()

    def recv(self) -> StepReply:
        p = self.f.readline().split()
        if not p or p[0] != b"D":
            raise RuntimeError(f"worker {self.port}: bad reply {p[:5]} rc={self.proc.poll()}")
        n = int(p[3])
        done = [(int(p[4 + 4 * k]), int(p[5 + 4 * k]), float(p[6 + 4 * k]), float(p[7 + 4 * k]))
                for k in range(n)]
        return StepReply(int(p[1]), float(p[2]), done)

    def rss_mb(self):
        try:
            for ln in open(f"/proc/{self.proc.pid}/status"):
                if ln.startswith("VmRSS"):
                    return int(ln.split()[1]) / 1024
        except OSError:
            return float("nan")

    def close(self):
        try:
            if self.f:
                self.f.write(b"Q\n")
                self.f.flush()
        except OSError:
            pass
        for x in (self.f, self.sock):
            try:
                x and x.close()
            except OSError:
                pass
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()


def encode_step(t, ues, frames):
    """ues: iterable of (ue, x, y, loss_db); frames: iterable of (ue, fid, bytes)."""
    ues, frames = list(ues), list(frames)
    parts = [f"S {t} {len(ues)}"]
    parts += [f"{u} {x:.3f} {y:.3f} {l:.3f}" for u, x, y, l in ues]
    parts.append(str(len(frames)))
    parts += [f"{u} {i} {b}" for u, i, b in frames]
    return (" ".join(parts) + "\n").encode()


class Ns3Pool:
    MAX_PROCS = 12   # shared-box cap on concurrent ns-3 processes

    def __init__(self, n_workers, n_ue, extra_args=(), launcher="local", base_port=None, seed_run=1,
                 log_dir=None):
        assert n_workers <= self.MAX_PROCS, f"cap is {self.MAX_PROCS} concurrent ns-3 processes"
        self.W, self.R = n_workers, n_ue
        self.extra, self.launcher, self.seed_run = list(extra_args), launcher, seed_run
        self.base_port = base_port or random.randint(20000, 60000 - 64)
        self.log_dir = log_dir
        self.workers = []
        self.timing = {"startup_s": [], "step_s": [], "worker_max_s": [], "worker_mean_s": []}

    def start(self, init_per_worker, runs=None):
        """init_per_worker[w] = list of (x, y, loss_db) of length R, or None. Spawns all, then connects.
        runs: optional explicit ns-3 RNG run per worker (default seed_run + w)."""
        self.close()
        t0 = time.time()
        for w in range(self.W):
            log = open(f"{self.log_dir}/worker{w}.log", "ab") if self.log_dir else None
            run = runs[w] if runs else self.seed_run + w   # every env gets its own ns-3 RNG run
            self.workers.append(Ns3Worker(self.R, self.base_port + w, init_per_worker[w], self.extra,
                                          self.launcher, run, log))
        for wk in self.workers:
            wk.connect()
        for wk in self.workers:
            wk.hello()
        self.timing["startup_s"].append(time.time() - t0)
        self.app_start, self.period = self.workers[0].app_start, self.workers[0].period

    def step(self, lines):
        """Scatter one encoded command per worker, then gather (barrier)."""
        t0 = time.perf_counter()
        for wk, ln in zip(self.workers, lines):
            wk.send(ln)
        replies = [wk.recv() for wk in self.workers]
        dt = time.perf_counter() - t0
        ws = [r.wall_us * 1e-6 for r in replies]
        self.timing["step_s"].append(dt)
        self.timing["worker_max_s"].append(max(ws))
        self.timing["worker_mean_s"].append(sum(ws) / len(ws))
        return replies

    def rss_mb(self):
        return [wk.rss_mb() for wk in self.workers]

    def close(self):
        for wk in self.workers:
            wk.close()
        self.workers = []

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
