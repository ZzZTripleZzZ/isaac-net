"""Lockstep driver: E envs over one or several ns-3 processes, one control step per call.

mode="procs":  one ns-3 process per env (nEnv=1 each), stepped in parallel (send all, then receive all).
mode="single": one ns-3 process that holds all E envs as independent cells (nEnv=E).
mode="groups": P processes with E/P envs each (procs_per_group envs per process).
transport: "tcp" | "unix" | "shm" (ns3-ai).  With spawn=False the caller passes endpoints of servers
that are already running (e.g. Windows client -> WSL servers), one per group.
"""
import os
import subprocess
import time

import numpy as np

from . import protocol as P
from .transport import ShmConn, StreamConn, free_port, uds_path

HERE = os.path.dirname(os.path.abspath(__file__))
BRIDGE_ROOT = os.environ.get("NS3BRIDGE_ROOT", "/home/zzhang66/experiments/bridge_lockstep")
ENVP = os.environ.get("NS3_TOOLCHAIN_ENV", "/home/zzhang66/experiments/ns3ref/env")

# Radio constants of the NetSlot / netslot-ref link budget (for the shadowing <-> SNR conversion).
P_TX_DBM, NI_DBM, PL0, PLEXP = 23.0, -90.0, 40.0, 3.5
_SHM_SEQ = [0]


def shadow_from_snr(pos_xy, snr_db):
    """Shadowing (dB, added to the path loss) that makes ns-3's large-scale SNR equal snr_db."""
    d = np.maximum(np.hypot(pos_xy[..., 0], pos_xy[..., 1]), 1.0)
    return P_TX_DBM - (PL0 + 10 * PLEXP * np.log10(d)) - NI_DBM - snr_db


class Ns3Lockstep:
    def __init__(self, E, R, mode="procs", transport="tcp", envs_per_proc=None, run=1,
                 ns3_args=None, binary=None, spawn=True, endpoints=None, log_dir=None, host="127.0.0.1"):
        self.E, self.R, self.mode, self.transport = E, R, mode, transport
        if mode == "procs":
            epp = 1
        elif mode == "single":
            epp = E
        else:
            epp = envs_per_proc
        assert E % epp == 0, "E must be a multiple of envs_per_proc"
        self.epp = epp
        self.G = E // epp
        self.base_run = run
        self.episode = 0
        self.ns3_args = dict(ns3_args or {})
        self.binary = binary or os.path.join(
            BRIDGE_ROOT, "bin", "netslot-bridge-ai" if transport == "shm" else "netslot-bridge")
        self.procs, self.conns, self.logs = [], [], []
        self.log_dir = log_dir or os.path.join(BRIDGE_ROOT, "logs")
        self.t_step = 0
        self.timing = {"rtt": [], "ns3_run": [], "ns3_other": []}
        if spawn:
            os.makedirs(self.log_dir, exist_ok=True)
            specs = []
            for g in range(self.G):
                if transport == "tcp":
                    spec_srv, spec_cli = (f"tcp:{free_port()}",) * 2
                elif transport == "unix":
                    spec_srv = spec_cli = f"unix:{uds_path(g)}"
                elif transport == "shm":
                    # unique per process lifetime: ns3-ai's creator removes its segment by name on destruction
                    _SHM_SEQ[0] += 1
                    spec_srv = spec_cli = f"shm:ns3b_{os.getpid()}_{_SHM_SEQ[0]}"
                else:
                    raise ValueError(transport)
                specs.append((spec_srv, spec_cli))
            for g, (srv, cli) in enumerate(specs):
                if transport == "shm":        # ns3-ai: Python creates the segment before ns-3 opens it
                    self.conns.append(ShmConn(cli[4:], slot=g))
                self._spawn(g, srv)
            for g, (srv, cli) in enumerate(specs):
                if transport != "shm":
                    self.conns.append(StreamConn(cli))
        else:
            for ep in endpoints:
                if ep.startswith("tcp:") and ep.count(":") == 1:
                    ep = f"tcp:{host}:{ep[4:]}"
                self.conns.append(StreamConn(ep))
        self.info = [self._expect(c, P.HELLO, P.parse_hello) for c in self.conns]
        self.t0 = self.info[0]["t0"]
        self.step_s = self.info[0]["step_s"]
        assert all(i["n_ue"] == R and i["n_env"] == epp for i in self.info), self.info

    # ------------------------------------------------------------------ process management
    def _spawn(self, g, spec):
        args = {"nEnv": self.epp, "nUe": self.R, "run": self.base_run + g * self.epp, **self.ns3_args}
        cmd = [self.binary, f"--bridge={spec}"] + [f"--{k}={_fmt(v)}" for k, v in args.items()]
        env = dict(os.environ)
        env["LD_LIBRARY_PATH"] = f"{BRIDGE_ROOT}/ns-3.48/build/lib:{ENVP}/lib:" + env.get("LD_LIBRARY_PATH", "")
        log = open(os.path.join(self.log_dir, f"ns3_{os.getpid()}_{g}.log"), "w")
        self.logs.append(log)
        self.procs.append(subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env))

    @staticmethod
    def _expect(conn, mtype, parse):
        t, p = conn.recv()
        if t != mtype:
            raise RuntimeError(f"expected message {mtype}, got {t}")
        return parse(p)

    def close(self):
        for c in self.conns:
            c.close()
        for pr in self.procs:
            try:
                pr.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pr.kill()
        for f in self.logs:
            f.close()
        self.conns, self.procs, self.logs = [], [], []

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    # ------------------------------------------------------------------ lockstep API
    def reset(self, groups=None, pos=None, shadow=None, runs=None):
        """Rebuild the scenario in the given process groups (default all).

        pos [E,R,3] env-local initial poses (NaN rows = keep the scenario's own drop),
        shadow [E,R] dB. runs: RNG run per group (default: a fresh run number per episode)."""
        groups = range(self.G) if groups is None else groups
        self.episode += 1
        for g in groups:
            sl = slice(g * self.epp, (g + 1) * self.epp)
            run = runs[g] if runs is not None else self.base_run + self.episode * self.E + g * self.epp
            pp = None if pos is None else _pos3(pos[sl]).reshape(-1, 3)
            sh = None if shadow is None else np.asarray(shadow[sl], np.float32).reshape(-1)
            self.conns[g].send(P.RESET, P.pack_reset(run, pp, sh))
        for g in groups:
            self._expect(self.conns[g], P.HELLO, P.parse_hello)

    def step(self, pos, frames, shadow=None, interp=False, t=None):
        """Advance every process by one control step.

        pos [E,R,2|3] env-local (NaN = keep); frames: structured array with fields env, ue, fid, bytes
        (env = global env index); shadow [E,R] dB or None.
        Returns dict: done (structured: env, ue, fid, frac in [0,1) of the step), per-UE stats [E,R],
        timing."""
        t = self.t_step if t is None else t
        pos = _pos3(pos)
        frames = np.asarray(frames, P.FRAME_IN) if len(frames) else np.zeros(0, P.FRAME_IN)
        genv = frames["env"] // self.epp if len(frames) else np.zeros(0, int)
        tic = time.perf_counter()
        for g, c in enumerate(self.conns):
            sl = slice(g * self.epp, (g + 1) * self.epp)
            fg = frames[genv == g].copy()
            fg["env"] -= g * self.epp
            sh = None if shadow is None else np.asarray(shadow[sl], np.float32).reshape(-1)
            c.send(P.STEP, P.pack_step(t, pos[sl].reshape(-1, 3), fg, sh, interp))
        res = []
        for c in self.conns:
            mt, p = c.recv()
            if mt != P.RESULT:
                raise RuntimeError(f"expected RESULT, got {mt}")
            res.append(P.parse_result(p))
        rtt = time.perf_counter() - tic
        self.t_step = t + 1
        out = {}
        for name in P.RESULT_F32 + P.RESULT_U32:
            out[name] = np.concatenate([r[name] for r in res]).reshape(self.E, self.R)
        done = []
        for g, r in enumerate(res):
            d = r["done"]
            if len(d):
                x = np.zeros(len(d), [("env", "i4"), ("ue", "i4"), ("fid", "i8"), ("frac", "f8")])
                x["env"] = d["env"].astype(np.int64) + g * self.epp
                x["ue"], x["fid"] = d["ue"], d["fid"]
                x["frac"] = (d["t"] - r["t0"]) / self.step_s
                done.append(x)
        out["done"] = np.concatenate(done) if done else np.zeros(0, [("env", "i4"), ("ue", "i4"),
                                                                    ("fid", "i8"), ("frac", "f8")])
        ns3_run = max(r["wall_run"] for r in res)
        ns3_other = max(r["wall_other"] for r in res)
        out["timing"] = {"rtt": rtt, "ns3_run_max": ns3_run, "ns3_other_max": ns3_other,
                         "ns3_run_sum": sum(r["wall_run"] for r in res)}
        self.timing["rtt"].append(rtt)
        self.timing["ns3_run"].append(ns3_run)
        self.timing["ns3_other"].append(ns3_other)
        return out


def _pos3(pos):
    pos = np.asarray(pos, np.float32)
    if pos.shape[-1] == 2:
        z = np.full(pos.shape[:-1] + (1,), np.nan, np.float32)
        pos = np.concatenate([pos, z], -1)
    return np.ascontiguousarray(pos)


def _fmt(v):
    if isinstance(v, bool):
        return "1" if v else "0"
    return str(v)
