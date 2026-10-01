"""Minimal Windows-side client for the ns-3 TCP lockstep bridge (stdlib only: no numpy, no torch).

Isaac Sim (Windows) talks to ns-3 (WSL2) through WSL's localhost port forwarding. This file is both a
tiny client library (Ns3Client) and a round-trip benchmark:

    python win_client.py --ports 57100,57101 --R 8 --steps 200

Start the servers in WSL first (scripts/serve_wsl.sh E R BASEPORT).
"""
import argparse
import math
import array
import json
import random
import socket
import struct
import time

MAGIC = 0x3142534E
HDR = struct.Struct("<III")
HELLO, STEP, RESULT, RESET, CLOSE = 1, 2, 3, 4, 5
FRAME_IN = struct.Struct("<HHII")
FRAME_DONE = struct.Struct("<HHId")
RES_HDR = struct.Struct("<iIIIddd")


class Ns3Client:
    def __init__(self, host, port, timeout=60.0):
        t_end = time.time() + timeout
        while True:
            try:
                self.s = socket.create_connection((host, port), timeout=5.0)
                break
            except OSError:
                if time.time() > t_end:
                    raise
                time.sleep(0.2)
        self.s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.s.settimeout(None)
        _, p = self._recv()
        ver, self.n_env, self.n_ue, self.run, self.t0, self.step_s = struct.unpack_from("<IIIIdd", p)
        self.N = self.n_env * self.n_ue

    def _read(self, n):
        buf = bytearray(n)
        v = memoryview(buf)
        got = 0
        while got < n:
            k = self.s.recv_into(v[got:], n - got)
            if not k:
                raise ConnectionError("server closed")
            got += k
        return bytes(buf)

    def _recv(self):
        magic, t, n = HDR.unpack(self._read(HDR.size))
        assert magic == MAGIC
        return t, self._read(n) if n else b""

    def _send(self, t, payload):
        self.s.sendall(HDR.pack(MAGIC, t, len(payload)) + payload)

    def step(self, t, pos_xyz, frames, shadow=None):
        """pos_xyz: flat list of 3N floats (NaN = keep); frames: list of (env, ue, fid, bytes)."""
        flags = 1 if shadow is not None else 0
        body = [struct.pack("<iII", t, flags, len(frames)), array.array("f", pos_xyz).tobytes()]
        if shadow is not None:
            body.append(array.array("f", shadow).tobytes())
        body.append(b"".join(FRAME_IN.pack(*f) for f in frames))
        self._send(STEP, b"".join(body))
        mt, p = self._recv()
        assert mt == RESULT, mt
        t_, n, n_done, k, t0, wall_run, wall_other = RES_HDR.unpack_from(p)
        off = RES_HDR.size
        f32 = {}
        for name in ("sinr_db", "rsrp_dbm", "rlc_bytes", "mcs"):
            f32[name] = array.array("f", p[off:off + 4 * n])
            off += 4 * n
        for name in ("n_tb", "n_retx", "n_corrupt", "n_tb_lost", "ok_bytes"):
            f32[name] = array.array("I", p[off:off + 4 * n])
            off += 4 * n
        done = [FRAME_DONE.unpack_from(p, off + 16 * j) for j in range(n_done)]
        return {"done": done, "t0": t0, "wall_run": wall_run, "wall_other": wall_other, **f32}

    def reset(self, run, pos_xyz=None):
        flags = 1 if pos_xyz is not None else 0
        p = struct.pack("<II", run, flags) + (array.array("f", pos_xyz).tobytes() if pos_xyz else b"")
        self._send(RESET, p)
        self._recv()

    def close(self):
        try:
            self._send(CLOSE, b"")
        finally:
            self.s.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--ports", default="57100")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--p", type=float, default=0.3)
    ap.add_argument("--tag", default="windows")
    a = ap.parse_args()
    rnd = random.Random(0)
    cl = [Ns3Client(a.host, int(p)) for p in a.ports.split(",")]
    nan = float("nan")
    fid = [[0] * c.N for c in cl]
    pos = []
    for c in cl:
        pp = []
        for i in range(c.N):
            d, ang = rnd.uniform(20, 80), rnd.uniform(0.05, 1.5)
            pp += [d * math.cos(ang), d * math.sin(ang), nan]
        pos.append(pp)
    for j, c in enumerate(cl):
        c.reset(1 + j, pos[j])
    rtt, run, other, done = [], [], [], 0
    for k in range(a.steps):
        frames_all = []
        for j, c in enumerate(cl):
            fr = []
            for i in range(c.N):
                pos[j][3 * i] += rnd.gauss(0, 0.3)
                pos[j][3 * i + 1] += rnd.gauss(0, 0.3)
                if rnd.random() < a.p:
                    fr.append((i // c.n_ue, i % c.n_ue, fid[j][i], 4000 if rnd.random() < 0.7 else 30000))
                    fid[j][i] += 1
            frames_all.append(fr)
        # send to all servers first, then receive all (parallel ns-3 processes)
        t_a = time.perf_counter()
        for j, c in enumerate(cl):
            flags = 0
            body = struct.pack("<iII", k, flags, len(frames_all[j])) + array.array("f", pos[j]).tobytes() + \
                b"".join(FRAME_IN.pack(*f) for f in frames_all[j])
            c._send(STEP, body)
        wr = []
        for c in cl:
            mt, p = c._recv()
            t_, n, n_done, kk, t0, wall_run, wall_other = RES_HDR.unpack_from(p)
            wr.append((wall_run, wall_other))
            done += n_done
        t_b = time.perf_counter()
        if k >= 5:
            rtt.append((t_b - t_a) * 1e3)
            run.append(max(w[0] for w in wr) * 1e3)
            other.append(max(w[1] for w in wr) * 1e3)
    for c in cl:
        c.close()
    srt = sorted(rtt)
    ov = sorted(r - s for r, s in zip(rtt, run))
    out = {"tag": a.tag, "servers": len(cl), "R": cl[0].n_ue, "steps": len(rtt),
           "rtt_ms_mean": sum(rtt) / len(rtt), "rtt_ms_p50": srt[len(srt) // 2],
           "rtt_ms_p95": srt[int(0.95 * len(srt))],
           "ns3_run_ms_mean": sum(run) / len(run), "ns3_io_ms_mean": sum(other) / len(other),
           "overhead_ms_mean": sum(ov) / len(ov), "overhead_ms_p50": ov[len(ov) // 2],
           "overhead_ms_p95": ov[int(0.95 * len(ov))], "frames_done": done}
    print(json.dumps(out))


if __name__ == "__main__":
    main()
