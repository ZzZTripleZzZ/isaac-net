"""Traffic agents of the OAI bridge: standard library only, so they run inside the OAI containers' network namespaces
(python:3.11-slim sidecars) as well as on the host for the fake stack.

    python agent.py ue    --ctl 0.0.0.0:5300                 # UE side: sends frames from the PDU session
    python agent.py sink  --ctl 0.0.0.0:5301 --port 5201     # data network side: receives and timestamps
    python agent.py proxy --listen 127.0.0.1:5202 --to 127.0.0.1:5201 --delay-ms 8 --jitter-ms 2 --loss 0.01
                                                             # fake stack: a UDP relay with delay, jitter and loss

Datagrams carry the probe header of isaac_net.tools.measure.probe (magic, flow, seq, frame id, fragment,
frame size, send time), so the logs the bridge writes are probe logs and go through tools/measure/owd.py and the
unified schema unchanged.

Control protocol: one TCP connection per agent, one JSON object per line in each direction.
  ue:    {"op": "config", "dst": ip, "port": p, "bind": ip or "", "bind_if": "oaitun_ue1" or "", "mtu": 1400}
         {"op": "send", "frames": [[flow, frame_id, bytes], ...]}  -> {"tx": [[flow, seq, frame_id, frag, n_frag,
                                                                        frame_bytes, pkt_bytes, t_tx_ns], ...]}
  sink:  {"op": "drain"} -> {"rx": [[src, flow, seq, frame_id, frag, n_frag, frame_bytes, pkt_bytes, t_tx_ns,
                                      t_rx_ns], ...]}   (every packet received since the last drain)
  both:  {"op": "ping"} -> {"t_ns": CLOCK_REALTIME}, {"op": "quit"}
Every reply carries "ok": 1, or "ok": 0 and "err".
"""
from __future__ import annotations

import argparse
import fcntl
import heapq
import json
import os
import random
import socket
import struct
import sys
import threading
import time

_here = os.path.dirname(os.path.abspath(__file__))
for _p in (_here, os.path.join(_here, "..", "..", "tools", "measure")):
    if os.path.exists(os.path.join(_p, "probe.py")) and _p not in sys.path:
        sys.path.insert(0, _p)
import probe  # noqa: E402  (tools/measure/probe.py, copied next to this file in the containers)

SIOCGIFADDR = 0x8915
# Python does not export SO_TIMESTAMPNS on Linux (value 35, also the SCM type of the ancillary message)
SO_TIMESTAMPNS = getattr(socket, "SO_TIMESTAMPNS", 35 if sys.platform.startswith("linux") else None)


def if_addr(name):
    """IPv4 address of interface `name` (Linux), or '' if it has none yet."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        r = fcntl.ioctl(s.fileno(), SIOCGIFADDR, struct.pack("256s", name[:15].encode()))
        return socket.inet_ntoa(r[20:24])
    except OSError:
        return ""
    finally:
        s.close()


def _hostport(s):
    h, p = s.rsplit(":", 1)
    return h, int(p)


def _kernel_ts_socket(bind, port, rcvbuf=8 << 20):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, rcvbuf)
    s.bind((bind, port))
    kts = SO_TIMESTAMPNS is not None
    if kts:
        try:
            s.setsockopt(socket.SOL_SOCKET, SO_TIMESTAMPNS, 1)
        except OSError:
            kts = False
    return s, kts


def _recv_ts(s, kts):
    if kts:
        buf, anc, _, addr = s.recvmsg(65535, 64)
        t_rx = time.time_ns()
        for lvl, typ, data in anc:
            if lvl == socket.SOL_SOCKET and typ == SO_TIMESTAMPNS and len(data) >= 16:
                sec, nsec = struct.unpack("qq", data[:16])
                t_rx = sec * 1_000_000_000 + nsec
        return buf, addr, t_rx
    buf, addr = s.recvfrom(65535)
    return buf, addr, time.time_ns()


class UeRole:
    def __init__(self):
        self.sock = None
        self.dst = None
        self.mtu = 1400
        self.seq = {}

    def handle(self, m):
        op = m["op"]
        if op == "config":
            bind = m.get("bind") or ""
            if not bind and m.get("bind_if"):
                deadline = time.time() + float(m.get("wait_s", 30))
                while not bind and time.time() < deadline:
                    bind = if_addr(m["bind_if"])
                    if not bind:
                        time.sleep(0.2)
                if not bind:
                    return {"ok": 0, "err": f"interface {m['bind_if']} has no IPv4 address"}
            if self.sock is not None:
                self.sock.close()
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4 << 20)
            if bind:
                self.sock.bind((bind, 0))
            self.dst = (m["dst"], int(m["port"]))
            self.mtu = int(m.get("mtu", 1400))
            self.seq = {}
            return {"ok": 1, "bind": bind}
        if op == "send":
            if self.sock is None:
                return {"ok": 0, "err": "not configured"}
            tx = []
            for flow, fid, nb in m["frames"]:
                sizes = probe.fragments(int(nb), self.mtu)
                for k, sz in enumerate(sizes):
                    q = self.seq.get(flow, 0)
                    self.seq[flow] = q + 1
                    t = time.time_ns()
                    try:
                        self.sock.sendto(probe.pack(flow, q, fid, k, len(sizes), nb, t, sz), self.dst)
                    except OSError:
                        t = -t                        # local send error: reported as a negative send time
                    tx.append([flow, q, fid, k, len(sizes), nb, sz, t])
            return {"ok": 1, "tx": tx}
        return None


class SinkRole:
    def __init__(self, port, bind="0.0.0.0"):
        self.sock, self.kts = _kernel_ts_socket(bind, port)
        self.buf = []
        self.lock = threading.Lock()
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while True:
            try:
                buf, addr, t_rx = _recv_ts(self.sock, self.kts)
            except OSError:
                continue
            h = probe.unpack(buf)
            if h is None:
                continue
            row = [addr[0], h["flow"], h["seq"], h["frame_id"], h["frag"], h["n_frag"], h["frame_bytes"],
                   h["pkt_bytes"], h["t_tx_ns"], t_rx]
            with self.lock:
                self.buf.append(row)

    def handle(self, m):
        if m["op"] == "drain":
            with self.lock:
                rx, self.buf = self.buf, []
            return {"ok": 1, "rx": rx, "kernel_ts": int(self.kts)}
        return None


def serve(ctl, role):
    host, port = _hostport(ctl)
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(4)
    print(f"agent listening on {host}:{port}", flush=True)
    while True:
        conn, _ = srv.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        f = conn.makefile("rwb", buffering=0)
        try:
            for line in f:
                try:
                    m = json.loads(line)
                    if m.get("op") == "ping":
                        r = {"ok": 1, "t_ns": time.time_ns()}
                    elif m.get("op") == "quit":
                        f.write(b'{"ok": 1}\n')
                        return 0
                    else:
                        r = role.handle(m) or {"ok": 0, "err": f"unknown op {m.get('op')!r}"}
                except Exception as e:  # noqa: BLE001 - report every failure to the client
                    r = {"ok": 0, "err": repr(e)}
                f.write((json.dumps(r) + "\n").encode())
        except (ConnectionError, OSError):
            pass
        finally:
            conn.close()


def proxy(listen, to, delay_ms, jitter_ms, loss, seed, rate_bps=0.0):
    """UDP relay that holds each datagram delay + U(0, jitter) ms (FIFO per relay when rate_bps > 0: a serializing
    link of that rate in front of the delay) and drops it with probability `loss`."""
    rng = random.Random(seed)
    s, _ = _kernel_ts_socket(*_hostport(listen))
    out = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    dst = _hostport(to)
    heap, cv = [], threading.Condition(threading.Lock())
    busy_until = [0.0]
    n = [0]

    def sender():
        while True:
            with cv:
                while not heap:
                    cv.wait()
                t, _, data = heap[0]
                d = t - time.time()
                if d > 0:
                    cv.wait(min(d, 0.05))
                    continue
                heapq.heappop(heap)
            out.sendto(data, dst)

    threading.Thread(target=sender, daemon=True).start()
    print(f"proxy {listen} -> {to}: {delay_ms} ms + U(0,{jitter_ms}) ms, loss {loss}, rate {rate_bps} bit/s",
          flush=True)
    while True:
        data, _ = s.recvfrom(65535)
        now = time.time()
        if rng.random() < loss:
            continue
        start = now
        if rate_bps > 0:
            start = max(now, busy_until[0])
            busy_until[0] = start + len(data) * 8 / rate_bps
            start = busy_until[0]
        t = start + (delay_ms + rng.random() * jitter_ms) / 1000.0
        with cv:
            n[0] += 1
            heapq.heappush(heap, (t, n[0], data))
            cv.notify()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="role", required=True)
    u = sub.add_parser("ue")
    u.add_argument("--ctl", default="0.0.0.0:5300")
    k = sub.add_parser("sink")
    k.add_argument("--ctl", default="0.0.0.0:5301")
    k.add_argument("--port", type=int, default=5201)
    k.add_argument("--bind", default="0.0.0.0")
    p = sub.add_parser("proxy")
    p.add_argument("--listen", required=True)
    p.add_argument("--to", required=True)
    p.add_argument("--delay-ms", type=float, default=10.0)
    p.add_argument("--jitter-ms", type=float, default=0.0)
    p.add_argument("--loss", type=float, default=0.0)
    p.add_argument("--rate-bps", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    if a.role == "ue":
        return serve(a.ctl, UeRole())
    if a.role == "sink":
        return serve(a.ctl, SinkRole(a.port, a.bind))
    proxy(a.listen, a.to, a.delay_ms, a.jitter_ms, a.loss, a.seed, a.rate_bps)
    return 0


if __name__ == "__main__":
    sys.exit(main())
