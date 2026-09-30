"""Timestamped UDP probes: a sender and a receiver, standard library only.

Every datagram starts with a 32-byte header (network byte order):

  magic "ILNP" | version u8 | flow u8 | reserved u16 | seq u32 | frame_id u32 | frag u16 | n_frag u16 |
  frame_bytes u32 | t_tx_ns u64 (sender CLOCK_REALTIME at send)

and is padded to its payload size. A frame larger than --mtu is split into back-to-back datagrams (fragments),
so frame delay (first fragment sent to last received) is directly comparable to the engine's frame delay.

  # receiver, on the host behind the UPF (or on the UE host for downlink)
  python -m isaaclab_net.tools.measure.probe recv --port 5201 --out rx.csv --duration 70
  # sender, on the UE host, bound to the modem's address so traffic takes the 5G path
  python -m isaaclab_net.tools.measure.probe send --dst 10.45.0.1 --port 5201 --bind 10.45.0.2 \
      --profile cbr --size 1000 --rate 100 --duration 60 --out tx.csv

Profiles: cbr (one frame of --size bytes every 1/--rate s), poisson (exponential gaps, mean 1/--rate),
robot (flow 1: --state-bytes at --state-hz, flow 2: --ctrl-bytes at --ctrl-hz, flow 3: --cam-bytes camera
frames at --cam-hz, all phase-randomized). The receiver uses kernel receive timestamps (SO_TIMESTAMPNS) when the
platform offers them.
"""
from __future__ import annotations

import argparse
import csv
import heapq
import random
import socket
import struct
import sys
import time

HDR = struct.Struct(">4sBBHIIHHIQ")
MAGIC = b"ILNP"
VERSION = 1
TX_COLS = ["flow", "seq", "frame_id", "frag", "n_frag", "frame_bytes", "pkt_bytes", "t_tx_ns"]
RX_COLS = ["src", "flow", "seq", "frame_id", "frag", "n_frag", "frame_bytes", "pkt_bytes", "t_tx_ns", "t_rx_ns"]


def pack(flow, seq, frame_id, frag, n_frag, frame_bytes, t_tx_ns, size):
    h = HDR.pack(MAGIC, VERSION, flow, 0, seq, frame_id, frag, n_frag, frame_bytes, t_tx_ns)
    return h + b"\x00" * max(0, size - HDR.size)


def unpack(buf):
    """Header fields of a probe datagram as a dict, or None if it is not one."""
    if len(buf) < HDR.size or buf[:4] != MAGIC:
        return None
    _, ver, flow, _, seq, fid, frag, nfr, fb, t = HDR.unpack(buf[:HDR.size])
    return {"flow": flow, "seq": seq, "frame_id": fid, "frag": frag, "n_frag": nfr, "frame_bytes": fb,
            "pkt_bytes": len(buf), "t_tx_ns": t, "version": ver}


def fragments(frame_bytes, mtu):
    """Payload sizes of the datagrams of one frame (each at least one header)."""
    mtu = max(mtu, HDR.size)
    n = max(1, -(-frame_bytes // mtu))
    sizes = [mtu] * (n - 1) + [frame_bytes - mtu * (n - 1)]
    return [max(s, HDR.size) for s in sizes]


def schedule(profile, duration, rate=10.0, size=1000, seed=0, state_hz=10.0, state_bytes=4000, ctrl_hz=50.0,
             ctrl_bytes=100, cam_hz=5.0, cam_bytes=30000):
    """Frame release times: list of (t_offset_s, flow, frame_bytes), sorted by time."""
    rng = random.Random(seed)
    ev = []
    if profile == "cbr":
        k = 0
        while k / rate < duration:
            ev.append((k / rate, 0, size))
            k += 1
    elif profile == "poisson":
        t = rng.expovariate(rate)
        while t < duration:
            ev.append((t, 0, size))
            t += rng.expovariate(rate)
    elif profile == "robot":
        for flow, hz, nb in ((1, state_hz, state_bytes), (2, ctrl_hz, ctrl_bytes), (3, cam_hz, cam_bytes)):
            if hz <= 0:
                continue
            t = rng.random() / hz
            while t < duration:
                ev.append((t, flow, nb))
                t += 1.0 / hz
    else:
        raise ValueError(f"unknown profile {profile!r}")
    return sorted(ev)


def _sleep_until(t_target):
    while True:
        d = t_target - time.perf_counter()
        if d <= 0:
            return
        time.sleep(d - 0.0005 if d > 0.002 else 0)


def send(args):
    ev = schedule(args.profile, args.duration, args.rate, args.size, args.seed, args.state_hz, args.state_bytes,
                  args.ctrl_hz, args.ctrl_bytes, args.cam_hz, args.cam_bytes)
    fam = socket.AF_INET6 if ":" in args.dst else socket.AF_INET
    s = socket.socket(fam, socket.SOCK_DGRAM)
    if args.bind:
        s.bind((args.bind, 0))
    if args.tos:
        s.setsockopt(socket.IPPROTO_IP, socket.IP_TOS, args.tos)
    seq, fid = {}, {}
    heap = [(t, i, fl, nb) for i, (t, fl, nb) in enumerate(ev)]
    heapq.heapify(heap)
    with open(args.out, "w", newline="") as out:
        w = csv.writer(out)
        w.writerow(TX_COLS)
        t0 = time.perf_counter() + 0.2
        while heap:
            t, _, flow, nb = heapq.heappop(heap)
            _sleep_until(t0 + t)
            f = fid.get(flow, 0)
            fid[flow] = f + 1
            sizes = fragments(nb, args.mtu)
            for k, sz in enumerate(sizes):
                q = seq.get(flow, 0)
                seq[flow] = q + 1
                tx = time.time_ns()
                s.sendto(pack(flow, q, f, k, len(sizes), nb, tx, sz), (args.dst, args.port))
                w.writerow([flow, q, f, k, len(sizes), nb, sz, tx])
    return 0


def recv(args):
    fam = socket.AF_INET6 if ":" in args.bind else socket.AF_INET
    s = socket.socket(fam, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8 << 20)
    s.bind((args.bind, args.port))
    kts = hasattr(socket, "SO_TIMESTAMPNS")
    if kts:
        try:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_TIMESTAMPNS, 1)
        except OSError:
            kts = False
    s.settimeout(0.5)
    out = open(args.out, "w", newline="")
    w = csv.writer(out)
    w.writerow(RX_COLS)
    t_end = time.time() + args.duration if args.duration > 0 else float("inf")
    try:
        _recv_loop(s, w, out, kts, t_end)
    finally:
        out.close()
    return 0


def _recv_loop(s, w, out, kts, t_end):
    n = 0
    while time.time() < t_end:
        try:
            if kts:
                buf, anc, _, addr = s.recvmsg(65535, 64)
                t_rx = time.time_ns()
                for lvl, typ, data in anc:
                    if lvl == socket.SOL_SOCKET and typ == socket.SO_TIMESTAMPNS and len(data) >= 16:
                        sec, nsec = struct.unpack("qq", data[:16])
                        t_rx = sec * 1_000_000_000 + nsec
            else:
                buf, addr = s.recvfrom(65535)
                t_rx = time.time_ns()
        except socket.timeout:
            continue
        h = unpack(buf)
        if h is None:
            continue
        w.writerow([addr[0]] + [h[c] for c in RX_COLS[1:-1]] + [t_rx])
        n += 1
        if n % 1000 == 0:
            out.flush()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("send")
    a.add_argument("--dst", required=True)
    a.add_argument("--port", type=int, default=5201)
    a.add_argument("--bind", default="", help="source address (the UE's PDU-session IP)")
    a.add_argument("--profile", default="cbr", choices=["cbr", "poisson", "robot"])
    a.add_argument("--size", type=int, default=1000, help="frame bytes (cbr, poisson)")
    a.add_argument("--rate", type=float, default=10.0, help="frames per second (cbr, poisson)")
    a.add_argument("--mtu", type=int, default=1400, help="max UDP payload per datagram")
    a.add_argument("--duration", type=float, default=60.0)
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--tos", type=int, default=0)
    a.add_argument("--state-hz", type=float, default=10.0)
    a.add_argument("--state-bytes", type=int, default=4000)
    a.add_argument("--ctrl-hz", type=float, default=50.0)
    a.add_argument("--ctrl-bytes", type=int, default=100)
    a.add_argument("--cam-hz", type=float, default=5.0)
    a.add_argument("--cam-bytes", type=int, default=30000)
    a.add_argument("--out", required=True)
    b = sub.add_parser("recv")
    b.add_argument("--bind", default="0.0.0.0")
    b.add_argument("--port", type=int, default=5201)
    b.add_argument("--duration", type=float, default=0.0, help="seconds, 0 = until interrupted")
    b.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    try:
        return send(args) if args.cmd == "send" else recv(args)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
