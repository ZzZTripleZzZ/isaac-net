"""Client-side transports. All carry the framed bytes of protocol.py."""
import os
import socket
import time

from . import protocol as P


class StreamConn:
    """TCP ("tcp:HOST:PORT" or "tcp:PORT") or Unix-socket ("unix:PATH") client."""

    def __init__(self, spec, connect_timeout=120.0):
        self.spec = spec
        deadline = time.time() + connect_timeout
        while True:
            try:
                if spec.startswith("tcp:"):
                    rest = spec[4:]
                    host, port = (rest.rsplit(":", 1) if ":" in rest else ("127.0.0.1", rest))
                    s = socket.create_connection((host, int(port)), timeout=5.0)
                    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                else:
                    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                    s.connect(spec[5:])
                break
            except (ConnectionRefusedError, FileNotFoundError, socket.timeout, OSError):
                if time.time() > deadline:
                    raise
                time.sleep(0.05)
        s.settimeout(None)
        self.s = s

    def send(self, mtype, payload=b""):
        self.s.sendall(P.frame(mtype, payload))

    def _read(self, n):
        buf = bytearray(n)
        view = memoryview(buf)
        got = 0
        while got < n:
            k = self.s.recv_into(view[got:], n - got)
            if k == 0:
                raise ConnectionError(f"{self.spec}: server closed the connection")
            got += k
        return bytes(buf)

    def recv(self):
        magic, mtype, n = P.HDR.unpack(self._read(P.HDR.size))
        if magic != P.MAGIC:
            raise RuntimeError("bad magic")
        return mtype, (self._read(n) if n else b"")

    def close(self):
        try:
            self.send(P.CLOSE)
        except OSError:
            pass
        self.s.close()


class ShmConn:
    """ns3-ai msg-interface (Boost managed shared memory). Python creates the segment (ns3-ai
    convention) before the ns-3 process is launched with --bridge=shm:NAME."""

    def __init__(self, name, slot=0):
        from . import _ns3ai_shm_loader as L
        self.name = name
        self.m = L.create(slot, name)

    def send(self, mtype, payload=b""):
        self.m.send(mtype, payload)

    def recv(self):
        return self.m.recv()

    def close(self):
        try:
            self.send(P.CLOSE)
        except Exception:
            pass
        self.m = None


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def uds_path(tag):
    d = os.environ.get("NS3BRIDGE_UDS_DIR", "/tmp")
    return os.path.join(d, f"ns3bridge_{os.getpid()}_{tag}.sock")
