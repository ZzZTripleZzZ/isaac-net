"""Client for the OAI softmodem telnet server (``--telnetsrv``, port 9090): run-time channel control and the rfsim
virtual clock.

Commands used (OAI 2026.w39, common/utils/telnetsrv and radio/rfsimulator):

    channelmod show current                    list the channel models, "model <id> <name> type <type>:" headers
    channelmod modify <id> ploss <dB>          "path loss" of model <id>, which rfsim applies as a GAIN: the received
                                               samples are scaled by 10^(ploss/20) (radio/rfsimulator/
                                               apply_channelmod.c: "path_loss_dB should contain the total path
                                               gain"). An attenuation of L dB is ploss = -L; positive values amplify
                                               and clip the int16 samples.
    channelmod modify <id> noise_power_dB <dB> noise power of model <id>
    rfsimu vtime                               "vtime measurement: TS <samples> sample_rate <Hz>": the rfsim sample
                                               clock of this node (answered from a worker thread, asynchronously)

The gNB is the rfsim server: its model ``rfsimu_channel_ue<k>`` is applied to the uplink of the k-th UE that
connected. Each UE is a client and applies ``rfsimu_channel_enB0`` to its downlink.

Reply framing: the server prints its prompt (``softmodem_<function>> ``) after every command, so cmd() reads up to
that prompt and nothing an earlier command printed can be taken for this command's reply. ``rfsimu vtime`` answers
from a worker thread after the prompt; a vtime query that times out drains the late reply before it raises, and a
query takes the newest vtime line it sees. set_ploss / set_noise raise if the reply reports an error.
"""
from __future__ import annotations

import re
import socket
import threading
import time

_VTIME = re.compile(rb"TS (\d+) sample_rate ([\d.]+)")
_MODEL = re.compile(r"model (\d+) (\S+) type (\S+):")
_PROMPT = re.compile(rb"softmodem\S*> ")
_ERROR = re.compile(r"error|unknown|not found|invalid|out of range|usage", re.IGNORECASE)


class OaiTelnet:
    """One persistent connection to a softmodem telnet server. The server serves one client at a time, so share
    one instance (it is thread-safe: every command holds a lock)."""

    def __init__(self, host, port=9090, timeout=2.0):
        self.host, self.port, self.timeout = host, port, timeout
        self.lock = threading.RLock()
        self.sock = socket.create_connection((host, port), timeout=timeout)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._buf = b""
        self._drain(0.3)

    def _drain(self, wait):
        self.sock.settimeout(wait)
        try:
            while True:
                d = self.sock.recv(65536)
                if not d:
                    break
                self._buf += d
        except (socket.timeout, BlockingIOError):
            pass
        out, self._buf = self._buf, b""
        return out

    def _read_until(self, pat, timeout, keep=False):
        t_end = time.monotonic() + timeout
        while True:
            m = pat.search(self._buf)
            if m:
                if not keep:
                    self._buf = self._buf[m.end():]
                return m
            left = t_end - time.monotonic()
            if left <= 0:
                raise TimeoutError(f"{self.host}:{self.port}: no match for {pat.pattern!r}")
            self.sock.settimeout(left)
            d = self.sock.recv(65536)
            if not d:
                raise ConnectionError("telnet server closed the connection")
            self._buf += d

    def cmd(self, line, wait=0.2):
        """Send one command and return what the server printed before its next prompt (at most `wait` seconds;
        without a prompt, whatever arrived within `wait`)."""
        with self.lock:
            self._drain(0.0)
            self.sock.sendall(line.encode() + b"\n")
            try:
                self._read_until(_PROMPT, wait, keep=True)
            except TimeoutError:
                pass
            m = _PROMPT.search(self._buf)
            if m:
                out, self._buf = self._buf[:m.start()], self._buf[m.end():]
            else:
                out, self._buf = self._buf, b""
            return out.decode(errors="replace")

    def vtime(self):
        """(virtual time in s, wall time in ns at the middle of the query, query round trip in ns)."""
        with self.lock:
            self._drain(0.0)
            t0 = time.time_ns()
            self.sock.sendall(b"rfsimu vtime\n")
            try:
                m = self._read_until(_VTIME, self.timeout)
            except TimeoutError:
                self._drain(self.timeout)       # swallow the late reply so the next query cannot match it
                raise
            t1 = time.time_ns()
            last = None
            for last in _VTIME.finditer(self._buf):   # more replies already here: the newest is this query's
                pass
            if last is not None:
                m, self._buf = last, self._buf[last.end():]
        return int(m.group(1)) / float(m.group(2)), (t0 + t1) // 2, t1 - t0

    def models(self):
        """{model name: id} of the channel models this node holds."""
        with self.lock:
            out = self.cmd("channelmod show current", wait=0.3)
            t_end = time.monotonic() + self.timeout
            while "----" not in out.rsplit("model ", 1)[-1] and time.monotonic() < t_end:
                out += self._drain(0.2).decode(errors="replace")
        return {name: int(i) for i, name, _ in _MODEL.findall(out)}

    def _checked(self, line):
        out = self.cmd(line, wait=max(0.05, min(self.timeout, 0.2)))   # returns at the prompt
        if _ERROR.search(out):
            raise RuntimeError(f"{self.host}:{self.port}: {line!r} failed: {out.strip()!r}")
        return out

    def set_ploss(self, model_id, db):
        """Raw ploss (a gain in dB, see the module docstring). Raises RuntimeError if the server reports an error."""
        return self._checked(f"channelmod modify {int(model_id)} ploss {float(db):.2f}")

    def set_noise(self, model_id, db):
        """Noise power of model_id in dB. Raises RuntimeError if the server reports an error."""
        return self._checked(f"channelmod modify {int(model_id)} noise_power_dB {int(round(db))}")

    def close(self):
        try:
            self.sock.close()
        except OSError:
            pass
