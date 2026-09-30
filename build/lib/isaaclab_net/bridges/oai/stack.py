"""The network stacks the OAI bridge drives: the OAI rfsim deployment (DockerOaiStack) and a local stand-in with the
same interfaces (FakeStack: UDP relay with configurable delay, jitter, loss and rate), so the bridge and its tests
run without OAI.

Both expose
    n_ue                      number of UEs (UE k carries the traffic of one robot)
    ue_ctl[k], sink_ctl       (host, port) of the traffic agents' control sockets (agent.py)
    sink_dst                  (ip, port) the UE agents send to
    ue_bind_if / ue_bind_ip   source interface or address of the UE agents
    vclock                    VClock: wall -> virtual time (identity for FakeStack)
    set_pathloss(k, ul_db, dl_db)   move UE k (rfsim channel path loss; recorded only by FakeStack)
    macstats()                gNB nrMAC_stats.log text or ""
    close()
"""
from __future__ import annotations

import os
import socket
import subprocess
import sys
import time

from .telnet import OaiTelnet
from .vclock import TTraceClock, VClock, default_textlog

AGENT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "agent.py")

# addresses of deploy/docker-compose.yaml
GNB_IP = "192.168.71.140"
UE_IPS = tuple(f"192.168.71.{150 + k}" for k in range(10))
SINK_IP = "192.168.72.135"
UE_CTL_PORT, SINK_CTL_PORT, SINK_PORT = 5300, 5301, 5201
TELNET_PORT = 9090


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def _wait_port(host, port, timeout=10.0):
    t_end = time.monotonic() + timeout
    while time.monotonic() < t_end:
        try:
            socket.create_connection((host, port), timeout=0.5).close()
            return True
        except OSError:
            time.sleep(0.05)
    raise TimeoutError(f"{host}:{port} did not open within {timeout} s")


class Stack:
    n_ue = 0
    ue_ctl = ()
    sink_ctl = None
    sink_dst = None
    ue_bind_if = ""
    ue_bind_ip = ""
    vclock = None
    kind = "abstract"

    def set_pathloss(self, k, ul_db, dl_db=None):
        raise NotImplementedError

    def macstats(self):
        return ""

    def describe(self):
        return {"kind": self.kind, "n_ue": self.n_ue}

    def close(self):
        pass


class FakeStack(Stack):
    """Local processes: one sink agent, one UDP relay shared by all UEs (delay_ms + U(0, jitter_ms), loss, and an
    optional serializing rate in bit/s that makes UEs contend), and one UE agent per UE. Wall time = virtual time."""
    kind = "fake"

    def __init__(self, n_ue=1, delay_ms=10.0, jitter_ms=0.0, loss=0.0, rate_bps=0.0, seed=0, python=None):
        self.n_ue = n_ue
        self.params = dict(delay_ms=delay_ms, jitter_ms=jitter_ms, loss=loss, rate_bps=rate_bps, seed=seed)
        py = python or sys.executable
        self.procs = []
        sink_port, sink_ctl, relay = _free_port(), _free_port(), _free_port()
        self._spawn([py, AGENT, "sink", "--ctl", f"127.0.0.1:{sink_ctl}", "--port", str(sink_port),
                     "--bind", "127.0.0.1"])
        px = subprocess.Popen([py, AGENT, "proxy", "--listen", f"127.0.0.1:{relay}", "--to", f"127.0.0.1:{sink_port}",
                               "--delay-ms", str(delay_ms), "--jitter-ms", str(jitter_ms), "--loss", str(loss),
                               "--rate-bps", str(rate_bps), "--seed", str(seed)],
                              stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        self.procs.append(px)
        px.stdout.readline()                      # the relay prints one line once its socket is bound
        ctl = []
        for _ in range(n_ue):
            p = _free_port()
            self._spawn([py, AGENT, "ue", "--ctl", f"127.0.0.1:{p}"])
            ctl.append(("127.0.0.1", p))
        self.ue_ctl = tuple(ctl)
        self.sink_ctl = ("127.0.0.1", sink_ctl)
        self.sink_dst = ("127.0.0.1", relay)
        self.ue_bind_ip = "127.0.0.1"
        for h, p in (*self.ue_ctl, self.sink_ctl):
            _wait_port(h, p)
        self.vclock = VClock(None)
        self.pathloss = [(0.0, 0.0)] * n_ue

    def _spawn(self, argv):
        self.procs.append(subprocess.Popen(argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))

    def set_pathloss(self, k, ul_db, dl_db=None):
        self.pathloss[k] = (float(ul_db), float(ul_db if dl_db is None else dl_db))

    def describe(self):
        return {"kind": self.kind, "n_ue": self.n_ue, **self.params}

    def close(self):
        for p in self.procs:                      # only the processes this stack started
            if p.poll() is None:
                p.terminate()
        for p in self.procs:
            try:
                p.wait(timeout=3)
            except subprocess.TimeoutExpired:
                p.kill()
        self.procs = []


class DockerOaiStack(Stack):
    """The deployment of deploy/docker-compose.yaml, already running (``docker compose up -d`` in a run directory made
    by deploy/make_configs.py). The bridge talks to the agents over the Docker bridge networks, to the gNB and UE
    telnet servers for channel control, and polls the gNB's rfsim clock for virtual time.

    UE k is the container ilnet-oai-ue<k+1>. Uplink channel models are per connection order on the gNB
    (rfsimu_channel_ue<k> for the k-th UE that connected), so the UEs must attach in index order, which the
    compose file's depends_on chain does.

    clock: "ttrace" (default; the gNB's T-tracer slot ticks through OAI's textlog, whose path and T_messages.txt
    come from `textlog` / `t_db` or $OAI_TEXTLOG / $OAI_T_MESSAGES; the gNB must run with --T_stdout 2 --T_nowait),
    "telnet" (``rfsimu vtime`` polls; crashed the 2026.w39 gNB, see vclock.py) or "wall" (no mapping: delays in
    wall time). t_events / t_log: further T events the same textlog process records to a file (the MAC trace)."""
    kind = "oai_rfsim"

    def __init__(self, n_ue=1, gnb_ip=GNB_IP, ue_ips=UE_IPS, sink_ip=SINK_IP, clock="ttrace", textlog=None,
                 t_db=None, t_events=(), t_log=None, mu=1, channel_control=True, run_dir=None):
        if n_ue > len(ue_ips):
            raise ValueError(f"{n_ue} UEs requested, {len(ue_ips)} addresses known")
        self.n_ue = n_ue
        self.gnb_ip, self.ue_ips, self.run_dir = gnb_ip, tuple(ue_ips[:n_ue]), run_dir
        self.ue_ctl = tuple((ip, UE_CTL_PORT) for ip in self.ue_ips)
        self.sink_ctl = (sink_ip, SINK_CTL_PORT)
        self.sink_dst = (sink_ip, SINK_PORT)
        self.ue_bind_if = "oaitun_ue1"
        for h, p in (*self.ue_ctl, self.sink_ctl):
            _wait_port(h, p, timeout=30)
        # the telnet server takes one client: one shared, thread-safe connection for the clock and the channels
        self.gnb = OaiTelnet(gnb_ip, TELNET_PORT)
        self.ue_tn = []
        self.ul_model, self.dl_model = [], []
        if channel_control:
            m = self.gnb.models()
            self.ul_model = [m[f"rfsimu_channel_ue{k}"] for k in range(n_ue)]
            for ip in self.ue_ips:
                t = OaiTelnet(ip, TELNET_PORT)
                self.ue_tn.append(t)
                self.dl_model.append(t.models()["rfsimu_channel_enB0"])
        self.channel_control = channel_control
        if clock == "ttrace":
            if textlog is None or t_db is None:
                textlog, t_db = default_textlog()
            if textlog is None:
                raise ValueError("clock='ttrace' needs textlog and t_db (or $OAI_TEXTLOG and $OAI_T_MESSAGES)")
            self.vclock = TTraceClock(textlog, t_db, host=gnb_ip, mu=mu, extra_events=t_events, log_path=t_log)
            try:
                self.vclock.wait_ready()
            except TimeoutError:
                self.vclock.close()
                raise TimeoutError("no T-tracer slot ticks: is the gNB running with --T_stdout 2 --T_nowait, and is "
                                   "another tracer (a textlog process) already connected to it?") from None
        elif clock == "telnet":
            self.vclock = VClock(self.gnb, period_s=0.1)
        elif clock == "wall":
            self.vclock = VClock(None)
        else:
            raise ValueError(f"clock {clock!r}")
        self.clock = clock
        self.pathloss = [(0.0, 0.0)] * n_ue

    def set_pathloss(self, k, ul_db, dl_db=None):
        """Extra uplink and downlink attenuation of UE k in dB, >= 0 (dl_db None = same as ul_db). rfsim's ploss is a
        gain, so the models get ploss = -attenuation."""
        if not self.channel_control:
            raise RuntimeError("channel_control=False")
        dl_db = ul_db if dl_db is None else dl_db
        self.gnb.set_ploss(self.ul_model[k], -float(ul_db))
        self.ue_tn[k].set_ploss(self.dl_model[k], -float(dl_db))
        self.pathloss[k] = (float(ul_db), float(dl_db))

    def macstats(self):
        r = subprocess.run(["docker", "exec", "ilnet-oai-gnb", "cat", "/opt/oai-gnb/nrMAC_stats.log"],
                           capture_output=True, text=True, timeout=10)
        return r.stdout

    def describe(self):
        d = {"kind": self.kind, "n_ue": self.n_ue, "gnb_ip": self.gnb_ip, "ue_ips": list(self.ue_ips),
             "clock": self.clock}
        if self.run_dir and os.path.exists(os.path.join(self.run_dir, "profile.json")):
            import json
            with open(os.path.join(self.run_dir, "profile.json")) as f:
                d["profile"] = json.load(f)
        return d

    def close(self):
        self.vclock.close()
        for t in [self.gnb, *self.ue_tn]:
            t.close()
