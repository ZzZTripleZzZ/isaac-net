"""OAI bridge against the fake stack (local UDP relay with delay, jitter, loss and rate): no OAI, Docker or root.

    python -m pytest tests/bridges/oai -q          # tests/bridges is not collected by the default suite

Covers the agents' protocol, fragmentation, loss and timeouts, NetBase semantics of OaiNet (delay, FIFO, partial
reset), the Isaac NetModule front end, virtual-time pacing and mapping, the probe-format logs, the telnet parsing and
the deployment config generator.
"""
import math
import os
import socket
import threading
import time

import numpy as np
import pytest
import torch

from isaaclab_net.bridges.oai import FakeStack, OaiBridge, VClock
from isaaclab_net.bridges.oai.deploy import make_configs as MC
from isaaclab_net.bridges.oai.net import OaiNet, make_oai_netmodule
from isaaclab_net.bridges.oai.telnet import OaiTelnet
from isaaclab_net.bridges.oai.vclock import map_wall_to_virtual
from isaaclab_net.core.config import NRConfig
from isaaclab_net.tools.measure import owd

DT = 0.05


@pytest.fixture
def stack2():
    s = FakeStack(n_ue=2, delay_ms=12.0, jitter_ms=2.0)
    yield s
    s.close()


def _run_steps(br, n):
    out = []
    for _ in range(n):
        out += br.step()["done"]
    return out


def test_agents_delay_and_fragments(stack2, tmp_path):
    br = OaiBridge(stack2, step_dt=DT, pacing="wall", log_dir=str(tmp_path))
    br.reset()
    fids = br.submit([(0, "a", 100), (1, "b", 30000)])
    assert set(fids) == {(0, "a"), (1, "b")}
    done = _run_steps(br, 20)                  # generous: the relay is a Python process on a shared host
    br.close()
    got = {(d[0], d[1]): d[2] for d in done}
    assert set(got) == {(0, "a"), (1, "b")}
    assert 0.012 <= got[(0, "a")] <= 0.5 and 0.012 <= got[(1, "b")] <= 0.5     # never faster than the relay
    # logs are probe logs: owd.py reads them, 1 + 22 datagrams, none lost, delays within the relay's range
    rows, frames, summ = owd.from_probe_logs(str(tmp_path / "rx.csv"), str(tmp_path / "tx_ue2.csv"))
    assert len(rows) == math.ceil(30000 / 1400) and summ["lost"] == 0
    assert 12.0 <= summ["min_ms"] and summ["p50_ms"] <= 500.0
    assert len(frames) == 1 and frames[0]["complete"] == 1
    assert (tmp_path / "tx_ue1_vt.csv").exists() and (tmp_path / "rx_vt.csv").exists()


def test_loss_and_timeout():
    s = FakeStack(n_ue=1, delay_ms=5.0, loss=1.0)
    try:
        br = OaiBridge(s, step_dt=DT, pacing="wall")
        net = OaiNet(1, 1, "cpu", (1000.0, 4000.0), br, timeout=3)
        send = torch.ones(1, 1, dtype=torch.long)
        net.submit(None, send)
        timed = 0
        for k in range(4):
            o = net.step(None, torch.zeros(1, 1))
            assert not o["delivered"].any()
            timed += int(o["timed_out"].sum())
            if k == 2:
                assert timed == 1                  # dropped after exactly `timeout` steps
        assert timed == 1 and int(net.queued().sum()) == 0
        net.close()
    finally:
        s.close()


def test_oainet_netbase_semantics(stack2):
    br = OaiBridge(stack2, step_dt=DT, pacing="wall")
    net = OaiNet(1, 2, "cpu", (1000.0, 30000.0), br)
    delays, n_sent, n_dlv = [], 0, 0
    for k in range(20):
        send = torch.tensor([[1, 2]]) if k < 8 else torch.zeros(1, 2, dtype=torch.long)
        n_sent += int((send > 0).sum())
        net.submit(None, send)
        o = net.step(None, torch.zeros(1, 2))
        assert int(o["t"][0]) == k
        d = o["delay"][o["delivered"]]
        delays += d.tolist()
        n_dlv += int(o["delivered"].sum())
        cap = o["cap"][o["delivered"]]
        assert (cap <= k).all() and (cap + d <= k + 1 + 1e-4).all()      # finished inside the step reported
    net.close()
    assert n_dlv == n_sent
    assert 0.012 / DT - 1e-3 <= min(delays)                            # never faster than the 12 ms relay


def test_partial_reset_ignores_old_frames():
    s = FakeStack(n_ue=2, delay_ms=70.0)            # longer than a step: frames are in flight at the reset
    try:
        br = OaiBridge(s, step_dt=DT, pacing="wall")
        net = OaiNet(2, 1, "cpu", (1000.0, 4000.0), br)
        net.submit(None, torch.ones(2, 1, dtype=torch.long))
        o = net.step(None, torch.zeros(2, 1))
        assert not o["delivered"].any()
        net.reset(torch.tensor([0]))                 # env 0 restarts; env 1 keeps its frame
        dlv = torch.zeros(2, dtype=torch.long)
        for _ in range(3):
            o = net.step(None, torch.zeros(2, 1))
            dlv += o["delivered"].sum((1, 2))
        assert dlv.tolist() == [0, 1]
        assert int(net.clock[0]) == 3 and int(net.clock[1]) == 4
        net.close()
    finally:
        s.close()


def test_netmodule_front_end(stack2):
    cfg = NRConfig(control_step_ms=DT * 1000, msg_sizes=(4000.0, 30000.0))
    br = OaiBridge(stack2, step_dt=DT, pacing="wall")
    mod = make_oai_netmodule(br, num_robots=2, config=cfg, pose_chunks=1)
    mod.reset()
    got = torch.zeros(1, 2, dtype=torch.bool)
    from isaaclab_net.isaac.net_module import TrafficRequest
    for k in range(10):
        send = torch.ones(1, 2, dtype=torch.long) if k == 0 else torch.zeros(1, 2, dtype=torch.long)
        mod.submit(None, TrafficRequest(send))
        out = mod.step(None, torch.zeros(1, 2, 3) + 10.0)
        for key in ("delivered", "newest_cap", "last_cap", "aoi_s", "queue_len", "delay_s", "sinr_db"):
            assert key in out
        got |= out["delivered"]
    assert got.all()
    assert torch.allclose(out["aoi_s"], torch.full((1, 2), 10 * DT))     # capture 0 is the newest, 10 steps later
    mod.eng.close()


class _FakeVtime:
    """vtime() source whose clock runs `speed` times faster than wall time."""

    def __init__(self, speed):
        self.speed, self.w0 = speed, time.time_ns()

    def vtime(self):
        w = time.time_ns()
        return 100.0 + (w - self.w0) / 1e9 * self.speed, w, 1000


def test_vclock_mapping_and_virtual_pacing():
    w = np.array([0, 1_000_000_000, 2_000_000_000], np.int64) + 10 ** 18
    v = np.array([5.0, 6.5, 7.5])                                      # speed 1.5, then 1.0
    got = map_wall_to_virtual(w + 500_000_000, w, v) - w[0]
    assert np.allclose(got / 1e9, [0.75, 2.0, 3.125])                  # interpolated; extrapolated at the mean slope
    vc = VClock(_FakeVtime(2.0), period_s=0.005)
    try:
        time.sleep(0.05)
        assert abs(vc.speed(0.04) - 2.0) < 0.1
        s = FakeStack(n_ue=1, delay_ms=10.0)
        s.vclock.close()
        s.vclock = vc
        br = OaiBridge(s, step_dt=0.1, pacing="virtual")
        br.reset()
        t0 = time.perf_counter()
        br.submit([(0, "x", 100)])
        done = _run_steps(br, 6)
        wall = time.perf_counter() - t0
        assert 0.25 <= wall <= 0.45                                    # 0.6 s of virtual time at speed 2
        assert len(done) == 1 and 0.019 <= done[0][2] <= 0.3           # 10 ms relay = at least 20 ms virtual
        br.close()
        s.close()
    finally:
        vc.close()


def test_telnet_parsing():
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]

    def serve():
        c, _ = srv.accept()
        c.sendall(b"softmodem_gnb> ")
        f = c.makefile("rb")
        for line in f:
            if line.startswith(b"rfsimu vtime"):
                c.sendall(b"softmodem_gnb> rfsimu_vtime_cmd: vtime measurement: TS 92160000 sample_rate 46080000.000\n")
            elif line.startswith(b"channelmod show current"):
                c.sendall(b"model 0 rfsimu_channel_ue0 type AWGN:\nmodel owner: rfsimulator\n----\n"
                          b"model 1 rfsimu_channel_ue1 type AWGN:\n")
        c.close()

    threading.Thread(target=serve, daemon=True).start()
    t = OaiTelnet("127.0.0.1", port)
    v, w, rtt = t.vtime()
    assert v == pytest.approx(2.0) and rtt > 0 and abs(w - time.time_ns()) < 1e9
    assert t.models() == {"rfsimu_channel_ue0": 0, "rfsimu_channel_ue1": 1}
    t.close()
    srv.close()


def test_make_configs_profiles():
    src = ("gNBs:\n  - gNB_ID: 0xe00\n    min_rxtxtime: 6\n    servingCellConfigCommon:\n"
           "      - absoluteFrequencySSB: 621312\n        dl_absoluteFrequencyPointA: 620040\n"
           "        dl_carrierBandwidth: 106\n        initialDLBWPlocationAndBandwidth: 28875\n"
           "        initialDLBWPcontrolResourceSetZero: 11\n        ul_carrierBandwidth: 106\n"
           "        initialULBWPlocationAndBandwidth: 28875\n        dl_UL_TransmissionPeriodicity: 6\n"
           "        nrofDownlinkSlots: 7\n        nrofDownlinkSymbols: 6\n        nrofUplinkSlots: 2\n"
           "        nrofUplinkSymbols: 4\nMACRLCs:\n  - tr_s_preference: local_L1\n    pusch_TargetSNRx10: 200\n"
           "log_config:\n  rlc_log_level: info\n  pdcp_log_level: info\n  ngap_log_level: debug\n"
           "  f1ap_log_level: debug\n")
    out = MC.gnb_yaml(src, "lena_match", {"ulsch_max_frame_inactivity": "1000", "pusch_TargetSNRx10": "300"})
    for k, v in (("dl_carrierBandwidth", 51), ("initialULBWPlocationAndBandwidth", 13750),
                 ("dl_UL_TransmissionPeriodicity", 5), ("nrofDownlinkSlots", 3), ("nrofDownlinkSymbols", 10),
                 ("nrofUplinkSlots", 1), ("nrofUplinkSymbols", 2), ("absoluteFrequencySSB", 640704),
                 ("ulsch_max_frame_inactivity", 1000), ("pusch_TargetSNRx10", 300)):
        assert f"{k}: {v}\n" in out, k
    assert out.count("model_name: rfsimu_channel_ue") == MC.MAX_UE
    assert "channelmod:" in MC.ue_yaml("uicc0:\n  imsi: 1\nchannelmod:\n  old: 1\n")
    assert os.path.exists(os.path.join(os.path.dirname(MC.__file__), "docker-compose.yaml"))
