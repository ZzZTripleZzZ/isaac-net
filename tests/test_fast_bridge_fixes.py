"""Regression tests for the fast-backend, bridge and measurement-tool fixes of the 2026-10-04 code review.

CPU
  F1  the prototype reference copies the caller's SNR / hid buffers, so it equals the eager backend when a caller
      reuses its buffers (review item 7)
  F2  CounterRNG.reset: an env listed twice starts one episode, as with a mask; distinct ids are unchanged
  F3  NRGraphEngine.submit forwards tag / priority / deadline_ms to NREngine.submit
  F4  Ns3NetModule runs on the isaac layer's real TrafficRequest / NetConfig (mocked transport): NetModule dict,
      tags, and the legacy step(poses, req) -> NetOutput form (item 26)
  F5  OAI bridge: rx_vt.csv maps t_tx_ns as well as t_rx_ns; negative send-error markers stay negative (item 27)
  F6  owd / calibrate drop negative t_tx_ns send-error rows instead of using them as send times (item 29)
  F7  OaiTelnet: replies framed by the prompt, a late vtime reply is not taken by the next query, set_ploss /
      set_noise raise on an error reply (item 29)
  F8  unwrap_slots(realtime=False) ignores a wall clock that does not run at the slot rate (item 29)
  F9  macnr: LCID 0 = 64-bit CCCH (8 B), LCID 52 = 48-bit CCCH (6 B) (item 28, TS 38.321 Table 6.2.1-2)
GPU (marker gpu)
  G1  NR graph backend: a float64 input after a float32 capture recaptures (graph key holds the dtype)
  G2  NR graph backend == reference bitwise with submit extras (tag / priority / deadline_ms)
  G3  L2-legacy triton reads rng.env through a pointer: set_env_offset after the capture takes effect
  G4  NetFast._capture leaves the CUDA generator untouched with rng="global"
"""
import csv
import socket
import threading
import time
import warnings

import numpy as np
import pytest
import torch

from engine_api import SIZES, lookup_params
from isaac_net.core.proto import netsim as ns
from isaac_net.core.proto.netsim_fast import make_net_fast
from isaac_net.core.proto.rng import CounterRNG


# ---------------------------------------------------------------------------------------------------------- F1
def _reused_buffer_run(level, backend, steps=30):
    E, R = 3, 4
    params = {"L05": lookup_params("L05"), "L05Q": lookup_params("L05Q"),
              "L0": {"mu": -0.7, "sig": 0.5, "p": 0.05}}.get(level)
    net = make_net_fast(level, E, R, "cpu", SIZES, params=params, backend=backend, seed=7, rng="engine")
    g = torch.Generator().manual_seed(0)
    snr = torch.zeros(E, R)
    hid = torch.zeros(E, dtype=torch.long)
    outs = []
    for _ in range(steps):
        send = (torch.rand(E, R, generator=g) < 0.5).long() * (1 + (torch.rand(E, R, generator=g) < 0.3).long())
        det = torch.rand(E, R, generator=g) < 0.5
        hid.copy_(torch.randint(0, 3, (E,), generator=g))
        net.submit(None, ns.Requests(send, det, hid))              # snr_db=None: the SNR of the previous step
        snr.copy_(torch.rand(E, R, generator=g) * 40 - 5)
        hid.copy_(torch.randint(0, 3, (E,), generator=g))           # the caller reuses its hid buffer ...
        outs.append(net.step(None, snr))
        snr.fill_(-99.0)                                            # ... and its SNR buffer
        hid.fill_(9)
    return net, outs


@pytest.mark.parametrize("level", ["L05", "L05Q", "L0", "L1", "L2"])
def test_f1_reference_copies_reused_buffers(level):
    a, oa = _reused_buffer_run(level, "reference")
    b, ob = _reused_buffer_run(level, "eager")
    for n in ns.NetBase.FIELDS:
        assert torch.equal(getattr(a, n), getattr(b, n)), n
    for x, y in zip(oa, ob):
        for k in x:
            assert torch.equal(x[k].nan_to_num(-7.0), y[k].nan_to_num(-7.0)), k
    assert torch.equal(oa[-1]["sinr_db"], ob[-1]["sinr_db"]) and float(oa[-1]["sinr_db"].min()) > -99.0


# ---------------------------------------------------------------------------------------------------------- F2
def test_f2_rng_reset_duplicate_ids_idempotent():
    a, b, c = (CounterRNG(5, 4, "cpu") for _ in range(3))
    for r in (a, b, c):
        r.reset(None)
        r.tick(2)
    a.reset(torch.tensor([1, 1, 2]))
    b.reset(torch.tensor([1, 2]))
    m = torch.tensor([False, True, True, False])
    c.episode.add_(m.long())
    c.ctr[2].masked_fill_(m, 0)
    for r in (a, c):
        assert torch.equal(r.episode, b.episode) and torch.equal(r.ctr[2], b.ctr[2])
    assert b.episode.tolist() == [0, 1, 1, 0] and b.ctr[2].tolist() == [1, 0, 0, 1]
    assert torch.equal(a.reset_normal(torch.tensor([1, 2]), 0, 3), b.reset_normal(torch.tensor([1, 2]), 0, 3))


# ---------------------------------------------------------------------------------------------------------- F3
def test_f3_graph_submit_forwards_extras(monkeypatch):
    from isaac_net.core import nr_fast
    from isaac_net.core.engine import NREngine
    seen = {}

    def fake_submit(self, t, requests, snr_db=None, **kw):
        seen.update(kw)
        return "acc"

    monkeypatch.setattr(NREngine, "submit", fake_submit)
    eng = object.__new__(nr_fast.NRGraphEngine)
    eng._rebind = lambda: seen.setdefault("rebound", True)
    tag = torch.ones(2, 3, dtype=torch.long)
    assert nr_fast.NRGraphEngine.submit(eng, None, torch.zeros(2, 3), tag=tag, priority=2, deadline_ms=50.0) == "acc"
    assert seen["tag"] is tag and seen["priority"] == 2 and seen["deadline_ms"] == 50.0 and seen["rebound"]


# ---------------------------------------------------------------------------------------------------------- F4
class _FakeLockstep:
    """Stands in for Ns3Lockstep: every frame sent in a step completes at half the step, SINR = 10 dB."""

    def __init__(self, E, R, mode="procs", **kw):
        self.E, self.R = E, R
        self.epp = 1 if mode == "procs" else E
        self.G = E // self.epp
        self.resets, self.steps = [], []

    def reset(self, groups=None, pos=None, shadow=None):
        self.resets.append(list(groups))

    def step(self, pos, frames, shadow=None, interp=False):
        from isaac_net.bridges.ns3_lockstep import protocol as P
        self.steps.append((np.array(pos), frames.copy(), interp))
        done = np.zeros(len(frames), [("env", "i4"), ("ue", "i4"), ("fid", "i8"), ("frac", "f8")])
        done["env"], done["ue"], done["fid"], done["frac"] = frames["env"], frames["ue"], frames["fid"], 0.5
        out = {k: np.full((self.E, self.R), 10.0, np.float32) for k in P.RESULT_F32}
        out.update({k: np.zeros((self.E, self.R), np.uint32) for k in P.RESULT_U32})
        out["done"], out["timing"] = done, {"rtt": 0.0}
        return out

    def close(self):
        pass


def test_f4_ns3_netmodule_real_request(monkeypatch):
    from isaac_net.bridges.ns3_lockstep import netmodule_ns3 as M
    from isaac_net.isaac.net_module import NetConfig, TrafficRequest
    assert M.TrafficRequest is TrafficRequest and M.NetConfig is NetConfig
    monkeypatch.setattr(M, "Ns3Lockstep", _FakeLockstep)
    E, R = 2, 3
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        cfg = NetConfig(num_envs=E, num_robots=R, device="cpu", msg_sizes=(4000.0, 30000.0))
    net = M.Ns3NetModule(cfg)
    assert (net.F, net.timeout, net.step_dt) == (16, 20, 0.1)
    net.reset()
    pos = torch.zeros(E, R, 3)
    send = torch.tensor([[1, 0, 2], [0, 0, 0]])
    tag = torch.tensor([[4, -1, 4], [-1, -1, -1]])
    acc = net.submit(None, TrafficRequest(send, tag))
    assert torch.equal(acc, send > 0)
    out = net.step(None, pos, torch.tensor([4, 4]))
    assert isinstance(out, dict)
    for k in ("delivered", "newest_cap", "last_cap", "aoi_s", "queue_len", "queue_bytes", "sinr_db", "rsrp_dbm",
              "serving", "blocked", "msg_delivered", "timed_out", "cap", "cls", "delay_s", "t", "tag_delivered",
              "dropped", "ns3"):
        assert k in out, k
    assert out["delivered"].tolist() == [[True, False, True], [False, False, False]]
    assert out["tag_delivered"].tolist() == [True, False]
    assert out["t"].tolist() == [0, 0] and net.clock.tolist() == [1, 1]
    assert float(out["delay_s"][0, 0, 0]) == pytest.approx(0.05)
    fr = net.core.steps[-1][1]
    assert sorted(fr["bytes"].tolist()) == [4000, 30000]
    # legacy call pattern: step(poses, TrafficRequest, blocked)
    blocked = torch.zeros(E, R, 1, dtype=torch.bool)
    blocked[:, 0] = True
    o = net.step(pos, TrafficRequest(torch.ones(E, R, dtype=torch.long)), blocked=blocked)
    assert isinstance(o, M.NetOutput) and o.delivered.all() and o.blocked[:, 0].all() and "sinr_db" in o.ns3
    assert net.core.steps[-1][2] is True
    net.close()


# ---------------------------------------------------------------------------------------------------------- F5
def test_f5_oai_virtual_logs_map_tx_and_rx(tmp_path):
    from types import SimpleNamespace

    from isaac_net.bridges.oai.bridge import OaiBridge
    from isaac_net.tools.measure.probe import RX_COLS, TX_COLS
    br = object.__new__(OaiBridge)
    br.log_dir = str(tmp_path)
    br.stack = SimpleNamespace(n_ue=1)
    br.vc = SimpleNamespace(to_virtual_ns=lambda w: np.asarray(w, np.int64) // 2)     # virtual = wall / 2
    with open(tmp_path / "tx_ue1.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(TX_COLS)
        w.writerow([1, 0, 0, 0, 1, 100, 100, 1000])
        w.writerow([1, 1, 1, 0, 1, 100, 100, -2000])                    # send error marker
    with open(tmp_path / "rx.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(RX_COLS)
        w.writerow(["10.0.0.2", 1, 0, 0, 0, 1, 100, 100, 1000, 1600])
    br._write_virtual_logs()
    with open(tmp_path / "rx_vt.csv", newline="") as f:
        rx = list(csv.DictReader(f))
    with open(tmp_path / "tx_ue1_vt.csv", newline="") as f:
        tx = list(csv.DictReader(f))
    assert (int(rx[0]["t_tx_ns"]), int(rx[0]["t_rx_ns"])) == (500, 800)
    assert [int(r["t_tx_ns"]) for r in tx] == [500, -1000]


# ---------------------------------------------------------------------------------------------------------- F6
def test_f6_owd_drops_send_errors(tmp_path):
    from isaac_net.tools.measure import calibrate, owd
    from isaac_net.tools.measure.probe import RX_COLS, TX_COLS
    tx, rx = tmp_path / "tx.csv", tmp_path / "rx.csv"
    with open(tx, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(TX_COLS)
        w.writerow([1, 0, 0, 0, 1, 100, 100, 5_000_000_000])
        w.writerow([1, 1, 1, 0, 2, 2800, 1400, -4_000_000_000])         # frame 1, fragment 0: send error
        w.writerow([1, 2, 1, 1, 2, 2800, 1400, 5_100_000_000])
    with open(rx, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(RX_COLS)
        w.writerow(["a", 1, 0, 0, 0, 1, 100, 100, 5_000_000_000, 5_010_000_000])
        w.writerow(["a", 1, 2, 1, 1, 2, 2800, 1400, 5_100_000_000, 5_120_000_000])
    rows, frames, summ = owd.from_probe_logs(str(rx), str(tx), run_id="r", ue="u")
    assert len(rows) == 2 and summ["send_errors"] == 1 and summ["lost"] == 0
    assert min(r["t_tx_s"] for r in rows) == pytest.approx(5.0)
    f1 = [f for f in frames if f["frame_id"] == 1][0]
    assert f1["complete"] == 0 and f1["t_tx_first_s"] == pytest.approx(5.1)
    poisoned = frames + [dict(frames[0], frame_id=9, t_tx_first_s=-4.0)]      # a frames table from before the fix
    arr, meas, ues = calibrate._arrivals(poisoned, "r")
    assert len(arr) == 2 and arr[0][0] == pytest.approx(0.0) and arr[1][0] == pytest.approx(0.1)


# ---------------------------------------------------------------------------------------------------------- F7
def _telnet_server(script):
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]

    def serve():
        c, _ = srv.accept()
        c.sendall(b"softmodem_gnb> ")
        f = c.makefile("rb")
        n = {"vtime": 0}
        for line in f:
            script(c, line, n)
        c.close()

    threading.Thread(target=serve, daemon=True).start()
    return srv, port


def test_f7_telnet_framing_late_vtime_and_checked_commands():
    from isaac_net.bridges.oai.telnet import OaiTelnet

    def script(c, line, n):
        if line.startswith(b"rfsimu vtime"):
            n["vtime"] += 1
            c.sendall(b"softmodem_gnb> ")
            if n["vtime"] == 1:
                time.sleep(0.7)                                  # answers after the client's 0.5 s timeout
                c.sendall(b"vtime measurement: TS 46080000 sample_rate 46080000.000\n")
            else:
                c.sendall(b"vtime measurement: TS 92160000 sample_rate 46080000.000\n")
        elif b"ploss" in line:
            if b" 7 " in line:
                c.sendall(b"ERROR: channel model 7 not defined\nsoftmodem_gnb> ")
            else:
                c.sendall(b"New model path loss -10.000000\nsoftmodem_gnb> ")
        elif b"noise_power_dB" in line:
            c.sendall(b"softmodem_gnb> ")

    srv, port = _telnet_server(script)
    t = OaiTelnet("127.0.0.1", port, timeout=0.5)
    with pytest.raises(TimeoutError):
        t.vtime()
    v, _, _ = t.vtime()
    assert v == pytest.approx(2.0)                               # not the late 1.0 s reply of the first query
    assert "path loss" in t.set_ploss(0, -10)
    with pytest.raises(RuntimeError, match="not defined"):
        t.set_ploss(7, -10)
    t.set_noise(0, -30)
    t.close()
    srv.close()


# ---------------------------------------------------------------------------------------------------------- F8
def test_f8_unwrap_slots_simulated_radio():
    from isaac_net.tools.measure.schema import unwrap_slots
    seq = [(1000, 0), (1023, 9), (10, 0)]                        # mu = 0: 10 slots per frame, one SFN wrap
    true = [0, 239, 340]
    # wall clock of a slow rfsim run (slot time / wall time = 0.03): the wall clock implies extra cycles
    t_wall = [0.0, 0.239 / 0.03, 0.340 / 0.03]
    assert unwrap_slots(seq, 0, t_wall, realtime=False) == true
    assert unwrap_slots(seq, 0, [0.0, 0.239, 0.340]) == true                 # a clock at the slot rate
    assert unwrap_slots(seq, 0) == true
    assert unwrap_slots(seq, 0, t_wall) != true                              # what realtime=True does there


# ---------------------------------------------------------------------------------------------------------- F9
def test_f9_macnr_ccch_sizes():
    from isaac_net.tools.measure import macnr
    assert macnr.UL_FIXED[0] == 8 and macnr.UL_FIXED[52] == 6
    sdu = bytes([0x01, 3]) + b"abc"                                         # LCID 1, 8-bit L = 3
    msg3 = bytes([52]) + b"\x11" * 6 + sdu + bytes([macnr.UL_PADDING])     # RRCSetupRequest (48-bit CCCH)
    r = macnr.walk_ul_pdu(msg3)
    assert r["ok"] and r["lcids"] == [52, 1, 63] and r["data_bytes"] == 3
    resume = bytes([0]) + b"\x22" * 8 + sdu + bytes([macnr.UL_PADDING])    # RRCResumeRequest1 (64-bit CCCH)
    r = macnr.walk_ul_pdu(resume)
    assert r["ok"] and r["lcids"] == [0, 1, 63] and r["data_bytes"] == 3


# ---------------------------------------------------------------------------------------------------------- GPU
@pytest.mark.gpu
def test_g1_nr_graph_recaptures_on_dtype():
    from isaac_net.core import NRConfig, make_engine
    E, R = 2, 4
    ref = make_engine("L2", E, R, "cuda", NRConfig(), "reference", seed=3)
    gr = make_engine("L2", E, R, "cuda", NRConfig(), "graph", seed=3)
    g = torch.Generator(device="cuda").manual_seed(1)
    for t in range(6):
        send = torch.randint(0, 3, (E, R), device="cuda", generator=g)
        snr = 25 * torch.rand(E, R, device="cuda", generator=g)
        if t >= 3:
            snr = snr.double()
        ref.submit(None, send)
        gr.submit(None, send)
        a, b = ref.step(None, snr), gr.step(None, snr)
        for k in a:
            if torch.is_tensor(a[k]):
                assert torch.equal(a[k].nan_to_num(-7.0), b[k].nan_to_num(-7.0)), (t, k)


@pytest.mark.gpu
def test_g2_nr_graph_submit_extras_bitwise():
    from isaac_net.core import NRConfig, make_engine
    E, R = 2, 4
    ref = make_engine("L2", E, R, "cuda", NRConfig(), "reference", seed=4)
    gr = make_engine("L2", E, R, "cuda", NRConfig(), "graph", seed=4)
    g = torch.Generator(device="cuda").manual_seed(2)
    for t in range(12):
        send = torch.randint(0, 3, (E, R), device="cuda", generator=g)
        snr = 25 * torch.rand(E, R, device="cuda", generator=g)
        kw = {}
        if t >= 2:
            kw = dict(tag=torch.randint(0, 5, (E, R), device="cuda", generator=g), priority=1, deadline_ms=150.0)
        ref.submit(None, send, **kw)
        gr.submit(None, send, **kw)
        a, b = ref.step(None, snr), gr.step(None, snr)
        assert set(a) == set(b), t
        for k in a:
            if torch.is_tensor(a[k]):
                assert torch.equal(a[k].nan_to_num(-7.0), b[k].nan_to_num(-7.0)), (t, k)
    assert "tag" in a


@pytest.mark.gpu
def test_g3_legacy_triton_env_offset_after_capture():
    pytest.importorskip("triton")

    def run(recapture):
        net = make_net_fast("L2", 3, 8, "cuda", SIZES, backend="triton", seed=5, rng="engine")
        g = torch.Generator(device="cuda").manual_seed(0)
        outs = []
        for t in range(8):
            if t == 1:
                net.rng.set_env_offset(3)                       # the step graph was captured at t = 0
                if recapture:
                    net._graphs.clear()                         # a fresh capture sees the new offset in any case
            send = torch.randint(0, 3, (3, 8), device="cuda", generator=g)
            snr = 30 * torch.rand(3, 8, device="cuda", generator=g)
            net.submit(None, send)
            outs.append(net.step(None, snr))
        return outs

    for x, y in zip(run(True), run(False)):
        for k in x:
            assert torch.equal(x[k].nan_to_num(-7.0), y[k].nan_to_num(-7.0)), k


@pytest.mark.gpu
def test_g4_capture_restores_cuda_rng():
    net = make_net_fast("L2", 2, 4, "cuda", SIZES, backend="graph", seed=5, rng="global")
    torch.cuda.manual_seed(11)
    s0 = torch.cuda.get_rng_state()
    net._capture(net._step_region)
    assert torch.equal(torch.cuda.get_rng_state(), s0)
