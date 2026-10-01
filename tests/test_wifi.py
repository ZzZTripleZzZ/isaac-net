"""Level WIFI (core/wifi): PHY tables, the mean-field solver against Bianchi's model and the event-driven simulator,
the engine API (dict outputs, clocks, conservation, exact timeouts, partial resets that leave other envs bitwise
unaffected), monotone delay vs contention, association / channels / hidden nodes, the validation numbers of
docs/wifi.md as regression bounds, and (GPU) the graph backend bitwise equal to the reference."""
import math

import numpy as np
import pytest
import torch

from isaac_net.core import NRConfig, Requests, make_engine
from isaac_net.core.traffic import TrafficModel
from isaac_net.core.wifi import WifiConfig, WifiNet, mcs_table
from isaac_net.core.wifi import meanfield as mf
from isaac_net.core.wifi.eventsim import Const, Station, run
from isaac_net.core.wifi.phy import AccessTiming, ctrl_us, ppdu_us
from isaac_net.core.wifi.validate import _wifi_station, engine_periodic, event_periodic, meanfield_saturated

KEYS = {"delivered", "timed_out", "cap", "cls", "delay", "newest", "det_env", "queue_len", "queue_bytes", "sinr_db",
        "t", "serving_cell", "wifi_mcs", "wifi_rate_mbps", "wifi_access_ms", "wifi_p_fail", "wifi_busy"}
E, R = 3, 6


def cfg(**kw):
    nr = {k: kw.pop(k) for k in list(kw) if k in NRConfig.__dataclass_fields__}
    return NRConfig(wifi=WifiConfig(**kw), **nr)


def net_(c=None, seed=0, En=E, Rn=R, device="cpu", backend="reference"):
    return make_engine("WIFI", En, Rn, device, c if c is not None else cfg(), seed=seed, backend=backend)


def drive(net, steps, gen, p_send=0.6, snr=(10.0, 40.0), reset_at=None, ids=None, pos=False, arena=60.0):
    outs = []
    En, Rn, d = net.E, net.R, net.dev
    for k in range(steps):
        if k == reset_at:
            net.reset(ids)
        send = (torch.rand(En, Rn, generator=gen) < p_send).long() * torch.randint(1, 3, (En, Rn), generator=gen)
        if pos:
            x = torch.rand(En, Rn, 2, generator=gen) * arena
        else:
            x = snr[0] + (snr[1] - snr[0]) * torch.rand(En, Rn, generator=gen)
        acc = net.submit(None, Requests(send.to(d)))
        o = net.step(None, x.to(d))
        o["accepted"] = acc
        outs.append(o)
    return outs


# ----------------------------------------------------------------------------- PHY
def test_phy_tables():
    r, thr = mcs_table("ax", 20)
    assert len(r) == 12 and r[0] == pytest.approx(8.6, abs=0.01) and r[7] == pytest.approx(86.0, abs=0.05)
    assert r[11] == pytest.approx(143.4, abs=0.05)
    assert thr[0] == pytest.approx(9.0, abs=0.02) and thr[11] == pytest.approx(39.0, abs=0.02)
    r, _ = mcs_table("ax", 80, n_ss=2)
    assert r[11] == pytest.approx(1201.0, abs=0.5)
    r, _ = mcs_table("ac", 20)
    assert len(r) == 9 and r[0] == pytest.approx(6.5) and r[-1] == pytest.approx(78.0)        # no MCS 9 at 20 MHz
    r, _ = mcs_table("ac", 80, gi_us=0.4)
    assert r[9] == pytest.approx(433.3, abs=0.1)
    r, thr = mcs_table("a")
    assert r == [6, 9, 12, 18, 24, 36, 48, 54] and thr[0] == pytest.approx(9.0, abs=0.02)
    assert ctrl_us(14, 24.0) == 28.0 and ctrl_us(32, 24.0) == 32.0                    # ACK, compressed Block Ack
    assert ppdu_us(1500, 54.0, "a") == 20.0 + 4.0 * math.ceil((22 + 12000) / 216)
    assert all(a < b for a, b in zip(mcs_table("ax", 20)[1], mcs_table("ax", 20)[1][1:]) if a != b)


def test_access_timing_and_cap():
    wc = WifiConfig()
    tm = AccessTiming(wc)
    rate = torch.tensor([86.0])
    ts, tc, tv = tm.times(torch.tensor([4000.0]), rate, torch.tensor([3.0]))
    assert ts.item() == pytest.approx(16 + 27 + ppdu_us(4000 * (1 + 70 / 1472), 86.0, "ax") + 16 + 32)
    assert tc.item() == pytest.approx(16 + 27 + ppdu_us(4000 * (1 + 70 / 1472), 86.0, "ax")) and tv.item() < ts.item()
    tca = AccessTiming(wc.with_(collision_time="ack")).times(torch.tensor([4000.0]), rate, torch.tensor([3.0]))[1]
    assert tca.item() == ts.item()
    wr = wc.with_(rts_cts=True)
    ts2, tc2, tv2 = AccessTiming(wr).times(torch.tensor([4000.0]), rate, torch.tensor([3.0]))
    assert tc2.item() < ts2.item() and tv2.item() == pytest.approx(28.0)
    cap = tm.cap_bytes(torch.tensor([8.6, 86.0, 1000.0]), torch.zeros(3))
    by_time = (5484.0 - 44.0) * 86.0 / 8 / (1 + 70 / 1472)                                   # aPPDUMaxTime at MCS 7
    assert cap[0] < cap[1] and cap[1].item() == pytest.approx(by_time, rel=1e-6)
    assert cap[2].item() == pytest.approx(65535 / (1 + 70 / 1472), rel=1e-6)                 # A-MPDU length limit
    vi = tm.cap_bytes(torch.tensor([86.0]), torch.tensor([3008.0]))                           # AC_VI TXOP limit
    assert vi.item() == pytest.approx((3008.0 - 44.0) * 86.0 / 8 / (1 + 70 / 1472), rel=1e-6)
    assert AccessTiming(wc.with_(max_ampdu_bytes=0)).cap_bytes(rate, torch.zeros(1)).item() == 1472


# ----------------------------------------------------------------------------- mean-field solver
@pytest.mark.parametrize("n", [1, 2, 5, 10, 20, 50])
def test_solver_matches_exact_fixed_point(n):
    f = torch.float64
    w = torch.ones(1, n, dtype=f)
    res = mf.solve(w, torch.tensor(16.0, dtype=f), torch.tensor(1024.0, dtype=f), 7, torch.full_like(w, 300.0),
                   torch.full_like(w, 300.0), 9.0, mf.DomainView(torch.ones(1, n, 1, dtype=f)), iters=60)
    tau, p = mf.fixed_point_scalar(n, 16, 1024, 7)
    assert res["tau"][0, 0].item() == pytest.approx(tau, rel=1e-9)
    assert res["p"][0, 0].item() == pytest.approx(p, rel=1e-8)


def test_tau_formula_is_bianchi():
    """(1) with no retry limit equals Bianchi's closed form 2(1-2p) / ((1-2p)(W+1) + pW(1-(2p)^m))."""
    for p in (0.05, 0.2, 0.4):
        for W, m in ((32, 3), (16, 6), (128, 5)):
            bianchi = 2 * (1 - 2 * p) / ((1 - 2 * p) * (W + 1) + p * W * (1 - (2 * p) ** m))
            assert mf.tau_scalar(p, W, W * 2 ** m, 2000) == pytest.approx(bianchi, rel=1e-12)


# Bianchi's saturation throughput (his Table I FHSS parameters), our exact root of his model. Regression values.
BIANCHI_S = {("basic", 32, 3): {5: 0.8097, 10: 0.7532, 20: 0.6788, 50: 0.5529},
             ("rts", 32, 3): {5: 0.8342, 10: 0.8371, 20: 0.8356, 50: 0.8270},
             ("basic", 128, 3): {5: 0.8250, 10: 0.8263, 20: 0.7981, 50: 0.7252}}


@pytest.mark.parametrize("case", list(BIANCHI_S))
def test_bianchi_curves(case):
    access, W, m = case
    for n, s in BIANCHI_S[case].items():
        assert mf.bianchi_throughput(n, W, m, access) == pytest.approx(s, abs=6e-5)


def test_event_sim_matches_bianchi():
    """The event-driven simulator against Bianchi's model (validation Table A: within 1.8% at 200 s)."""
    prm = mf.BIANCHI_FHSS
    ts, tc, P = mf.bianchi_times("basic")
    st = Station(W0=32, Wmax=256, aifsn=2, cap=1.0, busy_succ=Const(ts - prm["difs"]), busy_coll=Const(tc - prm["difs"]),
                 max_tx=10 ** 9)
    r = run([st] * 10, 30e6, prm["slot"], prm["sifs"], None, seed=7)
    s_ev = r["n_succ"].sum() * P / r["sim_us"]
    assert s_ev == pytest.approx(BIANCHI_S[("basic", 32, 3)][10], rel=0.03)


@pytest.mark.parametrize("n,B", [(2, 1472), (10, 1472), (5, 30000), (20, 8000)])
def test_meanfield_vs_event_saturation(n, B):
    """802.11ax 20 MHz MCS 7, AC_BE: aggregate throughput within 3% (validation Table B: within 1.6%)."""
    wc = WifiConfig(max_ampdu_bytes=0 if B == 1472 else 65535)
    rate = mcs_table("ax", 20)[0][7]
    st = _wifi_station(wc, "BE", rate, B=B)
    r = run([st] * n, 1.5e6, wc.slot_us, wc.sifs_us, None, seed=11 + n)
    m = meanfield_saturated(wc, [(n, "BE", rate, float(B))])[0]
    assert m["thr_mbps"] == pytest.approx(r["throughput_bps"].sum() / 1e6, rel=0.03)
    assert m["p"] == pytest.approx(r["n_coll"].sum() / r["n_att"].sum(), abs=0.03)


def test_ns3_reference_points():
    """ns-3.48 wifi-bianchi (802.11ax MCS 7, no aggregation, saturated), validation Table F: the mean-field model
    within 3.5 % of ns-3 with collision_time="difs" (the default)."""
    from isaac_net.core.wifi.validate import NS3_WIFI_BIANCHI
    wc = WifiConfig(msdu_payload_bytes=1500, msdu_overhead_bytes=38, max_ampdu_bytes=0, max_tx=1000)
    rate = mcs_table("ax", 20)[0][7]
    for n, (ref, _) in NS3_WIFI_BIANCHI.items():
        assert meanfield_saturated(wc, [(n, "BE", rate, 1500.0)])[0]["thr_mbps"] == pytest.approx(ref, rel=0.035)


def test_edca_priority():
    """VO takes most of the channel from BE; the model keeps the order and the total (validation Table C)."""
    wc = WifiConfig(max_ampdu_bytes=0)
    rate = mcs_table("ax", 20)[0][7]
    m = meanfield_saturated(wc, [(2, "VO", rate, 1472.0), (5, "BE", rate, 1472.0)])
    assert m[0]["thr_mbps"] > 5 * m[1]["thr_mbps"]
    sts = [_wifi_station(wc, "VO", rate)] * 2 + [_wifi_station(wc, "BE", rate)] * 5
    r = run(sts, 1.5e6, wc.slot_us, wc.sifs_us, None, seed=3)
    total = r["throughput_bps"].sum() / 1e6
    assert m[0]["thr_mbps"] + m[1]["thr_mbps"] == pytest.approx(total, rel=0.03)
    assert m[0]["thr_mbps"] == pytest.approx(r["throughput_bps"][:2].sum() / 1e6, rel=0.06)


# ----------------------------------------------------------------------------- engine API
def test_dict_outputs_and_clock(seeded):
    net = net_()
    assert isinstance(net, WifiNet) and net.level == "WIFI"
    outs = drive(net, 6, torch.Generator().manual_seed(1))
    o = outs[-1]
    assert KEYS <= set(o), KEYS - set(o)
    for k in ("delivered", "timed_out", "cap", "cls", "delay"):
        assert o[k].shape == (E, R, net.F)
    assert torch.equal(o["t"], torch.full((E,), 5)) and torch.equal(net.clock, torch.full((E,), 6))
    assert (o["wifi_mcs"] >= 0).all() and (o["wifi_rate_mbps"] > 0).all()
    d = torch.cat([x["delay"][x["delivered"]] for x in outs])
    assert (d > 0).all() and (d < 1).all()                    # light load: inside the capture step
    newest, det = net.step(None, torch.full((E, R), 30.0), torch.zeros(E, dtype=torch.long))       # legacy form
    assert newest.shape == (E, R) and det.shape == (E,)


def test_make_engine_checks():
    with pytest.raises(ValueError):
        make_engine("WIFI", 2, 2, "cpu", NRConfig(traffic=[TrafficModel.periodic(100, period_ms=10)]))
    with pytest.raises(ValueError):
        make_engine("WIFI", 2, 2, "cpu", NRConfig(wifi=WifiConfig(), olla_up_db=0.1), strict=True)
    make_engine("WIFI", 2, 2, "cpu", NRConfig(wifi=WifiConfig(), channel="tr38901_inf_sh"), strict=True)
    with pytest.raises(NotImplementedError):
        make_engine("WIFI", 2, 2, "cpu", backend="triton")
    with pytest.raises(ValueError):
        make_engine("WIFI", 2, 2, "cpu", NRConfig(control_step_ms=100.0, wifi=WifiConfig(substep_ms=3.0)))
    with pytest.raises(AssertionError):
        WifiConfig(standard="a", bandwidth_mhz=40)
    net = make_engine("WIFI", 2, 2, "cpu", NRConfig(wifi=WifiConfig(ap_positions_m=((0, 0), (50, 0)))))
    with pytest.raises(ValueError):
        net.step(None, torch.zeros(2, 2))                      # two APs need poses or rx_dbm


def test_conservation_and_timeouts(seeded):
    """accepted = delivered + timed out + queued (frames); nothing is delivered twice; an undelivered frame leaves at
    exactly the timeout. Overload so that queues, retry-limit losses and timeouts all happen."""
    c = cfg(timeout_steps=4, frame_buffer=8, msg_sizes=(30000.0, 60000.0), frame_error_rate=0.3, max_tx=2)
    net = net_(c, En=2, Rn=8)
    gen = torch.Generator().manual_seed(2)
    acc = dlv = tmo = 0
    for k in range(30):
        send = torch.randint(1, 3, (2, 8), generator=gen)
        a = net.submit(None, Requests(send))
        o = net.step(None, 5 + 30 * torch.rand(2, 8, generator=gen))
        acc += int(a.sum())
        dlv += int(o["delivered"].sum())
        tmo += int(o["timed_out"].sum())
        assert not (o["delivered"] & o["timed_out"]).any()
        age = o["t"][:, None, None] + 1 - o["cap"]
        assert (age[o["timed_out"]] == 4).all()
        assert (o["delay"][o["delivered"]] <= 4).all()
    assert acc == dlv + tmo + int(net.queued().sum())
    assert tmo > 0 and dlv > 0


def _state(net):
    names = list(net.FIELDS) + ["tau", "roam_left", "serv", "clock"]
    return {n: getattr(net, n).clone() for n in names}


@pytest.mark.parametrize("variant", ["snr", "poses", "hidden_multiap"])
def test_partial_reset_isolation(variant, seeded):
    """reset(env_ids) leaves every other env bitwise unaffected (outputs and state) and restores the reset env."""
    kw = {}
    pos = variant != "snr"
    if variant == "hidden_multiap":
        kw = dict(hidden_nodes=True, ap_positions_m=((0, 0), (60, 0), (30, 50)), n_channels=2, bg_stations=2,
                  roam_ms=10.0, rts_cts=True)
    c = cfg(**kw)
    a, b = net_(c, seed=5), net_(c, seed=5)
    ga, gb = torch.Generator().manual_seed(9), torch.Generator().manual_seed(9)
    oa = drive(a, 12, ga, reset_at=6, ids=torch.tensor([1]), pos=pos)
    ob = drive(b, 12, gb, pos=pos)
    keep = torch.tensor([0, 2])
    for x, y in zip(oa, ob):
        for k in KEYS:
            assert torch.equal(x[k][keep].nan_to_num(-7), y[k][keep].nan_to_num(-7)), k
    sa, sb = _state(a), _state(b)
    for n in sa:
        assert torch.equal(sa[n][keep], sb[n][keep]), n
    # the reset env restarted: its clock counts from the reset
    assert a.clock[1].item() == 6 and b.clock[1].item() == 12
    fresh = net_(c, seed=5)
    a.reset(torch.tensor([1]))
    for n in ("cap", "rem", "tau", "roam_left", "clock"):
        assert torch.equal(getattr(a, n)[1], getattr(fresh, n)[1]), n


def test_mask_reset_equals_index_reset(seeded):
    a, b = net_(seed=3), net_(seed=3)
    for net, ids in ((a, torch.tensor([0, 2])), (b, torch.tensor([True, False, True]))):
        drive(net, 5, torch.Generator().manual_seed(4), reset_at=3, ids=ids)
    for n, v in _state(a).items():
        assert torch.equal(v, _state(b)[n]), n


def test_determinism():
    oa = drive(net_(seed=8), 6, torch.Generator().manual_seed(1))
    ob = drive(net_(seed=8), 6, torch.Generator().manual_seed(1))
    oc = drive(net_(seed=9), 6, torch.Generator().manual_seed(1))
    assert all(torch.equal(x["delay"].nan_to_num(-1), y["delay"].nan_to_num(-1)) for x, y in zip(oa, ob))
    assert not all(torch.equal(x["delay"].nan_to_num(-1), y["delay"].nan_to_num(-1)) for x, y in zip(oa, oc))


# ----------------------------------------------------------------------------- contention behaviour
def _mean_delay(n, size, noise="mean", steps=6, **kw):
    c = cfg(msg_sizes=(float(size),), access_noise=noise, **kw)
    net = net_(c, En=2, Rn=n)
    ds, ps = [], []
    for s in range(steps):
        net.submit(None, Requests(torch.ones(2, n, dtype=torch.long)))
        o = net.step(None, torch.full((2, n), 30.0))
        ds.append(o["delay"][o["delivered"]])
        ps.append(o["wifi_p_fail"])
    return torch.cat(ds).mean().item(), torch.stack(ps).nanmean().item()


def test_delay_monotone_in_contention():
    for size in (1500, 4000, 30000):
        d, p = zip(*[_mean_delay(n, size) for n in (1, 2, 4, 8, 16)])
        assert all(x < y for x, y in zip(d, d[1:])), (size, d)
        assert p[0] == 0.0 and all(x < y for x, y in zip(p, p[1:])), (size, p)
    d = [_mean_delay(8, s)[0] for s in (1500, 4000, 16000, 30000)]
    assert all(x < y for x, y in zip(d, d[1:]))
    d_bg = [_mean_delay(4, 4000, bg_stations=b)[0] for b in (0, 2, 5)]
    assert all(x < y for x, y in zip(d_bg, d_bg[1:]))
    # lower SNR -> lower MCS -> longer delay
    c = cfg(msg_sizes=(4000.0,), access_noise="mean")
    out = []
    for snr in (35.0, 20.0, 12.0):
        net = net_(c, En=1, Rn=4)
        net.submit(None, Requests(torch.ones(1, 4, dtype=torch.long)))
        o = net.step(None, torch.full((1, 4), snr))
        out.append(o["delay"][o["delivered"]].mean().item())
    assert out[0] < out[1] < out[2]


def test_out_of_range_robot_never_sends():
    net = net_(cfg(), En=1, Rn=2)
    for _ in range(3):
        net.submit(None, Requests(torch.ones(1, 2, dtype=torch.long)))
        o = net.step(None, torch.tensor([[30.0, 2.0]]))
    assert o["wifi_mcs"][0, 1].item() == -1 and not o["delivered"][0, 1].any()
    assert o["queue_len"][0, 1].item() == 3 and o["queue_len"][0, 0].item() == 0


def test_association_channels_and_roaming():
    aps = ((0.0, 0.0), (80.0, 0.0))
    pos = torch.tensor([[[5.0, 0.0], [75.0, 0.0], [10.0, 3.0], [70.0, 2.0]]])
    sizes = (30000.0,)
    res = {}
    for nch in (1, 2):
        net = net_(cfg(ap_positions_m=aps, n_channels=nch, msg_sizes=sizes, shadow_sigma_db=0.0,
                       access_noise="mean"), En=1, Rn=4)
        net.submit(None, Requests(torch.ones(1, 4, dtype=torch.long)))
        o = net.step(None, pos)
        assert o["serving_cell"][0].tolist() == [0, 1, 0, 1]
        res[nch] = o
    # co-channel APs share the medium: more contention than on separate channels
    assert (res[1]["wifi_p_fail"] > res[2]["wifi_p_fail"]).all()
    assert res[1]["delay"][res[1]["delivered"]].mean() > res[2]["delay"][res[2]["delivered"]].mean()
    # roaming: hysteresis keeps the robot until the other AP is better by roam_hyst_db; then roam_ms without access
    net = net_(cfg(ap_positions_m=aps, roam_hyst_db=6.0, roam_ms=50.0, shadow_sigma_db=0.0), En=1, Rn=1)
    serv = []
    for x in (5.0, 38.0, 45.0, 60.0, 70.0):
        net.submit(None, Requests(torch.ones(1, 1, dtype=torch.long)))
        o = net.step(None, torch.tensor([[[x, 0.0]]]))
        serv.append(o["serving_cell"].item())
    assert serv[:3] == [0, 0, 0] and serv[-1] == 1
    assert net.roam_sub == 50


def test_hidden_nodes():
    """Two robots on opposite sides of the AP, out of each other's carrier sense: collisions rise; RTS/CTS helps; a
    user sensing matrix overrides the path-loss rule."""
    pos = torch.tensor([[[-60.0, 0.0], [60.0, 0.0]]])
    kw = dict(ap_positions_m=((0.0, 0.0),), msg_sizes=(30000.0,), shadow_sigma_db=0.0, access_noise="mean",
              pl_const_db=30.0, pathloss_exp=2.5)
    p = {}
    for name, extra in (("sensed", dict(hidden_nodes=False)), ("hidden", dict(hidden_nodes=True)),
                        ("hidden_rts", dict(hidden_nodes=True, rts_cts=True))):
        net = net_(cfg(**kw, **extra), En=1, Rn=2)
        net.submit(None, Requests(torch.ones(1, 2, dtype=torch.long)))
        p[name] = net.step(None, pos)["wifi_p_fail"][0, 0].item()
    assert p["hidden"] > 3 * p["sensed"] and p["hidden_rts"] < p["hidden"]
    net = net_(cfg(**kw, hidden_nodes=True), En=1, Rn=2)
    net.submit(None, Requests(torch.ones(1, 2, dtype=torch.long)))
    o = net.step(None, pos, sense=torch.ones(1, 2, 2, dtype=torch.bool))
    assert o["wifi_p_fail"][0, 0].item() == pytest.approx(p["sensed"], rel=1e-5)


def test_edca_classes_in_engine():
    """class_ac: a robot contends with the AC of its head-of-line message; VO messages get through first."""
    c = cfg(msg_sizes=(4000.0, 4000.0), class_ac=("VO", "BK"), access_noise="mean")
    net = net_(c, En=1, Rn=8)
    send = torch.tensor([[1, 1, 1, 1, 2, 2, 2, 2]])
    net.submit(None, Requests(send))
    o = net.step(None, torch.full((1, 8), 30.0))
    d = o["delay"][0, :, 0]
    assert d[:4].max() < d[4:].min()
    assert net.Z == 6 and net.per_class_ac


# ----------------------------------------------------------------------------- validation regression (docs/wifi.md)
def test_engine_vs_event_sim_periodic():
    """Level WIFI against the event-driven simulator, one message per robot per 100 ms step (validation Table D).
    Bounds: mean delay within 25%, 95th percentile within 35% (docs/wifi.md reports the full table)."""
    wc = WifiConfig()
    for n, size in ((10, 4000), (5, 30000)):
        d, deliv, sent = engine_periodic(n, size, 36, 8, wc)
        de = []
        for rep in range(4):
            args, steps, step_ms = event_periodic(n, size, 36, wc, seed=50 + rep)
            r = run(*args[:5], seed=args[5], timeout_us=args[6])
            arr = np.concatenate(r["msg_arrival_us"]) / 1e5
            de.append((np.concatenate(r["msg_delay_us"]) / 1000.0)[(arr >= 2) & (arr < 16)])
        de = np.concatenate(de)
        assert d.mean() == pytest.approx(de.mean(), rel=0.25), (n, size, d.mean(), de.mean())
        assert np.percentile(d, 95) == pytest.approx(np.percentile(de, 95), rel=0.35)
        assert deliv >= 0.99 * sent


# ----------------------------------------------------------------------------- GPU
@pytest.mark.gpu
@pytest.mark.parametrize("variant", ["default", "hidden_multiap", "mean"])
def test_graph_equals_reference(variant):
    kw = {"default": {}, "mean": dict(access_noise="mean", bg_stations=3),
          "hidden_multiap": dict(hidden_nodes=True, ap_positions_m=((0, 0), (60, 0)), n_channels=1, rts_cts=True,
                                 class_ac=("VI", "BE"))}[variant]
    c = cfg(**kw)
    ref = net_(c, seed=4, En=16, Rn=8, device="cuda")
    gr = net_(c, seed=4, En=16, Rn=8, device="cuda", backend="graph")
    pos = variant == "hidden_multiap"
    oa = drive(ref, 10, torch.Generator().manual_seed(2), reset_at=5, ids=torch.tensor([3, 7], device="cuda"), pos=pos)
    ob = drive(gr, 10, torch.Generator().manual_seed(2), reset_at=5, ids=torch.tensor([3, 7], device="cuda"), pos=pos)
    for x, y in zip(oa, ob):
        for k in KEYS:
            assert torch.equal(x[k].nan_to_num(-7), y[k].nan_to_num(-7)), k
    for n, v in _state(ref).items():
        assert torch.equal(v, _state(gr)[n]), n
