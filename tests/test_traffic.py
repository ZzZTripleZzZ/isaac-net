"""Traffic models (traffic.TrafficModel / TrafficGen) and their integration in the NR engine (level L2).

Generator: arrival counts, times and sizes per model, fixed shapes, partial reset, CUDA-graph capture == eager.
Engine: byte conservation, sub-step periodic arrivals get the delays of the same messages sent by the policy at a
10 ms control step, partial reset isolation, the per-message extras, rejection by every other level.
"""
import math

import pytest
import torch

from isaaclab_net.core import NRConfig, Requests, make_engine
from isaaclab_net.core.engine import LEVELS
from isaaclab_net.core.traffic import TrafficGen, TrafficModel

TM = TrafficModel
E, R = 3, 4


def _gen(models, steps_ms=100.0, N=200, seed=0, device="cpu", E=E, R=R):
    return TrafficGen(models, E, R, device, steps_ms, N, seed=seed)


def _times(arr, N=200, step_ms=100.0):
    return arr.slot.double() * step_ms / N


# ---------------------------------------------------------------------------------------------- generator
def test_periodic_aligned_substep_exact_slots():
    g = _gen([TM.periodic(300, period_ms=10, phase="aligned")])
    assert g.M == 11
    for _ in range(5):
        a = g.step()
        assert a.valid.shape == (E, R, 11)
        assert (a.valid.sum(-1) == 10).all()
        s = a.slot[..., :10]
        assert (s == torch.arange(0, 200, 20)).all()
        assert (a.nbytes[a.valid] == 300).all()
    assert int(g.deferred) == 0


@pytest.mark.parametrize("period", [7.0, 30.0, 100.0, 250.0])
def test_periodic_random_phase_counts(period):
    g = _gen([TM.periodic(100, period_ms=period, jitter_ms=period / 3)], E=8, R=5)
    K = 40
    n = torch.zeros(8, 5, dtype=torch.long)
    for _ in range(K):
        a = g.step()
        n += a.valid.sum(-1)
        # sorted by slot among the valid ones, invalid last
        key = torch.where(a.valid, a.slot, torch.full_like(a.slot, 10 ** 6))
        assert (key[..., 1:] >= key[..., :-1]).all()
    exp = K * 100.0 / period
    assert ((n - exp).abs() <= 2).all(), (n, exp)
    assert int(g.deferred) == 0


def test_periodic_jitter_bounds_no_drift():
    g = _gen([TM.periodic(100, period_ms=25, jitter_ms=5, phase="aligned")], E=2, R=2)
    for k in range(20):
        a = g.step()
        t = _times(a)[a.valid].view(2, 2, 4)
        nominal = torch.tensor([0.0, 25.0, 50.0, 75.0], dtype=torch.float64)
        assert ((t - nominal) >= -1e-9).all() and ((t - nominal) < 5.0).all()


def test_periodic_robot_subset_and_tags():
    g = _gen([TM.periodic(100, 50, phase="aligned").on([1, 3]), TM.periodic(200, 100, phase="aligned", tag=9)
              .on(0)])
    a = g.step()
    n = a.valid.sum(-1)
    assert (n[:, 0] == 1).all() and (n[:, 1] == 2).all() and (n[:, 2] == 0).all() and (n[:, 3] == 2).all()
    assert set(a.tag[a.valid].tolist()) == {1, 9}
    assert (a.nbytes[a.valid & (a.tag == 9)] == 200).all()


def test_video_gop_pattern_and_bytes():
    g = _gen([TM.video(fps=25, gop=(10_000, 1_000, 5), phase="aligned")], E=1, R=1)
    sizes = []
    for _ in range(10):
        a = g.step()
        sizes += a.nbytes[a.valid].tolist()
    assert len(sizes) == 25                      # 25 fps over 1 s
    assert sizes == [10_000.0 if k % 5 == 0 else 1_000.0 for k in range(25)]
    m = TM.video(fps=30, mean_frame_bytes=2_800, gop=(10_000, 1_000, 5))
    assert abs((m.gop[0] + 4 * m.gop[1]) / 5 - 2_800) < 1e-6
    assert abs(m.gop[0] / m.gop[1] - 10.0) < 1e-9


def test_bursty_bursts_and_rate():
    bs = 3
    g = _gen([TM.bursty(500, rate_hz=40, burst_size=bs, on_off=(0.5, 1.5))], E=64, R=8, seed=1)
    K, n = 100, 0
    for _ in range(K):
        a = g.step()
        v = a.valid
        n += int(v.sum())
        assert int(v.sum(-1).remainder(bs).sum()) == 0        # whole bursts
        # the bs messages of a burst share one arrival slot
        s = torch.where(v, a.slot, torch.full_like(a.slot, -1)).view(64, 8, -1, bs)
        assert (s == s[..., :1]).all()
    exp = 64 * 8 * K * 0.1 * 40 * bs * 0.25            # rate x burst x duty cycle 0.5 / 2.0
    assert abs(n - exp) / exp < 0.1, (n, exp)
    assert int(g.deferred) <= 0.001 * n + 5


def test_bursty_always_on_is_poisson():
    g = _gen([TM.bursty(100, rate_hz=20, burst_size=1, on_off=(1.0, 0.0))], E=128, R=4, seed=2)
    cnt = torch.stack([g.step().valid.sum(-1) for _ in range(50)]).double()
    lam = 2.0
    assert abs(cnt.mean() - lam) < 0.05 and abs(cnt.var() - lam) < 0.15


def test_event_triggers():
    g = _gen([TM.event(4000, trigger="alarm", det=True), TM.event(10, trigger=lambda clk: clk % 2 == 0).on(0)])
    m = torch.zeros(E, R, dtype=torch.bool)
    m[1, 2] = True
    a = g.step(clock=torch.tensor([0, 1, 2]), triggers={"alarm": m})
    big = a.valid & (a.nbytes == 4000)
    assert big.sum() == 1 and big[1, 2].any() and a.det[big].all()
    small = a.valid & (a.nbytes == 10)
    assert small.sum(-1)[:, 0].tolist() == [1, 0, 1] and small[:, 1:].sum() == 0
    assert (a.slot[a.valid] == 0).all()
    a = g.step(clock=torch.tensor([1, 1, 1]))                     # no triggers given: no alarm messages
    assert not (a.valid & (a.nbytes == 4000)).any()


def test_generator_partial_reset_isolation():
    ms = [TM.periodic(100, 30), TM.bursty(200, 30, 2, (0.3, 0.3)), TM.video(20, 1000, jitter_ms=3)]
    g1, g2 = _gen(ms, seed=5), _gen(ms, seed=5)
    for k in range(12):
        if k == 4:
            g2.reset(torch.tensor([False, True, False]))
        a, b = g1.step(), g2.step()
        for f in ("valid", "nbytes", "slot"):
            x, y = getattr(a, f), getattr(b, f)
            assert torch.equal(x[[0, 2]], y[[0, 2]]), (k, f)


@pytest.mark.gpu
def test_generator_cuda_graph_equals_eager():
    ms = [TM.periodic(300, 10, jitter_ms=2), TM.bursty(500, 40, 2, (0.2, 0.2)), TM.video(30, None, (9000, 900, 10)),
          TM.event(50, trigger="a")]
    dev = "cuda"
    trig = {"a": torch.zeros(8, 4, dtype=torch.bool, device=dev)}
    eager = _gen(ms, seed=3, device=dev, E=8, R=4)
    graph = _gen(ms, seed=3, device=dev, E=8, R=4)
    ref = []
    for k in range(6):
        trig["a"].copy_(torch.rand(8, 4, device=dev, generator=torch.Generator(dev).manual_seed(k)) < 0.3)
        a = eager.step(triggers=trig)
        ref.append((a.valid.clone(), a.nbytes.clone(), a.slot.clone()))
    # capture one step; the generator's state lives in fixed buffers and its RNG is registered with the graph
    trig["a"].zero_()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    cg = torch.cuda.CUDAGraph()
    cg.register_generator_state(graph.gen)
    held = {id(st): {n: getattr(st, n).clone() for n in st.state} for st in graph.streams}
    with torch.cuda.stream(s):
        graph.step(triggers=trig)                 # warm-up (advances state): restore it below
    torch.cuda.current_stream().wait_stream(s)
    for st in graph.streams:
        for n in st.state:
            getattr(st, n).copy_(held[id(st)][n])
    graph.gen.manual_seed(3)
    with torch.cuda.graph(cg):
        out = graph.step(triggers=trig)
    for k in range(6):
        trig["a"].copy_(torch.rand(8, 4, device=dev, generator=torch.Generator(dev).manual_seed(k)) < 0.3)
        cg.replay()
        v, b, sl = ref[k]
        assert torch.equal(out.valid, v) and torch.equal(out.nbytes, b) and torch.equal(out.slot, sl), k


# ---------------------------------------------------------------------------------------------- engine
def _drain(net, snr, steps, **kw):
    outs = []
    for _ in range(steps):
        outs.append(net.step(None, snr, **kw))
    return outs


def _queued_bytes(net):
    q = net.net.ul.q
    return int(((q.end - q.start) * (q.cap >= 0)).sum())


def test_engine_byte_conservation_all_models():
    ms = [TM.periodic(700, 20, jitter_ms=4).on([0, 1]), TM.video(15, 3000, gop=(6000, 2000, 4)).on(2),
          TM.bursty(1400, 30, 3, (0.4, 0.4)).on(3), TM.event(900, trigger="alarm")]
    cfg = NRConfig(traffic=ms, frame_buffer=64)
    net = make_engine("L2", E, R, "cpu", cfg, seed=0)
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(9)
    snr = torch.full((E, R), 12.0)
    out_b = out_n = 0
    acc_policy = 0
    for k in range(30):
        send = (torch.rand(E, R, generator=gen) < 0.3).long()
        acc = net.submit(None, Requests(send))
        acc_policy += int((acc * 4000).sum())
        alarm = torch.rand(E, R, generator=gen) < 0.2
        o = net.step(None, snr, triggers={"alarm": alarm})
        gone = o["delivered"] | o["timed_out"] | o["dropped"]
        out_b += int(o["bytes"][gone].sum())
        out_n += int(gone.sum())
        assert o["gen_accepted"].shape == (E, R) and o["gen_bytes"].shape == (E, R)
    st = net.traffic_stats
    assert int(st["generated"]) > 100 and int(st["refused"]) == int(st["generated"]) - int(st["accepted"])
    assert int(st["accepted_bytes"]) + acc_policy == out_b + _queued_bytes(net)


def _delays_ms(outs, step_ms):
    """Per robot, the delivered messages' (arrival ms, delay ms) sorted by arrival."""
    res = {}
    for o in outs:
        arr = o["arrival"] if "arrival" in o else o["cap"].double()
        d = o["delivered"]
        for e, r, f in d.nonzero().tolist():
            res.setdefault((e, r), []).append((float(arr[e, r, f]) * step_ms, float(o["delay"][e, r, f]) * step_ms))
    return {k: sorted(v) for k, v in res.items()}


@pytest.mark.parametrize("fading", [False, True])
def test_substep_periodic_equals_policy_at_short_step(fading):
    """A 10 ms periodic model inside 100 ms steps gives every message the delay the same message gets when the
    policy submits it at a 10 ms control step: arrival slots gate the MAC and the delay counts from them."""
    size, steps = 1200, 6
    # rng="global": both engines consume one global stream in the same slot order (the engine RNG keys the draws by
    # control step and slot index, which differ between a 100 ms and a 10 ms step)
    base = dict(frame_buffer=64, msg_sizes=(float(size),), fading=fading, timeout_steps=1000, rng="global")
    a = make_engine("L2", 2, 3, "cpu", NRConfig(traffic=TM.periodic(size, 10, phase="aligned"), **base), seed=4)
    b = make_engine("L2", 2, 3, "cpu", NRConfig(control_step_ms=10.0, **base), seed=4)
    snr = torch.tensor([[3.0, 10.0, 20.0], [0.0, 6.0, 25.0]])
    torch.manual_seed(11)
    oa = _drain(a, snr, steps + 2)
    torch.manual_seed(11)
    ob = []
    for k in range(10 * (steps + 2)):
        b.submit(None, Requests(torch.ones(2, 3, dtype=torch.long)))
        ob.append(b.step(None, snr))
    da, db = _delays_ms(oa, 100.0), _delays_ms(ob, 10.0)
    assert set(da) == set(db)
    for key in da:
        n = min(len(da[key]), len(db[key]))
        assert n >= 10, (key, n)
        x, y = torch.tensor(da[key][:n]), torch.tensor(db[key][:n])
        assert torch.allclose(x, y, atol=1e-4), (key, x[:12], y[:12])
        assert (x[:, 1] > 0).all()


@pytest.mark.parametrize("model", [TM.periodic(300, 7, jitter_ms=2), TM.video(25, 5000, (20_000, 2_000, 5)),
                                   TM.bursty(1400, 40, 4, (0.5, 1.0)), TM.event(800, trigger="a")])
def test_no_delivery_before_arrival(model):
    """Every generated message completes at least one slot after its arrival slot (regression: empty entries of the
    fixed-width arrival tensor must never open the stream gate)."""
    cfg = NRConfig(traffic=[model.on([1, 2])], frame_buffer=64, fading=False)
    net = make_engine("L2", 4, 3, "cpu", cfg, seed=0)
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(1)
    n = 0
    for _ in range(8):
        o = net.step(None, torch.full((4, 3), 20.0), triggers={"a": torch.rand(4, 3, generator=gen) < 0.5})
        d = o["delay"][o["delivered"]]
        n += d.numel()
        assert (d >= 1.0 / cfg.slots_per_step - 1e-6).all(), d.min()
    assert n > 0


@pytest.mark.parametrize("ho_rlc", ["carry", "flush"])
def test_multicell_traffic_conservation_and_arrival(ho_rlc):
    """Several NR cells with handovers: generated bytes are conserved, and no message completes before it arrives
    (a flush at a handover spares messages that arrive later in the step)."""
    from isaaclab_net.core import multicell
    cfg = multicell(3, traffic=[TM.periodic(600, 10, jitter_ms=2), TM.bursty(1400, 30, 2, (0.3, 0.3)).on(0)],
                    frame_buffer=64, ho_rlc=ho_rlc, a3_ttt_ms=0.0, a3_hyst_db=0.5)
    E_, R_ = 4, 4
    net = make_engine("L2", E_, R_, "cpu", cfg, seed=0)
    ul, hooked = net.net.ul, net.net.ul.handover
    seen = {"ho": 0}

    def check(ho, flush=False):                   # messages not yet arrived must survive a handover flush
        q = ul.q
        pending = (q.cap >= 0) & (q.start >= q.enq[..., None]) & ~q.lost
        r = hooked(ho, flush)
        seen["ho"] += int(ho.sum())
        assert not (pending & q.lost).any()
        return r

    ul.handover = check
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(2)
    pos = torch.rand(E_, R_, 2, generator=gen) * 150
    vel = (torch.rand(E_, R_, 2, generator=gen) - 0.5) * 60
    out_b = 0
    for k in range(20):
        pos = (pos + vel).clamp(0, 150)
        o = net.step(None, pos)
        gone = o["delivered"] | o["timed_out"] | o["dropped"]
        out_b += int(o["bytes"][gone].sum())
        d = o["delay"][o["delivered"]]
        assert (d >= 1.0 / cfg.slots_per_step - 1e-6).all(), d.min()
    assert int(net.traffic_stats["accepted_bytes"]) == out_b + _queued_bytes(net)
    assert seen["ho"] > 0


def test_edge_loop_sees_arrival_offsets():
    """EdgeLoop over L2 with a traffic model: a message reaches the edge at capture + in-step offset + delay."""
    from isaaclab_net.core import EdgeConfig
    cfg = NRConfig(traffic=TM.periodic(300, 100, phase="random"), fading=False,
                   edge=EdgeConfig(service_ms=1.0, return_path="instant"))
    net = make_engine("L2", 8, 2, "cpu", cfg, seed=0)
    torch.manual_seed(0)
    checked = 0
    for _ in range(6):
        o = net.step(None, torch.full((8, 2), 20.0))
        d = o["delivered"]
        one = d.sum(-1) == 1
        ul = torch.where(d, o["arrival"] - o["cap"].double() + o["delay"].double(), torch.zeros_like(o["arrival"]))
        ul = ul.sum(-1)
        cap = torch.where(d, o["cap"], torch.full_like(o["cap"], -1)).max(-1).values
        m = o["act_new"] & one & (o["act_cap"] == cap)
        checked += int(m.sum())
        assert torch.allclose(o["act_ul_delay"][m].double(), ul[m], atol=1e-5)
    assert checked > 10


def test_substep_delay_counts_from_arrival_slot():
    cfg = NRConfig(traffic=TM.periodic(200, 25, phase="aligned"), fading=False, proactive_grant="every_ul_slot")
    net = make_engine("L2", 1, 1, "cpu", cfg, seed=0)
    torch.manual_seed(0)
    outs = _drain(net, torch.full((1, 1), 25.0), 12)
    d = _delays_ms(outs, 100.0)[(0, 0)]
    assert [t for t, _ in d[:8]] == [0.0, 25.0, 50.0, 75.0, 100.0, 125.0, 150.0, 175.0]
    # 25 ms = 50 slots = 10 TDD periods: every arrival offset sees the same slot pattern, so the modal delay
    # (first-transmission success) is the same for the four offsets inside a step
    by_off = {}
    for t, x in d:
        by_off.setdefault(round(t % 100.0, 6), []).append(round(x, 6))
    modes = {k: max(set(v), key=v.count) for k, v in by_off.items()}
    assert sorted(modes) == [0.0, 25.0, 50.0, 75.0] and len(set(modes.values())) == 1
    assert all(0 < x < 10.0 for v in by_off.values() for x in v if x == modes[0.0])
    assert all(o["arrival_slot"][o["delivered"]].remainder(50).eq(0).all() for o in outs)


def test_engine_partial_reset_isolation_with_traffic():
    ms = [TM.periodic(500, 15), TM.bursty(900, 20, 2, (0.3, 0.3)).on([1, 2])]
    cfg = NRConfig(traffic=ms)
    x, y = make_engine("L2", E, R, "cpu", cfg, seed=2), make_engine("L2", E, R, "cpu", cfg, seed=2)
    snr = torch.full((E, R), 8.0)
    torch.manual_seed(0)
    ox = _drain(x, snr, 10)
    torch.manual_seed(0)
    oy = []
    for k in range(10):
        if k == 4:
            y.reset([1])
        oy.append(y.step(None, snr))
    for k, (a, b) in enumerate(zip(ox, oy)):
        for key in ("delivered", "delay", "cap", "arrival", "tag", "bytes"):
            u, v = a[key][[0, 2]], b[key][[0, 2]]
            assert torch.equal(u.nan_to_num(-7), v.nan_to_num(-7)), (k, key)
    assert int(oy[4]["t"][1]) == 0 and int(oy[4]["t"][0]) == 4


def test_submit_extras_and_deadline():
    cfg = NRConfig(fading=False)
    net = make_engine("L2", 1, 2, "cpu", cfg, seed=0)
    net.submit(None, Requests(torch.tensor([[1, 2]])), tag=torch.tensor([[5, 6]]), priority=3,
               deadline_ms=torch.tensor([[1e9, 1.0]]))
    outs = _drain(net, torch.full((1, 2), 15.0), 5)
    seen = {}
    for o in outs:
        for e, r, f in o["delivered"].nonzero().tolist():
            seen[r] = (int(o["tag"][e, r, f]), int(o["priority"][e, r, f]), bool(o["deadline_miss"][e, r, f]))
    assert seen == {0: (5, 3, False), 1: (6, 3, True)}


def test_no_traffic_keeps_outputs_bitwise():
    """policy() alone creates no generator and no extras: the same outputs as no traffic field at all."""
    a = make_engine("L2", E, R, "cpu", NRConfig(), seed=1)
    b = make_engine("L2", E, R, "cpu", NRConfig(traffic=[TM.policy()]), seed=1)
    assert b.traffic is None and not b._extras
    gen = torch.Generator().manual_seed(0)
    for k in range(6):
        send = (torch.rand(E, R, generator=gen) < 0.5).long()
        snr = 5 + 20 * torch.rand(E, R, generator=gen)
        torch.manual_seed(k)
        a.submit(None, Requests(send))
        oa = a.step(None, snr)
        torch.manual_seed(k)
        b.submit(None, Requests(send))
        ob = b.step(None, snr)
        assert set(oa) == set(ob)
        for key in oa:
            assert torch.equal(oa[key].nan_to_num(-1), ob[key].nan_to_num(-1)), key


@pytest.mark.parametrize("level", [lv for lv in LEVELS if lv != "L2"])
def test_other_levels_refuse_traffic_models(level):
    cfg = NRConfig(traffic=[TM.periodic(100, 10), TM.policy()])
    assert "traffic" in cfg.unused_fields(level) and "traffic" not in cfg.unused_fields("L2")
    with pytest.raises(ValueError, match="traffic"):
        make_engine(level, E, R, "cpu", cfg)
    assert "traffic" not in NRConfig(traffic=TM.policy()).unused_fields(level)


def test_config_normalizes_and_validates():
    c = NRConfig(traffic=TM.periodic(100, 10))
    assert isinstance(c.traffic, tuple) and len(c.traffic) == 1
    assert NRConfig(traffic=[]).traffic is None
    with pytest.raises(TypeError):
        NRConfig(traffic=["periodic"])
    with pytest.raises(AssertionError):
        TM.periodic(100, 10, jitter_ms=10)
    with pytest.raises(ValueError, match="robot"):
        make_engine("L2", 1, 2, "cpu", NRConfig(traffic=TM.periodic(100, 10).on([5])))
    assert TM.periodic(1, 7.0).max_per_step(100.0) == 15 and TM.periodic(1, 250.0).max_per_step(100.0) == 1
    assert math.isinf(TM.event(1).deadline_ms)
