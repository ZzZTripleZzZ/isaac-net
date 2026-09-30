"""EdgeLoop (core/edge.py): edge compute stage, return path and loop accounting on top of any engine.

Exact timing against hand-computed schedules (a scripted engine that delivers messages at chosen times),
conservation (arrived = completed + dropped + at the edge) on real levels, exactness when the event budget runs
out, partial resets that leave other envs bitwise unaffected, make_engine integration, the NR downlink return
path, and (gpu) CUDA-graph capture of the edge stage around the L2-legacy graph backend.
"""
import math

import pytest
import torch

from isaaclab_net.core import EdgeConfig, EdgeLoop, NRConfig, Requests, make_engine, netslot_compat

CFG = NRConfig(frame_buffer=4)


class Scripted:
    """Engine stub with the contract API: step t delivers the messages of plan[t] = [(e, r, cap, arrival, cls)]."""

    def __init__(self, E, R, plan, cfg=CFG, device="cpu"):
        self.E, self.R, self.dev, self.config, self.plan = E, R, torch.device(device), cfg, plan
        self.F = cfg.frame_buffer
        self.clock = torch.zeros(E, dtype=torch.long, device=self.dev)

    def reset(self, env_ids=None):
        if env_ids is None:
            self.clock.zero_()
        else:
            self.clock[torch.as_tensor(env_ids)] = 0

    def submit(self, t, requests, snr_db=None):
        return requests.send > 0

    def step(self, t=None, x=None):
        E, R, F, d = self.E, self.R, self.F, self.dev
        t = self.clock.clone()
        dlv = torch.zeros(E, R, F, dtype=torch.bool, device=d)
        cap = torch.full((E, R, F), -1, dtype=torch.long, device=d)
        cls = torch.zeros(E, R, F, dtype=torch.long, device=d)
        delay = torch.full((E, R, F), math.nan, device=d)
        used = {}
        for e, r, c, a, k in self.plan.get(int(t[0]), []):
            f = used.get((e, r), 0)
            used[(e, r)] = f + 1
            dlv[e, r, f], cap[e, r, f], cls[e, r, f], delay[e, r, f] = True, c, k, a - c
        self.clock += 1
        return {"delivered": dlv, "cap": cap, "cls": cls, "delay": delay, "t": t,
                "sinr_db": torch.full((E, R), 20.0, device=d)}


def _run(plan, cfg, steps, E=1, R=4):
    net = EdgeLoop(Scripted(E, R, plan), cfg)
    return net, [net.step(None) for _ in range(steps)]


def _done_times(outs, e=0):
    """{(robot, cap): completion time} of the newest result per robot and step."""
    res = {}
    for o in outs:
        for r in range(o["edge_done_cap"].shape[1]):
            c = int(o["edge_done_cap"][e, r])
            if c >= 0:
                res[(r, c)] = round(float(o["edge_done_time"][e, r]), 6)
    return res


def test_fifo_single_server_deterministic():
    plan = {0: [(0, r, 0, 0.0, 1) for r in range(3)]}
    net, outs = _run(plan, EdgeConfig(service_ms=30.0), 2)
    o = outs[0]
    assert o["edge_done_time"][0, :3].tolist() == pytest.approx([0.3, 0.6, 0.9])
    assert o["edge_done"][0].tolist() == [1, 1, 1, 0]
    assert o["act_new"][0].tolist() == [True, True, True, False]
    assert o["act_latency"][0, :3].tolist() == pytest.approx([0.3, 0.6, 0.9])
    assert o["act_edge_delay"][0, :3].tolist() == pytest.approx([0.3, 0.6, 0.9])
    assert o["act_ul_delay"][0, :3].tolist() == [0.0, 0.0, 0.0] and o["act_ret_delay"][0, :3].tolist() == [0, 0, 0]
    assert math.isnan(float(o["act_time"][0, 3])) and int(o["act_cap"][0, 3]) == -1
    assert o["act_age"][0].tolist() == [1.0, 1.0, 1.0, 2.0]
    assert outs[1]["act_age"][0, :3].tolist() == [2.0, 2.0, 2.0] and not outs[1]["act_new"].any()


def test_fifo_two_servers_and_step_boundary():
    plan = {0: [(0, r, 0, 0.1, 1) for r in range(4)]}
    _, outs = _run(plan, EdgeConfig(servers_per_env=2, service_ms=30.0), 1)
    assert outs[0]["edge_done_time"][0].tolist() == pytest.approx([0.4, 0.4, 0.7, 0.7])
    # 2.5 steps of service from 0.5 ends exactly at the end of step 2
    _, outs = _run({0: [(0, 0, 0, 0.5, 1)]}, EdgeConfig(service_ms=250.0), 4)
    assert [int(o["edge_done"][0, 0]) for o in outs] == [0, 0, 1, 0]
    assert float(outs[2]["edge_done_time"][0, 0]) == pytest.approx(3.0)
    assert [int(o["edge_in_service"][0]) for o in outs] == [1, 1, 0, 0]


def test_waiting_across_steps_and_class_table():
    # class 2 takes 150 ms: robot 0 (class 2) at 0.2 -> 1.7; robot 1 (class 1, 40 ms) at 0.3 waits -> 2.1
    plan = {0: [(0, 0, 0, 0.2, 2), (0, 1, 0, 0.3, 1)]}
    _, outs = _run(plan, EdgeConfig(service_ms=(40.0, 150.0)), 3)
    assert _done_times(outs) == {(0, 0): 1.7, (1, 0): 2.1}


def test_processor_sharing():
    plan = {0: [(0, 0, 0, 0.0, 1), (0, 1, 0, 0.2, 1)]}
    _, outs = _run(plan, EdgeConfig(discipline="ps", service_ms=40.0), 1)
    assert outs[0]["edge_done_time"][0, :2].tolist() == pytest.approx([0.6, 0.8])
    # two servers of capacity: no sharing below two jobs
    _, outs = _run(plan, EdgeConfig(discipline="ps", servers_per_env=2, service_ms=40.0), 1)
    assert outs[0]["edge_done_time"][0, :2].tolist() == pytest.approx([0.4, 0.6])
    # three jobs on two servers: rate 2/3 each
    plan = {0: [(0, r, 0, 0.0, 1) for r in range(3)]}
    _, outs = _run(plan, EdgeConfig(discipline="ps", servers_per_env=2, service_ms=40.0), 1)
    assert outs[0]["edge_done_time"][0, :3].tolist() == pytest.approx([0.6, 0.6, 0.6])


def test_queue_full_and_deadline():
    plan = {0: [(0, r, 0, 0.0, 1) for r in range(4)]}
    net, outs = _run(plan, EdgeConfig(queue_cap=1, service_ms=30.0), 1)
    o = outs[0]
    assert o["edge_done"][0].tolist() == [1, 1, 0, 0] and o["edge_dropped_full"][0].tolist() == [0, 0, 1, 1]
    assert o["edge_dropped"][0].sum() == 2 and o["edge_dropped_deadline"].sum() == 0
    # deadline 50 ms from capture: the third job would start at 0.6 > 0.5 and is dropped at 0.5
    plan = {0: [(0, r, 0, 0.0, 1) for r in range(3)]}
    _, outs = _run(plan, EdgeConfig(service_ms=30.0, deadline_ms=50.0), 1)
    o = outs[0]
    assert o["edge_done"][0].tolist() == [1, 1, 0, 0] and o["edge_dropped_deadline"][0].tolist() == [0, 0, 1, 0]
    # a message that reaches the edge after its deadline is dropped on arrival
    _, outs = _run({0: [(0, 0, 0, 0.6, 1)]}, EdgeConfig(service_ms=30.0, deadline_ms=50.0), 1)
    assert int(outs[0]["edge_dropped_deadline"][0, 0]) == 1 and int(outs[0]["edge_queue_len"][0]) == 0
    # PS aborts at the deadline
    _, outs = _run(plan, EdgeConfig(discipline="ps", service_ms=30.0, deadline_ms=50.0), 1)
    assert outs[0]["edge_dropped_deadline"][0].tolist() == [1, 1, 1, 0]


def test_return_delay_model():
    plan = {0: [(0, 0, 0, 0.1, 1)]}
    cfg = EdgeConfig(service_ms=20.0, return_path="delay", ret_fixed_ms=5.0, cmd_bytes=1000, ret_share=0.5)
    _, outs = _run(plan, cfg, 1)
    snr = 10 ** ((20.0 + cfg.ret_snr_offset_db) / 10)
    rate = cfg.ret_rate_eta * 0.5 * CFG.nprb * 12 * CFG.scs_khz * 1e3 * math.log2(1 + snr)
    ret = (5.0 + 8000 / rate * 1e3) / CFG.control_step_ms
    o = outs[0]
    assert float(o["act_ret_delay"][0, 0]) == pytest.approx(ret, rel=1e-5)
    assert float(o["act_time"][0, 0]) == pytest.approx(0.3 + ret, rel=1e-5)
    assert float(o["act_ul_delay"][0, 0]) == pytest.approx(0.1) and float(o["act_edge_delay"][0, 0]) == pytest.approx(0.2)
    # a return that crosses the step boundary arrives in the next step
    cfg = EdgeConfig(service_ms=20.0, return_path="delay", ret_fixed_ms=90.0)
    _, outs = _run(plan, cfg, 2)
    assert not outs[0]["act_new"].any() and bool(outs[1]["act_new"][0, 0])
    assert int(outs[0]["edge_done"][0, 0]) == 1


def test_newest_action_wins_and_inflight_replacement():
    # robot 0: caps 0 and 1 complete in the same step; only the newer one is sent
    plan = {1: [(0, 0, 0, 1.0, 1), (0, 0, 1, 1.05, 1)]}
    _, outs = _run(plan, EdgeConfig(service_ms=10.0), 2)
    assert int(outs[1]["edge_done"][0, 0]) == 2 and int(outs[1]["act_cap"][0, 0]) == 1
    # a slow return path with one command in flight: the newer command replaces the older one
    plan = {0: [(0, 0, 0, 0.0, 1)], 1: [(0, 0, 1, 1.0, 1)]}
    cfg = EdgeConfig(service_ms=10.0, return_path="delay", ret_fixed_ms=150.0, ret_inflight=1)
    _, outs = _run(plan, cfg, 4)
    assert [int(o["cmd_dropped"][0, 0]) for o in outs] == [0, 1, 0, 0]
    assert [int(o["act_cap"][0, 0]) for o in outs] == [-1, -1, 1, 1]


def test_event_budget_lag_is_exact():
    plan = {0: [(0, r, 0, 0.01 * r, 1) for r in range(8)], 1: [(0, r, 1, 1.0 + 0.01 * r, 1) for r in range(8)]}
    cfg = EdgeConfig(service_ms=4.0, discipline="fifo")
    net_r, ref = _run(plan, cfg, 12, R=8)
    net_l, lag = _run(plan, EdgeConfig(service_ms=4.0, max_events_per_step=3), 12, R=8)
    assert not any(o["edge_lag"].any() for o in ref) and any(o["edge_lag"].any() for o in lag)
    assert not lag[-1]["edge_lag"].any()
    tr, tl = _done_times(ref), _done_times(lag)
    assert len(tr) == 16 and all(tl[k] == tr[k] for k in tl)          # lagged results carry their exact times
    assert all(tl[(r, 1)] == tr[(r, 1)] for r in range(8))
    assert int(net_l.counters()["completed"].sum()) == int(net_r.counters()["completed"].sum()) == 16


@pytest.mark.parametrize("level", ["L0", "L1", "L2-legacy", "L2"])
@pytest.mark.parametrize("disc", ["fifo", "ps"])
def test_conservation_real_levels(level, disc, seeded):
    E, R = 3, 6
    ecfg = EdgeConfig(discipline=disc, service_dist="exponential", service_ms=(15.0, 60.0), queue_cap=3,
                      servers_per_env=2, deadline_ms=250.0, return_path="delay", ret_jitter_ms=20.0)
    cfg = (netslot_compat() if level == "L2" else NRConfig()).with_(edge=ecfg)
    net = make_engine(level, E, R, "cpu", cfg)
    assert isinstance(net, EdgeLoop)
    gen = torch.Generator().manual_seed(3)
    delivered = torch.zeros(E, R, dtype=torch.long)
    tot = {"done": 0, "drop": 0}
    for k in range(30):
        send = (torch.rand(E, R, generator=gen) < 0.7).long() * torch.randint(1, 3, (E, R), generator=gen)
        net.submit(None, Requests(send))
        o = net.step(None, 10 + 15 * torch.rand(E, R, generator=gen))
        delivered += o["delivered"].sum(-1)
        tot["done"] += int(o["edge_done"].sum())
        tot["drop"] += int(o["edge_dropped"].sum())
        c = net.counters()
        assert torch.equal(c["arrived"], delivered)
        assert torch.equal(c["arrived"], c["completed"] + c["dropped_full"] + c["dropped_deadline"] + c["at_edge"])
        assert torch.equal(o["edge_queue_len"], c["at_edge"].sum(-1))
        assert (o["act_cap"] <= o["t"][:, None]).all()
        lat = o["act_latency"][o["act_cap"] >= 0]
        parts = (o["act_ul_delay"] + o["act_edge_delay"] + o["act_ret_delay"])[o["act_cap"] >= 0]
        assert torch.allclose(lat, parts, atol=1e-4) and (lat >= 0).all()
    c = net.counters()
    assert tot["done"] == int(c["completed"].sum()) and tot["drop"] == int((c["dropped_full"] + c["dropped_deadline"]).sum())
    assert tot["done"] > 0


@pytest.mark.parametrize("level", ["L1", "L2-legacy", "L2"])
def test_partial_reset_isolation(level):
    E, R = 4, 5
    ecfg = EdgeConfig(service_dist="exponential", service_ms=40.0, queue_cap=4, return_path="delay",
                      ret_jitter_ms=10.0)
    cfg = (netslot_compat() if level == "L2" else NRConfig()).with_(edge=ecfg)

    def run(reset):
        torch.manual_seed(7)
        net = make_engine(level, E, R, "cpu", cfg, seed=1)
        gen = torch.Generator().manual_seed(5)
        outs = []
        for k in range(14):
            if reset and k == 7:
                net.reset(torch.tensor([1, 3]))
                c = net.counters()
                assert all((v[[1, 3]] == 0).all() for v in c.values())
                assert (net.state["acap"][[1, 3]] == -1).all() and (net.state["scap"][[1, 3]] == -1).all()
                assert (net.state["now"][[1, 3]] == 0).all()
            send = (torch.rand(E, R, generator=gen) < 0.8).long()
            net.submit(None, Requests(send))
            outs.append(net.step(None, 15 + 10 * torch.rand(E, R, generator=gen)))
        return net, outs

    _, a = run(False)
    net, b = run(True)
    keep = [0, 2]
    for oa, ob in zip(a, b):
        for k in EdgeLoop.OUT_KEYS:
            assert torch.equal(oa[k][keep], ob[k][keep]) or (
                oa[k].is_floating_point() and torch.equal(oa[k][keep].nan_to_num(-7), ob[k][keep].nan_to_num(-7))), k
    assert net.clock.tolist() == [14, 7, 14, 7]
    assert (b[7]["t"][[1, 3]] == 0).all() and (b[7]["act_cap"][[1, 3]] <= 0).all()


def test_make_engine_integration_and_passthrough():
    net = make_engine("L0", 2, 3, "cpu", NRConfig(edge=EdgeConfig()), strict=True)
    assert isinstance(net, EdgeLoop) and net.level == "L0" and net.config.edge == EdgeConfig()
    assert not isinstance(make_engine("L0", 2, 3, "cpu"), EdgeLoop)
    net.submit(None, Requests(torch.ones(2, 3, dtype=torch.long)))
    o = net.step(None, torch.full((2, 3), 20.0))
    assert set(EdgeLoop.OUT_KEYS) <= set(o) and (net.clock == 1).all() and net.queued().shape == (2, 3)
    newest, det = net.step(None, torch.full((2, 3), 20.0), torch.zeros(2, dtype=torch.long))   # legacy form
    assert newest.shape == (2, 3)
    with pytest.raises(ValueError):
        make_engine("L0", 2, 3, "cpu", NRConfig(edge=EdgeConfig(return_path="nr_dl")))
    with pytest.raises(AssertionError):
        EdgeConfig(discipline="lifo")


def test_nr_downlink_return_path(seeded):
    E, R = 2, 4
    cfg = netslot_compat(dl=True, edge=EdgeConfig(service_ms=5.0, return_path="nr_dl", cmd_bytes=200))
    net = make_engine("L2", E, R, "cpu", cfg)
    done = {}
    got = 0
    for k in range(25):
        if k == 12:
            net.reset(torch.tensor([1]))
        send = torch.ones(E, R, dtype=torch.long) * (k % 3 == 0)
        net.submit(None, Requests(send))
        o = net.step(None, torch.full((E, R), 18.0))
        for e, r in zip(*torch.nonzero(o["edge_done_cap"] >= 0, as_tuple=True)):
            done[(int(e), int(r), int(o["edge_done_cap"][e, r]))] = float(o["edge_done_time"][e, r])
        for e, r in zip(*torch.nonzero(o["act_new"], as_tuple=True)):
            e, r = int(e), int(r)
            key = (e, r, int(o["act_cap"][e, r]))
            assert key in done
            # the command leaves at the next step boundary and needs at least part of a step on the DL
            assert float(o["act_time"][e, r]) > math.ceil(done[key] - 1e-9) - 1e-6
            assert float(o["act_time"][e, r]) <= float(o["t"][e]) + 1 + 1e-6
            assert float(o["act_ret_delay"][e, r]) > 0
            got += 1
    assert got >= E * R * 4


@pytest.mark.gpu
def test_graph_capture_l2_legacy(cuda, seeded):
    E, R = 64, 8
    eng = make_engine("L2-legacy", E, R, cuda, backend="graph", seed=0)
    ecfg = EdgeConfig(service_ms=(12.0, 45.0), servers_per_env=2, queue_cap=6, deadline_ms=300.0,
                      return_path="delay", ret_fixed_ms=3.0)
    ref, gr = EdgeLoop(eng, ecfg), EdgeLoop(eng, ecfg, graph=True)
    gen = torch.Generator(device=cuda).manual_seed(0)
    for k in range(40):
        if k in (15, 30):
            ids = torch.tensor([1, 5, 9], device=cuda)
            eng.reset(ids)
            ref.reset_edge(ids)
            gr.reset_edge(ids)
        send = (torch.rand(E, R, device=cuda, generator=gen) < 0.6).long() * torch.randint(
            1, 3, (E, R), device=cuda, generator=gen)
        eng.submit(None, Requests(send))
        out = eng.step(None, 5 + 20 * torch.rand(E, R, device=cuda, generator=gen))
        a, b = ref.process(out), gr.process(out)
        for key in EdgeLoop.OUT_KEYS:
            assert torch.equal(a[key].nan_to_num(-7) if a[key].is_floating_point() else a[key],
                               b[key].nan_to_num(-7) if b[key].is_floating_point() else b[key]), (k, key)
    for key, v in ref.state.items():
        assert torch.equal(v.nan_to_num(-7) if v.is_floating_point() else v,
                           gr.state[key].nan_to_num(-7) if v.is_floating_point() else gr.state[key]), key
    assert int(ref.counters()["completed"].sum()) > 0
    # random service and jitter inside the graph: conservation
    ecfg = EdgeConfig(discipline="ps", service_dist="exponential", service_ms=30.0, queue_cap=4,
                      return_path="delay", ret_jitter_ms=10.0)
    gx = EdgeLoop(eng, ecfg, graph=True)
    for k in range(20):
        eng.submit(None, Requests((torch.rand(E, R, device=cuda) < 0.6).long()))
        gx.process(eng.step(None, torch.full((E, R), 15.0, device=cuda)))
        c = gx.counters()
        assert torch.equal(c["arrived"], c["completed"] + c["dropped_full"] + c["dropped_deadline"] + c["at_edge"])
