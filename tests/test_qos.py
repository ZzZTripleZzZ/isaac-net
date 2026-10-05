"""QoS-aware scheduling of the NR engine (NRConfig.scheduler="qos", 5G-LENA NrMacSchedulerOfdmaQos), CPU.

  Q1 off = before: scheduler="pf" with the qos_* fields set (unread) is bitwise the frozen engine of tests/nr_frozen/
     and the live engine with the qos defaults; a "pf" link never enters the QoS path
  Q2 two classes on a saturated cell: class-0 messages see a lower delay distribution (median, 90th percentile) than
     class-1 messages, the gap grows with the priority-level difference (10 / 70 against 40 / 60), and "pf" serves
     both classes alike
  Q3 delay budget: a class-1 frame (P = 70, PDB 30 ms) beats a fresh class-0 frame (P = 10) exactly when
     (100 - 70) * PDB / (PDB - HOL) > 100 - 10, i.e. HOL > 20 ms (and always once HOL >= PDB); without a PDB the
     class-0 frame always wins; the UL and DL weights follow 5G-LENA's formulas
  Q4 FrameQueue.reorder: class order inside the unsent visible block, sent / gated frames untouched, holes skipped
  Q5 conservation and in-order delivery per class: every accepted message is delivered once, bytes decoded equal
     bytes accepted, each robot's class keeps its arrival order, the stream stays contiguous, and a class-0 message
     enqueued behind unsent class-1 messages completes before them
  Q6 E-independence and partial-reset isolation (test_radio_rng.py patterns), one cell and three cells (per-RBG PF,
     gamma != 1), poses through the engine radio, random priorities
  Q7 unused_fields / fields_read_by: the qos_* fields are read only with scheduler="qos"
GPU equivalence (graph bitwise, triton teacher forced): configs "qos" / "qos_rbg" in tests/nr_equiv.py, run by
tests/test_nr_fast.py G1 / G2.
"""
import math

import pytest
import torch

from isaac_net.core import NRConfig, Requests, make_engine, multicell
from isaac_net.core.config import QOS_FIELDS, fields_read_by
from isaac_net.core.nr_engine import NRNet
from isaac_net.core.queues import FrameQueue
from nr_frozen.nr_engine import NRNet as FrozenNRNet

SIZES = (4000.0, 30000.0)
INF = math.inf
OFF_FIELDS = dict(qos_classes=3, qos_priority=(5, 20, 80), qos_pdb_ms=(10.0, 50.0, INF), qos_gamma=0.7)


# ---------------------------------------------------------------- Q1 off = before
def _run_single(cls, cfg, E=3, R=4, steps=8, reset_at=4):
    torch.manual_seed(11)
    net = cls(E, R, "cpu", SIZES, cfg, generator=torch.Generator().manual_seed(3))
    g = torch.Generator().manual_seed(5)
    outs = []
    for t in range(steps):
        if t == reset_at:
            net.reset(torch.tensor([1]))
        send = torch.randint(0, 3, (E, R), generator=g)
        snr = -5 + 30 * torch.rand(E, R, generator=g)
        hid = torch.randint(0, 3, (E,), generator=g)
        net.add_frames(t, send, torch.rand(E, R, generator=g) < 0.3, hid, snr)
        if cfg.dl:
            net.add_dl_frames(t, torch.where(send > 0, net.sizes[(send - 1).clamp(min=0)], torch.zeros_like(snr)))
        o = net.step(t, snr, hid, full=True) if t % 2 == 0 else net.step_rx(t, snr - 113.0, hid, full=True)
        outs.append({k: v.clone() for k, v in o.items()})
        for lk in (net.ul, net.dl):
            if lk is not None:
                outs[-1].update({f"{lk.dir}.{n}": getattr(lk, n).clone() for n in ("olla", "avg", "csi", "sent")})
    return outs


def _equal(a, b):
    for t, (x, y) in enumerate(zip(a, b)):
        assert x.keys() == y.keys()
        for k in x:
            assert torch.equal(x[k].nan_to_num(-7.0), y[k].nan_to_num(-7.0)), (t, k)


def test_q1_pf_equals_frozen_engine_and_defaults():
    cfg = NRConfig(dl=True, rng="global", control_step_ms=20.0, **OFF_FIELDS)    # the frozen engine: global RNG
    _equal(_run_single(FrozenNRNet, cfg), _run_single(NRNet, cfg))
    live = NRConfig(dl=True, control_step_ms=20.0)
    _equal(_run_single(NRNet, live), _run_single(NRNet, live.with_(**OFF_FIELDS)))
    _equal(_run_single(NRNet, live.with_(pf_update="rbg")), _run_single(NRNet, live.with_(pf_update="rbg",
                                                                                           **OFF_FIELDS)))
    net = NRNet(2, 2, "cpu", SIZES, live)
    assert not net.ul.qos and not net.dl.qos and "prio" not in net.ul.q.extras


# ---------------------------------------------------------------- Q2 delay by class under saturation
def _class_delays(sched, prio=(10, 70), size=2000.0, steps=70, warm=20):
    """Four cells of eight robots, robots 0, 2, ... send class 0 and 1, 3, ... class 1, one message of size bytes
    per 10 ms control step each (more than the UL carries); delays (control steps) of the messages delivered after
    the warm-up, per class."""
    cfg = NRConfig(scheduler=sched, msg_sizes=(size,), control_step_ms=10.0, frame_buffer=32, qos_priority=prio)
    E, R = 4, 8
    eng = make_engine("L2", E, R, "cpu", cfg, seed=1)
    pr = (torch.arange(R) % 2).expand(E, R)
    g = torch.Generator().manual_seed(0)
    d = {0: [], 1: []}
    for t in range(steps):
        eng.submit(None, Requests(torch.ones(E, R, dtype=torch.long)), priority=pr)
        o = eng.step(None, 10 + 10 * torch.rand(E, R, generator=g))
        if t >= warm:
            for c in (0, 1):
                d[c].append(o["delay"][o["delivered"] & (o["priority"] == c)])
    return [torch.cat(d[c]) for c in (0, 1)]


def test_q2_class0_lower_delay_cdf_and_gap_grows_with_priority_gap():
    pf0, pf1 = _class_delays("pf")
    a0, a1 = _class_delays("qos", (10, 70))
    b0, b1 = _class_delays("qos", (40, 60))
    assert min(map(len, (pf0, pf1, a0, a1, b0, b1))) > 100
    q = torch.tensor([0.5, 0.9])
    for x0, x1 in ((a0, a1), (b0, b1)):
        assert (torch.quantile(x0, q) < torch.quantile(x1, q)).all()
    gap = lambda x0, x1: float(x1.mean() - x0.mean())     # noqa: E731
    assert gap(a0, a1) > gap(b0, b1) > 1.0                 # control steps of 10 ms
    assert abs(gap(pf0, pf1)) < 0.25 * gap(b0, b1)
    assert len(a1) < len(b1) < len(pf1)                    # class 1 delivers less as its relative weight shrinks


# ---------------------------------------------------------------- Q3 delay-budget factor
def _two_robot_slot(age_ms, pdb=30.0, direction="ul"):
    """Robot 0 holds a class-1 frame captured age_ms ago, robot 1 a fresh class-0 frame, equal channel and PF
    average, both backlogged beyond one slot; returns (weights qw [2], bytes granted [2]) of one data slot."""
    cfg = NRConfig(scheduler="qos", qos_priority=(10, 70), qos_pdb_ms=(INF, pdb), fading=False, control_step_ms=1.0,
                   dl=direction == "dl", olla=False)
    net = NRNet(1, 2, "cpu", (1e5,), cfg, seed=0)
    link = net.ul if direction == "ul" else net.dl
    q, k = link.q, int(age_ms / cfg.control_step_ms)
    q.add(0, torch.tensor([[True, False]]), torch.full((1, 2), 1e5))
    q.add(k, torch.tensor([[False, True]]), torch.full((1, 2), 1e5))
    q.prio[0, 0, 0] = 1
    link.qos_prepare(float(k))
    if direction == "ul":
        link.bsr = link.unsent()
    S = cfg.n_subbands
    g = next(s for s in range(len(cfg.tdd_pattern)) if cfg.slot_symbols(s)[0 if direction == "dl" else 1] >= 12)
    ns = cfg.slot_symbols(g)[0 if direction == "dl" else 1]
    sinr = torch.full((1, 2, S), 12.0)
    link.slot(g, k + 0.5, ns, sinr, torch.zeros(1, 2, S), ack_slot=g + 4)
    return link.qw[0].clone(), link.sent[0].clone()


@pytest.mark.parametrize("direction", ["ul", "dl"])
def test_q3_delay_budget_factor(direction):
    for age, old_first in ((0.0, False), (10.0, False), (19.0, False), (25.0, True), (30.0, True), (200.0, True)):
        qw, sent = _two_robot_slot(age, direction=direction)
        d = 30.0 / (0.1 if age >= 30.0 else 30.0 - age)
        assert torch.allclose(qw, torch.tensor([30.0 * d, 90.0]), rtol=1e-5), (age, qw)
        assert (sent[0] > 0, sent[1] > 0) == ((True, False) if old_first else (False, True)), (age, sent)
    qw, sent = _two_robot_slot(500.0, pdb=INF, direction=direction)     # no budget: priority level only
    assert qw.tolist() == [30.0, 90.0] and sent[0] == 0 and sent[1] > 0


def test_q3_ul_takes_the_best_class_dl_sums_the_classes():
    cfg = NRConfig(scheduler="qos", qos_priority=(10, 70), qos_pdb_ms=(INF, 30.0), dl=True, control_step_ms=10.0)
    net = NRNet(1, 3, "cpu", (1e4,), cfg, seed=0)
    for lk in (net.ul, net.dl):
        both = torch.tensor([[True, True, False]])
        lk.q.add(0, both, torch.full((1, 3), 1e4))                       # robots 0, 1: an old class-1 frame
        lk.q.prio[0, :2, 0] = 1
        lk.q.add(2, torch.tensor([[True, False, False]]), torch.full((1, 3), 1e4))   # robot 0: a class-0 frame
        lk.qos_prepare(2.0)                                              # HOL of class 1: 20 ms, D = 3
        # robot 0's class-0 frame was moved ahead of its unsent class-1 frame (byte assignment by class)
        assert lk.q.prio[0, 0, :2].tolist() == [0, 1] and lk.q.start[0, 0, 0] == 0
    assert net.ul.qw[0].tolist() == [90.0, 90.0, 30.0]                   # UL: 100 - min P; nothing queued: 100 - 70
    assert torch.allclose(net.dl.qw[0], torch.tensor([90.0 + 90.0, 90.0, 30.0]))   # DL: sum over classes


# ---------------------------------------------------------------- Q4 FrameQueue.reorder
def test_q4_reorder_block():
    q = FrameQueue(1, 1, 8, "cpu")
    q.enable_extras()
    for i, (nb, pr) in enumerate([(10, 1), (10, 1), (5, 1), (7, 0), (3, 1), (4, 0), (6, 0)]):
        q.add(i, torch.ones(1, 1, dtype=torch.bool), torch.full((1, 1), float(nb)))
        q.prio[0, 0, i] = pr
    # stream: 0:[0,10) 1:[10,20) 2:[20,25) 3:[25,32) 4:[32,35) 5:[35,39) 6:[39,45); the last frame is behind the gate
    q.enq = torch.tensor([[39]])
    sent = torch.tensor([[15]])                                          # frame 1 partly sent
    new = q.reorder(sent, q.prio, 2)
    assert q.cap[0, 0].tolist() == [0, 1, 3, 5, 2, 4, 6, -1]
    assert q.start[0, 0, :7].tolist() == [0, 10, 20, 27, 31, 36, 39]
    assert q.end[0, 0, :7].tolist() == [10, 20, 27, 31, 36, 39, 45]
    assert new.tolist() == [[15]] and q.enq.tolist() == [[39]]
    # a purged frame leaves a hole in front of the block: skipped when nothing below it is unsent
    q.remove(torch.tensor([[[False, False, False, False, True, False, False, False]]]))   # frame 2 (5 bytes)
    new = q.reorder(torch.tensor([[20]]), q.prio, 2)
    assert q.cap[0, 0, :6].tolist() == [0, 1, 3, 5, 4, 6]
    assert q.start[0, 0, :6].tolist() == [0, 10, 25, 32, 36, 39] and new.tolist() == [[25]]


# ---------------------------------------------------------------- Q5 conservation and in-order per class
def test_q5_conservation_and_in_order_per_class():
    cfg = NRConfig(scheduler="qos", qos_classes=3, qos_priority=(10, 40, 70), qos_pdb_ms=(20.0, INF, 60.0), dl=True,
                   discard="none", frame_buffer=64, control_step_ms=10.0, msg_sizes=(1000.0, 4000.0))
    E, R = 3, 4
    eng = make_engine("L2", E, R, "cpu", cfg, seed=2)
    g = torch.Generator().manual_seed(3)
    seq = torch.zeros(E, R, dtype=torch.long)
    acc_tags, acc_bytes, got = set(), 0, {}
    for t in range(70):
        send = torch.randint(0, 3, (E, R), generator=g) if t < 40 else torch.zeros(E, R, dtype=torch.long)
        pr = torch.randint(0, 3, (E, R), generator=g)
        a = eng.submit(None, Requests(send), tag=seq + 1, priority=pr)
        for e, r in a.nonzero().tolist():
            acc_tags.add((e, r, int(seq[e, r]) + 1, int(pr[e, r])))
            acc_bytes += int(eng.net.air_bytes(torch.tensor(cfg.msg_sizes[int(send[e, r]) - 1])))
        seq = seq + a.long()
        o = eng.step(None, 5 + 15 * torch.rand(E, R, generator=g))
        done = o["arrival"] + o["delay"].double()
        for e, r, f in o["delivered"].nonzero().tolist():
            key = (e, r, int(o["tag"][e, r, f]), int(o["priority"][e, r, f]))
            assert key not in got
            got[key] = float(done[e, r, f])
        q, ul = eng.net.ul.q, eng.net.ul
        v = q.cap >= 0                                    # stream contiguous and in slot order, ending at enq
        last = torch.where(v, q.end, torch.zeros_like(q.end)).max(-1).values
        assert (torch.where(v[..., 1:], q.start[..., 1:] - q.end[..., :-1], torch.zeros_like(q.end[..., 1:])) == 0).all()
        assert torch.equal(torch.where(v.any(-1), last, q.enq), q.enq) and (ul.sent <= q.enq).all()
    assert set(got) == acc_tags
    assert int(eng.net.ul.ctr["bytes_ok"]) == acc_bytes and (eng.net.ul.q.count() == 0).all()
    for (e, r, c) in {(k[0], k[1], k[3]) for k in got}:
        times = [got[k] for k in sorted(k for k in got if (k[0], k[1], k[3]) == (e, r, c))]
        assert times == sorted(times), (e, r, c)


def test_q5_high_class_overtakes_unsent_low_class():
    cfg = NRConfig(scheduler="qos", msg_sizes=(30000.0, 2000.0), control_step_ms=10.0, frame_buffer=16, fading=False,
                   discard="none")
    eng = make_engine("L2", 1, 1, "cpu", cfg, seed=0)
    snr = torch.full((1, 1), 20.0)
    for _ in range(3):                                    # three large class-1 messages
        eng.submit(None, Requests(torch.ones(1, 1, dtype=torch.long)), tag=1, priority=1)
    eng.step(None, snr)
    eng.submit(None, Requests(torch.full((1, 1), 2)), tag=2, priority=0)
    done = {}
    for _ in range(40):
        o = eng.step(None, snr)
        for f in o["delivered"][0, 0].nonzero().flatten().tolist():
            done.setdefault(int(o["tag"][0, 0, f]), []).append(float(o["arrival"][0, 0, f] + o["delay"][0, 0, f]))
    assert len(done[1]) == 3 and len(done[2]) == 1
    assert done[2][0] < sorted(done[1])[1]               # ahead of every class-1 message not started yet


# ---------------------------------------------------------------- Q6 E-independence and partial reset
QOS_ONE = NRConfig(scheduler="qos", qos_pdb_ms=(20.0, INF), dl=True, frame_buffer=32, control_step_ms=20.0)
QOS_MC = multicell(3, scheduler="qos", qos_pdb_ms=(30.0, INF), dl=True, control_step_ms=20.0, pf_update="rbg",
                   qos_gamma=0.8)


def _drive(cfg, E, resets=None, steps=6, emax=6, seed=4):
    eng = make_engine("L2", E, 4, "cpu", cfg, seed=seed)
    g = torch.Generator().manual_seed(1)
    pos = torch.rand(emax, 4, 2, generator=g) * 150
    outs = []
    for k in range(steps):
        send = torch.randint(0, 3, (emax, 4), generator=g)
        pr = torch.randint(0, 2, (emax, 4), generator=g)
        eng.submit(None, Requests(send[:E], None, torch.full((E,), k)), priority=pr[:E])
        pos = (pos + 3 * (2 * torch.rand(emax, 4, 2, generator=g) - 1)).clamp(0, 150)
        o = eng.step(None, pos[:E])
        outs.append({n: v.clone() for n, v in o.items() if torch.is_tensor(v) and v.dim() >= 1 and v.shape[0] == E})
        outs[-1].update({f"{lk.dir}.qw": lk.qw.clone() for lk in (eng.net.ul, eng.net.dl)})
        if resets and k in resets:
            eng.reset(torch.tensor(resets[k]))
    return outs


def _rows(a, b, rows_a, rows_b=None):
    rows_b = rows_a if rows_b is None else rows_b
    for x, y in zip(a, b):
        for k in x:
            assert torch.equal(x[k][rows_a].nan_to_num(-7.0), y[k][rows_b].nan_to_num(-7.0)), k


@pytest.mark.parametrize("cfg", [QOS_ONE, QOS_MC], ids=["one_cell", "three_cells"])
def test_q6_env_independence_and_partial_reset(cfg):
    a = _drive(cfg, E=3, resets={2: [1]})
    b = _drive(cfg, E=4, resets={2: [1]})
    c = _drive(cfg, E=3, resets={1: [2], 2: [1]})
    d = _drive(cfg, E=3)
    _rows(a, b, slice(0, 3))                      # env rows do not depend on E
    _rows(a, c, 0)                                # other envs' resets leave env 0 alone
    _rows(a[:3], d[:3], slice(None))
    _rows(a, d, [0, 2])                           # a partial reset leaves the other envs bitwise unaffected
    assert any(bool((x["ul.qw"] != 30.0).any()) for x in a)        # the class weights moved


# ---------------------------------------------------------------- Q7 field gating
def test_q7_unused_fields():
    off = NRConfig(**OFF_FIELDS)
    assert set(QOS_FIELDS) <= set(off.unused_fields("L2"))
    on = off.with_(scheduler="qos")
    assert not set(QOS_FIELDS) & set(on.unused_fields("L2"))
    assert set(QOS_FIELDS) <= fields_read_by("L2") and not set(QOS_FIELDS) & fields_read_by("L2", NRConfig())
    assert set(QOS_FIELDS) <= set(off.unused_fields("L0"))
    assert NRConfig().unused_fields("L2") == [] and NRConfig(scheduler="qos").unused_fields("L2") == []
    with pytest.raises(AssertionError):
        NRConfig(scheduler="qos", qos_classes=3)                       # one priority and PDB per class
    with pytest.raises(AssertionError):
        NRConfig(qos_priority=(10, 100))
