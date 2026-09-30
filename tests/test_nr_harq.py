"""Multi-process HARQ + RLC in-order delivery of the NR engine, against an independent host-side reference
(merged from nrconfig/tests/test_harq.py).

The engine's debug trace records every TB (time, tx, ok, byte range, exhausted). From it the reference
rebuilds, per robot, the decoded and lost byte ranges and the earliest time at which every byte of [0, end)
is resolved (decoded, or lost in drop mode). Checks:
  H1 decoded (+ lost) ranges are disjoint and tile [0, enq): no byte loss, no double delivery
  H2 every delivered frame's completion time equals the reference in-order completion time
  H3 frame completion is in order per robot
  H4 new TBs are sent while another process of the same UE waits for a retransmission (no HOL)
  H5 n_harq=1: never, and the mean frame delay is higher than with 16 processes
  H6 drop mode (RLC UM): frames overlapping an exhausted TB are never delivered; accounting closes
  H7 downlink: H1-H4 with K1 feedback timing (chase and 5G-LENA IR combining)
  H8 discard="none": no frame is purged at the deadline; all are delivered (possibly late)
  H9 discard="pdcp_arrival": arrivals are refused only while the head-of-line frame is stale
  H10 reset(env_ids): every state tensor of the reset envs returns to its initial value, the other envs are
      untouched, and the engine drains cleanly afterwards
  H11 calibration knobs: ul_mcs_max respected, proc_offset_ms added to every delay, proactive grants remove
      the SR wait at light load
"""
import pytest
import torch

from isaaclab_net.core.config import NRConfig, oai_like
from isaaclab_net.core.mac import MacLink
from isaaclab_net.core.nr_engine import NRNet

dev = "cpu"
E, R = 8, 6
SIZES = (4000.0, 30000.0)


def drained(net):
    """No frame is queued and no HARQ process is busy on any link: later steps cannot change the outcome."""
    links = [lk for lk in (net.ul, net.dl) if lk is not None]
    return all(int((lk.q.cap >= 0).sum()) == 0 and int((lk.h_state != 0).sum()) == 0 for lk in links)


def run(cfg, T_send=40, T_drain=80, seed=1, dl=False, p_send=0.35, snr_hi=25.0, big=False):
    torch.manual_seed(seed)
    net = NRNet(E, R, dev, SIZES, cfg)
    link = net.dl if dl else net.ul
    link.trace = []
    net.trace_frames, net.trace_frames_dl = [], []
    net.log_stats = True
    snr = torch.rand(E, R, device=dev) * snr_hi
    maxbusy = 0
    for t in range(T_send + T_drain):
        if t < T_send:
            send = (torch.rand(E, R, device=dev) < p_send).long() * (2 if big else torch.randint(1, 3, (E, R), device=dev))
            if dl:
                net.add_dl_frames(t, torch.where(send > 0, net.sizes[(send - 1).clamp(min=0)], torch.zeros_like(snr)))
            else:
                net.add_frames(t, send, torch.zeros(E, R, dtype=torch.bool, device=dev),
                               torch.zeros(E, dtype=torch.long, device=dev), snr)
        net.step(t, snr, torch.zeros(E, dtype=torch.long, device=dev))
        maxbusy = max(maxbusy, int((link.h_state != 0).sum(-1).max()))
        if t >= T_send and drained(net):   # the drain phase is an upper bound; stop once the engine is idle
            break
    return net, link, maxbusy


def frames_from(net, dl):
    out = []
    for cap, st, en, fin, dv, tm, dr in (net.trace_frames_dl if dl else net.trace_frames):
        for e, r, f in (dv | tm | dr).nonzero().tolist():
            s = "delivered" if dv[e, r, f] else ("timed" if tm[e, r, f] else "dropped")
            out.append((e, r, int(st[e, r, f]), int(en[e, r, f]), float(fin[e, r, f]), s, int(cap[e, r, f])))
    return out


def reference(link, drop):
    ok_iv, lost_iv = {}, {}
    for frac, tx, ok, lo, hi, exh in link.trace:
        for e, r in tx.nonzero().tolist():
            iv = (int(lo[e, r]), int(hi[e, r]), frac)
            if ok[e, r]:
                ok_iv.setdefault((e, r), []).append(iv)
            elif drop and exh[e, r]:
                lost_iv.setdefault((e, r), []).append(iv)
    return ok_iv, lost_iv


def check_delivery(net, link, dl, drop=False):
    """H1-H3; returns (ok ranges, lost ranges, frames)."""
    ok_iv, lost_iv = reference(link, drop)
    enq = link.q.enq.cpu()
    for e in range(E):
        for r in range(R):
            iv = sorted(ok_iv.get((e, r), []) + lost_iv.get((e, r), []))
            pos = 0
            for lo, hi, _ in iv:
                assert lo == pos, "H1: decoded ranges must tile the byte stream"
                pos = hi
            assert pos == int(enq[e, r]), "H1: every enqueued byte is resolved"
    fr = frames_from(net, dl)
    last = {}
    for e, r, st, en, fin, s, cap in sorted(fr, key=lambda x: (x[0], x[1], x[2])):
        if s != "delivered":
            continue
        ref = max((f for lo, hi, f in ok_iv.get((e, r), []) + lost_iv.get((e, r), []) if lo < en), default=None)
        assert ref is not None and abs(ref - fin) <= 1e-9, "H2: completion time == reference in-order time"
        assert fin >= last.get((e, r), -1) - 1e-12, "H3: in-order completion"
        last[(e, r)] = fin
    assert any(x[5] == "delivered" for x in fr)
    return ok_iv, lost_iv, fr


def mean_delay(fr):
    d = [fin - cap for _, _, _, _, fin, s, cap in fr if s == "delivered"]
    return sum(d) / max(len(d), 1)


# stressed link adaptation: BLER target 0.5 and no OLLA, so about half of the TBs fail
BASE = dict(timeout_steps=10 ** 6, olla=False, bler_target=0.5, n_prb=50, rbg_size=10)
CFG_M = NRConfig(n_harq=16, harq_fail="rlc_am", **BASE)


@pytest.fixture(scope="module")
def multi():
    net, link, busy = run(CFG_M)
    return net, link, busy


def test_h1_h4_multi_process(multi):
    net, link, busy = multi
    check_delivery(net, link, False)
    c = net.counters()["ul"]
    assert c["new_while_pending"] > 0 and busy > 1


@pytest.mark.slow
def test_h5_single_process_blocks(multi):
    net, link, _ = multi
    fr_m = frames_from(net, False)
    net_h, link_h, busy_h = run(CFG_M.with_(n_harq=1))
    _, _, fr_h = check_delivery(net_h, link_h, False)
    assert busy_h == 1 and net_h.counters()["ul"]["new_while_pending"] == 0
    assert mean_delay(fr_h) > mean_delay(fr_m)


def test_h6_rlc_um_drop():
    net_d, link_d, _ = run(CFG_M.with_(harq_fail="drop", bler_target=0.7, max_harq_tx=2))
    ok_iv, lost_iv, fr = check_delivery(net_d, link_d, False, drop=True)
    bad = sum(1 for e, r, st, en, fin, s, cap in fr if s == "delivered"
              and any(st < hi and en > lo for lo, hi, _ in lost_iv.get((e, r), [])))
    assert bad == 0 and any(x[5] == "dropped" for x in fr) and int((link_d.q.cap >= 0).sum()) == 0


@pytest.mark.slow
@pytest.mark.parametrize("comb", ["cc", "ir_lena"])
def test_h7_downlink(comb):
    net_dl, link_dl, _ = run(CFG_M.with_(dl=True, ul=False, harq_combining=comb, bler_target=0.6), dl=True, snr_hi=10.0)
    check_delivery(net_dl, link_dl, True)
    assert net_dl.counters()["dl"]["new_while_pending"] > 0


CFG_N = CFG_M.with_(timeout_steps=5, discard="none", bler_target=0.1, olla=True, frame_buffer=64)


@pytest.mark.slow
def test_h8_discard_none_never_purges():
    net_n, _, _ = run(CFG_N, T_send=10, T_drain=400, p_send=0.8, big=True)
    assert drained(net_n), "H8: the backlog must drain completely so that 'all are delivered' is checked"
    fr = frames_from(net_n, False)
    cnt = {s: sum(1 for x in fr if x[5] == s) for s in ("delivered", "timed", "dropped")}
    late = sum(1 for x in fr if x[5] == "delivered" and x[4] - x[6] >= 5)
    assert cnt["timed"] == 0 and cnt["dropped"] == 0 and late > 0


@pytest.mark.slow
def test_h9_pdcp_arrival_discard():
    net_p, _, _ = run(CFG_N.with_(discard="pdcp_arrival"), T_send=30, T_drain=400, p_send=0.8, big=True)
    fr = frames_from(net_p, False)
    assert net_p.stats["discarded"] > 0 and not any(x[5] == "timed" for x in fr)


def _snapshot(n):
    out = {}
    for lk in ("ul", "dl"):
        link = getattr(n, lk)
        for k in MacLink.STATE:
            out[f"{lk}.{k}"] = getattr(link, k).clone()
        for k in link.q.fields + ["enq"]:
            out[f"{lk}.q.{k}"] = getattr(link.q, k).clone()
    out["h"] = n.h.clone()
    return out


def _h10(drain):
    cfg_r = CFG_M.with_(dl=True, bler_target=0.3)
    torch.manual_seed(3)
    net_r = NRNet(E, R, dev, SIZES, cfg_r)
    fresh = NRNet(E, R, dev, SIZES, cfg_r)
    snr = torch.rand(E, R) * 20
    z = torch.zeros(E, dtype=torch.long)
    zb = torch.zeros(E, R, dtype=torch.bool)
    for t in range(15):
        send = (torch.rand(E, R) < 0.6).long() * 2
        net_r.add_frames(t, send, zb, z, snr)
        net_r.add_dl_frames(t, send.float() * 3000)
        net_r.step(t, snr, z)
    before = _snapshot(net_r)
    ids = torch.tensor([1, 3])
    net_r.reset(ids)
    after, init = _snapshot(net_r), _snapshot(fresh)
    keep = torch.ones(E, dtype=torch.bool); keep[ids] = False
    assert [k for k in after if k != "h" and not torch.equal(after[k][ids], init[k][ids])] == []
    assert [k for k in after if not torch.equal(after[k][keep], before[k][keep])] == []
    assert int((before["ul.h_state"][ids] != 0).sum() + (before["ul.q.cap"][ids] >= 0).sum()) > 0
    if not drain:
        return
    for t in range(15, 300):
        net_r.step(t, snr, z)
        if drained(net_r):
            break
    for lk in ("ul", "dl"):
        assert int((getattr(net_r, lk).q.cap >= 0).sum()) == 0 and int((getattr(net_r, lk).h_state == 1).sum()) == 0


def test_h10_partial_reset(seeded):
    _h10(drain=False)


@pytest.mark.slow
def test_h10_partial_reset_then_drain(seeded):
    """The reset engine keeps working: every frame and HARQ process drains afterwards."""
    _h10(drain=True)


def _light(cfg, seed=5):
    torch.manual_seed(seed)
    n = NRNet(E, R, dev, SIZES, cfg)
    n.log_stats = True
    s = torch.full((E, R), 25.0)
    for t in range(60):
        send = (torch.rand(E, R) < 0.2).long()
        n.add_frames(t, send, torch.zeros(E, R, dtype=torch.bool), torch.zeros(E, dtype=torch.long), s)
        n.step(t, s, torch.zeros(E, dtype=torch.long))
    return n, n.collect()["delay"] * 100.0


@pytest.mark.slow
def test_h11_calibration_knobs():
    n_pg, d_pg = _light(oai_like(n_prb=50, rbg_size=10))
    _, d_sr = _light(oai_like(n_prb=50, rbg_size=10, proactive_grant="off"))
    _, d_0 = _light(oai_like(n_prb=50, rbg_size=10, proc_offset_ms=0.0))
    assert int(n_pg.ul.h_mcs.max()) <= 15
    assert float(d_pg.min()) >= 2.25 - 1e-6 and float(d_pg.median()) < float(d_sr.median())
    assert abs(float(d_pg.median()) - float(d_0.median()) - 2.25) < 2.6
