"""5G-LENA MAC switches of the NR engine under load (NRConfig pf_update, pf_avg_idle, ul_retx_sched, ul_amc_alloc,
ul_grant_model; docs/fidelity-load-gap.md). CPU, reference backend; graph == reference with the switches on is G7 in
tests/test_nr_fast.py.

  L0 every switch set is bitwise the load-gap prototype (frozen copy in tests/nr_frozen/loadfix_proto.py): outputs
     every step and the MAC state and counters at the end, through a partial reset, rng="global" and rng="engine"
  L1 defaults: the presets set no switch, the UL MAC carries no extra state or counters (bitwise equality of the
     defaults with the frozen engine is M1 in test_nr_multicell.py), the compatibility shim with every switch off
     is NRNet, and the v2 presets set exactly the switches
  L2 BSR quantization: the reported value is the 38.321 level upper bound
  L3 grant pipeline, one UE, forced decoding success: SR at the SR opportunity, one-RBG bootstrap grant
     sr_boot_slots later, first data grant bsr_delay_slots after the bootstrap PUSCH, padded over-grant after the
     stale report, and the frame completes only when the RLC residue goes out after the buffer-status timer
  L4 per-RBG PF with frozen averages: backlogged UEs of equal SNR share the RBGs of a slot one each; without the
     switches one UE takes all
  L5 TDMA retransmission: a slot that carries a retransmission carries nothing else, one retx per slot
  L6 previous-allocation AMC: the MCS is chosen for the PRB count of the previous PUSCH
  L7 partial reset of the extra state: reset envs return to their initial values, the others are untouched
  L8 frame and byte conservation with the switches on under random load
  L9 three cells with every switch on (UL + DL, handovers): conservation, one retransmission per cell and slot
  L10 the triton backend refuses the BSR grant pipeline with a clear message; unused_fields reports the lumped SR
      delay under the pipeline and the pipeline fields under the lumped model
"""
import pytest
import torch

from isaaclab_net.core import make_engine, multicell
from isaaclab_net.core.config import LENA_MAC_V2, NRConfig, lena_like, lena_match, lena_match_v2, lena_validation, \
    lena_validation_v2
from isaaclab_net.core.mac_ul import AMC_STATE, BSR_LEVELS, BSR_STATE
from isaaclab_net.core.nr_engine import NRNet
from isaaclab_net.core.nr_loadfix import LoadFixConfig, LoadFixNet, apply_loadfix
from nr_frozen.loadfix_proto import LoadFixNet as ProtoNet
from nr_frozen.loadfix_proto import make_arm as proto_arm

dev = "cpu"
SIZES = (4000.0, 30000.0)

ARMS = {
    "pf": dict(pf_update="rbg", pf_avg_idle="freeze"),
    "pf_intra": dict(pf_update="rbg"),
    "pf_active": dict(pf_avg_idle="freeze"),
    "retx": dict(ul_retx_sched="tdma"),
    "amc": dict(ul_amc_alloc="previous"),
    "pipe": dict(ul_grant_model="bsr", tb_overhead_bytes=8),
    "all": dict(LENA_MAC_V2),
}


def lena_cfg(**kw):
    """The validation config without the locally generated 5G-LENA tables (Sionna curves, same TB size rule)."""
    kw.setdefault("rng", "global")          # the tests below patch torch.rand_like
    return lena_validation(bler_source="pdsch", **kw)


def drive(net, T, E, R, p=0.3, seed=0, reset_at=None, snr_lo=5.0, snr_hi=25.0):
    g = torch.Generator().manual_seed(seed)
    snr = snr_lo + (snr_hi - snr_lo) * torch.rand(E, R, generator=g)
    zb = torch.zeros(E, R, dtype=torch.bool)
    zh = torch.zeros(E, dtype=torch.long)
    outs = []
    for t in range(T):
        send = ((torch.rand(E, R, generator=g) < p).long() * (1 + (torch.rand(E, R, generator=g) < 0.3).long()))
        if t < T - 25:
            net.add_frames(t, send, zb, zh, snr)
        o = net.step(t, snr, zh, full=True)
        outs.append({k: v.clone() for k, v in o.items() if torch.is_tensor(v)})
        if reset_at is not None and t == reset_at:
            net.reset(torch.tensor([0, 2]))
    return outs


def _same(x, y):
    return torch.equal(x.nan_to_num(-7.0), y.nan_to_num(-7.0)) if x.is_floating_point() else torch.equal(x, y)


@pytest.mark.parametrize("arm,rng", [(a, "global") for a in ARMS] + [("all", "engine"), ("pf", "engine")])
def test_l0_switches_bitwise_equal_the_prototype(arm, rng):
    E, R = 4, 6
    base = lena_cfg(rng=rng, seed=11)
    torch.manual_seed(7)
    a = NRNet(E, R, dev, SIZES, base.with_(**ARMS[arm]))
    oa = drive(a, 70, E, R, p=0.45, reset_at=30)
    torch.manual_seed(7)
    b = ProtoNet(E, R, dev, SIZES, base, lf=proto_arm(arm))
    ob = drive(b, 70, E, R, p=0.45, reset_at=30)
    for t, (x, y) in enumerate(zip(oa, ob)):
        for k in x:
            assert _same(x[k], y[k]), (t, k)
    for n in a.ul.STATE:
        if hasattr(b.ul, n):
            assert _same(getattr(a.ul, n), getattr(b.ul, n)), n
    for k, v in b.ul.ctr.items():
        assert float(a.ul.ctr[k]) == float(v), k
    for n in ("ntx_hist", "rv_tx", "rv_fail", "prb_used_env"):
        assert torch.equal(getattr(a.ul, n), getattr(b.ul, n)), n


def test_l1_defaults_and_presets():
    for cfg in (NRConfig(), lena_like(), lena_match(), lena_validation()):
        assert cfg.lena_mac_switches() == {}
    net = NRNet(2, 3, dev, SIZES, NRConfig(rng="global"))
    assert set(net.ul.STATE) == set(type(net.ul).STATE)
    assert not any(hasattr(net.ul, n) for n in list(BSR_STATE) + list(AMC_STATE))
    assert "tb_bytes" not in net.ul.ctr
    shim = LoadFixNet(2, 3, dev, SIZES, NRConfig(rng="global"), lf=LoadFixConfig())
    assert shim.cfg == net.cfg and type(shim.ul) is type(net.ul)
    v2 = lena_match_v2()
    assert v2 == lena_like(**LENA_MAC_V2)
    assert set(v2.lena_mac_switches()) == {"pf_update", "pf_avg_idle", "ul_retx_sched", "ul_amc_alloc",
                                           "ul_grant_model"}
    vv = lena_validation_v2()
    assert vv == lena_validation(sr_grant_delay_slots=None, **LENA_MAC_V2)
    assert apply_loadfix(lena_validation(), proto_arm("all")) == lena_validation(**LENA_MAC_V2)
    net = NRNet(2, 3, dev, SIZES, vv.with_(bler_source="pdsch"))
    assert set(BSR_STATE) | set(AMC_STATE) <= set(net.ul.STATE) and "tb_bytes" in net.ul.ctr


def test_l2_bsr_quantization():
    net = NRNet(1, 1, dev, SIZES, lena_cfg(ul_grant_model="bsr"))
    x = torch.tensor([0, 1, 10, 11, 1480, 1552, 1553, 31100, 149999, 150000, 10 ** 6])
    q = net.ul._quant(x)
    for v, qq in zip(x.tolist(), q.tolist()):
        want = next((lv for lv in BSR_LEVELS if lv >= v), 150000)
        assert qq == want, (v, qq)


class SlotLog:
    """Wraps UlMac.slot: per UL slot (g, RBGs per UE, new-TB data bytes per UE, TB bytes per UE)."""

    def __init__(self, ul):
        self.ul, self.rows, self._orig = ul, [], ul.slot
        ul.slot = self

    def __call__(self, g, frac, nsym, sinr, gain, ack=0, **kw):
        ul = self.ul
        sent0 = ul.sent.clone()
        tbb0 = ul.ctr["tb_bytes"].clone() if "tb_bytes" in ul.ctr else torch.zeros(())
        prb0, rx0 = ul.prb_used_env.clone(), float(ul.ctr["tb_retx"])
        new0 = float(ul.ctr["tb_new"])
        self._orig(g, frac, nsym, sinr, gain, ack, **kw)
        tbb1 = ul.ctr["tb_bytes"] if "tb_bytes" in ul.ctr else torch.zeros(())
        self.rows.append(dict(g=g, rbg=(ul.prb_used_env - prb0) / 10, data=ul.sent - sent0,
                              tb=float(tbb1 - tbb0), retx=float(ul.ctr["tb_retx"]) - rx0,
                              new=float(ul.ctr["tb_new"]) - new0))


def test_l3_grant_pipeline_timeline(monkeypatch):
    monkeypatch.setattr(torch, "rand_like", lambda x: torch.ones_like(x))       # every TB decodes
    cfg = lena_cfg(ul_grant_model="bsr", tb_overhead_bytes=8)
    net = NRNet(1, 1, dev, SIZES, cfg)
    ul = net.ul
    log = SlotLog(ul)
    srs = []
    osr = ul.sr_step

    def sr(g):
        a = int(ul.sr_t)
        osr(g)
        if int(ul.sr_t) != a and int(ul.sr_t) >= 0:
            srs.append(g)
    ul.sr_step = sr
    snr = torch.tensor([[21.23]])
    zb, zh = torch.zeros(1, 1, dtype=torch.bool), torch.zeros(1, dtype=torch.long)
    fin = None
    for t in range(3):
        if t == 0:
            net.add_frames(0, torch.tensor([[2]]), zb, zh, snr)
        o = net.step(t, snr, zh, full=True)
        if bool(o["delivered"].any()):
            fin = float(o["delay"][o["delivered"]][0]) * 100.0
    tx = [r for r in log.rows if r["new"] > 0]
    assert srs[0] == 3                                           # first SR opportunity (S slot)
    assert tx[0]["g"] == 3 + cfg.sr_boot_slots and float(tx[0]["rbg"]) == 1.0   # bootstrap: one RBG
    assert tx[1]["g"] == tx[0]["g"] + cfg.bsr_delay_slots       # the bootstrap BSR arrives 2 UL slots later
    last_data = max(r["g"] for r in tx if float(r["data"]) > cfg.rlc_tail_bytes)
    over = [r for r in tx if r["g"] > last_data and float(r["data"]) == 0]
    assert over, "the stale report must cause an empty (padded) grant after the buffer drains"
    tail = [r for r in tx if float(r["data"]) == cfg.rlc_tail_bytes]
    assert len(tail) == 1 and tail[0]["g"] >= last_data + cfg.rlc_tail_timer_slots
    assert srs[-1] >= last_data + cfg.rlc_tail_timer_slots
    assert fin == pytest.approx((tail[0]["g"] + 1) * 0.5, abs=1e-3)   # completes with the residue's slot


def _backlog_rbgs(**sw):
    torch.manual_seed(0)
    E, R = 1, 5
    net = NRNet(E, R, dev, SIZES, lena_cfg(**sw))
    log = SlotLog(net.ul)
    snr = torch.full((E, R), 15.0)
    zb, zh = torch.zeros(E, R, dtype=torch.bool), torch.zeros(E, dtype=torch.long)
    for t in range(3):
        net.add_frames(t, torch.full((E, R), 2), zb, zh, snr)
        net.step(t, snr, zh)
    return [r for r in log.rows if float(r["rbg"].sum()) == 5 and r["retx"] == 0 and r["new"] > 0][-10:]


def test_l4_per_rbg_pf_spreads_rbgs():
    spread = _backlog_rbgs(pf_update="rbg", pf_avg_idle="freeze")
    assert spread and all(int((r["data"][0] > 0).sum()) == 5 for r in spread)       # one RBG each
    greedy = _backlog_rbgs()
    assert greedy and all(int((r["data"][0] > 0).sum()) == 1 for r in greedy)      # one UE takes the slot


def test_l5_tdma_retx(monkeypatch):
    E, R = 2, 4
    g = torch.Generator().manual_seed(3)
    monkeypatch.setattr(torch, "rand_like", lambda x: torch.rand(x.shape, generator=g) * 0.02)   # many failures
    net = NRNet(E, R, dev, SIZES, lena_cfg(ul_retx_sched="tdma"))
    ul = net.ul
    orig = ul.slot
    per_slot = []

    def chk(gs, frac, nsym, sinr, gain, ack=0, **kw):
        rv0 = ul.rv_tx.clone()
        orig(gs, frac, nsym, sinr, gain, ack, **kw)
        per_slot.append(ul.rv_tx - rv0)                    # [E, ntx]: TBs by transmission number in this slot
    ul.slot = chk
    snr = torch.full((E, R), 6.0)
    zb, zh = torch.zeros(E, R, dtype=torch.bool), torch.zeros(E, dtype=torch.long)
    for t in range(6):
        net.add_frames(t, torch.full((E, R), 1), zb, zh, snr)
        net.step(t, snr, zh)
    had = 0
    for d in per_slot:
        retx, new = d[:, 2:].sum(-1), d[:, 1]
        assert bool((retx <= 1).all())                     # at most one retransmission per slot
        assert bool(((retx == 0) | (new == 0)).all())     # and nothing else in its slot
        had += int((retx > 0).sum())
    assert had > 0 and float(ul.ctr["retx_block"]) > 0


def test_l6_amc_previous_allocation():
    net = NRNet(1, 1, dev, SIZES, lena_cfg(ul_amc_alloc="previous"))
    ul, phy = net.ul, net.ul.phy
    s_bw = 6.644                                              # per-PRB SNR over the band (whole-band power)
    ones = torch.ones(1, 1, 5, dtype=torch.bool)
    pick = lambda nprb: int(phy.select_mcs(torch.full((1, 1, 5), s_bw), ones, torch.zeros(1, 1),
                                           torch.full((1, 1), nprb), 13, 0, 0, "eesm", ul.sb_prb)[0][0, 0])
    first = []
    orig = ul.slot

    def rec(g, frac, nsym, sinr, gain, ack=0, **kw):
        n0 = float(ul.ctr["tb_new"])
        orig(g, frac, nsym, sinr, gain, ack, **kw)
        if float(ul.ctr["tb_new"]) > n0:
            first.append((int(ul.h_mcs[0, 0, 0]), float(ul.prb_used_env[0])))
    ul.slot = rec
    snr = torch.full((1, 1), s_bw + 10 * torch.log10(torch.tensor(5.0)).item())   # snr1 is over 10 of the 50 PRB
    zb, zh = torch.zeros(1, 1, dtype=torch.bool), torch.zeros(1, dtype=torch.long)
    net.add_frames(0, torch.tensor([[2]]), zb, zh, snr)
    ul.last_nprb.fill_(10.0)                                  # previous PUSCH: one RBG
    net.step(0, snr, zh)
    assert len(first) >= 2
    assert first[0][0] == pick(10.0)                          # chosen for the previous 10-PRB PUSCH
    assert first[1][0] == pick(50.0)                          # then for the first TB's 50 PRB


def test_l7_partial_reset_of_extra_state():
    E, R = 4, 5
    torch.manual_seed(1)
    cfg = lena_cfg(**LENA_MAC_V2)
    net = NRNet(E, R, dev, SIZES, cfg)
    drive(net, 32, E, R, p=0.6)
    extra = list(BSR_STATE) + list(AMC_STATE)
    before = {n: getattr(net.ul, n).clone() for n in extra}
    assert any(bool((before[n] != getattr(NRNet(E, R, dev, SIZES, cfg).ul, n)).any()) for n in extra)
    net.reset(torch.tensor([1, 3]))
    fresh = NRNet(E, R, dev, SIZES, cfg).ul
    for n in extra:
        x = getattr(net.ul, n)
        assert torch.equal(x[[1, 3]], getattr(fresh, n)[[1, 3]]), n
        assert torch.equal(x[[0, 2]], before[n][[0, 2]]), n


@pytest.mark.parametrize("arm", ["pf", "retx", "amc", "pipe", "all"])
def test_l8_conservation(arm):
    E, R = 3, 6
    torch.manual_seed(2)
    net = NRNet(E, R, dev, SIZES, lena_cfg(**ARMS[arm]))
    net.log_stats = True
    g = torch.Generator().manual_seed(5)
    snr = 5 + 20 * torch.rand(E, R, generator=g)
    zb, zh = torch.zeros(E, R, dtype=torch.bool), torch.zeros(E, dtype=torch.long)
    acc = 0
    T = 60
    for t in range(T):
        if t < T - 30:
            send = (torch.rand(E, R, generator=g) < 0.5).long()
            acc += int(net.add_frames(t, send, zb, zh, snr).sum())
        net.step(t, snr, zh)
    st = net.collect()
    delivered = len(st["delay"])
    left = int(net.queued().sum())
    assert delivered + st["dropped"] + left == acc
    assert left == 0 and int((net.ul.h_state != 0).sum()) == 0     # drained, residue included


def test_l9_three_cells_with_every_switch():
    E, R = 3, 8
    cfg = multicell(3, dl=True, frame_buffer=32, **LENA_MAC_V2)
    eng = make_engine("L2", E, R, dev, cfg, seed=4)
    ul = eng.net.ul
    orig = ul.slot
    per_slot = []

    def chk(gs, frac, nsym, sinr, gain, ack=0, **kw):
        h0 = ul.h_ntx.clone()
        orig(gs, frac, nsym, sinr, gain, ack, **kw)
        member = ul.member                                     # [E, C, R]
        retx = ((ul.h_ntx > h0) & (h0 > 0)).any(-1)            # robots that sent a retransmission
        per_slot.append((member & retx[:, None, :]).sum(-1))   # [E, C]
    ul.slot = chk
    from isaaclab_net.core.traffic import Requests
    g = torch.Generator().manual_seed(0)
    pos = torch.rand(E, R, 2, generator=g) * 150
    delivered = 0
    for t in range(80):
        if t < 50:
            send = (torch.rand(E, R, generator=g) < 0.4).long() * 2
            eng.submit(None, Requests(send, torch.zeros(E, R, dtype=torch.bool), torch.zeros(E, dtype=torch.long)))
        pos = (pos + 4.0 * (2 * torch.rand(E, R, 2, generator=g) - 1)).clamp(0, 150)
        delivered += int(eng.step(None, pos)["delivered"].sum())
        if t == 40:
            eng.reset(torch.tensor([1]))
    assert delivered > 0 and float(ul.ctr["tb_new"]) > 0
    assert all(bool((c <= 1).all()) for c in per_slot)         # at most one UL retx per cell and slot
    for n in list(BSR_STATE) + list(AMC_STATE):
        assert getattr(ul, n).shape[:2] == (E, R)
    assert int(eng.net.queued().sum()) == 0                     # drained, residue included


def test_l10_triton_refuses_and_unused_fields():
    cfg = lena_match_v2(bler_source="pdsch", tbs_mode="38214", harq_combining="cc")
    with pytest.raises(NotImplementedError, match="grant pipeline"):
        make_engine("L2", 2, 2, dev, cfg, "triton")
    assert "sr_grant_delay_slots" in cfg.with_(sr_grant_delay_slots=40).unused_fields("L2")
    assert "sr_grant_delay_slots" not in lena_validation().unused_fields("L2")
    assert "rlc_tail_bytes" in NRConfig(rlc_tail_bytes=4).unused_fields("L2")
    assert "rlc_tail_bytes" not in cfg.with_(rlc_tail_bytes=4).unused_fields("L2")
