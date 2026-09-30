"""Prototype 5G-LENA uplink-pipeline switches of the NR engine (isaaclab_net/core/nr_loadfix.py).

  L1 every switch off: LoadFixNet is bitwise NRNet (default config with fading and OLLA, and the 5G-LENA
     validation config), through a partial reset
  L2 BSR quantization: the reported value is the 38.321 level upper bound
  L3 grant pipeline, one UE, forced decoding success: SR at the SR opportunity, one-RBG bootstrap grant
     sr_boot_slots later, first data grant bsr_delay_slots after the bootstrap PUSCH, padded over-grant after the
     stale report, and the frame completes only when the RLC residue goes out after the buffer-status timer
  L4 intra-slot PF: backlogged UEs of equal SNR share the RBGs of a slot one each; without it one UE takes all
  L5 TDMA retransmission: a slot that carries a retransmission carries nothing else, one retx per slot
  L6 previous-allocation AMC: the MCS is chosen for the PRB count of the previous PUSCH
  L7 partial reset of the extra state: reset envs return to their initial values, the others are untouched
  L8 frame and byte conservation with every switch on under random load
"""
import pytest
import torch

from isaaclab_net.core.config import NRConfig, lena_validation
from isaaclab_net.core.nr_engine import NRNet
from isaaclab_net.core.nr_loadfix import BSR_LEVELS, LoadFixConfig, LoadFixNet, LoadFixUlMac, make_arm

dev = "cpu"
SIZES = (4000.0, 30000.0)


def lena_cfg(**kw):
    """The validation config without the locally generated 5G-LENA tables (Sionna curves, same TB size rule)."""
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


@pytest.mark.parametrize("cfgf", [NRConfig, lena_cfg])
def test_l1_all_off_is_bitwise_nrnet(cfgf):
    E, R = 4, 6
    torch.manual_seed(7)
    a = NRNet(E, R, dev, SIZES, cfgf())
    oa = drive(a, 60, E, R, reset_at=30)
    torch.manual_seed(7)
    b = LoadFixNet(E, R, dev, SIZES, cfgf(), lf=LoadFixConfig())
    assert isinstance(b.ul, LoadFixUlMac)
    ob = drive(b, 60, E, R, reset_at=30)
    for x, y in zip(oa, ob):
        for k in x:
            assert torch.equal(x[k].nan_to_num(-7.0), y[k].nan_to_num(-7.0)), k
    for k in ("tb_new", "tb_retx", "bytes_ok", "prb_used"):
        assert float(a.ul.ctr[k]) == float(b.ul.ctr[k])


def test_l2_bsr_quantization():
    net = LoadFixNet(1, 1, dev, SIZES, lena_cfg(), lf=LoadFixConfig(grant_pipeline=True))
    x = torch.tensor([0, 1, 10, 11, 1480, 1552, 1553, 31100, 149999, 150000, 10 ** 6])
    q = net.ul._quant(x)
    for v, qq in zip(x.tolist(), q.tolist()):
        want = next((lv for lv in BSR_LEVELS if lv >= v), 150000)
        assert qq == want, (v, qq)


class SlotLog:
    """Wraps LoadFixUlMac.slot: per UL slot (g, RBGs per UE, new-TB data bytes per UE, TB bytes per UE)."""

    def __init__(self, ul):
        self.ul, self.rows, self._orig = ul, [], ul.slot
        ul.slot = self

    def __call__(self, g, frac, nsym, sinr, gain, ack=0):
        ul = self.ul
        sent0, tbb0 = ul.sent.clone(), ul.ctr["tb_bytes"].clone()
        prb0, rx0 = ul.prb_used_env.clone(), float(ul.ctr["tb_retx"])
        new0 = float(ul.ctr["tb_new"])
        self._orig(g, frac, nsym, sinr, gain, ack)
        self.rows.append(dict(g=g, rbg=(ul.prb_used_env - prb0) / 10, data=ul.sent - sent0,
                              tb=float(ul.ctr["tb_bytes"] - tbb0), retx=float(ul.ctr["tb_retx"]) - rx0,
                              new=float(ul.ctr["tb_new"]) - new0))


def test_l3_grant_pipeline_timeline(monkeypatch):
    monkeypatch.setattr(torch, "rand_like", lambda x: torch.ones_like(x))       # every TB decodes
    lf = LoadFixConfig(grant_pipeline=True, tb_overhead_bytes=8)
    net = LoadFixNet(1, 1, dev, SIZES, lena_cfg(), lf=lf)
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
    assert tx[0]["g"] == 3 + 6 and float(tx[0]["rbg"]) == 1.0   # bootstrap: one RBG at the first U slot >= SR + 6
    assert tx[1]["g"] == tx[0]["g"] + 10                         # the bootstrap BSR arrives 2 UL slots later
    last_data = max(r["g"] for r in tx if float(r["data"]) > lf.rlc_tail_bytes)
    over = [r for r in tx if r["g"] > last_data and float(r["data"]) == 0]
    assert over, "the stale report must cause an empty (padded) grant after the buffer drains"
    tail = [r for r in tx if float(r["data"]) == lf.rlc_tail_bytes]
    assert len(tail) == 1 and tail[0]["g"] >= last_data + lf.rlc_tail_timer_slots
    assert srs[-1] >= last_data + lf.rlc_tail_timer_slots
    assert fin == pytest.approx((tail[0]["g"] + 1) * 0.5, abs=1e-3)   # completes with the residue's slot


def _backlog_rbgs(lf):
    torch.manual_seed(0)
    E, R = 1, 5
    net = LoadFixNet(E, R, dev, SIZES, lena_cfg(), lf=lf)
    log = SlotLog(net.ul)
    snr = torch.full((E, R), 15.0)
    zb, zh = torch.zeros(E, R, dtype=torch.bool), torch.zeros(E, dtype=torch.long)
    for t in range(3):
        net.add_frames(t, torch.full((E, R), 2), zb, zh, snr)
        net.step(t, snr, zh)
    return [r for r in log.rows if float(r["rbg"].sum()) == 5 and r["retx"] == 0 and r["new"] > 0][-10:]


def test_l4_intra_slot_pf_spreads_rbgs():
    spread = _backlog_rbgs(LoadFixConfig(pf_intra_slot=True, pf_active_only=True))
    assert spread and all(int((r["data"][0] > 0).sum()) == 5 for r in spread)       # one RBG each
    greedy = _backlog_rbgs(LoadFixConfig(tb_overhead_bytes=6))        # a switch on, PF unchanged
    assert greedy and all(int((r["data"][0] > 0).sum()) == 1 for r in greedy)      # one UE takes the slot


def test_l5_tdma_retx(monkeypatch):
    E, R = 2, 4
    g = torch.Generator().manual_seed(3)
    monkeypatch.setattr(torch, "rand_like", lambda x: torch.rand(x.shape, generator=g) * 0.02)   # many failures
    net = LoadFixNet(E, R, dev, SIZES, lena_cfg(), lf=LoadFixConfig(retx_tdma=True))
    ul = net.ul
    orig = ul.slot
    per_slot = []

    def chk(gs, frac, nsym, sinr, gain, ack=0):
        rv0 = ul.rv_tx.clone()
        orig(gs, frac, nsym, sinr, gain, ack)
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
    assert had > 0


def test_l6_amc_previous_allocation():
    cfg = lena_cfg()
    net = LoadFixNet(1, 1, dev, SIZES, cfg, lf=LoadFixConfig(amc_prev_alloc=True))
    ul, phy = net.ul, net.ul.phy
    s_bw = 6.644                                              # per-PRB SNR over the band (whole-band power)
    ones = torch.ones(1, 1, 5, dtype=torch.bool)
    pick = lambda nprb: int(phy.select_mcs(torch.full((1, 1, 5), s_bw), ones, torch.zeros(1, 1),
                                           torch.full((1, 1), nprb), 13, 0, 0, "eesm", ul.sb_prb)[0][0, 0])
    first = []
    orig = ul.slot

    def rec(g, frac, nsym, sinr, gain, ack=0):
        n0 = float(ul.ctr["tb_new"])
        orig(g, frac, nsym, sinr, gain, ack)
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
    net = LoadFixNet(E, R, dev, SIZES, lena_cfg(), lf=make_arm("all"))
    drive(net, 20, E, R, p=0.6)
    before = {n: getattr(net.ul, n).clone() for n in LoadFixUlMac.X_INIT}
    net.reset(torch.tensor([1, 3]))
    fresh = LoadFixNet(E, R, dev, SIZES, lena_cfg(), lf=make_arm("all")).ul
    for n in LoadFixUlMac.X_INIT:
        x = getattr(net.ul, n)
        assert torch.equal(x[[1, 3]], getattr(fresh, n)[[1, 3]]), n
        assert torch.equal(x[[0, 2]], before[n][[0, 2]]), n


@pytest.mark.parametrize("arm", ["pf", "retx", "amc", "pipe", "all"])
def test_l8_conservation(arm):
    E, R = 3, 6
    torch.manual_seed(2)
    net = LoadFixNet(E, R, dev, SIZES, lena_cfg(), lf=make_arm(arm))
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
