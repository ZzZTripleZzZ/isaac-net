"""Regression tests for the MAC fixes of code review 2026-10-04 (section B, items 8-12), reference engine on CPU.

  F8  HARQ processes whose bytes were all purged are freed at compaction (no endless retransmission of dead bytes);
      a partially purged process and an empty (padding) TB are kept
  F9  an idle robot whose PF average decayed to almost zero cannot take RBGs from an admitted retransmission
      (lexicographic retransmission priority), and the PF average is floored (no float32 underflow / infinite metric)
  F10 retx_priority=False: RBGs won by a retransmission that falls short of its RBG count go to new data
  F11 the wideband PF rate estimate respects the MCS cap (ul_mcs_max)
  F12 SlotTap.ul_tx_j charges the PUSCH symbols of the slot, not 14
"""
import math

import pytest
import torch

from isaac_net.core.config import NRConfig
from isaac_net.core.engine import make_engine
from isaac_net.core.mac import AVG_MIN
from isaac_net.core.nr_engine import NRNet
from isaac_net.core.slot_tap import SlotTap
from isaac_net.core.traffic import Requests

dev = "cpu"


def _ul_slot(net, g, sinr_db):
    """Run one U data slot of the UL link directly (flat per-subband SINR, no fading)."""
    E, R, S = net.E, net.R, net.S
    nsym = net.cfg.ul_data_symbols
    ref = torch.full((E, R, S), float(sinr_db))
    net.ul.slot(g, 0.0, nsym, ref, torch.zeros(E, R, S), 0, gh=4)


# ---------------------------------------------------------------- F8
def test_purged_harq_processes_are_freed():
    """The review's scenario: one frame at an SNR where nothing decodes; after the frame times out (purge) the
    processes that carried it are free and tb_retx stops growing."""
    torch.manual_seed(0)
    E, R = 1, 1
    cfg = NRConfig()
    assert cfg.discard == "purge" and cfg.harq_fail == "rlc_am"
    net = NRNet(E, R, dev, (4000.0,), cfg)
    z, zh = torch.zeros(E, R, dtype=torch.bool), torch.zeros(E, dtype=torch.long)
    snr = torch.full((E, R), -30.0)
    net.add_frames(0, torch.ones(E, R, dtype=torch.long), z, zh, snr)
    u = net.ul
    busy_before = 0
    for t in range(cfg.timeout_steps + 1):
        busy_before = max(busy_before, int((u.h_state == 1).sum()))
        net.step(t, snr, zh)
    assert busy_before > 0                                   # the frame was in flight on HARQ processes
    assert int(net.queued().sum()) == 0                      # and has been purged
    assert int((u.h_state != 0).sum()) == 0                  # every process that carried it is free again
    assert int(u.h_ntx.sum()) == 0 and float(u.h_comb.sum()) == 0.0
    retx = float(u.ctr["tb_retx"])
    exh = float(u.ctr["exhaust"])
    for t in range(cfg.timeout_steps + 1, cfg.timeout_steps + 21):
        net.step(t, snr, zh)
    assert float(u.ctr["tb_retx"]) == retx                   # no phantom retransmissions
    assert float(u.ctr["exhaust"]) == exh


def test_compact_boundary_cases():
    """Fully purged process freed; partially purged process (its upper bytes belong to the still-queued head frame)
    kept; an empty TB (no stream bytes) kept; a process above the floor untouched."""
    E, R = 1, 1
    net = NRNet(E, R, dev, (4000.0,), NRConfig())
    z, zh = torch.zeros(E, R, dtype=torch.bool), torch.zeros(E, dtype=torch.long)
    one = torch.ones(E, R, dtype=torch.long)
    for t in range(3):
        net.add_frames(t, one, z, zh, torch.zeros(E, R))
    u = net.ul
    q = u.q
    s1, s2 = int(q.start[0, 0, 1]), int(q.start[0, 0, 2])    # frame 0 = [0, s1), frame 1 = [s1, s2)
    ranges = [(0, s1 // 2), (s1 // 2, s1 + 10), (s1 // 2, s1 // 2), (s1 + 10, s2)]
    for p, (lo, hi) in enumerate(ranges):
        u.h_state[0, 0, p], u.h_lo[0, 0, p], u.h_hi[0, 0, p] = 1, lo, hi
        u.h_ready[0, 0, p], u.h_ntx[0, 0, p], u.h_comb[0, 0, p] = 7, 2, 3.0
    u.sent[:] = s2
    gone = torch.zeros_like(q.cap, dtype=torch.bool)
    gone[0, 0, 0] = True                                      # frame 0 timed out
    u.compact(gone)
    assert int(u.floor[0, 0]) == s1
    st = u.h_state[0, 0, :4].tolist()
    assert st == [0, 1, 1, 1]
    assert (int(u.h_ready[0, 0, 0]), int(u.h_ntx[0, 0, 0]), float(u.h_comb[0, 0, 0])) == (0, 0, 0.0)
    assert math.isinf(float(u.h_lexp[0, 0, 0])) and float(u.h_lexp[0, 0, 0]) < 0
    assert int(u.h_ntx[0, 0, 1]) == 2 and int(u.h_ready[0, 0, 1]) == 7     # partially purged: kept as it was
    assert int(u.ack_ptr()[0, 0]) == s1                      # the head frame's bytes are not acknowledged


# ---------------------------------------------------------------- F9
def _retx_vs_idle(retx_priority=True):
    """Robot 0: an admitted retransmission that needs every RBG. Robot 1: new data, PF average decayed to ~0."""
    E, R = 1, 2
    net = NRNet(E, R, dev, (4000.0,), NRConfig(retx_priority=retx_priority))
    u, S = net.ul, net.S
    z, zh = torch.zeros(E, R, dtype=torch.bool), torch.zeros(E, dtype=torch.long)
    net.add_frames(0, torch.ones(E, R, dtype=torch.long), z, zh, torch.zeros(E, R))
    u.h_state[0, 0, 0], u.h_lo[0, 0, 0], u.h_hi[0, 0, 0] = 1, 0, 100
    u.h_ready[0, 0, 0], u.h_ntx[0, 0, 0], u.h_mcs[0, 0, 0] = 0, 1, 0
    u.h_tbs[0, 0, 0], u.h_nsb[0, 0, 0] = 2000, S
    u.sent[0, 0] = 100
    u.bsr[:] = 10 ** 6                                        # robot 1 is granted (lumped BSR)
    u.avg[0, 1] = 1e-30                                       # a long-idle robot under pf_avg_idle="decay"
    return net


def test_idle_robot_cannot_block_admitted_retx():
    net = _retx_vs_idle()
    u = net.ul
    _ul_slot(net, 10, 20.0)
    assert float(u.ctr["tb_retx"]) == 1.0                    # the retransmission got all its RBGs and was sent
    assert float(u.ctr["tb_new"]) == 0.0                     # no RBG left for new data
    assert float(u.ctr["prb_used"]) == float(u.sb_prb.sum())


def test_pf_average_floor():
    net = NRNet(1, 2, dev, (4000.0,), NRConfig())
    u = net.ul
    zero = torch.zeros(1, 2, dtype=torch.long)
    nope = torch.zeros(1, 2, dtype=torch.bool)
    for g in range(5000):                                     # idle: the average decays every data slot
        u._pf_update(g, zero, nope, None, zero, nope, zero, nope)
    assert float(u.avg.min()) == pytest.approx(AVG_MIN)
    rate = torch.tensor(1e4)
    assert torch.isfinite(rate / u.avg).all()


def test_retx_priority_unchanged_without_tiny_average():
    """With ordinary averages the lexicographic rule picks what +1e9 picked: same outputs as a fresh run of the
    same scenario with a robot-1 average of 100 (the retransmission wins every RBG it needs)."""
    net = _retx_vs_idle()
    net.ul.avg[0, 1] = 100.0
    _ul_slot(net, 10, 20.0)
    assert float(net.ul.ctr["tb_retx"]) == 1.0 and float(net.ul.ctr["tb_new"]) == 0.0


# ---------------------------------------------------------------- F10
def test_no_retx_priority_releases_short_rbgs():
    """retx_priority=False on a frequency-selective channel: robot 0's retransmission (11 RBGs) wins the RBGs where
    it is strong, robot 1 (a large frame, strong on the other RBGs) the rest, so the retransmission falls short and
    is not sent. Its RBGs go to robot 1, which then fills the carrier, instead of staying empty."""
    E, R = 1, 2
    net = NRNet(E, R, dev, (4000.0, 30000.0), NRConfig(retx_priority=False, olla=False))
    u, S = net.ul, net.S
    z, zh = torch.zeros(E, R, dtype=torch.bool), torch.zeros(E, dtype=torch.long)
    net.add_frames(0, torch.tensor([[1, 2]]), z, zh, torch.zeros(E, R))
    u.h_state[0, 0, 0], u.h_lo[0, 0, 0], u.h_hi[0, 0, 0] = 1, 0, 100
    u.h_ready[0, 0, 0], u.h_ntx[0, 0, 0], u.h_mcs[0, 0, 0] = 0, 1, 0
    u.h_tbs[0, 0, 0], u.h_nsb[0, 0, 0] = 2000, S - 2
    u.sent[0, 0] = 100
    u.bsr[:] = 10 ** 6
    half = S // 2
    ref = torch.full((E, R, S), 5.0)
    ref[0, 0, :half] = 25.0                                   # robot 0 strong on the lower RBGs
    ref[0, 1, half:] = 25.0                                   # robot 1 strong on the upper RBGs
    u.slot(10, 0.0, net.cfg.ul_data_symbols, ref, torch.zeros(E, R, S), 0, gh=4)
    assert float(u.ctr["tb_retx"]) == 0.0                     # short of its 11 RBGs: not sent
    assert float(u.ctr["tb_new"]) == 1.0
    assert int(u.h_state[0, 0, 0]) == 1                       # the retransmission waits for the next slot
    assert int(u.h_nsb[0, 1, :].max()) == S                   # robot 1's new TB spans every RBG
    assert float(u.ctr["prb_used"]) == float(u.sb_prb.sum())


# ---------------------------------------------------------------- F11
def test_wideband_rate_respects_mcs_cap():
    """pf_wideband at high SNR with ul_mcs_max=4: the scheduler sizes the grant with the capped rate, so a need of
    1.5 capped RBG-rates takes 2 RBGs (the uncapped rate would cover it with one)."""
    cfg = NRConfig(scheduler="pf_wideband", ul_mcs_max=4, olla=False)
    net = NRNet(1, 1, dev, (4000.0,), cfg)
    u, phy = net.ul, net.ul.phy
    nsym = cfg.ul_data_symbols
    re_prb = float(min(12 * nsym - cfg.dmrs_re_per_prb - cfg.overhead_re_per_prb, 156))
    w0 = float(u.sb_prb[0])
    rate_cap = float(phy.se[4]) * re_prb / 8 * w0
    rate_top = float(phy.se[phy.M - 1]) * re_prb / 8 * w0
    need = int(1.5 * rate_cap)
    assert need < rate_top
    u.q.enq[:] = need
    u.bsr[:] = need
    _ul_slot(net, 10, 40.0)
    assert float(u.ctr["tb_new"]) == 1.0
    assert float(u.ctr["prb_used"]) == 2 * w0


# ---------------------------------------------------------------- F12
def test_slot_tap_energy_scales_with_pusch_symbols():
    E, R = 2, 3
    eng = make_engine("L2", E, R, "cpu", NRConfig(), seed=1)
    tap = SlotTap.of(eng)
    link = eng.net.ul
    S = eng.net.S
    won = torch.ones(E, R, S, dtype=torch.bool)
    n_prb = torch.full((E, R), float(link.sb_prb.sum()))
    act = torch.zeros(E, R, S)
    e = {}
    for nsym in (12, 2):
        tap.begin()
        link.slot_nsym = nsym
        link.sinr_hook(0, "ul", won, n_prb, act)
        e[nsym] = tap.ul_tx_j.clone()
    c = eng.config
    p_w = 10 ** ((c.ue_tx_dbm - 30) / 10)                     # whole UE power, no power control
    assert torch.allclose(e[12], torch.full((E, R), p_w * c.slot_ms * 1e-3 * 12 / 14), rtol=1e-5)
    assert torch.allclose(e[2] * 6, e[12], rtol=1e-6)
    # the engine sets the symbol count of every UL data slot it runs (U slots: ul_data_symbols)
    eng.submit(None, Requests(torch.ones(E, R, dtype=torch.long)))
    tap.begin()
    eng.step(None, torch.full((E, R), 20.0))
    assert link.slot_nsym == c.ul_data_symbols
    assert float(tap.ul_slots.sum()) > 0
