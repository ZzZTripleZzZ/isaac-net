"""Mini-slot (type B) uplink grants of the NR engine (NRConfig.ul_mini_slot_symbols, mini_slot_dl; docs/configurability.md
"Mini-slot grants").

CPU
  (a) feature off is bitwise the pre-feature engine: the live NRNet against the same class with the frozen pre-feature
      slot loops (tests/nr_frozen/minislot_pre.py: step, step_cells), one and three cells, UL + DL, rng engine and
      global, with a partial reset; plus the occasion layout (occasion_symbols) and config validation
  (b) timing and capacity: an isolated small frame whose TB fits the first occasion completes exactly 10/14 slot
      earlier with m = 2 than with whole slots (same SR and K2 timing in slots); at light load the mean UL delay of
      10-byte commands decreases strictly and the delay quantiles do not increase as m shrinks (7, 4, 2), and the DL
      delay (mini_slot_dl) drops too; a 200-byte frame that outgrows the first occasion's TB waits for the next slot's
      grant (the occasions share the slot's BSR); at saturation the throughput ratio to whole slots stays within the
      data-RE ratio (one DMRS symbol per occasion) up to the TBS / BLER-table slack
  (c) conservation (every accepted frame is delivered, timed out or dropped once, or still queued), HARQ
      retransmissions complete and never come earlier than ul_rtt slots after the failed transmission
  (d) E-independence and partial-reset isolation with mini-slots on (one and three cells, poses through the radio)
  (e) unused_fields honesty, validation errors, the triton refusal; SlotTap counts an occasion as its slot share
GPU: nr_equiv's ul_minislot2 / ul_minislot4 are in the G1 and G7 (graph bitwise) lists of test_nr_fast.py; triton
refuses mini-slots (not in G2).
"""
import math

import pytest
import torch

import nr_equiv
from isaac_net.core import NRConfig, Requests, make_engine
from isaac_net.core.config import multicell, occasion_symbols
from isaac_net.core.nr_engine import NRNet
from isaac_net.core.nr_fast import NRTritonEngine
from isaac_net.core.slot_tap import SlotTap
from nr_frozen.minislot_pre import PreMiniSlot

SIZES = (4000.0, 30000.0)
FAST = dict(control_step_ms=10.0)      # 20 slots per control step


class OldLoop(NRNet):
    """The live engine with the pre-feature slot loops (verbatim, frozen at 95f2f59)."""
    step = PreMiniSlot.step
    step_cells = PreMiniSlot.step_cells


def _drive_net(net, steps=10, reset_at=4, seed=0, snr_lo=0.0):
    E, R, C = net.E, net.R, net.C
    g = torch.Generator().manual_seed(seed)
    z = torch.zeros(E, dtype=torch.long)
    outs = []
    for t in range(steps):
        if t == reset_at:
            net.reset(torch.tensor([1]))
        send = torch.randint(0, 3, (E, R), generator=g)
        net.add_frames(t, send, torch.zeros(E, R, dtype=torch.bool), z, 25 * torch.rand(E, R, generator=g))
        if net.dl is not None:
            net.add_dl_frames(t, send.float() * 2000)
        if C == 1:
            outs.append(net.step(t, snr_lo + 25 * torch.rand(E, R, generator=g), z, full=True))
        else:
            outs.append(net.step_cells(t, -60 - 40 * torch.rand(E, R, C, generator=g), z, full=True))
    return outs


def _same_outs(a, b):
    for x, y in zip(a, b):
        assert x.keys() == y.keys()
        for k in x:
            assert torch.equal(x[k].nan_to_num(-7.0), y[k].nan_to_num(-7.0)), k


# ---------------------------------------------------------------------------------------------------------- (a)
@pytest.mark.parametrize("cfg", [NRConfig(dl=True, **FAST), multicell(3, dl=True, **FAST), NRConfig(rng="global", **FAST),
                                 NRConfig(dl=True, special_ul_data=True, ul_grant_model="bsr", **FAST)],
                         ids=["c1_dl", "c3_dl", "c1_global", "c1_bsr_sul"])
def test_a_off_is_bitwise_the_pre_feature_engine(cfg):
    assert cfg.ul_mini_slot_symbols is None and not cfg.mini_slot_dl
    outs = []
    for cls in (NRNet, OldLoop):
        torch.manual_seed(0)
        net = cls(3, 4, "cpu", SIZES, cfg, generator=torch.Generator().manual_seed(4), seed=8)
        net.log_stats = True
        outs.append((_drive_net(net), net.counters(), net.collect()))
    (a, ca, sa), (b, cb, sb) = outs
    _same_outs(a, b)
    assert ca == cb
    assert torch.equal(sa["delay"], sb["delay"])
    assert sum(int(o["delivered"].sum()) for o in a) > 0


def test_a_occasion_layout():
    assert occasion_symbols(12) == (12,) and occasion_symbols(0, 2) == ()
    assert occasion_symbols(12, 2) == (2,) * 6
    assert occasion_symbols(12, 4) == (4, 4, 4)
    assert occasion_symbols(12, 7) == (7, 5)
    assert occasion_symbols(13, 2) == (2,) * 5 + (3,)          # an exact half rounds down: no 1-symbol occasion
    assert occasion_symbols(2, 4) == (2,) and occasion_symbols(3, 7) == (3,)
    for n in range(1, 15):
        for m in (2, 4, 7):
            occ = occasion_symbols(n, m)
            assert sum(occ) == n and all(x == m for x in occ[:-1])
            assert len(occ) == 1 or m / 2 < occ[-1] <= 1.5 * m
    c = NRConfig(ul_mini_slot_symbols=4, special_ul_data=True, mini_slot_dl=True, dl=True)
    assert c.slot_occasions(4) == ((), (4, 4, 4))                       # DDDSU: U slot
    assert c.slot_occasions(3) == ((4, 5), (2,))                        # S slot: 9 DL data symbols, 2 UL
    assert c.slot_occasions(0) == ((4, 4, 5), ())
    assert NRConfig().ul_occasions_per_step == NRConfig().ul_slots_per_step
    assert NRConfig(ul_mini_slot_symbols=2).ul_occasions_per_step == 6 * NRConfig().ul_slots_per_step


# ---------------------------------------------------------------------------------------------------------- (b)
def _light(m, steps=60, size=200.0, dl=False, seed=1, **kw):
    cfg = NRConfig(control_step_ms=10.0, msg_sizes=(size,), ul_mini_slot_symbols=m, dl=dl,
                   mini_slot_dl=dl and m is not None, **kw)
    E, R = 8, 4
    net = NRNet(E, R, "cpu", cfg.msg_sizes, cfg, seed=3)
    net.log_stats = True
    g = torch.Generator().manual_seed(seed)
    z = torch.zeros(E, dtype=torch.long)
    for t in range(steps):
        send = (torch.rand(E, R, generator=g) < 0.5).long()
        net.add_frames(t, send, torch.zeros(E, R, dtype=torch.bool), z, torch.full((E, R), 15.0))
        if dl:
            net.add_dl_frames(t, send.float() * size)
        net.step(t, torch.full((E, R), 15.0), z)
    return net, net.collect()


def test_b_first_occasion_timing_exact():
    """No fading, high SNR, one 10-byte frame per robot every other step: the SR, the grant (sr_delay slots) and the
    granted slot are the same, and with m = 2 the TB of the first occasion (one RBG, 2 symbols) carries the frame, so
    every delay is the whole-slot delay minus the 10 symbols that follow the first occasion."""
    res = {}
    for m in (None, 2):
        cfg = NRConfig(control_step_ms=10.0, msg_sizes=(10.0,), ul_mini_slot_symbols=m, fading=False)
        E, R = 2, 3
        net = NRNet(E, R, "cpu", cfg.msg_sizes, cfg, seed=3)
        net.log_stats = True
        z = torch.zeros(E, dtype=torch.long)
        for t in range(12):
            send = torch.full((E, R), t % 2, dtype=torch.long)
            net.add_frames(t, send, torch.zeros(E, R, dtype=torch.bool), z, torch.full((E, R), 30.0))
            net.step(t, torch.full((E, R), 30.0), z)
        res[m] = net.collect()["delay"].double()
        assert net.counters()["ul"]["tb_retx"] == 0
    assert res[None].numel() == res[2].numel() == 6 * 6
    slot_steps = 1.0 / NRConfig(control_step_ms=10.0).slots_per_step
    assert torch.allclose(res[None] - res[2], torch.full_like(res[2], 10 / 14 * slot_steps), atol=1e-6)


def test_b_light_load_delay_decreases_with_m():
    """10-byte commands (they fit the TB of the SR's one-RBG grant in every occasion length)."""
    stats = {m: _light(m, size=10.0)[1]["delay"].double() for m in (None, 7, 4, 2)}
    n = stats[None].numel()
    assert n > 500 and all(abs(v.numel() - n) <= 2 for v in stats.values())
    means = [float(stats[m].mean()) for m in (None, 7, 4, 2)]
    assert all(a > b for a, b in zip(means, means[1:])), means
    for q in (0.5, 0.9):
        qs = [float(torch.quantile(stats[m], q)) for m in (None, 7, 4, 2)]
        assert all(a >= b - 1e-9 for a, b in zip(qs, qs[1:])), (q, qs)


def test_b_frame_larger_than_the_first_occasion_waits_for_the_next_slot():
    """Occasions share the slot's timing: the BSR that the first occasion's PUSCH carries reaches the scheduler at the
    next slot, as after a whole-slot PUSCH. A 200-byte frame that needs more than the TB of the SR's one-RBG grant
    (about 30 bytes in 2 symbols, about 330 bytes in 12) therefore finishes one TDD period later with m = 2."""
    d0 = _light(None)[1]["delay"].double()
    d2 = _light(2)[1]["delay"].double()
    assert float(torch.quantile(d2, 0.5)) > float(torch.quantile(d0, 0.5))


def test_b_mini_slot_dl_cuts_dl_delay():
    d0 = _light(None, dl=True)[1]["dl_delay"].double()
    d2 = _light(2, dl=True)[1]["dl_delay"].double()
    assert d0.numel() > 500 and abs(d0.numel() - d2.numel()) <= 2
    assert float(d2.mean()) < float(d0.mean())
    assert float(torch.quantile(d2, 0.5)) < float(torch.quantile(d0, 0.5))


def _re(nsym, cfg):
    return 12 * nsym - cfg.dmrs_re_per_prb - cfg.overhead_re_per_prb


@pytest.mark.parametrize("m", [7, 4, 2])
def test_b_saturation_within_the_re_ratio(m):
    """Every robot backlogged: the UL carrier is full in both runs, so the served bytes scale with the data RE per
    slot, sum over the occasions of (12 L - DMRS - overhead) against 12 x 12 - DMRS - overhead for the whole slot
    (m = 2: 72 / 132 = 0.545, 4: 108 / 132 = 0.818, 7: 120 / 132 = 0.909). The TBS of TS 38.214 quantizes a small
    N_info on a finer grid and the BLER depends on the TB size, so the measured ratio lies within a few percent."""
    served = {}
    for mm in (None, m):
        net, _ = _light(mm, steps=30, size=30000.0)
        c = net.counters()["ul"]
        assert c["prb_used"] / c["prb_avail"] > 0.98
        served[mm] = c["bytes_ok"]
    cfg = NRConfig()
    ratio = sum(_re(L, cfg) for L in occasion_symbols(cfg.ul_data_symbols, m)) / _re(cfg.ul_data_symbols, cfg)
    got = served[m] / served[None]
    assert ratio * 0.9 < got < ratio * 1.08, (got, ratio)


# ---------------------------------------------------------------------------------------------------------- (c)
@pytest.mark.parametrize("kw", [dict(harq_fail="drop", discard="purge"), dict(harq_fail="rlc_am", discard="pdcp_arrival"),
                                dict(dl=True, mini_slot_dl=True, n_harq=4)], ids=["drop", "rlc_am", "dl"])
def test_c_conservation_and_harq(kw):
    cfg = NRConfig(control_step_ms=10.0, ul_mini_slot_symbols=2, frame_buffer=6, timeout_steps=4,
                   msg_sizes=(300.0, 3000.0), **kw)
    E, R = 3, 5
    net = NRNet(E, R, "cpu", cfg.msg_sizes, cfg, seed=11)
    net.ul.trace = []
    g = torch.Generator().manual_seed(2)
    z = torch.zeros(E, dtype=torch.long)
    n_acc = n_out = 0
    seen = set()
    for t in range(40):
        send = torch.randint(0, 3, (E, R), generator=g)
        acc = net.add_frames(t, send, torch.zeros(E, R, dtype=torch.bool), z, torch.zeros(E, R))
        n_acc += int(acc.sum())
        if cfg.dl:
            net.add_dl_frames(t, send.float() * 500)
        o = net.step(t, -4 + 12 * torch.rand(E, R, generator=g), z, full=True)
        gone = o["delivered"] | o["timed_out"] | o["dropped"]
        assert not ((o["delivered"] & o["timed_out"]) | (o["delivered"] & o["dropped"])).any()
        e, r, f = gone.nonzero(as_tuple=True)
        ids = set(zip(e.tolist(), r.tolist(), o["cap"][e, r, f].tolist()))
        assert not ids & seen, "frame resolved twice"
        seen |= ids
        n_out += len(ids)
    assert n_acc == n_out + int(net.queued().sum())
    c = net.counters()["ul"]
    assert c["tb_retx"] > 0 and sum(c["ntx_hist"][2:]) > 0           # retransmissions decoded
    # a retransmission (same stream range) of a TB that failed in slot g comes no earlier than slot g + ul_rtt
    N = cfg.slots_per_step
    last_fail = {}
    n_chk = 0
    for frac, tx, ok, lo, hi, exh in net.ul.trace:
        gs = math.floor(float(frac) * N - 1e-6)
        for e, r in tx.nonzero().tolist():
            key = (e, r, int(lo[e, r]), int(hi[e, r]))
            if key in last_fail:
                assert gs - last_fail.pop(key) >= cfg.ul_rtt
                n_chk += 1
            if not ok[e, r] and not exh[e, r]:
                last_fail[key] = gs
    assert n_chk > 0


# ---------------------------------------------------------------------------------------------------------- (d)
RR, EMAX, STEPS = 3, 6, 7
D_CASES = {"c1": NRConfig(ul_mini_slot_symbols=2, dl=True, mini_slot_dl=True, **FAST),
           "c3": multicell(3, ul_mini_slot_symbols=4, **FAST)}


def _inputs():
    g = torch.Generator().manual_seed(0)
    return [(torch.randint(0, 3, (EMAX, RR), generator=g), torch.rand(EMAX, RR, 2, generator=g) * 150)
            for _ in range(STEPS)]


def _drive(case, E, resets=()):
    eng = make_engine("L2", E, RR, "cpu", D_CASES[case], seed=5)
    rs = dict(resets)
    outs = []
    for k, (send, pos) in enumerate(_inputs()):
        eng.submit(None, Requests(send[:E], None, torch.full((E,), k, dtype=torch.long)))
        o = eng.step(None, pos[:E])
        outs.append({n: v.clone() for n, v in o.items() if torch.is_tensor(v) and v.dim() >= 1 and v.shape[0] == E})
        if k in rs:
            eng.reset(torch.tensor(rs[k]))
    return outs


def _rows_equal(a, b, rows):
    n = 0
    for x, y in zip(a, b):
        assert x.keys() == y.keys()
        for name in x:
            assert torch.equal(x[name][rows].nan_to_num(-7.0), y[name][rows].nan_to_num(-7.0)), name
            n += 1
    assert n > 0


@pytest.mark.parametrize("case", list(D_CASES))
def test_d_env_independence_and_partial_reset(case):
    a = _drive(case, 3, {3: [1]})
    b = _drive(case, 6, {3: [1]})
    _rows_equal(a, b, slice(0, 3))
    c = _drive(case, 4)
    d = _drive(case, 4, {2: [1]})
    _rows_equal(c, d, [0, 2, 3])
    assert sum(int(o["delivered"].sum()) for o in a) > 0


# ---------------------------------------------------------------------------------------------------------- (e)
def test_e_unused_fields_and_validation():
    assert NRConfig(ul_mini_slot_symbols=2).unused_fields("L2") == []
    assert NRConfig(ul_mini_slot_symbols=2, dl=True, mini_slot_dl=True).unused_fields("L2") == []
    assert "ul_mini_slot_symbols" in NRConfig(ul_mini_slot_symbols=2).unused_fields("L1")
    assert "ul_mini_slot_symbols" in NRConfig(ul_mini_slot_symbols=2).unused_fields("L2-legacy")
    assert NRConfig(ul_mini_slot_symbols=2, mini_slot_dl=True).unused_fields("L2") == ["mini_slot_dl"]   # dl off
    assert NRConfig(ul=False, dl=True, ul_mini_slot_symbols=4).unused_fields("L2") == ["ul_mini_slot_symbols"]
    assert NRConfig(ul=False, dl=True, ul_mini_slot_symbols=4, mini_slot_dl=True).unused_fields("L2") == []
    with pytest.raises(ValueError, match="ul_mini_slot_symbols"):
        NRConfig(mini_slot_dl=True)
    with pytest.raises(ValueError, match="one of"):
        NRConfig(ul_mini_slot_symbols=3)
    with pytest.raises(ValueError, match="data RE"):
        NRConfig(ul_mini_slot_symbols=2, dmrs_re_per_prb=24)
    make_engine("L2", 2, 2, "cpu", NRConfig(ul_mini_slot_symbols=2), strict=True, seed=1)    # read: no error
    with pytest.raises(ValueError, match="ul_mini_slot_symbols"):
        make_engine("L1", 2, 2, "cpu", NRConfig(ul_mini_slot_symbols=2), strict=True, seed=1)


def test_e_triton_refuses_mini_slots():
    assert NRTritonEngine.refusals(NRConfig()) == []
    no = NRTritonEngine.refusals(NRConfig(ul_mini_slot_symbols=2))
    assert len(no) == 1 and "mini-slot" in no[0][0]
    with pytest.raises(ValueError):
        make_engine("L2", 2, 2, "cpu", NRConfig(ul_mini_slot_symbols=2), "triton")


@pytest.mark.parametrize("m", [None, 2])
def test_e_slot_tap_counts_occasion_shares(m):
    """Backlogged robots: SlotTap's UL PRB-slots per env and step never exceed the carrier (nprb per UL data slot),
    and its PUSCH time per robot never exceeds the slot time of the step's UL data slots."""
    cfg = NRConfig(ul_mini_slot_symbols=m, msg_sizes=(30000.0,), **FAST)
    E, R = 2, 4
    eng = make_engine("L2", E, R, "cpu", cfg, seed=2)
    tap = SlotTap.of(eng)
    for t in range(4):
        eng.submit(None, Requests(torch.ones(E, R, dtype=torch.long), None, torch.zeros(E, dtype=torch.long)))
        tap.begin()
        eng.step(None, torch.full((E, R), 20.0))
        per_env = tap.ul_prb.sum(-1)
        cap = cfg.nprb * cfg.ul_slots_per_step
        assert (per_env <= cap + 1e-3).all() and (per_env > 0.9 * cap).all() if t else True
        assert (tap.ul_slots <= cfg.ul_slots_per_step + 1e-6).all()
        assert (tap.ul_tx_s <= cfg.ul_slots_per_step * cfg.slot_ms * 1e-3 * cfg.ul_data_symbols / 14 + 1e-9).all()


@pytest.mark.parametrize("name", nr_equiv.MINISLOT_CFGS)
def test_e_nr_equiv_minislot_configs_run_on_the_reference(name):
    """The GPU lists G1 / G7 of test_nr_fast.py run these configs on the graph backend; on the CPU they run
    reference against reference (deterministic) and deliver frames on both links."""
    cfg = nr_equiv.CFGS[name]()
    assert cfg.ul_mini_slot_symbols is not None and NRTritonEngine.refusals(cfg)
    r = nr_equiv.run("reference", name, E=4, R=3, steps=6, seed=3, device="cpu", p_reset=0.3, phase_offset=25)
    assert r["bitwise"] and r["frames_delivered"][0] > 0, r
