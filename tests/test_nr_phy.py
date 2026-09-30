"""NR PHY abstraction (merged from nrconfig/tests/test_phy.py and bler_curves.py).

A: hand-worked TS 38.214 examples and config tables (always).
B: exhaustive cross-check against Sionna SYS 2.2.0 (skipped unless sionna is importable; marked slow).
C: 5G-LENA table semantics (skipped unless the locally generated LENA tables exist).
D: BLER table sanity: monotone, gap to Shannon, agreement of identical (Qm, R) across MCS tables.
"""
import math
import os

import numpy as np
import pytest
import torch

from isaaclab_net.core.config import NRConfig, netslot_compat
from isaaclab_net.core.phy import MCS_TABLES, PHY, lena_tables_path, segment, segment_ldpc_k, tbs_38214, tbs_lena

dev = "cpu"
T = lambda *a: torch.tensor(a, device=dev)


# ---------------- A. spec examples ----------------
def test_a1_tbs_mcs0_one_prb():
    # N'_RE = 12*12-12 = 132, N_info = 30.9, n = 3, N'_info = 24 -> TBS 24
    assert int(tbs_38214(T(2.), T(120 / 1024), T(1.), 12, 12)) == 24


def test_a2_tbs_mcs28_50prb():
    # N_info = 36660.9 > 3824, N'_info = 36864, C = 5, TBS = 8*5*ceil(36888/40) - 24 = 36896
    assert int(tbs_38214(T(6.), T(948 / 1024), T(50.), 12, 12)) == 36896


def test_a3_tbs_mcs10_10prb():
    assert int(tbs_38214(T(4.), T(340 / 1024), T(10.), 12, 12)) == 1800


def test_a4_tbs_low_rate_branch():
    assert int(tbs_38214(T(2.), T(193 / 1024), T(273.), 13, 12)) == 14856


def test_a5_re_cap_156():
    assert int(tbs_38214(T(2.), T(120 / 1024), T(1.), 14, 0)) == int(tbs_38214(T(2.), T(120 / 1024), T(1.), 13, 0))


def test_a6_segmentation():
    cb, c = segment(T(36896), T(948 / 1024))
    assert int(cb) == 7408 and int(c) == 5


def test_a7_config_tables():
    c1 = NRConfig(mu=1, bandwidth_mhz=20)
    c0 = NRConfig(mu=0, bandwidth_mhz=20, tdd_pattern="DDDSU")
    c2 = NRConfig(mu=2, bandwidth_mhz=100, tdd_pattern="DDSU")
    c3 = NRConfig(mu=1, bandwidth_mhz=100, tdd_pattern="DSUUU")
    assert (c1.nprb, c1.rbg, c1.n_subbands, c1.subband_prbs[-1]) == (51, 4, 13, 3)
    assert (c0.nprb, c0.rbg) == (106, 8) and (c2.nprb, c2.rbg, c2.n_subbands) == (135, 8, 17)
    assert (c3.nprb, c3.rbg, c3.n_subbands) == (273, 16, 18)


def test_a8_compat_slots():
    cc, c2, c3 = netslot_compat(), NRConfig(mu=2, bandwidth_mhz=100, tdd_pattern="DDSU"), \
        NRConfig(mu=1, bandwidth_mhz=100, tdd_pattern="DSUUU")
    assert (cc.slots_per_step, cc.ul_slots_per_step, cc.n_subbands) == (200, 40, 5)
    assert (c3.ul_slots_per_step, c2.slots_per_step, c2.ul_slots_per_step) == (120, 400, 100)


def test_a9_special_slot_symbols():
    cs = NRConfig(tdd_pattern="DDDSU", special_ul_data=True, special_split=(6, 4, 4))
    assert cs.slot_symbols(3) == (5, 4) and cs.slot_symbols(0) == (13, 0) and cs.slot_symbols(4) == (0, 12)


# ---------------- B. Sionna cross-check ----------------
@pytest.mark.slow
def test_b_sionna_cross_check():
    pytest.importorskip("sionna")
    from sionna.phy.nr.utils import MCSDecoderNR, calculate_tb_size
    from sionna.sys import PHYAbstraction
    dec = MCSDecoderNR()
    for tab in (1, 2):
        idx = torch.arange(len(MCS_TABLES[tab]))
        for cat in (0, 1):
            qm, r = dec(idx, torch.full_like(idx, tab), torch.full_like(idx, cat))
            ours = torch.tensor(MCS_TABLES[tab], dtype=torch.float64)
            assert torch.equal(qm.long().cpu(), ours[:, 0].long())
            assert torch.allclose(r.double().cpu(), ours[:, 1] / 1024, atol=1e-6)
    bad = 0
    for tab in (1, 2):
        mc = torch.tensor(MCS_TABLES[tab], dtype=torch.float64)
        for dmrs in (0, 12, 24):
            for nsym in range(3, 15):
                if 12 * nsym - dmrs <= 0:
                    continue
                prb = torch.arange(1, 276, dtype=torch.float64)
                q, rr = mc[:, 0][:, None].expand(-1, 275), (mc[:, 1] / 1024)[:, None].expand(-1, 275)
                p = prb[None, :].expand(len(mc), -1)
                ours = tbs_38214(q, rr, p, nsym, dmrs)
                tb, cbs, ncb, *_ = calculate_tb_size(modulation_order=q.int(), target_coderate=rr.float(),
                                                     num_prbs=p.int(), num_ofdm_symbols=nsym, num_dmrs_per_prb=dmrs,
                                                     return_cw_length=True, device="cpu")
                cb_o, c_o = segment(ours, rr)
                bad += int(((ours != tb.long()) | (cb_o.round().long() != cbs.long()) | (c_o.long() != ncb.long())).sum())
    assert bad == 0
    pa = PHYAbstraction()
    for tab in (1, 2):
        phy = PHY("ul", tab, dev)
        avail = [m for m in sorted(pa.bler_table["category"][1]["index"][tab]["MCS"].keys()) if not (tab == 2 and m == 27)]
        n = 5000
        m = torch.tensor(avail)[torch.randint(len(avail), (n,))]
        snr, cbs = torch.rand(n) * 25 - 5, torch.rand(n) * (8424 - 24) + 24
        err = (phy.bler_lookup(m, snr, cbs) - pa.get_bler(m, torch.full_like(m, tab), torch.full_like(m, 1), cbs,
                                                          10 ** (snr / 10)).float()).abs()
        assert float(err.mean()) < 0.01


# ---------------- C. 5G-LENA (local tables only) ----------------
def test_c_lena_tbs_and_cb_size_examples():
    # these re-implement 5G-LENA's observable semantics and need no LENA data
    assert int(tbs_lena(T(6.), T(948 / 1024), T(50.), 13)) == 4949 * 8
    k, c, bg = segment_ldpc_k(T(39592), T(948 / 1024))
    assert (int(k), int(c), int(bg)) == (8448, 5, 0)


def test_c_lena_tables_missing_gives_instructions(tmp_path, monkeypatch):
    from isaaclab_net.core import phy as phy_mod
    monkeypatch.setenv("ISAACLAB_NET_LENA_TABLES", str(tmp_path / "none.npz"))
    phy_mod._TAB_CACHE.clear()
    with pytest.raises(FileNotFoundError, match="extract_lena_tables"):
        PHY("ul", 1, dev, source="lena")


@pytest.mark.skipif(not os.path.exists(lena_tables_path()), reason="5G-LENA tables not generated locally")
def test_c_lena_betas_equal_sionna():
    for tab in (1, 2):
        assert torch.allclose(PHY("ul", tab, "cpu", source="lena").beta, PHY("ul", tab, "cpu").beta)


# ---------------- D. BLER table sanity ----------------
@pytest.mark.parametrize("direction", ["ul", "dl"])
@pytest.mark.parametrize("tab", [1, 2])
def test_d_bler_monotone_and_shannon_gap(direction, tab):
    phy = PHY(direction, tab, dev)
    assert bool((phy.bler[..., 1:] <= phy.bler[..., :-1] + 1e-6).all())
    thr = phy.thr_lookup(torch.full((phy.M,), 1524.0))
    assert bool((thr[1:] >= thr[:-1] - 1e-6).all())
    gaps = [float(thr[m]) - 10 * math.log10(2 ** (q * r / 1024) - 1) for m, (q, r) in enumerate(MCS_TABLES[tab])]
    assert 0.3 < min(gaps) and max(gaps) < 6.5, (min(gaps), max(gaps))


@pytest.mark.parametrize("direction", ["ul", "dl"])
def test_d_same_qm_r_agree_across_tables(direction):
    p1, p2 = PHY(direction, 1, dev), PHY(direction, 2, dev)
    diffs = [float(p1.thr_ref[MCS_TABLES[1].index(e)] - p2.thr_ref[m2]) for m2, e in enumerate(MCS_TABLES[2])
             if e in MCS_TABLES[1]]
    assert diffs and max(abs(d) for d in diffs) <= 0.9


def test_d_link_adaptation_respects_target():
    """select_mcs picks the highest MCS whose TB error at the effective SINR is <= the BLER target."""
    phy = PHY("ul", 1, dev, bler_target=0.1)
    E, R, S = 3, 4, 5
    est = torch.linspace(-5, 25, E * R * S).view(E, R, S)
    won = torch.ones(E, R, S, dtype=torch.bool)
    w = torch.full((S,), 10.0)
    mcs, tbs = phy.select_mcs(est, won, torch.zeros(E, R), torch.full((E, R), 50.0), 12, 12, 0, "eesm", w)
    eff = phy.eff_sinr(est, won, mcs, "eesm", w)
    p = phy.tb_error_prob(mcs, eff, tbs)
    assert bool(((p <= 0.1 + 1e-6) | (mcs == 0)).all()) and (mcs.max() > mcs.min())
    assert np.all(tbs.numpy() > 0)
