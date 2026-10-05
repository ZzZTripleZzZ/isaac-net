"""Regression tests for the 2026-10-04 channel and Wi-Fi fixes: TR 38.901 height applicability (UMa / UMi
breakpoint, InF-SH / DH clutter), the Wi-Fi Poisson access cap, the busy time of frames lost to the residual FER,
the log-distance 1 m loss at the Wi-Fi carrier, and the invalid VHT-MCS combinations."""
import math

import pytest
import torch

from isaac_net.core import NRConfig, Requests, make_engine
from isaac_net.core.channels import tr38901 as tr
from isaac_net.core.radio import RadioMC
from isaac_net.core.wifi import WifiConfig
from isaac_net.core.wifi.engine import free_space_1m_db, poisson_cap, poisson_icdf, radio_config
from isaac_net.core.wifi.eventsim import run
from isaac_net.core.wifi.phy import VHT_INVALID, mcs_numbers, mcs_table
from isaac_net.core.wifi.validate import WifiBusy, _wifi_station, meanfield_saturated


# ---------------------------------------------------------------- TR 38.901 heights
@pytest.mark.parametrize("scn", ["UMa", "UMi"])
@pytest.mark.parametrize("h_ut", [0.3, 0.5, 1.0, 1.2, 23.0])
def test_uma_umi_height_out_of_range_raises(scn, h_ut):
    h_bs = tr.SCENARIOS[scn].h_bs
    with pytest.raises(ValueError, match="ue_height_m"):
        tr.check_heights(scn, h_bs, h_ut)
    with pytest.raises(ValueError, match="InF|log_distance"):
        tr.breakpoint_m(scn, 3.5, h_bs, h_ut)
    d2 = torch.tensor([50.0, 500.0])
    with pytest.raises(ValueError):
        tr.pl_los(scn, d2, torch.sqrt(d2 ** 2 + (h_bs - h_ut) ** 2), 3.5, h_bs, h_ut)


@pytest.mark.parametrize("scn", ["UMa", "UMi", "RMa"])
@pytest.mark.parametrize("h_ut", [1.5, 2.0, 10.0])
def test_breakpoint_positive_in_range(scn, h_ut):
    h_bs = tr.SCENARIOS[scn].h_bs
    tr.check_heights(scn, h_bs, h_ut)
    assert tr.breakpoint_m(scn, 3.5, h_bs, h_ut) > 0
    # the hand value of the default height: d'_BP = 4 (h_BS - 1)(h_UT - 1) fc / c
    if scn != "RMa" and h_ut == 1.5:
        assert abs(tr.breakpoint_m(scn, 3.5, h_bs, 1.5) - 4 * (h_bs - 1) * 0.5 * 3.5e9 / 3e8) < 1e-9


def test_rma_range_and_bs_height():
    tr.check_heights("RMa", 35.0, 1.0)                      # RMa allows 1 m <= h_UT <= 10 m
    with pytest.raises(ValueError, match="RMa"):
        tr.check_heights("RMa", 35.0, 0.5)
    with pytest.raises(ValueError, match="RMa"):
        tr.check_heights("RMa", 35.0, 12.0)
    with pytest.raises(ValueError, match="BS height"):
        tr.check_heights("UMi", 1.0, 1.5)                   # h_BS = h_E: d'_BP = 0


def test_uma_low_ue_raises_through_radio():
    with pytest.raises(ValueError, match="UMa"):                # checked when the channel is built
        RadioMC(NRConfig(channel="tr38901_uma", ue_height_m=0.5), 2, "cpu", R=3)
    ok = RadioMC(NRConfig(channel="tr38901_uma"), 2, "cpu", R=3).rx_dbm(torch.full((2, 3, 2), 60.0))
    assert torch.isfinite(ok).all()


@pytest.mark.parametrize("scn", ["InF-SH", "InF-DH"])
def test_inf_clutter_heights(scn):
    h_bs, hc = tr.SCENARIOS[scn].h_bs, tr.INF_CLUTTER[scn][2]
    for h_ut in (hc, hc + 0.5):                             # UT at or above the clutter
        with pytest.raises(ValueError, match="clutter"):
            tr.inf_k_subsce(scn, h_bs, h_ut)
    with pytest.raises(ValueError, match="clutter"):        # BS below the clutter
        tr.inf_k_subsce(scn, 1.0, 0.5, h_c=hc)
    with pytest.raises(ValueError, match="clutter"):
        RadioMC(NRConfig(channel="tr38901", tr38901_scenario=scn, ue_height_m=hc + 1.0), 2, "cpu", R=3)


@pytest.mark.parametrize("scn", ["InF-SL", "InF-DL", "InF-SH", "InF-DH"])
@pytest.mark.parametrize("h_ut", [0.3, 0.5, 1.5])
def test_inf_plos_in_unit_interval(scn, h_ut):
    """Ground robots (h_UT down to 0.3 m) are valid in every InF sub-scenario, with 0 <= Pr_LOS <= 1."""
    k = tr.inf_k_subsce(scn, tr.SCENARIOS[scn].h_bs, h_ut)
    assert k > 0 and math.isfinite(k)
    d = torch.linspace(0.0, 200.0, 101)
    p = tr.p_los(scn, d, h_ut, k)
    assert (p >= 0).all() and (p <= 1).all()


# ---------------------------------------------------------------- Wi-Fi Poisson cap
def test_poisson_cap_does_not_bind():
    g = torch.Generator().manual_seed(0)
    for lam_max in (1.0, 5.0, 13.0, 40.0):
        n = poisson_cap(lam_max)
        lam = torch.full((200_000,), lam_max)
        x = poisson_icdf(lam, torch.rand(200_000, generator=g), n)
        assert abs(float(x.mean()) / lam_max - 1) < 0.01
        assert float((x >= n).float().mean()) < 1e-4


def _goodput(noise, bw, sub, steps=25, E=6):
    wc = WifiConfig(standard="ax", bandwidth_mhz=bw, max_ampdu_bytes=0, substep_ms=sub, access_noise=noise)
    c = NRConfig(wifi=wc, msg_sizes=(800_000.0,), frame_buffer=64, timeout_steps=1000)
    net = make_engine("WIFI", E, 1, "cpu", c, seed=0)
    served = 0.0
    for _ in range(steps):
        net.submit(None, Requests(torch.ones(E, 1, dtype=torch.long)))
        before = float(net.rem.sum())
        net.step(None, torch.full((E, 1), 50.0))
        served += before - float(net.rem.sum())
    return served * 8 / (steps * c.control_step_ms * 1e-3) / E / 1e6


@pytest.mark.parametrize("bw,sub", [(80, 2.0), (160, 5.0)])
def test_poisson_goodput_matches_mean(bw, sub):
    """The review's scenario: one robot per env, 800 kB per step, no aggregation. The cap of 8 accesses per
    sub-step gave 42.5 vs 52.2 Mb/s (80 MHz / 2 ms) and 18.8 vs 55.5 (160 MHz / 5 ms)."""
    p, m = _goodput("poisson", bw, sub), _goodput("mean", bw, sub)
    assert abs(p / m - 1) < 0.03, (p, m)


# ---------------------------------------------------------------- Wi-Fi FER busy time
def test_fer_failure_busy_time_is_ts():
    """A lone frame lost to the residual FER occupies the medium for Ts, as in meanfield.slot_time: with one
    station, RTS/CTS (where Tc << Ts) and FER = 0.5, every attempt, received or not, is busy for Ts - AIFS."""
    wc = WifiConfig(rts_cts=True, frame_error_rate=0.5)
    rate = mcs_table("ax", 20)[0][7]
    st = _wifi_station(wc, "BE", rate, B=30000)
    ts_busy, tc_busy = WifiBusy(wc, rate, 3, 0)(30000), WifiBusy(wc, rate, 3, 1)(30000)
    assert tc_busy < 0.1 * ts_busy
    r = run([st], 4e6, wc.slot_us, wc.sifs_us, None, seed=1)
    busy_per_att = r["busy_frac"] * r["sim_us"] / r["n_att"].sum()
    assert abs(busy_per_att / ts_busy - 1) < 0.01


@pytest.mark.parametrize("rts", [False, True])
def test_fer_meanfield_vs_event(rts):
    wc = WifiConfig(rts_cts=rts, frame_error_rate=0.3)
    rate = mcs_table("ax", 20)[0][7]
    st = _wifi_station(wc, "BE", rate, B=30000)
    thr_ev = sum(float(run([st] * 2, 4e6, wc.slot_us, wc.sifs_us, None, seed=s)["throughput_bps"].sum())
                 for s in range(3)) / 3 / 1e6
    thr_mf = meanfield_saturated(wc, [(2, "BE", rate, st.cap)])[0]["thr_mbps"]
    assert abs(thr_mf / thr_ev - 1) < 0.04, (thr_mf, thr_ev)


# ---------------------------------------------------------------- Wi-Fi log-distance constant
def test_wifi_log_distance_constant_from_carrier():
    assert abs(free_space_1m_db(5.2) - 46.77) < 0.01
    assert abs(radio_config(NRConfig(), WifiConfig()).pl_const_db - free_space_1m_db(5.2)) < 1e-9
    assert abs(radio_config(NRConfig(), WifiConfig(carrier_ghz=6.0)).pl_const_db - free_space_1m_db(6.0)) < 1e-9
    assert radio_config(NRConfig(pl_const_db=44.0), WifiConfig()).pl_const_db == 44.0          # user value kept
    assert radio_config(NRConfig(channel="tr38901_inf_sh"), WifiConfig()).pl_const_db == 40.0  # not log_distance
    assert NRConfig().pl_const_db == 40.0                                                      # NR unchanged


def test_wifi_snr_from_poses_uses_wifi_constant():
    c = NRConfig(wifi=WifiConfig(), shadow_sigma_db=0.0)
    net = make_engine("WIFI", 1, 1, "cpu", c, seed=0)
    net.submit(None, Requests(torch.zeros(1, 1, dtype=torch.long)))
    out = net.step(None, torch.tensor([[[10.0, 0.0]]]))
    wc = WifiConfig()
    want = wc.sta_tx_dbm - free_space_1m_db(5.2) - 10 * c.pathloss_exp * 1.0 - wc.noise_dbm()
    assert abs(float(out["sinr_db"]) - want) < 0.5


# ---------------------------------------------------------------- VHT-MCS validity
def test_vht_invalid_combinations_excluded():
    assert mcs_numbers("ac", 80, 3) == [0, 1, 2, 3, 4, 5, 7, 8, 9]
    assert mcs_numbers("ac", 160, 3) == list(range(9))
    assert mcs_numbers("ac", 20, 3) == list(range(10))
    assert mcs_numbers("ac", 20, 1) == list(range(9))
    assert mcs_numbers("ax", 80, 3) == list(range(12))
    for bw in (20, 40, 80, 160):
        for n in (1, 2, 3, 4):
            ids = mcs_numbers("ac", bw, n)
            assert not any((bw, m, n) in VHT_INVALID for m in ids)
            rates, thr = mcs_table("ac", bw, n)
            assert len(rates) == len(ids) and all(a < b for a, b in zip(rates, rates[1:]))
    # 80 MHz, 3 SS, MCS 7 at 0.8 us GI: 234 x 6 x 5/6 x 3 / 4 us = 877.5 Mb/s
    assert abs(mcs_table("ac", 80, 3)[0][6] - 877.5) < 1e-9


def test_wifi_mcs_output_is_mcs_number():
    c = NRConfig(wifi=WifiConfig(standard="ac", bandwidth_mhz=80, n_ss=3))
    net = make_engine("WIFI", 1, 1, "cpu", c, seed=0)
    net.submit(None, Requests(torch.zeros(1, 1, dtype=torch.long)))
    thr = mcs_table("ac", 80, 3)[1]
    snr = (thr[6] + thr[7]) / 2                               # between table entries 6 and 7 = MCS 7 and MCS 8
    out = net.step(None, torch.tensor([[snr]]))
    assert int(out["wifi_mcs"]) == 7
