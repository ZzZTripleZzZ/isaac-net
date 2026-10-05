"""Regression tests for the config / wrapper fixes of the 2026-10-04 code review (items 13, 17-21).

  13  fading_rho_from_speed is monotone non-increasing in speed (rho = 0 past the first zero of J0)
  17  make_adaptive applies NRConfig.energy around the adaptive engine (batteries drain) and refuses background users
  18  unused_fields follows the switches (dl, n_cells, noise_model, channel, blockage, fading, tbs_mode, NetSlotMC)
  19  EnergyConfig.seed wins over the engine seed; EnergyLoop refuses the legacy step form; the offered-load
      BackgroundLoop accounts the legacy form exactly like the dict form
  20  EdgeLoop nr_dl refuses the graph / triton NR backends; act_age is NaN before the first action; an adaptive
      L0 with an empirical {"q", "p"} marginal takes frames back without KeyError
  21  srsran_like / oai_like derive slot counts from the merged mu / tdd_pattern; with_ keeps an explicit
      fading_rho_per_ms; isaac_net.LEVELS lists WIFI
"""
import math

import pytest
import torch

import isaac_net
from isaac_net.core import EdgeConfig, EdgeLoop, NRConfig, Requests, make_engine
from isaac_net.core.adaptive import AdaptiveEngine, FidelityConfig, make_adaptive
from isaac_net.core.background import BackgroundConfig
from isaac_net.core.config import (fading_rho_from_speed, lena_like, lena_match_v2, lena_validation,
                                   lena_validation_v2, multicell, netslot_compat, oai_like, srsran_like)
from isaac_net.core.energy import EnergyConfig, EnergyLoop

E, R = 4, 3


# ---------------------------------------------------------------- 13: fading from speed
def test_fading_rho_monotone_in_speed():
    v = [0.25 * i for i in range(241)]                       # 0 .. 60 m/s
    rho = [fading_rho_from_speed(x) for x in v]
    assert all(b <= a for a, b in zip(rho, rho[1:]))
    assert fading_rho_from_speed(13.0) > 0.0 and fading_rho_from_speed(13.2) == 0.0
    assert fading_rho_from_speed(35.0) == 0.0 == fading_rho_from_speed(55.0)    # was 0.57 at 35 m/s
    assert NRConfig(ue_speed_mps=20.0).fading_rho_per_ms == 0.0


# ---------------------------------------------------------------- 18: unused_fields follows the switches
def test_unused_fields_review_example():
    cfg = NRConfig(k1=5, dl_mcs_max=10, a3_hyst_db=1, li_alpha=0.5, gnb_nf_db=9, tr38901_los="los",
                   radio_map_path="x", blockage_radius_m=1)
    assert cfg.unused_fields("L2") == sorted(["k1", "dl_mcs_max", "a3_hyst_db", "li_alpha", "gnb_nf_db",
                                              "tr38901_los", "radio_map_path", "blockage_radius_m"])


def test_unused_fields_read_when_switched_on():
    assert NRConfig(dl=True, k1=5, dl_mcs_max=10, gnb_tx_dbm=40.0).unused_fields("L2") == []
    assert multicell(3, a3_hyst_db=1.0, li_alpha=0.5, dl=True, dl_interference=False).unused_fields("L2") == []
    assert NRConfig(noise_model="thermal", gnb_nf_db=9.0).unused_fields("L2") == []
    assert NRConfig(noise_model="thermal", ue_nf_db=7.0).unused_fields("L2") == ["ue_nf_db"]   # DL noise only
    assert NRConfig(noise_model="thermal", ni_fixed_dbm=-95.0).unused_fields("L2") == ["ni_fixed_dbm"]
    assert NRConfig(channel="tr38901", tr38901_los="los").unused_fields("L2") == []
    assert NRConfig(channel="radio_map", radio_map_path="synthetic").unused_fields("L2") == []
    assert NRConfig(blockage=True, blockage_radius_m=1.0).unused_fields("L2") == []
    assert NRConfig(fading=False, ue_speed_mps=3.0).unused_fields("L2") == ["fading_rho_per_ms", "ue_speed_mps"]
    assert NRConfig(doppler_min_speed_mps=0.5).unused_fields("L2") == ["doppler_min_speed_mps"]
    assert NRConfig(fading_doppler="per_robot", doppler_min_speed_mps=0.5).unused_fields("L2") == []
    assert NRConfig(lena_ref_sc_per_rb=2).unused_fields("L2") == ["lena_ref_sc_per_rb"]
    assert NRConfig(tbs_mode="lena", lena_ref_sc_per_rb=2).unused_fields("L2") == []
    assert NRConfig(ul_pc_alpha=0.8).unused_fields("L2") == ["ul_pc_alpha"]
    assert NRConfig(ul_pc=True, ul_pc_alpha=0.8).unused_fields("L2") == []


def test_unused_fields_netslot_mc_frame():
    """NetSlotMC reads the frame only to convert the handover times to UL slots (several cells)."""
    cfg = multicell(2, bandwidth_mhz=40, rbg_config=2)
    un = cfg.unused_fields("L2-legacy")
    assert "bandwidth_mhz" in un and "rbg_config" in un and "n_prb" in un
    assert "a3_hyst_db" not in multicell(2, a3_hyst_db=1.0).unused_fields("L2-legacy")
    one = NRConfig(noise_model="thermal", a3_hyst_db=1.0, tdd_pattern="DDSUU")     # one non-legacy cell
    assert {"a3_hyst_db", "tdd_pattern"} <= set(one.unused_fields("L2-legacy"))


def test_presets_read_everything_they_set_on_l2():
    for f in (netslot_compat, lena_like, lena_validation, srsran_like, oai_like, lena_match_v2, lena_validation_v2):
        assert f().unused_fields("L2") == [], f.__name__
    assert multicell(3).unused_fields("L2") == []


# ---------------------------------------------------------------- 21: presets and with_
def test_presets_follow_mu_and_pattern():
    s = srsran_like()
    assert (s.sr_period_slots, s.sr_grant_delay_slots, s.ul_harq_rtt_slots) == (40, 5, 20)      # unchanged default
    o = oai_like()
    assert (o.sr_period_slots, o.sr_grant_delay_slots, o.ul_harq_rtt_slots) == (40, 50, 20)
    assert srsran_like(mu=0).sr_period_slots == 20 and oai_like(mu=2).sr_period_slots == 80
    d = srsran_like(tdd_pattern="DSUUU")                      # 3 U slots per 5: 1 UL slot -> 2 slots, 4 -> 7
    assert (d.sr_grant_delay_slots, d.ul_harq_rtt_slots) == (2, 7)
    assert srsran_like(sr_period_slots=7, ul_harq_rtt_slots=3).sr_period_slots == 7
    assert srsran_like(sr_period_slots=7, ul_harq_rtt_slots=3).ul_harq_rtt_slots == 3
    with pytest.raises(ValueError, match="no U slot"):
        oai_like(tdd_pattern="DDDS")


def test_with_keeps_explicit_fading_rho():
    c = NRConfig(ue_speed_mps=3.0)
    assert c.with_(fading_rho_per_ms=0.5).fading_rho_per_ms == 0.5
    assert c.with_(fading_rho_per_ms=0.5).ue_speed_mps is None
    assert c.with_(carrier_ghz=28.0).fading_rho_per_ms < c.fading_rho_per_ms      # speed still drives it
    assert c.with_(ue_speed_mps=1.0, fading_rho_per_ms=0.5).fading_rho_per_ms == fading_rho_from_speed(1.0)


def test_top_level_levels_list_wifi():
    assert "WIFI" in isaac_net.LEVELS and set(isaac_net.core.LEVELS) < set(isaac_net.LEVELS)


# ---------------------------------------------------------------- 17: make_adaptive wrappers
def _drive(net, steps, seed=0):
    g = torch.Generator().manual_seed(seed)
    outs = []
    for _ in range(steps):
        send = (torch.rand(E, R, generator=g) < 0.6).long() * 2
        snr = 5.0 + 20.0 * torch.rand(E, R, generator=g)
        net.submit(None, Requests(send))
        outs.append(net.step(None, snr))
    return outs


def test_make_adaptive_energy_drains_battery():
    fid = FidelityConfig(cheap="L1", expensive="L2-legacy", mode="load", up_threshold=4000.0)
    cfg = NRConfig(seed=5, energy=EnergyConfig(battery_j=10.0, idle_power_w=1.0), fidelity=fid)
    net = make_adaptive(E, R, "cpu", cfg)
    assert isinstance(net, EnergyLoop) and isinstance(net.engine, AdaptiveEngine)
    outs = _drive(net, 6)
    assert all("energy_j" in o and "fidelity" in o for o in outs)
    b = torch.stack([o["battery_j"] for o in outs])
    assert bool((b[1:] < b[:-1]).all())                       # idle power alone drains every step
    assert bool((outs[-1]["energy_cum_j"] > 0.6 - 1e-6).all())
    assert net.engine.exp.config.energy is None and net.engine.cheap.config.energy is None   # levels unwrapped


def test_make_adaptive_edge_and_energy_order():
    fid = FidelityConfig(cheap="L1", expensive="L2-legacy")
    net = make_adaptive(E, R, "cpu", NRConfig(seed=5, edge=EdgeConfig(), energy=EnergyConfig()), fid)
    assert isinstance(net, EnergyLoop) and isinstance(net.engine, EdgeLoop)
    assert isinstance(net.engine.engine, AdaptiveEngine)
    o = _drive(net, 2)[-1]
    assert "act_cap" in o and "battery_j" in o


def test_make_adaptive_refuses_background():
    fid = FidelityConfig(cheap="L1", expensive="L2-legacy")
    with pytest.raises(ValueError, match="background"):
        make_adaptive(E, R, "cpu", NRConfig(background=BackgroundConfig(n_background=2)), fid)
    make_adaptive(E, R, "cpu", NRConfig(background=BackgroundConfig(n_background=0)), fid)    # nothing to apply


# ---------------------------------------------------------------- 19: energy seed, legacy forms
def test_energy_config_seed_wins():
    def soc(engine_seed, energy_seed):
        en = EnergyConfig(initial_soc=(0.2, 0.9), seed=energy_seed)
        net = make_engine("L1", E, R, "cpu", NRConfig(energy=en), seed=engine_seed)
        return net.state["battery"].clone()
    assert torch.equal(soc(1, 7), soc(2, 7))
    assert not torch.equal(soc(1, 7), soc(1, 8))
    assert not torch.equal(soc(1, None), soc(2, None))        # None: derived from the engine seed


def test_energy_refuses_legacy_step():
    net = make_engine("L1", E, R, "cpu", NRConfig(energy=EnergyConfig()), seed=1)
    net.submit(None, Requests(torch.ones(E, R, dtype=torch.long)))
    with pytest.raises(NotImplementedError, match="dict form"):
        net.step(None, torch.full((E, R), 10.0), torch.zeros(E, dtype=torch.long))


@pytest.mark.parametrize("level", ["L1", "L2-legacy"])
def test_background_load_legacy_matches_dict(level):
    cfg = NRConfig(background=BackgroundConfig(n_background=3, mobility="random_waypoint", speed_mps=5.0))
    a = make_engine(level, E, R, "cpu", cfg, seed=3)
    b = make_engine(level, E, R, "cpu", cfg, seed=3)
    g = torch.Generator().manual_seed(1)
    for k in range(25):
        send = (torch.rand(E, R, generator=g) < 0.7).long() * (1 + (torch.rand(E, R, generator=g) < 0.5).long())
        det = torch.rand(E, R, generator=g) < 0.3
        hid = torch.randint(0, 3, (E,), generator=g)
        snr = 0.0 + 20.0 * torch.rand(E, R, generator=g)
        for n in (a, b):
            n.submit(None, Requests(send, det, hid))
        if k == 12:
            a.reset([1])
            b.reset([1])
        oa = a.step(None, snr)
        newest, det_env = b.step(None, snr, hid)
        assert torch.equal(oa["newest"], newest) and torch.equal(oa["det_env"], det_env)
        assert torch.equal(a.kappa_f, b.kappa_f) and torch.equal(a.rho, b.rho)
        assert torch.equal(a.engine.rem, b.engine.rem)


# ---------------------------------------------------------------- 20: edge
def test_edge_nr_dl_refuses_fast_backends():
    eng = make_engine("L2", E, R, "cpu", NRConfig(dl=True), seed=1)
    EdgeLoop(eng, EdgeConfig(return_path="nr_dl"))                  # reference: accepted
    for backend in ("graph", "triton"):
        eng = make_engine("L2", E, R, "cpu", NRConfig(dl=True), seed=1)
        eng.backend = backend                                       # NRGraphEngine / NRTritonEngine need CUDA
        with pytest.raises(ValueError, match="reference backend"):
            EdgeLoop(eng, EdgeConfig(return_path="nr_dl"))


def test_edge_act_age_nan_before_first_action():
    net = make_engine("L1", E, R, "cpu", NRConfig(edge=EdgeConfig(service_ms=10.0)), seed=1)
    o = net.step(None, torch.full((E, R), 20.0))
    assert bool(torch.isnan(o["act_age"]).all()) and bool((o["act_cap"] < 0).all())
    send = torch.zeros(E, R, dtype=torch.long)
    send[:, 0] = 1
    net.submit(None, Requests(send))
    for _ in range(3):
        o = net.step(None, torch.full((E, R), 20.0))
    got = o["act_cap"] >= 0
    assert bool(got[:, 0].all()) and not bool(got[:, 1:].any())
    assert bool(torch.isfinite(o["act_age"][got]).all()) and bool(torch.isnan(o["act_age"][~got]).all())


# ---------------------------------------------------------------- 20: adaptive L0 with an empirical marginal
def test_adaptive_l0_empirical_handback():
    q = torch.tensor([0.3, 0.5, 0.8, 1.5, 4.0, 9.0, 30.0])       # delays in control steps (sorted sample)
    fid = FidelityConfig(cheap="L0", expensive="L1", cheap_params={"q": q, "p": 0.0}, mode="static", fraction=0.0)
    net = make_adaptive(E, R, "cpu", NRConfig(seed=4, timeout_steps=60), fid)
    send = torch.full((E, R), 2, dtype=torch.long)
    snr = torch.full((E, R), -5.0)                                # slow L1: messages stay queued
    net.set_fraction(1.0)
    for _ in range(3):
        net.submit(None, Requests(send))
        net.step(None, snr)
    queued = net.queued().clone()
    assert int(queued.sum()) > 0
    net.set_fraction(0.0)                                         # hand the queues back to L0: conditional draw
    lc = net.lc
    dlv, cap = net.cheap.dlv, net.cheap.cap
    valid = cap >= 0
    assert torch.equal(valid.sum(-1), queued)
    waited = net.cheap.clock.double()[:, None, None] - cap.double()
    d = dlv.double() - cap.double()
    # every finite delivery time is cap + an entry of q above the time already waited (or now, if it is due)
    fin = valid & torch.isfinite(dlv)
    ok = torch.isclose(d[..., None], q.double()).any(-1) | torch.isclose(dlv.double(), net.cheap.clock.double()[:, None, None])
    assert bool(ok[fin].all())
    assert bool((d[fin] >= waited[fin] - 1e-6).all())
    assert lc.lvl()["q"].numel() == q.numel()
    for _ in range(40):                                           # the handed-back frames drain on L0
        net.step(None, snr)
    assert int(net.queued().sum()) == 0
    assert not math.isnan(float(net.cheap.clock[0]))
