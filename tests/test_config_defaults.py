"""The configurability fields keep today's behavior at their defaults, and change it when set.

Plumbed fields: L0 delay/loss (l0_*), L0DR ranges (dr_*), the L1 goodput factor (l1_eta), fading from a UE speed
(ue_speed_mps, carrier_ghz), and the ignored-field check (NRConfig.unused_fields, make_engine(strict=True)).
The bitwise tests compare against inline copies of the constants and formulas the code had before.
"""
import math
from dataclasses import fields

import pytest
import torch

from isaaclab_net import NRConfig, make_engine
from isaaclab_net.core import multicell
from isaaclab_net.core.config import FIELD_GROUPS, fading_rho_from_speed
from isaaclab_net.core.proto import netsim as ns

E, R = 4, 6


def _old_l0dr_draw(n, gen):
    kw = dict(generator=gen)
    mu = math.log(0.05) + (math.log(10.0) - math.log(0.05)) * torch.rand(n, **kw)
    sig = 0.2 + 1.0 * torch.rand(n, **kw)
    p = 0.2 * torch.rand(n, **kw)
    return mu, sig, p


def test_l0dr_default_ranges_are_bitwise_the_old_draws():
    g1, g2 = torch.Generator().manual_seed(3), torch.Generator().manual_seed(3)
    new = ns.l0dr_draw(1000, ns.l0dr_ranges(None), dict(generator=g1))
    old = _old_l0dr_draw(1000, g2)
    assert all(torch.equal(a, b) for a, b in zip(new, old))


def test_l0dr_engine_defaults_match_and_ranges_apply():
    a = make_engine("L0DR", E, R, "cpu", seed=5)
    g = torch.Generator().manual_seed(5)
    mu, sig, p = _old_l0dr_draw(E, g)
    assert torch.equal(a.mu, mu) and torch.equal(a.sig, sig) and torch.equal(a.p, p)
    b = make_engine("L0DR", 256, R, "cpu", NRConfig(dr_delay_median_steps=(1.0, 2.0), dr_loss=(0.5, 0.6)), seed=5)
    assert (b.mu.exp() >= 1.0 - 1e-5).all() and (b.mu.exp() <= 2.0 + 1e-5).all()
    assert (b.p >= 0.5).all() and (b.p <= 0.6).all()


class _OldFluid(ns.NetFluid):
    """NetFluid._transmit as it was, with the literal 0.9."""

    def _transmit(self, t, snr_db):
        fin_t = torch.full_like(self.rem, float("inf"))
        for k in range(ns.UL_PER_STEP):
            q = self.rem.sum(-1)
            back = q > 0
            nb = back.sum(-1, keepdim=True).clamp(min=1).float()
            share = ns.S / nb
            split = share.clamp(min=1.0)
            snr_sb = snr_db - 10 * torch.log10(split)
            se = (0.75 * torch.log2(1 + 10 ** (snr_sb / 10))).clamp(max=ns.SE_MAX) * 0.9
            b = share * se * ns.BYTES_PER_SE * back
            self.rem, fin = ns.serve_fifo(self.rem, b)
            fin_t = torch.where(fin, self._finvals(t, k), fin_t)
        return fin_t


def _run(net, steps=30, seed=11):
    g = torch.Generator().manual_seed(seed)
    outs = []
    for _ in range(steps):
        send = torch.randint(0, 3, (E, R), generator=g)
        snr = 25 * torch.rand(E, R, generator=g) - 5
        net.submit(None, send)
        o = net.step(None, snr)
        outs.append((o["delivered"], o["delay"].nan_to_num(-1.0), o["queue_bytes"]))
    return outs


def _same(a, b):
    return all(torch.equal(x, y) for oa, ob in zip(a, b) for x, y in zip(oa, ob))


def test_l1_default_eta_is_bitwise_the_old_engine():
    new = make_engine("L1", E, R, "cpu", seed=1)
    assert new.eta == 0.9
    assert _same(_run(new), _run(_OldFluid(E, R, "cpu", NRConfig().msg_sizes, seed=1)))
    slow = make_engine("L1", E, R, "cpu", NRConfig(l1_eta=0.5), seed=1)
    assert not _same(_run(slow), _run(make_engine("L1", E, R, "cpu", seed=1)))


def test_l0_without_params_uses_the_config():
    torch.manual_seed(0)
    a = _run(make_engine("L0", E, R, "cpu", seed=2))
    torch.manual_seed(0)
    b = _run(make_engine("L0", E, R, "cpu", params={"mu": math.log(0.05), "sig": 0.5, "p": 0.0}, seed=2))
    assert _same(a, b)
    torch.manual_seed(0)
    c = _run(make_engine("L0", E, R, "cpu", NRConfig(l0_loss=1.0), seed=2))
    assert not any(bool(o[0].any()) for o in c)


def test_fading_from_speed():
    assert NRConfig().fading_rho_per_ms == 0.93 ** (1 / 2.5)
    rho3 = NRConfig(ue_speed_mps=3.0).fading_rho_per_ms
    assert abs(rho3 ** 2.5 - 0.93) < 0.01
    assert fading_rho_from_speed(10.0) < rho3 < fading_rho_from_speed(1.0) < 1.0
    assert NRConfig(ue_speed_mps=3.0, carrier_ghz=28.0).fading_rho_per_ms < rho3


def test_every_field_is_classified():
    grouped = [f for g in FIELD_GROUPS.values() for f in g]
    assert len(grouped) == len(set(grouped))
    assert set(grouped) == {f.name for f in fields(NRConfig)}


def test_unused_fields_and_strict():
    for level in ("L0", "L0DR", "L05", "L1", "L2", "L2-legacy", "QA", "ORACLE"):
        assert NRConfig().unused_fields(level) == []
    assert NRConfig(olla_up_db=0.1).unused_fields("L2") == []
    assert NRConfig(olla_up_db=0.1).unused_fields("L2-legacy") == ["olla_up_db"]
    assert NRConfig(pathloss_exp=3.0).unused_fields("L0") == ["pathloss_exp"]
    assert multicell(3).unused_fields("L2") == [] and multicell(3).unused_fields("L2-legacy") == []
    assert multicell(3, dl_interference=False).unused_fields("L2-legacy") == ["dl_interference"]   # NR engine only
    with pytest.raises(ValueError, match="olla_up_db"):
        make_engine("L2-legacy", E, R, "cpu", NRConfig(olla_up_db=0.1), strict=True)
    make_engine("L2-legacy", E, R, "cpu", NRConfig(olla_up_db=0.1))          # not strict: ignored silently
