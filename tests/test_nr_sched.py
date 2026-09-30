"""Scheduler variants of the NR engine (NRConfig.scheduler).

  S1 "pf" is the default and bitwise the engine before the option (the frozen-engine regression in
     test_nr_multicell.py M1 covers it); "pf_wideband" is bitwise pf_metric="wideband"
  S2 full buffer, robots spread over 30 dB of SNR, fixed transmit PSD (ul_power="whole_band", so the rate of an RBG
     does not depend on the grant size): max C/I carries the most bytes and starves the weakest robot, PF carries less
     and round robin the least; max C/I has the least fair byte split (Jain index). (With the UE power split over the
     grant, the default, PF can carry more than max C/I: max C/I gives the strongest robot every RBG and spreads its
     power thin.)
  S3 round robin rotates: with equal channels every backlogged robot is served, none twice before the others
"""
import torch

from isaaclab_net.core.config import NRConfig
from isaaclab_net.core.nr_engine import NRNet

E, R = 4, 6
SIZES = (4000.0, 30000.0)


def _run(cfg, steps=30, snr=None, seed=3):
    torch.manual_seed(seed)
    net = NRNet(E, R, "cpu", SIZES, cfg, seed=seed)
    net.ul.trace = []
    snr = torch.linspace(-3.0, 27.0, R).expand(E, R).clone() if snr is None else snr
    z, zb = torch.zeros(E, dtype=torch.long), torch.zeros(E, R, dtype=torch.bool)
    outs = []
    for t in range(steps):
        net.add_frames(t, torch.full((E, R), 2, dtype=torch.long), zb, z, snr)
        outs.append(net.step(t, snr, z, full=True))
    tbs = sum(tx.long() for _, tx, *_ in net.ul.trace)                                   # [E,R]
    byt = sum(((hi - lo) * ok).double() for _, _, ok, lo, hi, _ in net.ul.trace)
    return net, outs, tbs.sum(0).double(), byt.sum(0)


def jain(x):
    return float(x.sum() ** 2 / (len(x) * (x ** 2).sum()))


def test_s1_pf_wideband_is_pf_with_wideband_metric():
    a = _run(NRConfig(scheduler="pf_wideband"), steps=8)
    b = _run(NRConfig(pf_metric="wideband"), steps=8)
    for oa, ob in zip(a[1], b[1]):
        for k in oa:
            assert torch.equal(oa[k].nan_to_num(-7.0), ob[k].nan_to_num(-7.0)), k
    assert torch.equal(a[0].ul.olla, b[0].ul.olla) and torch.equal(a[0].ul.avg, b[0].ul.avg)


def test_s2_throughput_fairness_ordering():
    res = {s: _run(NRConfig(scheduler=s, ul_power="whole_band", phr_cap=False)) for s in ("pf", "maxci", "rr")}
    tot = {s: float(r[3].sum()) for s, r in res.items()}
    fair = {s: jain(r[3]) for s, r in res.items()}
    worst = {s: float(r[3].min()) for s, r in res.items()}
    assert tot["maxci"] >= tot["pf"] >= tot["rr"], tot
    assert fair["maxci"] < min(fair["pf"], fair["rr"]), fair
    assert worst["maxci"] < min(worst["pf"], worst["rr"]), worst          # max C/I starves the weakest robot
    assert tot["rr"] > 0.3 * tot["maxci"]


def test_s3_round_robin_rotates():
    snr = torch.full((E, R), 45.0)             # top MCS decodes with certainty: no retransmission reorders the round
    net, _, tbs, _ = _run(NRConfig(scheduler="rr", fading=False), steps=6, snr=snr)
    assert (tbs > 0).all()
    served = [tx for _, tx, *_ in net.ul.trace if tx.any()]
    # within the first round every robot of an env transmits before any robot transmits twice
    for e in range(E):
        seen = torch.zeros(R, dtype=torch.long)
        for tx in served:
            seen += tx[e].long()
            if int(seen.min()) == 0:
                assert int(seen.max()) <= 1
            else:
                break
        assert int(seen.min()) >= 1
