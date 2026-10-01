"""L05 / L05Q lookup tables: the bins used when fitting (from logged per-frame features) match the
bins the engine queries at frame arrival.

Each bin gets a unique constant delay (0.05 + 0.1 * flat bin index) and odd bins drop every frame.
A delivered frame's delay then encodes the bin the engine looked up; it must equal the bin that a
fit computes from the frame's logged features with the same `lookup_key`.
"""
import pytest
import torch

from isaac_net.core.proto import netsim as ns
from engine_api import Workload, collect, enable_stats, lookup_params, lookup_shape, make_ref, run


def _fit_bins(mode, st, prefix):
    key = ns.lookup_key(mode, st[prefix + "f_nact"], st[prefix + "f_snr"], st[prefix + "f_own"], st[prefix + "cls"])
    shape = lookup_shape(mode)
    flat = torch.zeros_like(key[0])
    for k, n in zip(key, shape):
        assert ((k >= 0) & (k < n)).all(), "bin index out of table range"
        flat = flat * n + k
    return flat


@pytest.mark.parametrize("mode", ["L05", "L05Q"])
def test_fit_and_query_bucketing_match(mode, seeded):
    shape = lookup_shape(mode)
    nbins = int(torch.tensor(shape).prod())
    code = 0.05 + 0.1 * torch.arange(nbins, dtype=torch.float32)
    assert float(code.max()) < ns.TIMEOUT - 1
    q = code.view(*shape, 1).expand(*shape, 101).clone()
    pdrop = (torch.arange(nbins) % 2 == 1).float().view(shape)
    E, R = 4, 12
    net = make_ref(mode, E, R, "cpu", params=lookup_params(mode, quantiles=q, pdrop=pdrop))
    enable_stats(net)
    wl = Workload(E, R, "cpu", seed=21, period=8, snr_lo=-15.0, snr_hi=40.0)
    run(net, wl, steps=80)
    st = collect(net)
    d_bins = _fit_bins(mode, st, "d_")
    x_bins = _fit_bins(mode, st, "x_")
    queried = ((st["delay"] - 0.05) / 0.1).round().long()
    assert torch.equal(queried, d_bins), "query bin differs from fit bin"
    assert (d_bins % 2 == 0).all(), "a frame from a drop-all bin was delivered"
    assert (x_bins % 2 == 1).all(), "a frame from a keep-all bin timed out"
    assert len(set(d_bins.tolist()) | set(x_bins.tolist())) >= (12 if mode == "L05" else 20), "too few bins exercised"


def test_lookup_key_edges():
    """Bins follow torch.bucketize(right=False) on the documented edges and cover every table row."""
    nact = torch.tensor([0, 2, 3, 5, 9, 10])
    snr = torch.tensor([-5.0, 0.0, 5.0, 10.0, 30.0, 31.0])
    own = torch.tensor([0, 1, 2, 3, 4, 15])
    cls = torch.tensor([1, 2, 1, 2, 1, 2])
    nb, sb, ob, c = ns.lookup_key("L05Q", nact, snr, own, cls)
    assert torch.equal(nb, torch.bucketize(nact, torch.tensor(ns.NACT_EDGES)))
    assert torch.equal(sb, torch.bucketize(snr, torch.tensor(ns.SNR_EDGES)))
    assert torch.equal(ob, torch.bucketize(own, torch.tensor(ns.OWNQ_EDGES)))
    assert torch.equal(c, cls - 1)
    assert nb.max() == len(ns.NACT_EDGES) and sb.max() == len(ns.SNR_EDGES) and ob.max() == len(ns.OWNQ_EDGES)
