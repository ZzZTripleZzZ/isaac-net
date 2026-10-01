"""NR engine with netslot_compat() vs the legacy NetSlot on the same workload distribution (statistical match;
merged from nrconfig/tests/test_compat.py). The PHYs differ by construction (38.214 MCS/TBS + Sionna BLER vs a
Shannon-gap SE and logistic BLER), so the check is approximate agreement at light load, where the nrconfig
report found loss within about 5 points and goodput within about 10%."""
import pytest
import torch

from isaac_net.core import make_engine, netslot_compat
from isaac_net.core.proto.netsim import Radio

E, R, T, WARM = 16, 16, 80, 20


def _kpis(net, snr, p, seed):
    g = torch.Generator().manual_seed(1000 + seed)
    net.log_stats = True
    net.log_cap_max = T - 25
    z, zb = torch.zeros(E, dtype=torch.long), torch.zeros(E, R, dtype=torch.bool)
    for t in range(T):
        send = (torch.rand(E, R, generator=g) < p).long()
        net.add_frames(t, send, zb, z, snr)
        net.step(t, snr, z)
    st = net.collect()
    n_del = st["delay"].numel()
    n_lost = st["x_cls"].numel() + st.get("discarded", 0)
    return n_lost / max(n_del + n_lost, 1), n_del * 4000 / 1000 / (T - 25) / E


@pytest.mark.slow
def test_netslot_compat_close_to_netslot_at_light_load():
    torch.manual_seed(0)
    radio = Radio(E, "cpu")
    snr = radio.snr_db(torch.rand(E, R, 2) * 150)
    torch.manual_seed(1)
    loss_ref, gp_ref = _kpis(make_engine("L2-legacy", E, R, "cpu", seed=2), snr, 0.5, 0)
    torch.manual_seed(1)
    loss_nr, gp_nr = _kpis(make_engine("L2", E, R, "cpu", netslot_compat(), seed=2), snr, 0.5, 0)
    assert abs(loss_nr - loss_ref) < 0.1, (loss_nr, loss_ref)
    assert 0.75 < gp_nr / gp_ref < 1.4, (gp_nr, gp_ref)
