"""Smoke test of the backend-agnostic NetModule skeleton (prototype/isaac/netmodule.py) on CPU:
every rung steps, and a partial reset re-initializes exactly the chosen envs."""
import pytest
import torch


def _mod():
    from isaac import netmodule
    return netmodule


@pytest.mark.parametrize("rung", ["L0", "L1", "L2"])
def test_netmodule_steps_and_partial_reset(rung, seeded):
    nm = _mod()
    E, R = 4, 3
    cfg = nm.NetConfig(num_envs=E, num_robots=R, device="cpu", rung=rung, pose_chunks=2)
    net = nm.NetModule(cfg)
    net.reset()
    delivered = 0
    for _ in range(12):
        pos = torch.rand(E, R, 3) * 60
        out = net.step(pos, nm.TrafficRequest(send=torch.randint(0, 3, (E, R))))
        assert out.newest_cap.shape == (E, R) and out.delay_s.shape == (E, R, cfg.frame_depth)
        assert torch.isfinite(out.sinr_db).all() and (out.queue_len <= cfg.frame_depth).all()
        delivered += int(out.delivered.sum())
    assert delivered > 0
    before = net.state_dict()
    net.reset([1])
    after = net.state_dict()
    keep = torch.tensor([0, 2, 3])
    for k in before:
        assert torch.equal(before[k][keep], after[k][keep]), f"reset([1]) touched other envs in {k}"
    assert int(net.t[1]) == 0 and (net.cap[1] == -1).all() and (net.rem[1] == 0).all()
    assert (net.t[keep] == 12).all()
