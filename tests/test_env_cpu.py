"""Example task smoke test on CPU with the reference L2 engine."""
import torch

from engine_api import make_ref


def test_fleet_env_runs_on_cpu(seeded):
    from env import OBS_DIM, TASK_SIZES, FleetEnv
    E, R = 3, 4
    net = make_ref("L2", E, R, "cpu", sizes=TASK_SIZES["T1"])
    env = FleetEnv(E, R, net, torch.device("cpu"))
    env.T = 12                                          # exercise the episode end and auto-reset
    obs = env.reset()
    assert obs.shape == (E, R, OBS_DIM)
    infos = []
    for _ in range(15):
        vel = torch.rand(E, R, 2) * 2 - 1
        send = torch.randint(0, 3, (E, R))
        obs, rew, done, info = env.step(vel, send)
        assert obs.shape == (E, R, OBS_DIM) and rew.shape == (E, R)
        assert torch.isfinite(obs).all() and torch.isfinite(rew).all()
        if info is not None:
            infos.append(info)
    assert len(infos) == 1 and env.t == 3
