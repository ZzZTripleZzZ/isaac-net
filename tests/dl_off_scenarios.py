"""Scenarios of tests/test_dl_traffic_fdd_rem.py whose outputs must stay bitwise those of the engine before the DL
traffic / DL background / FDD fields existed (golden digests in fixtures/dl_off_golden.json, written by
`python tests/dl_off_scenarios.py` on the commit before those features). Only the pre-feature API is used here."""
import hashlib
import json
import os
import platform
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
GOLDEN = os.path.join(HERE, "fixtures", "dl_off_golden.json")
E, R, STEPS = 3, 4, 10


def _digest(outs):
    h = hashlib.sha256()
    for o in outs:
        if isinstance(o, tuple):
            o = dict(enumerate(o))
        for k in sorted(o, key=str):
            v = o[k]
            if torch.is_tensor(v):
                h.update(str(k).encode())
                h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _drive(net, pos=True, dl=False, triggers=False, seed=5):
    from isaac_net.core import Requests
    g = torch.Generator().manual_seed(seed)
    outs = []
    for k in range(STEPS):
        send = torch.where(torch.rand(E, R, generator=g) < 0.4, torch.randint(1, 3, (E, R), generator=g),
                           torch.zeros(E, R, dtype=torch.long))
        x = torch.rand(E, R, 2, generator=g) * 120 if pos else 25 * torch.rand(E, R, generator=g) - 5
        net.submit(None, Requests(send))
        if dl:
            nb = torch.where(torch.rand(E, R, generator=g) < 0.5, torch.full((E, R), 3000.0), torch.zeros(E, R))
            net.add_dl_frames(None, nb, torch.full((E, R), 1, dtype=torch.long))
        kw = {"triggers": {"alarm": torch.rand(E, R, generator=g) < 0.3}} if triggers else {}
        outs.append(net.step(None, x, **kw))
        if k == 6:
            net.reset([1])
    return outs


def scenarios():
    from isaac_net.core import NRConfig, TrafficModel as TM, make_engine, multicell
    from isaac_net.core.background import BackgroundConfig
    ul_models = [TM.periodic(700, 20, jitter_ms=4).on([0, 1]), TM.video(15, 3000, gop=(6000, 2000, 4)).on(2),
                 TM.bursty(1400, 30, 3, (0.4, 0.4)).on(3), TM.event(900, trigger="alarm")]
    bg = BackgroundConfig(n_background=3, traffic=(TM.video(fps=30, mean_frame_bytes=2500),),
                          mobility="random_waypoint", speed_mps=1.5)
    return {
        "l2_ul_traffic_snr": lambda: _drive(make_engine("L2", E, R, "cpu", NRConfig(traffic=ul_models, frame_buffer=32, control_step_ms=20.0),
                                                        seed=2), pos=False, triggers=True),
        "l2_ul_traffic_dl_pose": lambda: _drive(make_engine("L2", E, R, "cpu", NRConfig(
            traffic=ul_models, dl=True, channel="tr38901_inf_sh", control_step_ms=20.0), seed=4), dl=True, triggers=True),
        "l2_plain_dl": lambda: _drive(make_engine("L2", E, R, "cpu", NRConfig(dl=True, control_step_ms=20.0), seed=6), dl=True),
        "l2_ghost_bg_dl": lambda: _drive(make_engine("L2", E, R, "cpu", NRConfig(
            background=bg, dl=True, traffic=ul_models[:1], control_step_ms=20.0), seed=7), dl=True),
        "l2_ghost_bg_multicell": lambda: _drive(make_engine("L2", E, R, "cpu", multicell(
            3, background=bg, dl=True, control_step_ms=20.0), seed=8), dl=True),
        "l1_load_bg": lambda: _drive(make_engine("L1", E, R, "cpu", NRConfig(background=bg, control_step_ms=20.0), seed=9)),
        "l2legacy_load_bg": lambda: _drive(make_engine("L2-legacy", E, R, "cpu", NRConfig(background=bg, control_step_ms=20.0), seed=10)),
        "l2_multicell_traffic_dl": lambda: _drive(make_engine("L2", E, R, "cpu", multicell(
            3, traffic=ul_models, dl=True, control_step_ms=20.0), seed=11), dl=True, triggers=True),
    }


def env_key():
    return {"torch": torch.__version__, "machine": platform.machine(), "system": platform.system()}


def compute():
    torch.set_num_threads(1)
    out = {}
    for name, f in scenarios().items():
        torch.manual_seed(0)
        out[name] = _digest(f())
    return out


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(HERE))
    res = {"env": env_key(), "digests": compute()}
    with open(GOLDEN, "w") as fh:
        json.dump(res, fh, indent=1, sort_keys=True)
    print(json.dumps(res, indent=1))
