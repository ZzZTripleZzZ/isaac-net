"""Scenarios of tests/test_limits_closed.py whose outputs must stay bitwise those of the engine before the
known-limits follow-ups (DL traffic on graph, EdgeLoop delay path under FDD, RLF re-establishment through RACH,
sector-baked radio maps, one place for the triton refusals). Golden digests in fixtures/limits_off_golden.json,
written by `python tests/limits_off_scenarios.py` on the commit before those changes. Only configurations that do not
use the new paths are here: RLF without RACH (with and without the access stage), RACH without RLF, the EdgeLoop
"delay" return path with one carrier, DL traffic models on the reference backend and the run-time sector pattern on
a radio map."""
import hashlib
import json
import os
import platform
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
GOLDEN = os.path.join(HERE, "fixtures", "limits_off_golden.json")
E, R, STEPS = 3, 4, 12


def _digest(outs, extra=None):
    h = hashlib.sha256()
    for o in outs:
        for k in sorted(o, key=str):
            v = o[k]
            if torch.is_tensor(v):
                h.update(str(k).encode())
                h.update(v.detach().cpu().contiguous().numpy().tobytes())
    if extra is not None:
        h.update(json.dumps(extra, sort_keys=True).encode())
    return h.hexdigest()


def _drive(net, x="pose", dl=False, seed=5, area=150.0, reset_at=7):
    from isaac_net.core import Requests
    g = torch.Generator().manual_seed(seed)
    outs = []
    pos = torch.rand(E, R, 2, generator=g) * area
    for k in range(STEPS):
        send = torch.where(torch.rand(E, R, generator=g) < 0.5, torch.randint(1, 3, (E, R), generator=g),
                           torch.zeros(E, R, dtype=torch.long))
        pos = (pos + 6.0 * (2 * torch.rand(E, R, 2, generator=g) - 1)).clamp(0, area)
        net.submit(None, Requests(send))
        if dl:
            nb = torch.where(torch.rand(E, R, generator=g) < 0.5, torch.full((E, R), 3000.0), torch.zeros(E, R))
            net.add_dl_frames(None, nb)
        if x == "pose":
            o = net.step(None, pos)
        else:
            o = net.step(None, 25 * torch.rand(E, R, generator=g) - 5)
        outs.append(o)
        if k == reset_at:
            net.reset([1])
    return outs


def _fail(net, cfg, seed=3):
    """Robot 0 of each env loses its serving cell (SINR -15 dB) from step 3 on: RLF with re-establishment."""
    from isaac_net.core import Requests
    outs = []
    for t in range(STEPS):
        s0 = [20.0, 10.0, 0.0] if t < 3 else [-15.0, 5.0, 2.0]
        sinr = torch.tensor([s0] + [[20.0, 0.0, 0.0]] * (R - 1))
        pg = (sinr - cfg.ue_tx_dbm + cfg.subband_noise_dbm)[None].expand(E, R, 3).clone()
        net.submit(None, Requests(torch.ones(E, R, dtype=torch.long)))
        outs.append(net.step(None, pathgain_db=pg))
    return outs


def _ctr(net):
    return json.loads(json.dumps(net.counters(), default=float))


def scenarios():
    from isaac_net.core import EdgeConfig, EdgeLoop, NRConfig, TrafficModel as TM, make_engine, multicell
    quiet = dict(ul_interference=False, dl_interference=False, control_step_ms=20.0)

    def rlf(extra, fail=True):
        def run():
            kw = dict(rlf=True, a3_ttt_ms=1e6, t310_ms=100.0, t311_ms=300.0, reest_delay_ms=30.0, frame_buffer=32,
                      timeout_steps=100, **quiet)
            kw.update(extra)
            cfg = multicell(3, **kw)
            net = make_engine("L2", E, R, "cpu", cfg, seed=4)
            outs = _fail(net, cfg) if fail else _drive(net, dl=cfg.dl)
            return outs, _ctr(net)
        return run

    def plain(cfg, level="L2", seed=6, **kw):
        def run():
            net = make_engine(level, E, R, "cpu", cfg, seed=seed)
            outs = _drive(net, **kw)
            return outs, _ctr(net) if level == "L2" else None
        return run

    def edge(cfg, level="L2"):
        def run():
            net = EdgeLoop(make_engine(level, E, R, "cpu", cfg, seed=7), EdgeConfig(return_path="delay",
                                                                                    service_ms=5.0, cmd_bytes=4000))
            return _drive(net, x="snr"), None
        return run

    gnb2 = ((25.0, 75.0), (125.0, 75.0))
    return {
        "rlf_no_rach": rlf({}),
        "rlf_no_rach_dl_poses": rlf(dict(dl=True, a3_ttt_ms=40.0), fail=False),
        "rlf_drx_no_rach": rlf(dict(drx=True, drx_cycle_ms=40.0, drx_on_ms=8.0)),
        "rach_single_cell": plain(NRConfig(rach=True, rach_initial="idle", rach_release_after_ms=60.0, dl=True,
                                           control_step_ms=20.0), x="snr", dl=True),
        "rach_multicell_no_rlf": plain(multicell(3, rach=True, rach_initial="idle", control_step_ms=20.0)),
        "edge_delay_tdd_l2": edge(NRConfig(control_step_ms=20.0)),
        "edge_delay_l1": edge(NRConfig(control_step_ms=20.0), level="L1"),
        "edge_delay_fdd_same_carrier": edge(NRConfig(duplex="fdd", dl=True, control_step_ms=20.0)),
        "dl_traffic_reference": plain(NRConfig(dl=True, control_step_ms=20.0, frame_buffer=32,
                                               traffic=[TM.periodic(1200, 5).downlink().on([0, 1]),
                                                        TM.periodic(700, 10)]), x="snr"),
        "radio_map_sector_runtime": plain(NRConfig(channel="radio_map", radio_map_path="synthetic", n_cells=2,
                                                   cell_positions_m=gnb2, gnb_antenna="sector",
                                                   cell_azimuth_deg=(0.0, 180.0), control_step_ms=20.0)),
    }


def env_key():
    return {"torch": torch.__version__, "machine": platform.machine(), "system": platform.system()}


def compute():
    torch.set_num_threads(1)
    out = {}
    for name, f in scenarios().items():
        torch.manual_seed(0)
        outs, extra = f()
        out[name] = _digest(outs, extra)
    return out


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(HERE))
    res = {"env": env_key(), "digests": compute()}
    with open(GOLDEN, "w") as fh:
        json.dump(res, fh, indent=1, sort_keys=True)
    print(json.dumps(res, indent=1))
