"""Bitwise regression of every level and backend against another checkout (e.g. main before the configurable
application constants and the engine-owned RNG).

Run the same command in both trees, then compare:
    python tests/scripts/regress_main.py --out /tmp/new.pt [--device cuda]      # in this tree
    python tests/scripts/regress_main.py --out /tmp/old.pt [--device cuda]      # in the old tree
    python tests/scripts/regress_main.py --compare /tmp/old.pt /tmp/new.pt

Defaults are compared: NRConfig() with msg_sizes, plus rng="global" where the tree has the field (the earlier
stepping randomness), with a mid-run partial reset. Every output tensor of every step must be equal.
"""
import argparse
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))

import torch  # noqa: E402

SIZES = (4000.0, 30000.0)


def cases(device):
    has_triton = True
    try:
        import triton  # noqa: F401
    except Exception:
        has_triton = False
    out = []
    gpu = device == "cuda"
    for lvl in ("L0", "L0DR", "L05", "L05Q", "L1", "L2-legacy"):
        out.append((lvl, "reference"))
        out.append((lvl, "eager"))
        if gpu:
            out.append((lvl, "graph"))
            if lvl in ("L0DR", "L1", "L2-legacy"):
                out.append((lvl, "compile"))
            if lvl in ("L1", "L2-legacy") and has_triton:
                out.append((lvl, "triton"))
    for lvl in ("TR", "GE", "QA", "NN", "ORACLE", "NOCOMM"):
        out.append((lvl, "reference"))
        if gpu:
            out.append((lvl, "graph"))
    out.append(("MC", "reference"))
    out.append(("L2", "reference"))
    return out


def params_of(level):
    from engine_api import lookup_params
    from test_levels import synth_params
    if level == "L0":
        return {"mu": math.log(0.5), "sig": 0.5, "p": 0.05}
    if level in ("L05", "L05Q"):
        return lookup_params(level)
    if level in ("TR", "GE", "QA", "NN"):
        return synth_params(level)
    return None


def run_case(level, backend, device, E, R, steps):
    from isaac_net.core import NRConfig, Requests, make_engine, multicell
    from dataclasses import fields
    has_rng = "rng" in {f.name for f in fields(NRConfig)}
    kw = {"rng": "global"} if has_rng else {}
    if level == "MC":
        lvl, cfg = "L2-legacy", multicell(3, msg_sizes=SIZES, **kw)
    else:
        lvl, cfg = level, NRConfig(msg_sizes=SIZES, **kw)
    dev = torch.device(device)
    torch.manual_seed(123)
    if dev.type == "cuda":
        torch.cuda.manual_seed(123)
    net = make_engine(lvl, E, R, dev, cfg, backend, params=params_of(level), seed=17)
    g = torch.Generator().manual_seed(5)
    outs = []
    for t in range(steps):
        if t == steps // 2:
            net.reset(torch.tensor([1, E - 1], device=dev))
        send = ((torch.rand(E, R, generator=g) < 0.7).long() * torch.randint(1, 3, (E, R), generator=g)).to(dev)
        snr = (-5 + 25 * torch.rand(E, R, generator=g)).to(dev)
        if level == "MC":                   # the multi-cell engine takes poses
            pos = (150 * torch.rand(E, R, 2, generator=g)).to(dev)
            net.submit(None, Requests(send))
            o = net.step(None, pos)
        else:
            net.submit(None, Requests(send), snr)
            o = net.step(None, snr)
        outs.append({k: v.detach().cpu().clone() for k, v in o.items() if torch.is_tensor(v)})
    return outs


def compare(a_path, b_path):
    a, b = torch.load(a_path), torch.load(b_path)
    bad = 0
    for key in sorted(set(a) | set(b)):
        if key not in a or key not in b:
            print(f"{key}: only in {'new' if key in b else 'old'}")
            continue
        x, y = a[key], b[key]
        diff = None
        for t, (ox, oy) in enumerate(zip(x, y)):
            for k in ox:
                if k not in oy:
                    continue
                u, v = ox[k], oy[k]
                if u.is_floating_point():
                    u, v = u.nan_to_num(-7.0), v.nan_to_num(-7.0)
                if not torch.equal(u, v):
                    diff = (t, k)
                    break
            if diff:
                break
        status = "OK   " if diff is None and len(x) == len(y) else f"DIFF {diff}"
        bad += status != "OK   "
        print(f"{status} {key}")
    print("all bitwise equal" if not bad else f"{bad} cases differ")
    return bad


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="")
    p.add_argument("--compare", nargs=2, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--E", type=int, default=16)
    p.add_argument("--R", type=int, default=8)
    p.add_argument("--steps", type=int, default=50)
    a = p.parse_args()
    if a.compare:
        sys.exit(1 if compare(*a.compare) else 0)
    res = {}
    for level, backend in cases(a.device):
        steps = 20 if level == "L2" else a.steps
        res[f"{level}/{backend}"] = run_case(level, backend, a.device, a.E, a.R, steps)
        print("ran", level, backend, flush=True)
    torch.save(res, a.out)


if __name__ == "__main__":
    main()
