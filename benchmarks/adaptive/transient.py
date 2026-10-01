"""Cost of the handoff itself: forced switching against a fixed mix with the same share of env-steps per level.

usage: python benchmarks/adaptive/transient.py --E 2048 --R 16 --out transient.jsonl [--cheap L05Q --cheap_params f]

Runs on one workload (sweep.workload, uniform load p): pure L2-legacy (reference), pure L2-legacy with another seed
(noise floor), the pure cheap level, a static 50 % mix (a fixed random half of the envs on L2-legacy), and toggle-N:
the same half and its complement swap levels every N steps, so every env alternates N steps on each level. Static 50 %
and toggle-N put the same share of env-steps on each level; the difference between them is the effect of the
handoffs (the transient after entering L2-legacy with steady-state defaults, and the cheap level serving what
L2-legacy queued). Per run: W1 and KS of the per-frame delay distribution against the reference and against the
static mix, drop share, delay quantiles.

reentry-N isolates the entry rule of L2-legacy: all envs stay on L2-legacy (mask layout), and every N steps each env is
handed to the cheap level and straight back between two steps. The queue passes through exactly and the envs' random
streams stay those of the pure reference, so the only difference from pure L2-legacy is the MAC state that the entry
rule sets (BSR, no SR, no HARQ in flight, OLLA 0, PF average from the delivered-bytes EWMA, a fresh stationary fading
draw). Its W1 / KS against the reference, next to the seed-to-seed noise floor, measure the entry transient.
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _root)
from sweep import NB, dist, quantile, workload  # noqa: E402
from isaac_net.core import NRConfig, make_engine  # noqa: E402
from isaac_net.core.adaptive import FidelityConfig, make_adaptive  # noqa: E402
from isaac_net.core.proto.netsim import Requests  # noqa: E402


def run(net, wl, warm, toggle=None, reentry=None):
    dev = wl[0][0].device
    hist = torch.zeros(NB, dtype=torch.long, device=dev)
    timed = torch.zeros((), dtype=torch.long, device=dev)
    for k, (send, pos, rs) in enumerate(wl):
        if toggle is not None and k > 0 and k % toggle == 0:
            net.set_assignment(~net.static_mask)
        if reentry is not None and k > 0 and k % reentry == 0:
            m = net.static_mask.clone()
            net.set_assignment(~m)                 # to the cheap level ...
            net.set_assignment(m)                  # ... and straight back, before the next step
        if rs is not None:
            net.reset(rs)
        net.submit(None, Requests(send))
        o = net.step(None, pos)
        if k < warm:
            continue
        b = (o["delay"] * 40).round().nan_to_num(0).long().clamp(0, NB - 1)
        hist += torch.bincount(torch.where(o["delivered"], b, torch.full_like(b, NB)).flatten(), minlength=NB + 1)[:NB]
        timed += o["timed_out"].sum()
    return hist.cpu(), int(timed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--E", type=int, default=2048)
    ap.add_argument("--R", type=int, default=16)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--warm", type=int, default=50)
    ap.add_argument("--loads", default="0.1,0.3,0.5")
    ap.add_argument("--toggles", default="1,2,5,10,25,50")
    ap.add_argument("--reentries", default="1,5,25")
    ap.add_argument("--cheap", default="L1")
    ap.add_argument("--backend", default="triton")
    ap.add_argument("--cheap_params", default=None)
    ap.add_argument("--arena", default="5,60")
    ap.add_argument("--out", default="transient.jsonl")
    a = ap.parse_args()
    dev = torch.device("cuda")
    E, R = a.E, a.R
    cfg = NRConfig(seed=7)
    arena = tuple(float(v) for v in a.arena.split(","))
    cparams = torch.load(a.cheap_params, weights_only=True)[a.cheap] if a.cheap_params else None
    fh = open(a.out, "a")
    for p in a.loads.split(","):
        wl = workload(E, R, a.steps, f"uniform-{p}", dev, arena=arena)
        fid = lambda: FidelityConfig(cheap=a.cheap, expensive="L2-legacy", cheap_backend=a.backend,  # noqa: E731
                                     expensive_backend="triton", cheap_params=cparams, mode="static", fraction=0.5)
        runs = [("ref", lambda: make_engine("L2-legacy", E, R, dev, cfg, "triton", seed=7), None),
                ("ref-seed2", lambda: make_engine("L2-legacy", E, R, dev, cfg, "triton", seed=8), None),
                ("cheap", lambda: make_engine(a.cheap, E, R, dev, cfg, a.backend, seed=7, params=cparams), None),
                ("static-50", lambda: make_adaptive(E, R, dev, cfg, fid(), seed=7), None)]
        runs += [(f"toggle-{n}", lambda: make_adaptive(E, R, dev, cfg, fid(), seed=7), int(n))
                 for n in a.toggles.split(",")]
        full = lambda: FidelityConfig(cheap=a.cheap, expensive="L2-legacy", cheap_backend=a.backend,  # noqa: E731
                                      expensive_backend="triton", cheap_params=cparams, mode="expensive", layout="mask")
        runs += [(f"reentry-{n}", lambda: make_adaptive(E, R, dev, cfg, full(), seed=7), -int(n))
                 for n in a.reentries.split(",") if n]
        ref = mix = None
        for name, build, tog in runs:
            h, t = run(build(), wl, a.warm, tog if (tog or 0) > 0 else None, -tog if (tog or 0) < 0 else None)
            row = {"load": float(p), "run": name, "toggle": tog, "cheap": a.cheap, "E": E, "R": R, "arena": arena,
                   "delivered": int(h.sum()), "timed_out": t, "drop": t / max(1, t + int(h.sum())),
                   "p50": quantile(h, 0.5), "p95": quantile(h, 0.95), "p99": quantile(h, 0.99), "hist": h.tolist()}
            ref = row if name == "ref" else ref
            mix = row if name == "static-50" else mix
            row["w1"], row["ks"] = dist(h, torch.tensor(ref["hist"]))
            if mix is not None:
                row["w1_vs_static"], row["ks_vs_static"] = dist(h, torch.tensor(mix["hist"]))
                row["drop_vs_static"] = row["drop"] - mix["drop"]
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            print(json.dumps({k: v for k, v in row.items() if k != "hist"}), flush=True)
            torch.cuda.empty_cache()
    fh.close()


if __name__ == "__main__":
    main()
