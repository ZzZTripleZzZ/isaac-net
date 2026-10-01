"""Fit L05 / L05Q lookup tables (delay quantiles and drop share per bin) from L2-legacy rollouts.

usage: python benchmarks/adaptive/fit_l05q.py --out l05q_fit.pt [--E 512 --R 16 --steps 400 --arena 5,60]

The rollout is the sweep's mixed workload (sweep.workload, its own seed) on L2-legacy (triton, log_stats). Every
delivered or timed-out frame is binned with netsim.lookup_key on the features the engine logged at its arrival
(backlogged robots, SNR, own queue, class). Per bin: the 101 delay quantiles (0 %, 1 %, ..., 100 %) of the delivered
frames, and pdrop = timed out / (delivered + timed out). Bins with fewer than --min_n frames back off to the bin
without the own-queue index, then to the class marginal. The file holds {"L05": {...}, "L05Q": {...}} in the params
format of make_engine (q [*bins, 101], pdrop [*bins]); it stays outside the source tree.
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _root)
from sweep import workload  # noqa: E402
from isaac_net.core import NRConfig, make_engine  # noqa: E402
from isaac_net.core.proto import netsim as ns  # noqa: E402
from isaac_net.core.proto.netsim import Requests  # noqa: E402

QS = torch.linspace(0, 1, 101)


def tables(mode, dl, xf, delay, min_n):
    """(q, pdrop) for mode from delivered features dl, timed-out features xf and delays of delivered frames."""
    kd = ns.lookup_key(mode, dl["f_nact"], dl["f_snr"], dl["f_own"], dl["cls"])
    kx = ns.lookup_key(mode, xf["f_nact"], xf["f_snr"], xf["f_own"], xf["cls"])
    shape = [len(ns.NACT_EDGES) + 1, len(ns.SNR_EDGES) + 1] + ([len(ns.OWNQ_EDGES) + 1] if mode == "L05Q" else []) + [2]
    q = torch.zeros(*shape, 101)
    pd = torch.zeros(*shape)
    idx = torch.cartesian_prod(*[torch.arange(n) for n in shape])

    def pick(key, k, dims):
        m = torch.ones(key[0].shape[0], dtype=torch.bool)
        for d in dims:
            m &= key[d] == k[d]
        return m

    all_dims = list(range(len(shape)))
    backoffs = [all_dims, [d for d in all_dims if not (mode == "L05Q" and d == 2)], [len(shape) - 1]]
    for k in idx:
        for dims in backoffs:
            md, mx = pick(kd, k, dims), pick(kx, k, dims)
            nd, nx = int(md.sum()), int(mx.sum())
            if nd + nx >= min_n and nd > 0:
                break
        q[tuple(k)] = torch.quantile(delay[md], QS) if nd else torch.full((101,), float(ns.TIMEOUT))
        pd[tuple(k)] = nx / max(1, nd + nx)
    return q, pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--E", type=int, default=512)
    ap.add_argument("--R", type=int, default=16)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--arena", default="5,60")
    ap.add_argument("--min_n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=99)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if os.path.abspath(a.out).startswith(_root + os.sep):
        raise SystemExit("write fitted tables outside the source tree")
    dev = torch.device("cuda")
    lo, hi = (float(x) for x in a.arena.split(","))
    wl = workload(a.E, a.R, a.steps, "mixed", dev, seed=a.seed, arena=(lo, hi))
    net = make_engine("L2-legacy", a.E, a.R, dev, NRConfig(seed=a.seed), "triton", seed=a.seed)
    net.log_stats = True
    net.log_cap_max = a.steps - ns.TIMEOUT - 1            # frames whose fate is known by the end of the rollout
    for send, pos, rs in wl:
        if rs is not None:
            net.reset(rs)
        net.submit(None, Requests(send))
        net.step(None, pos)
    st = net.collect()
    dl = {f: st["d_" + f] for f in ns.NetBase.FEATS}
    xf = {f: st["x_" + f] for f in ns.NetBase.FEATS}
    out = {}
    for mode in ("L05", "L05Q"):
        q, pd = tables(mode, dl, xf, st["delay"].float(), a.min_n)
        out[mode] = {"q": q, "pdrop": pd}
    out["meta"] = {"frames_delivered": int(st["delay"].numel()), "frames_timed_out": int(xf["cls"].numel()),
                   "arena": (lo, hi), "E": a.E, "R": a.R, "steps": a.steps, "seed": a.seed}
    torch.save(out, a.out)
    print(out["meta"])


if __name__ == "__main__":
    main()
