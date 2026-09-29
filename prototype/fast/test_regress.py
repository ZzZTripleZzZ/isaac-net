"""Regression test: the reworked reference (netsim.py: per-env clock, partial resets, submit/step API) against a
frozen copy of the original reference (netsim_v0.py, global int clock) on every level, without resets.

Both engines get identical draws: L2 through randn_like / rand_like injection, NetDelay levels by mapping the
new per-robot draws z, u1, u2 [E,R] onto v0's compacted draws z[new], u1[new], u2[new]. Initial random state
(fading h, L0DR parameters) is copied from the new engine into v0. Everything (state, newest, det_env, stats)
must be bitwise equal, with both the legacy API (int t) and the new API (engine clock).

usage: python test_regress.py --E 16 --R 16 --steps 200
"""
import argparse
import json
import sys

import os
import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

import netsim
import netsim_v0
from netsim import F, Requests
from testlib import SIZES, Inject, InjectArrival, Workload, same, synthetic_params


class InjectV0:
    """v0 NetDelay draws torch.randn(n) / torch.rand(n) over the enqueuing robots (nonzero order)."""

    def __init__(self, new, z, u1, u2):
        self.n, self.u = [z[new]], [u1[new], u2[new]]

    def __enter__(self):
        self._rn, self._ru = torch.randn, torch.rand
        torch.randn = lambda *a, **k: self.n.pop(0)
        torch.rand = lambda *a, **k: self.u.pop(0)

    def __exit__(self, *exc):
        torch.randn, torch.rand = self._rn, self._ru


def run(rung, api, a):
    dev = "cuda"
    params = synthetic_params(rung, dev)
    new = netsim.make_net(rung, a.E, a.R, dev, SIZES, params, seed=a.seed)
    old = netsim_v0.make_net(rung, a.E, a.R, dev, SIZES, params)
    names = list(netsim.NetBase.FIELDS)
    if rung == "L2":
        old.h.copy_(new.h)
        names += netsim.NetSlot.MAC
    if rung == "L0DR":
        for n in ("mu", "sig", "p"):
            getattr(old, n).copy_(getattr(new, n))
    new.log_stats = old.log_stats = True
    wl = Workload(a.E, a.R, dev, a.seed + 1)
    bad = {}
    first_bad = None
    for t in range(a.steps):
        send, det, hid, snr = wl.inputs(t)
        nz, u = wl.noise()
        arr = wl.arrival_noise()
        delay_rung = rung in ("L0", "L0DR", "L05", "L05Q")
        # v0
        m = (send > 0) & (old.queued() < F)
        if delay_rung:
            with InjectV0(m, *arr):
                old.add_frames(t, send, det, hid, snr)
        else:
            old.add_frames(t, send, det, hid, snr)
        if rung == "L2":
            with Inject(nz, u):
                o_old = old.step(t, snr, hid)
        else:
            o_old = old.step(t, snr, hid)
        # reworked reference
        if delay_rung:
            with InjectArrival(*arr):
                new.submit(t if api == "legacy" else None, Requests(send, det, hid), snr)
        else:
            new.submit(t if api == "legacy" else None, Requests(send, det, hid), snr)
        ctx = Inject(nz, u) if rung == "L2" else None
        if ctx:
            ctx.__enter__()
        if api == "legacy":
            o_new = new.step(t, snr, hid)
        else:
            d = new.step(None, snr)
            o_new = (d["newest"], d["det_env"])
            assert int(d["t"][0]) == t
        if ctx:
            ctx.__exit__(None, None, None)
        ok = torch.equal(o_old[0], o_new[0]) and torch.equal(o_old[1], o_new[1])
        if not ok:
            bad["outputs"] = bad.get("outputs", 0) + 1
        for n in names:
            if not same(getattr(old, n), getattr(new, n)):
                bad[n] = bad.get(n, 0) + 1
                ok = False
        if not ok and first_bad is None:
            first_bad = t
    co, cn = old.collect(), new.collect()
    stats_ok = all(torch.equal(co[k], cn[k]) if k != "overflow" else co[k] == cn[k] for k in co)
    return {"rung": rung, "api": api, "E": a.E, "R": a.R, "steps": a.steps, "bitwise_vs_v0": first_bad is None and stats_ok,
            "first_mismatch_step": first_bad, "mismatch_counts": bad, "stats_identical": stats_ok,
            "delivered_frames": int(cn["delay"].numel()), "timeouts": int(cn["x_cls"].numel()), "overflow": cn["overflow"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rung", default="all")
    ap.add_argument("--E", type=int, default=16)
    ap.add_argument("--R", type=int, default=16)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    rungs = netsim.RUNGS if a.rung == "all" else a.rung.split(",")
    ok = True
    for rung in rungs:
        for api in ("legacy", "new"):
            torch.manual_seed(a.seed)
            res = run(rung, api, a)
            ok &= res["bitwise_vs_v0"]
            print(json.dumps(res), flush=True)
            if a.out:
                with open(a.out, "a") as f:
                    f.write(json.dumps(res) + "\n")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
