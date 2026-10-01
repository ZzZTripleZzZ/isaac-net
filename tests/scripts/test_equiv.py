"""Equivalence test: reference engine (netsim) vs a fast backend (netsim_fast) under identical random draws.

The reference draws its randomness through torch.randn_like / rand_like (L2) and torch.randn / rand (NetDelay
levels). During the reference call these functions are temporarily replaced by readers of pre-generated
tensors, and the fast engine receives the same tensors via set_noise(). Inputs (send, det, hid, SNR) come from
a synthetic, regime-switching workload that exercises idle, loaded and overloaded cells (SR, HARQ, RLC wait,
buffer overflow, timeouts). Optional random partial resets (--reset_every) use the per-env clock.

usage: python test_equiv.py --rung L2 --E 16 --R 16 --steps 300 --backend graph --api new --reset_every 7
       python test_equiv.py --rung all --backend graph            # every level
Prints one JSON line per level; exit code 1 if a bitwise backend (eager/graph) is not bitwise identical.
"""
import argparse
import json
import sys
import time

import os
import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root
_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from isaac_net.core.proto import netsim
from isaac_net.core.proto.netsim import F, Requests
from isaac_net.core.proto.netsim_fast import NetFast
from testlib import SIZES, Workload, drive, is_ref, maxdiff, same, state_names, synthetic_params

OUT_KEYS = ["delivered", "timed_out", "cap", "cls", "delay", "newest", "det_env", "queue_len", "queue_bytes", "t"]


def run(rung, a):
    dev = "cuda"
    params = synthetic_params(rung, dev)
    ref = netsim.make_net(rung, a.E, a.R, dev, SIZES, params, seed=a.seed)
    fast = NetFast(rung, a.E, a.R, dev, SIZES, params=params, backend=a.backend, inject=True, seed=a.seed)
    ref.log_stats = fast.log_stats = bool(a.log_stats)
    fin_ref = {}
    orig_tx = ref._transmit
    def tx_wrap(t, snr):
        f = orig_tx(t, snr); fin_ref["f"] = f.clone(); return f
    ref._transmit = tx_wrap

    wl = Workload(a.E, a.R, dev, a.seed + 1)
    names = state_names(rung)
    first_bad, worst, out_bad = None, {}, {}
    n_fin = n_fin_mismatch = n_resets = 0
    max_fin_diff = 0.0
    q_hist = []
    t0 = time.time()
    for step in range(a.steps):
        if a.reset_every and step > 0 and step % a.reset_every == 0:
            ids = wl.reset_ids(a.reset_frac)
            ref.reset(ids); fast.reset(ids); n_resets += int(ids.numel())
        if a.teacher_force:
            for n in names:
                getattr(fast, n).copy_(getattr(ref, n))
        send, det, hid, snr = wl.inputs(step)
        noise, arr = wl.noise(), wl.arrival_noise()
        req = Requests(send, det, hid)
        t = None if a.api == "new" else step        # new API: per-env engine clock
        o_r = drive(ref, rung, t, req, snr, noise, arr, a.api, hid)
        o_f = drive(fast, rung, t, req, snr, noise, arr, a.api, hid)
        fr, ff = fin_ref["f"], fast._fin
        fin_both = torch.isfinite(fr) | torch.isfinite(ff)
        n_fin += int(torch.isfinite(fr).sum())
        mism = (fr != ff) & fin_both
        n_fin_mismatch += int(mism.sum())
        if fin_both.any():
            dd = (fr - ff)[fin_both].abs()
            dd = torch.where(torch.isnan(dd), torch.full_like(dd, float("inf")), dd)
            max_fin_diff = max(max_fin_diff, float(dd.max()))
        ok = int(mism.sum()) == 0
        if a.api == "new":
            for k in OUT_KEYS:
                s = same(o_r[k], o_f[k])
                ok &= s
                if not s:
                    out_bad[k] = out_bad.get(k, 0) + 1
        else:
            ok &= torch.equal(o_r[0], o_f[0]) and torch.equal(o_r[1], o_f[1])
        for n in names:
            x, y = getattr(ref, n), getattr(fast, n)
            s = same(x, y)
            ok &= s
            if not s:
                worst[n] = max(worst.get(n, 0.0), maxdiff(x, y))
        if not ok and first_bad is None:
            first_bad = step
        q_hist.append(float(ref.queued().float().mean()))
    torch.cuda.synchronize()
    res = {"rung": rung, "E": a.E, "R": a.R, "steps": a.steps, "backend": a.backend, "api": a.api, "seed": a.seed,
           "reset_every": a.reset_every, "envs_reset": n_resets,
           "bitwise_identical_all_steps": first_bad is None, "first_mismatch_step": first_bad,
           "frames_delivered_ref": n_fin, "finish_time_mismatches": n_fin_mismatch,
           "max_finish_time_diff_steps": max_fin_diff, "max_state_diff": worst, "output_mismatch_steps": out_bad,
           "mean_queue": round(sum(q_hist) / len(q_hist), 2), "max_mean_queue": round(max(q_hist), 2),
           "wall_s": round(time.time() - t0, 1)}
    if a.log_stats:
        cr, cf = ref.collect(), fast.collect()
        res["stats_identical"] = all(torch.equal(cr[k], cf[k]) if k != "overflow" else cr[k] == cf[k] for k in cr)
        res["delivered_frames"] = int(cr["delay"].numel()); res["timeouts"] = int(cr["x_cls"].numel())
        res["overflow"] = cr["overflow"]
        res["fast_delivered"] = int(cf["delay"].numel()); res["fast_timeouts"] = int(cf["x_cls"].numel())
        res["mean_delay_ref"] = float(cr["delay"].mean()) if cr["delay"].numel() else None
        res["mean_delay_fast"] = float(cf["delay"].mean()) if cf["delay"].numel() else None
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rung", default="L2", help="level, comma list, or 'all'")
    ap.add_argument("--E", type=int, default=16)
    ap.add_argument("--R", type=int, default=16)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--backend", default="graph")
    ap.add_argument("--api", default="new", choices=["new", "legacy"])
    ap.add_argument("--reset_every", type=int, default=0, help="partial reset of a random env subset every N steps")
    ap.add_argument("--reset_frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log_stats", type=int, default=1)
    ap.add_argument("--teacher_force", type=int, default=0, help="copy reference state into the fast engine before every step")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    rungs = netsim.RUNGS if a.rung == "all" else a.rung.split(",")
    if a.backend == "triton":
        rungs = [r for r in rungs if r in ("L1", "L2")]
    fail = False
    for rung in rungs:
        torch.manual_seed(a.seed)
        res = run(rung, a)
        print(json.dumps(res), flush=True)
        if a.out:
            with open(a.out, "a") as f:
                f.write(json.dumps(res) + "\n")
        if a.backend in ("eager", "graph") and not (res["bitwise_identical_all_steps"] and res.get("stats_identical", True)):
            fail = True
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
