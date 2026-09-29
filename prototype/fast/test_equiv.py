"""Equivalence test: NetSlot (original, unmodified) vs NetSlotFast under identical random draws.

The original draws its per-slot noise through torch.randn_like / torch.rand_like. During the reference
step these two functions are temporarily replaced by readers of a pre-generated noise tensor, and the
fast engine receives the same tensor via set_noise(). Inputs (send, det, hid, SNR) are a synthetic,
regime-switching workload that exercises idle, loaded and overloaded cells (SR, HARQ, RLC wait,
buffer overflow, timeouts).

usage: python test_equiv.py --E 16 --R 16 --steps 300 --backend graph
"""
import argparse
import json
import time

import torch

import netsim
from netsim import NetSlot, UL_PER_STEP, S
from netsim_fast import NetSlotFast

SIZES = (4000.0, 30000.0)


class Inject:
    def __init__(self, nz, u):
        self.nz, self.u, self.i, self.j = nz, u, 0, 0

    def __enter__(self):
        self._rn, self._ru = torch.randn_like, torch.rand_like
        def rn(x, *a, **k):
            assert x.shape == self.nz.shape[1:]
            v = self.nz[self.i]; self.i += 1; return v
        def ru(x, *a, **k):
            assert x.shape == self.u.shape[1:]
            v = self.u[self.j]; self.j += 1; return v
        torch.randn_like, torch.rand_like = rn, ru
        return self

    def __exit__(self, *exc):
        torch.randn_like, torch.rand_like = self._rn, self._ru
        assert self.i == UL_PER_STEP and self.j == UL_PER_STEP, (self.i, self.j)


class Workload:
    def __init__(self, E, R, dev, seed):
        self.g = torch.Generator(device=dev).manual_seed(seed)
        self.E, self.R, self.dev = E, R, dev
        self.base = -5 + 40 * torch.rand(E, R, device=dev, generator=self.g)
        self.hid = torch.zeros(E, dtype=torch.long, device=dev)

    def inputs(self, t):
        E, R, d, g = self.E, self.R, self.dev, self.g
        phase = (t // 25) % 4          # idle, medium, burst (all large), medium-small
        p = [0.03, 0.3, 0.9, 0.5][phase]
        big = [0.3, 0.5, 1.0, 0.1][phase]
        tx = torch.rand(E, R, device=d, generator=g) < p
        lg = torch.rand(E, R, device=d, generator=g) < big
        send = tx.long() * (1 + lg.long())
        det = tx & (torch.rand(E, R, device=d, generator=g) < 0.3)
        self.hid = self.hid + (torch.rand(E, device=d, generator=g) < 0.05).long()
        self.base = (self.base + 0.5 * torch.randn(E, R, device=d, generator=g)).clamp(-10, 40)
        return send, det, self.hid.clone(), self.base.clone()

    def noise(self):
        E, R, d, g = self.E, self.R, self.dev, self.g
        return (torch.randn(UL_PER_STEP, E, R, S, 2, device=d, generator=g),
                torch.rand(UL_PER_STEP, E, R, device=d, generator=g))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--E", type=int, default=16)
    ap.add_argument("--R", type=int, default=16)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--backend", default="graph")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log_stats", type=int, default=1)
    ap.add_argument("--teacher_force", type=int, default=0, help="copy NetSlot state into the fast engine before every step")
    a = ap.parse_args()
    dev = "cuda"
    torch.manual_seed(a.seed)
    ref = NetSlot(a.E, a.R, dev, SIZES)
    fast = NetSlotFast(a.E, a.R, dev, SIZES, backend=a.backend, inject=True)
    fast.h.copy_(ref.h)
    ref.log_stats = fast.log_stats = bool(a.log_stats)

    fin_ref = {}
    orig_tx = ref._transmit
    def tx_wrap(t, snr):
        f = orig_tx(t, snr); fin_ref["f"] = f.clone(); return f
    ref._transmit = tx_wrap

    wl = Workload(a.E, a.R, dev, a.seed + 1)
    names = NetSlotFast.FIELDS + NetSlotFast.MAC
    first_bad, worst = None, {}
    n_frames_fin = 0
    n_fin_mismatch = 0
    max_fin_diff = 0.0
    q_hist = []
    t0 = time.time()
    tf_rows = tf_rows_bad = 0
    tf_state_diff = {}
    for t in range(a.steps):
        if a.teacher_force:
            for n in names:
                getattr(fast, n).copy_(getattr(ref, n))
        send, det, hid, snr = wl.inputs(t)
        nz, u = wl.noise()
        ref.add_frames(t, send, det, hid, snr)
        fast.add_frames(t, send, det, hid, snr)
        with Inject(nz, u):
            n_r, d_r = ref.step(t, snr, hid)
        fast.set_noise(nz, u)
        n_f, d_f = fast.step(t, snr, hid)
        fr, ff = fin_ref["f"], fast._fin
        fin_both = torch.isfinite(fr) | torch.isfinite(ff)
        n_frames_fin += int(torch.isfinite(fr).sum())
        mism = (fr != ff) & fin_both
        n_fin_mismatch += int(mism.sum())
        if fin_both.any():
            dd = (fr - ff)[fin_both].abs()
            dd = torch.where(torch.isnan(dd), torch.full_like(dd, float("inf")), dd)
            max_fin_diff = max(max_fin_diff, float(dd.max()))
        if a.teacher_force:   # one-step agreement from an identical state
            active = (ref.queued() > 0) | torch.isfinite(fr).any(-1) | torch.isfinite(ff).any(-1)
            rowbad = (mism.any(-1) | (n_r != n_f)) & active
            tf_rows += int(active.sum()); tf_rows_bad += int(rowbad.sum())
            for n in ["rem", "bsr", "avg", "olla", "h"]:
                dd = (getattr(ref, n) - getattr(fast, n)).abs()
                tf_state_diff[n] = max(tf_state_diff.get(n, 0.0), float(dd.max()))
        ok = torch.equal(n_r, n_f) and torch.equal(d_r, d_f) and int(mism.sum()) == 0
        for n in names:
            x, y = getattr(ref, n), getattr(fast, n)
            same = torch.equal(x, y)
            ok &= same
            if not same:
                diff = (x.double() - y.double()).abs()
                diff = diff[~torch.isnan(diff)]
                worst[n] = max(worst.get(n, 0.0), float(diff.max()) if diff.numel() else 0.0)
        if not ok and first_bad is None:
            first_bad = t
        q_hist.append(float(ref.queued().float().mean()))
    torch.cuda.synchronize()
    res = {"E": a.E, "R": a.R, "steps": a.steps, "backend": a.backend, "seed": a.seed,
           "bitwise_identical_all_steps": first_bad is None, "first_mismatch_step": first_bad,
           "frames_delivered_ref": n_frames_fin, "finish_time_mismatches": n_fin_mismatch,
           "max_finish_time_diff_steps": max_fin_diff, "max_state_diff": worst,
           "mean_queue": round(sum(q_hist)/len(q_hist), 2), "max_mean_queue": round(max(q_hist), 2),
           "wall_s": round(time.time() - t0, 1)}
    if a.teacher_force:
        res["tf_robot_steps"] = tf_rows; res["tf_robot_steps_mismatch"] = tf_rows_bad
        res["tf_max_state_diff"] = tf_state_diff
    if a.log_stats:
        cr, cf = ref.collect(), fast.collect()
        res["stats_identical"] = all(torch.equal(cr[k], cf[k]) if k != "overflow" else cr[k] == cf[k] for k in cr)
        res["delivered_frames"] = int(cr["delay"].numel()); res["timeouts"] = int(cr["x_cls"].numel())
        res["overflow"] = cr["overflow"]
        res["fast_delivered"] = int(cf["delay"].numel()); res["fast_timeouts"] = int(cf["x_cls"].numel())
        res["mean_delay_ref"] = float(cr["delay"].mean()); res["mean_delay_fast"] = float(cf["delay"].mean())
    print(json.dumps(res))


if __name__ == "__main__":
    main()
