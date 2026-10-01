"""Partial-reset test: reset(env_ids) re-initializes exactly those envs and leaves every other env bit-for-bit
unaffected, on every level and backend.

Two engines A and B with the same seed get the same inputs and the same random draws. A partially resets a
random env subset every --reset_every steps (index tensor, bool mask and python list formats in turn); B never
resets. After every step, every state tensor and every step() output of the envs that A never reset must be
bitwise equal between A and B. Right after each reset the reset rows must hold their initial values (clock 0,
empty FIFOs, initial MAC state, fading/L0DR draws taken from the engine generator), and the fast backends must
not have reallocated any buffer (data_ptr unchanged, so the captured CUDA graphs stay valid).

--inject 1 feeds both engines pre-drawn noise; --inject 0 lets them draw from the default CUDA generator,
restoring the generator state between A and B (checks that a reset does not shift the other envs' stream).

usage: python test_reset.py --rung all --backend reference,graph,triton --E 32 --R 8 --steps 120
Prints one JSON line per (level, backend); exit code 1 on any failure.
"""
import argparse
import json
import math
import sys

import os
import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root
_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from isaac_net.core.proto import netsim
from isaac_net.core.proto.netsim import Requests, S
from isaac_net.core.proto.netsim_fast import NetFast
from testlib import SIZES, Workload, drive, is_ref, same, state_names, synthetic_params

OUT_KEYS = ["delivered", "timed_out", "cap", "cls", "delay", "newest", "det_env", "queue_len", "queue_bytes", "t"]


def build(rung, backend, E, R, params, seed, inject):
    if backend == "reference":
        return netsim.make_net(rung, E, R, "cuda", SIZES, params, seed=seed)
    return NetFast(rung, E, R, "cuda", SIZES, params=params, backend=backend, inject=inject, seed=seed)


def ptrs(net):
    return {n: getattr(net, n).data_ptr() for n in dir(net)
            if not n.startswith("__") and isinstance(getattr(net, n, None), torch.Tensor)}


def check_reset_rows(net, rung, ids, gen_state):
    """Reset rows hold initial values; random rows equal fresh draws from the engine generator."""
    errs = []
    for n in netsim.NetBase.FIELDS:
        x = getattr(net, n)[ids]
        if not torch.equal(x, torch.full_like(x, netsim.NetBase.INIT[n])):
            errs.append(n)
    if not torch.equal(net.clock[ids], torch.zeros_like(net.clock[ids])):
        errs.append("clock")
    g = torch.Generator(device="cuda")
    g.set_state(gen_state)
    n = ids.numel()
    if rung == "L2":
        for name, v in netsim.NetSlot.MAC_INIT.items():
            x = getattr(net, name)[ids]
            if not torch.equal(x, torch.full_like(x, v)):
                errs.append(name)
        h = torch.randn(n, net.R, S, 2, device="cuda", generator=g) / math.sqrt(2)
        if not torch.equal(net.h[ids], h):
            errs.append("h")
    if rung == "L0DR":
        mu = math.log(0.05) + (math.log(10.0) - math.log(0.05)) * torch.rand(n, device="cuda", generator=g)
        if not torch.equal(net.mu[ids], mu):
            errs.append("mu")
    return errs


def run(rung, backend, a):
    E, R = a.E, a.R
    params = synthetic_params(rung, "cuda")
    A = build(rung, backend, E, R, params, a.seed, bool(a.inject))
    B = build(rung, backend, E, R, params, a.seed, bool(a.inject))
    if not is_ref(A):
        B._seed.copy_(A._seed)
    wl = Workload(E, R, "cuda", a.seed + 1)
    names = state_names(rung)
    touched = torch.zeros(E, dtype=torch.bool, device="cuda")
    p0 = None
    res = {"rung": rung, "backend": backend, "E": E, "R": R, "steps": a.steps, "inject": a.inject,
           "resets": 0, "envs_reset": 0, "untouched_envs_at_end": None, "untouched_bitwise": True,
           "first_bad_step": None, "bad_tensors": [], "reset_rows_ok": True, "reset_row_errors": [],
           "no_realloc": True, "delivered_untouched": 0}
    for step in range(a.steps):
        if a.reset_every and step > 0 and step % a.reset_every == 0:
            ids = wl.reset_ids(a.reset_frac)
            fmt = res["resets"] % 3
            arg = ids if fmt == 0 else (torch.zeros(E, dtype=torch.bool, device="cuda").index_fill_(0, ids, True)
                                        if fmt == 1 else ids.tolist())
            gst = A.gen.get_state()
            A.reset(arg)
            errs = check_reset_rows(A, rung, ids, gst)
            if errs:
                res["reset_rows_ok"] = False
                res["reset_row_errors"] = sorted(set(res["reset_row_errors"]) | set(errs))
            touched[ids] = True
            res["resets"] += 1
            res["envs_reset"] += int(ids.numel())
        send, det, hid, snr = wl.inputs(step)
        noise, arr = wl.noise(), wl.arrival_noise()
        req = Requests(send, det, hid)
        if a.inject:
            oa = drive(A, rung, None, req, snr, noise, arr)
            ob = drive(B, rung, None, req, snr, noise, arr)
        else:
            st = torch.cuda.get_rng_state()
            A.submit(None, req, snr); oa = A.step(None, snr)
            torch.cuda.set_rng_state(st)
            B.submit(None, req, snr); ob = B.step(None, snr)
        keep = ~touched
        bad = []
        for n in names:
            if not same(getattr(A, n)[keep], getattr(B, n)[keep]):
                bad.append(n)
        for k in OUT_KEYS:
            if not same(oa[k][keep], ob[k][keep]):
                bad.append("out." + k)
        res["delivered_untouched"] += int(ob["delivered"][keep].sum())
        if bad:
            res["untouched_bitwise"] = False
            res["bad_tensors"] = sorted(set(res["bad_tensors"]) | set(bad))
            if res["first_bad_step"] is None:
                res["first_bad_step"] = step
        if not is_ref(A):
            if p0 is None:
                p0 = ptrs(A)          # after the first step (graphs captured)
            elif ptrs(A) != p0:
                res["no_realloc"] = False
    torch.cuda.synchronize()
    res["untouched_envs_at_end"] = int((~touched).sum())
    res["pass"] = res["untouched_bitwise"] and res["reset_rows_ok"] and res["no_realloc"] and res["untouched_envs_at_end"] > 0
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rung", default="all")
    ap.add_argument("--backend", default="reference,graph,triton")
    ap.add_argument("--E", type=int, default=32)
    ap.add_argument("--R", type=int, default=8)
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--reset_every", type=int, default=9)
    ap.add_argument("--reset_frac", type=float, default=0.15)
    ap.add_argument("--inject", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    rungs = netsim.RUNGS if a.rung == "all" else a.rung.split(",")
    ok = True
    for rung in rungs:
        for backend in a.backend.split(","):
            if backend == "triton" and rung not in ("L1", "L2"):
                continue
            torch.manual_seed(a.seed)
            res = run(rung, backend, a)
            ok &= res["pass"]
            print(json.dumps(res), flush=True)
            if a.out:
                with open(a.out, "a") as f:
                    f.write(json.dumps(res) + "\n")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
