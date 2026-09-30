"""Gradient sanity of the differentiable models against finite-difference sensitivities of the real engines.

Decisions (shared by all robots, at the operating point p0, B0 = 12 kB, 23 dBm, the drawn positions):
    p      send probability                          FD step +-0.05
    kB     message size in kB                        FD step +-10 %
    tx     transmit power offset in dB               FD step +-1 dB
    dist   radial scale of every robot's position    FD step +-5 %
    p_own / p_cross: the send probability of robot 0 only; own = robot 0's KPI, cross = the other robots' mean.
KPIs: mean delay and mean AoI (control steps), delivery ratio.

Finite differences (central, common random numbers: the same send uniforms and engine seed on both sides) of
    L2-legacy (the slot-level engine: SR/BSR, PF, fading, OLLA, BLER, HARQ), L1 and QA (the discrete levels the
    relaxations come from), over n_seeds independent scenarios (mean and standard deviation of the slope),
and autograd derivatives of
    L1D and QAD at tau = 0.05, with mean-field sends (weight p) and with relaxed-Bernoulli sends (the same
    uniforms, lam = 0.1), averaged over the same scenarios.

usage: python sensitivity.py [device] [E] [T] [n_seeds] [loads] [decisions]  -> results/sensitivity_<load>_<dec>.csv
       python sensitivity.py summarize   -> results/sensitivity.csv and results/sensitivity.md (a cross marks a sign
       that disagrees with a resolved L2-legacy slope, |mean| > 2 std over the seeds)
"""
import csv
import glob
import os
import statistics
import sys

import torch

from common import OUT, Scenario

B0 = 12000.0
LOADS = [0.3, 0.5]
STEP = {"p": 0.05, "kB": 1.2, "tx": 1.0, "dist": 0.05, "p_own": 0.05}
KPIS = ["mean_delay", "mean_aoi", "mean_delivery"]
MODELS = [("L2-legacy", "fd", None), ("L1", "fd", None), ("QA", "fd", None),
          ("L1D-mean", "grad", ("L1", "mean")), ("L1D-concrete", "grad", ("L1", "concrete")),
          ("QAD-mean", "grad", ("QA", "mean")), ("QAD-concrete", "grad", ("QA", "concrete"))]


def _robot_kpis(r):
    """own (robot 0) and cross (robots 1..) delivered-weighted mean delay and mean AoI."""
    d, dl = r["delay"], r["delivered"]
    own = (d[:, 0] * dl[:, 0]).sum() / dl[:, 0].sum().clamp(min=1e-9)
    cross = (d[:, 1:] * dl[:, 1:]).sum() / dl[:, 1:].sum().clamp(min=1e-9)
    return {"own_delay": own, "cross_delay": cross, "own_aoi": r["aoi"][:, 0].mean(), "cross_aoi": r["aoi"][:, 1:].mean()}


def inputs(sc, p, th, dec, dtype):
    """(p [E,R], B, tx, pos) at the operating point with decision dec moved to th."""
    E, R = sc.E, sc.R
    P = torch.full((E, R), p, device=sc.dev, dtype=dtype)
    B = torch.full((E, R), B0, device=sc.dev, dtype=dtype)
    tx, pos = None, sc.pos.to(dtype)
    if dec == "p":
        P = th.expand(E, R) if torch.is_tensor(th) else torch.full((E, R), th, device=sc.dev, dtype=dtype)
    elif dec == "p_own":
        col = th.expand(E, 1) if torch.is_tensor(th) else torch.full((E, 1), th, device=sc.dev, dtype=dtype)
        P = torch.cat([col, P[:, 1:]], 1)
    elif dec == "kB":
        B = (th * 1000.0) * torch.ones_like(B)
    elif dec == "tx":
        tx = 23.0 + th * torch.ones_like(P)
    elif dec == "dist":
        pos = pos * th
    return P, B, tx, pos


def base_value(dec, p):
    return {"p": p, "p_own": p, "kB": B0 / 1000.0, "tx": 0.0, "dist": 1.0}[dec]


def kpis_of(r, dec):
    out = {k: r[k] for k in KPIS}
    if dec == "p_own":
        out.update(_robot_kpis(r))
    return out


def fd(sc, level, p, dec):
    th0, h = base_value(dec, p), STEP[dec]
    vals = []
    for s in (+1, -1):
        P, B, tx, pos = inputs(sc, p, th0 + s * h, dec, torch.float32)
        r = sc.discrete_kpis(level, P, B, tx_dbm=tx, pos=pos)
        vals.append({k: float(v) for k, v in kpis_of(r, dec).items()})
    return {k: (vals[0][k] - vals[1][k]) / (2 * h) for k in vals[0]}


def grad(sc, mode, send, p, dec):
    th = torch.tensor(base_value(dec, p), device=sc.dev, dtype=torch.float64, requires_grad=True)
    P, B, tx, pos = inputs(sc, p, th, dec, torch.float64)
    r = sc.diff_kpis(P, B, tx_dbm=tx, pos=pos, mode=mode, send=send, dtype=torch.float64)
    ks = kpis_of(r, dec)
    return {k: float(torch.autograd.grad(v, th, retain_graph=True)[0]) for k, v in ks.items()}


def main():
    if sys.argv[1:2] == ["summarize"]:
        return summarize()
    dev = sys.argv[1] if len(sys.argv) > 1 else "cpu"
    E = int(sys.argv[2]) if len(sys.argv) > 2 else 128
    T = int(sys.argv[3]) if len(sys.argv) > 3 else 300
    n_seeds = int(sys.argv[4]) if len(sys.argv) > 4 else 3
    loads = [float(v) for v in sys.argv[5].split(",")] if len(sys.argv) > 5 else LOADS
    decs = sys.argv[6].split(",") if len(sys.argv) > 6 else list(STEP)
    R = 8
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for p in loads:
        for dec in decs:
            acc = {m[0]: [] for m in MODELS}
            for seed in range(n_seeds):
                sc = Scenario(E, R, T, dev, seed=100 + seed)
                for name, kind, arg in MODELS:
                    acc[name].append(fd(sc, name, p, dec) if kind == "fd" else grad(sc, *arg, p, dec))
            for name, _, _ in MODELS:
                for k in acc[name][0]:
                    xs = [a[k] for a in acc[name]]
                    row = dict(load=p, decision=dec, kpi=k, model=name, mean=statistics.fmean(xs),
                               std=statistics.stdev(xs) if len(xs) > 1 else 0.0, E=E, T=T, n_seeds=n_seeds)
                    rows.append(row)
                    print(row, flush=True)
            tag = f"{p}_{dec}"
            with open(os.path.join(OUT, f"sensitivity_{tag}.csv"), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[-1]))
                w.writeheader()
                w.writerows([r for r in rows if r["load"] == p and r["decision"] == dec])


def summarize():
    """Merge results/sensitivity_*.csv into sensitivity.csv and the table sensitivity.md."""
    rows = []
    for fn in sorted(glob.glob(os.path.join(OUT, "sensitivity_*.csv"))):
        with open(fn) as f:
            for r in csv.DictReader(f):
                r["load"], r["mean"], r["std"] = float(r["load"]), float(r["mean"]), float(r["std"])
                rows.append(r)
    with open(os.path.join(OUT, "sensitivity.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    names = [m[0] for m in MODELS]
    lines = ["| load | decision | KPI | " + " | ".join(names) + " |", "|" + "---|" * (3 + len(names))]
    for p in sorted({r["load"] for r in rows}):
        for dec in STEP:
            for k in sorted({r["kpi"] for r in rows if r["decision"] == dec}):
                cell = {r["model"]: r for r in rows if r["load"] == p and r["decision"] == dec and r["kpi"] == k}
                if "L2-legacy" not in cell:
                    continue
                ref = cell["L2-legacy"]
                parts = []
                for n in names:
                    c = cell[n]
                    s = f"{c['mean']:+.3g}"
                    if c["std"] > 0:
                        s += f" ± {c['std']:.2g}"
                    if n != "L2-legacy" and abs(ref["mean"]) > 2 * ref["std"]:
                        s += "" if (c["mean"] > 0) == (ref["mean"] > 0) else " ✗"
                    parts.append(s)
                resolved = abs(ref["mean"]) > 2 * ref["std"]
                lines.append(f"| {p} | {dec} | {k}{'' if resolved else ' (L2 unresolved)'} | " + " | ".join(parts) + " |")
    with open(os.path.join(OUT, "sensitivity.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
