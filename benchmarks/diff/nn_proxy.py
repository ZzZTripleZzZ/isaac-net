"""Recipe run (L2D option b): fit the neural proxy (isaaclab_net.core.diff.proxy) to L2-legacy rollouts, report
held-out fit quality, and compare its autograd sensitivities with the finite differences of L2-legacy at the
sensitivity.py operating points.

Proxy KPIs are per robot; the batch KPIs here are plain means over robots (mean delay = mean of the per-robot
mean delays), and the L2-legacy finite differences are taken on the same plain means so that both sides measure
the same quantity.

usage: python nn_proxy.py [device] [n_batches] [E] [T] [n_seeds]   -> results/nn_proxy.json
"""
import json
import os
import statistics
import sys
import time

import torch

from common import OUT, Scenario, radio_snr_db
from isaaclab_net.core.diff import proxy
from sensitivity import B0, LOADS, STEP, base_value, inputs

DECS = ("p", "kB", "tx", "dist")


def plain(r):
    m = r["delivered"] > 0
    return {"delay": float(r["delay"][m].mean()), "aoi": float(r["aoi"].mean()), "delivery": float(r["delivery"].mean())}


def main():
    dev = sys.argv[1] if len(sys.argv) > 1 else "cpu"
    nb = int(sys.argv[2]) if len(sys.argv) > 2 else 24
    E = int(sys.argv[3]) if len(sys.argv) > 3 else 128
    T = int(sys.argv[4]) if len(sys.argv) > 4 else 200
    n_seeds = int(sys.argv[5]) if len(sys.argv) > 5 else 3
    R = 8
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    X, Y = proxy.collect(n_batches=nb, E=E, R=R, T=T, warmup=T // 10, device=dev, seed=0)
    Xv, Yv = proxy.collect(n_batches=max(2, nb // 6), E=E, R=R, T=T, warmup=T // 10, device=dev, seed=1)
    t_collect = time.time() - t0
    t0 = time.time()
    model = proxy.fit(X, Y, epochs=300, device=dev)
    t_fit = time.time() - t0
    with torch.no_grad():
        pv = model(Xv.to(dev)).cpu()
    r2 = {}
    for i, k in enumerate(("log1p_delay", "logit_delivery", "log_aoi")):
        m = torch.isfinite(Yv[:, i])
        y, yh = Yv[m, i], pv[m, i]
        r2[k] = float(1 - ((y - yh) ** 2).mean() / y.var())
    res = {"n_train": int(X.shape[0]), "n_val": int(Xv.shape[0]), "collect_s": t_collect, "fit_s": t_fit,
           "val_r2": r2, "sens": []}
    print(res, flush=True)
    model = model.double()
    for p in LOADS:
        for dec in DECS:
            fds, gs = [], []
            for seed in range(n_seeds):
                sc = Scenario(E, R, T, dev, seed=100 + seed)
                th0, h = base_value(dec, p), STEP[dec]
                vals = []
                for s in (+1, -1):
                    P, B, tx, pos = inputs(sc, p, th0 + s * h, dec, torch.float32)
                    vals.append(plain(sc.discrete_kpis("L2-legacy", P, B, tx_dbm=tx, pos=pos)))
                fds.append({k: (vals[0][k] - vals[1][k]) / (2 * h) for k in vals[0]})
                th = torch.tensor(th0, device=dev, dtype=torch.float64, requires_grad=True)
                P, B, tx, pos = inputs(sc, p, th, dec, torch.float64)
                snr = radio_snr_db(sc.radio, pos) + (0.0 if tx is None else tx - 23.0)
                kp = proxy.predict(model, P, B, snr)
                gs.append({k: float(torch.autograd.grad(kp[k].mean(), th, retain_graph=True)[0]) for k in kp})
            for k in fds[0]:
                a, b = [f[k] for f in fds], [g[k] for g in gs]
                row = {"load": p, "decision": dec, "kpi": k, "l2_fd": statistics.fmean(a), "l2_fd_std": statistics.stdev(a),
                       "proxy_grad": statistics.fmean(b), "proxy_grad_std": statistics.stdev(b)}
                res["sens"].append(row)
                print(row, flush=True)
    res["B0"] = B0
    with open(os.path.join(OUT, "nn_proxy.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
