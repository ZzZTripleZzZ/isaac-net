"""Relaxation error of L1D / QAD against the discrete L1 / QA levels as tau -> 0.

Binary sends (u < p, the same draws for every tau), so the tau = 0 model is the discrete level exactly (the table
records the tau = 0 difference). For each load and tau: relative error of the batch mean delay, mean AoI and
delivery ratio, and the largest per-robot absolute delay error (control steps). Also the mean-field KPIs (send
weight p instead of Bernoulli sends) at tau = 0.05, which show the gap between the fluid-of-expectations model
and the Bernoulli model it relaxes.

usage: python convergence.py [device] [E] [T]      -> results/convergence.csv, results/convergence.png
"""
import csv
import os
import sys

import torch

from common import OUT, DiffFluid, Scenario, discrete_rollout, make_discrete, rollout

TAUS = [0.3, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
LOADS = [0.2, 0.4, 0.6]
B = 12000.0


def main():
    dev = sys.argv[1] if len(sys.argv) > 1 else "cpu"
    E = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    T = int(sys.argv[3]) if len(sys.argv) > 3 else 200
    R = 8
    os.makedirs(OUT, exist_ok=True)
    sc = Scenario(E, R, T, dev, seed=0)
    snr = sc.snr()
    rows = []
    for mode in ("L1", "QA"):
        for p in LOADS:
            send = (sc.u < p).float()
            Bt = torch.full((E, R), B, device=dev)
            ref = discrete_rollout(make_discrete(mode, Bt, device=dev), T, sc.u < p, snr, warmup=T // 10)
            mf = rollout(DiffFluid(E, R, dev, mode=mode, tau=0.05), T, torch.full((E, R), p, device=dev), Bt,
                         snr_db=snr, warmup=T // 10)
            for tau in [0.0] + TAUS:
                r = rollout(DiffFluid(E, R, dev, mode=mode, tau=tau), T, send, Bt, snr_db=snr, warmup=T // 10,
                            time_axis={"send"})
                rel = lambda k: float(r[k] / ref[k] - 1)                                  # noqa: E731
                m = ref["delivered"] > 0
                row = dict(mode=mode, p=p, tau=tau, delay=float(r["mean_delay"]), ref_delay=float(ref["mean_delay"]),
                           err_delay=rel("mean_delay"), err_aoi=rel("mean_aoi"), err_delivery=rel("mean_delivery"),
                           max_robot_delay_err=float((r["delay"] - ref["delay"])[m].abs().max()),
                           meanfield_delay=float(mf["mean_delay"]), meanfield_aoi=float(mf["mean_aoi"]),
                           ref_aoi=float(ref["mean_aoi"]), ref_delivery=float(ref["mean_delivery"]))
                rows.append(row)
                print(row, flush=True)
    with open(os.path.join(OUT, "convergence.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    for ax, k, name in zip(axes, ("err_delay", "err_aoi", "err_delivery"), ("mean delay", "mean AoI", "delivery")):
        for mode, ls in (("L1", "-"), ("QA", "--")):
            for p in LOADS:
                rr = [r for r in rows if r["mode"] == mode and r["p"] == p and r["tau"] > 0]
                ax.loglog([r["tau"] for r in rr], [max(abs(r[k]), 1e-9) for r in rr], ls, marker="o", ms=3,
                          label=f"{mode}D p={p}")
        ax.set_xlabel("temperature tau")
        ax.set_ylabel(f"relative error, {name}")
        ax.grid(True, which="both", alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "convergence.png"), dpi=130)


if __name__ == "__main__":
    main()
