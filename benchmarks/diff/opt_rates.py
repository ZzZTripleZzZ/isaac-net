"""Demo: gradient-based choice of per-robot send probability, message size and transmit power, against random
search, on a batch of 64 independent envs (64 separate problems solved in parallel).

Problem (per env, R = 8 robots). Robot r must move a data stream of d_r bytes per control step (d_r drawn in
1..6 kB). It sends a message with probability p_r each step; a message of B_r bytes carries B_r - H payload
bytes (H = 300 B of headers), at transmit power tx_r in [0, 23] dBm. The cost is

    J = mean AoI + mean delay                       (control steps, delivered-weighted delay)
        + LAMBDA_E * energy                          (transmit joules per robot per step)
        + LAMBDA_S * mean_r relu(1 - goodput_r / d_r) (goodput = p_r * delivery_r * (B_r - H))

Frequent small messages lower the AoI but pay the header, rare large ones raise the queueing delay, and power buys
spectral efficiency at an energy price.

Methods (same starting point and the same simulator for scoring):
    grad    Adam on unconstrained parameters through L1D (tau = 0.05, relaxed-Bernoulli sends with fresh uniforms,
            lam = 0.2), ITERS iterations
    random  per env, N_RAND random decisions (p, B, tx drawn uniformly per robot) scored on the true cost, the
            best kept
Scoring ("true" cost): the exact discrete L1 (L1D at tau = 0 is the discrete L1 level, see tests/test_diff.py),
Bernoulli sends from held-out uniforms, T steps. The chosen decisions are also run on L2-legacy (delay, AoI and
delivery; that engine has no energy model), to show how the choice transfers to the slot-level engine.

usage: python opt_rates.py [device] [E] [T]      -> results/opt_rates.json, results/opt_rates.png
"""
import json
import math
import os
import sys
import time

import torch

from common import OUT, DiffFluid, Scenario, discrete_rollout, make_discrete, radio_snr_db, relaxed_bernoulli, rollout

H = 300.0
LAMBDA_E, LAMBDA_S = 50.0, 10.0
P_LO, P_HI, B_LO, B_HI, TX_LO, TX_HI = 0.02, 1.0, 400.0, 40000.0, 0.0, 23.0
ITERS, N_RAND = 150, 256


def cost(r, p, B, d):
    good = p * r["delivery"] * (B - H)
    short = torch.relu(1 - good / d).mean(-1)                                           # [E]
    E = p.shape[0]
    delay_e = r["delay_sum_e"] / r["delivered_e"].clamp(min=1e-9) if "delay_sum_e" in r else r["delay"].mean(-1)
    parts = {"aoi": r["aoi"].mean(-1), "delay": delay_e, "energy": LAMBDA_E * r["energy_j"].mean(-1),
             "shortfall": LAMBDA_S * short}
    return sum(parts.values()), {k: v.reshape(E) for k, v in parts.items()}


def per_env(r):
    """Add per-env delivered-weighted delay pieces to a rollout() result."""
    r = dict(r)
    r["delivered_e"] = r["delivered"].sum(-1)
    r["delay_sum_e"] = (r["delay"] * r["delivered"]).sum(-1)
    return r


def decode(z):
    zp, zb, zt = z.unbind(-1)
    p = P_LO + (P_HI - P_LO) * torch.sigmoid(zp)
    B = torch.exp(math.log(B_LO) + (math.log(B_HI) - math.log(B_LO)) * torch.sigmoid(zb))
    tx = TX_LO + (TX_HI - TX_LO) * torch.sigmoid(zt)
    return p, B, tx


@torch.no_grad()
def true_cost(snr, u, p, B, tx, d, T):
    """Discrete L1 (L1D at tau = 0) with Bernoulli sends u < p; returns cost [E] and parts."""
    E, R = p.shape
    net = DiffFluid(E, R, snr.device, tau=0.0)
    r = per_env(rollout(net, T, (u < p).float(), B, snr_db=snr, tx_dbm=tx, warmup=T // 10, time_axis={"send"}))
    return cost(r, p, B, d)


def main():
    dev = sys.argv[1] if len(sys.argv) > 1 else "cpu"
    E = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    T = int(sys.argv[3]) if len(sys.argv) > 3 else 150
    R = 8
    os.makedirs(OUT, exist_ok=True)
    sc = Scenario(E, R, T, dev, seed=11)
    g = torch.Generator().manual_seed(12)
    d = (1000 + 5000 * torch.rand(E, R, generator=g)).to(dev)
    snr0 = radio_snr_db(sc.radio, sc.pos)                     # at 23 dBm
    u_eval = torch.rand(T, E, R, generator=g).to(dev)
    res = {"E": E, "R": R, "T": T}

    # starting point: p = 0.5, B = payload for the demand + H, full power
    p0 = torch.full((E, R), 0.5, device=dev)
    B0 = (d / p0 + H).clamp(B_LO, B_HI)
    tx0 = torch.full((E, R), 23.0, device=dev)
    c0, parts0 = true_cost(snr0, u_eval, p0, B0, tx0, d, T)
    res["start"] = {"cost": float(c0.mean()), **{k: float(v.mean()) for k, v in parts0.items()}}
    inv = lambda y, lo, hi: torch.logit(((y - lo) / (hi - lo)).clamp(1e-4, 1 - 1e-4))          # noqa: E731
    z = torch.stack([inv(p0, P_LO, P_HI), inv(torch.log(B0), math.log(B_LO), math.log(B_HI)),
                     inv(tx0, TX_LO, TX_HI)], -1).requires_grad_()

    # ---- gradient descent through L1D
    opt = torch.optim.Adam([z], lr=0.1)
    hist, t0 = [], time.time()
    for it in range(ITERS):
        p, B, tx = decode(z)
        u = torch.rand(T, E, R, device=dev)
        net = DiffFluid(E, R, dev, tau=0.05)
        r = per_env(rollout(net, T, relaxed_bernoulli(p.expand(T, E, R), u, 0.2), B, snr_db=snr0, tx_dbm=tx,
                            warmup=T // 10, time_axis={"send"}))
        c, _ = cost(r, p, B, d)
        opt.zero_grad()
        c.sum().backward()
        opt.step()
        if it % 10 == 9 or it == ITERS - 1:
            with torch.no_grad():
                pc, Bc, txc = decode(z)
                ct, _ = true_cost(snr0, u_eval, pc, Bc, txc, d, T)
            hist.append({"iter": it + 1, "rollouts": 2 * (it + 1), "wall_s": time.time() - t0,
                         "train_cost": float(c.detach().mean()), "true_cost": float(ct.mean())})
            print("grad", hist[-1], flush=True)
    p_g, B_g, tx_g = (v.detach() for v in decode(z))
    cg, parts_g = true_cost(snr0, u_eval, p_g, B_g, tx_g, d, T)
    res["grad"] = {"cost": float(cg.mean()), **{k: float(v.mean()) for k, v in parts_g.items()}, "curve": hist,
                   "wall_s": hist[-1]["wall_s"], "p_mean": float(p_g.mean()), "B_mean": float(B_g.mean()),
                   "tx_mean": float(tx_g.mean()), "win_vs_random_envs": None}

    # ---- random search: N_RAND candidates per env, scored on the true cost with training uniforms
    t0 = time.time()
    u_train = torch.rand(T, E, R, generator=g).to(dev)
    best_c = torch.full((E,), float("inf"), device=dev)
    best = [p0.clone(), B0.clone(), tx0.clone()]
    rhist = []
    chunk = 32
    for i0 in range(0, N_RAND, chunk):
        n = min(chunk, N_RAND - i0)
        pr = P_LO + (P_HI - P_LO) * torch.rand(n, E, R, device=dev)
        Br = torch.exp(math.log(B_LO) + (math.log(B_HI) - math.log(B_LO)) * torch.rand(n, E, R, device=dev))
        txr = TX_LO + (TX_HI - TX_LO) * torch.rand(n, E, R, device=dev)
        flat = lambda x: x.reshape(n * E, R)                                                    # noqa: E731
        cr, _ = true_cost(snr0.repeat(n, 1), u_train.repeat(1, n, 1), flat(pr), flat(Br), flat(txr), d.repeat(n, 1), T)
        cr = cr.view(n, E)
        cmin, imin = cr.min(0)
        upd = cmin < best_c
        ar = torch.arange(E, device=dev)
        for j, x in enumerate((pr, Br, txr)):
            best[j] = torch.where(upd[:, None], x[imin, ar], best[j])
        best_c = torch.minimum(best_c, cmin)
        ce, _ = true_cost(snr0, u_eval, *best, d, T)
        rhist.append({"candidates": i0 + n, "rollouts": i0 + n, "wall_s": time.time() - t0, "true_cost": float(ce.mean())})
        print("random", rhist[-1], flush=True)
    cr_, parts_r = true_cost(snr0, u_eval, *best, d, T)
    res["random"] = {"cost": float(cr_.mean()), **{k: float(v.mean()) for k, v in parts_r.items()}, "curve": rhist,
                     "wall_s": rhist[-1]["wall_s"], "p_mean": float(best[0].mean()), "B_mean": float(best[1].mean()),
                     "tx_mean": float(best[2].mean())}
    res["grad"]["win_vs_random_envs"] = float((cg < cr_).float().mean())

    # ---- transfer to L2-legacy (no energy model there): delay, AoI, delivery of the three decision sets
    res["l2_legacy"] = {}
    for name, (p, B, tx) in (("start", (p0, B0, tx0)), ("grad", (p_g, B_g, tx_g)), ("random", tuple(best))):
        net = make_discrete("L2-legacy", B, device=dev, seed=5)
        r = discrete_rollout(net, T, u_eval < p, snr0 + (tx - 23.0), warmup=T // 10)
        good = (p * r["delivery"] * (B - H) / d).clamp(max=1)
        res["l2_legacy"][name] = {"delay": float(r["mean_delay"]), "aoi": float(r["mean_aoi"]),
                                  "delivery": float(r["mean_delivery"]), "demand_met": float(good.mean())}
        print("L2-legacy", name, res["l2_legacy"][name], flush=True)
    with open(os.path.join(OUT, "opt_rates.json"), "w") as f:
        json.dump(res, f, indent=1)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.3))
    for ax, key in zip(axes, ("rollouts", "wall_s")):
        ax.plot([h[key] for h in hist], [h["true_cost"] for h in hist], "o-", ms=3, label="gradient (L1D)")
        ax.plot([h[key] for h in rhist], [h["true_cost"] for h in rhist], "s--", ms=3, label="random search")
        ax.axhline(res["start"]["cost"], color="gray", lw=0.8, label="start")
        ax.set_xlabel("simulated rollouts" if key == "rollouts" else "wall time [s]")
        ax.set_ylabel("true cost (discrete L1, held-out sends)")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "opt_rates.png"), dpi=130)


if __name__ == "__main__":
    main()
