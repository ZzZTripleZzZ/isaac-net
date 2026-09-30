"""Toy: communication-aware placement. Each robot has a task point c_r and may stand anywhere; standing away from
it costs ALPHA * (distance / 10 m)^2, while its position sets its SNR through the radio model (log-distance path
loss plus the spatially correlated shadowing field of proto.netsim.Radio). Robots send 20 kB messages with
probability 0.5 per control step, which loads the cell, so poor positions raise the delay and lose messages.

    J = mean delay + mean AoI + BETA * (1 - delivery) + ALPHA * mean (|x_r - c_r| / 10 m)^2

Methods, on 64 envs x 8 robots in parallel:
    grad    Adam on the positions through L1D (tau = 0.05, mean-field sends), ITERS iterations
    random  per env, N_RAND candidate layouts (each robot moved uniformly within 25 m of its task point), the
            best by the true cost
    stay    every robot at its task point (the start)
True cost: discrete L1 with Bernoulli sends (held-out uniforms). The layouts are also run on L2-legacy (delay,
AoI, delivery). A position the Radio sees within 1 m of the gNB is clamped to 1 m by the path-loss model.

usage: python placement.py [device] [E] [T]      -> results/placement.json, results/placement.png
"""
import json
import os
import sys
import time

import torch

from common import OUT, DiffFluid, Scenario, discrete_rollout, make_discrete, radio_snr_db, rollout

P_SEND, B_MSG = 0.5, 20000.0
ALPHA, BETA = 0.5, 20.0
ITERS, N_RAND, RADIUS = 150, 256, 25.0


def cost(r, x, c):
    move = (((x - c).norm(dim=-1) / 10.0) ** 2).mean(-1)
    dl = (r["delay"] * r["delivered"]).sum(-1) / r["delivered"].sum(-1).clamp(min=1e-9)
    deliv = r["delivered"].sum(-1) / r["offered"].sum(-1).clamp(min=1e-9)
    parts = {"delay": dl, "aoi": r["aoi"].mean(-1), "loss": BETA * (1 - deliv), "move": ALPHA * move}
    return sum(parts.values()), parts


@torch.no_grad()
def true_cost(sc, x, c, u, T, radio=None):
    E, R = x.shape[:2]
    snr = radio_snr_db(radio or sc.radio, x)
    net = DiffFluid(E, R, x.device, tau=0.0)
    r = rollout(net, T, (u < P_SEND).float(), torch.full((E, R), B_MSG, device=x.device), snr_db=snr,
                warmup=T // 10, time_axis={"send"})
    return cost(r, x, c)


class _Tiled:
    """A Radio view repeated n times along the env axis (for scoring n candidate layouts per env at once)."""

    def __init__(self, radio, n):
        self.k, self.phi, self.amp = radio.k.repeat(n, 1, 1), radio.phi.repeat(n, 1), radio.amp


def main():
    dev = sys.argv[1] if len(sys.argv) > 1 else "cpu"
    E = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    T = int(sys.argv[3]) if len(sys.argv) > 3 else 150
    R = 8
    os.makedirs(OUT, exist_ok=True)
    sc = Scenario(E, R, T, dev, seed=21, r_min=40.0, r_max=140.0)
    c = sc.pos.clone()
    g = torch.Generator().manual_seed(22)
    u_eval = torch.rand(T, E, R, generator=g).to(dev)
    res = {"E": E, "R": R, "T": T}
    c0, parts0 = true_cost(sc, c, c, u_eval, T)
    res["stay"] = {"cost": float(c0.mean()), **{k: float(v.mean()) for k, v in parts0.items()}}
    print("stay", res["stay"], flush=True)

    x = c.clone().requires_grad_()
    opt = torch.optim.Adam([x], lr=1.0)
    hist, t0 = [], time.time()
    for it in range(ITERS):
        net = DiffFluid(E, R, dev, tau=0.05)
        r = rollout(net, T, torch.full((E, R), P_SEND, device=dev), torch.full((E, R), B_MSG, device=dev),
                    snr_db=radio_snr_db(sc.radio, x), warmup=T // 10)
        cst, _ = cost(r, x, c)
        opt.zero_grad()
        cst.sum().backward()
        opt.step()
        if it % 10 == 9 or it == ITERS - 1:
            ct, _ = true_cost(sc, x.detach(), c, u_eval, T)
            hist.append({"iter": it + 1, "rollouts": 2 * (it + 1), "wall_s": time.time() - t0,
                         "train_cost": float(cst.detach().mean()), "true_cost": float(ct.mean())})
            print("grad", hist[-1], flush=True)
    xg = x.detach()
    cg, parts_g = true_cost(sc, xg, c, u_eval, T)
    res["grad"] = {"cost": float(cg.mean()), **{k: float(v.mean()) for k, v in parts_g.items()}, "curve": hist,
                   "moved_m": float((xg - c).norm(dim=-1).mean()),
                   "snr_gain_db": float((radio_snr_db(sc.radio, xg) - radio_snr_db(sc.radio, c)).mean())}

    t0 = time.time()
    u_train = torch.rand(T, E, R, generator=g).to(dev)
    best_c, best = c0.clone(), c.clone()
    rhist, chunk = [], 32
    for i0 in range(0, N_RAND, chunk):
        n = min(chunk, N_RAND - i0)
        ang = 2 * torch.pi * torch.rand(n, E, R, device=dev)
        rad = RADIUS * torch.sqrt(torch.rand(n, E, R, device=dev))
        xr = c + torch.stack([rad * torch.cos(ang), rad * torch.sin(ang)], -1)
        cr, _ = true_cost(sc, xr.reshape(n * E, R, 2), c.repeat(n, 1, 1), u_train.repeat(1, n, 1), T,
                          radio=_Tiled(sc.radio, n))
        cr = cr.view(n, E)
        cmin, imin = cr.min(0)
        upd = cmin < best_c
        best = torch.where(upd[:, None, None], xr[imin, torch.arange(E, device=dev)], best)
        best_c = torch.minimum(best_c, cmin)
        ce, _ = true_cost(sc, best, c, u_eval, T)
        rhist.append({"candidates": i0 + n, "rollouts": i0 + n, "wall_s": time.time() - t0, "true_cost": float(ce.mean())})
        print("random", rhist[-1], flush=True)
    cr_, parts_r = true_cost(sc, best, c, u_eval, T)
    res["random"] = {"cost": float(cr_.mean()), **{k: float(v.mean()) for k, v in parts_r.items()}, "curve": rhist,
                     "moved_m": float((best - c).norm(dim=-1).mean())}
    res["grad"]["win_vs_random_envs"] = float((cg < cr_).float().mean())

    res["l2_legacy"] = {}
    for name, xx in (("stay", c), ("grad", xg), ("random", best)):
        net = make_discrete("L2-legacy", torch.full((E, R), B_MSG, device=dev), device=dev, seed=5)
        r = discrete_rollout(net, T, u_eval < P_SEND, radio_snr_db(sc.radio, xx), warmup=T // 10)
        res["l2_legacy"][name] = {"delay": float(r["mean_delay"]), "aoi": float(r["mean_aoi"]),
                                  "delivery": float(r["mean_delivery"])}
        print("L2-legacy", name, res["l2_legacy"][name], flush=True)
    with open(os.path.join(OUT, "placement.json"), "w") as f:
        json.dump(res, f, indent=1)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    ax = axes[0]
    ax.plot([h["rollouts"] for h in hist], [h["true_cost"] for h in hist], "o-", ms=3, label="gradient (L1D)")
    ax.plot([h["rollouts"] for h in rhist], [h["true_cost"] for h in rhist], "s--", ms=3, label="random search")
    ax.axhline(res["stay"]["cost"], color="gray", lw=0.8, label="stay at task point")
    ax.set_xlabel("simulated rollouts")
    ax.set_ylabel("true cost (discrete L1)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax = axes[1]                        # env 0: shadowing map, task points and optimized positions
    e = 0
    lim = float(c[e].abs().max()) + 30
    gx = torch.linspace(-lim, lim, 160, device=dev)
    XX, YY = torch.meshgrid(gx, gx, indexing="xy")
    grid = torch.stack([XX, YY], -1).reshape(1, -1, 2)
    view = type("V", (), {"k": sc.radio.k[e:e + 1], "phi": sc.radio.phi[e:e + 1], "amp": sc.radio.amp})
    snr_map = radio_snr_db(view, grid).reshape(160, 160).cpu()
    im = ax.imshow(snr_map, origin="lower", extent=[-lim, lim, -lim, lim], cmap="viridis")
    fig.colorbar(im, ax=ax, label="SNR at 23 dBm [dB]")
    ax.scatter(c[e, :, 0].cpu(), c[e, :, 1].cpu(), c="white", marker="x", s=25, label="task point")
    ax.scatter(xg[e, :, 0].cpu(), xg[e, :, 1].cpu(), c="red", s=12, label="gradient")
    ax.scatter([0], [0], c="black", marker="^", s=30, label="gNB")
    ax.legend(fontsize=7, loc="upper right")
    ax.set_title("env 0")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "placement.png"), dpi=130)


if __name__ == "__main__":
    main()
