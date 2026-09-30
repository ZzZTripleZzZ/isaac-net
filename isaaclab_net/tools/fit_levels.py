"""Fit the surrogate levels TR, GE, QA and NN from L2 or L2-legacy rollouts, and save one parameter file.

    python -m isaaclab_net.tools.fit_levels --source L2-legacy --task T1 --device cuda
    python -m isaaclab_net.tools.fit_levels --source L2 --preset netslot_compat --task T1 --out ~/fits/T1_nr.pt

The fitted file goes outside the repository (default ~/.cache/isaaclab_net/levels/<source>_<task>.pt, or
$ISAACLAB_NET_LEVELS_DIR); the tool refuses a path inside the source tree, and fitted files are never committed.
Load it with make_engine("NN", E, R, device, params=<path>) (any of TR, GE, QA, NN). A JSON summary of the fit
(sizes, fit statistics, the QA grid, held-out fit quality) is written next to it.

Data. The source level runs the example fleet task (isaaclab_net.examples.fleet_task.FleetEnv) under a behavior
policy: every env draws a log-uniform send rate over [0.01, 1] and a random large-frame fraction per episode, and
half of the envs send more ([0.1, 0.3, 0.6] for nothing / small / large) while a hazard siren is on. A
RolloutLogger wraps the engine through its public API only (submit / step dicts), so any level can be the
source, and records every resolved frame with its send-time features. Only frames captured at or before step
T - TIMEOUT - 1 are kept, so no outcome is right-censored. Delays are in control steps from the capture step.

Fits.
  TR  every (episode, env) of the training rollouts is one trace. For trace j, class c and step t the file stores
      the slice of outcomes of the class-c frames captured at t* = the step of trace j nearest to t with at least
      one class-c frame (ties to the earlier step). A trace with no class-c frame at all falls back to the pooled
      class-c outcomes.
  GE  thresholding and counting, not EM. The state of env e at step t comes from its end-of-step backlog b (bytes
      queued in the env): 0 if b = 0, 1 if 0 < b <= theta, 2 otherwise, theta = the median of the positive b.
      Transition matrix from consecutive-step counts (+1 smoothing), initial distribution from step 0; per
      (state, class) the 101-point empirical quantile function of the delivered delay and the loss rate, with
      cells of fewer than 30 frames backing off to the class-pooled values.
  NN  an MLP (3 x 128 SiLU) with a drop logit (BCE on all frames) and 32 monotone quantiles of the log delay of
      delivered frames (pinball loss), on the 13 send-time features of levels.surrogates.nn_features.
  QA  eta over a grid and pf on / off, picked by the frame-weighted per-cell 1-Wasserstein distance of the
      censored delay (a lost frame counts as TIMEOUT) between QA and the held-out source rollouts, both under the
      behavior policy. Cells are the L05 lookup cells (backlogged robots x SNR x class).
Held-out fit quality (teacher-forced, true send-time features, GE with its true state) goes into the JSON.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import torch

from ..core import config as _config
from ..core.engine import make_engine
from ..core.levels import DelayNet, nn_features
from ..core.levels.surrogates import H
from ..core.proto import netsim as _ns
from ..examples.fleet_task import TASK_SIZES, FleetEnv

TIMEOUT, F = _ns.TIMEOUT, _ns.F
FEATURES = ("cls", "snr", "own", "ownb", "nact", "totb", "to") + tuple(f"h{k}" for k in range(H))
PRESETS = ("default", "netslot_compat", "lena_like", "srsran_like", "oai_like")
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ----------------------------------------------------------------------------- logging
class RolloutLogger:
    """Engine proxy for FleetEnv (legacy calls add_frames / step(t, snr, hid) / reset / queued) that drives the
    wrapped engine through submit / step dicts and records each resolved frame with its send-time features:
      cls, snr        traffic class and the SNR passed at submit
      own, ownb       frames queued ahead in the robot's FIFO, and their nominal bytes
      nact, totb      robots with a non-empty FIFO after the submit, nominal bytes queued in the env (with arrivals)
      to              EWMA (alpha 0.3) of timed-out frames per robot per step in the env
      h0 .. h3        mean delivered delay of the env in each of the last 4 steps (carried over empty steps)
    plus cap, env, ep (episode) and the outcome delay (inf if lost). Nominal (not remaining) bytes, so that the NN
    level, which serves no bytes, computes the same features. backlog records the end-of-step env backlog."""

    def __init__(self, net, sizes, cap_max):
        self.net, self.E, self.R = net, net.E, net.R
        self.dev = torch.device(net.dev) if not isinstance(net.dev, torch.device) else net.dev
        self.sizes = torch.tensor(sizes, dtype=torch.float32, device=self.dev)
        self.cap_max = cap_max
        self.W = TIMEOUT + 2                            # a frame lives at most TIMEOUT steps
        E, R, d = self.E, self.R, self.dev
        self.ring = {k: torch.zeros(E, R, self.W, device=d) for k in FEATURES}
        self.hist = torch.zeros(E, H, device=d)
        self.to = torch.zeros(E, device=d)
        self.nomq = torch.zeros(E, R, device=d)
        self.ep, self._stepped = -1, True
        self.frames, self.backlog = [], []

    def reset(self, env_ids=None):
        assert env_ids is None, "the logger drives full-episode resets"
        self.net.reset()
        self.hist.zero_()
        self.to.zero_()
        self.nomq.zero_()
        if self._stepped:
            self.ep += 1
            self._stepped = False

    def queued(self):
        return self.net.queued()

    def add_frames(self, t, send, det, hid, snr):
        own = self.net.queued()
        acc = self.net.submit(t, _ns.Requests(send, det, hid), snr)
        size_new = torch.where(acc, self.sizes[(send - 1).clamp(min=0)], torch.zeros_like(self.nomq))
        nact = ((own + acc.long()) > 0).sum(-1)
        totb = (self.nomq + size_new).sum(-1)
        E, R = self.E, self.R
        vals = {"cls": send.float(), "snr": snr.float(), "own": own.float(), "ownb": self.nomq.clone(),
                "nact": nact[:, None].expand(E, R).float(), "totb": totb[:, None].expand(E, R),
                "to": self.to[:, None].expand(E, R)}
        for k in range(H):
            vals[f"h{k}"] = self.hist[:, k, None].expand(E, R)
        slot = int(t) % self.W
        for k, v in vals.items():
            self.ring[k][..., slot] = torch.where(acc, v, self.ring[k][..., slot])
        self.nomq += size_new

    def step(self, t, snr, hid):
        out = self.net.step(t, snr)
        self._stepped = True
        gone = out["delivered"] | out["timed_out"]
        lost_now = out["timed_out"]
        if "dropped" in out:
            gone = gone | out["dropped"]
            lost_now = lost_now | out["dropped"]
        cap = out["cap"]
        delay = torch.where(out["delivered"], out["delay"], torch.full_like(out["delay"], math.inf))
        e, r, f = (gone & (cap >= 0) & (cap <= self.cap_max)).nonzero(as_tuple=True)
        if e.numel():
            slot = cap[e, r, f] % self.W
            rec = {k: self.ring[k][e, r, slot].cpu() for k in FEATURES}
            rec.update(cap=cap[e, r, f].cpu(), env=e.cpu(), ep=torch.full((e.numel(),), self.ep),
                       delay=delay[e, r, f].cpu())
            self.frames.append(rec)
        dl = out["delivered"]
        n = dl.sum((1, 2))
        dsum = torch.where(dl, out["delay"], torch.zeros_like(out["delay"])).sum((1, 2))
        newest = torch.where(n > 0, dsum / n.clamp(min=1), self.hist[:, 0])
        self.hist = torch.cat([newest[:, None], self.hist[:, :-1]], 1)
        self.to = 0.7 * self.to + 0.3 * lost_now.sum((1, 2)).float() / self.R
        nom_gone = torch.where(gone & (cap >= 0), self.sizes[(out["cls"] - 1).clamp(min=0)],
                               torch.zeros_like(out["delay"]))
        self.nomq = (self.nomq - nom_gone.sum(-1)).clamp(min=0)
        self.backlog.append((self.ep, int(t), out["queue_bytes"].sum(-1).float().cpu()))
        return out["newest"], out["det_env"]

    def table(self, T):
        """All logged frames as one dict of 1-D tensors (ep renumbered 0..n_ep-1), and the backlog B [n_ep,T,E]."""
        fr = {k: torch.cat([x[k] for x in self.frames]) for k in self.frames[0]}
        for k in ("cls", "own", "nact"):
            fr[k] = fr[k].round().long()
        eps = sorted({ep for ep, _, _ in self.backlog})
        lut = torch.full((max(eps) + 1,), -1, dtype=torch.long)
        lut[torch.tensor(eps)] = torch.arange(len(eps))
        fr["ep"] = lut[fr["ep"]]
        fr["drop"] = ~torch.isfinite(fr["delay"])
        B = torch.zeros(len(eps), T, self.E)
        for ep, t, b in self.backlog:
            B[lut[ep], t] = b
        return fr, B


@torch.no_grad()
def behavior(env, net, episodes, mode="fit"):
    """Drive FleetEnv for `episodes` episodes. fit: log-uniform per-env send rate over [0.01, 1], random large
    fraction, siren boost in half the envs. greedy: large frames w.p. 0.5. polite: a small frame w.p. 0.5 only
    when the robot's own queue is empty. Uses the global torch RNG."""
    E, R, dev = env.E, env.R, env.dev
    for _ in range(episodes):
        env.reset()
        rate = torch.exp(math.log(0.01) - math.log(0.01) * torch.rand(E, device=dev))
        large = torch.rand(E, device=dev)
        probs = torch.stack([1 - rate, rate * (1 - large), rate * large], -1)
        siren_boost = torch.rand(E, device=dev) < 0.5
        boost = torch.tensor([0.1, 0.3, 0.6], device=dev)
        for _ in range(env.T):
            to_goal = env.goal - env.pos
            vel = to_goal / to_goal.norm(dim=-1, keepdim=True).clamp(min=1e-3) + 0.5 * torch.randn_like(to_goal)
            if mode == "fit":
                siren = (env.h_on & (env.t - env.h_start < env.SIREN) & siren_boost)[:, None, None]
                p = torch.where(siren, boost, probs[:, None, :]).expand(E, R, 3)
                send = torch.distributions.Categorical(probs=p).sample()
            elif mode == "greedy":
                send = 2 * (torch.rand(E, R, device=dev) < 0.5).long()
            else:
                send = ((torch.rand(E, R, device=dev) < 0.5) & (net.queued() == 0)).long()
            env.step(vel.clamp(-1, 1), send)


def source_config(source, preset):
    cfg = _config.NRConfig() if preset in (None, "default") else getattr(_config, preset)()
    want = {"frame_buffer": F, "timeout_steps": TIMEOUT, "control_step_ms": 100.0}
    bad = {k: getattr(cfg, k) for k, v in want.items() if getattr(cfg, k) != v}
    if bad:
        raise ValueError(f"the surrogate levels run with frame_buffer={F}, timeout_steps={TIMEOUT} and 100 ms "
                         f"steps; preset {preset!r} has {bad}")
    return cfg


def log_rollouts(source, E, R, episodes, seed, device, sizes, config=None, T=300, backend="reference",
                 mode="fit", params=None):
    """Roll out `source` (any make_engine level) under the behavior policy; returns (frames, B, seconds/step)."""
    torch.manual_seed(seed)
    net = make_engine(source, E, R, device, config, backend, sizes=sizes, params=params, seed=seed)
    log = RolloutLogger(net, sizes, cap_max=T - TIMEOUT - 1)
    env = FleetEnv(E, R, log, torch.device(device))
    env.T = T
    t0 = time.time()
    behavior(env, log, episodes, mode)
    sps = (time.time() - t0) / (episodes * T)
    fr, B = log.table(T)
    return fr, B, sps


# ----------------------------------------------------------------------------- fits
def fit_tr(fr, E, n_ep, T):
    J = n_ep * E
    j = fr["ep"] * E + fr["env"]
    c = fr["cls"] - 1
    t = fr["cap"]
    key = (j * 2 + c) * T + t
    order = key.argsort(stable=True)
    delay = fr["delay"][order]
    counts = torch.bincount(key, minlength=J * 2 * T).view(J, 2, T)
    starts = (counts.flatten().cumsum(0) - counts.flatten()).view(J, 2, T)
    has = counts > 0
    idx = torch.arange(T).expand(J, 2, T)
    big = 10 ** 6
    prev = torch.where(has, idx, torch.full_like(idx, -big)).cummax(-1).values
    nxt = torch.where(has, idx, torch.full_like(idx, big)).flip(-1).cummin(-1).values.flip(-1)
    tstar = torch.where((idx - prev) <= (nxt - idx), prev, nxt).clamp(0, T - 1)
    valid = has.any(-1, keepdim=True)
    start = starts.gather(-1, tstar)
    cnt = counts.gather(-1, tstar) * valid
    corder = c.argsort(stable=True)
    pool = fr["delay"][corder]
    pool_cnt = torch.bincount(c, minlength=2)
    pool_start = pool_cnt.cumsum(0) - pool_cnt
    info = {"traces": J, "frames": int(delay.numel()),
            "frac_trace_class_empty": float((~valid).float().mean()),
            "mean_abs_time_shift_steps": float(((tstar - idx).abs().float() * valid).sum()
                                               / (valid.sum() * T).clamp(min=1))}
    return {"delay": delay, "start": start, "cnt": cnt, "pool": pool, "pool_start": pool_start,
            "pool_cnt": pool_cnt}, info


def ge_states(B, theta):
    return torch.where(B <= 1.0, 0, torch.where(B <= theta, 1, 2))


def fit_ge(fr, B, min_n=30):
    pos = B[B > 1.0]
    theta = float(pos.median()) if pos.numel() else 1.0
    s = ge_states(B, theta)                                # [n_ep, T, E]
    K = 3
    a, b = s[:, :-1].flatten(), s[:, 1:].flatten()
    Pc = torch.bincount(a * K + b, minlength=K * K).view(K, K).float() + 1.0
    P = Pc / Pc.sum(1, keepdim=True)
    pi0 = torch.bincount(s[:, 0].flatten(), minlength=K).float() + 1.0
    pi0 = pi0 / pi0.sum()
    fs = s[fr["ep"], fr["cap"], fr["env"]]
    mu, sig, p, n = (torch.zeros(K, 2) for _ in range(4))
    q = torch.zeros(K, 2, 101)
    qs = torch.linspace(0, 1, 101)
    all_dl = fr["delay"][~fr["drop"]].clamp(min=1e-3).log()
    for c in range(2):
        mc = fr["cls"] == c + 1
        dl_c = fr["delay"][mc & ~fr["drop"]].clamp(min=1e-3).log()
        if dl_c.numel() == 0:
            dl_c = all_dl if all_dl.numel() else torch.zeros(1)
        p_c = fr["drop"][mc].float().mean() if mc.any() else fr["drop"].float().mean()
        for k in range(K):
            m = mc & (fs == k)
            dl = fr["delay"][m & ~fr["drop"]].clamp(min=1e-3).log()
            src = dl if dl.numel() >= min_n else dl_c
            mu[k, c], sig[k, c] = src.mean(), (src.std() if src.numel() > 1 else torch.tensor(0.0)).clamp(min=0.05)
            q[k, c] = src.exp().quantile(qs)
            p[k, c] = fr["drop"][m].float().mean() if m.sum() >= min_n else p_c
            n[k, c] = m.sum()
    occ = torch.bincount(s.flatten(), minlength=K).float()
    info = {"theta_bytes": theta, "P": P.tolist(), "pi0": pi0.tolist(), "occupancy": (occ / occ.sum()).tolist(),
            "mean_sojourn_steps": (1 / (1 - P.diag())).tolist(), "loss": p.tolist(), "frames": n.tolist()}
    return {"P": P, "pi0": pi0, "mu": mu, "sig": sig, "p": p, "q": q, "theta": torch.tensor(theta)}, info


def nn_x(fr, R):
    hist = torch.stack([fr[f"h{k}"] for k in range(H)], -1)
    return nn_features(fr["cls"], fr["snr"], fr["own"], fr["ownb"], fr["nact"], fr["totb"], fr["to"], hist, R)


def fit_nn(fr, R, device, Q=32, h=128, steps=6000, bs=8192, lr=2e-3, seed=0):
    torch.manual_seed(seed)
    dev = torch.device(device)
    X = nn_x(fr, R)
    xm, xs = X.mean(0), X.std(0).nan_to_num(1.0).clamp(min=1e-3)
    Xn = ((X - xm) / xs).to(dev)
    drop = fr["drop"].float().to(dev)
    logd = torch.where(fr["drop"], torch.zeros_like(fr["delay"]), fr["delay"].clamp(min=1e-3).log()).to(dev)
    net = DelayNet(X.shape[1], Q, h).to(dev)
    taus = ((torch.arange(Q) + 0.5) / Q).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(1, steps))
    N = Xn.shape[0]
    t0 = time.time()
    l_drop = l_q = torch.zeros(())
    for _ in range(steps):
        ix = torch.randint(0, N, (min(bs, N),), device=dev)
        logit, q = net(Xn[ix])
        l_drop = torch.nn.functional.binary_cross_entropy_with_logits(logit, drop[ix])
        dm = drop[ix] < 0.5
        err = logd[ix][dm, None] - q[dm]
        l_q = torch.maximum(taus * err, (taus - 1) * err).mean() if err.numel() else torch.zeros((), device=dev)
        loss = l_drop + l_q
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    params = {"state": {k: v.detach().cpu() for k, v in net.state_dict().items()}, "xm": xm, "xs": xs,
              "din": int(X.shape[1]), "Q": Q, "h": h, "fifo": True}
    return params, {"train_frames": N, "steps": steps, "fit_sec": time.time() - t0,
                    "final_loss_drop": float(l_drop.detach()), "final_loss_q": float(l_q.detach())}


# ----------------------------------------------------------------------------- metrics and QA calibration
QS = torch.linspace(0.0025, 0.9975, 199, dtype=torch.float64)


def w1(a, b):
    """1-Wasserstein distance between two 1-D samples via 199 matched quantiles."""
    if a.numel() == 0 or b.numel() == 0:
        return float("nan")
    return float((a.double().quantile(QS) - b.double().quantile(QS)).abs().mean())


def l05_cells(fr):
    nb, sb, c = _ns.lookup_key("L05", fr["nact"], fr["snr"], fr["own"], fr["cls"])
    return (nb * (len(_ns.SNR_EDGES) + 1) + sb) * 2 + c


def censored_cell_w1(ref, st, min_ref=100, min_st=20):
    """Frame-weighted per-L05-cell W1 (ms) of the censored delay (lost = TIMEOUT)."""
    cr, cs = l05_cells(ref), l05_cells(st)
    a_all, b_all = ref["delay"].clamp(max=TIMEOUT), st["delay"].clamp(max=TIMEOUT)
    tot, ws = 0.0, 0
    for cid in cr.unique():
        a, b = a_all[cr == cid], b_all[cs == cid]
        if a.numel() < min_ref or b.numel() < min_st:
            continue
        tot += w1(a * 100, b * 100) * a.numel()
        ws += a.numel()
    return tot / ws if ws else float("nan")


def calibrate_qa(ref, E, R, device, sizes, T, seed, etas=(0.4, 0.55, 0.7, 0.85, 1.0, 1.15), episodes=1):
    rows = []
    for pf in (True, False):
        for eta in etas:
            st, _, _ = log_rollouts("QA", E, R, episodes, seed, device, sizes, T=T,
                                    params={"eta": eta, "pf": pf})
            rows.append({"eta": eta, "pf": pf, "w1_ms": censored_cell_w1(ref, st),
                         "drop_rate": float(st["drop"].float().mean())})
    ok = [r for r in rows if not math.isnan(r["w1_ms"])]
    best = min(ok, key=lambda r: r["w1_ms"]) if ok else {"eta": 0.9, "pf": True}
    return {"eta": float(best["eta"]), "pf": bool(best["pf"])}, rows


@torch.no_grad()
def heldout(te, Bte, fit, R, device, seed=4242):
    """Teacher-forced fit quality of NN and GE on held-out frames: pooled delay W1 (ms) and drop Brier score."""
    torch.manual_seed(seed)
    dm = ~te["drop"]
    ref = te["delay"][dm]
    out = {"frames": int(te["delay"].numel()), "drop_rate": float(te["drop"].float().mean())}
    p = fit["NN"]
    net = DelayNet(p["din"], p["Q"], p["h"])
    net.load_state_dict(p["state"])
    logit, q = net((nn_x(te, R) - p["xm"]) / p["xs"])
    pd = torch.sigmoid(logit)
    samp = torch.exp(DelayNet.sample_quantiles(q, torch.rand(q.shape[0])))[dm]
    out["NN"] = {"w1_pooled_ms": w1(ref * 100, samp * 100), "brier": float(((pd - te["drop"].float()) ** 2).mean())}
    g = fit["GE"]
    s = ge_states(Bte, float(g["theta"]))[te["ep"], te["cap"], te["env"]]
    c = te["cls"] - 1
    u = torch.rand(int(dm.sum())) * 100
    lo = u.floor().long().clamp(max=99)
    qf = g["q"][s[dm], c[dm]]
    gs = qf.gather(1, lo[:, None]).squeeze(1) * (1 - (u - lo)) + qf.gather(1, (lo + 1)[:, None]).squeeze(1) * (u - lo)
    out["GE"] = {"w1_pooled_ms": w1(ref * 100, gs * 100),
                 "brier": float(((g["p"][s, c] - te["drop"].float()) ** 2).mean())}
    return out


# ----------------------------------------------------------------------------- driver
def fit_levels(source="L2-legacy", E=64, R=16, episodes=12, test_episodes=4, T=300, sizes=TASK_SIZES["T1"],
               device="cuda", preset=None, backend="reference", seed=31337, nn_steps=6000, qa_envs=32,
               qa_etas=(0.4, 0.55, 0.7, 0.85, 1.0, 1.15), log=print):
    """Roll out `source`, fit TR / GE / NN, calibrate QA; returns (fit dict for torch.save, info dict)."""
    if source not in ("L2", "L2-legacy"):
        raise ValueError("fit from 'L2' or 'L2-legacy'")
    sizes = tuple(float(s) for s in sizes)
    if len(sizes) != 2:
        raise ValueError("the surrogate fits have two traffic classes; pass two message sizes")
    cfg = source_config(source, preset)
    t0 = time.time()
    tr, Btr, sps = log_rollouts(source, E, R, episodes, seed, device, sizes, cfg, T, backend)
    te, Bte, _ = log_rollouts(source, E, R, test_episodes, seed + 1, device, sizes, cfg, T, backend)
    log(f"logged {tr['delay'].numel()} training frames ({int(tr['drop'].sum())} lost), {te['delay'].numel()} "
        f"held-out, {source} {sps * 1e3:.1f} ms/step, {time.time() - t0:.0f} s")
    info = {"source": source, "preset": preset or "default", "sizes": list(sizes), "E": E, "R": R, "T": T,
            "episodes": episodes, "test_episodes": test_episodes, "seed": seed, "source_ms_per_step": sps * 1e3}
    fit = {}
    fit["TR"], info["TR"] = fit_tr(tr, E, int(Btr.shape[0]), T)
    fit["GE"], info["GE"] = fit_ge(tr, Btr)
    fit["NN"], info["NN"] = fit_nn(tr, R, device, steps=nn_steps)
    fit["QA"], info["QA_grid"] = calibrate_qa(te, qa_envs, R, device, sizes, T, seed + 2, qa_etas)
    info["QA"] = fit["QA"]
    info["heldout"] = heldout(te, Bte, fit, R, device)
    fit["meta"] = {"version": 1, "source": source, "preset": preset or "default", "sizes": list(sizes), "R": R,
                   "T": T, "frame_buffer": F, "timeout_steps": TIMEOUT}
    log(f"fitted in {time.time() - t0:.0f} s: QA {fit['QA']}, held-out {json.dumps(info['heldout'])}")
    return fit, info


def default_fit_path(name):
    root = os.environ.get("ISAACLAB_NET_LEVELS_DIR", os.path.join(os.path.expanduser("~"), ".cache",
                                                                   "isaaclab_net", "levels"))
    return os.path.join(root, name + ".pt")


def save_fit(fit, path, info=None):
    """torch.save the fit (plus a JSON summary next to it). Refuses paths inside the source tree."""
    path = os.path.abspath(os.path.expanduser(path))
    root = os.path.realpath(REPO_ROOT)
    if os.path.realpath(path).startswith(root + os.sep):
        raise ValueError(f"{path} is inside the source tree {root}; fitted parameters stay outside the repository")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(fit, path)
    if info is not None:
        with open(os.path.splitext(path)[0] + ".json", "w") as f:
            json.dump(info, f, indent=1)
    return path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--source", default="L2-legacy", choices=["L2-legacy", "L2"])
    p.add_argument("--preset", default="default", choices=PRESETS, help="NRConfig preset of the source")
    p.add_argument("--task", default="T1", choices=sorted(TASK_SIZES), help="message sizes of the example task")
    p.add_argument("--sizes", default="", help="message sizes in bytes, e.g. 4000,30000 (overrides --task)")
    p.add_argument("--envs", type=int, default=64)
    p.add_argument("--robots", type=int, default=16)
    p.add_argument("--episodes", type=int, default=12)
    p.add_argument("--test-episodes", type=int, default=4)
    p.add_argument("--episode-steps", type=int, default=300)
    p.add_argument("--nn-steps", type=int, default=6000)
    p.add_argument("--qa-envs", type=int, default=32)
    p.add_argument("--seed", type=int, default=31337)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--backend", default="reference", help="source backend (graph for L2-legacy on CUDA)")
    p.add_argument("--out", default="", help="output .pt (default: $ISAACLAB_NET_LEVELS_DIR or "
                                             "~/.cache/isaaclab_net/levels/<source>_<task>.pt)")
    a = p.parse_args(argv)
    sizes = tuple(float(s) for s in a.sizes.split(",")) if a.sizes else TASK_SIZES[a.task]
    fit, info = fit_levels(a.source, a.envs, a.robots, a.episodes, a.test_episodes, a.episode_steps, sizes,
                           a.device, a.preset, a.backend, a.seed, a.nn_steps, a.qa_envs)
    tag = a.task if not a.sizes else "s" + "-".join(str(int(s)) for s in sizes)
    path = save_fit(fit, a.out or default_fit_path(f"{a.source}_{tag}"), info)
    print("saved", path)


if __name__ == "__main__":
    main()
