"""Closed-loop run of one fixed controller over every network model (docs/closed-loop.md).

The same scripted Fleet-Alert controller drives E x R robots for T control steps against each arm:

  ideal      level ORACLE: every frame arrives in its capture step, delay 0, never lost
  L2         NR engine, lena_validation_v2(), graph backend
  L0         i.i.d. lognormal delay and loss, fitted to the L2 arm's pooled marginals (l0_fit.csv)
  L0-emp     level L0 in its empirical mode: i.i.d. delay resampled from the pooled L2 delays (their empirical
             marginal) and i.i.d. loss, both from the same L2 frames as L0 (l0emp_fit.csv)
  L1         fluid level, graph backend
  L2-legacy  prototype slot-level engine, triton backend
  ns3        ns-3.48 + 5G-LENA v5.1 through the lockstep bridge (one process per env, TCP)

Geometry (all arms): a 60 m x 60 m arena with the gNB at the corner (0, 0), log-distance path loss
40 + 35 log10(d) dB, 23 dBm UE power, thermal noise with a 7 dB noise figure (-101.44 dBm per 10-PRB subband),
6 dB per-robot shadowing drawn at reset, fading off, UE power over the whole band in ns-3. The coverage rule of the
5G-LENA validation (single-subband full-power SNR >= 7 dB) is applied to the shadowing: a draw that would put the
robot below 7 dB anywhere in the arena is redrawn, so every robot stays inside the validated range at every pose.
Every engine arm gets the same per-robot SNR [E, R] each step, computed from the end-of-step pose; ns-3 gets the
end-of-step pose and the same shadowing and computes the same path loss itself.

The task random numbers come from one generator seeded per seed, with a fixed number of draws per step, so every
arm sees the same initial poses, goals, shadowing and hazard draws; the runs differ only through what the network
delivers to the controller.

usage (lab box): python benchmarks/closedloop/run_closedloop.py --out benchmarks/results/closedloop
  heavier loads (docs/closed-loop.md, "Under load"): --interval 1 (one frame per 0.2 s) or --R 32
  L0 / L0-emp without rerunning L2: --arms L0,L0-emp --fit-from <folder of an earlier run that has the L2 arm>
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))

from isaac_net.core import NRConfig, lena_validation_v2, make_engine  # noqa: E402

ARMS = ("ideal", "L2", "L0", "L0-emp", "L1", "L2-legacy", "ns3")
SIZES = (4000.0, 30000.0)
STEP_S = 0.1
TIMEOUT = 20                       # application deadline, control steps (2 s)
ARENA = 60.0
P_TX, NI_DBM, PL0, PLEXP = 23.0, -101.44, 40.0, 3.5
SHADOW_STD, MIN_SNR = 6.0, 7.0
SEND_CLS = 2                       # the controller always sends the large (30 kB) camera frame
MIN_INTERVAL = 5                   # default: at most one frame per robot every 5 steps (2 Hz); --interval


def snr_db(pos, sh):
    """Single-subband full-power UL SNR [E,R] (dB) at positions [E,R,2] with shadowing sh [E,R]."""
    d = pos.norm(dim=-1).clamp(min=1.0)
    return P_TX - (PL0 + 10 * PLEXP * torch.log10(d)) - sh - NI_DBM


# ------------------------------------------------------------------------------------------------ task
class FleetLoop:
    """Fleet-Alert (isaac_net.bench.tasks.fleet_alert) in the 60 m arena, with its scripted controller.

    Robots drive to random goals at 3 m/s. A hazard appears near a random robot, grows to 15 m radius and lasts 10 s.
    A large frame captured within 50 m of an active hazard detects it, and the fleet learns the hazard only when a
    detecting frame is delivered. The controller drives to the goal, away from a known hazard, and sends a large
    frame when the robot has no frame in flight and at least MIN_INTERVAL steps have passed since its last send.
    Reward per robot-step: progress to the goal (m) - 1 inside an active hazard + 2 per goal reached."""
    VMAX, H_R, H_GROW, H_LIFE, H_RATE, DET_RANGE = 0.3, 15.0, 0.3, 100, 1 / 80, 50.0

    def __init__(self, E, R, seed, dev, min_interval=MIN_INTERVAL):
        self.E, self.R, self.dev = E, R, dev
        self.min_interval = int(min_interval)
        self.gen = torch.Generator(device=dev)
        self.gen.manual_seed(int(seed) * 7919 + 17)
        L = ARENA
        self.pos = self.rand(E, R, 2) * L
        self.goal = self.rand(E, R, 2) * L
        far = torch.tensor([L, L], device=dev).norm()
        worst = P_TX - (PL0 + 10 * PLEXP * torch.log10(far)) - NI_DBM - MIN_SNR      # largest allowed shadowing
        sh = self.randn(E, R) * SHADOW_STD
        for _ in range(100):
            bad = sh > worst
            if not bool(bad.any()):
                break
            sh = torch.where(bad, self.randn(E, R) * SHADOW_STD, sh)
        self.sh = sh.clamp(max=float(worst))
        self.t = 0
        self.h_on = torch.zeros(E, dtype=torch.bool, device=dev)
        self.h_pos = torch.zeros(E, 2, device=dev)
        self.h_id = torch.zeros(E, dtype=torch.long, device=dev)
        self.h_start = torch.zeros(E, dtype=torch.long, device=dev)
        self.known = torch.zeros(E, dtype=torch.bool, device=dev)
        self.since = torch.full((E, R), 10 ** 6, dtype=torch.long, device=dev)

    def rand(self, *s):
        return torch.rand(*s, device=self.dev, generator=self.gen)

    def randn(self, *s):
        return torch.randn(*s, device=self.dev, generator=self.gen)

    def radius(self):
        return ((self.t - self.h_start + 1).float() * self.H_GROW).clamp(max=self.H_R) * self.h_on.float()

    @staticmethod
    def unit(v):
        return v / v.norm(dim=-1, keepdim=True).clamp(min=1e-6)

    def act(self, in_flight):
        """Scripted controller: velocity [E,R,2] in [-1,1] and send class [E,R] (0 = none)."""
        to_goal = self.unit(self.goal - self.pos)
        away = self.pos - self.h_pos[:, None, :]
        near = (self.known[:, None] & (away.norm(dim=-1) < self.radius()[:, None] + 5.0))[..., None]
        vel = torch.where(near, self.unit(away), to_goal)
        ok = ~in_flight & (self.since >= self.min_interval)
        send = torch.where(ok, torch.full_like(self.since, SEND_CLS), torch.zeros_like(self.since))
        self.since = torch.where(ok, torch.zeros_like(self.since), self.since + 1)
        return vel, send

    def pre_step(self, send):
        """Hazard lifecycle and detection at the start-of-step pose; returns det [E,R] (frame detects the hazard)."""
        E, R, t = self.E, self.R, self.t
        end = self.h_on & (t - self.h_start >= self.H_LIFE)
        self.h_on &= ~end
        self.known &= ~end
        spawn = ~self.h_on & (self.rand(E) < self.H_RATE)
        j = torch.randint(0, R, (E,), device=self.dev, generator=self.gen)
        c = self.pos[torch.arange(E, device=self.dev), j] + 10 * self.randn(E, 2)
        self.h_pos = torch.where(spawn[:, None], c.clamp(0, ARENA), self.h_pos)
        self.h_id = self.h_id + spawn.long()
        self.h_start = torch.where(spawn, torch.full_like(self.h_start, t), self.h_start)
        self.h_on |= spawn
        self.known &= ~spawn
        dist = (self.pos - self.h_pos[:, None, :]).norm(dim=-1)
        return (send > 0) & self.h_on[:, None] & (dist < self.DET_RANGE)

    def post_step(self, vel, learned):
        """Move, apply the delivered detections (learned [E] bool), reward [E,R] and hazard exposure [E,R]."""
        d_old = (self.goal - self.pos).norm(dim=-1)
        self.pos = (self.pos + vel * self.VMAX).clamp(0.0, ARENA)
        self.known |= learned & self.h_on
        d_new = (self.goal - self.pos).norm(dim=-1)
        inside = self.h_on[:, None] & ((self.pos - self.h_pos[:, None, :]).norm(dim=-1) < self.radius()[:, None])
        reached = d_new < 3.0
        rew = (d_old - d_new) - inside.float() + 2.0 * reached.float()
        newg = self.rand(self.E, self.R, 2) * ARENA
        self.goal = torch.where(reached[..., None], newg, self.goal)
        self.t += 1
        return rew, inside, reached


# ------------------------------------------------------------------------------------------------ network arms
class EngineArm:
    """Any make_engine level through the contract API (submit, then step with the SNR [E,R])."""

    def __init__(self, level, backend, cfg, E, R, dev, seed, params=None):
        self.eng = make_engine(level, E, R, dev, cfg, backend, seed=seed, params=params)
        self.eng.reset()
        # capture step of a frame the engine lost before its deadline (RLC UM loss after HARQ exhaustion, key
        # "dropped" of the NR engine), -1 = none. The application only learns of the loss at the deadline, as with
        # ns-3, so the frame counts as in flight until then.
        self.lost = torch.full((E, R), -1, dtype=torch.long, device=dev)

    def step(self, t, send, snr, pos, tag):
        self.eng.submit(None, send)
        out = self.eng.step(None, snr)
        dlv = out["delivered"]
        e, r, f = dlv.nonzero(as_tuple=True)
        cap = out["cap"][e, r, f]
        delay_steps = out["delay"][e, r, f].float()
        gone = out["timed_out"]
        if "dropped" in out:
            dr_ = out["dropped"] & ~gone
            capd = torch.where(dr_, out["cap"], torch.full_like(out["cap"], -1)).max(-1).values
            self.lost = torch.where(capd >= 0, capd, self.lost)
        te, tr, tf = gone.nonzero(as_tuple=True)
        tc = out["cap"][te, tr, tf]
        due = (self.lost >= 0) & (t + 1 - self.lost >= TIMEOUT)
        le, lr = due.nonzero(as_tuple=True)
        lc = self.lost[le, lr]
        self.lost = torch.where(due, torch.full_like(self.lost, -1), self.lost)
        return (e, r, cap, delay_steps), (torch.cat([te, le]), torch.cat([tr, lr]), torch.cat([tc, lc]))

    def close(self):
        pass


class Ns3Arm:
    """ns-3 + 5G-LENA through the lockstep bridge; frame id = capture step, completion time = t + frac."""

    def __init__(self, E, R, dev, seed):
        from isaac_net.bridges.ns3_lockstep import protocol as P
        from isaac_net.bridges.ns3_lockstep.core import Ns3Lockstep
        self.P, self.E, self.R, self.dev = P, E, R, dev
        args = dict(fading=False, ulPowerAlloc="UniformPowerAllocBw", niPerSubbandDbm=NI_DBM, side=ARENA,
                    deadline=TIMEOUT * STEP_S, shadowStd=SHADOW_STD)
        self.core = Ns3Lockstep(E, R, mode="procs", transport="tcp", run=1 + 1000 * seed, ns3_args=args)
        self.seed = seed
        self.inflight = np.full((E, R), -1, np.int64)       # capture step of the frame in flight, -1 = none
        self.first = True

    def step(self, t, send, snr, pos, sh):
        P = self.P
        p = pos.detach().float().cpu().numpy()
        shn = sh.detach().float().cpu().numpy()
        if self.first:
            self.core.reset(pos=p, shadow=shn, runs=[1 + 1000 * self.seed + g for g in range(self.core.G)])
            self.first = False
        s = send.cpu().numpy()
        e, r = np.nonzero(s > 0)
        fr = np.zeros(len(e), P.FRAME_IN)
        fr["env"], fr["ue"], fr["fid"] = e, r, t
        fr["bytes"] = np.round(np.asarray(SIZES)[s[e, r] - 1]).astype(np.uint32)
        self.inflight[e, r] = t
        res = self.core.step(p, fr, shadow=shn, interp=True)
        de, dr, dc, dd = [], [], [], []
        for ee, uu, fid, frac in zip(res["done"]["env"], res["done"]["ue"], res["done"]["fid"], res["done"]["frac"]):
            if self.inflight[ee, uu] == fid:
                de.append(ee)
                dr.append(uu)
                dc.append(fid)
                dd.append(t + frac - fid)
                self.inflight[ee, uu] = -1
        tim = (self.inflight >= 0) & (t + 1 - self.inflight >= TIMEOUT)
        te, tr = np.nonzero(tim)
        tc = self.inflight[te, tr].copy()
        self.inflight[te, tr] = -1
        T = lambda x, dt=torch.long: torch.as_tensor(np.asarray(x), dtype=dt, device=self.dev)
        return (T(de), T(dr), T(dc), T(dd, torch.float32)), (T(te), T(tr), T(tc))

    def close(self):
        self.core.close()


def make_arm(arm, E, R, dev, seed, l0_params=None):
    """l0_params: {"L0": lognormal params, "L0-emp": quantile-table params} (fitted to the L2 arm)."""
    l0_params = l0_params or {}
    eseed = int(seed) * 1000 + 1
    app = dict(msg_sizes=SIZES, timeout_steps=TIMEOUT, control_step_ms=STEP_S * 1000, frame_buffer=16, seed=eseed)
    if arm == "ideal":
        return EngineArm("ORACLE", "graph", NRConfig(**app), E, R, dev, eseed)
    if arm == "L2":
        return EngineArm("L2", "graph", lena_validation_v2(**app), E, R, dev, eseed)
    if arm == "L0":
        return EngineArm("L0", "graph", NRConfig(**app), E, R, dev, eseed, params=l0_params["L0"])
    if arm == "L0-emp":
        return EngineArm("L0", "graph", NRConfig(**app), E, R, dev, eseed, params=l0_params["L0-emp"])
    if arm == "L1":
        return EngineArm("L1", "graph", NRConfig(**app), E, R, dev, eseed)
    if arm == "L2-legacy":
        return EngineArm("L2-legacy", "triton", NRConfig(**app), E, R, dev, eseed)
    if arm == "ns3":
        return Ns3Arm(E, R, dev, seed)
    raise ValueError(arm)


# ------------------------------------------------------------------------------------------------ one episode
def run_episode(arm, E, R, T, seed, dev, l0_params=None, min_interval=MIN_INTERVAL):
    task = FleetLoop(E, R, seed, dev, min_interval)
    t0 = time.perf_counter()
    net = make_arm(arm, E, R, dev, seed, l0_params)
    t_build = time.perf_counter() - t0
    inflight = torch.full((E, R), -1, dtype=torch.long, device=dev)
    tagtab = torch.full((E, R, T), -1, dtype=torch.long, device=dev)      # hazard id carried by the frame of step t
    last_cap = torch.zeros(E, R, dtype=torch.long, device=dev)
    aoi, frames = [], []
    ret = torch.zeros(E, R, device=dev)
    expo = torch.zeros(E, R, device=dev)
    goals = torch.zeros(E, R, device=dev)
    sent = timed = 0
    t1 = time.perf_counter()
    for t in range(T):
        vel, send = task.act(inflight >= 0)
        det = task.pre_step(send)
        tagtab[..., t] = torch.where(det, task.h_id[:, None].expand(E, R), torch.full_like(send, -1))
        inflight = torch.where(send > 0, torch.full_like(inflight, t), inflight)
        sent += int((send > 0).sum())
        pos_end = (task.pos + vel * task.VMAX).clamp(0.0, ARENA)
        snr = snr_db(pos_end, task.sh)
        extra = task.sh if arm == "ns3" else None
        (de, dr, dc, dd), (te, tr, tc) = net.step(t, send, snr, pos_end, extra)
        # delivered frames: the fleet learns a hazard when a frame tagged with the current hazard id arrives
        learned = torch.zeros(E, dtype=torch.bool, device=dev)
        if de.numel():
            tg = tagtab[de, dr, dc]
            hit = (tg >= 0) & (tg == task.h_id[de])
            learned[de[hit]] = True
            last_cap[de, dr] = torch.maximum(last_cap[de, dr], dc)
            inflight[de, dr] = -1
            frames.append(torch.stack([de.float(), dr.float(), dc.float(), dd * STEP_S * 1000.0], -1).cpu())
        if te.numel():
            inflight[te, tr] = -1
            timed += int(te.numel())
        rew, inside, reached = task.post_step(vel, learned)
        ret += rew
        expo += inside.float()
        goals += reached.float()
        aoi.append(((t + 1 - last_cap).float() * STEP_S).flatten().cpu())
    if dev.type == "cuda":
        torch.cuda.synchronize()
    wall = time.perf_counter() - t1
    net.close()
    fr = torch.cat(frames).numpy() if frames else np.zeros((0, 4))
    aoi = torch.cat(aoi).numpy()
    delivered = len(fr)
    d = fr[:, 3]
    row = dict(arm=arm, seed=seed, E=E, R=R, steps=T, sent=sent, delivered=delivered, timed_out=timed,
               in_flight_at_end=sent - delivered - timed,
               delivery_ratio=delivered / max(delivered + timed, 1),
               delay_p50_ms=float(np.percentile(d, 50)) if delivered else math.nan,
               delay_p95_ms=float(np.percentile(d, 95)) if delivered else math.nan,
               delay_mean_ms=float(d.mean()) if delivered else math.nan,
               aoi_mean_s=float(aoi.mean()), aoi_p95_s=float(np.percentile(aoi, 95)),
               task_return=float(ret.mean()), hazard_exposure=float(expo.mean() / T),
               goals_per_robot=float(goals.mean()), wall_s=wall, build_s=t_build, ms_per_step=1000 * wall / T)
    return row, fr


# ------------------------------------------------------------------------------------------------ distances
def w1(a, b):
    """Wasserstein-1 between two 1-D samples (same units as the samples)."""
    a, b = np.sort(a), np.sort(b)
    q = np.linspace(0, 1, 2001)[1:-1]
    return float(np.mean(np.abs(np.quantile(a, q) - np.quantile(b, q))))


def ks(a, b):
    a, b = np.sort(a), np.sort(b)
    x = np.concatenate([a, b])
    return float(np.max(np.abs(np.searchsorted(a, x, "right") / len(a) - np.searchsorted(b, x, "right") / len(b))))


def fit_l0(delays_ms, delivered, resolved):
    """L0 parameters from the pooled L2 marginals: lognormal delay in control steps (median, log sigma) and an i.i.d.
    loss p chosen so that L0's delivery ratio, including its own lognormal tail beyond the deadline, equals L2's."""
    x = np.maximum(np.asarray(delays_ms) / (STEP_S * 1000.0), 1e-4)
    lx = np.log(x)
    mu, sig = float(np.median(lx)), float(lx.std())
    p_in = 0.5 * (1 + math.erf((math.log(TIMEOUT) - mu) / (sig * math.sqrt(2))))
    dr = delivered / max(resolved, 1)
    p = float(min(max(1 - dr / p_in, 0.0), 1.0))
    return {"mu": mu, "sig": sig, "p": p}, dict(median_steps=math.exp(mu), log_sigma=sig, loss=p,
                                                l2_delivery_ratio=dr, lognormal_within_deadline=p_in)


def fit_l0_emp(delays_ms, delivered, resolved):
    """L0-emp parameters from the same pooled L2 frames as fit_l0: the L0 level's empirical mode ({"q", "p"}), where q
    is the sorted pooled sample in control steps, so each delay is resampled i.i.d. from the empirical marginal
    (inverted CDF, the extreme tail included), and an i.i.d. loss p chosen as in fit_l0 so that L0-emp's delivery
    ratio, including the sample's share beyond the deadline, equals L2's. Returns the params and 101 summary rows."""
    x = np.sort(np.asarray(delays_ms, float)) / (STEP_S * 1000.0)
    p_in = float(np.mean(x < TIMEOUT))
    dr = delivered / max(resolved, 1)
    p = float(min(max(1 - dr / max(p_in, 1e-12), 0.0), 1.0))
    qs = np.quantile(x, np.linspace(0, 1, 101), method="inverted_cdf")
    rows = [dict(quantile=i / 100, delay_steps=float(v), loss=p, l2_delivery_ratio=dr, sample_within_deadline=p_in,
                 n_sample=len(x)) for i, v in enumerate(qs)]
    return {"q": torch.tensor(x, dtype=torch.float32), "p": p}, rows


def l2_from_folder(path):
    """Pooled L2 delays (ms), delivered and resolved frame counts of an earlier run (frames.csv.gz, per_seed.csv)."""
    d = []
    with gzip.open(os.path.join(path, "frames.csv.gz"), "rt") as f:
        for r in csv.DictReader(f):
            if r["arm"] == "L2":
                d.append(float(r["delay_ms"]))
    with open(os.path.join(path, "per_seed.csv")) as f:
        rows = [r for r in csv.DictReader(f) if r["arm"] == "L2"]
    if not d or not rows:
        raise SystemExit(f"{path} has no L2 arm to fit L0 / L0-emp to")
    dl = sum(int(r["delivered"]) for r in rows)
    return np.asarray(d), dl, dl + sum(int(r["timed_out"]) for r in rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--E", type=int, default=8)
    ap.add_argument("--R", type=int, default=16)
    ap.add_argument("--T", type=int, default=300)
    ap.add_argument("--interval", type=int, default=MIN_INTERVAL,
                    help="send counter: 5 = one frame per robot per 0.6 s, 1 = per 0.2 s (docs/closed-loop.md)")
    ap.add_argument("--fit-from", default=None,
                    help="fit L0 / L0-emp to the L2 arm of this earlier run folder instead of an L2 arm of this run")
    ap.add_argument("--out", default=os.path.join(HERE, "..", "results", "closedloop"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    dev = torch.device(a.device)
    os.makedirs(a.out, exist_ok=True)
    seeds = [int(s) for s in a.seeds.split(",")]
    arms = a.arms.split(",")
    rows, pooled = [], {}
    l0 = {}
    allframes = []
    for arm in arms:
        if arm in ("L0", "L0-emp"):
            if a.fit_from:
                src = l2_from_folder(a.fit_from)
            elif "L2" in pooled:
                l2 = [r for r in rows if r["arm"] == "L2"]
                src = (pooled["L2"], sum(r["delivered"] for r in l2), sum(r["delivered"] + r["timed_out"] for r in l2))
            else:
                raise SystemExit(f"{arm} is calibrated to the L2 arm: run L2 before it, or pass --fit-from")
            if arm == "L0":
                l0[arm], fit = fit_l0(*src)
                fit, name = [fit], "l0_fit.csv"
            else:
                l0[arm], fit = fit_l0_emp(*src)
                name = "l0emp_fit.csv"
            with open(os.path.join(a.out, name), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(fit[0]))
                w.writeheader()
                w.writerows(fit)
        ds = []
        ta = time.perf_counter()
        for s in seeds:
            row, fr = run_episode(arm, a.E, a.R, a.T, s, dev, l0, a.interval)
            rows.append(row)
            ds.append(fr[:, 3])
            allframes += [(arm, s, int(x[0]), int(x[1]), int(x[2]), float(x[3])) for x in fr]
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
        pooled[arm] = np.concatenate(ds)
        print(f"# {arm}: {time.perf_counter() - ta:.1f} s for {len(seeds)} seeds", flush=True)
    keys = list(rows[0])
    with open(os.path.join(a.out, "per_seed.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    with gzip.open(os.path.join(a.out, "frames.csv.gz"), "wt", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "seed", "env", "robot", "capture_step", "delay_ms"])
        w.writerows(allframes)
    # summary: mean and 95% t interval over seeds
    from statistics import mean, stdev
    tq = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571}
    mets = ["delivery_ratio", "delay_p50_ms", "delay_p95_ms", "delay_mean_ms", "aoi_mean_s", "aoi_p95_s",
            "task_return", "hazard_exposure", "goals_per_robot", "sent", "ms_per_step", "wall_s"]
    with open(os.path.join(a.out, "summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "n_seeds"] + [x for m in mets for x in (m, m + "_ci95")] + ["wall_total_s"])
        for arm in arms:
            rr = [r for r in rows if r["arm"] == arm]
            line = [arm, len(rr)]
            for m in mets:
                v = [r[m] for r in rr if not math.isnan(r[m])]
                line += [mean(v) if v else math.nan,
                         tq.get(len(v), 2.0) * stdev(v) / math.sqrt(len(v)) if len(v) > 1 else math.nan]
            line.append(sum(r["wall_s"] + r["build_s"] for r in rr))
            w.writerow(line)
    # delay CDF on a log grid (pooled over seeds)
    grid = np.logspace(-1, math.log10(TIMEOUT * STEP_S * 1000.0), 241)
    with open(os.path.join(a.out, "delay_cdf.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "delay_ms", "cdf"])
        for arm in arms:
            x = np.sort(pooled[arm])
            for g in grid:
                w.writerow([arm, f"{g:.4g}", f"{np.searchsorted(x, g, 'right') / max(len(x), 1):.5f}"])
    # pairwise distances, pooled over seeds, plus split-half floors (even vs odd seeds within one arm)
    pairs = [("L0", "L2"), ("L0-emp", "L2"), ("L1", "L2"), ("L2", "ns3"), ("L2-legacy", "ns3"), ("L0", "ns3"),
             ("L0-emp", "ns3"), ("L1", "ns3"), ("ideal", "ns3")]
    with open(os.path.join(a.out, "distances.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["a", "b", "n_a", "n_b", "w1_ms", "ks"])
        for x, y in pairs:
            if x in pooled and y in pooled and len(pooled[x]) and len(pooled[y]):
                w.writerow([x, y, len(pooled[x]), len(pooled[y]), f"{w1(pooled[x], pooled[y]):.3f}",
                            f"{ks(pooled[x], pooled[y]):.4f}"])
        fr = np.array([(r[0], r[1], r[5]) for r in allframes], dtype=object)
        for arm in ("L2", "ns3"):
            if arm not in pooled:
                continue
            m = fr[:, 0] == arm
            ev = np.array([d for (s, d) in zip(fr[m, 1], fr[m, 2]) if s % 2 == 0], float)
            od = np.array([d for (s, d) in zip(fr[m, 1], fr[m, 2]) if s % 2 == 1], float)
            if len(ev) and len(od):
                w.writerow([f"{arm} even seeds", f"{arm} odd seeds", len(ev), len(od), f"{w1(ev, od):.3f}",
                            f"{ks(ev, od):.4f}"])
        # disjoint seeds: each arm's even seeds against ns-3's odd seeds, so the two samples share no trajectory
        # (the pooled pairs above share seeds, hence trajectories, and can lie closer than the split-half floors)
        if "ns3" in pooled:
            m3 = fr[:, 0] == "ns3"
            od3 = np.array([d for (s, d) in zip(fr[m3, 1], fr[m3, 2]) if s % 2 == 1], float)
            for arm in ("L2", "L0", "L0-emp", "L1", "L2-legacy"):
                if arm not in pooled:
                    continue
                m = fr[:, 0] == arm
                ev = np.array([d for (s, d) in zip(fr[m, 1], fr[m, 2]) if s % 2 == 0], float)
                if len(ev) and len(od3):
                    w.writerow([f"{arm} even seeds", "ns3 odd seeds", len(ev), len(od3), f"{w1(ev, od3):.3f}",
                                f"{ks(ev, od3):.4f}"])
    with open(os.path.join(a.out, "setup.json"), "w") as f:
        json.dump(dict(arms=arms, seeds=seeds, E=a.E, R=a.R, T=a.T, arena_m=ARENA, ni_dbm=NI_DBM,
                       shadow_std_db=SHADOW_STD, min_snr_db=MIN_SNR, send_class_bytes=SIZES[SEND_CLS - 1],
                       min_interval_steps=a.interval, timeout_steps=TIMEOUT, fit_from=a.fit_from,
                       l0_params=l0.get("L0"),
                       device=str(dev), gpu=torch.cuda.get_device_name(0) if dev.type == "cuda" else None,
                       torch=torch.__version__), f, indent=1)


if __name__ == "__main__":
    main()
