"""Accuracy vs cost of the adaptive engine on a synthetic contention workload.

usage: python benchmarks/adaptive/sweep.py --E 1024 --R 16 --backend triton --out adaptive.jsonl
       python benchmarks/adaptive/report.py adaptive.jsonl > tables.md

Workload (random send policies, no learning): E envs x R robots on the legacy cell (gNB at the origin), positions a
random walk inside the square --arena (default 5-60 m per axis, SNR above about 5 dB: backlog then comes from load,
not from robots out of coverage), each robot sends one message per step with probability p_e (70% 4000 B, 30%
30000 B). Scenarios:
  uniform-<p>   every env at the same p (a load sweep)
  mixed         each env alternates between a light regime (p = 0.03) and a heavy regime (p ~ U(0.3, 0.6)) as a
                two-state Markov chain (mean dwell 50 light steps, 25 heavy steps), so at any time about a third of
                the envs are congested and the set changes over time
Every run uses the same seed, the same poses, sends and partial-reset schedule (0.5% of envs per step), so the
runs differ only in the network model. The reference is pure L2-legacy; pure L2-legacy with another seed gives the
noise floor of the distance measures.

Metrics per run: the per-frame delay distribution of delivered frames (exact histogram in UL slots, 1/40 step) and
the drop share (timed out / resolved), compared with the reference by W1 (control steps) and KS; the share of
env-steps on the expensive level, switches; and ms per control step (submit + step, CUDA-synchronized, median of
--reps windows of --window steps, without the metric bookkeeping), without and with the partial resets; all engines
of a scenario are timed round-robin so that load changes on a shared GPU hit them alike. nvidia-smi utilization (all
processes) is sampled during the timing and recorded.
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time

import torch

_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _root)
from isaac_net.core import NRConfig, make_engine  # noqa: E402
from isaac_net.core.adaptive import FidelityConfig, make_adaptive  # noqa: E402
from isaac_net.core.proto.netsim import Requests  # noqa: E402

NB = 40 * 20 + 2            # delay histogram bins: UL slots 0 .. 800 (timeout 20 steps), last bin = overflow


class Util:
    def __enter__(self):
        self.v, self.stop = [], threading.Event()

        def run():
            while not self.stop.is_set():
                try:
                    o = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                                       capture_output=True, text=True, timeout=5).stdout
                    self.v.append(float(o.strip().split()[0]))
                except Exception:
                    pass
                self.stop.wait(0.5)
        self.th = threading.Thread(target=run, daemon=True)
        self.th.start()
        return self

    def __exit__(self, *a):
        self.stop.set()
        self.th.join()

    def mean(self):
        return round(sum(self.v) / len(self.v), 1) if self.v else None


def workload(E, R, steps, scenario, dev, seed=0, arena=(5.0, 60.0)):
    """List of (send [E,R], poses [E,R,2], reset ids or None) per step; generated once, reused by every run."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    lo, hi = arena
    pos = lo + (hi - lo) * torch.rand(E, R, 2, generator=g)
    heavy = torch.rand(E, generator=g) < 1 / 3
    hp = 0.3 + 0.3 * torch.rand(E, generator=g)
    out = []
    for _ in range(steps):
        if scenario == "mixed":
            flip = torch.rand(E, generator=g) < torch.where(heavy, 1 / 25, 1 / 50)
            heavy = heavy ^ flip
            p = torch.where(heavy, hp, torch.full_like(hp, 0.03))[:, None]
        else:
            p = float(scenario.split("-")[1])
        send = (torch.rand(E, R, generator=g) < p).long() * (1 + (torch.rand(E, R, generator=g) < 0.3).long())
        pos = (pos + 0.5 * torch.randn(E, R, 2, generator=g)).clamp(lo, hi)
        rs = (torch.rand(E, generator=g) < 0.005).nonzero().squeeze(-1)
        out.append((send.to(dev), pos.to(dev), rs.to(dev) if rs.numel() else None))
    return out


def run_metrics(net, wl, warm):
    """Delay histogram (UL slots) of delivered frames, drop and delivered counts, share of expensive env-steps."""
    dev = wl[0][0].device
    hist = torch.zeros(NB, dtype=torch.long, device=dev)
    timed = torch.zeros((), dtype=torch.long, device=dev)
    fid = torch.zeros((), dtype=torch.long, device=dev)
    n = 0
    for k, (send, pos, rs) in enumerate(wl):
        if rs is not None:
            net.reset(rs)
        net.submit(None, Requests(send))
        o = net.step(None, pos)
        if k < warm:
            continue
        d = o["delivered"]
        b = (o["delay"] * 40).round().nan_to_num(0).long().clamp(0, NB - 1)
        hist += torch.bincount(torch.where(d, b, torch.full_like(b, NB)).flatten(), minlength=NB + 1)[:NB]
        timed += o["timed_out"].sum()
        if "fidelity" in o:
            fid += o["fidelity"].sum()
        n += 1
    return hist.cpu(), int(timed), float(fid) / (n * wl[0][0].shape[0])


def time_round_robin(nets, wl, reps, window):
    """ms per control step of every engine in nets {name: engine}, timed round-robin (one window of each engine per
    round, so drift in the shared GPU's load hits all of them alike). Per round and engine: a window without resets
    (submit + step) and a window with the workload's partial resets. Returns {name: (median, median with resets,
    all samples, all samples with resets)} and the mean GPU utilization."""
    for net in nets.values():                                 # warm-up: graph capture, compiles
        for send, pos, rs in wl[:6]:
            if rs is not None:
                net.reset(rs)
            net.submit(None, Requests(send))
            net.step(None, pos)
    torch.cuda.synchronize()
    ts = {k: ([], []) for k in nets}
    n = len(wl) - 10
    with Util() as u:
        for r in range(reps):
            for name, net in nets.items():
                for j, with_resets in enumerate((False, True)):
                    base = 10 + ((2 * r + j) * window) % max(1, n - window)
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    for send, pos, rs in wl[base: base + window]:
                        if with_resets and rs is not None:
                            net.reset(rs)
                        net.submit(None, Requests(send))
                        net.step(None, pos)
                    torch.cuda.synchronize()
                    ts[name][j].append((time.perf_counter() - t0) * 1e3 / window)
    med = lambda v: sorted(v)[len(v) // 2]                          # noqa: E731
    return {k: (med(a), med(b), a, b) for k, (a, b) in ts.items()}, u.mean()


def dist(h1, h2):
    """W1 (control steps) and KS between two delay histograms in UL slots."""
    c1 = h1.double().cumsum(0) / max(1, int(h1.sum()))
    c2 = h2.double().cumsum(0) / max(1, int(h2.sum()))
    return float((c1 - c2).abs().sum()) / 40.0, float((c1 - c2).abs().max())


def quantile(h, q):
    c = h.double().cumsum(0) / max(1, int(h.sum()))
    return float((c < q).sum()) / 40.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--E", type=int, default=1024)
    ap.add_argument("--R", type=int, default=16)
    ap.add_argument("--backend", default="triton", help="backend of both levels (see --exp_backend)")
    ap.add_argument("--exp_backend", default=None, help="backend of the expensive level (default --backend)")
    ap.add_argument("--expensive", default="L2-legacy", help="L2-legacy or L2 (NR engine, reference backend)")
    ap.add_argument("--decision_period", default="1", help="one or more (comma-separated) decision periods")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--warm", type=int, default=50)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--window", type=int, default=40)
    ap.add_argument("--scenarios", default="mixed,uniform-0.05,uniform-0.15,uniform-0.3,uniform-0.5")
    ap.add_argument("--thresholds", default="1000,4000,16000")
    ap.add_argument("--budgets", default="mask,0.1,0.25,0.5")
    ap.add_argument("--indicator", default="backlog")
    ap.add_argument("--static", default="0.1,0.25")
    ap.add_argument("--cheap", default="L1")
    ap.add_argument("--cheap_params", default=None, help="fit file of fit_l05q.py (for --cheap L05 / L05Q)")
    ap.add_argument("--arena", default="5,60")
    ap.add_argument("--no_timing", action="store_true")
    ap.add_argument("--out", default="adaptive.jsonl")
    a = ap.parse_args()
    dev = torch.device("cuda")
    E, R = a.E, a.R
    cfg = NRConfig(seed=7)
    xb = a.exp_backend or a.backend
    X = a.expensive
    arena = tuple(float(v) for v in a.arena.split(","))
    cparams = torch.load(a.cheap_params, weights_only=True)[a.cheap] if a.cheap_params else None
    nsteps = max(a.steps, 10 + 4 * a.window)
    fh = open(a.out, "a")

    def emit(row):
        row.update(E=E, R=R, backend=a.backend, exp_backend=xb, expensive=X, cheap=a.cheap, arena=arena,
                   gpu=torch.cuda.get_device_name())
        fh.write(json.dumps(row) + "\n")
        fh.flush()
        print(json.dumps({k: row[k] for k in row if k not in ("hist", "times")}), flush=True)

    for sc in a.scenarios.split(","):
        wl = workload(E, R, nsteps, sc, dev, arena=arena)
        runs = [("ref", None, 7), ("ref-seed2", None, 8), (a.cheap, None, 7)]
        for thr in a.thresholds.split(","):
            for dp in (int(v) for v in str(a.decision_period).split(",")):
                for b in a.budgets.split(","):
                    runs.append(("adaptive", dict(mode="load", indicator=a.indicator, up_threshold=float(thr),
                                                  layout="mask" if b == "mask" else "subbatch",
                                                  active_budget=None if b == "mask" else float(b),
                                                  decision_period=dp), 7))
        for f in (a.static.split(",") if a.static else []):
            runs.append(("static", dict(mode="static", fraction=float(f)), 7))
        ref = None
        rows, builders = [], {}
        for i, (name, kw, seed) in enumerate(runs):
            def build(name=name, kw=kw, seed=seed):
                torch.manual_seed(seed)                 # the NR engine steps on the global RNG
                if name.startswith("ref"):
                    return make_engine(X, E, R, dev, cfg, xb, seed=seed)
                if kw is None:
                    return make_engine(name, E, R, dev, cfg, a.backend, seed=seed, params=cparams)
                fid = FidelityConfig(cheap=a.cheap, expensive=X, cheap_backend=a.backend, expensive_backend=xb,
                                     cheap_params=cparams, **kw)
                return make_adaptive(E, R, dev, cfg, fid, seed=seed)
            net = build()
            hist, timed, share = run_metrics(net, wl[: a.steps], a.warm)
            st = net.fidelity_stats() if hasattr(net, "fidelity_stats") else {}
            row = {"scenario": sc, "run": name, "cfg": kw, "delivered": int(hist.sum()), "timed_out": timed,
                   "drop": timed / max(1, timed + int(hist.sum())), "exp_share": share if kw else
                   (1.0 if name.startswith("ref") else 0.0), "p50": quantile(hist, 0.5), "p95": quantile(hist, 0.95),
                   "p99": quantile(hist, 0.99), "switch_up": st.get("up"), "switch_down": st.get("down"),
                   "denied": st.get("denied"), "graph": getattr(net, "graph", None), "hist": hist.tolist()}
            if name == "ref":
                ref = row
            row["w1"], row["ks"] = dist(hist, torch.tensor(ref["hist"]))
            row["drop_diff"] = row["drop"] - ref["drop"]
            del net
            torch.cuda.empty_cache()
            rows.append(row)
            if name != "ref-seed2":
                builders[i] = build
        if not a.no_timing:
            nets = {i: b() for i, b in builders.items()}
            res, util = time_round_robin(nets, wl, a.reps, a.window)
            for i, (ms, msr, ts, tsr) in res.items():
                rows[i].update(ms=ms, ms_resets=msr, times=ts, times_resets=tsr, gpu_util=util)
            del nets
            torch.cuda.empty_cache()
        for row in rows:
            emit(row)
    fh.close()


if __name__ == "__main__":
    main()
