"""Tables and cost-model fits for the pool scaling runs (bench_scaling.py output).

usage: python analyze_scaling.py TAG=file.jsonl[,file2.jsonl] ...  (e.g. vanilla=main.jsonl,r100.jsonl)
Prints markdown. Per-worker cost t (worker-side ms per 100 ms step, W=1 points):
  model A (as requested):  t = a + b * R * p           (p = per-robot send probability = load)
  model B:                 t = a + b * R + c * R^2 + d * R * p
Then extrapolates 4096 envs x 100 robots against the GPU engine (fastnet_report: triton 60.15 ms,
graph 817.7 ms per step at E=4096, R=100 under contention).
"""
import json
import sys

import numpy as np

GPU_MS = {"triton": 60.15, "graph": 817.7}


def load(spec):
    rows = []
    for f in spec.split(","):
        rows += [json.loads(l) for l in open(f)]
    return rows


def fit(X, y):
    # least squares on relative error (weights 1/y): t spans two orders of magnitude
    coef, *_ = np.linalg.lstsq(X / y[:, None], np.ones_like(y), rcond=None)
    pred = X @ coef
    r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    rel = np.abs(pred - y) / y
    return coef, r2, rel.max()


def report(tag, rows):
    print(f"\n### {tag}\n")
    print("| W | R | p | env-steps/s | pool step ms | worker mean / max ms | protocol ms | env ms | "
          "par. eff. | startup s | RSS MB | load1 | busy |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    base = {(r["R"], r["p"]): r["env_steps_per_s"] for r in rows if r["W"] == 1}
    for r in sorted(rows, key=lambda r: (r["R"], r["p"], r["W"])):
        eff = r["env_steps_per_s"] / (r["W"] * base[(r["R"], r["p"])]) if (r["R"], r["p"]) in base else float("nan")
        print(f"| {r['W']} | {r['R']} | {r['p']} | {r['env_steps_per_s']:.2f} | {r['pool_step_ms']:.0f} | "
              f"{r['worker_mean_ms']:.0f} / {r['worker_max_ms']:.0f} | {r['protocol_ms']:.1f} | "
              f"{r['env_python_ms']:.1f} | {eff:.2f} | {r['startup_s']:.2f} | {r['rss_mb_per_worker']:.0f} | "
              f"{r['loadavg_1m_before']:.0f} | {r['cpu_busy_before']:.2f} |")
    w1 = [r for r in rows if r["W"] == 1]
    R = np.array([r["R"] for r in w1], float)
    p = np.array([r["p"] for r in w1], float)
    t = np.array([r["worker_mean_ms"] for r in w1], float)
    cA, r2A, eA = fit(np.stack([np.ones_like(R), R * p], 1), t)
    cB, r2B, eB = fit(np.stack([np.ones_like(R), R, R ** 2, R * p], 1), t)
    kk, lc = np.polyfit(np.log(R), np.log(t), 1)
    print(f"\nW=1 fits over R in {sorted(set(R.astype(int).tolist()))}:")
    print(f"- model A: t = {cA[0]:.1f} + {cA[1]:.2f}*R*p ms  (R^2 = {r2A:.3f}, max rel err {eA:.0%})")
    print(f"- model B: t = {cB[0]:.1f} + {cB[1]:.2f}*R + {cB[2]:.4f}*R^2 + {cB[3]:.2f}*R*p ms  "
          f"(R^2 = {r2B:.3f}, max rel err {eB:.0%})")
    print(f"- power law: t = {np.exp(lc):.2f} * R^{kk:.2f} ms")
    t100 = {q: cB[0] + cB[1] * 100 + cB[2] * 1e4 + cB[3] * 100 * q for q in (0.1, 0.5)}
    meas100 = {r["p"]: r["worker_mean_ms"] for r in w1 if r["R"] == 100}
    rss100 = [r["rss_mb_per_worker"] for r in w1 if r["R"] == 100]
    for q in (0.1, 0.5):
        tq = meas100.get(q, t100[q])
        src = "measured" if q in meas100 else "model B"
        cores_rt = 4096 * tq / 100.0
        print(f"- 4096 envs x 100 robots, p={q}: {tq:.0f} ms per env-step on one core ({src}); "
              f"cores for real time (100 ms/step): {cores_rt:,.0f}; "
              + "; ".join(f"cores to match GPU {k} ({v} ms/step): {4096 * tq / v:,.0f}" for k, v in GPU_MS.items())
              + (f"; RAM {4096 * rss100[0] / 1024:,.0f} GB" if rss100 else ""))


if __name__ == "__main__":
    for a in sys.argv[1:]:
        tag, spec = a.split("=", 1)
        report(tag, load(spec))
