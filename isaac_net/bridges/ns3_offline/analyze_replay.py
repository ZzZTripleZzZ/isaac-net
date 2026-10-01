"""Markdown summary of replay_error.jsonl (mean over seeds, with min..max)."""
import json
import sys

rows = [json.loads(l) for l in open(sys.argv[1])]
M = ["frames", "overflow_share", "delivery_rate", "delay_p50_ms", "delay_p95_ms", "delay_mean_ms", "aoi_mean_steps",
     "sends_per_robot_step", "large_share", "expo", "ret"]


def agg(vals):
    v = [x for x in vals if x is not None]
    if not v:
        return "–"
    m = sum(v) / len(v)
    return f"{m:.3g} [{min(v):.3g}, {max(v):.3g}]" if len(v) > 1 else f"{m:.3g}"


print(f"seeds {[r['seed'] for r in rows]}, E={rows[0]['E']} envs x R={rows[0]['R']} robots x T={rows[0]['T']} steps\n")
names = list(rows[0]["metrics"].keys())
runs = [n for n in names if isinstance(rows[0]["metrics"][n], dict)]
print("| run | " + " | ".join(M) + " | lookup exact / nearest / env / lost |")
print("|---|" + "---|" * (len(M) + 1))
for n in runs:
    lk = [r["metrics"][n].get("lookup") for r in rows]
    lks = "–"
    if lk[0]:
        keys = ["exact", "nearest", "env", "lost"]
        lks = " / ".join(f"{sum(l.get(k, 0) for l in lk) / len(lk):.2f}" for k in keys)
    print(f"| {n} | " + " | ".join(agg([r["metrics"][n].get(m) for r in rows]) for m in M) + f" | {lks} |")

C = ["ks_delay", "delivery_rate_err", "delay_p50_ms_relerr", "delay_p95_ms_relerr", "delay_mean_ms_relerr",
     "aoi_mean_steps_relerr", "sends_per_robot_step_relerr", "expo_relerr", "ret_relerr",
     "common_frames", "delivered_both", "same_frames_1ms", "fate_mismatch", "mean_abs_delay_err_ms"]
print("\nErrors vs the closed-loop truth for the same policy (relerr = (replay - truth) / truth):\n")
print("| replay | " + " | ".join(C) + " |")
print("|---|" + "---|" * len(C))
for n in rows[0]["vs_truth"]:
    print(f"| {n} | " + " | ".join(agg([r["vs_truth"][n].get(c) for r in rows]) for c in C) + " |")
walls = {k: agg([r["metrics"].get(k) for r in rows]) for k in rows[0]["metrics"] if k.endswith("_wall_s")}
print("\nwall s: closed-loop (both policies) " + agg([r["closed_wall_s"] for r in rows]) + "; "
      + "; ".join(f"{k} {v}" for k, v in walls.items()))
