"""Print a markdown table from bench JSON lines. usage: python summarize.py results/grid.jsonl"""
import json
import sys

rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip().startswith("{")]
print("| E | R | E·R | network | env steps/s | robot steps/s | control steps/s | net-only ms/step | GPU util before / during | torch peak MiB | device mem before → max MiB | startup s |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|")
for r in rows:
    net = "off" if r["rung"] == "off" else f'{r["rung"]} {r.get("backend", "")}'
    nm = r.get("net_only_ms_per_step")
    print(f'| {r["E"]} | {r["R"]} | {r["E"]*r["R"]:,} | {net} | {r["env_steps_per_s"]:,.0f} | {r["robot_steps_per_s"]:,.0f} | '
          f'{r["iter_per_s"]:.2f} | {"" if nm is None else f"{nm:.1f}"} | {r["gpu_util_before"]:.0f}% / {r["gpu_util_during_mean"]:.0f}% | '
          f'{r["torch_peak_alloc_mib"]:.0f} | {r["gpu_mem_before_mib"]:.0f} → {r["gpu_mem_during_max_mib"]:.0f} | {r["startup_s"]:.0f} |')
