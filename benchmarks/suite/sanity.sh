#!/usr/bin/env bash
# Pipeline sanity runs of the benchmark suite (docs/benchmark-suite.md, "Sanity results"). Small on purpose:
# every task, random and heuristic baselines, a short PPO run (MLP and GRU), two seeds, on one cheap level (L0,
# graph backend) and on L2-legacy (triton backend), plus the light negative-control variants, and the load
# calibration. These numbers show that the pipeline works end to end; they are not tuned baseline results.
#
#   benchmarks/suite/sanity.sh [out_dir]          # needs a CUDA GPU (graph / triton backends)
set -euo pipefail
cd "$(dirname "$0")/../.."
OUT=${1:-results/sanity}
mkdir -p "$OUT"
COMMON=(--envs 64 --robots 16 --eval_envs 64 --eval_episodes 1 --ppo_envs 64 --ppo_iters 20 --ppo_horizon 32
        --seeds 0,1 --label sanity --out "$OUT" --quiet)
LEVELS=("L0 graph" "L2-legacy triton")

for lb in "${LEVELS[@]}"; do
  set -- $lb
  python -m isaaclab_net.bench calibrate --task all --level "$1" --backend "$2" --envs 64 --robots 16 \
    > "$OUT/calibration_$1_$2.jsonl"
  python -m isaaclab_net.bench calibrate --task all --variant light --level "$1" --backend "$2" --envs 64 \
    --robots 16 > "$OUT/calibration_light_$1_$2.jsonl"
done
# the same with one robot per env: no contention, so what remains is the per-robot link (coverage)
python -m isaaclab_net.bench calibrate --task all --level L2-legacy --backend triton --envs 64 --robots 1 \
  > "$OUT/calibration_R1_L2-legacy_triton.jsonl"
for lb in "${LEVELS[@]}"; do
  set -- $lb
  python -m isaaclab_net.bench run --task all --level "$1" --backend "$2" \
    --baselines random,heuristic,ppo_mlp,ppo_gru "${COMMON[@]}"
  python -m isaaclab_net.bench run --task all --variant light --level "$1" --backend "$2" \
    --baselines random,heuristic "${COMMON[@]}"
done
python -m isaaclab_net.bench report "$OUT" --json "$OUT/aggregate.json" > "$OUT/report.md"
echo DONE > "$OUT/DONE"
