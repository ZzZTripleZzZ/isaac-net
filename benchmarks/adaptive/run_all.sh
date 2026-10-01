#!/bin/bash
# Accuracy vs cost sweeps of the adaptive engine (docs/adaptive-fidelity.md) on one GPU.
# usage: bash benchmarks/adaptive/run_all.sh [outdir] [fitdir]   (fitdir: outside the source tree)
# Writes <outdir>/adaptive_*.jsonl, transient_*.jsonl, kernel_time.jsonl and tables.md.
set -e
cd "$(dirname "$0")/../.."
OUT=${1:-runs}
FIT=${2:-$HOME/.cache/isaac_net/adaptive}
mkdir -p "$OUT" "$FIT"
S=benchmarks/adaptive/sweep.py
T=benchmarks/adaptive/transient.py
LOAD="--thresholds 1000,4000,16000 --budgets mask,0.1,0.25,0.5 --static 0.1,0.25"
# 0. L05 / L05Q tables fitted from L2-legacy (its own workload seed)
python benchmarks/adaptive/fit_l05q.py --E 1024 --R 16 --steps 400 --out "$FIT/l05q_fit.pt"
# 1. L1 (triton) -> L2-legacy (triton), E = 4096 x 16
python $S --E 4096 --R 16 --backend triton --scenarios mixed,uniform-0.05,uniform-0.2,uniform-0.5 $LOAD \
    --out "$OUT/adaptive_l1_triton.jsonl"
# 1b. the cost table of the docs: decision periods 1 and 5, all engines timed round-robin in one run
python $S --E 4096 --R 16 --backend triton --scenarios mixed --thresholds 1000 --budgets mask,0.1,0.25,0.5 \
    --decision_period 1,5 --static 0.1,0.25 --reps 7 --out "$OUT/adaptive_cost.jsonl"
# 2. L05Q (graph, fitted) -> L2-legacy (triton), E = 4096 x 16
python $S --E 4096 --R 16 --cheap L05Q --cheap_params "$FIT/l05q_fit.pt" --backend graph --exp_backend triton \
    --scenarios mixed,uniform-0.2,uniform-0.5 $LOAD --out "$OUT/adaptive_l05q_triton.jsonl"
# 3. L1 (triton) -> L2-legacy on the graph backend (E = 1024 x 16): an expensive level much costlier than the cheap one
python $S --E 1024 --R 16 --backend triton --exp_backend graph --scenarios mixed --thresholds 4000 \
    --budgets mask,0.1,0.25,0.5 --static 0.1,0.25 --out "$OUT/adaptive_l1_graph.jsonl"
# 4. L1 (triton) -> L2, the NR engine (reference backend), E = 256 x 8
python $S --E 256 --R 8 --expensive L2 --backend triton --exp_backend reference --scenarios mixed --steps 300 \
    --thresholds 4000 --budgets mask,0.25 --static 0.1 --reps 3 --window 20 --out "$OUT/adaptive_nr.jsonl"
# 5. handoff transient: forced switching every N steps against a fixed 50 % mix
python $T --E 2048 --R 16 --cheap L1 --out "$OUT/transient_l1.jsonl"
python $T --E 2048 --R 16 --cheap L05Q --backend graph --cheap_params "$FIT/l05q_fit.pt" --out "$OUT/transient_l05q.jsonl"
# 6. GPU kernel time per step (contention-light cost estimate)
python benchmarks/adaptive/kernel_time.py --E 4096 --R 16 --cheap_params "$FIT/l05q_fit.pt" --out "$OUT/kernel_time.jsonl"
python benchmarks/adaptive/report.py "$OUT"/adaptive_*.jsonl > "$OUT/tables.md"
echo DONE > "$OUT/adaptive_DONE"
