#!/bin/bash
# Grid of benchmarks/mjx/bench.py: network off / L0 (graph) / L2-legacy (triton) x E x R, REPS interleaved passes.
# Usage (repo root, venv active): bash benchmarks/mjx/run_grid.sh results/bench_mjx.jsonl [REPS]
OUT=${1:-bench_mjx.jsonl}; REPS=${2:-2}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
for rep in $(seq 1 "$REPS"); do
  for R in 16 32; do
    for E in 256 1024 4096; do
      for cfg in "off graph" "L0 graph" "L2-legacy triton"; do
        set -- $cfg
        timeout 600 python benchmarks/mjx/bench.py --num_envs "$E" --num_robots "$R" --level "$1" --backend "$2" \
          --steps 200 --tag "rep$rep" --out "$OUT" 2>&1 | grep -E "^RESULT|Error" | cut -c1-200
      done
    done
  done
done
echo DONE
