#!/bin/bash
# Run every differentiable-model benchmark; results land in benchmarks/diff/results/.
#   bash run_all.sh [device] [threads per job]
# The sensitivity study runs as one job per (load, decision) in parallel, then merges.
set -e
cd "$(dirname "$0")"
DEV=${1:-cuda}
export OMP_NUM_THREADS=${2:-2}
mkdir -p results logs
pids=()
python -u convergence.py "$DEV" 64 200 > logs/convergence.log 2>&1 & pids+=($!)
for load in 0.3 0.5; do
  for dec in p kB tx dist p_own; do
    python -u sensitivity.py "$DEV" 128 300 3 "$load" "$dec" > "logs/sens_${load}_${dec}.log" 2>&1 & pids+=($!)
  done
done
python -u opt_rates.py "$DEV" 64 150 > logs/opt_rates.log 2>&1 & pids+=($!)
python -u placement.py "$DEV" 64 150 > logs/placement.log 2>&1 & pids+=($!)
python -u nn_proxy.py "$DEV" 24 128 200 3 > logs/nn_proxy.log 2>&1 & pids+=($!)
echo "${pids[@]}" > logs/pids
wait
python sensitivity.py summarize
touch results/DONE
