#!/bin/bash
cd /home/zzhang66/experiments/lenafid_rerun1005
source /home/zzhang66/anaconda3/etc/profile.d/conda.sh && conda activate i5g
export ISAAC_NET_LENA_TABLES=/home/zzhang66/experiments/loadfix/lena_eesm_tables.npz
echo "$(date +%T) start commit $(git -C /home/zzhang66/experiments/isaac-net rev-parse --short HEAD)"
bash gpu_jobs.sh > gpu.log 2>&1 &
G=$!
bash cpu_jobs.sh > cpu.log 2>&1
wait $G
echo "$(date +%T) CHAIN_DONE"; cat gpu.log; tail -3 cpu.log
