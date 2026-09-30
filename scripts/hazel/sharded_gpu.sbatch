#!/bin/bash
#SBATCH -J inet-shard
#SBATCH -A hpcproject_gpu
#SBATCH -p gpu
#SBATCH -q gpu
#SBATCH --gres=gpu:l40:2
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 01:00:00
#SBATCH -o /share/hpcproject/zzhang66/isaacnet/logs/sharded_gpu.%j.log
# ShardedEngine on a two-GPU node: the sharded test file, then equality + throughput of one engine vs two shards.
# Usage: sbatch sharded_gpu.sbatch (or --gres=gpu:a30:2). Prerequisite: install_core.sbatch.
set -uxo pipefail
B=/share/hpcproject/zzhang66/isaacnet
export PYTHONNOUSERSITE=1 XDG_CACHE_HOME=$B/xdgcache TRITON_CACHE_DIR=$B/triton_cache TMPDIR=$B/tmp
mkdir -p $TMPDIR
OUT=$B/results/sharded_$SLURM_JOB_ID; mkdir -p $OUT; rm -f $OUT/DONE
source $B/envs/core/bin/activate
cd $B/repo
hostname; nvidia-smi --query-gpu=index,name,memory.total,utilization.gpu --format=csv
python -m pytest -q tests/test_sharded.py > $OUT/pytest_sharded.log 2>&1; echo "pytest_rc=$?" | tee -a $OUT/summary.txt
for cfg in "L2-legacy graph" "L2-legacy triton" "L1 triton"; do
  set -- $cfg
  for E in 4096 16384; do
    python benchmarks/bench_sharded.py --level $1 --backend $2 --E $E --R 16 --devices cuda:0,cuda:1 \
      >> $OUT/bench_sharded.jsonl 2>> $OUT/bench_sharded.err
  done
done
cat $OUT/bench_sharded.jsonl | tee -a $OUT/summary.txt
touch $OUT/DONE
