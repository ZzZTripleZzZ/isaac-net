#!/bin/bash
#SBATCH -J inet-bench2
#SBATCH -A hpcproject_gpu
#SBATCH -p gpu
#SBATCH -q gpu
#SBATCH --gres=gpu:l40:1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 02:30:00
#SBATCH -o /share/hpcproject/zzhang66/isaacnet/logs/bench2.%j.log
# Uncontended campaign, NR engine and L2-legacy rows on a dedicated Hazel L40 (whole GPU allocated to the job).
# Usage (login node): sbatch benchmarks/uncontended/hazel_nr.sbatch, from a copy of the repository at $B/bench2.
# Prerequisite: the core venv of scripts/hazel/install_core.sbatch. All compute goes through sbatch.
set -uxo pipefail
B=/share/hpcproject/zzhang66/isaacnet
export PYTHONNOUSERSITE=1 XDG_CACHE_HOME=$B/xdgcache TRITON_CACHE_DIR=$B/triton_cache TMPDIR=$B/tmp
OUT=$B/results/bench2_$SLURM_JOB_ID; mkdir -p $OUT $TMPDIR; rm -f $OUT/DONE
source $B/envs/core/bin/activate
cd $B/bench2
export PYTHONPATH=$PWD
hostname; nvidia-smi
nvidia-smi --query-gpu=name,driver_version,memory.total,utilization.gpu --format=csv | tee $OUT/gpu.txt
python -c "import torch,triton;print(torch.__version__, torch.version.cuda, triton.__version__, torch.cuda.get_device_name())" | tee -a $OUT/gpu.txt
for S in "256 16" "1024 32" "4096 16" "4096 100"; do
  set -- $S
  for c in "--case L2-legacy --backend graph" "--case L2-legacy --backend triton" \
           "--case L2 --cfg ul --backend triton" "--case L2 --cfg ul --backend graph" \
           "--case L2 --cfg ul_dl --backend triton" "--case L2 --cfg ul_dl --backend graph" \
           "--case L2 --cfg ul --backend reference" "--case L2 --cfg ul_dl --backend reference"; do
    timeout 1800 python benchmarks/uncontended/bench_net.py $c --E $1 --R $2 --out $OUT/net_hazel.jsonl 2>> $OUT/stderr.log | tail -1
  done
done
touch $OUT/DONE
