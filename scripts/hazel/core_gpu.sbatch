#!/bin/bash
#SBATCH -J inet-core
#SBATCH -A hpcproject_gpu
#SBATCH -p gpu
#SBATCH -q gpu
#SBATCH --gres=gpu:l40:1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 03:00:00
#SBATCH -o /share/hpcproject/zzhang66/isaacnet/logs/core_gpu.%j.log
# Core package on a GPU node: GPU and CPU test suites, then benchmarks (fast backends, reference, NR engine).
# Usage: sbatch [--gres=gpu:a30:1] core_gpu.sbatch. Prerequisite: install_core.sbatch.
set -uxo pipefail
B=/share/hpcproject/zzhang66/isaacnet
export PYTHONNOUSERSITE=1 XDG_CACHE_HOME=$B/xdgcache TRITON_CACHE_DIR=$B/triton_cache TMPDIR=$B/tmp
mkdir -p $TMPDIR
OUT=$B/results/core_$SLURM_JOB_ID; mkdir -p $OUT; rm -f $OUT/DONE
source $B/envs/core/bin/activate
cd $B/repo
hostname; nvidia-smi; nvidia-smi --query-gpu=name,driver_version,memory.total,utilization.gpu --format=csv
python -c "import torch,triton;print(torch.__version__, torch.version.cuda, triton.__version__, torch.cuda.get_device_name())"
T=$(date +%s); python -m pytest -m gpu -v --durations=15 > $OUT/pytest_gpu.log 2>&1; echo "gpu_rc=$? s=$(( $(date +%s)-T ))" | tee -a $OUT/summary.txt
T=$(date +%s); python -m pytest -m "not gpu" -v --durations=15 > $OUT/pytest_cpu.log 2>&1; echo "cpu_rc=$? s=$(( $(date +%s)-T ))" | tee -a $OUT/summary.txt
tail -3 $OUT/pytest_gpu.log $OUT/pytest_cpu.log | tee -a $OUT/summary.txt
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv | tee -a $OUT/summary.txt
T=$(date +%s); python benchmarks/bench.py --grid 256x16,1024x16,4096x16,1024x64 --levels all --backends graph,triton --steps 50 --out $OUT/bench_fast.jsonl > $OUT/bench_fast.log 2>&1; echo "bench_fast_rc=$? s=$(( $(date +%s)-T ))" | tee -a $OUT/summary.txt
T=$(date +%s); python benchmarks/bench.py --grid 16x16,256x16 --levels all --backends reference,eager,compile --steps 30 --ref_steps 10 --out $OUT/bench_ref.jsonl > $OUT/bench_ref.log 2>&1; echo "bench_ref_rc=$? s=$(( $(date +%s)-T ))" | tee -a $OUT/summary.txt
sed '/"NR lena_like/d' benchmarks/bench_nr.py > $OUT/bench_nr_nolena.py    # lena_like needs the GPL 5G-LENA tables (not shipped)
T=$(date +%s); python $OUT/bench_nr_nolena.py $OUT/bench_nr.json > $OUT/bench_nr.log 2>&1; echo "bench_nr_rc=$? s=$(( $(date +%s)-T ))" | tee -a $OUT/summary.txt
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv | tee -a $OUT/summary.txt
touch $OUT/DONE
