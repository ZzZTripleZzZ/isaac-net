#!/bin/bash
#SBATCH -J inet-isaac
#SBATCH -A hpcproject_gpu
#SBATCH -p gpu
#SBATCH -q gpu
#SBATCH --gres=gpu:l40:1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH -o /share/hpcproject/zzhang66/isaacnet/logs/isaac_gpu.%j.log
# Isaac Lab 3.0 kit-less (Newton, OV PhysX) in an Apptainer glibc-2.36 userland on a Hazel GPU node.
# GPU nodes are offline: ISAACSIM_ASSET_ROOT points at a local mirror of the S3 asset tree (prefetch_assets.sbatch).
set -uxo pipefail
B=/share/hpcproject/zzhang66/isaacnet
module load apptainer/1.4.2-1
OUT=$B/results/isaac_$SLURM_JOB_ID; mkdir -p $OUT; rm -f $OUT/DONE
X="apptainer exec --nv --bind /gpfs_common,/gpfs_common/share:/share --pwd $B/repo \
   --env TMPDIR=$B/tmp,PYTHONNOUSERSITE=1,XDG_CACHE_HOME=$B/xdgcache,TRITON_CACHE_DIR=$B/triton_cache,OMNI_KIT_ACCEPT_EULA=YES,ISAACSIM_ASSET_ROOT=$B/assets/Assets/Isaac/6.1,PATH=$B/envs/lab_ctr/bin:/usr/local/bin:/usr/bin:/bin \
   $B/sif/py312-bookworm.sif"
run() { local name=$1; shift; local T=$(date +%s); timeout 1200 $X "$@" > $OUT/$name.log 2>&1; echo "$name rc=$? s=$(( $(date +%s)-T ))" | tee -a $OUT/summary.txt; }
hostname | tee -a $OUT/summary.txt
$X nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv | tee -a $OUT/summary.txt
$X python -c "import torch,triton;print(torch.__version__,torch.version.cuda,triton.__version__,torch.cuda.get_device_name())" | tee -a $OUT/summary.txt
# 1. official Isaac Lab smoke: headless cartpole, 16 envs x 10 iterations
run cartpole_newton isaaclab train --rl_library rsl_rl --task Isaac-Cartpole-Direct --num_envs 16 --max_iterations 10 physics=newton_mjwarp
run cartpole_ovphysx isaaclab train --rl_library rsl_rl --task Isaac-Cartpole-Direct --num_envs 16 --max_iterations 10 physics=ovphysx
run cartpole_newton_4096 isaaclab train --rl_library rsl_rl --task Isaac-Cartpole-Direct --num_envs 4096 --max_iterations 30 physics=newton_mjwarp
# 2. our fleet env on each physics backend (bench smoke, then the isaac-marked tests)
for P in isaacsim_physx newton ovphysx; do
  run fleet_bench_$P env ISAACLAB_NET_PHYSICS=$P python benchmarks/isaac/bench.py --num_envs 16 --num_robots 4 --level L2-legacy --backend triton --steps 20 --warmup 3 --max_time 60 --out $OUT/fleet_bench.jsonl
done
for P in newton ovphysx; do
  run pytest_isaac_$P env ISAACLAB_NET_PHYSICS=$P python -m pytest -m isaac tests/test_isaac_env.py -v --durations=10
done
# 3. core GPU suite in the Isaac Lab env (torch 2.12 cu130, Triton 3.7)
run pytest_gpu_labenv python -m pytest -m gpu -q
$X nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv | tee -a $OUT/summary.txt
touch $OUT/DONE
