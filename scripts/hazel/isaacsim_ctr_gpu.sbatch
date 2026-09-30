#!/bin/bash
#SBATCH -J inet-kit
#SBATCH -A hpcproject_gpu
#SBATCH -p gpu
#SBATCH -q gpu
#SBATCH --gres=gpu:l40:1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 03:00:00
#SBATCH -o /share/hpcproject/zzhang66/isaacnet/logs/isaacsim_ctr.%j.log
# Isaac Sim (Kit) + Isaac Lab through the official NGC image nvcr.io/nvidia/isaac-lab:3.0.0-rc1 (Isaac Sim 6.1.0-rc.26),
# pulled as a SIF by pull_isaaclab_container.sbatch. LAB=release binds the Isaac Lab release/3.0.0 checkout over the
# image's rc1 sources (/workspace/isaaclab), which the fleet env needs (see docs/isaac-lab-linux.md). Kit's writable directories are bound to $B/kit/*, HOME to $B/kit/home.
set -uxo pipefail
B=/share/hpcproject/zzhang66/isaacnet
module load apptainer/1.4.2-1
OUT=$B/results/kit${LAB:+_$LAB}_$SLURM_JOB_ID; mkdir -p $OUT; rm -f $OUT/DONE
EXTRA=""; [ "${LAB:-}" = release ] && EXTRA=",$B/IsaacLab:/workspace/isaaclab"
K=$B/kit; mkdir -p $K/home $K/cache $K/data $K/logs $K/run
X="apptainer exec --nv --writable-tmpfs --home $K/home \
   --bind /gpfs_common,/gpfs_common/share:/share,$K/cache:/isaac-sim/kit/cache,$K/data:/isaac-sim/kit/data,$K/logs:/isaac-sim/kit/logs$EXTRA \
   --env OMNI_KIT_ACCEPT_EULA=YES,ACCEPT_EULA=Y,PRIVACY_CONSENT=Y,TMPDIR=$B/tmp,PYTHONNOUSERSITE=1,TRITON_CACHE_DIR=$B/triton_cache,ISAACSIM_ASSET_ROOT=$B/assets/Assets/Isaac/6.1 \
   $B/sif/isaac-lab_3.0.0-rc1.sif"
PY=/isaac-sim/python.sh
run() { local name=$1; shift; local T=$(date +%s); timeout 1200 $X "$@" > $OUT/$name.log 2>&1; echo "$name rc=$? s=$(( $(date +%s)-T ))" | tee -a $OUT/summary.txt; }
hostname | tee -a $OUT/summary.txt
$X nvidia-smi --query-gpu=name,driver_version --format=csv | tee -a $OUT/summary.txt
$X bash -c "ls /usr/share/vulkan/icd.d /etc/vulkan/icd.d 2>&1; ldconfig -p | grep -E 'libGLX_nvidia|libnvidia-glcore|libEGL_nvidia' | head" | tee -a $OUT/summary.txt
# 1. official smoke, Kit PhysX
run cartpole_kit bash -c "cd $K/run && /workspace/isaaclab/isaaclab.sh train --rl_library rsl_rl --task Isaac-Cartpole-Direct --num_envs 16 --max_iterations 10 physics=isaacsim_physx"
# 2. our fleet env on the default physics (PhysX through Isaac Sim), as on Windows
run fleet_check_kit bash -c "cd $B/repo && $PY tests/scripts/isaac_fleet_check.py --level L2-legacy --backend eager"
run fleet_bench_kit bash -c "cd $B/repo && $PY benchmarks/isaac/bench.py --num_envs 16 --num_robots 4 --level L2-legacy --backend graph --steps 20 --warmup 3 --max_time 60 --out $OUT/fleet_bench.jsonl"
if grep -q "fleet_check_kit rc=0" $OUT/summary.txt; then
  run pytest_isaac_kit bash -c "cd $B/repo && $PY -m pytest -m isaac tests/test_isaac_env.py -v -p no:cacheprovider"
fi
touch $OUT/DONE
