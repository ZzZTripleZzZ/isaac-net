#!/bin/bash
#SBATCH -J inet-scale
#SBATCH -A hpcproject_gpu
#SBATCH -p gpu
#SBATCH -q gpu
#SBATCH --gres=gpu:l40:1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 04:00:00
#SBATCH -o /share/hpcproject/zzhang66/isaacnet/logs/isaac_scale.%j.log
# Fleet env throughput, kit-less Isaac Lab 3.0 (Newton, OV PhysX), dedicated L40; plus bench_nr without the
# GPL 5G-LENA presets. Options: sbatch --export=ALL,PHYS=newton,SKIP_NR=1 isaac_scale.sbatch
set -uxo pipefail
B=/share/hpcproject/zzhang66/isaacnet
module load apptainer/1.4.2-1
OUT=$B/results/scale_$SLURM_JOB_ID; mkdir -p $OUT; rm -f $OUT/DONE
X="apptainer exec --nv --bind /gpfs_common,/gpfs_common/share:/share --pwd $B/repo \
   --env TMPDIR=$B/tmp,PYTHONNOUSERSITE=1,XDG_CACHE_HOME=$B/xdgcache,TRITON_CACHE_DIR=$B/triton_cache,OMNI_KIT_ACCEPT_EULA=YES,ISAACSIM_ASSET_ROOT=$B/assets/Assets/Isaac/6.1,PATH=$B/envs/lab_ctr/bin:/usr/local/bin:/usr/bin:/bin \
   $B/sif/py312-bookworm.sif"
run() { local name=$1; shift; local T=$(date +%s); timeout 1500 $X "$@" > $OUT/$name.log 2>&1; echo "$name rc=$? s=$(( $(date +%s)-T ))" | tee -a $OUT/summary.txt; }
hostname | tee -a $OUT/summary.txt
sed '/"NR lena_like/d' $B/repo/benchmarks/bench_nr.py > $OUT/bench_nr_nolena.py
[ -n "${SKIP_NR:-}" ] || run bench_nr python $OUT/bench_nr_nolena.py $OUT/bench_nr.json
for P in ${PHYS:-newton ovphysx}; do
  for ER in 256x16 1024x16 1024x64 4096x32; do
    E=${ER%x*}; R=${ER#*x}
    for L in off L2-legacy; do
      run fleet_${P}_${ER}_$L env ISAAC_NET_PHYSICS=$P python benchmarks/isaac/bench.py --num_envs $E --num_robots $R \
          --level $L --backend triton --steps 100 --warmup 20 --max_time 60 --out $OUT/fleet_scale.jsonl
    done
  done
done
$X nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv | tee -a $OUT/summary.txt
touch $OUT/DONE
