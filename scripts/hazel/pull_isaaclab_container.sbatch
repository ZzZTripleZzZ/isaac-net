#!/bin/bash
#SBATCH -J inet-pull
#SBATCH -p xfer
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=22G
#SBATCH -t 06:00:00
#SBATCH -o /share/hpcproject/zzhang66/isaacnet/logs/pull.%j.log
set -uxo pipefail
B=/share/hpcproject/zzhang66/isaacnet
module load apptainer/1.4.2-1
export APPTAINER_CACHEDIR=$B/apptainer_cache APPTAINER_TMPDIR=$B/tmp_apptainer
mkdir -p $APPTAINER_CACHEDIR $APPTAINER_TMPDIR $B/sif; rm -f $B/logs/pull.DONE
apptainer --version
# mksquashfs was OOM-killed at the xfer default memory; 22G (the xfer QOS cap is 24G per user) (layers stay cached in APPTAINER_CACHEDIR)
cd $B/sif
T0=$(date +%s)
[ -f isaac-lab_3.0.0-rc1.sif ] || apptainer pull isaac-lab_3.0.0-rc1.sif docker://nvcr.io/nvidia/isaac-lab:3.0.0-rc1; echo "pull_lab_rc=$? s=$(( $(date +%s) - T0 ))"
# fallback if mksquashfs still dies: an unpacked sandbox directory needs no squashfs step
if [ ! -f isaac-lab_3.0.0-rc1.sif ] && [ ! -d isaac-lab_3.0.0-rc1.sandbox ]; then
  T0=$(date +%s); apptainer build --sandbox isaac-lab_3.0.0-rc1.sandbox docker://nvcr.io/nvidia/isaac-lab:3.0.0-rc1; echo "sandbox_rc=$? s=$(( $(date +%s) - T0 ))"
fi
ls -la $B/sif
touch $B/logs/pull.DONE
