#!/bin/bash
#SBATCH -J inet-labctr
#SBATCH -p xfer
#SBATCH -n 1
#SBATCH -t 04:00:00
#SBATCH -o /share/hpcproject/zzhang66/isaacnet/logs/install_lab_ctr.%j.log
# Kit-less Isaac Lab 3.0 (release/3.0.0: Newton + OV PhysX + rsl-rl, no Isaac Sim) plus isaaclab-net, installed
# inside a glibc 2.36 userland (python:3.12-bookworm under Apptainer), because Isaac Lab's core dependencies
# omniverseclient and usd-exchange ship only manylinux_2_35 wheels and RHEL 9 has glibc 2.34.
# Runs on the xfer partition (internet). Prerequisites: uv unpacked in $B/tools (install_core.sbatch does it) and
# the isaaclab-net checkout in $B/repo. Edit B (and the -o path) for another cluster or user.
set -uxo pipefail
B=/share/hpcproject/zzhang66/isaacnet
UV=$B/tools/uv-x86_64-unknown-linux-gnu/uv
module load apptainer/1.4.2-1
export APPTAINER_CACHEDIR=$B/apptainer_cache APPTAINER_TMPDIR=$B/tmp_apptainer
export PYTHONNOUSERSITE=1 PIP_NO_CACHE_DIR=1 HF_HOME=$B/hf
export UV_CACHE_DIR=$B/uvcache UV_PYTHON_INSTALL_DIR=$B/uvpython_ctr XDG_CACHE_HOME=$B/xdgcache TMPDIR=$B/tmp
# /share is a symlink to /gpfs_common/share on Hazel: bind both so every path resolves inside the container
BIND=/gpfs_common,/gpfs_common/share:/share
rm -f $B/logs/install_lab_ctr.DONE
mkdir -p $B/sif $B/envs $APPTAINER_TMPDIR $TMPDIR
cd $B
# Isaac Lab release/3.0.0 (validated at 60e28c1); git-lfs holds only docs media and test images
[ -d IsaacLab ] || GIT_LFS_SKIP_SMUDGE=1 git -c filter.lfs.smudge= -c filter.lfs.required=false \
    clone --branch release/3.0.0 https://github.com/isaac-sim/IsaacLab.git
(cd IsaacLab && git log -1 --format='IsaacLab %H %cd')
cd $B/sif
[ -f py312-bookworm.sif ] || apptainer pull py312-bookworm.sif docker://python:3.12-bookworm; echo "pull_rc=$?"
apptainer exec --bind $BIND $B/sif/py312-bookworm.sif bash -lc "ldd --version | head -1; git --version"
T0=$(date +%s)
apptainer exec --bind $BIND --env UV_CACHE_DIR=$UV_CACHE_DIR,UV_PYTHON_INSTALL_DIR=$UV_PYTHON_INSTALL_DIR,UV_PROJECT_ENVIRONMENT=$B/envs/lab_ctr,TMPDIR=$TMPDIR,PYTHONNOUSERSITE=1 \
  $B/sif/py312-bookworm.sif bash -c "cd $B/IsaacLab && $UV sync --extra rsl-rl --extra ovphysx && \
  $UV pip install --python $B/envs/lab_ctr/bin/python --no-deps -e $B/repo && \
  $UV pip install --python $B/envs/lab_ctr/bin/python pytest"; echo "ctr_sync_rc=$? s=$(( $(date +%s)-T0 ))"
apptainer exec --bind $BIND $B/sif/py312-bookworm.sif $B/envs/lab_ctr/bin/python - <<'PY'
import importlib, torch
print("torch", torch.__version__, torch.version.cuda)
for m in ["isaaclab", "isaaclab_tasks", "isaaclab_newton", "isaaclab_physx", "isaaclab_ov", "ovphysx", "newton", "warp",
          "mujoco_warp", "rsl_rl", "triton", "isaacsim", "isaaclab_net"]:
    try:
        mod = importlib.import_module(m); print("OK  ", m, getattr(mod, "__version__", ""))
    except Exception as e:
        print("FAIL", m, type(e).__name__, str(e)[:200])     # isaacsim is expected to fail (kit-less)
PY
$UV pip freeze --python $B/envs/lab_ctr/bin/python > $B/logs/lab_ctr_freeze.txt 2>&1 || true
touch $B/logs/install_lab_ctr.DONE
