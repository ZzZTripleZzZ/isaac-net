#!/bin/bash
#SBATCH -J inet-assets
#SBATCH -p xfer
#SBATCH -n 1
#SBATCH -t 01:00:00
#SBATCH -o /share/hpcproject/zzhang66/isaacnet/logs/prefetch.%j.log
# GPU nodes have no internet: mirror the cloud USD assets the smoke runs need into $TMPDIR, where Isaac Lab
# looks for local copies before going online.
set -uxo pipefail
B=/share/hpcproject/zzhang66/isaacnet
module load apptainer/1.4.2-1
rm -f $B/logs/prefetch.DONE
apptainer exec --bind /gpfs_common,/gpfs_common/share:/share --env TMPDIR=$B/tmp,PYTHONNOUSERSITE=1,XDG_CACHE_HOME=$B/xdgcache \
  $B/sif/py312-bookworm.sif $B/envs/lab_ctr/bin/python - <<'PY'
import tempfile
from isaaclab.utils.assets import retrieve_file_path, ISAACLAB_NUCLEUS_DIR, ISAAC_NUCLEUS_DIR
from isaaclab.sim.spawners.from_files.from_files_cfg import _DEFAULT_GROUND_PLANE_USD
print("tempdir", tempfile.gettempdir(), "root", ISAAC_NUCLEUS_DIR)
for url in [_DEFAULT_GROUND_PLANE_USD, f"{ISAACLAB_NUCLEUS_DIR}/Robots/Classic/Cartpole/cartpole.usd"]:
    print(url, "->", retrieve_file_path(url))
PY
echo "prefetch_rc=$?"
find $B/tmp -maxdepth 6 -type d | head -30
touch $B/logs/prefetch.DONE
