#!/bin/bash
#SBATCH -J inet-inst
#SBATCH -p xfer
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -o /share/hpcproject/zzhang66/isaacnet/logs/install_core.%j.log
# Core isaac-net on Linux: Python 3.11 uv venv, torch 2.14 cu126 (bundles Triton), editable install with dev extras.
# Runs on the xfer partition (internet, no GPU). Prerequisite: the isaac-net checkout in $B/repo, e.g.
#   git archive main | ssh hazel "mkdir -p $B/repo && tar x -C $B/repo"
# Edit B (and the -o path) for another cluster or user.
set -euxo pipefail
B=/share/hpcproject/zzhang66/isaacnet
export PYTHONNOUSERSITE=1 PIP_NO_CACHE_DIR=1 CONDA_PKGS_DIRS=$B/pkgs HF_HOME=$B/hf
export UV_CACHE_DIR=$B/uvcache UV_PYTHON_INSTALL_DIR=$B/uvpython XDG_CACHE_HOME=$B/xdgcache
export PATH=$B/tools/uv-x86_64-unknown-linux-gnu:$PATH
rm -f $B/logs/install_core.DONE
mkdir -p $B/tools $B/envs $B/logs
[ -x $B/tools/uv-x86_64-unknown-linux-gnu/uv ] || curl -sL https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-unknown-linux-gnu.tar.gz | tar xz -C $B/tools
T0=$(date +%s)
uv python install 3.11
uv venv --python 3.11 --seed $B/envs/core
source $B/envs/core/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cu126
uv pip install triton numpy      # already satisfied by the torch wheel's pins; kept explicit
cd $B/repo && uv pip install -e ".[dev]"
python -c "import torch, triton, numpy, isaac_net; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'triton', triton.__version__, 'numpy', numpy.__version__)"
uv pip freeze > $B/logs/core_freeze.txt
echo "install_s=$(( $(date +%s) - T0 ))"
touch $B/logs/install_core.DONE
