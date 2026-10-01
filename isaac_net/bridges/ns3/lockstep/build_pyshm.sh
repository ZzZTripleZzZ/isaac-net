#!/bin/bash
# Build the Python side of the ns3-ai shm transport for the i5g interpreter (Python 3.11).
set -e
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # this directory: the C++ sources
B=${NS3BRIDGE_ROOT:-/home/zzhang66/experiments/bridge_lockstep}   # build root: ns-3.48 copy, envrc.sh, bin/
source $B/envrc.sh
source ~/anaconda3/etc/profile.d/conda.sh && conda activate i5g
PYINC=$(python -c "import sysconfig; print(sysconfig.get_paths()['include'])")
EXT=$(python -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
PYB=$B/pydeps/pybind11/include
mkdir -p $B/sysboost && ln -sfn /usr/include/boost $B/sysboost/boost
$CXX -O2 -shared -fPIC -std=c++20 -I$PYB -isystem $PYINC -isystem $B/sysboost \
  -I$B/ns3-ai-src/model/msg-interface -I$B/ns-3.48/build/include \
  -o $B/bin/ns3ai_bridge_py$EXT $SRC/pyshm/ns3ai_bridge_py.cc -lrt -lpthread \
  -static-libstdc++ -static-libgcc
echo BUILT $B/bin/ns3ai_bridge_py$EXT
