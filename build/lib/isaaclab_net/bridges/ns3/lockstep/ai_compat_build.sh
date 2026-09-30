#!/bin/bash
# Compatibility check: build ns3-ai (latest main) as a contrib module of ns-3.48 with the ns3ref toolchain.
# Separate source tree (ns-3.48-ai), modules limited to ai + core deps, examples on (a-plus-b pybind modules).
set -x
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # this directory: the C++ sources
B=${NS3BRIDGE_ROOT:-/home/zzhang66/experiments/bridge_lockstep}   # build root: ns-3.48 copy, envrc.sh, bin/
cd $B
source $B/envrc.sh
export PYTHONPATH=$B/pydeps:$PYTHONPATH
mkdir -p $B/sysboost && ln -sfn /usr/include/boost $B/sysboost/boost
if [ ! -d ns-3.48-ai ]; then
  rsync -a --exclude build --exclude cmake-cache --exclude contrib/nr /home/zzhang66/experiments/ns3ref/ns-3.48/ ns-3.48-ai/
  cp -a ns3-ai-src ns-3.48-ai/contrib/ai
fi
cd ns-3.48-ai
PYB=$(python3 -c "import sys; sys.path.insert(0,'$B/pydeps'); import pybind11; print(pybind11.get_cmake_dir())")
./ns3 configure -d optimized --enable-examples --disable-tests --disable-python-bindings -G Ninja \
  --enable-modules "ai" \
  -- -DCMAKE_PREFIX_PATH="$B/pbenv;$ENVP" -DCMAKE_C_COMPILER=$CC -DCMAKE_CXX_COMPILER=$CXX -Dpybind11_DIR=$PYB \
     -DBoost_DIR=/usr/lib/x86_64-linux-gnu/cmake/Boost-1.71.0 -DPython_EXECUTABLE=/home/zzhang66/anaconda3/envs/i5g/bin/python -DCMAKE_EXE_LINKER_FLAGS="-Wl,--no-as-needed -lrt" -DCMAKE_SHARED_LINKER_FLAGS="-lrt" -DCMAKE_CXX_FLAGS="-isystem $B/sysboost" -DProtobuf_PROTOC_EXECUTABLE=$B/pbenv/bin/protoc > $B/results/ai_configure.log 2>&1
echo "CONFIGURE_RC=$?"
./ns3 build -j6 > $B/results/ai_build.log 2>&1
echo "BUILD_RC=$?"
touch $B/results/AI_BUILD_DONE
