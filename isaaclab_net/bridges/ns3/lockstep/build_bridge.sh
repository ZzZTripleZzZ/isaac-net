#!/bin/bash
# Build the bridge server against the copied ns-3.48 + NR v5.1 libraries (no ns-3 reconfigure needed).
# Usage: bridges/ns3/lockstep/build_bridge.sh [ai]   ("ai" adds the ns3-ai shared-memory transport)
set -e
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # this directory: the C++ sources
B=${NS3BRIDGE_ROOT:-/home/zzhang66/experiments/bridge_lockstep}   # build root: ns-3.48 copy, envrc.sh, bin/
source $B/envrc.sh        # ns3ref/env toolchain (read-only)
NS=$B/ns-3.48
mkdir -p $B/bin $B/obj
DEFS="-DEIGEN_MPL2_ONLY -DHAVE_EIGEN3 -DHAVE_GSL -DHAVE_SQLITE3 -DNS3_BUILD_PROFILE_OPTIMIZED -DPROJECT_SOURCE_PATH=\"$NS\" -DSTACKTRACE_LIBRARY_IS_LINKED=1 -D__LINUX__"
INC="-I$NS/build/include -isystem $ENVP/include -isystem $ENVP/include/eigen3"
OUT=$B/bin/netslot-bridge
EXTRA=""
if [ "$1" == "ai" ]; then
  mkdir -p $B/sysboost && ln -sfn /usr/include/boost $B/sysboost/boost
  EXTRA="-DWITH_NS3AI -I$B/ns3-ai-src/model/msg-interface -isystem $B/sysboost"
  OUT=$B/bin/netslot-bridge-ai
fi
LIBS=$(ls $NS/build/lib/libns3.48-*-optimized.so | tr '\n' ' ')
$CXX $DEFS $INC $EXTRA -O3 -DNDEBUG -std=c++23 -fPIE -Wall -march=native -mtune=native \
  -o $OUT $SRC/netslot-bridge.cc \
  -Wl,-rpath,$NS/build/lib -Wl,-rpath,$ENVP/lib -Wl,--no-as-needed $LIBS \
  $ENVP/lib/libgsl.so $ENVP/lib/libgslcblas.so $ENVP/lib/libsqlite3.so -lpthread -lrt -lstdc++exp
echo BUILT $OUT
