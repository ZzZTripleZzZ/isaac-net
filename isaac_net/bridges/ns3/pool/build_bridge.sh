#!/bin/bash
# Compile netslot-bridge.cc against the copied, already-built ns-3.48 + nr v5.1 libraries
# (same flags as the ns-3 build of netslot-ref; no ns-3 reconfigure needed).
set -e
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # this directory: the C++ sources
ROOT=${BRIDGE_ROOT:-/home/zzhang66/experiments/bridge_parallel}   # build root: ns-3.48 copy, bin/
NS=$ROOT/ns-3.48
ENVP=${NS3_TOOLCHAIN_ENV:-/home/zzhang66/experiments/ns3ref/env}
CXX=$ENVP/bin/x86_64-conda-linux-gnu-g++
mkdir -p $ROOT/bin
LIBS=""
for m in nr config-store csma virtual-net-device spectrum buildings propagation mobility antenna internet-apps point-to-point flow-monitor applications internet traffic-control bridge network stats core; do
  LIBS="$LIBS $NS/build/lib/libns3.48-$m-optimized.so"
done
$CXX -DEIGEN_MPL2_ONLY -DHAVE_EIGEN3 -DHAVE_GSL -DHAVE_SQLITE3 -DNS3_BUILD_PROFILE_OPTIMIZED \
  -DPROJECT_SOURCE_PATH=\"$NS\" -DSTACKTRACE_LIBRARY_IS_LINKED=1 -D__LINUX__ \
  -I$NS/build/include -isystem $ENVP/include -isystem $ENVP/include/eigen3 \
  -O3 -DNDEBUG -std=c++23 -fPIE -fno-semantic-interposition -Wall -march=native -mtune=native \
  $SRC/netslot-bridge.cc -o $ROOT/bin/netslot-bridge.new \
  -Wl,-rpath,$NS/build/lib:$ENVP/lib -L$NS/build/lib -Wl,--no-as-needed $LIBS -Wl,--as-needed \
  $ENVP/lib/libgsl.so $ENVP/lib/libgslcblas.so $ENVP/lib/libsqlite3.so -lpthread -lstdc++exp
mv $ROOT/bin/netslot-bridge.new $ROOT/bin/netslot-bridge   # atomic: running workers keep the old inode
# standalone reference binary from the same copy (unmodified netslot-ref), rpath -> our copy
cp $NS/build/scratch/netslot-ref/ns3.48-netslot-ref-optimized $ROOT/bin/netslot-ref.new && ~/anaconda3/bin/patchelf --set-rpath $NS/build/lib:$ENVP/lib $ROOT/bin/netslot-ref.new && mv $ROOT/bin/netslot-ref.new $ROOT/bin/netslot-ref
echo BUILD_OK
