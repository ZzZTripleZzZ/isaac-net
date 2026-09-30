#!/bin/bash
# 5G-LENA outcome depends on process heap layout: the same command script, seed and binary give a
# different (statistically equivalent) sample path when only the length of an argv string changes.
# usage: heap_sensitivity.sh CMDS INIT RUN NUE
B=/home/zzhang66/experiments/bridge_parallel/bin/netslot-bridge
CMDS=$1; INIT=$2; RUN=$3; NUE=${4:-16}
W=$(mktemp -d); cp $CMDS $W/c; mkdir -p $W/a_much_longer_directory_name_to_move_the_heap
cd $W
common="--nUe=$NUE --init=$INIT --run=$RUN --outDir=/tmp --macTraces=0 --pktLog=0 --flowmon=0 --trafficTime=0 --ueUeFilter=1"
$B --io=file:c:r1 $common
$B --io=file:c:r2 $common
$B --io=file:c:a_much_longer_directory_name_to_move_the_heap/r3 $common
for f in r1 r2 a_much_longer_directory_name_to_move_the_heap/r3; do echo "$f $(grep ^D $f | cut -d' ' -f1-2,4- | md5sum | cut -c1-12)"; done
python3 - <<'P'
a=[l.split() for l in open('r1') if l.startswith('D')]; b=[l.split() for l in open('a_much_longer_directory_name_to_move_the_heap/r3') if l.startswith('D')]
k=next((i for i,(x,y) in enumerate(zip(a,b)) if x[:2]+x[3:]!=y[:2]+y[3:]),None)
print("short vs long argv: identical" if k is None else f"short vs long argv: first difference at step {k}")
P
rm -rf $W
