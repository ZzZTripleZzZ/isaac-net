#!/bin/bash
# Speed benchmark (<= 8 ns-3 processes at a time) + WSL-internal run of the Windows client for comparison.
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # this directory: the C++ sources
B=${NS3BRIDGE_ROOT:-/home/zzhang66/experiments/bridge_lockstep}   # build root: ns-3.48 copy, envrc.sh, bin/
cd $B/tests; source ~/anaconda3/etc/profile.d/conda.sh && conda activate i5g
rm -f $B/results/bench.jsonl
python bench_speed.py --configs tcp-procs,unix-procs,tcp-single,unix-single --E 1,2,4,8 --R 8,16 > $B/results/bench.out 2>&1
python bench_speed.py --configs shm-procs --E 1,2,4 --R 8,16 >> $B/results/bench.out 2>&1
touch $B/results/BENCH_DONE
