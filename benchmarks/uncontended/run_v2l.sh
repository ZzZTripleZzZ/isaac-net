#!/bin/bash
# Network-only NR uplink on triton with the scale configuration of docs/fidelity-vs-lena.md ("Scale configurations"):
# --cfg ul_v2l, and --cfg ul (NRConfig()) in the same passes for a same-session comparison, at the four shapes of
# the campaign, 3 passes (3 processes per case), 3 windows each. Expects the marker to be held by the caller.
# usage: ISAACLAB_NET_LENA_TABLES=<tables> bash benchmarks/uncontended/run_v2l.sh <outdir>
set -u
D=$(cd "$(dirname "$0")" && pwd); ROOT=$(cd "$D/../.." && pwd); OUT=${1:-$D/results_v2l}
mkdir -p "$OUT"
cd "$ROOT"
export PYTHONPATH=$ROOT
for pass in 1 2 3; do
  for sz in "256 16" "1024 32" "4096 16" "4096 100"; do
    set -- $sz
    for cfg in ul_v2l ul; do
      echo "== $(date +%T) pass $pass $cfg $1 x $2"
      timeout 1800 python "$D/bench_net.py" --case L2 --backend triton --cfg $cfg --E "$1" --R "$2" \
        --out "$OUT/net_r$pass.jsonl" 2>> "$OUT/stderr.log" | tail -1
    done
  done
done
echo "DONE $(date -Is)"
