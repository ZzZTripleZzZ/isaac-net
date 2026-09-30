#!/bin/bash
# Uncontended campaign driver (lab box, WSL): runs every line of plan_net.txt through bench_net.py and every line of
# plan_env.txt through bench_env.py, one process at a time, skipping lines already in done.txt (resumable).
# Holds the marker /home/zzhang66/experiments/BENCH_RUNNING while it runs.
# usage: bash benchmarks/uncontended/run.sh <outdir>; run it three times (three outdirs) for three process repeats
set -u
D=$(cd "$(dirname "$0")" && pwd); ROOT=$(cd "$D/../.." && pwd); OUT=${1:-$D/results}
MARK=/home/zzhang66/experiments/BENCH_RUNNING
mkdir -p "$OUT"; touch "$OUT/done.txt"
echo "bench2 $(date -Is) pid $$" > $MARK
[ -z "${KEEP_MARK:-}" ] && trap 'rm -f $MARK' EXIT      # KEEP_MARK=1: a later phase still needs the GPU
cd "$ROOT"
export PYTHONPATH=$ROOT
python "$D/make_plan.py" "$OUT/plan_net.txt" "$OUT/plan_env.txt"
run_plan() {   # $1 plan, $2 script, $3 jsonl
  while read -r line; do
    [ -z "$line" ] && continue
    grep -qxF "$2 $line" "$OUT/done.txt" && continue
    echo "== $(date +%T) $2 $line"
    timeout 1800 python "$D/$2" $line --out "$OUT/$3" 2>> "$OUT/stderr.log" | tail -1
    rc=${PIPESTATUS[0]}
    [ $rc -ne 0 ] && echo "{\"args\": \"$line\", \"status\": \"rc=$rc\"}" >> "$OUT/$3"
    echo "$2 $line" >> "$OUT/done.txt"
  done < "$1"
}
run_plan "$OUT/plan_net.txt" bench_net.py net.jsonl
run_plan "$OUT/plan_env.txt" bench_env.py env.jsonl
echo "DONE $(date -Is)"
