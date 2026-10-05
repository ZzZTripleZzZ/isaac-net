#!/bin/bash
# CPU reference replays: v1 (global RNG = paper tables), v1 engine RNG, v2, v2 global RNG, held-out v2
cd /home/zzhang66/experiments/lenafid_rerun1005
export REPO=/home/zzhang66/experiments/isaac-net PYTHONPATH=/home/zzhang66/experiments/isaac-net NRF_THREADS=2 CUDA_VISIBLE_DEVICES=
F=$REPO/benchmarks/fidelity
mkdir -p replay logs heldout/replay heldout/logs
jobs=()
for N in 64 32 16 8 4 2 1; do
  jobs+=("data|replay|primary|lena_validation|$N|rng=global")
  jobs+=("data|replay|primary_eng|lena_validation|$N|")
  jobs+=("data|replay|v2|lena_validation_v2|$N|")
  jobs+=("data|replay|v2_global|lena_validation_v2|$N|rng=global")
  case $N in 32|16|8) jobs+=("heldout/data|heldout/replay|v2|lena_validation_v2|$N|bandwidth_mhz=10 n_prb=20");; esac
done
printf '%s\n' "${jobs[@]}" | xargs -P 14 -I{} bash -c '
  IFS="|" read -r data out arm preset N ov <<< "{}"
  [ -f $out/$arm/.done_$N ] && exit 0
  s=$(date +%s)
  NRF_PRESET=$preset python '"$F"'/nr_replay.py $data $out $arm 4 $N $ov > logs/$(echo $out|tr / _)_${arm}_$N.log 2>&1 \
    && touch $out/$arm/.done_$N && echo "$(date +%T) done $out/$arm N=$N $(( $(date +%s)-s )) s"
'
echo "$(date +%T) CPU_DONE"
