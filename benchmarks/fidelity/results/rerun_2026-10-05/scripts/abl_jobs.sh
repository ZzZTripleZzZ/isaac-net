#!/bin/bash
# v1 ablation arms (run_all.sh), SR grid + fading arms (run_extra.sh), global RNG as in the original campaign
R=/home/zzhang66/experiments/lenafid_rerun1005; cd $R
source /home/zzhang66/anaconda3/etc/profile.d/conda.sh && conda activate i5g
export ISAAC_NET_LENA_TABLES=/home/zzhang66/experiments/loadfix/lena_eesm_tables.npz
export REPO=/home/zzhang66/experiments/isaac-net PYTHONPATH=/home/zzhang66/experiments/isaac-net NRF_THREADS=2 CUDA_VISIBLE_DEVICES=
F=$REPO/benchmarks/fidelity
ARMS=("olla_on|olla=True" "phr_cap_wholeband|phr_cap=True" "ul_power_alloc|ul_power=allocated"
  "ul_power_alloc_phr|ul_power=allocated phr_cap=True" "harq1|n_harq=1" "sr_default3|sr_grant_delay_slots=None"
  "sr_measured14|sr_grant_delay_slots=14" "bler_sionna|bler_source=pdsch")
for v in 8 20 26 32 36 44 50; do ARMS+=("sr_grid$v|sr_grant_delay_slots=$v"); done
jobs=()
for N in 64 32 16 8 4 2 1; do
  for a in "${ARMS[@]}"; do jobs+=("data|replay|${a%%|*}|$N|${a#*|} rng=global"); done
  case $N in 64|16|4) jobs+=("data_fade|replay_fade|fade_matched|$N|fading=True ue_speed_mps=3.0 rng=global")
                      jobs+=("data_fade|replay_fade|fade_olla|$N|fading=True ue_speed_mps=3.0 olla=True rng=global");; esac
done
printf '%s\n' "${jobs[@]}" | xargs -P 10 -I{} bash -c '
  IFS="|" read -r data out arm N ov <<< "{}"
  [ -f $out/$arm/.done_$N ] && exit 0
  s=$(date +%s)
  python '"$F"'/nr_replay.py $data $out $arm 4 $N $ov > logs/${out}_${arm}_$N.log 2>&1 \
    && touch $out/$arm/.done_$N && echo "$(date +%T) done $out/$arm N=$N $(( $(date +%s)-s )) s" || echo "FAIL $out/$arm N=$N"
'
echo "$(date +%T) ABL_DONE"
