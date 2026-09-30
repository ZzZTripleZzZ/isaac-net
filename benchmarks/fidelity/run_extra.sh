#!/usr/bin/env bash
# Second part of the campaign (after run_all.sh): the SR->grant grid for the fit / hold-out split, and the
# fading-ON arm. Same conventions as run_all.sh. data_fade/ is lena_extract.py run on the ns3ref fade arm.
set -u
REPO=${REPO:-repo}
REPS=${REPS:-4}
PAR=${PAR:-8}
export PYTHONPATH=$REPO NRF_THREADS=${NRF_THREADS:-2}
jobs=()
# SR -> first-PUSCH grid (3, 14 and 40 slots are run_all.sh's sr_default3, sr_measured14 and primary)
for N in 64 32 16 8 4 2 1; do
  for v in 8 20 26 32 36 44 50; do jobs+=("data|replay|sr_grid$v|$N|sr_grant_delay_slots=$v"); done
done
# fading ON: engine AR(1) Rayleigh at 3 m/s (LENA: 38.901 UMi NLOS at 3 m/s), LENA-matched (OLLA off) and OLLA on
for N in 64 16 4; do
  jobs+=("data_fade|replay_fade|fade_matched|$N|fading=True ue_speed_mps=3.0")
  jobs+=("data_fade|replay_fade|fade_olla|$N|fading=True ue_speed_mps=3.0 olla=True")
done
mkdir -p replay replay_fade logs
printf '%s\n' "${jobs[@]}" | xargs -P "$PAR" -I{} bash -c '
  IFS="|" read -r data out arm N ov <<< "{}"
  [ -f $out/$arm/.done_$N ] && exit 0
  python $PYTHONPATH/benchmarks/fidelity/nr_replay.py $data $out $arm '"$REPS"' $N $ov > logs/${arm}_$N.log 2>&1 \
    && touch $out/$arm/.done_$N
'
touch EXTRA_DONE
