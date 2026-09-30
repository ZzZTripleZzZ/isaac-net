#!/usr/bin/env bash
# Full fidelity campaign on the lab box (CPU only, reference backend). Run from the directory that holds
# data/ (lena_extract.py output); REPO points at an isaaclab-net checkout. Results go to replay/<arm>/.
#   ISAACLAB_NET_LENA_TABLES=<local 5G-LENA EESM tables> REPO=... bash run_all.sh
set -u
REPO=${REPO:-repo}
REPS=${REPS:-4}
PAR=${PAR:-8}
export PYTHONPATH=$REPO NRF_THREADS=${NRF_THREADS:-2}
# arm name | overrides of lena_validation(); one switch per ablation arm
ARMS=(
  "primary|"
  "olla_on|olla=True"
  "phr_cap_wholeband|phr_cap=True"
  "ul_power_alloc|ul_power=allocated"
  "ul_power_alloc_phr|ul_power=allocated phr_cap=True"
  "harq1|n_harq=1"
  "sr_default3|sr_grant_delay_slots=None"
  "sr_measured14|sr_grant_delay_slots=14"
  "bler_sionna|bler_source=pdsch"
)
jobs=()
for N in 64 32 16 8 4 2 1; do
  for a in "${ARMS[@]}"; do
    jobs+=("${a%%|*}|$N|${a#*|}")
  done
done
mkdir -p replay logs
printf '%s\n' "${jobs[@]}" | xargs -P "$PAR" -I{} bash -c '
  IFS="|" read -r arm N ov <<< "{}"
  [ -f replay/$arm/.done_$N ] && exit 0
  python $PYTHONPATH/benchmarks/fidelity/nr_replay.py data replay $arm '"$REPS"' $N $ov > logs/${arm}_$N.log 2>&1 \
    && touch replay/$arm/.done_$N
'
touch ALL_DONE
