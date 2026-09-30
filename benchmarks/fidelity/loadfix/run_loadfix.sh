#!/usr/bin/env bash
# Load-gap campaign on the lab box (CPU, reference backend): every LoadFixNet arm over the 153 primary-arm runs.
# Run from a directory holding data/ (lena_extract.py output); REPO points at an isaaclab-net checkout.
#   ISAACLAB_NET_LENA_TABLES=<local 5G-LENA EESM tables> REPO=... ARMS="pf pipe all" bash run_loadfix.sh
set -u
REPO=${REPO:-repo}
REPS=${REPS:-4}
PAR=${PAR:-10}
NS=${NS:-"64 32 16 8 4 2 1"}
ARMS=${ARMS:-"pf pf_intra pf_active retx amc oh8 pipe pf_pipe all"}
OUT=${OUT:-replay}
export PYTHONPATH=$REPO NRF_THREADS=${NRF_THREADS:-2}
jobs=()
for N in $NS; do
  for a in $ARMS; do
    jobs+=("$a|$N")
  done
done
mkdir -p $OUT logs
printf '%s\n' "${jobs[@]}" | xargs -P "$PAR" -I{} bash -c '
  IFS="|" read -r arm N <<< "{}"
  [ -f '"$OUT"'/$arm/.done_$N ] && exit 0
  python $PYTHONPATH/benchmarks/fidelity/loadfix/loadfix_replay.py data '"$OUT"' $arm $arm '"$REPS"' $N > logs/${arm}_$N.log 2>&1 \
    && touch '"$OUT"'/$arm/.done_$N
'
touch $OUT/ALL_DONE
