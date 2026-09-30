#!/usr/bin/env bash
# The engine-integrated 5G-LENA MAC switches over the 153 primary-arm runs (CPU, reference backend):
#   v2           lena_validation_v2() (every switch on, engine RNG: the default)
#   v2_global    the same with rng="global" (the prototype's random stream: bitwise the prototype's `all` arm)
#   primary_eng  lena_validation() with the engine RNG (the v1 baseline under today's default RNG)
# Run from a directory holding data/ (lena_extract.py output); REPO points at an isaaclab-net checkout.
#   ISAACLAB_NET_LENA_TABLES=<local 5G-LENA EESM tables> REPO=... bash run_v2.sh
set -u
REPO=${REPO:-repo}
REPS=${REPS:-4}
PAR=${PAR:-8}
OUT=${OUT:-replay}
export PYTHONPATH=$REPO NRF_THREADS=${NRF_THREADS:-2}
ARMS=(
  "v2|lena_validation_v2|"
  "v2_global|lena_validation_v2|rng=global"
  "primary_eng|lena_validation|"
)
jobs=()
for N in 64 32 16 8 4 2 1; do
  for a in "${ARMS[@]}"; do
    jobs+=("$a|$N")
  done
done
mkdir -p $OUT logs
printf '%s\n' "${jobs[@]}" | xargs -P "$PAR" -I{} bash -c '
  IFS="|" read -r arm preset ov N <<< "{}"
  [ -f '"$OUT"'/$arm/.done_$N ] && exit 0
  NRF_PRESET=$preset python $PYTHONPATH/benchmarks/fidelity/nr_replay.py data '"$OUT"' $arm '"$REPS"' $N $ov \
    > logs/${arm}_$N.log 2>&1 && touch '"$OUT"'/$arm/.done_$N
'
touch $OUT/ALL_DONE
