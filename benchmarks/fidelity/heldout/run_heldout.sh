#!/usr/bin/env bash
# Held-out 5G-LENA scenario, end to end (lab box). Needs the 5G-LENA reference program of the validation
# (NETSLOT_REF_BIN), its parse_run.py (PARSE_RUN), the local 5G-LENA tables (ISAAC_NET_LENA_TABLES) and REPO
# pointing at an isaac-net checkout. Run from an empty directory; ns-3 runs 8 at a time.
set -u
REPO=${REPO:-repo}
H=$REPO/benchmarks/fidelity/heldout
F=$REPO/benchmarks/fidelity
export PYTHONPATH=$REPO NRF_THREADS=${NRF_THREADS:-2}
python $H/make_manifest.py > manifest.txt
job() { d=$1; shift; [ -f $d/summary.json ] && exit 0; rm -rf $d; mkdir -p $d
  (cd $d && $NETSLOT_REF_BIN --outDir=. "$@" > stdout.txt 2>&1); python3 $PARSE_RUN $d > /dev/null 2>&1 || echo "PARSE_FAIL $d"; }
export -f job
xargs -P 8 -L 1 bash -c 'job "$@"' _ < manifest.txt
python $F/lena_extract.py sweep data > extract.log 2>&1
mkdir -p replay logs
for N in 32 16 8; do    # only the carrier fields of the v2 preset change
  NRF_PRESET=lena_validation_v2 python $F/nr_replay.py data replay v2 4 $N bandwidth_mhz=10 n_prb=20 > logs/v2_$N.log 2>&1 &
done
wait
python $F/compare.py data replay results > compare.log 2>&1
python $H/summarize.py results/per_run_v2.csv held-out > table.md
python $H/mcs_heldout.py sweep 20 10 > mcs_heldout.txt
