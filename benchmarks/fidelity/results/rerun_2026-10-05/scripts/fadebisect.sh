#!/bin/bash
R=/home/zzhang66/experiments/lenafid_rerun1005; cd $R
source /home/zzhang66/anaconda3/etc/profile.d/conda.sh && conda activate i5g
export ISAAC_NET_LENA_TABLES=/home/zzhang66/experiments/loadfix/lena_eesm_tables.npz NRF_THREADS=2 CUDA_VISIBLE_DEVICES=
F=/home/zzhang66/experiments/isaac-net/benchmarks/fidelity
for c in 415c7ff 1f3c90d; do for N in 64 16 4; do
 ( s=$(date +%s); PYTHONPATH=$R/wt_$c python $F/nr_replay.py data_fade fbisect_$c fade_matched 4 $N fading=True ue_speed_mps=3.0 rng=global > logs/fbisect_${c}_$N.log 2>&1; echo "$(date +%T) $c fade_matched N=$N $(( $(date +%s)-s )) s rc=$?" ) &
done; done; wait; echo FB_DONE
