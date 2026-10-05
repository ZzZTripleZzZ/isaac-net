#!/bin/bash
R=/home/zzhang66/experiments/lenafid_rerun1005; cd $R
source /home/zzhang66/anaconda3/etc/profile.d/conda.sh && conda activate i5g
export ISAAC_NET_LENA_TABLES=/home/zzhang66/experiments/loadfix/lena_eesm_tables.npz NRF_THREADS=2
F=/home/zzhang66/experiments/isaac-net/benchmarks/fidelity
# CPU: v1 N=64 at base / B8 / B9
for c in 2bb77f8 415c7ff 1f3c90d; do
  ( s=$(date +%s); CUDA_VISIBLE_DEVICES= PYTHONPATH=$R/wt_$c NRF_PRESET=lena_validation python $F/nr_replay.py data bisect_$c primary 4 64 rng=global > logs/bisect_${c}_primary64.log 2>&1; echo "$(date +%T) cpu $c primary N=64 $(( $(date +%s)-s )) s rc=$?" ) &
  ( s=$(date +%s); CUDA_VISIBLE_DEVICES= PYTHONPATH=$R/wt_$c NRF_PRESET=lena_validation python $F/nr_replay.py data bisect_$c primary_eng 4 64 > logs/bisect_${c}_eng64.log 2>&1; echo "$(date +%T) cpu $c primary_eng N=64 $(( $(date +%s)-s )) s rc=$?" ) &
done
# GPU: the three changed scale arms at each commit
for c in 2bb77f8 415c7ff 1f3c90d d153c37 a002df8; do
  s=$(date +%s)
  for a in "v1_triton|lena_validation|" "nrconfig|NRConfig|" "nrconfig_nofade|NRConfig|fading=False"; do
    IFS="|" read -r arm preset ov <<< "$a"
    for N in 64 32 16 8 4 2 1; do
      PYTHONPATH=$R/wt_$c NRF_PRESET=$preset python $F/scalecfg/nr_replay_fast.py data bisect_$c $arm 4 $N triton $ov > logs/bisect_${c}_${arm}_$N.log 2>&1 || echo "FAIL $c $arm $N"
    done
  done
  echo "$(date +%T) gpu $c $(( $(date +%s)-s )) s"
done
wait
echo "$(date +%T) BISECT_DONE"
