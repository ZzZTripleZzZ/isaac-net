#!/usr/bin/env bash
# The configurations of the scale runs replayed against the 153 5G-LENA runs, on the GPU (fast backends):
#   v1_triton        lena_validation() on triton (checks this replay against the CPU primary_eng arm)
#   v2_graph         lena_validation_v2() on graph (checks this replay against the CPU v2 arm; triton refuses v2)
#   nrconfig         NRConfig() exactly: the configuration of the Isaac Lab and network-only NR speed rows
#   nrconfig_nofade  NRConfig(fading=False): splits the fading channel from the MAC and PHY defaults
#   v2_lumped40      lena_validation_v2(ul_grant_model="lumped", sr_grant_delay_slots=40): v2 minus the SR / BSR
#                    grant pipeline, the closest configuration the triton kernel accepts
#   v2_lumped40_fb16 the same with frame_buffer=16, the task's buffer depth: the configuration of the scale runs
# Run from a directory holding data/ (lena_extract.py output); REPO points at an isaaclab-net checkout.
#   ISAACLAB_NET_LENA_TABLES=<local 5G-LENA EESM tables> REPO=... bash run_scalecfg.sh
set -u
REPO=${REPO:-repo}
REPS=${REPS:-4}
OUT=${OUT:-replay_scalecfg}
export PYTHONPATH=$REPO:${PYTHONPATH:-}
ARMS=(
  "v1_triton|lena_validation|triton|"
  "v2_graph|lena_validation_v2|graph|"
  "nrconfig|NRConfig|triton|"
  "nrconfig_nofade|NRConfig|triton|fading=False"
  "v2_lumped40|lena_validation_v2|triton|ul_grant_model=lumped sr_grant_delay_slots=40"
  "v2_lumped40_fb16|lena_validation_v2|triton|ul_grant_model=lumped sr_grant_delay_slots=40 frame_buffer=16"
)
mkdir -p "$OUT" logs
for a in "${ARMS[@]}"; do
  IFS="|" read -r arm preset backend ov <<< "$a"
  for N in 64 32 16 8 4 2 1; do
    [ -f "$OUT/$arm/.done_$N" ] && continue
    # shellcheck disable=SC2086
    NRF_PRESET=$preset python "$REPO/benchmarks/fidelity/scalecfg/nr_replay_fast.py" data "$OUT" "$arm" "$REPS" "$N" \
      "$backend" $ov > "logs/${arm}_$N.log" 2>&1 && touch "$OUT/$arm/.done_$N"
  done
done
touch "$OUT/ALL_DONE"
