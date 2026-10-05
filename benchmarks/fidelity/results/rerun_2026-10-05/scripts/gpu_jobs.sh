#!/bin/bash
cd /home/zzhang66/experiments/lenafid_rerun1005
export REPO=/home/zzhang66/experiments/isaac-net PYTHONPATH=/home/zzhang66/experiments/isaac-net
s=$(date +%s); OUT=replay_scalecfg bash $REPO/benchmarks/fidelity/scalecfg/run_scalecfg.sh; echo "$(date +%T) scalecfg $(( $(date +%s)-s )) s"
s=$(date +%s); python $REPO/benchmarks/fidelity/legacy_replay.py data replay legacy_graph 4 cuda graph > logs/legacy.log 2>&1; echo "$(date +%T) legacy $(( $(date +%s)-s )) s rc=$?"
s=$(date +%s); python $REPO/benchmarks/fidelity/legacy_replay.py data_fade replay_fade legacy_graph 4 cuda graph > logs/legacy_fade.log 2>&1; echo "$(date +%T) legacy_fade $(( $(date +%s)-s )) s"
echo "$(date +%T) GPU_DONE"
