#!/bin/bash
# Start E single-env ns-3 bridge servers on TCP ports BASE..BASE+E-1 (for clients outside WSL, e.g.
# Isaac Sim on Windows via localhost forwarding). Each server exits after its client sends CLOSE.
# Usage: bridges/ns3/lockstep/serve_wsl.sh E R BASEPORT [extra ns-3 args...]
E=${1:-1}; R=${2:-8}; BASE=${3:-57100}; shift 3
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # this directory: the C++ sources
B=${NS3BRIDGE_ROOT:-/home/zzhang66/experiments/bridge_lockstep}   # build root: ns-3.48 copy, envrc.sh, bin/
export LD_LIBRARY_PATH=$B/ns-3.48/build/lib:/home/zzhang66/experiments/ns3ref/env/lib
mkdir -p $B/logs
for ((e=0; e<E; e++)); do
  P=$((BASE+e))
  setsid nohup $B/bin/netslot-bridge --bridge=tcp:$P --nEnv=1 --nUe=$R --run=$((1+e)) "$@" \
     > $B/logs/serve_$P.log 2>&1 < /dev/null &
done
sleep 1; grep -h LISTENING $B/logs/serve_*.log | tail -n $E
