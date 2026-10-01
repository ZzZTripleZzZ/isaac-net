#!/bin/bash
# Logs the host CPU state (WSL side) every 30 s while a marker file exists: load average and the five busiest
# processes with their user. Used next to the Isaac Lab runs, whose step is bound by host work.
# usage: bash cpu_log.sh <marker> <log>
M=$1; LOG=$2
while [ -e "$M" ]; do
  { echo "== $(date -Is) load $(cut -d' ' -f1-3 /proc/loadavg)"
    ps -eo user:12,pid,pcpu,etime,comm --sort=-pcpu | head -6; } >> "$LOG"
  sleep 30
done
echo "== $(date -Is) marker gone, stop" >> "$LOG"
