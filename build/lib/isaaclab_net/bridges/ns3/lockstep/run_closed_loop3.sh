#!/bin/bash
cd /home/zzhang66/experiments/bridge_lockstep/tests; source ~/anaconda3/etc/profile.d/conda.sh && conda activate i5g
python test_closed_loop.py --E 2 --R 16 --variants shm-procs,tcp-single >> ../results/closed_loop.out 2>&1
python test_closed_loop.py --E 4 --R 8,16 >> ../results/closed_loop.out 2>&1
python test_closed_loop.py --E 2 --R 8 --policy none --variants tcp-procs,shm-procs,tcp-single >> ../results/closed_loop.out 2>&1
touch ../results/CL_DONE
