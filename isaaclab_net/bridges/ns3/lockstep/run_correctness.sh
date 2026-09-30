#!/bin/bash
cd /home/zzhang66/experiments/bridge_lockstep; source ~/anaconda3/etc/profile.d/conda.sh && conda activate i5g
P=tests/test_correctness.py
python $P --trafficTime 20 > results/corr1.json 2>&1 &
python $P --trafficTime 20 --transport unix --run 3 > results/corr2.json 2>&1 &
python $P --trafficTime 20 --placement random --shadowStd 6 --extra minSnrDb=0 --run 2 > results/corr3.json 2>&1 &
python $P --trafficTime 20 --frameBytes 30000 --p 0.25 --dists 10,15,20,25,30,35,40,45 --run 4 > results/corr4.json 2>&1 &
python $P --trafficTime 20 --nUe 16 --frameBytes 4000 --p 0.5 --dists 10,20,30,40,50,60,70,80 --extra rlc=AM --run 5 > results/corr5.json 2>&1 &
wait
touch results/CORR_DONE
