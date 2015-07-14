#!/bin/bash

# changes the DT value in the helper file and then runs the proper parun commands with specialized output
# to distinguish between the running locations


sed 's/DT = 0\.[0-9]*/DT = 0.1/g' navierstokes_vardensity.py > tmp.py && mv -f tmp.py navierstokes_vardensity.py 
parun navierstokes_vardensity_so.py -l5 -G -b L2_batch.py -D navierstokes_dt_0_100000_results > dt_0_100000.out

sed 's/DT = 0\.[0-9]*/DT = 0.05/g' navierstokes_vardensity.py > tmp.py && mv -f tmp.py navierstokes_vardensity.py 
parun navierstokes_vardensity_so.py -l5 -G -b L2_batch.py -D navierstokes_dt_0_050000_results > dt_0_050000.out

sed 's/DT = 0\.[0-9]*/DT = 0.025/g' navierstokes_vardensity.py > tmp.py && mv -f tmp.py navierstokes_vardensity.py 
parun navierstokes_vardensity_so.py -l5 -G -b L2_batch.py -D navierstokes_dt_0_025000_results > dt_0_025000.out

sed 's/DT = 0\.[0-9]*/DT = 0.0125/g' navierstokes_vardensity.py > tmp.py && mv -f tmp.py navierstokes_vardensity.py 
parun navierstokes_vardensity_so.py -l5 -G -b L2_batch.py -D navierstokes_dt_0_012500_results > dt_0_012500.out

sed 's/DT = 0\.[0-9]*/DT = 0.00625/g' navierstokes_vardensity.py > tmp.py && mv -f tmp.py navierstokes_vardensity.py 
parun navierstokes_vardensity_so.py -l5 -G -b L2_batch.py -D navierstokes_dt_0_006250_results > dt_0_006250.out


