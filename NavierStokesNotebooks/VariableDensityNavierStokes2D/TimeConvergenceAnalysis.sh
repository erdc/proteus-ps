#!/bin/bash

# changes the DT value in the helper file and then runs the proper parun commands with specialized output
# to distinguish between the running locations


sed 's/DT = 0\.[0-9]*/DT = 0.1/g' bdf1_variabledensity.py > tmp.py && mv -f tmp.py bdf1_variabledensity.py 
parun bdf1_variabledensity_so.py -l5 -G -b L2_batch.py -D bdf1_dt_0_100000_results > dt_0_100000.out

sed 's/DT = 0\.[0-9]*/DT = 0.05/g' bdf1_variabledensity.py > tmp.py && mv -f tmp.py bdf1_variabledensity.py 
parun bdf1_variabledensity_so.py -l5 -G -b L2_batch.py -D bdf1_dt_0_050000_results > dt_0_050000.out

sed 's/DT = 0\.[0-9]*/DT = 0.025/g' bdf1_variabledensity.py > tmp.py && mv -f tmp.py bdf1_variabledensity.py 
parun bdf1_variabledensity_so.py -l5 -G -b L2_batch.py -D bdf1_dt_0_025000_results > dt_0_025000.out

sed 's/DT = 0\.[0-9]*/DT = 0.0125/g' bdf1_variabledensity.py > tmp.py && mv -f tmp.py bdf1_variabledensity.py 
parun bdf1_variabledensity_so.py -l5 -G -b L2_batch.py -D bdf1_dt_0_012500_results > dt_0_012500.out

sed 's/DT = 0\.[0-9]*/DT = 0.006125/g' bdf1_variabledensity.py > tmp.py && mv -f tmp.py bdf1_variabledensity.py 
parun bdf1_variabledensity_so.py -l5 -G -b L2_batch.py -D bdf1_dt_0_006250_results > dt_0_006250.out


