#!/bin/bash

# changes the DT value in the helper file and then runs the proper parun commands with specialized output
# to distinguish between the running locations

# bdf1 algorithm runs
sed 's/globalBDFTimeOrder = [1-2]/globalBDFTimeOrder = 1/g' navierstokes_vardensity.py > tmp.py && mv -f tmp.py navierstokes_vardensity.py

sed 's/DT = 0\.[0-9]*/DT = 0.1/g' navierstokes_vardensity.py > tmp.py && mv -f tmp.py navierstokes_vardensity.py
parun navierstokes_vardensity_so.py -l5 -b L2_batch.py -C "parallel=False analytical=True" -D navierstokes_bdf1_1_dt_0_100000_results > bdf1_1_dt_0_100000.out&

sed 's/DT = 0\.[0-9]*/DT = 0.05/g' navierstokes_vardensity.py > tmp.py && mv -f tmp.py navierstokes_vardensity.py
parun navierstokes_vardensity_so.py -l5 -b L2_batch.py -C "parallel=False analytical=True" -D navierstokes_bdf1_2_dt_0_050000_results > bdf1_2_dt_0_050000.out&

sed 's/DT = 0\.[0-9]*/DT = 0.025/g' navierstokes_vardensity.py > tmp.py && mv -f tmp.py navierstokes_vardensity.py
parun navierstokes_vardensity_so.py -l5 -b L2_batch.py -C "parallel=False analytical=True" -D navierstokes_bdf1_3_dt_0_025000_results > bdf1_3_dt_0_025000.out&

sed 's/DT = 0\.[0-9]*/DT = 0.0125/g' navierstokes_vardensity.py > tmp.py && mv -f tmp.py navierstokes_vardensity.py
parun navierstokes_vardensity_so.py -l5 -b L2_batch.py -C "parallel=False analytical=True" -D navierstokes_bdf1_4_dt_0_012500_results > bdf1_4_dt_0_012500.out&

sed 's/DT = 0\.[0-9]*/DT = 0.00625/g' navierstokes_vardensity.py > tmp.py && mv -f tmp.py navierstokes_vardensity.py
parun navierstokes_vardensity_so.py -l5 -b L2_batch.py -C "parallel=False analytical=True" -D navierstokes_bdf1_5_dt_0_006250_results > bdf1_5_dt_0_006250.out&



# bdf2 algorithm runs single processor
sed 's/globalBDFTimeOrder = [1-2]/globalBDFTimeOrder = 2/g' navierstokes_vardensity.py > tmp.py && mv -f tmp.py navierstokes_vardensity.py

sed 's/DT = 0\.[0-9]*/DT = 0.1/g' navierstokes_vardensity.py > tmp.py && mv -f tmp.py navierstokes_vardensity.py
parun navierstokes_vardensity_so.py -l5 -b L2_batch.py -O petsc.options.superlu_dist -C "parallel=False analytical=True" -D navierstokes_bdf2_1_dt_0_100000_results > bdf2_1_dt_0_100000.out&

sed 's/DT = 0\.[0-9]*/DT = 0.05/g' navierstokes_vardensity.py > tmp.py && mv -f tmp.py navierstokes_vardensity.py
parun navierstokes_vardensity_so.py -l5 -b L2_batch.py -O petsc.options.superlu_dist -C "parallel=False analytical=True" -D navierstokes_bdf2_2_dt_0_050000_results > bdf2_2_dt_0_050000.out&

sed 's/DT = 0\.[0-9]*/DT = 0.025/g' navierstokes_vardensity.py > tmp.py && mv -f tmp.py navierstokes_vardensity.py
parun navierstokes_vardensity_so.py -l5 -b L2_batch.py -O petsc.options.superlu_dist -C "parallel=False analytical=True" -D navierstokes_bdf2_3_dt_0_025000_results > bdf2_3_dt_0_025000.out&

sed 's/DT = 0\.[0-9]*/DT = 0.0125/g' navierstokes_vardensity.py > tmp.py && mv -f tmp.py navierstokes_vardensity.py
parun navierstokes_vardensity_so.py -l5 -b L2_batch.py -O petsc.options.superlu_dist -C "parallel=False analytical=True" -D navierstokes_bdf2_4_dt_0_012500_results > bdf2_4_dt_0_012500.out&

sed 's/DT = 0\.[0-9]*/DT = 0.00625/g' navierstokes_vardensity.py > tmp.py && mv -f tmp.py navierstokes_vardensity.py
parun navierstokes_vardensity_so.py -l5 -b L2_batch.py -O petsc.options.superlu_dist -C "parallel=False analytical=True" -D navierstokes_bdf2_5_dt_0_006250_results > bdf2_5_dt_0_006250.out&





############## 2 mpi processes ##########################

# bdf1 algorithm run 2 mpi

sed 's/globalBDFTimeOrder = [1-2]/globalBDFTimeOrder = 1/g' navierstokes_vardensity.py > tmp.py && mv -f tmp.py navierstokes_vardensity.py

sed 's/DT = 0\.[0-9]*/DT = 0.1/g' navierstokes_vardensity.py > tmp.py && mv -f tmp.py navierstokes_vardensity.py
mpiexec -np 2 parun navierstokes_vardensity_so.py -O petsc.options.superlu_dist -l5 -b L2_batch.py -C "parallel=True analytical=True" -D navierstokes_bdf1_1_dt_0_100000_results > bdf1_1_dt_0_100000.out&

sed 's/DT = 0\.[0-9]*/DT = 0.05/g' navierstokes_vardensity.py > tmp.py && mv -f tmp.py navierstokes_vardensity.py
mpiexec -np 2 parun navierstokes_vardensity_so.py -O petsc.options.superlu_dist -l5 -b L2_batch.py -C "parallel=True analytical=True" -D navierstokes_bdf1_2_dt_0_050000_results > bdf1_2_dt_0_050000.out&

sed 's/DT = 0\.[0-9]*/DT = 0.025/g' navierstokes_vardensity.py > tmp.py && mv -f tmp.py navierstokes_vardensity.py
mpiexec -np 2 parun navierstokes_vardensity_so.py -O petsc.options.superlu_dist -l5 -b L2_batch.py -C "parallel=True analytical=True" -D navierstokes_bdf1_3_dt_0_025000_results > bdf1_3_dt_0_025000.out&

sed 's/DT = 0\.[0-9]*/DT = 0.0125/g' navierstokes_vardensity.py > tmp.py && mv -f tmp.py navierstokes_vardensity.py
mpiexec -np 2 parun navierstokes_vardensity_so.py -O petsc.options.superlu_dist -l5 -b L2_batch.py -C "parallel=True analytical=True" -D navierstokes_bdf1_4_dt_0_012500_results > bdf1_4_dt_0_012500.out&

sed 's/DT = 0\.[0-9]*/DT = 0.00625/g' navierstokes_vardensity.py > tmp.py && mv -f tmp.py navierstokes_vardensity.py
mpiexec -np 2 parun navierstokes_vardensity_so.py -O petsc.options.superlu_dist -l5 -b L2_batch.py -C "parallel=True analytical=True" -D navierstokes_bdf1_5_dt_0_006250_results > bdf1_5_dt_0_006250.out&



# bdf2 algorithm runs 2 mpi
sed 's/globalBDFTimeOrder = [1-2]/globalBDFTimeOrder = 2/g' navierstokes_vardensity.py > tmp.py && mv -f tmp.py navierstokes_vardensity.py

sed 's/DT = 0\.[0-9]*/DT = 0.1/g' navierstokes_vardensity.py > tmp.py && mv -f tmp.py navierstokes_vardensity.py
mpiexec -np 2 parun navierstokes_vardensity_so.py -O petsc.options.superlu_dist -l5 -b L2_batch.py -C "parallel=True analytical=True" -D navierstokes_bdf2_1_dt_0_100000_results > bdf2_1_dt_0_100000.out&

sed 's/DT = 0\.[0-9]*/DT = 0.05/g' navierstokes_vardensity.py > tmp.py && mv -f tmp.py navierstokes_vardensity.py
mpiexec -np 2 parun navierstokes_vardensity_so.py -O petsc.options.superlu_dist -l5 -b L2_batch.py -C "parallel=True analytical=True" -D navierstokes_bdf2_2_dt_0_050000_results > bdf2_2_dt_0_050000.out&

sed 's/DT = 0\.[0-9]*/DT = 0.025/g' navierstokes_vardensity.py > tmp.py && mv -f tmp.py navierstokes_vardensity.py
mpiexec -np 2 parun navierstokes_vardensity_so.py -O petsc.options.superlu_dist -l5 -b L2_batch.py -C "parallel=True analytical=True" -D navierstokes_bdf2_3_dt_0_025000_results > bdf2_3_dt_0_025000.out&

sed 's/DT = 0\.[0-9]*/DT = 0.0125/g' navierstokes_vardensity.py > tmp.py && mv -f tmp.py navierstokes_vardensity.py
mpiexec -np 2 parun navierstokes_vardensity_so.py -O petsc.options.superlu_dist -l5 -b L2_batch.py -C "parallel=True analytical=True" -D navierstokes_bdf2_4_dt_0_012500_results > bdf2_4_dt_0_012500.out&

sed 's/DT = 0\.[0-9]*/DT = 0.00625/g' navierstokes_vardensity.py > tmp.py && mv -f tmp.py navierstokes_vardensity.py
mpiexec -np 2 parun navierstokes_vardensity_so.py -O petsc.options.superlu_dist -l5 -b L2_batch.py -C "parallel=True analytical=True" -D navierstokes_bdf2_5_dt_0_006250_results > bdf2_5_dt_0_006250.out&



##################### 4 mpi processes #########################

# bdf1 algorithm run 4 mpi

sed 's/globalBDFTimeOrder = [1-2]/globalBDFTimeOrder = 1/g' navierstokes_vardensity.py > tmp.py && mv -f tmp.py navierstokes_vardensity.py

sed 's/DT = 0\.[0-9]*/DT = 0.1/g' navierstokes_vardensity.py > tmp.py && mv -f tmp.py navierstokes_vardensity.py
mpiexec -np 4 parun navierstokes_vardensity_so.py -O petsc.options.superlu_dist -l5 -b L2_batch.py -C "parallel=True analytical=True" -D navierstokes_bdf1_1_dt_0_100000_results > bdf1_1_dt_0_100000.out&

sed 's/DT = 0\.[0-9]*/DT = 0.05/g' navierstokes_vardensity.py > tmp.py && mv -f tmp.py navierstokes_vardensity.py
mpiexec -np 4 parun navierstokes_vardensity_so.py -O petsc.options.superlu_dist -l5 -b L2_batch.py -C "parallel=True analytical=True" -D navierstokes_bdf1_2_dt_0_050000_results > bdf1_2_dt_0_050000.out&

sed 's/DT = 0\.[0-9]*/DT = 0.025/g' navierstokes_vardensity.py > tmp.py && mv -f tmp.py navierstokes_vardensity.py
mpiexec -np 4 parun navierstokes_vardensity_so.py -O petsc.options.superlu_dist -l5 -b L2_batch.py -C "parallel=True analytical=True" -D navierstokes_bdf1_3_dt_0_025000_results > bdf1_3_dt_0_025000.out&

sed 's/DT = 0\.[0-9]*/DT = 0.0125/g' navierstokes_vardensity.py > tmp.py && mv -f tmp.py navierstokes_vardensity.py
mpiexec -np 4 parun navierstokes_vardensity_so.py -O petsc.options.superlu_dist -l5 -b L2_batch.py -C "parallel=True analytical=True" -D navierstokes_bdf1_4_dt_0_012500_results > bdf1_4_dt_0_012500.out&

sed 's/DT = 0\.[0-9]*/DT = 0.00625/g' navierstokes_vardensity.py > tmp.py && mv -f tmp.py navierstokes_vardensity.py
mpiexec -np 4 parun navierstokes_vardensity_so.py -O petsc.options.superlu_dist -l5 -b L2_batch.py -C "parallel=True analytical=True" -D navierstokes_bdf1_5_dt_0_006250_results > bdf1_5_dt_0_006250.out&



# bdf2 algorithm runs 4 mpi
sed 's/globalBDFTimeOrder = [1-2]/globalBDFTimeOrder = 2/g' navierstokes_vardensity.py > tmp.py && mv -f tmp.py navierstokes_vardensity.py

sed 's/DT = 0\.[0-9]*/DT = 0.1/g' navierstokes_vardensity.py > tmp.py && mv -f tmp.py navierstokes_vardensity.py
mpiexec -np 4 parun navierstokes_vardensity_so.py -O petsc.options.superlu_dist -l5 -b L2_batch.py -C "parallel=True analytical=True" -D navierstokes_bdf2_1_dt_0_100000_results > bdf2_1_dt_0_100000.out&

sed 's/DT = 0\.[0-9]*/DT = 0.05/g' navierstokes_vardensity.py > tmp.py && mv -f tmp.py navierstokes_vardensity.py
mpiexec -np 4 parun navierstokes_vardensity_so.py -O petsc.options.superlu_dist -l5 -b L2_batch.py -C "parallel=True analytical=True" -D navierstokes_bdf2_2_dt_0_050000_results > bdf2_2_dt_0_050000.out&

sed 's/DT = 0\.[0-9]*/DT = 0.025/g' navierstokes_vardensity.py > tmp.py && mv -f tmp.py navierstokes_vardensity.py
mpiexec -np 4 parun navierstokes_vardensity_so.py -O petsc.options.superlu_dist -l5 -b L2_batch.py -C "parallel=True analytical=True" -D navierstokes_bdf2_3_dt_0_025000_results > bdf2_3_dt_0_025000.out&

sed 's/DT = 0\.[0-9]*/DT = 0.0125/g' navierstokes_vardensity.py > tmp.py && mv -f tmp.py navierstokes_vardensity.py
mpiexec -np 4 parun navierstokes_vardensity_so.py -O petsc.options.superlu_dist -l5 -b L2_batch.py -C "parallel=True analytical=True" -D navierstokes_bdf2_4_dt_0_012500_results > bdf2_4_dt_0_012500.out&

sed 's/DT = 0\.[0-9]*/DT = 0.00625/g' navierstokes_vardensity.py > tmp.py && mv -f tmp.py navierstokes_vardensity.py
mpiexec -np 4 parun navierstokes_vardensity_so.py -O petsc.options.superlu_dist -l5 -b L2_batch.py -C "parallel=True analytical=True" -D navierstokes_bdf2_5_dt_0_006250_results > bdf2_5_dt_0_006250.out&
