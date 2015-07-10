#
#  Command Line options for running variable density navier stokes flow bdf1 variable density algorithm
#


# sequential computing:
#   make sure that `usePETSc = False` in navierstokes_vardensity.py

parun navierstokes_vardensity_so.py -l5 -v -G -b L2_batch.py -D bdf1_dt_0_10000_results

# or if running the bdf2 version switched in navierstokes_vardensity.py then output to 

parun navierstokes_vardensity_so.py -l5 -v -G -b L2_batch.py -D bdf2_dt_0_10000_results




# mpi computing
#   make sure that `usePETSc = True` in navierstokes_vardensity.py

mpiexec -np 1 parun navierstokes_vardensity_so.py -O petsc.options.superlu_dist -l5 -v -G -b L2_batch.py -D bdf1_dt_0_10000_results
mpiexec -np 2 parun navierstokes_vardensity_so.py -O petsc.options.superlu_dist -l5 -v -G -b L2_batch.py
