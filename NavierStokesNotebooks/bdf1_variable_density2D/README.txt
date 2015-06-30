#
#  Command Line options for running variable density navier stokes flow bdf1 variable density algorithm
#


# sequential computing:
#   make sure that `usePETSc = False` in bdf1_variabledensity.py

parun bdf1_variabledensity_so.py -l5 -v -G



# mpi computing
#   make sure that `usePETSc = True` in bdf1_variabledensity.py

mpiexec -np 1 parun bdf1_variabledensity_so.py -O petsc.options.superlu_dist -l5 -v -G
mpiexec -np 2 parun bdf1_variabledensity_so.py -O petsc.options.superlu_dist -l5 -v -G
