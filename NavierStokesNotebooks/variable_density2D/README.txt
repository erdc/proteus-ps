#
#  Command Line options for running variable density navier stokes flow fully coupled
#


# sequential computing:
#   make sure that `usePETSc = False` in guermond_example_variable_density.py

parun guermond_example_variable_density_so.py -l5 -v -G



# mpi computing
#   make sure that `usePETSc = True` in guermond_example_variable_density.py

mpiexec -np 1 parun guermond_example_variable_density_so.py -O petsc.options.superlu_dist -l5 -v -G
mpiexec -np 2 parun guermond_example_variable_density_so.py -O petsc.options.superlu_dist -l5 -v -G
