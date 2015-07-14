#
#  Command Line options for running variable density navier stokes flow bdf1 variable density algorithm
#


# sequential computing:
#   make sure that `usePETSc = False` in navierstokes_vardensity.py

parun navierstokes_vardensity_so.py -l5 -v -G -b L2_batch.py -D bdf1_dt_0_10000_results

# or if running the bdf2 version switched in navierstokes_vardensity.py then output to

parun navierstokes_vardensity_so.py -l5 -v -G -b L2_batch.py -D bdf2_dt_0_10000_results



#  running the processing from VariableDensityNavierStokes2D/  folder when results
# are stored in results/  folder.  This will produce the error reports and tables.

python processVelocity.py results/velocity_DT_0_100000_BDF1.db \
                          results/velocity_DT_0_050000_BDF1.db \
                          results/velocity_DT_0_025000_BDF1.db \
                          results/velocity_DT_0_012500_BDF1.db \
                          results/velocity_DT_0_006125_BDF1.db

python processPressure.py results/pressure_DT_0_100000_BDF1.db \
                          results/pressure_DT_0_050000_BDF1.db \
                          results/pressure_DT_0_025000_BDF1.db \
                          results/pressure_DT_0_012500_BDF1.db \
                          results/pressure_DT_0_006125_BDF1.db

python processDensity.py results/density_DT_0_100000_BDF1.db \
                        results/density_DT_0_050000_BDF1.db \
                        results/density_DT_0_025000_BDF1.db \
                        results/density_DT_0_012500_BDF1.db \
                        results/density_DT_0_006125_BDF1.db



# mpi computing
#   make sure that `usePETSc = True` in navierstokes_vardensity.py

mpiexec -np 1 parun navierstokes_vardensity_so.py -O petsc.options.superlu_dist -l5 -v -G -b L2_batch.py -D bdf1_dt_0_10000_results
mpiexec -np 2 parun navierstokes_vardensity_so.py -O petsc.options.superlu_dist -l5 -v -G -b L2_batch.py
