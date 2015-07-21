#
#  Command Line options for running variable density navier stokes flow bdf1 variable density algorithm
#


# Options for running:
#
#  -l5    will give level 5 information to log
#  -v     will give verbose output to stdout which will give the output the full triangle construction statistics
#  -C  "opt1=True  opt2=False"   various context options to be set.  examples
#                                "parallel=True or False" turn on parallel linear algebra or not
#                                "analytical=True or False" give analyticalSolutions to NumericalSolution
#  -b batch_file_name.py    will run the norm analysis as defined in batch_file_name.py  here it is by default called "L2_batch.py"
#                           The option "analytical=True" needs to be set in flag -C to give tru functions to compare against.
#
#  -D folder_name     output log and .xml and .h5 results to this folder


# sequential computing:

#
# BDF1:
#   Set in navierstokes_vardensity.py  (for example)
#   globalBDFTimeOrder = 1
#   DT = 0.1

parun navierstokes_vardensity_so.py -l5 -v -b L2_batch.py -C "parallel=False analytical=True" -D navierstokes_bdf1_1_dt_0_100000_results > bdf1_1_dt_0_100000.out


#
# BDF2:
#   Set in navierstokes_vardensity.py
#   globalBDFTimeOrder = 2
#   DT = 0.1

parun navierstokes_vardensity_so.py -l5 -v -b L2_batch.py -C "parallel=False analytical=True" -D navierstokes_bdf2_1_dt_0_100000_results > bdf2_1_dt_0_100000.out



# After you have run the various time steps and want to make the analysis of them, you can run
# the following sequences of commands which correspond to the different dt values run

#  running the processing from VariableDensityNavierStokes2D/  folder when results
# are stored in results/  folder.  This will produce the error reports and tables.

python processVelocity.py results/velocity_BDF1_dt_0_100000_0.db \
                          results/velocity_BDF1_dt_0_050000_0.db \
                          results/velocity_BDF1_dt_0_025000_0.db \
                          results/velocity_BDF1_dt_0_012500_0.db \
                          results/velocity_BDF1_dt_0_006250_0.db

python processPressure.py results/pressure_BDF1_dt_0_100000_0.db \
                          results/pressure_BDF1_dt_0_050000_0.db \
                          results/pressure_BDF1_dt_0_025000_0.db \
                          results/pressure_BDF1_dt_0_012500_0.db \
                          results/pressure__BDF1DT_0_006250_0.db

python processDensity.py results/density_BDF1_dt_0_100000_0.db \
                        results/density_BDF1_dt_0_050000_0.db \
                        results/density_BDF1_dt_0_025000_0.db \
                        results/density_BDF1_dt_0_012500_0.db \
                        results/density_BDF1_dt_0_006250_0.db

mv -f *.png results/

python processVelocity.py results/velocity_BDF2_dt_0_100000_0.db \
                          results/velocity_BDF2_dt_0_050000_0.db \
                          results/velocity_BDF2_dt_0_025000_0.db \
                          results/velocity_BDF2_dt_0_012500_0.db \
                          results/velocity_BDF2_dt_0_006250_0.db

python processPressure.py results/pressure_BDF2_dt_0_100000_0.db \
                          results/pressure_BDF2_dt_0_050000_0.db \
                          results/pressure_BDF2_dt_0_025000_0.db \
                          results/pressure_BDF2_dt_0_012500_0.db \
                          results/pressure_BDF2_dt_0_006250_0.db

python processDensity.py results/density_BDF2_dt_0_100000_0.db \
                        results/density_BDF2_dt_0_050000_0.db \
                        results/density_BDF2_dt_0_025000_0.db \
                        results/density_BDF2_dt_0_012500_0.db \
                        results/density_BDF2_dt_0_006250_0.db

mv -f *.png results/



# Parallel computing with MPI

#
# BDF1:
#   Set in navierstokes_vardensity.py
#   globalBDFTimeOrder = 1
#   DT = 0.1

mpiexec -np 4 parun navierstokes_vardensity_so.py -O petsc.options.superlu_dist -l5 -v -b L2_batch.py -C "parallel=True analytical=True" -D navierstokes_bdf1_1_dt_0_100000_results > bdf1_1_dt_0_100000.out

#
# BDF2:
#   Set in navierstokes_vardensity.py
#   globalBDFTimeOrder = 2
#   DT = 0.1

mpiexec -np 4 parun navierstokes_vardensity_so.py -O petsc.options.superlu_dist -l5 -v -b L2_batch.py -C "parallel=True analytical=True" -D navierstokes_bdf2_1_dt_0_100000_results > bdf2_1_dt_0_100000.out
