from proteus import *
from proteus.default_n import *
from vof_p import *

if ctx.timeDiscretization=='vbdf':
    timeIntegration = TimeIntegration.VBDF
    timeOrder=2
    stepController  = StepControl.Min_dt_cfl_controller
elif ctx.timeDiscretization=='flcbdf':
    timeIntegration = TimeIntegration.FLCBDF
    #stepController = FLCBDF_controller
    stepController  = StepControl.Min_dt_cfl_controller
    time_tol = 10.0*ctx.vof_nl_atol_res
    atol_u = {0:time_tol}
    rtol_u = {0:time_tol}
else:
    timeIntegration = TimeIntegration.BackwardEuler_cfl
    stepController  = StepControl.Min_dt_cfl_controller
runCFL = ctx.runCFL
triangleOptions = ctx.triangleOptions
nLayersOfOverlapForParallel = ctx.nLayersOfOverlapForParallel
parallelPartitioningType = ctx.parallelPartitioningType
nLevels = ctx.nLevels
femSpaces = {0:ctx.basis}
elementQuadrature = ctx.elementQuadrature
elementBoundaryQuadrature = ctx.elementBoundaryQuadrature

massLumping       = False
numericalFluxType = VOF.NumericalFlux
conservativeFlux  = None
subgridError      = VOF.SubgridError(coefficients=coefficients,nd=nd)
shockCapturing    = VOF.ShockCapturing(coefficients,
                                       ctx.nd,
                                       shockCapturingFactor= \
                                       ctx.vof_shockCapturingFactor,
                                       lag=ctx.vof_lag_shockCapturing)

fullNewtonFlag = True
multilevelNonlinearSolver = Newton
levelNonlinearSolver      = Newton

nonlinearSmoother = None
linearSmoother    = None

matrix = SparseMatrix

if ctx.useOldPETSc:
    multilevelLinearSolver = PETSc
    levelLinearSolver      = PETSc
else:
    multilevelLinearSolver = KSP_petsc4py
    levelLinearSolver      = KSP_petsc4py

if ctx.useSuperlu:
    multilevelLinearSolver = LU
    levelLinearSolver      = LU

linear_solver_options_prefix = 'vof_'
levelNonlinearSolverConvergenceTest = 'r'
linearSolverConvergenceTest         = 'r-true'

tolFac      = 0.0
linTolFac   = 0.0
l_atol_res = 0.001*ctx.vof_nl_atol_res
nl_atol_res = ctx.vof_nl_atol_res
useEisenstatWalker = True

maxNonlinearIts = 50
maxLineSearches = 0
