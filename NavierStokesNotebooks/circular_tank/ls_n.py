from proteus import *
from proteus.default_n import *
from ls_p import *
from proteus import Context
ctx = Context.get()

if ctx.timeDiscretization=='vbdf':
    timeIntegration = VBDF
    timeOrder=2
    stepController  = TimeIntegration.Min_dt_cfl_controller
elif ctx.timeDiscretization=='flcbdf':
    timeIntegration = TimeIntegration.FLCBDF
    stepController  = TimeIntegration.Min_dt_cfl_controller
    time_tol = 10.0*ctx.ls_nl_atol_res
    atol_u = {0:time_tol}
    rtol_u = {0:time_tol}
else:
    timeIntegration = TimeIntegration.BackwardEuler_cfl
    stepController  = ctx.Min_dt_cfl_controller
runCFL = ctx.runCFL
triangleOptions = ctx.triangleOptions
nLayersOfOverlapForParallel = ctx.nLayersOfOverlapForParallel
parallelPartitioningType = ctx.parallelPartitioningType
nLevels = ctx.nLevels
femSpaces = {0:ctx.basis}
elementQuadrature = ctx.elementQuadrature
elementBoundaryQuadrature = ctx.elementBoundaryQuadrature

massLumping       = False
conservativeFlux  = None
numericalFluxType = NCLS.NumericalFlux
subgridError      = NCLS.SubgridError(coefficients,nd)
shockCapturing    = NCLS.ShockCapturing(coefficients,
                                        ctx.nd,
                                        shockCapturingFactor= \
                                        ctx.ls_shockCapturingFactor,
                                        lag=ctx.ls_lag_shockCapturing)

fullNewtonFlag  = True
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

linear_solver_options_prefix = 'ncls_'
levelNonlinearSolverConvergenceTest = 'r'
linearSolverConvergenceTest         = 'r-true'

tolFac = 0.0
linTolFac = 0.0
l_atol_res = 0.001*ctx.ls_nl_atol_res
nl_atol_res = ctx.ls_nl_atol_res
useEisenstatWalker = True

maxNonlinearIts = 50
maxLineSearches = 0
