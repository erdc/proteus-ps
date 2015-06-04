from proteus import *
from proteus.default_n import *
from twp_navier_stokes_p import *

if ctx.timeDiscretization=='vbdf':
    timeIntegration = VBDF
    timeOrder=2
    stepController  = StepControl.Min_dt_cfl_controller
elif ctx.timeDiscretization=='flcbdf':
    timeIntegration = TimeIntegration.FLCBDF
    #stepController = FLCBDF_controller_sys
    stepController  = StepControl.Min_dt_cfl_controller
    time_tol = 10.0*ctx.ns_nl_atol_res
    atol_u = {1:time_tol,2:time_tol}
    rtol_u = {1:time_tol,2:time_tol}
else:
    timeIntegration = TimeIntegration.BackwardEuler_cfl
    stepController  = StepControl.Min_dt_cfl_controller

runCFL = ctx.runCFL
triangleOptions = ctx.triangleOptions
nLayersOfOverlapForParallel = ctx.nLayersOfOverlapForParallel
parallelPartitioningType = ctx.parallelPartitioningType
nLevels = ctx.nLevels
femSpaces = {0:ctx.basis,
	     1:ctx.basis,
	     2:ctx.basis}

elementQuadrature = ctx.elementQuadrature
elementBoundaryQuadrature = ctx.elementBoundaryQuadrature

massLumping       = False
numericalFluxType = None
conservativeFlux  = None

numericalFluxType = RANS2P.NumericalFlux
subgridError = RANS2P.SubgridError(coefficients,
                                   ctx.nd,
                                   lag=ctx.ns_lag_subgridError,
                                   hFactor=ctx.hFactor)
shockCapturing = RANS2P.ShockCapturing(coefficients,
                                       ctx.nd,
                                       ctx.ns_shockCapturingFactor,
                                       lag=ctx.ns_lag_shockCapturing)

fullNewtonFlag = True
multilevelNonlinearSolver = NonlinearSolvers.Newton
levelNonlinearSolver      = NonlinearSolvers.Newton

nonlinearSmoother = None

linearSmoother    = LinearSolvers.SimpleNavierStokes2D

matrix = LinearSolvers.SparseMatrix

if ctx.useOldPETSc:
    multilevelLinearSolver = LinearSolvers.PETSc
    levelLinearSolver      = Linearsolvers.PETSc
else:
    multilevelLinearSolver = LinearSolvers.KSP_petsc4py
    levelLinearSolver      = LinearSolvers.KSP_petsc4py

if ctx.useSuperlu:
    multilevelLinearSolver = LinearSolvers.LU
    levelLinearSolver      = LinearSolvers.LU

linear_solver_options_prefix = 'rans2p_'
levelNonlinearSolverConvergenceTest = 'r'
linearSolverConvergenceTest             = 'r-true'

linTolFac = 0.01
l_atol_res = 0.001*ctx.ns_nl_atol_res
tolFac = 0.0
nl_atol_res = ctx.ns_nl_atol_res
useEisenstatWalker = True
maxNonlinearIts = 50
maxSolverFailures = 10
maxErrorFailures = 10
maxLineSearches = 0
computeLinearSolverRates = False
conservativeFlux = {0:'pwl-bdm-opt'}
auxiliaryVariables=[]
reactionLumping = False
printLinearSolverInfo = False
