from proteus import *
from proteus.default_n import *
from rho_p import *
from proteus import Context
ctx = Context.get()

triangleOptions = ctx.triangleOptions

femSpaces = {0:FemTools.C0_AffineQuadraticOnSimplexWithNodalBasis}#density space

# numerics.timeIntegration = TimeIntegration.BackwardEuler
# timeIntegration = TimeIntegration.BackwardEuler_cfl
timeIntegration = TimeIntegration.VBDF
timeOrder = 2

# stepController  = StepControl.Min_dt_cfl_controller
# runCFL= 0.99
# runCFL= 0.5

stepController=FixedStep
DT = ctx.DT


# domain stuff for parallel
parallelPartitioningType = ctx.parallelPartitioningType
nLayersOfOverlapForParallel = ctx.nLayersOfOverlapForParallel
structured=ctx.structured

#Quadrature rules for elements and element  boundaries
elementQuadrature = Quadrature.SimplexGaussQuadrature(ctx.nd,ctx.quad_degree)
elementBoundaryQuadrature = Quadrature.SimplexGaussQuadrature(ctx.nd-1,ctx.quad_degree)
#number of nodes in the x and y direction

subgridError = SubgridError.Advection_ASGS(coefficients,
                                           ctx.nd,
                                           lag=False)
                                           
#numerics.shockCapturing = ShockCapturing.ResGradQuadDelayLag_SC(physics.coefficients,
#                                                                physics.nd,
#                                                                lag = True,
#                                                                nStepsToDelay=1)
#numerics.nny= 41

#matrix type
#numericalFluxType = NumericalFlux.StrongDirichletFactory(fluxBoundaryConditions) #strong boundary conditions
numericalFluxType = NumericalFlux.Advection_DiagonalUpwind_Diffusion_IIPG_exterior #weak boundary conditions (upwind)
matrix = LinearAlgebraTools.SparseMatrix

if ctx.usePetsc:
    #use petsc solvers wrapped by petsc4py
    multilevelLinearSolver = LinearSolvers.KSP_petsc4py
    levelLinearSolver = LinearSolvers.KSP_petsc4py
    #using petsc4py requires weak boundary condition enforcement
else:
    #can also use our internal wrapper for SuperLU
    multilevelLinearSolver = LinearSolvers.LU
    levelLinearSolver = LinearSolvers.LU

linear_solver_options_prefix = 'rans2p_'
# linearSolverConvergenceTest  = 'r-true'
# levelNonlinearSolverConvergenceTest = 'r'

multilevelNonlinearSolver = NonlinearSolvers.Newton
levelNonlinearSolver = NonlinearSolvers.Newton

#linear solve rtolerance

linTolFac = 0.001
l_atol_res = 0.001*ctx.ns_nl_atol_res
tolFac = 0.0
nl_atol_res = ctx.ns_nl_atol_res

periodicDirichletConditions=None
