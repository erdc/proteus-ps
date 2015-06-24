from proteus import *
from proteus.default_n import *
from mom_p import *
from proteus import Context
ctx = Context.get()

triangleOptions = ctx.triangleOptions


femSpaces = {0:FemTools.C0_AffineQuadraticOnSimplexWithNodalBasis, # u velocity space
             1:FemTools.C0_AffineQuadraticOnSimplexWithNodalBasis, # v velocity space
             2:FemTools.C0_AffineLinearOnSimplexWithNodalBasis} #p pressure space

from TimeIntegrationPS import NonConservativeBackwardEuler, NonConservativeVBDF
# numerics.timeIntegration = TimeIntegration.BackwardEuler
#timeIntegration = NonConservativeBackwardEuler
timeIntegration = NonConservativeVBDF
timeOrder = 2


# stepController  = StepControl.Min_dt_cfl_controller
# runCFL = 0.5
stepController  = FixedStep
DT = ctx.DT


# domain stuff for parallel
parallelPartitioningType = ctx.parallelPartitioningType
nLayersOfOverlapForParallel = ctx.nLayersOfOverlapForParallel
structured=ctx.structured


#Quadrature rules for elements and element  boundaries
elementQuadrature = Quadrature.SimplexGaussQuadrature(ctx.nd,ctx.quad_degree)
elementBoundaryQuadrature = Quadrature.SimplexGaussQuadrature(ctx.nd-1,ctx.quad_degree)


#Matrix type
numericalFluxType = NumericalFlux.StrongDirichletFactory(fluxBoundaryConditions)
#numericalFluxType = MixedDarcy_exterior
# numericalFluxType = NumericalFlux.Advection_DiagonalUpwind_Diffusion_IIPG_exterior
#numericalFluxType = NumericalFlux.Advection_Diagonal_average
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

# linTolFac = 0.001  # relative tolerance for linear solver
# tolFac = 0.0 # absolute tolerance
#
# l_atol_res = 1.0e-5
# nl_atol_res = 1.0e-5

linTolFac = 0.001
l_atol_res = 0.001*ctx.ns_nl_atol_res
tolFac = 0.0
nl_atol_res = ctx.ns_nl_atol_res

periodicDirichletConditions=None

# post processing 

#all of these  should work
#conservativeFlux = {2:'point-eval'}
#conservativeFlux = {2:'pwl-bdm'}
conservativeFlux = {2:'pwl-bdm-opt'}
