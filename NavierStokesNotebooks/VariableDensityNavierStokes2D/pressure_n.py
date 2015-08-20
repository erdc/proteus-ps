from proteus import *
from proteus.default_n import *
from pressure_p import *


triangleOptions = ctx.triangleOptions


femSpaces = {0:FemTools.C0_AffineLinearOnSimplexWithNodalBasis} #pressure P1 space


# stepController  = StepControl.Min_dt_cfl_controller
# runCFL= 0.99
# runCFL= 0.5

stepController=FixedStep
DT = ctx.DT

#Quadrature rules for elements and element  boundaries
elementQuadrature = Quadrature.SimplexGaussQuadrature(ctx.nd,ctx.quad_degree)
elementBoundaryQuadrature = Quadrature.SimplexGaussQuadrature(ctx.nd-1,ctx.quad_degree)
#number of nodes in the x and y direction


#matrix type
#numericalFluxType = NumericalFlux.StrongDirichletFactory(fluxBoundaryConditions) #strong boundary conditions
numericalFluxType = NumericalFlux.ConstantAdvection_exterior
matrix = LinearAlgebraTools.SparseMatrix
#use petsc solvers wrapped by petsc4py
#numerics.multilevelLinearSolver = LinearSolvers.KSP_petsc4py
#numerics.levelLinearSolver = LinearSolvers.KSP_petsc4py
#using petsc4py requires weak boundary condition enforcement
#can also use our internal wrapper for SuperLU
multilevelLinearSolver = LinearSolvers.LU
levelLinearSolver = LinearSolvers.LU

linear_solver_options_prefix = 'pressure_'

if ctx.opts.parallel:
    multilevelLinearSolver = KSP_petsc4py
    levelLinearSolver      = KSP_petsc4py
    parallelPartitioningType = ctx.parallelPartitioningType
    nLayersOfOverlapForParallel = ctx.nLayersOfOverlapForParallel
    nonlinearSmoother = None
    linearSmoother    = None

multilevelNonlinearSolver = NonlinearSolvers.Newton
levelNonlinearSolver = NonlinearSolvers.Newton

#linear solve rtolerance

linTolFac = 0.0
l_atol_res = 0.1*ctx.pressure_atol_res
tolFac = 0.0
nl_atol_res = ctx.pressure_atol_res

nonlinearSolverConvergenceTest      = 'r'
levelNonlinearSolverConvergenceTest = 'r'
linearSolverConvergenceTest         = 'r-true'

periodicDirichletConditions=None

conservativeFlux=None
