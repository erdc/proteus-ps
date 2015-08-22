from proteus import *
from proteus.default_n import *
from pressureincrement_p import *


triangleOptions = ctx.triangleOptions


femSpaces = {0:FemTools.C0_AffineLinearOnSimplexWithNodalBasis} # = pressure space = P1 with zero average

stepController=FixedStep
DT = ctx.DT

#Quadrature rules for elements and element  boundaries
elementQuadrature = Quadrature.SimplexGaussQuadrature(ctx.nd,ctx.quad_degree)
elementBoundaryQuadrature = Quadrature.SimplexGaussQuadrature(ctx.nd-1,ctx.quad_degree)

#numericalFluxType = NumericalFlux.StrongDirichletFactory(fluxBoundaryConditions) #strong boundary conditions
numericalFluxType = NumericalFlux.ConstantAdvection_Diffusion_SIPG_exterior #weak boundary conditions (upwind)
matrix = LinearAlgebraTools.SparseMatrix
#use petsc solvers wrapped by petsc4py
#numerics.multilevelLinearSolver = LinearSolvers.KSP_petsc4py
#numerics.levelLinearSolver = LinearSolvers.KSP_petsc4py
#using petsc4py requires weak boundary condition enforcement
#can also use our internal wrapper for SuperLU

if ctx.useNoFluxPressureIncrementBC:
    linearSmoother    = LinearSolvers.NavierStokesPressureCorrection
    multilevelLinearSolver = LinearSolvers.KSP_petsc4py
    levelLinearSolver = LinearSolvers.KSP_petsc4py
else:
    multilevelLinearSolver = LinearSolvers.LU
    levelLinearSolver = LinearSolvers.LU

linear_solver_options_prefix = 'phi_'

if ctx.opts.parallel:
    multilevelLinearSolver = LinearSolvers.KSP_petsc4py
    levelLinearSolver      = LinearSolvers.KSP_petsc4py
    parallelPartitioningType = ctx.parallelPartitioningType
    nLayersOfOverlapForParallel = ctx.nLayersOfOverlapForParallel

multilevelNonlinearSolver = NonlinearSolvers.Newton
levelNonlinearSolver = NonlinearSolvers.Newton

#linear solve rtolerance

linTolFac = 0.0
l_atol_res = 0.1*ctx.phi_atol_res
tolFac = 0.0
nl_atol_res = ctx.phi_atol_res
nonlinearSolverConvergenceTest = 'r'
levelNonlinearSolverConvergenceTest = 'r'
linearSolverConvergenceTest             = 'r-true'

periodicDirichletConditions=None

# post processing

if not ctx.useVelocityComponents:
    conservativeFlux = {0:'pwl-bdm-opt'} #'point-eval','pwl-bdm'
else:
    conservativeFlux=None
