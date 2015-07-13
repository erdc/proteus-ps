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
multilevelLinearSolver = LinearSolvers.LU
levelLinearSolver = LinearSolvers.LU

multilevelNonlinearSolver = NonlinearSolvers.Newton
levelNonlinearSolver = NonlinearSolvers.Newton

#linear solve rtolerance

linTolFac = 0.001
l_atol_res = 0.001*ctx.ns_nl_atol_res
tolFac = 0.0
nl_atol_res = ctx.ns_nl_atol_res

periodicDirichletConditions=None

# post processing

conservativeFlux=None
#all of these  should work
#conservativeFlux = {0:'point-eval'}
# conservativeFlux = {0:'pwl-bdm'}
# conservativeFlux = {0:'pwl-bdm-opt'}
