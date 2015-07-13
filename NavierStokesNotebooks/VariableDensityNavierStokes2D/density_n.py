from proteus import *
from proteus.default_n import *
from density_p import *


triangleOptions = ctx.triangleOptions


femSpaces = {0:FemTools.C0_AffineQuadraticOnSimplexWithNodalBasis} # density space = P2

# timeIntegration = TimeIntegration.BackwardEuler
# timeIntegration = TimeIntegration.BackwardEuler_cfl
# timeIntegration = TimeIntegration.VBDF

timeOrder = ctx.globalBDFTimeOrder

if timeOrder == 1:
    timeIntegration = TimeIntegration.BackwardEuler
elif timeOrder == 2:
    timeIntegration = TimeIntegration.VBDF
else:
    assert False, "BDF order %d for time integration is not supported." % timeOrder

# stepController  = StepControl.Min_dt_cfl_controller
# runCFL= 0.99
# runCFL= 0.5

stepController=FixedStep
DT = ctx.DT

#Quadrature rules for elements and element  boundaries
elementQuadrature = Quadrature.SimplexGaussQuadrature(ctx.nd,ctx.quad_degree)
elementBoundaryQuadrature = Quadrature.SimplexGaussQuadrature(ctx.nd-1,ctx.quad_degree)


if not ctx.useStabilityTerms:
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

conservativeFlux=None
