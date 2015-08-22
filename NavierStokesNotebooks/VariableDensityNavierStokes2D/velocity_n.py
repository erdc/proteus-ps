from proteus import *
from proteus.default_n import *
from velocity_p import *


triangleOptions = ctx.triangleOptions


femSpaces = {0:FemTools.C0_AffineQuadraticOnSimplexWithNodalBasis, # u velocity space  = P2
             1:FemTools.C0_AffineQuadraticOnSimplexWithNodalBasis} # v velocity space  = P2

timeOrder = ctx.globalBDFTimeOrder

from TimeIntegrationPS import NonConservativeBackwardEuler, NonConservativeVBDF
if timeOrder == 1:
    timeIntegration = NonConservativeBackwardEuler
elif timeOrder == 2:
    timeIntegration = NonConservativeVBDF
else:
    assert False, "BDF order %d for time integration is not supported." % timeOrder


# stepController  = StepControl.Min_dt_cfl_controller
# runCFL = 0.5

stepController  = FixedStep
DT = ctx.DT # overwritten by _so.py file

#Quadrature rules for elements and element  boundaries
elementQuadrature = Quadrature.SimplexGaussQuadrature(ctx.nd,ctx.quad_degree)
elementBoundaryQuadrature = Quadrature.SimplexGaussQuadrature(ctx.nd-1,ctx.quad_degree)


#Matrix type
#numericalFluxType = NumericalFlux.StrongDirichletFactory(fluxBoundaryConditions)
#numericalFluxType = MixedDarcy_exterior
#numericalFluxType = NumericalFlux.Advection_DiagonalUpwind_Diffusion_SIPG_exterior
#numericalFluxType = NumericalFlux.Advection_Diagonal_average
numericalFluxType = NumericalFlux.HamiltonJacobi_DiagonalLesaintRaviart_Diffusion_SIPG_exterior

if ctx.useASGS:
    subgridError = HamiltonJacobiDiffusionReaction_ASGS(coefficients,
                                                        nd=ctx.nd,
                                                        stabFlag='1',
                                                        lag=False)
matrix = LinearAlgebraTools.SparseMatrix
#use petsc solvers wrapped by petsc4py
#numerics.multilevelLinearSolver = LinearSolvers.KSP_petsc4py
#numerics.levelLinearSolver = LinearSolvers.KSP_petsc4py
#using petsc4py requires weak boundary condition enforcement
#can also use our internal wrapper for SuperLU
multilevelLinearSolver = LinearSolvers.LU
levelLinearSolver = LinearSolvers.LU

linear_solver_options_prefix = 'velocity_'

if ctx.opts.parallel:
    multilevelLinearSolver = LinearSolvers.KSP_petsc4py
    levelLinearSolver      = LinearSolvers.KSP_petsc4py
    parallelPartitioningType = ctx.parallelPartitioningType
    nLayersOfOverlapForParallel = ctx.nLayersOfOverlapForParallel
    nonlinearSmoother = None
    linearSmoother    = None

multilevelNonlinearSolver = NonlinearSolvers.Newton
levelNonlinearSolver = NonlinearSolvers.Newton

#linear solve rtolerance

# linTolFac = 0.001  # relatice tolerance for linear solver
# tolFac = 0.0 # absolute tolerance
#
# l_atol_res = 1.0e-5
# nl_atol_res = 1.0e-5

linTolFac = 0.0
l_atol_res = 0.1*ctx.velocity_atol_res
tolFac = 0.0
nl_atol_res = ctx.velocity_atol_res
nonlinearSolverConvergenceTest      = 'r'
levelNonlinearSolverConvergenceTest = 'r'
linearSolverConvergenceTest         = 'r-true'

periodicDirichletConditions=None

# post processing

conservativeFlux=None
