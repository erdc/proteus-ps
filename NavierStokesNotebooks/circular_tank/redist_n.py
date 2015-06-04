from proteus.default_n import *
from proteus import *
from redist_p import *
from proteus import Context
ctx = Context.get()

nl_atol_res = ctx.rd_nl_atol_res
tolFac = 0.0
linTolFac = 0.0
l_atol_res = 0.001*ctx.rd_nl_atol_res

if ctx.redist_Newton:
    timeIntegration = TimeIntegration.NoIntegration
    stepController = StepControl.Newton_controller
    maxNonlinearIts = 25
    maxLineSearches = 0
    nonlinearSolverConvergenceTest = 'r'
    levelNonlinearSolverConvergenceTest = 'r'
    linearSolverConvergenceTest = 'r-true'
    useEisenstatWalker = True
else:
    timeIntegration = TimeIntegration.BackwardEuler_cfl
    stepController = RDLS.PsiTC
    runCFL=1.0
    psitc['nStepsForce']=3
    psitc['nStepsMax']=25
    psitc['reduceRatio']=2.0
    psitc['startRatio']=1.0
    rtol_res[0] = 0.0
    atol_res[0] = ctx.rd_nl_atol_res
    useEisenstatWalker = False
    maxNonlinearIts = 1
    maxLineSearches = 0
    nonlinearSolverConvergenceTest = 'rits'
    levelNonlinearSolverConvergenceTest = 'rits'
    linearSolverConvergenceTest = 'r-true'

femSpaces = {0:ctx.basis}
elementQuadrature = ctx.elementQuadrature
elementBoundaryQuadrature = ctx.elementBoundaryQuadrature

massLumping       = False
numericalFluxType = DoNothing
conservativeFlux  = None
subgridError      = RDLS.SubgridError(coefficients,nd)
shockCapturing    = RDLS.ShockCapturing(coefficients,
                                        ctx.nd,
                                        shockCapturingFactor= \
                                        ctx.rd_shockCapturingFactor,
                                        lag=ctx.rd_lag_shockCapturing)

fullNewtonFlag = True
multilevelNonlinearSolver  = Newton
levelNonlinearSolver       = Newton

nonlinearSmoother = NLGaussSeidel
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

linear_solver_options_prefix = 'rdls_'

auxiliaryVariables=[]
