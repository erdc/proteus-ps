from math import *
from proteus import *
from proteus.default_p import *
from proteus import Context
ctx = Context.get()

import NavierStokesVariableDensity as NavierStokes

domain = ctx.domain
nd = ctx.nd
name = "pressure"


#the object for evaluating the coefficients
# from pnList in *_so.py  0 = density,  1 = (u,v), 2 = (pressureincrement),  3 = (pressure)
coefficients=NavierStokes.Pressure2D(bdf=ctx.globalBDFTimeOrder,
                                     mu=ctx.mu,
                                     chiValue=ctx.chi,
                                     velocityModelIndex=1,
                                     velocityFunction=None, # use ctx.velocityFunction for exact velocity
                                     useVelocityComponents=ctx.useVelocityComponents,
                                     pressureIncrementModelIndex=2,
                                     pressureIncrementFunction=None,  # use ctx.gradpitrue for exact pressure increment (=0)
                                     useRotationalModel=ctx.useRotationalModel,
                                     currentModelIndex=3,
                                     pressureFunction=None,
                                     setFirstTimeStepValues=ctx.setFirstTimeStepValues,
                                     usePressureExtrapolations=ctx.usePressureExtrapolations)

if ctx.opts.analytical:
    analyticalSolution = {0:ctx.AnalyticSolutionConverter(ctx.ptrue,ctx.gradptrue)}

# analyticalSolutionVelocity = {0:ctx.AnalyticSolutionConverter(ctx.velocityFunctionLocal)}


# Define boundary conditions and initial conditions of system

def getDBC_p(x,flag):
    return None
#    if flag == ctx.boundaryTags['top']:
#        return lambda x,t: 0.0
#    else:
#        return None

def getFlux(x,flag):
    if flag == 0:
        return lambda x,t: 0.0
    else:
        return None

class getIBC_p:
    def __init__(self):
        pass
    def uOfXT(self,x,t):
        from math import cos,pi
        z = - 0.1*cos(2.0*pi*x[0]/ctx.d)
        if (x[1] - z) > 0:
            p = (2*ctx.d - x[1])*ctx.rho_max*ctx.g
        if (x[1] - z) < 0:
            p = (2-z)*ctx.d*ctx.rho_max*ctx.g + (z - x[1])*ctx.rho_min*ctx.g
        return 0.0#self.ptrue(x,t)

initialConditions = {0:getIBC_p()}

dirichletConditions = {0:getDBC_p } # pressure bc are explicitly set
advectiveFluxBoundaryConditions = {0:getFlux}
