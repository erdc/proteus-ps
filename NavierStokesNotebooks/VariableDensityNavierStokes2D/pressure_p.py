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
coefficients=NavierStokes.Pressure2D(mu=ctx.mu,
                                     chiValue=ctx.chi,
                                     velocityModelIndex=1,
                                     pressureIncrementModelIndex=2,
                                     pressureFunction=ctx.ptrue,
                                     useRotationalModel=ctx.useRotationalModel,
                                     currentModelIndex=3,
                                     setFirstTimeStepValues=ctx.setFirstTimeStepValues)

if ctx.opts.analytical:
    analyticalSolution = {0:ctx.AnalyticSolutionConverter(ctx.ptrue,ctx.gradptrue)}

# analyticalSolutionVelocity = {0:ctx.AnalyticSolutionConverter(ctx.velocityFunctionLocal)}


# Define boundary conditions and initial conditions of system

def getDBC_p(x,flag):
    if flag in [ctx.boundaryTags['bottom'],
                ctx.boundaryTags['top'],
                ctx.boundaryTags['fixed']]:
        return lambda x,t: ctx.ptrue(x,t)
    else:
        return None

def getNone(x,flag):
    return None

def getZeroFlux(x,flag):
    if flag in [ctx.boundaryTags['bottom'],
                ctx.boundaryTags['top'],
                ctx.boundaryTags['fixed']]:
        return lambda x,t: 0.0
    else:
        return None

class getIBC_p:
    def __init__(self):
        self.ptrue=ctx.ptrue
        pass
    def uOfXT(self,x,t):
        return self.ptrue(x,t)

initialConditions = {0:getIBC_p()}

if ctx.useDirichletPressureBC:
    dirichletConditions = {0:getDBC_p } # pressure bc are explicitly set
else:
    dirichletConditions = {0:getNone } # pressure bc are set by pressure increment

advectiveFluxBoundaryConditions = {0:getNone} # check this?
