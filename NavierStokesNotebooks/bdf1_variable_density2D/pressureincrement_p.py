from math import *
from proteus import *
from proteus.default_p import *
from proteus import Context
ctx = Context.get()

import NavierStokesVariableDensity as NavierStokes

domain = ctx.domain
nd = ctx.nd
name = "pressureincrement_2d"

#the object for evaluating the coefficients   

# from pnList in *_so.py  0 = density,  1 = (u,v), 2 = (pressureincrement),  3 = (pressure)
coefficients=NavierStokes.PressureIncrement2D(velocityModelIndex=1,
                                              velocityFunction=None, # use ctx.velocityFunction for exact velocity
                                              useVelocityComponents=ctx.useVelocityComponents,
                                              densityModelIndex=0,
                                              chiValue=None)


analyticalSolution = {0:ctx.AnalyticSolutionConverter(ctx.pitrue,ctx.gradpitrue)}
                      
# analyticalSolutionVelocity = {2:ctx.AnalyticSolutionConverter(ctx.velocityFunctionLocal)}


# Define boundary conditions and initial conditions of system

def getNBC_p(x,flag):
    if flag == ctx.boundaryTags['top']:
        return lambda x,t: 0.0*x
    elif flag == ctx.boundaryTags['bottom']:
        return lambda x,t: 0.0*x
    else:
        return None

def getNone(x,flag):
    return None

def getZeroFlux(x,flag):
    if flag == ctx.boundaryTags['top']:
        return lambda x,t: 0.0
    elif flag == ctx.boundaryTags['bottom']:
        return lambda x,t: 0.0
    else:
        return None

class getIBC_p:
    def __init__(self):
        self.pitrue=ctx.pitrue
        pass
    def uOfXT(self,x,t):
        return self.pitrue(x,t)

initialConditions = {0:getIBC_p()}

# dirichletConditions = {0:getDBC_p }

# advectiveFluxBoundaryConditions = {0:getNone} # check this?

diffusiveFluxBoundaryConditions = {0:{0:getZeroFlux}}

fluxBoundaryConditions = {0:'noFlow'}
