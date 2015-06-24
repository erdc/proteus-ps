from math import *
from proteus import *
from proteus.default_p import *
from proteus import Context
ctx = Context.get()

import NavierStokes

# import stuff from guermond_example_variable_density.py
nd = ctx.nd
name = "navier_stokes_2d"
domain = ctx.domain



#the object for evaluating the coefficients   
# # uncoupled system
# coefficients=NavierStokes.NavierStokes2D(f1ofx=ctx.f1true,
#                                          f2ofx=ctx.f2true,
#                                          mu=ctx.mu,
#                                          densityFunction=ctx.rhotrue)

coefficients=NavierStokes.NavierStokes2D(f1ofx=ctx.f1true,
                                         f2ofx=ctx.f2true,
                                         mu=ctx.mu,
                                         densityFunction=None, #set to ctx.rhotrue for exact density (uncoupled  flow)
                                         densityModelIndex=0)  # from pnList in *_so.py  0 = density,  1 = (u,v,p)


analyticalSolution = {0:ctx.AnalyticSolutionConverter(ctx.utrue,ctx.gradutrue),
                      1:ctx.AnalyticSolutionConverter(ctx.vtrue,ctx.gradvtrue),
                      2:ctx.AnalyticSolutionConverter(ctx.ptrue,ctx.gradptrue)}
                      
analyticalSolutionVelocity = {2:ctx.AnalyticSolutionConverter(ctx.velocityFunctionLocal)}


# Define boundary conditions and initial conditions of system

def getDBC_p(x,flag):
    if flag == ctx.boundaryTags['top']:
        return lambda x,t: ctx.ptrue(x,t)
    elif flag == ctx.boundaryTags['bottom']:
        return lambda x,t: ctx.ptrue(x,t)
    else:
        return None

def getDBC_u(x,flag):
    if flag == ctx.boundaryTags['top']:
        return lambda x,t: ctx.utrue(x,t)
    elif flag == ctx.boundaryTags['bottom']:
        return lambda x,t: ctx.utrue(x,t)
    else:
        return None

def getDBC_v(x,flag):
    if flag == ctx.boundaryTags['top']:
        return lambda x,t: ctx.vtrue(x,t)
    elif flag == ctx.boundaryTags['bottom']:
        return lambda x,t: ctx.vtrue(x,t)
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
        self.ptrue=ctx.ptrue
        pass
    def uOfXT(self,x,t):
        return self.ptrue(x,t)

class getIBC_u:
    def __init__(self):
        self.utrue=ctx.utrue
        pass
    def uOfXT(self,x,t):
        return self.utrue(x,t)

class getIBC_v:
    def __init__(self):
        self.vtrue=ctx.vtrue
        pass
    def uOfXT(self,x,t):
        return self.vtrue(x,t)

initialConditions = {0:getIBC_u(),
                     1:getIBC_v(),
                     2:getIBC_p()}

dirichletConditions = {0:getDBC_u,
                       1:getDBC_v,
                       2:getDBC_p }

advectiveFluxBoundaryConditions = {2:getNone}

diffusiveFluxBoundaryConditions = {0:{0:getZeroFlux},
                                   1:{1:getZeroFlux}}

fluxBoundaryConditions = {0:'noFlow',1:'noFlow',2:'noFlow'}
