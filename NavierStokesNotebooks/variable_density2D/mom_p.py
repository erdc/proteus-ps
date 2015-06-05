from math import *
from proteus import *
from proteus.default_p import *
from proteus import Context
ctx = Context.get()

import NavierStokes

domain = ctx.domain
nd = ctx.nd
name = "navier_stokes_2d"


#the object for evaluating the coefficients   
coefficients=NavierStokes.NavierStokes2D(f1ofx=ctx.f1true,
                                         f2ofx=ctx.f2true,
                                         mu=ctx.mu,
                                         densityFunction=ctx.rhotrue)   



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
    
physics.initialConditions = {0:getIBC_u(),
                             1:getIBC_v(),
                             2:getIBC_p()}

physics.dirichletConditions = {0:getDBC_u,
                               1:getDBC_v,
                               2:getDBC_p }

physics.advectiveFluxBoundaryConditions = {2:getNone}#dummy condition for non-existent  advective flux
#physics.advectiveFluxBoundaryConditions = {1:getZeroFlux}#dummy condition for non-existent  advective flux

physics.diffusiveFluxBoundaryConditions = {0:{0:getZeroFlux},
                                           1:{1:getZeroFlux}}#viscous flux
physics.fluxBoundaryConditions = {0:'outFlow',1:'outFlow',2:'mixedFlow'}
#physics.fluxBoundaryConditions = {0:'setFlow',1:'setFlow',2:'setFlow'}