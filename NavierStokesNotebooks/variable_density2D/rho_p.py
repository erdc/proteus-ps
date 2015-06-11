from math import *
from proteus import *
from proteus.default_p import *
from proteus import Context
ctx = Context.get()

import NavierStokes

domain = ctx.domain
nd = ctx.nd
name = "mass_transport"

# coefficients=NavierStokes.MassTransport(velocityFunction=ctx.velocityFunction,useVelocityFunction=True)
# coefficients=NavierStokes.MassTransport(velocityFunction=None,velocityModelIndex=1,useVelocityFunction=False) # from pnList in *_so.py  0 = density,  1 = (u,v,p)
coefficients=NavierStokes.MassTransport(velocityFunction=ctx.velocityFunction,
                                        velocityModelIndex=1,
                                        useVelocityFunction=False,
                                        useVelocityComponents=True) # from pnList in *_so.py  0 = density,  1 = (u,v,p)

#this function's job is to return another function holding the Dirichlet boundary conditions 
# wherever they are set

def getDBC_rho(x,flag):
    if flag in [ctx.boundaryTags['bottom'],
                ctx.boundaryTags['top']]:
        return lambda x,t: ctx.rhotrue(x,t)
    
def getNone(x,flag):
    return None

class getIBC_rho:
    def __init__(self):
        self.rhotrue=ctx.rhotrue
        pass
    def uOfXT(self,x,t):
        return self.rhotrue(x,t)
    
initialConditions = {0:getIBC_rho()}

dirichletConditions = {0:getDBC_rho}

advectiveFluxBoundaryConditions = {0:getNone}
# fluxBoundaryConditions = {0:'outFlow'}
fluxBoundaryConditions = {0:'noFlux'}
