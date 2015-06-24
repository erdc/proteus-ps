from math import *
from proteus import *
from proteus.default_p import *
from proteus import Context
ctx = Context.get()

import NavierStokes

# import stuff from guermond_example_variable_density.py
nd = ctx.nd
name = "mass_transport"
domain = ctx.domain


# from pnList in *_so.py  0 = density,  1 = (u,v,p)
coefficients=NavierStokes.MassTransport(velocityFunction=None, #or ctx.velocityFunction to use exact solution (uncoupled transport)
                                        velocityModelIndex=1,  #don't change this unless the order in so-file is changed
                                        useVelocityComponents=False) #set to false to use 'velocity' (possible post-processed)

analyticalSolution = {0:ctx.AnalyticSolutionConverter(ctx.rhotrue)}

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

fluxBoundaryConditions = {0:'outFlow'} #this only has an effect when numericalFlux is not used
