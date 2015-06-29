from math import *
from proteus import *
from proteus.default_p import *
from proteus import Context
ctx = Context.get()

import NavierStokesVariableDensity as NavierStokes

domain = ctx.domain
nd = ctx.nd
name = "density_2d"


# from pnList in *_so.py  0 = density,  1 = (u,v), 2 = (pressureincrement),  3 = (pressure)
coefficients=NavierStokes.MassTransport(velocityFunction=None, #or ctx.velocityFunction to use exact solution (uncoupled transport)
                                        velocityModelIndex=1,  #don't change this unless the order in so-file is changed
                                        divVelocityFunction=False, # or ctx.divVelocityFunction to use exact divergence solution
                                        useVelocityComponents=ctx.useVelocityComponents, #set to false to use 'velocity' (possible post-processed)
                                        useStabilityTerms=ctx.useStabilityTerms) 

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
