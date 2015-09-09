from math import *
from proteus import *
from proteus.default_p import *
from proteus import Context
ctx = Context.get()

import NavierStokesVariableDensity as NavierStokes

domain = ctx.domain
nd = ctx.nd
name = "density"


# from pnList in *_so.py  0 = density,  1 = (u,v), 2 = (pressureincrement),  3 = (pressure)
coefficients=NavierStokes.DensityTransport2D(bdf=ctx.globalBDFTimeOrder,
                                             chiValue=ctx.chi,
                                             currentModelIndex=0,
                                             densityFunction=None,#ctx.rhotrue,
                                             velocityModelIndex=1,  #don't change this unless the order in so-file is changed
                                             velocityFunction=None, #or ctx.velocityFunction to use exact solution (uncoupled transport)
                                             divVelocityFunction=None, # or ctx.divVelocityFunction to use exact divergence solution
                                             useVelocityComponents=ctx.useVelocityComponents, #set to false to use 'velocity' (possible post-processed)
                                             pressureIncrementModelIndex=2,
                                             useStabilityTerms=ctx.useStabilityTerms,
                                             setFirstTimeStepValues=ctx.setFirstTimeStepValues,
                                             useNumericalFluxEbqe=ctx.useNumericalFluxEbqe)

if ctx.opts.analytical:
   analyticalSolution = {0:ctx.AnalyticSolutionConverter(ctx.rhotrue)}

#this function's job is to return another function holding the Dirichlet boundary conditions
# wherever they are set

def getDBC_rho(x,flag):
   return None

def getAFlux(x,flag):
   return lambda x,t: 0.0

def getDFlux(x,flag):
   return lambda x,t: 0.0

class getIBC_rho:
    def __init__(self):
        pass
    def uOfXT(self,x,t):
       return ctx.rho_init(x[0],x[1])

initialConditions = {0:getIBC_rho()}

dirichletConditions = {0:getDBC_rho}

advectiveFluxBoundaryConditions = {0:getAFlux}
diffusiveFluxBoundaryConditions = {0:{0:getDFlux}}
