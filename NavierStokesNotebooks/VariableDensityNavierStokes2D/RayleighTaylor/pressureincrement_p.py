from math import *
from proteus import *
from proteus.default_p import *
from proteus import Context
ctx = Context.get()

import NavierStokesVariableDensity as NavierStokes

domain = ctx.domain
nd = ctx.nd
name = "pressureincrement"

#the object for evaluating the coefficients

# from pnList in *_so.py  0 = density,  1 = (u,v),  2 = (pressureincrement),  3 = (pressure)
coefficients=NavierStokes.PressureIncrement2D(bdf=ctx.globalBDFTimeOrder,
                                              chiValue=ctx.chi,
                                              zeroMean=not ctx.useDirichletPressureIncrementBC,
                                              densityModelIndex=0,
                                              velocityModelIndex=1,
                                              velocityFunction=None, # use ctx.velocityFunction for exact velocity
                                              pressureFunction=None,
                                              currentModelIndex=2,
                                              setFirstTimeStepValues=ctx.setFirstTimeStepValues)

def getDBC_p(x,flag):
    return None
#    if flag == ctx.boundaryTags['top']:
#        return lambda x,t: 0.0

def getAdvectiveFlux(x,flag):
    return lambda x,t: 0.0

def getDiffusiveFlux_p(x,flag):
    return lambda x,t: 0.0
#    if flag != ctx.boundaryTags['top']:
#        return None
#    else:
#        return lambda x,t: 0.0

class getIBC_p:
    def __init__(self):
        pass
    def uOfXT(self,x,t):
        return 0.0

initialConditions = {0:getIBC_p()}

dirichletConditions = {0:getDBC_p }
advectiveFluxBoundaryConditions = {0:getAdvectiveFlux}
diffusiveFluxBoundaryConditions = {0:{0:getDiffusiveFlux_p}}
