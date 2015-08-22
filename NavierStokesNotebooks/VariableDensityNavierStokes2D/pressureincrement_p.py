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
                                              pressureFunction=ctx.ptrue,
                                              currentModelIndex=2,
                                              setFirstTimeStepValues=ctx.setFirstTimeStepValues)

if ctx.opts.analytical:
    analyticalSolution = {0:ctx.AnalyticSolutionConverter(ctx.pitrue,ctx.gradpitrue)}

def getDBC_p(x,flag):
    if flag in [ctx.boundaryTags['bottom'],
                ctx.boundaryTags['top'],
                ctx.boundaryTags['fixed']]:
        return lambda x,t: 0.0

def getDBC_fixed(x,flag):
    if flag == ctx.boundaryTags['fixed']:
        return lambda x,t: 0.0
    else:
        return None

def getNone(x,flag):
    return None

def getAdvectiveFlux(x,flag):
    if flag == 0:
        return lambda x,t: 0.0

def getDiffusiveFlux_p(x,flag):
    if flag == 0:
        return lambda x,t: 0.0

def getDiffusiveFlux_fixed(x,flag):
    if flag != ctx.boundaryTags['fixed']:
        return lambda x,t: 0.0

def getDiffusiveFlux_None(x,flag):
    return lambda x,t: 0.0

class getIBC_p:
    def __init__(self):
        self.pitrue=ctx.pitrue
        pass
    def uOfXT(self,x,t):
        return self.pitrue(x,t)

initialConditions = {0:getIBC_p()}

if ctx.useDirichletPressureIncrementBC:  # there are Dirichlet BC somewhere on boundary
    dirichletConditions = {0:getDBC_p }
    advectiveFluxBoundaryConditions = {0:getAdvectiveFlux}
    diffusiveFluxBoundaryConditions = {0:{0:getDiffusiveFlux_p}}
else: # pure neumann laplacian problem
    if ctx.useNoFluxPressureIncrementBC:  # solve pure neumann laplacian problem by removing null space using petsc solver
        dirichletConditions = {0:getNone }
        advectiveFluxBoundaryConditions = {0:getAdvectiveFlux}
        diffusiveFluxBoundaryConditions = {0:{0:getDiffusiveFlux_None}}
    else:   # solve pure neumann laplacian problem using a fixed dof then adjust for zero mean
        dirichletConditions = {0:getDBC_fixed }
        advectiveFluxBoundaryConditions = {0:getAdvectiveFlux}
        diffusiveFluxBoundaryConditions = {0:{0:getDiffusiveFlux_fixed}}
