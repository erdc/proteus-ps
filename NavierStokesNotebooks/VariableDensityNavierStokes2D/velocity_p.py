from math import *
from proteus import *
from proteus.default_p import *
from proteus import Context
ctx = Context.get()

import NavierStokesVariableDensity as NavierStokes

domain = ctx.domain
nd = ctx.nd
name = "velocity"

#the object for evaluating the coefficients

# from pnList in *_so.py  0 = density,  1 = (u,v), 2 = (pressureincrement),  3 = (pressure)
coefficients=NavierStokes.VelocityTransport2D(bdf=ctx.globalBDFTimeOrder,
                                              f1ofx=ctx.f1true,
                                              f2ofx=ctx.f2true,
                                              mu=ctx.mu,
                                              densityModelIndex=0,
                                              densityFunction=None, #set to ctx.rhotrue for exact density (uncoupled  flow)
                                              densityGradFunction=None,  #set to ctx.gradrhotrue for exact grad density
                                              currentModelIndex=1,
                                              uFunction=ctx.utrue,
                                              vFunction=ctx.vtrue,
                                              pressureIncrementModelIndex=2,
                                              pressureIncrementFunction=None, # set to ctx.pitrue for exact pressure increment
                                              pressureIncrementGradFunction=None, # set to ctx.gradpitrue for exact pressure increment
                                              pressureModelIndex=3,
                                              pressureFunction=None, # set to ctx.ptrue for exact pressure
                                              pressureGradFunction=None, # set to ctx.gradptrue for exact pressure
                                              useStabilityTerms=ctx.useStabilityTerms,
                                              setFirstTimeStepValues=ctx.setFirstTimeStepValues,
                                              useNonlinearAdvection=ctx.useNonlinearAdvection,
                                              usePressureExtrapolations=ctx.usePressureExtrapolations,
                                              useConservativePressureTerm=ctx.useConservativePressureTerm)

if ctx.opts.analytical:
    analyticalSolution = {0:ctx.AnalyticSolutionConverter(ctx.utrue,ctx.gradutrue),
                          1:ctx.AnalyticSolutionConverter(ctx.vtrue,ctx.gradvtrue)}

# Define boundary conditions and initial conditions of system

def getDBC_u(x,flag):
    if flag in [ctx.boundaryTags['bottom'],
                ctx.boundaryTags['top'],
                ctx.boundaryTags['fixed']]:
       return lambda x,t: ctx.utrue(x,t)
    else:
        return None

def getDBC_v(x,flag):
    if flag in [ctx.boundaryTags['bottom'],
                ctx.boundaryTags['top'],
                ctx.boundaryTags['fixed']]:
        return lambda x,t: ctx.vtrue(x,t)
    else:
        return None

def getDFlux(x,flag):
    if flag == 0: # artificial boundary from parallelization
       return lambda x,t: 0.0
    else:
        return None

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
                     1:getIBC_v()}

dirichletConditions = {0:getDBC_u,
                       1:getDBC_v }

diffusiveFluxBoundaryConditions = {0:{0:getDFlux},
                                   1:{1:getDFlux}}
