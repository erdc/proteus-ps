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
                                              f1ofx=None,#ctx.f1true,
                                              f2ofx=None,#ctx.f2true,
                                              mu=ctx.mu,
                                              densityModelIndex=0,
                                              densityFunction=None, #set to ctx.rhotrue for exact density (uncoupled  flow)
                                              densityGradFunction=None,  #set to ctx.gradrhotrue for exact grad density
                                              currentModelIndex=1,
                                              uFunction=None,#ctx.utrue,
                                              vFunction=None,#ctx.vtrue,
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
                                              useConservativePressureTerm=ctx.useConservativePressureTerm,
                                              g=ctx.gvec)

if ctx.opts.analytical:
    analyticalSolution = {0:ctx.AnalyticSolutionConverter(ctx.utrue,ctx.gradutrue),
                          1:ctx.AnalyticSolutionConverter(ctx.vtrue,ctx.gradvtrue)}

# Define boundary conditions and initial conditions of system

def getDBC_u(x,flag):
    return None

def getDBC_v(x,flag):
    return None

def getDFlux(x,flag):
    return lambda x,t: 0.0

class getIBC_u:
    def __init__(self):
        pass
    def uOfXT(self,x,t):
        return 0.0

class getIBC_v:
    def __init__(self):
        pass
    def uOfXT(self,x,t):
        return 0.0

initialConditions = {0:getIBC_u(),
                     1:getIBC_v()}

dirichletConditions = {0:getDBC_u,
                       1:getDBC_v }

diffusiveFluxBoundaryConditions = {0:{0:getDFlux},
                                   1:{1:getDFlux}}
