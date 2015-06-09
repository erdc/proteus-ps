from math import *
from proteus import *
from proteus.default_p import *
from proteus.mprans import RANS2P
from proteus import Context
ctx = Context.get()

domain = ctx.domain
nd = ctx.nd
name = 'rans2p'
genMesh = ctx.genMesh
LevelModelType = RANS2P.LevelModel
if ctx.useOnlyVF:
    LS_model = None
else:
    LS_model = 2
if ctx.useRANS >= 1:
    Closure_0_model = 5; Closure_1_model=6
    if ctx.useOnlyVF:
        Closure_0_model=2; Closure_1_model=3
else:
    Closure_0_model = None
    Closure_1_model = None

coefficients = RANS2P.Coefficients(epsFact=ctx.epsFact_viscosity,
                                   sigma=0.0,
                                   rho_0 = ctx.rho_0,
                                   nu_0 = ctx.nu_0,
                                   rho_1 = ctx.rho_1,
                                   nu_1 = ctx.nu_1,
                                   g=ctx.g,
                                   nd=ctx.nd,
                                   VF_model=1,
                                   LS_model=LS_model,
                                   Closure_0_model=Closure_0_model,
                                   Closure_1_model=Closure_1_model,
                                   epsFact_density=ctx.epsFact_density,
                                   stokes=False,
                                   useVF=ctx.useVF,
                                   useRBLES=ctx.useRBLES,
                                   useMetrics=ctx.useMetrics,
                                   eb_adjoint_sigma=1.0,
                                   forceStrongDirichlet = \
                                   ctx.ns_forceStrongDirichlet,
                                   turbulenceClosureModel=ctx.ns_closure)

def getDBC_p(x,flag):
    if flag == ctx.boundaryTags['top']:
        return lambda x,t: 0.0

def getDBC_u(x,flag):
    #return None
    if flag == ctx.boundaryTags['top']:
        return lambda x,t: 0.0

def getDBC_v(x,flag):
    return None

dirichletConditions = {0:getDBC_p,
                       1:getDBC_u,
                       2:getDBC_v}

def getAFBC_p(x,flag):
    if flag != ctx.boundaryTags['top']:
        return lambda x,t: 0.0

def getAFBC_u(x,flag):
    if flag != ctx.boundaryTags['top']:
        return lambda x,t: 0.0

def getAFBC_v(x,flag):
    if flag != ctx.boundaryTags['top']:
        return lambda x,t: 0.0

def getDFBC_u(x,flag):
    if flag != ctx.boundaryTags['top']:
        return lambda x,t: 0.0

def getDFBC_v(x,flag):
    return lambda x,t: 0.0

advectiveFluxBoundaryConditions =  {0:getAFBC_p,
                                    1:getAFBC_u,
                                    2:getAFBC_v}

diffusiveFluxBoundaryConditions = {0:{},
                                   1:{1:getDFBC_u},
                                   2:{2:getDFBC_v}}

fluxBoundaryConditions = {}
stressFluxBoundaryConditions = {}
weakDirichletConditions = None
periodicDirichletConditions = None
sd = True
movingDomain = False
bcsTimeDependent=False
class PerturbedSurface_p:
    def __init__(self,waterdepth,amplitude):
        self.waterdepth=waterdepth
        self.amplitude=amplitude
    def uOfXT(self,x,t):
        if ctx.signedDistance(x) < 0:
            return -(L[1] -
                     self.waterdepth -
                     self.amplitude * math.cos(x[0])
            )*ctx.rho_1*ctx.g[1] - (self.waterdepth +
                            self.amplitude * math.cos(x[0]) -
                            x[1])*ctx.rho_0*ctx.g[1]
        else:
            return -(ctx.L[1] - x[1])*ctx.rho_1*ctx.g[1]

class AtRest:
    def __init__(self):
        pass
    def uOfXT(self,x,t):
        return 0.0

initialConditions = {0:PerturbedSurface_p(ctx.h,ctx.A),
                     1:AtRest(),
                     2:AtRest()}
