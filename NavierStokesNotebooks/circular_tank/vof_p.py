from proteus import *
from proteus.default_p import *
from proteus.ctransportCoefficients import smoothedHeaviside
from proteus.mprans import VOF
from proteus import Context
ctx = Context.get()
domain = ctx.domain
nd = ctx.nd
genMesh = ctx.genMesh
LevelModelType = VOF.LevelModel
if ctx.useOnlyVF:
    RD_model = None
    LS_model = None
else:
    RD_model = 3
    LS_model = 2
coefficients = VOF.Coefficients(LS_model=LS_model,
                                V_model=0,
                                RD_model=RD_model,
                                ME_model=1,
                                checkMass=False,
                                useMetrics=ctx.useMetrics,
                                epsFact=ctx.epsFact_vof,
                                sc_uref=ctx.vof_sc_uref,
                                sc_beta=ctx.vof_sc_beta)

def getDBC_vof(x,flag):
   if flag == ctx.boundaryTags['top']:
       return lambda x,t: 1.0

dirichletConditions = {0:getDBC_vof}

def getAFBC_vof(x,flag):
    if flag == ctx.boundaryTags['top']:
        return None
    else:
        return lambda x,t: 0.0

advectiveFluxBoundaryConditions = {0:getAFBC_vof}
diffusiveFluxBoundaryConditions = {0:{}}

class PerturbedSurface_H:
    def uOfXT(self,x,t):
        return smoothedHeaviside(ctx.epsFact_consrv_heaviside*ctx.he,
                                 ctx.signedDistance(x))

initialConditions  = {0:PerturbedSurface_H()}
