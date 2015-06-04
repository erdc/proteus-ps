from proteus import *
from proteus.default_p import *
from proteus.mprans import NCLS
from proteus import Context
ctx = Context.get()
domain = ctx.domain
nd = ctx.nd
genMesh = ctx.genMesh

LevelModelType = NCLS.LevelModel

coefficients = NCLS.Coefficients(V_model=0,
                                 RD_model=3,
                                 ME_model=2,
                                 checkMass=False,
                                 useMetrics=ctx.useMetrics,
                                 epsFact=ctx.epsFact_consrv_heaviside,
                                 sc_uref=ctx.ls_sc_uref,
                                 sc_beta=ctx.ls_sc_beta)

def getDBC_ls(x,flag):
    return None

dirichletConditions = {0:getDBC_ls}

advectiveFluxBoundaryConditions =  {}
diffusiveFluxBoundaryConditions = {0:{}}

class PerturbedSurface_phi:
    def uOfXT(self,x,t):
        return ctx.signedDistance(x)

initialConditions  = {0:PerturbedSurface_phi()}
