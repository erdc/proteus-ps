"""
The redistancing equation in the sloshbox test problem.
"""
from proteus import *
from proteus.default_p import *
from math import *
from proteus.mprans import RDLS
from proteus import Context
ctx = Context.get()
domain = ctx.domain
nd = ctx.nd
genMesh = ctx.genMesh

LevelModelType = RDLS.LevelModel

coefficients = RDLS.Coefficients(applyRedistancing=True,
                                 epsFact=ctx.epsFact_redistance,
                                 nModelId=2,
                                 rdModelId=3,
		                 useMetrics=ctx.useMetrics)

def getDBC_rd(x,flag):
    pass

dirichletConditions     = {0:getDBC_rd}
weakDirichletConditions = {0:RDLS.setZeroLSweakDirichletBCsSimple}

advectiveFluxBoundaryConditions =  {}
diffusiveFluxBoundaryConditions = {0:{}}

class PerturbedSurface_phi:
    def uOfXT(self,x,t):
        return ctx.signedDistance(x)

initialConditions  = {0:PerturbedSurface_phi()}
