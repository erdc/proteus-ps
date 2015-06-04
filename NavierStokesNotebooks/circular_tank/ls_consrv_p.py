from proteus import *
from proteus.default_p import *
from proteus.mprans import MCorr
from proteus import Context
ctx = Context.get()
domain = ctx.domain
nd = ctx.nd
genMesh = ctx.genMesh

LevelModelType = MCorr.LevelModel

coefficients = MCorr.Coefficients(LSModel_index=2,
                                  V_model=0,
                                  me_model=4,
                                  VOFModel_index=1,
                                  applyCorrection=ctx.applyCorrection,
                                  nd=ctx.nd,
                                  checkMass=False,
                                  useMetrics=ctx.useMetrics,
                                  epsFactHeaviside= \
                                  ctx.epsFact_consrv_heaviside,
                                  epsFactDirac=ctx.epsFact_consrv_dirac,
                                  epsFactDiffusion= \
                                  ctx.epsFact_consrv_diffusion)

class zero_phi:
    def __init__(self):
        pass
    def uOfX(self,X):
        return 0.0
    def uOfXT(self,X,t):
        return 0.0

initialConditions  = {0:zero_phi()}
