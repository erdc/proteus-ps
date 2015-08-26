from math import *
from proteus import *
from proteus.default_p import *
from proteus import Context
import navierstokes_vardensity
Context.setFromModule(navierstokes_vardensity)
ctx = Context.get()


import NavierStokesVariableDensity as NavierStokes

domain = ctx.domain
nd = ctx.nd
name = "pi_phi"
coefficients=NavierStokes.L2Projection(projectTime=0.0,
                                       toName='pi',
                                       myModelIndex=5,
                                       toModelIndex=3,
                                       toModel_u_ci=0,
                                       exactFunction = ctx.pitrue)
analyticalSolution = {0:ctx.AnalyticSolutionConverter(ctx.pitrue,T=0.0)}
class getIBC_u:
    def __init__(self):
        self.pitrue=ctx.pitrue
        pass
    def uOfXT(self,x,t):
        return self.pitrue(x,t)
initialConditions = {0:getIBC_u()}
def getDBC_none(x,flag):
    return None
dirichletConditions = {0:getDBC_none}
