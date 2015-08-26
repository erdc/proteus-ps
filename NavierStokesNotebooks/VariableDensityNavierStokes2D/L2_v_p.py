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
name = "pi_v"
coefficients=NavierStokes.L2Projection(projectTime=0.0,
                                       toName='v',
                                       myModelIndex=7,
                                       toModelIndex=1,
                                       toModel_u_ci=1,
                                       exactFunction = ctx.vtrue)
analyticalSolution = {0:ctx.AnalyticSolutionConverter(ctx.vtrue,T=0.0)}
class getIBC_u:
    def __init__(self):
        self.vtrue=ctx.vtrue
        pass
    def uOfXT(self,x,t):
        return self.vtrue(x,0.0)
initialConditions = {0:getIBC_u()}
def getDBC_none(x,flag):
    return None
dirichletConditions = {0:getDBC_none}
