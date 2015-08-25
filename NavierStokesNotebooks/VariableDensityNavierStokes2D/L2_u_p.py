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
name = "pi_u"
coefficients=NavierStokes.L2Projection(projectTime=0.0,
                                       toName='u',
                                       myModelIndex=0,
                                       exactFunction = ctx.utrue)
analyticalSolution = {0:ctx.AnalyticSolutionConverter(ctx.utrue,T=0.0)}
class getIBC_u:
    def __init__(self):
        self.utrue=ctx.utrue
        pass
    def uOfXT(self,x,t):
        return self.utrue(x,0.0)
initialConditions = {0:getIBC_u()}
def getDBC_none(x,flag):
    return None
dirichletConditions = {0:getDBC_none}
