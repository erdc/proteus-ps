from math import *
from proteus import *
from proteus.default_p import *
from proteus import Context
ctx = Context.get()


import NavierStokesVariableDensity as NavierStokes

domain = ctx.domain
nd = ctx.nd
name = "StokesProjection"
coefficients=NavierStokes.StokesProjection2D(myModelIndex=4,
                                             toModelIndex_v=1,
                                             toModelIndex_p=3,
                                             projectTime=0.0,
                                             grad_u_function = ctx.gradutrue,
                                             grad_v_function = ctx.gradvtrue,
                                             p_function = ctx.ptrue)

analyticalSolution = {0:ctx.AnalyticSolutionConverter(ctx.utrue,T=0.0),
                      1:ctx.AnalyticSolutionConverter(ctx.vtrue,T=0.0),
                      2:ctx.AnalyticSolutionConverter(ctx.ptrue,T=0.0)}
class getIBC_u:
    def __init__(self):
        self.utrue=ctx.utrue
        pass
    def uOfXT(self,x,t):
        return self.utrue(x,0.0)

class getIBC_v:
    def __init__(self):
        self.vtrue=ctx.vtrue
        pass
    def uOfXT(self,x,t):
        return self.vtrue(x,0.0)

class getIBC_p:
    def __init__(self):
        self.ptrue=ctx.ptrue
        pass
    def uOfXT(self,x,t):
        return self.ptrue(x,0.0)

initialConditions = {0:getIBC_u(),
                     1:getIBC_v(),
                     2:getIBC_p()}

def getDBC_u(x,flag):
    if flag > 0:
        return lambda x,t: ctx.utrue(x,0.0)

def getDBC_v(x,flag):
    if flag > 0:
        return lambda x,t: ctx.vtrue(x,0.0)
    #return None

def getDBC_p(x,flag):
    if flag > 0:
        return lambda x,t: ctx.ptrue(x,0.0)
    #return None

dirichletConditions = {0:getDBC_u,
                       1:getDBC_v,
                       2:getDBC_p}
