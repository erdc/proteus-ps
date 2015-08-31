from math import *
import proteus.MeshTools
from proteus import Domain
from proteus.default_n import *
from proteus.Profiling import logEvent
import numpy as np

from proteus import Context
opts = Context.Options([
    ("parallel", False, "Run in parallel mode"),
    ("analytical", False, "Archive the analytical solution")
])


#  Discretization
nd = 2

# Numerics
quad_degree = 5  # exact for polynomials of this degree

# Model Flags
globalBDFTimeOrder = 2 # 1 or 2 for time integration algorithms
useRotationalModel = True #  Standard vs Rotational models in pressure update
useStabilityTerms = True  # stability terms in density and velocity models (should always be True)
useNonlinearAdvection = False # switches between extrapolated and fully nonlinear advection in velocity model
useNumericalFluxEbqe = True # ebqe history manipulation use ebqe or numericalFlux.ebqe which is exact
useDirichletPressureBC = False  # Dirichlet bc pressure or zeroMean pressure increment
useDirichletPressureIncrementBC = False  # Dirichlet bc pressure or zeroMean pressure increment
useNoFluxPressureIncrementBC = True # use petsc builtin pure neumann laplacian solver
useVelocityComponents = False# False uses post processed velocity,
useScaleUpTimeStepsBDF2 = False  # Time steps = [dt^2, 2dt^2, 4dt^2, ... dt, ... , dt, T-tLast]
setFirstTimeStepValues = False # interpolate the first step as well as the 0th step from exact solutions
usePressureExtrapolations = False # use p_star instead of p_last in velocity and pressure model
useConservativePressureTerm = False # use < -pI, grad w>  instead of < grad p, w> in velocity update

useDensityASGS=True  # turn on/off Algebraic Subgrid Stabilization for density transport
useVelocityASGS=True # turn on/off Algebraic Subgrid Stabilization for velocity  transport

# choose initial condition format
useInitialConditions=int(0) # 0 = use Interpolation initial conditions
                            # 1 = use L2 Projection for (rho,u,v,p) and calculate (pi) from [u,v]
                            # 2 = use (u,v,p) Stokes Projection, (rho) L2 projection, calculate (pi) from [u,v]

# Spatial Discretization  he = he_coeff*2*Pi/150.0
he_coeff = 0.025
#he_coeff *= 2.0
#he_coeff *= 0.5
time_offset_coeff = 0.0  # offsets time by coeff*pi ie (t0 = 0 + coeff pi)
# setup time variables

g = 1.0
gvec = [0.0,-g]
d = 1.0 #reference length
rho_min = 1.0
chi = 0.95*rho_min
At = 0.5 #(rho_max - rho_min)/(rho_max+rho_min)
rho_max = (1.0+At)*rho_min / (1.0-At) # 1.5/0.5 = 3.0
Re = 1000.0
mu = (rho_min*d**(3.0/2.0)*g**(1.0/2.0))/Re
T = 2.5/sqrt(At)  # length of time interval
DT = 0.00125
#DT *= 2.0
#DT*= 0.5
class AnalyticSolutionConverter:
    """
    wrapper for function f(x) that satisfies proteus interface for analytical solutions
    """
    def __init__(self,fx,gradfx=None,T=None):
        self.exact_function = fx
        self.exact_grad_function = gradfx
        self.fixed_T=T
    def get_t(self,t):
        if self.fixed_T is not None:
            return self.fixed_T
        else:
            return t
    def uOfXT(self,x,t):
        return self.exact_function(x,self.get_t(t))
    def uOfX(self,x):
        return self.exact_function(x,0.0)
    def duOfXT(self,x,t):
        return self.exact_grad_function(x,self.get_t(t))
    def duOfX(self,x):
        return self.exact_grad_function(x,0.0)


# setup tnList
if globalBDFTimeOrder == 1 or not useScaleUpTimeStepsBDF2:
    nFrames = int(T/DT) + 1
    tnList =  [i*DT for i in range(nFrames) ]
elif globalBDFTimeOrder == 2:
    # spin up to DT from DT**2  doubling each time until we have DT then continue
    DTstep = DT*DT
    Tval = 0.0
    tnList = [0.0]
    while DTstep < DT:
        Tval = Tval + DTstep
        tnList.append(Tval)
        DTstep *= 2.0

    remainingDTSteps = int(np.floor((T-tnList[-1])/DT))
    lastVal = tnList[-1]
    tnList[len(tnList):] =  [lastVal + (i+1)*DT for i in range(remainingDTSteps) ]
    if tnList[-1] < T :
        tnList.append(T)
    nFrames = len(tnList)
else:
    assert False, \
      "Error: BDF time order = % is not supported.  It must be in {1,2}." % globalBDFTimeOrder

# for outputting in file names without '.' and the like
decimal_length = 6
DT_string = "{:1.{dec_len}f}".format(DT, dec_len=decimal_length)[2:]

if not useVelocityComponents:
    DT_string+='_velpp'

# solutions

#use numpy for evaluations
# from IPython.display import  display
# from sympy.interactive import printing
# printing.init_printing(use_latex=True)

# Domain and mesh
boundaries=['left','right','bottom','top','front','back','fixed']
    # fixed is for if we need to fix a single dof for solving a poisson problem
    # with natural boundary conditions, where we need to fix a sngle dof to pin
    # them down and then adjust to have average 0 in postStep()
boundaryTags=dict([(key,i+1) for (i,key) in enumerate(boundaries)])
domain = Domain.PlanarStraightLineGraphDomain(vertices=[[0.0,-2.0*d],
                                                        [0.0, 2.0*d],
                                                        [0.5*d, 2.0*d],
                                                        [0.5*d,-2.0*d]],
                                              vertexFlags=[boundaryTags['bottom'],
                                                           boundaryTags['top'],
                                                           boundaryTags['top'],
                                                           boundaryTags['bottom']],
                                              segments=[[0,1],
                                                        [1,2],
                                                        [2,3],
                                                        [3,0]],
                                              segmentFlags=[boundaryTags['left'],
                                                            boundaryTags['top'],
                                                            boundaryTags['right'],
                                                            boundaryTags['bottom']])
domain.boundaryTags = boundaryTags
domain.writePoly("mesh")
triangleOptions="VApq33Dena%8.8f" % ((he_coeff**2)/2.0,)
logEvent("""Mesh generated using: triangle -%s %s"""  % (triangleOptions,domain.polyfile+".poly"))

def rho_init(x,y):
    from math import cos,pi,tanh
    eta = -0.1*d*cos(2.0*pi*x/d)
    rho = rho_min*(2.0+tanh((y - eta)/(0.01*d)))
    return max(rho_min,min(rho_max,rho))

rho_init_v = np.vectorize(rho_init)

def rhotrue(x,t):
    return rho_init_v(x[...,0],x[...,1])

def utrue(x,t):
    return np.zeros(x[...,0].shape)

def vtrue(x,t):
    return np.zeros(x[...,0].shape)

def gradutrue(x,t):
    return np.zeros(x.shape)

def gradvtrue(x,t):
    return np.zeros(x.shape)

def pitrue(x,t): # pressure increment
    return np.zeros(x[...,0].shape)

def ptrue(x,t):
    return np.zeros(x[...,0].shape)

def gradptrue(x,t):
    return np.zeros(x.shape)

# numerical tolerances
density_atol_res = 1.0e-4
velocity_atol_res = 1.0e-4
phi_atol_res = 1.0e-4
pressure_atol_res = 1.0e-4


parallelPartitioningType = proteus.MeshTools.MeshParallelPartitioningTypes.node
nLayersOfOverlapForParallel = 0
