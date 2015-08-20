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
useRotationalModel = False #  Standard vs Rotational models in pressure update
useStabilityTerms = True  # stability terms in density and velocity models
useNonlinearAdvection = False # switches between extrapolated and fully nonlinear advection in velocity model
useNumericalFluxEbqe = True # ebqe history manipulation use ebqe or numericalFlux.ebqe which is exact
useDirichletPressureBC = False  # Dirichlet bc pressure or zeroMean pressure increment
useDirichletPressureIncrementBC = False  # Dirichlet bc pressure or zeroMean pressure increment
useNoFluxPressureIncrementBC = True
useVelocityComponents = True  # False uses post processed velocity,
useScaleUpTimeStepsBDF2 = False  # Time steps = [dt^2, 2dt^2, 4dt^2, ... dt, ... , dt, T-tLast]
setFirstTimeStepValues = False # interpolate the first step as well as the 0th step from exact solutions
usePressureExtrapolations = False # use p_star instead of p_last in velocity and pressure model
useConservativePressureTerm = False # use < -pI, grad w>  instead of < grad p, w> in velocity update

# Spatial Discretization  he = he_coeff*2*Pi/150.0
he_coeff = 0.75 # default to match Guermond paper: 0.75

# setup time variables
T = 2.0
DT = 0.1  # target time step size
#DT *= 0.5
#DT *= 0.5
#DT *= 0.5
# setup tnList
if globalBDFTimeOrder == 1 or not useScaleUpTimeStepsBDF2:
    nFrames = int(T/DT) + 1
    tnList =  [ i*DT for i in range(nFrames) ]
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


# solutions

#use numpy for evaluations
# from IPython.display import  display
# from sympy.interactive import printing
# printing.init_printing(use_latex=True)

# Create the manufactured solution and run through sympy
# to create the forcing function and solutions etc
#
# Import specific sympy functions to avoid overloading
# numpy etc functions
from sympy.utilities.lambdify import lambdify
from sympy import (symbols,
                   simplify,
                   diff)
from sympy.functions import (sin as sy_sin,
                             cos as sy_cos,
                             atan2 as sy_atan2,
                             sqrt as sy_sqrt)
from sympy import pi as sy_pi

# use xs and ts to represent symbolic x and t
xs,ys,ts = symbols('x y t')

# viscosity coefficient
mu = 1.0 # the viscosity coefficient
chi = 1.0  # 1.0 is the minimal value of rho density.

# Given solution: (Modify here and if needed add more sympy.functions above with
#                  notation sy_* to distinguish as symbolic functions)
rs = sy_sqrt(xs*xs + ys*ys)
thetas = sy_atan2(ys,xs)
rhos = 2 + rs*sy_cos(thetas-sy_sin(ts))


# rhos = 2 + rs*sy_cos(thetas-sy_sin(ts))
ps = sy_sin(xs)*sy_sin(ys)*sy_sin(ts)
us = -ys*sy_cos(ts)
vs = xs*sy_cos(ts)

# manufacture the source terms:

f1s = simplify((rhos*(diff(us,ts) + us*diff(us,xs) + vs*diff(us,ys)) + diff(ps,xs) - diff(mu*us,xs,xs) - diff(mu*us,ys,ys)))
f2s = simplify((rhos*(diff(vs,ts) + us*diff(vs,xs) + vs*diff(vs,ys)) + diff(ps,ys) - diff(mu*vs,xs,xs) - diff(mu*vs,ys,ys)))

# use lambdify to convert from sympy to python expressions
pl = lambdify((xs, ys, ts), ps, "numpy")
ul = lambdify((xs, ys, ts), us, "numpy")
vl = lambdify((xs, ys, ts), vs, "numpy")
rhol = lambdify((xs, ys, ts), rhos, "numpy")
f1l = lambdify((xs, ys, ts), f1s, "numpy")
f2l = lambdify((xs, ys, ts), f2s, "numpy")

drhol_dx = lambdify((xs, ys, ts), simplify(diff(rhos,xs)), "numpy")
drhol_dy = lambdify((xs, ys, ts), simplify(diff(rhos,ys)), "numpy")
dul_dx = lambdify((xs, ys, ts), simplify(diff(us,xs)), "numpy")
dul_dy = lambdify((xs, ys, ts), simplify(diff(us,ys)), "numpy")
dvl_dx = lambdify((xs, ys, ts), simplify(diff(vs,xs)), "numpy")
dvl_dy = lambdify((xs, ys, ts), simplify(diff(vs,ys)), "numpy")
dpl_dx = lambdify((xs, ys, ts), simplify(diff(ps,xs)), "numpy")
dpl_dy = lambdify((xs, ys, ts), simplify(diff(ps,ys)), "numpy")

# convert python expressions to the format we need for multidimensional x values
def rhotrue(x,t):
    return rhol(x[...,0],x[...,1],t)

def utrue(x,t):
    return ul(x[...,0],x[...,1],t)

def vtrue(x,t):
    return vl(x[...,0],x[...,1],t)

def pitrue(x,t): # pressure increment
    return np.zeros(x[...,0].shape)

def ptrue(x,t):
    return pl(x[...,0],x[...,1],t)

def f1true(x,t):
    return f1l(x[...,0],x[...,1],t)

def f2true(x,t):
    return f2l(x[...,0],x[...,1],t)

def velocityFunction(x,t):
    return np.vstack((utrue(x,t)[...,np.newaxis].transpose(),
                      vtrue(x,t)[...,np.newaxis].transpose())
                    ).transpose()

def velocityFunctionLocal(x,t):
    return np.array([utrue(x,t),vtrue(x,t)])


# analytic derivatives
def drhodxtrue(x,t):
    return drhol_dx(x[...,0],x[...,1],t)
def drhodytrue(x,t):
    return drhol_dy(x[...,0],x[...,1],t)

def dudxtrue(x,t):
    return dul_dx(x[...,0],x[...,1],t)
def dudytrue(x,t):
    return dul_dy(x[...,0],x[...,1],t)

def dvdxtrue(x,t):
    return dvl_dx(x[...,0],x[...,1],t)
def dvdytrue(x,t):
    return dvl_dy(x[...,0],x[...,1],t)

def dpdxtrue(x,t):
    return dpl_dx(x[...,0],x[...,1],t)
def dpdytrue(x,t):
    return dpl_dy(x[...,0],x[...,1],t)

def divVelocityFunction(x,t):
    return dudxtrue(x,t) + dvdytrue(x,t)

# analytic gradients
def gradrhotrue(x,t):
    return np.vstack((drhodxtrue(x,t)[...,np.newaxis].transpose(),
                      drhodytrue(x,t)[...,np.newaxis].transpose())
                      ).transpose()

# These velocity gradients get plugged into AnalyticSolutionConverter class
# below which makes them behve properly (give the right shape)
def gradutrue(x,t):
    return np.array([dudxtrue(x,t), dudytrue(x,t)])

def gradvtrue(x,t):
    return np.array([dvdxtrue(x,t), dvdytrue(x,t)])

def gradptrue(x,t):
    return np.vstack((dpdxtrue(x,t)[...,np.newaxis].transpose(),
                      dpdytrue(x,t)[...,np.newaxis].transpose())
                      ).transpose()

def gradpitrue(x,t): # pressure increment
    return np.vstack((np.zeros(x[...,0].shape)[...,np.newaxis].transpose(),
                      np.zeros(x[...,0].shape)[...,np.newaxis].transpose())
                      ).transpose()

class AnalyticSolutionConverter:
    """
    wrapper for function f(x) that satisfies proteus interface for analytical solutions
    """
    def __init__(self,fx,gradfx=None):
        self.exact_function = fx
        self.exact_grad_function = gradfx

    def uOfXT(self,x,t):
        return self.exact_function(x,t)
    def uOfX(self,x):
        return self.exact_function(x)
    def duOfXT(self,x,t):
        return self.exact_grad_function(x,t)
    def duOfX(self,x):
        return self.exact_grad_function(x)



# Domain and mesh
unitCircle = True
if unitCircle:
    from math import pi, ceil, cos, sin

    # modify these for changing circular domain location and size
    radius = 1.0
    center_x = 0.0
    center_y = 0.0

    he = he_coeff*2.0*pi/150.0  # h size for edges of circle

    # no need to modify past here
    nvertices = nsegments = int(ceil(2.0*pi/he))
    dtheta = 2.0*pi/float(nsegments)
    vertices= []
    vertexFlags = []
    segments = []
    segmentFlags = []

    # boundary tags and dictionary
    boundaries=['left','right','bottom','top','front','back','fixed']
    # fixed is for if we need to fix a single dof for solving a poisson problem
    # with natural boundary conditions, where we need to fix a sngle dof to pin
    # them down and then adjust to have average 0 in postStep()
    boundaryTags=dict([(key,i+1) for (i,key) in enumerate(boundaries)])

    # set domain with top and bottom
    for i in range(nsegments):
        theta = pi/2.0 - i*dtheta
        vertices.append([center_x+radius*cos(theta),center_y+radius*sin(theta)])
        if i in [nvertices-1,0,1]:
            vertexFlags.append(boundaryTags['top'])
        elif i == 2:
            vertexFlags.append(boundaryTags['fixed'])
        else:
            vertexFlags.append(boundaryTags['bottom'])

        segments.append([i,(i+1)%nvertices])
        if i in [nsegments-1,0]:
            segmentFlags.append(boundaryTags['top'])
        elif i == 1:
            segmentFlags.append(boundaryTags['fixed'])
        else:
            segmentFlags.append(boundaryTags['bottom'])

    domain = Domain.PlanarStraightLineGraphDomain(vertices=vertices,
                                                  vertexFlags=vertexFlags,
                                                  segments=segments,
                                                  segmentFlags=segmentFlags)
    #go ahead and add a boundary tags member
    domain.boundaryTags = boundaryTags
    domain.writePoly("mesh")

    #
    #finished setting up circular domain
    #
    triangleOptions="VApq30Dena%8.8f" % ((he**2)/2.0,)

    logEvent("""Mesh generated using: triangle -%s %s"""  % (triangleOptions,domain.polyfile+".poly"))


# numerical tolerances
density_atol_res = 1.0e-6
velocity_atol_res = 1.0e-6
phi_atol_res = 1.0e-6
pressure_atol_res = 1.0e-6


parallelPartitioningType = proteus.MeshTools.MeshParallelPartitioningTypes.node
nLayersOfOverlapForParallel = 0





# Time stepping for output
# T=10.0
# DT = 0.1
# nFrames = 51
# dt = T/(nFrames-1)
# tnList = [0, DT] + [ i*dt for i in range(1,nFrames) ]

# tnList =  [ i*dt for i in range(nFrames) ]
