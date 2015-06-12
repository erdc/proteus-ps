from math import *
import proteus.MeshTools
from proteus import Domain
from proteus.default_n import *
from proteus.Profiling import logEvent
import numpy as np


#  Discretization
nd = 2

# Numerics
quad_degree = 5  # exact for polynomials of this degree


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
mu = 1.0

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

# display(f1s)
# display(f2s)
# print "f1(x,y,t) = ", f1s
# print "f2(x,y,t) = ", f2s

# use lambdify to convert from sympy to python expressions
pl = lambdify((xs, ys, ts), ps, "numpy")
ul = lambdify((xs, ys, ts), us, "numpy")
vl = lambdify((xs, ys, ts), vs, "numpy")
rhol = lambdify((xs, ys, ts), rhos, "numpy")
f1l = lambdify((xs, ys, ts), f1s, "numpy")
f2l = lambdify((xs, ys, ts), f2s, "numpy")

# convert python expressions to the format we need for multidimensional x values
def ptrue(x,t):
    return pl(x[...,0],x[...,1],t)

def utrue(x,t):
    return ul(x[...,0],x[...,1],t)

def vtrue(x,t):
    return vl(x[...,0],x[...,1],t)

def rhotrue(x,t):
    return rhol(x[...,0],x[...,1],t)

def f1true(x,t):
    return f1l(x[...,0],x[...,1],t)

def f2true(x,t):
    return f2l(x[...,0],x[...,1],t)

def velocityFunction(x,t):
    return np.vstack((utrue(x,t)[...,np.newaxis].transpose(),
                      vtrue(x,t)[...,np.newaxis].transpose())
                    ).transpose()


# Domain and mesh

unitCircle = True
if unitCircle:
    from math import pi, ceil, cos, sin

    # modify these for changing circular domain location and size
    radius = 1.0
    center_x = 0.0
    center_y = 0.0
    he = 2.0*pi/150.0  # h size for edges of circle

    # no need to modify past here
    nvertices = nsegments = int(ceil(2.0*pi/he))
    dtheta = 2.0*pi/float(nsegments)
    vertices= []
    vertexFlags = []
    segments = []
    segmentFlags = []

    # boundary tags and dictionary
    boundaries=['left','right','bottom','top','front','back']
    boundaryTags=dict([(key,i+1) for (i,key) in enumerate(boundaries)])

    # set domain with top and bottom
    for i in range(nsegments):
        theta = pi/2.0 - i*dtheta
        vertices.append([center_x+radius*cos(theta),center_y+radius*sin(theta)])
        if i in [nvertices-1,0,1]:
            vertexFlags.append(boundaryTags['top'])
        else:
            vertexFlags.append(boundaryTags['bottom'])
        segments.append([i,(i+1)%nvertices])
        if i in [nsegments-1,0]:
            segmentFlags.append(boundaryTags['top'])
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
ns_nl_atol_res = max(1.0e-8,0.01*he**2)

# Time stepping for output
T=10.0
nFrames = 161#41
dt = T/(nFrames-1)
tnList = [i*dt for i in range(nFrames)]

# actual time step size
DT = 0.005

