"""
This module contains the coefficients classes for various NavierStokes equation definitions.

In particular this will contain classes for density, momentum, pressure increment, and pressure updates
according to the variable density incompressible navier stokes fractional time stepping schemes
as described in Guermand and Salgado 2011.
"""

from proteus.iproteus import TransportCoefficients
import numpy as np
from copy import deepcopy, copy # for cacheing _last values of variables

class DensityTransport2D(TransportCoefficients.TC_base):
    r"""
    The coefficients for conservative mass transport

    Conservation of mass is given by

    .. math::

       \frac{\partial\rho}{\partial t}+\nabla\cdot\left(\rho\mathbf{v}\right)-rho/2*\nabla\cdot\mathbf{v}=0
    """
    def __init__(self,velocityModelIndex=-1,
                 velocityFunction=None,
                 divVelocityFunction=None,
                 useVelocityComponents=True,
                 useStabilityTerms=False):
        """Construct a coefficients object

        :param velocityModelIndex: The index into the proteus model list

        :param velocityFunction: A function taking as input an array of spatial
        locations :math: `x`, time :math: `t`, and velocity :math: `v`, setting
        the velocity parameter as a side effect.
        
        :param divVelocityFunction: A function taking as input an array of spatial
        locations :math: `x`, time :math: `t`, and velocity :math: `v`, setting
        the divVelocity parameter as a side effect.
        
        :param useStabilityTerms: A boolean switch to include the stabilizing
        terms in the model as a reaction term or not

        TODO: decide how the parameters interact. I think velocityFunction
        should override the velocity from another model

        """
        TransportCoefficients.TC_base.__init__(self,
                                               nc = 1,
                                               variableNames = ['rho'],
                                               mass = {0:{0:'linear'}},
                                               advection = {0:{0:'linear'}},
                                               reaction = {0:{0:'linear'}} if useStabilityTerms else {} ) # for the stability term
        self.velocityModelIndex = velocityModelIndex
        self.velocityFunction = velocityFunction
        self.divVelocityFunction = divVelocityFunction
        self.c_u = {}
        self.c_v = {}
        self.c_velocity = {}
        self.useVelocityComponents = useVelocityComponents
        self.useStabilityTerms = useStabilityTerms

    def attachModels(self,modelList):
        """
        Attach the model for velocity
        
        Note that we must use ('velocity',2) since we want the velocity post processor to associate itself
        with the divergence free property which is our third (2) equation.  Then ('velocity',2) will extract
        the velocity components from the post processor and pass those along.  There are at least three possible 
        post processors for divergence free equation of velocity set in mom_n.py:
        
        conservativeFlux = {2:'point-eval'}  - will return computed velocities without change 
                                               (since there is no diffusion in eqn (2) )
        conservativeFlux = {2:'pwl-bdm'}     - will return velocities projected onto the bdm space (CG 
                                               Taylor-Hood enriched with DG pw linears on each element)
        conservativeFlux = {2:'pwl-bdm-opt'} - same as pwl-bdm but optimized in a special way to be more 
                                               effective.  any additional comments ?
        
        Notice that again we are applying the conservativeFlux post processing to the divergence free equation (2).
        """
        if not self.useVelocityComponents and self.velocityModelIndex >= 0:
            assert self.velocityModelIndex < len(modelList), \
                "velocity model index out of  range 0," + repr(len(modelList))
            self.velocityModel = modelList[self.velocityModelIndex]
            if ('velocity',2) in self.velocityModel.q:
                vel = self.velocityModel.q[('velocity',2)]
                self.c_velocity[vel.shape] = vel
                if self.useStabilityTerms:
                    gradu = self.velocityModel.q[('grad(u)',0)]
                    gradv = self.velocityModel.q[('grad(u)',1)]
                    self.c_u[gradu.shape] = gradu
                    self.c_v[gradv.shape] = gradv
            if ('velocity',2) in self.velocityModel.ebq:
                vel = self.velocityModel.ebq[('velocity',2)]
                self.c_velocity[vel.shape] = vel
                if self.useStabilityTerms:
                    gradu = self.velocityModel.ebq[('grad(u)',0)]
                    gradv = self.velocityModel.ebq[('grad(u)',1)]
                    self.c_u[gradu.shape] = gradu
                    self.c_v[gradv.shape] = gradv
            if ('velocity',2) in self.velocityModel.ebqe:
                vel = self.velocityModel.ebqe[('velocity',2)]
                self.c_velocity[vel.shape] = vel
                if self.useStabilityTerms:
                    gradu = self.velocityModel.ebqe[('grad(u)',0)]
                    gradv = self.velocityModel.ebqe[('grad(u)',1)]
                    self.c_u[gradu.shape] = gradu
                    self.c_v[gradv.shape] = gradv
            if ('velocity',2) in self.velocityModel.ebq_global:
                vel = self.velocityModel.ebq_global[('velocity',2)]
                self.c_velocity[vel.shape] = vel
                if self.useStabilityTerms:
                    gradu = self.velocityModel.ebq_global[('grad(u)',0)]
                    gradv = self.velocityModel.ebq_global[('grad(u)',1)]
                    self.c_u[gradu.shape] = gradu
                    self.c_v[gradv.shape] = gradv
        elif self.useVelocityComponents and self.velocityModelIndex >= 0:
            assert self.velocityModelIndex < len(modelList), \
                "velocity model index out of  range 0," + repr(len(modelList))
            self.velocityModel = modelList[self.velocityModelIndex]
            if ('u',0) in self.velocityModel.q:
                u = self.velocityModel.q[('u',0)]
                v = self.velocityModel.q[('u',1)]
                self.c_u[u.shape] = u
                self.c_v[v.shape] = v
                if self.useStabilityTerms:
                    gradu = self.velocityModel.q[('grad(u)',0)]
                    gradv = self.velocityModel.q[('grad(u)',1)]
                    self.c_u[gradu.shape] = gradu
                    self.c_v[gradv.shape] = gradv
            if ('u',0) in self.velocityModel.ebq:
                u = self.velocityModel.ebq[('u',0)]
                v = self.velocityModel.ebq[('u',1)]
                self.c_u[u.shape] = u
                self.c_v[v.shape] = v
                if self.useStabilityTerms:
                    gradu = self.velocityModel.ebq[('grad(u)',0)]
                    gradv = self.velocityModel.ebq[('grad(u)',1)]
                    self.c_u[gradu.shape] = gradu
                    self.c_v[gradv.shape] = gradv
            if ('u',0) in self.velocityModel.ebqe:
                u = self.velocityModel.ebqe[('u',0)]
                v = self.velocityModel.ebqe[('u',1)]
                self.c_u[u.shape] = u
                self.c_v[v.shape] = v
                if self.useStabilityTerms:
                    gradu = self.velocityModel.ebqe[('grad(u)',0)]
                    gradv = self.velocityModel.ebqe[('grad(u)',1)]
                    self.c_u[gradu.shape] = gradu
                    self.c_v[gradv.shape] = gradv
            if ('u',0) in self.velocityModel.ebq_global:
                u = self.velocityModel.ebq_global[('u',0)]
                v = self.velocityModel.ebq_global[('u',1)]
                self.c_u[u.shape] = u
                self.c_v[v.shape] = v
                if self.useStabilityTerms:
                    gradu = self.velocityModel.ebq_global[('grad(u)',0)]
                    gradv = self.velocityModel.ebq_global[('grad(u)',1)]
                    self.c_u[gradu.shape] = gradu
                    self.c_v[gradv.shape] = gradv
    def preStep(self,t,firstStep=False):
        """
        Give the TC object an opportunity to modify itself before the time step.
        """
        for ci in range(self.nc):
            self.model.q[('u_last',ci)] = deepcopy(self.model.q[('u',ci)])
            # self.model.ebq[('u_last',ci)] = deepcopy(self.model.ebq[('u',ci)])
            self.model.ebqe[('u_last',ci)] = deepcopy(self.model.ebqe[('u',ci)])
            # self.model.ebq_global[('u_last',ci)] = deepcopy(self.model.ebq_global[('u',ci)])
            
            # self.model.q[('grad(u)_last',ci)] = deepcopy(self.model.q[('grad(u)',ci)])
            # self.model.ebq[('grad(u)_last',ci)] = deepcopy(self.model.ebq[('grad(u)',ci)])
            # self.model.ebqe[('grad(u)_last',ci)] = deepcopy(self.model.ebqe[('grad(u)',ci)])
            # self.model.ebq_global[('grad(u)_last',ci)] = deepcopy(self.model.ebq_global[('grad(u)',ci)])
        copyInstructions = {}
        return copyInstructions
    def evaluate(self,t,c):
        """
        Evaluate the coefficients after getting the specified velocity
        """
        if self.velocityFunction != None:
            u = self.velocityFunction(c['x'],t)[...,0]
            v = self.velocityFunction(c['x'],t)[...,1]
            if self.useStabilityTerms:
                div_vel = self.divVelocityFunction(c['x'],t)
        elif self.useVelocityComponents:
            u = self.c_u[c[('m',0)].shape]
            v = self.c_v[c[('m',0)].shape]
            if self.useStabilityTerms:
                div_vel = self.c_u[c[('f',0)].shape][0] + self.c_u[c[('f',0)].shape][1]
        else:
            u = self.c_velocity[c[('f',0)].shape][...,0]
            v = self.c_velocity[c[('f',0)].shape][...,1]
            if self.useStabilityTerms:
                div_vel = self.c_u[c[('f',0)].shape][0] + self.c_u[c[('f',0)].shape][1]
        c[('m',0)][:] = c[('u',0)]
        c[('dm',0,0)][:] = 1.0
        c[('f',0)][...,0] = c[('u',0)]*u
        c[('f',0)][...,1] = c[('u',0)]*v
        c[('df',0,0)][...,0] = u
        c[('df',0,0)][...,1] = v
        if useStabilityTerms:
            c[('r',0)][:]     = -0.5*c[('u',0)]*div_vel
            c[('dr',0,0)][:]   = -0.5*div_vel




class VelocityTransport2D(TransportCoefficients.TC_base):
    r"""
    The coefficients of the 2D Navier Stokes momentum equation with variable density.  This coefficient class
    will only represent the momentum equation but not the incompressibility equation and not the conservation of mass.

    For completeness we give them all and note that this class only represents the 2nd and 3rd equation.
    .. math::
       :nowrap:

       \begin{equation}
       \begin{cases}
       \rho_t + \nabla\cdot(\rho\vb) = 0,&\\
       \rho\left(\frac{\partial \vb}{\partial t} + \vb\cdot\nabla\vb\right) +  \nabla p  - \nabla \cdot \left(\mu \nabla\vb\right) = \fb(x,t),&\\
       \nabla \cdot \vb  = 0,&
       \end{cases}
       \end{equation}

    where :math:`\rho>0` is the density, :math:`\mathbf{v}` is the velocity field,  :math:`p` is the pressure and :math:`\mu` is the dynamic
    viscosity which could depend on density :math:`\rho`.

    We solve this equation on a 2D disk :math:`\Omega=\{(x,y) \:|\: x^2+y^2<1\}`

    :param densityModelIndex: The index into the proteus model list

    :param densityFunction: A function taking as input an array of spatial
    locations :math: `x`, time :math: `t`, and density :math: `\rho`, setting
    the density parameter as a side effect.

    TODO: decide how the parameters interact. I think densityFunction
    should override the density from another model

    """
    def __init__(self,
                 f1ofx=None,
                 f2ofx=None,
                 mu=1.0,
                 densityModelIndex=-1,
                 densityFunction=None,
                 pressureModelIndex=-1,
                 pressureGradFunction=None,
                 pressureIncrementModelIndex=-1,
                 pressureIncrementGradFunction=None,
                 useStabilityTerms=False):

        sdInfo  = {(0,0):(np.array([0,1,2],dtype='i'),  # sparse diffusion uses diagonal element for diffusion coefficient
                          np.array([0,1],dtype='i')),
                   (1,1):(np.array([0,1,2],dtype='i'),
                          np.array([0,1],dtype='i'))}
        dim=2; # dimension of space
        xi=0; yi=1; # indices for first component or second component of dimension
        eu=0; ev=1; # equation numbers  momentum u, momentum v, 
        ui=0; vi=1; # variable name ordering

        TransportCoefficients.TC_base.__init__(self,
                         nc=dim, #number of components  u, v
                         variableNames=['u','v'], # defines variable reference order [0, 1]
                         mass = {eu:{ui:'linear'}, # du/dt
                                 ev:{vi:'linear'}}, # dv/dt
                         hamiltonian = {eu:{ui:'linear'}, #  rho*(u u_x + v u_y)   convection term
                                        ev:{vi:'linear'}}, # rho*(u v_x + v v_y)   convection term
                         diffusion = {eu:{ui:{ui:'constant'}},  # - \mu * \grad u
                                      ev:{vi:{vi:'constant'}}}, # - \mu * \grad v
                         potential = {eu:{ui:'u'},
                                      ev:{vi:'u'}}, # define the potential for the diffusion term to be the solution itself
                         reaction  = {eu:{ui:'constant'}, # -f1(x) + d/dx p^* + (stability terms) * u
                                      ev:{vi:'constant'}}, # -f2(x) + d/dy p^* + (stability terms) * v
                         sparseDiffusionTensors=sdInfo,
                         useSparseDiffusion = True),
        self.vectorComponents=[ui,vi]
        self.f1ofx=f1ofx
        self.f2ofx=f2ofx
        self.mu=mu
        self.densityModelIndex = densityModelIndex
        self.densityFunction = densityFunction
        self.pressureModelIndex = pressureModelIndex
        self.pressureGradFunction = pressureGradFunction
        self.pressureIncrementModelIndex = pressureIncrementModelIndex
        self.pressureIncrementGradFunction = pressureIncrementGradFunction
        self.useStabilityTerms = useStabilityTerms
        self.c_rho = {} # density
        self.c_rho_last = {} # density of cached values
        self.c_p = {}  # pressure
        self.c_phi = {} # pressure increment phi

    def attachModels(self,modelList):
        """
        Attach the models for density, pressure increment and pressure
        """
        if self.densityModelIndex >= 0:
            assert self.densityModelIndex < len(modelList), \
                "density model index out of range 0," + repr(len(modelList))
            self.densityModel = modelList[self.densityModelIndex]
            if ('u',0) in self.densityModel.q:
                rho = self.densityModel.q[('u',0)]
                self.c_rho[rho.shape] = rho
                if self.useStabilityTerms:
                    rho_last = self.densityModel.q[('u_last',0)]
                    grad_rho = self.densityModel.q[('grad(u)',0)]
                    self.c_rho_last[rho_last.shape] = rho_last
                    self.c_rho[grad_rho.shape] = grad_rho
            if ('u',0) in self.densityModel.ebq:
                rho = self.densityModel.ebq[('u',0)]
                self.c_rho[rho.shape] = rho
                if self.useStabilityTerms:
                    rho_last = self.densityModel.ebq[('u_last',0)]
                    grad_rho = self.densityModel.ebq[('grad(u)',0)]
                    self.c_rho_last[rho_last.shape] = rho_last
                    self.c_rho[grad_rho.shape] = grad_rho
            if ('u',0) in self.densityModel.ebqe:
                rho = self.densityModel.ebqe[('u',0)]
                self.c_rho[rho.shape] = rho
                if self.useStabilityTerms:
                    rho_last = self.densityModel.ebqe[('u_last',0)]
                    grad_rho = self.densityModel.ebqe[('grad(u)',0)]
                    self.c_rho_last[rho_last.shape] = rho_last
                    self.c_rho[grad_rho.shape] = grad_rho
            if ('u',0) in self.densityModel.ebq_global:
                rho = self.densityModel.ebq_global[('u',0)]
                self.c_rho[rho.shape] = rho
                if self.useStabilityTerms:
                    rho_last = self.densityModel.ebq_global[('u_last',0)]
                    grad_rho = self.densityModel.ebq_global[('grad(u)',0)]
                    self.c_rho_last[rho_last.shape] = rho_last
                    self.c_rho[grad_rho.shape] = grad_rho
        if self.pressureIncrementModelIndex >= 0:
            assert self.pressureIncrementModelIndex < len(modelList), \
                "pressure increment model index out of range 0," + repr(len(modelList))
            self.pressureIncrementModel = modelList[self.pressureIncrementModelIndex]
            if ('u',0) in self.pressureIncrementModel.q:
                grad_phi = self.pressureIncrementModel.q[('grad(u)',0)]
                self.c_phi[grad_phi.shape] = grad_phi
            if ('u',0) in self.pressureIncrementModel.ebq:
                grad_phi = self.pressureIncrementModel.ebq[('grad(u)',0)]
                self.c_phi[grad_phi.shape] = grad_phi
            if ('u',0) in self.pressureIncrementModel.ebqe:
                grad_phi = self.pressureIncrementModel.ebqe[('grad(u)',0)]
                self.c_phi[grad_phi.shape] = grad_phi
            if ('u',0) in self.pressureIncrementModel.ebq_global:
                grad_phi = self.pressureIncrementModel.ebq_global[('grad(u)',0)]
                self.c_phi[grad_phi.shape] = grad_phi
        if self.pressureModelIndex >= 0:
            assert self.pressureModelIndex < len(modelList), \
                "pressure model index out of range 0," + repr(len(modelList))
            self.pressureModel = modelList[self.pressureModelIndex]
            if ('u',0) in self.pressureModel.q:
                grad_p = self.pressureModel.q[('grad(u)',0)]
                self.c_p[grad_p.shape] = grad_p
            if ('u',0) in self.pressureModel.ebq:
                grad_p = self.pressureModel.ebq[('grad(u)',0)]
                self.c_p[grad_p.shape] = grad_p
            if ('u',0) in self.pressureModel.ebqe:
                grad_p = self.pressureModel.ebqe[('grad(u)',0)]
                self.c_p[grad_p.shape] = grad_p
            if ('u',0) in self.pressureModel.ebq_global:
                grad_p = self.pressureModel.ebq_global[('grad(u)',0)]
                self.c_p[grad_p.shape] = grad_p

    def preStep(self,t,firstStep=False):
        """
        Give the TC object an opportunity to modify itself before the time step.
        """
        for ci in range(self.nc):
            # deep copy so we have a cached value instead of pointer to current values
            self.model.q[('u_last',ci)] = deepcopy(self.model.q[('u',ci)]) 
            # self.model.ebq[('u_last',ci)] = deepcopy(self.model.ebq[('u',ci)])
            self.model.ebqe[('u_last',ci)] = deepcopy(self.model.ebqe[('u',ci)])
            # self.model.ebq_global[('u_last',ci)] = deepcopy(self.model.ebq_global[('u',ci)])
            
            self.model.q[('grad(u)_last',ci)] = deepcopy(self.model.q[('grad(u)',ci)])
            # self.model.ebq[('grad(u)_last',ci)] = deepcopy(self.model.ebq[('grad(u)',ci)])
            self.model.ebqe[('grad(u)_last',ci)] = deepcopy(self.model.ebqe[('grad(u)',ci)])
            # self.model.ebq_global[('grad(u)_last',ci)] = deepcopy(self.model.ebq_global[('grad(u)',ci)])
        copyInstructions = {}
        return copyInstructions
    def evaluate(self,t,c):
        """
        evaluate quadrature point values held in the dictionary c
        These are labelled according to the 'master equation.' For example,

        c[('a',0,0)] = diffusion coefficient for the 0th equation (first) with respect to the
                       0th potential (the solution itself)
                       The value at each quadrature point is a n_d x n_d tensor (n_d=number of space dimensions).
                       Usually the tensor values are stored as a flat array in compressed sparse row format to save space.
                       By default, we assume the tensor is full though.

        c[('r',0)]   = reaction term for the 0th equation. This is where we will put the source term
        """
        xi=0; yi=1; # indices for first component or second component of dimension
        eu=0; ev=1; # equation numbers  momentum u, momentum v, divergencefree
        ui=0; vi=1; # variable name ordering
        # current velocity and grad velocity
        u = c[('u',ui)]
        v = c[('u',vi)]
        grad_u = c[('grad(u)',ui)]
        grad_v = c[('grad(u)',vi)]
        
        # previous velocity and grad velocity
        u_last = c[('u_last',ui)] # figure out how to extract old solutons
        v_last = c[('u_last',vi)]
        grad_u_last = c[('grad(u)_last',ui)]
        grad_v_last = c[('grad(u)_last',vi)]
                
        # time management
        dt = self.timeIntegration.dt
        tLast = self.timeIntegration.tLast 
        
        # extract rho, rho_last and grad_rho
        if self.densityFunction != None:
            rho = self.densityFunction(c['x'],t)
            if self.useStabilityTerms:
                rho_last =  self.densityFunction(c['x'],tLast)
                grad_rho = self.gradDensityFunction(c['x'],t)
        else:#use mass shape as key since it is same shape as density
            rho = self.c_rho[c[('m',0)].shape]
            if self.useStabilityTerms:
                rho_last = self.c_rho_last[c[('m',0)].shape]
                grad_rho = self.c_rho[c[('f',0)].shape] # use flux shape since it is same shape as gradient
                
        if self.pressureFunction != None:
            grad_p = self.pressureGradFunction(c['x'],t)
        else:#use flux shape as key since it is same shape as gradient of pressure
            grad_p = self.c_p[c[('f',0)].shape]
                
        if self.pressureIncrementFunction != None:
            grad_phi = self.pressureIncrementGradFunction(c['x'],t)
        else:#use flux shape as key since it is same shape as gradient of pressure increment
            grad_phi = self.c_phi[c[('f',0)].shape]
        
        # gradient of pressure term
        grad_psharp = grad_p + grad_phi
        
        # solve for stability terms
        if self.useStabilityTerms:
            div_vel_last = grad_u_last[...,xi] + grad_v_last[...,yi]
            div_rho_vel = grad_rho[...,xi]*u + grad_rho[...,yi]*v + rho*div_vel_last
                
        #equation eu = 0  rho*(u_t + u ux + v uy ) + px + div(-mu grad(u)) - f1 = 0
        c[('m',eu)][:] = rho_last*u  # d/dt ( rho_last * u) = d/dt (m_0)
        c[('dm',eu,ui)][:] = rho_last  # dm^0_du
        c[('r',eu)][:] = -self.f1ofx(c['x'][:],t) + grad_psharp[...,xi]
        c[('dr',eu,ui)][:] = 0.0
        if self.useStabilityTerms:
            c[('r',eu)][:] += 0.5*((rho - rho_last)/dt + div_rho_vel)*u
            c[('dr',eu,ui)][:] += 0.5*((rho - rho_last)/dt + div_rho_vel)
        c[('H',eu)][:] = rho*(u_last*grad_u[...,xi] + v_last*grad_u[...,yi])
        c[('dH',eu,ui)][...,xi] = rho*u_last #  dH d(u_x)
        c[('dH',eu,ui)][...,yi] = rho*v_last #  dH d(u_y)
        c[('a',eu,ui)][...,0] = self.mu # -mu*\grad v :   tensor  [ mu  0;  0  mu] ordered [0 1; 2 3]  in our
        c[('a',eu,ui)][...,1] = self.mu # -mu*\grad v :       new diagonal notation from sDInfo above is [0 .; . 1] -> [0; 1]
        c[('da',eu,ui,ui)][...,0] = 0.0 # -(da/d ui)_0   # could leave these off since it is 0
        c[('da',eu,ui,ui)][...,1] = 0.0 # -(da/d ui)_1   # could leave these off since it is 0

        # equation ev = 1  rho*(v_t + u vx + v vy ) + py + div(-mu grad(v)) - f2 = 0
        c[('m',ev)][:] = rho_last*v  # d/dt ( rho * v) = d/dt (m_1)
        c[('dm',ev,vi)][:] = rho_last  # dm^1_dv
        c[('r',ev)][:] = -self.f2ofx(c['x'][:],t) + grad_psharp[...,yi]
        c[('dr',ev,vi)][:] = 0.0
        if self.useStabilityTerms:
            c[('r',eu)][:] += 0.5*((rho - rho_last)/tau + div_rho_vel)*v
            c[('dr',eu,ui)][:] += 0.5*((rho - rho_last)/tau + div_rho_vel)
        c[('H',ev)][:] = rho*(u_last*grad_v[...,xi] + v_last*grad_v[...,yi])  # add rho term
        c[('dH',ev,vi)][...,xi] = rho*u_last #  dH d(v_x)
        c[('dH',ev,vi)][...,yi] = rho*v_last #  dH d(v_y)
        c[('a',ev,vi)][...,0] = self.mu # -mu*\grad v :   tensor  [ mu  0;  0  mu] ordered [0 1; 2 3]  in our
        c[('a',ev,vi)][...,1] = self.mu # -mu*\grad v :       new diagonal notation from sDInfo above is [0 .; . 1] -> [0; 1]
        c[('da',ev,vi,vi)][...,0] = 0.0 # -(da/d vi)_0   # could leave these off since it is 0
        c[('da',ev,vi,vi)][...,1] = 0.0 # -(da/d vi)_1   # could leave these off since it is 0




class PressureIncrement2D(TransportCoefficients.TC_base):
    r"""
    The coefficients for pressure increment solution

    Update is given by 

    .. math::

       -\Delta \phi^{k+1} + \chi/dt \nabla\cdot\ub^{k+1}=0
    """
    def __init__(self,
                 velocityModelIndex=-1,
                 velocityFunction=None,
                 useVelocityComponents=True,
                 densityModelIndex=-1,
                 chiValue=None):
        """Construct a coefficients object

        :param velocityModelIndex: The index into the proteus model list
        
        :param divVelocityFunction: A function taking as input an array of spatial
        locations :math: `x`, time :math: `t`, and velocity :math: `v`, setting
        the divVelocity parameter as a side effect.

        TODO: decide how the parameters interact. I think divVelocityFunction
        should override the velocity from another model

        """
        sdInfo  = {(0,0):(np.array([0,1,2],dtype='i'),  # sparse diffusion uses diagonal element for diffusion coefficient
                          np.array([0,1],dtype='i')),
                   (1,1):(np.array([0,1,2],dtype='i'),
                          np.array([0,1],dtype='i'))}
        
        TransportCoefficients.TC_base.__init__(self,
                                               nc = 1,
                                               variableNames = ['phi'],
                                               diffusion = {0:{0:{0:'constant'}}}, # -  \grad phi
                                               potential = {0:{0:'u'}}, # define the potential for the diffusion term to be the solution itself
                                               advection = {0:{0:'constant'}}, # div  (chi/dt velocity)
                                               sparseDiffusionTensors=sdInfo,
                                               useSparseDiffusion = True)
                                               # reaction = {0:{0:'linear'}} ) # chi / dt div velocity
        self.velocityModelIndex = velocityModelIndex
        self.velocityFunction = velocityFunction
        self.c_u = {}
        self.c_v = {}
        self.c_velocity = {}
        self.c_rho = {}
        self.useVelocityComponents = useVelocityComponents
        self.chiValue = chiValue

    def attachModels(self,modelList):
        """
        Attach the model for velocity
        
        Note that we must use ('velocity',2) since we want the velocity post processor to associate itself
        with the divergence free property which is our third (2) equation.  Then ('velocity',2) will extract
        the velocity components from the post processor and pass those along.  There are at least three possible 
        post processors for divergence free equation of velocity set in mom_n.py:
        
        conservativeFlux = {2:'point-eval'}  - will return computed velocities without change 
                                               (since there is no diffusion in eqn (2) )
        conservativeFlux = {2:'pwl-bdm'}     - will return velocities projected onto the bdm space (CG 
                                               Taylor-Hood enriched with DG pw linears on each element)
        conservativeFlux = {2:'pwl-bdm-opt'} - same as pwl-bdm but optimized in a special way to be more 
                                               effective.  any additional comments ?
        
        Notice that again we are applying the conservativeFlux post processing to the divergence free equation (2).
        """
        if not self.useVelocityComponents and self.velocityModelIndex >= 0:
            assert self.velocityModelIndex < len(modelList), \
                "velocity model index out of range 0," + repr(len(modelList))
            self.velocityModel = modelList[self.velocityModelIndex]
            if ('velocity',2) in self.velocityModel.q:
                vel = self.velocityModel.q[('velocity',2)]
                self.c_velocity[vel.shape] = vel
            if ('velocity',2) in self.velocityModel.ebq:
                vel = self.velocityModel.ebq[('velocity',2)]
                self.c_velocity[vel.shape] = vel
            if ('velocity',2) in self.velocityModel.ebqe:
                vel = self.velocityModel.ebqe[('velocity',2)]
                self.c_velocity[vel.shape] = vel
            if ('velocity',2) in self.velocityModel.ebq_global:
                vel = self.velocityModel.ebq_global[('velocity',2)]
                self.c_velocity[vel.shape] = vel
        elif self.useVelocityComponents and self.velocityModelIndex >= 0:
            assert self.velocityModelIndex < len(modelList), \
                "velocity model index out of  range 0," + repr(len(modelList))
            self.velocityModel = modelList[self.velocityModelIndex]
            if ('u',0) in self.velocityModel.q:
                u = self.velocityModel.q[('u',0)]
                v = self.velocityModel.q[('u',1)]
                self.c_u[u.shape] = u
                self.c_v[v.shape] = v
            if ('u',0) in self.velocityModel.ebq:
                u = self.velocityModel.ebq[('u',0)]
                v = self.velocityModel.ebq[('u',1)]
                self.c_u[u.shape] = u
                self.c_v[v.shape] = v
            if ('u',0) in self.velocityModel.ebqe:
                u = self.velocityModel.ebqe[('u',0)]
                v = self.velocityModel.ebqe[('u',1)]
                self.c_u[u.shape] = u
                self.c_v[v.shape] = v
            if ('u',0) in self.velocityModel.ebq_global:
                u = self.velocityModel.ebq_global[('u',0)]
                v = self.velocityModel.ebq_global[('u',1)]
                self.c_u[u.shape] = u
                self.c_v[v.shape] = v
        if self.densityModelIndex >= 0:
            assert self.densityModelIndex < len(modelList), \
                "density model index out of range 0," + repr(len(modelList))
            self.densityModel = modelList[self.densityModelIndex]
            if ('u',0) in self.densityModel.q:
                rho = self.densityModel.q[('u',0)]
                self.c_rho[rho.shape] = rho
            if ('u',0) in self.densityModel.ebq:
                rho = self.densityModel.ebq[('u',0)]
                self.c_rho[rho.shape] = rho
            if ('u',0) in self.densityModel.ebqe:
                rho = self.densityModel.ebqe[('u',0)]
                self.c_rho[rho.shape] = rho
            if ('u',0) in self.densityModel.ebq_global:
                rho = self.densityModel.ebq_global[('u',0)]
                self.c_rho[rho.shape] = rho
    def preStep(self,t,firstStep=False):
        """
        Give the TC object an opportunity to modify itself before the time step.
        """
        for ci in range(self.nc):
            # self.model.q[('u_last',ci)] = deepcopy(self.model.q[('u',ci)])
            # self.model.ebq[('u_last',ci)] = deepcopy(self.model.ebq[('u',ci)])
            # self.model.ebqe[('u_last',ci)] = deepcopy(self.model.ebqe[('u',ci)])
            # self.model.ebq_global[('u_last',ci)] = deepcopy(self.model.ebq_global[('u',ci)])
            
            self.model.q[('grad(u)_last',ci)] = deepcopy(self.model.q[('grad(u)',ci)])
            # self.model.ebq[('grad(u)_last',ci)] = deepcopy(self.model.ebq[('grad(u)',ci)])
            self.model.ebqe[('grad(u)_last',ci)] = deepcopy(self.model.ebqe[('grad(u)',ci)])
            # self.model.ebq_global[('grad(u)_last',ci)] = deepcopy(self.model.ebq_global[('grad(u)',ci)])
        copyInstructions = {}
        return copyInstructions
    def evaluate(self,t,c):
        """
        Evaluate the coefficients after getting the specified velocity and density
        """        
        # time management
        dt = self.timeIntegration.dt
        
        # find minimal density value set it to be chi
        chi = 0.0
        if self.chiValue != None:
            chi = self.chiValue
        else:
            chi = np.min(self.c_rho[c[('m',0)].shape])
            
        # extract velocity components
        if self.velocityFunction != None:
            u = self.velocityFunction(c['x'],t)[...,0]
            v = self.velocityFunction(c['x'],t)[...,1]
        elif self.useVelocityComponents:
            u = self.c_u[c[('m',0)].shape]
            v = self.c_v[c[('m',0)].shape]
        else:
            u = self.c_velocity[c[('f',0)].shape][...,0]
            v = self.c_velocity[c[('f',0)].shape][...,1]
        
        # set coefficients
        c[('f',0)][...,0] = chi/dt*u
        c[('f',0)][...,1] = chi/dt*v
        c[('df',0,0)][...,0] = 0.0
        c[('df',0,0)][...,1] = 0.0
        c[('a',ev,vi)][...,0] = 1.0 # -mu*\grad v :   tensor  [ mu  0;  0  mu] ordered [0 1; 2 3]  in our
        c[('a',ev,vi)][...,1] = 1.0 # -mu*\grad v :       new diagonal notation from sDInfo above is [0 .; . 1] -> [0; 1]
        c[('da',ev,vi,vi)][...,0] = 0.0 # -(da/d vi)_0   # could leave these off since it is 0
        c[('da',ev,vi,vi)][...,1] = 0.0 # -(da/d vi)_1   # could leave these off since it is 0


class Pressure2D(TransportCoefficients.TC_base):
    r"""
    The coefficients for pressure solution

    Update is given by 

    .. math::

       p^{k+1} - p^{k} - phi^{k+1} + \mu \div u^{k+1} = 0
    """
    def __init__(self,
                 mu=1.0,
                 velocityModelIndex=-1,
                 velocityFunction=None,
                 useVelocityComponents=True,
                 pressureIncrementModelIndex=-1,
                 pressureIncrementFunction=None): 
        """Construct a coefficients object

        :param velocityModelIndex: The index into the proteus model list
        
        :param divVelocityFunction: A function taking as input an array of spatial
        locations :math: `x`, time :math: `t`, and velocity :math: `v`, setting
        the divVelocity parameter as a side effect.

        TODO: decide how the parameters interact. I think divVelocityFunction
        should override the velocity from another model

        """
        sdInfo  = {(0,0):(np.array([0,1,2],dtype='i'),  # sparse diffusion uses diagonal element for diffusion coefficient
                          np.array([0,1],dtype='i')),
                   (1,1):(np.array([0,1,2],dtype='i'),
                          np.array([0,1],dtype='i'))}
        
        TransportCoefficients.TC_base.__init__(self,
                                               nc = 1,
                                               variableNames = ['p'],
                                               reaction = {0:{0:'linear'}}, # p - p_last - phi
                                               advection = {0:{0:'constant'}})#, # div  (\mu velocity)
        self.mu = mu
        self.velocityModelIndex = velocityModelIndex
        self.velocityFunction = velocityFunction
        self.useVelocityComponents = useVelocityComponents
        self.pressureIncrementModelIndex = pressureIncrementModelIndex
        self.pressureIncrementFunction = pressureIncrementFunction
        self.c_u = {}
        self.c_v = {}
        self.c_velocity = {}
        self.c_phi = {}


    def attachModels(self,modelList):
        """
        Attach the model for velocity
        
        Note that we must use ('velocity',2) since we want the velocity post processor to associate itself
        with the divergence free property which is our third (2) equation.  Then ('velocity',2) will extract
        the velocity components from the post processor and pass those along.  There are at least three possible 
        post processors for divergence free equation of velocity set in mom_n.py:
        
        conservativeFlux = {2:'point-eval'}  - will return computed velocities without change 
                                               (since there is no diffusion in eqn (2) )
        conservativeFlux = {2:'pwl-bdm'}     - will return velocities projected onto the bdm space (CG 
                                               Taylor-Hood enriched with DG pw linears on each element)
        conservativeFlux = {2:'pwl-bdm-opt'} - same as pwl-bdm but optimized in a special way to be more 
                                               effective.  any additional comments ?
        
        Notice that again we are applying the conservativeFlux post processing to the divergence free equation (2).
        """
        if not self.useVelocityComponents and self.velocityModelIndex >= 0:
            assert self.velocityModelIndex < len(modelList), \
                "velocity model index out of range 0," + repr(len(modelList))
            self.velocityModel = modelList[self.velocityModelIndex]
            if ('velocity',2) in self.velocityModel.q:
                vel = self.velocityModel.q[('velocity',2)]
                self.c_velocity[vel.shape] = vel
            if ('velocity',2) in self.velocityModel.ebq:
                vel = self.velocityModel.ebq[('velocity',2)]
                self.c_velocity[vel.shape] = vel
            if ('velocity',2) in self.velocityModel.ebqe:
                vel = self.velocityModel.ebqe[('velocity',2)]
                self.c_velocity[vel.shape] = vel
            if ('velocity',2) in self.velocityModel.ebq_global:
                vel = self.velocityModel.ebq_global[('velocity',2)]
                self.c_velocity[vel.shape] = vel
        elif self.useVelocityComponents and self.velocityModelIndex >= 0:
            assert self.velocityModelIndex < len(modelList), \
                "velocity model index out of  range 0," + repr(len(modelList))
            self.velocityModel = modelList[self.velocityModelIndex]
            if ('u',0) in self.velocityModel.q:
                u = self.velocityModel.q[('u',0)]
                v = self.velocityModel.q[('u',1)]
                self.c_u[u.shape] = u
                self.c_v[v.shape] = v
            if ('u',0) in self.velocityModel.ebq:
                u = self.velocityModel.ebq[('u',0)]
                v = self.velocityModel.ebq[('u',1)]
                self.c_u[u.shape] = u
                self.c_v[v.shape] = v
            if ('u',0) in self.velocityModel.ebqe:
                u = self.velocityModel.ebqe[('u',0)]
                v = self.velocityModel.ebqe[('u',1)]
                self.c_u[u.shape] = u
                self.c_v[v.shape] = v
            if ('u',0) in self.velocityModel.ebq_global:
                u = self.velocityModel.ebq_global[('u',0)]
                v = self.velocityModel.ebq_global[('u',1)]
                self.c_u[u.shape] = u
                self.c_v[v.shape] = v
        if self.pressureIncrementModelIndex >= 0:
            assert self.pressureIncrementModelIndex < len(modelList), \
                "pressure increment model index out of range 0," + repr(len(modelList))
            self.pressureIncrementModel = modelList[self.pressureIncrementModelIndex]
            if ('u',0) in self.pressureIncrementModel.q:
                phi = self.pressureIncrementModel.q[('u',0)]
                self.c_phi[phi.shape] = phi
            if ('u',0) in self.pressureIncrementModel.ebq:
                phi = self.pressureIncrementModel.ebq[('u',0)]
                self.c_phi[phi.shape] = phi
            if ('u',0) in self.pressureIncrementModel.ebqe:
                phi = self.pressureIncrementModel.ebqe[('u',0)]
                self.c_phi[phi.shape] = phi
            if ('u',0) in self.pressureIncrementModel.ebq_global:
                phi = self.pressureIncrementModel.ebq_global[('u',0)]
                self.c_phi[phi.shape] = phi
    def preStep(self,t,firstStep=False):
        """
        Give the TC object an opportunity to modify itself before the time step.
        """
        for ci in range(self.nc):
            self.model.q[('u_last',ci)] = deepcopy(self.model.q[('u',ci)])
            # self.model.ebq[('u_last',ci)] = deepcopy(self.model.ebq[('u',ci)])
            self.model.ebqe[('u_last',ci)] = deepcopy(self.model.ebqe[('u',ci)])
            # self.model.ebq_global[('u_last',ci)] = deepcopy(self.model.ebq_global[('u',ci)])
            
            self.model.q[('grad(u)_last',ci)] = deepcopy(self.model.q[('grad(u)',ci)])
            # self.model.ebq[('grad(u)_last',ci)] = deepcopy(self.model.ebq[('grad(u)',ci)])
            self.model.ebqe[('grad(u)_last',ci)] = deepcopy(self.model.ebqe[('grad(u)',ci)])
            # self.model.ebq_global[('grad(u)_last',ci)] = deepcopy(self.model.ebq_global[('grad(u)',ci)])
        copyInstructions = {}
        return copyInstructions
    def evaluate(self,t,c):
        """
        Evaluate the coefficients after getting the specified velocity and density
        """        
        
        # current pressure
        p = c[('u',0)]
        
        # extract previous pressure value
        p_last = c[('u_last',0)] 
        
        # extract pressure increment
        if self.velocityFunction != None:
            phi = self.pressureIncrementFunction(c['x'],t)
        else:
            phi = self.c_phi[c[('m',0)].shape]
            
        # extract velocity components
        if self.velocityFunction != None:
            u = self.velocityFunction(c['x'],t)[...,0]
            v = self.velocityFunction(c['x'],t)[...,1]
        elif self.useVelocityComponents:
            u = self.c_u[c[('m',0)].shape]
            v = self.c_v[c[('m',0)].shape]
        else:
            u = self.c_velocity[c[('f',0)].shape][...,0]
            v = self.c_velocity[c[('f',0)].shape][...,1]
        
        # set coefficients
        c[('f',0)][...,0] = self.mu*u
        c[('f',0)][...,1] = self.mu*v
        c[('df',0,0)][...,0] = 0.0
        c[('df',0,0)][...,1] = 0.0
        c[('r',0)][:] = p - p_last - phi
        c[('dr',0,0)][:] = 1.0
