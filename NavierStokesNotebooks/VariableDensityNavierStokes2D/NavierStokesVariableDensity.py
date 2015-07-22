"""
This module contains the coefficients classes for various NavierStokes equation definitions.

In particular this will contain classes for density, momentum, pressure increment, and pressure updates
according to the variable density incompressible navier stokes fractional time stepping schemes
as described in Guermand and Salgado 2011.
"""

from proteus.iproteus import TransportCoefficients
import numpy as np
from copy import deepcopy, copy # for cacheing _last values of variables

from proteus.Profiling import logEvent as log


class DensityTransport2D(TransportCoefficients.TC_base):
    r"""
    The coefficients for conservative mass transport

    Conservation of mass is given by

    .. math::

       \frac{\partial\rho}{\partial t}+\nabla\cdot\left(\rho\mathbf{v}\right)-rho/2*\nabla\cdot\mathbf{v}=0
    """
    def __init__(self,
                 bdf=1,
                 currentModelIndex=0,
                 velocityModelIndex=-1,
                 velocityFunction=None,
                 divVelocityFunction=None,
                 useVelocityComponents=True,
                 chiValue=1.0,  # only needed for the scaling adjustment in case of post processed velocity
                 pressureIncrementModelIndex=-1,
                 pressureIncrementFunction=None,
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
                                               #hamiltonian = {0:{0:'linear'}},
                                               reaction = {0:{0:'linear'}} if useStabilityTerms else {} ) # for the stability term
        self.bdf=int(bdf)
        self.currentModelIndex = currentModelIndex
        self.velocityModelIndex = velocityModelIndex
        self.velocityFunction = velocityFunction
        self.divVelocityFunction = divVelocityFunction
        self.useVelocityComponents = useVelocityComponents
        self.chiValue = chiValue
        self.pressureIncrementModelIndex=pressureIncrementModelIndex
        self.pressureIncrementFunction=pressureIncrementFunction
        self.c_u = {}
        self.c_v = {}
        self.c_u_last = {}
        self.c_v_last = {}
        self.c_velocity = {}
        self.c_grad_u = {}
        self.c_grad_v = {}
        self.c_grad_u_last = {}
        self.c_grad_v_last = {}
        self.c_grad_velocity = {}
        self.useStabilityTerms = useStabilityTerms
        self.firstStep = True # manipulated in preStep()

    def attachModels(self,modelList):
        """
        Attach the model for velocity

        We are implementing the post processing in the pressureIncrement model which is
        essentially the divergence free velocity equation.  The velocity
        is then extracted from the pressureIncrement Model as ('velocity',0).  In order to
        get a physical velocity, we must then scale it by the constants dt/chi  since the pressure
        equation is  -div(  grad\phi - chi/dt [u v] ) = 0  so that the flux F has local integrals matching chi/dt [u v]
        and hopefully has locally divergence free velocity matching chi/dt [u v].  Thus the scaling by dt/chi
        to get physical velocity.

        In pressureincrement_n.py, the following could be set.  we recommend the 'pwl-bdm' as the best
        for this current situation:

        conservativeFlux = {0:'point-eval'}  - will return computed velocities with diffusive flux
                                               projected and evaluated to match conservation law
        conservativeFlux = {0:'pwl-bdm'}     - will return velocities projected onto the bdm space (CG
                                               Taylor-Hood enriched with DG pw linears on each element)
        conservativeFlux = {0:'pwl-bdm-opt'} - same as pwl-bdm but optimized in a special way to be more
                                               effective.
        """
        self.model = modelList[self.currentModelIndex] # current model
        self.model.points_quadrature.add(('u_last',0))
        self.model.points_elementBoundaryQuadrature.add(('u_last',0))
        self.model.numericalFlux.ebqe[('u_last',0)] = self.model.ebqe[('u_last',0)]
        if self.bdf is int(2):
            self.model.points_quadrature.add(('u_lastlast',0))
            self.model.points_elementBoundaryQuadrature.add(('u_lastlast',0))
            self.model.numericalFlux.ebqe[('u_lastlast',0)] = self.model.ebqe[('u_lastlast',0)]

        if not self.useVelocityComponents and self.velocityModelIndex >= 0:
            assert self.velocityModelIndex < len(modelList), \
                "velocity model index out of  range 0," + repr(len(modelList))
            assert self.pressureIncrementModelIndex < len(modelList), \
                "pressureIncrement model index out of  range 0," + repr(len(modelList))

            self.velocityModel = modelList[self.velocityModelIndex]
            self.pressureIncrementModel = modelList[self.pressureIncrementModelIndex]
            if ('velocity',0) in self.pressureIncrementModel.q:
                vel = self.pressureIncrementModel.q[('velocity',0)]
                self.c_velocity[vel.shape] = vel
                if self.useStabilityTerms:
                    grad_u = self.velocityModel.q[('grad(u)',0)]
                    grad_v = self.velocityModel.q[('grad(u)',1)]
                    self.c_grad_u[grad_u.shape] = grad_u
                    self.c_grad_v[grad_v.shape] = grad_v
            if ('velocity',0) in self.pressureIncrementModel.ebq:
                vel = self.pressureIncrementModel.ebq[('velocity',0)]
                self.c_velocity[vel.shape] = vel
                if self.useStabilityTerms:
                    grad_u = self.velocityModel.ebq[('grad(u)',0)]
                    grad_v = self.velocityModel.ebq[('grad(u)',1)]
                    self.c_grad_u[grad_u.shape] = grad_u
                    self.c_grad_v[grad_v.shape] = grad_v
            if ('velocity',0) in self.pressureIncrementModel.ebqe:
                vel = self.pressureIncrementModel.ebqe[('velocity',0)]
                self.c_velocity[vel.shape] = vel
                if self.useStabilityTerms:
                    grad_u = self.velocityModel.ebqe[('grad(u)',0)]
                    grad_v = self.velocityModel.ebqe[('grad(u)',1)]
                    self.c_grad_u[grad_u.shape] = grad_u
                    self.c_grad_v[grad_v.shape] = grad_v
            if ('velocity',0) in self.pressureIncrementModel.ebq_global:
                vel = self.pressureIncrementModel.ebq_global[('velocity',0)]
                self.c_velocity[vel.shape] = vel
                if self.useStabilityTerms:
                    grad_u = self.velocityModel.ebq_global[('grad(u)',0)]
                    grad_v = self.velocityModel.ebq_global[('grad(u)',1)]
                    self.c_grad_u[grad_u.shape] = grad_u
                    self.c_grad_v[grad_v.shape] = grad_v
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
                    grad_u = self.velocityModel.q[('grad(u)',0)]
                    grad_v = self.velocityModel.q[('grad(u)',1)]
                    self.c_grad_u[grad_u.shape] = grad_u
                    self.c_grad_v[grad_v.shape] = grad_v
                if self.bdf is int(2):
                    u_last = self.velocityModel.q[('u_last',0)]
                    v_last = self.velocityModel.q[('u_last',1)]
                    self.c_u_last[u_last.shape] = u_last
                    self.c_v_last[v_last.shape] = v_last
                    if self.useStabilityTerms:
                        grad_u_last = self.velocityModel.q[('grad(u)_last',0)]
                        grad_v_last = self.velocityModel.q[('grad(u)_last',1)]
                        self.c_grad_u_last[grad_u_last.shape] = grad_u_last
                        self.c_grad_v_last[grad_v_last.shape] = grad_v_last
            if ('u',0) in self.velocityModel.ebq:
                u = self.velocityModel.ebq[('u',0)]
                v = self.velocityModel.ebq[('u',1)]
                self.c_u[u.shape] = u
                self.c_v[v.shape] = v
                if self.useStabilityTerms:
                    grad_u = self.velocityModel.ebq[('grad(u)',0)]
                    grad_v = self.velocityModel.ebq[('grad(u)',1)]
                    self.c_grad_u[grad_u.shape] = grad_u
                    self.c_grad_v[grad_v.shape] = grad_v
                if self.bdf is int(2):
                    u_last = self.velocityModel.ebq[('u_last',0)]
                    v_last = self.velocityModel.ebq[('u_last',1)]
                    self.c_u_last[u_last.shape] = u_last
                    self.c_v_last[v_last.shape] = v_last
                    if self.useStabilityTerms:
                        grad_u_last = self.velocityModel.ebq[('grad(u)_last',0)]
                        grad_v_last = self.velocityModel.ebq[('grad(u)_last',1)]
                        self.c_grad_u_last[grad_u_last.shape] = grad_u_last
                        self.c_grad_v_last[grad_v_last.shape] = grad_v_last
            if ('u',0) in self.velocityModel.ebqe:
                u = self.velocityModel.ebqe[('u',0)]
                v = self.velocityModel.ebqe[('u',1)]
                self.c_u[u.shape] = u
                self.c_v[v.shape] = v
                if self.useStabilityTerms:
                    grad_u = self.velocityModel.ebqe[('grad(u)',0)]
                    grad_v = self.velocityModel.ebqe[('grad(u)',1)]
                    self.c_grad_u[grad_u.shape] = grad_u
                    self.c_grad_v[grad_v.shape] = grad_v
                if self.bdf is int(2):
                    u_last = self.velocityModel.ebqe[('u_last',0)]
                    v_last = self.velocityModel.ebqe[('u_last',1)]
                    self.c_u_last[u_last.shape] = u_last
                    self.c_v_last[v_last.shape] = v_last
                    if self.useStabilityTerms:
                        grad_u_last = self.velocityModel.ebqe[('grad(u)_last',0)]
                        grad_v_last = self.velocityModel.ebqe[('grad(u)_last',1)]
                        self.c_grad_u_last[grad_u_last.shape] = grad_u_last
                        self.c_grad_v_last[grad_v_last.shape] = grad_v_last
            if ('u',0) in self.velocityModel.ebq_global:
                u = self.velocityModel.ebq_global[('u',0)]
                v = self.velocityModel.ebq_global[('u',1)]
                self.c_u[u.shape] = u
                self.c_v[v.shape] = v
                if self.useStabilityTerms:
                    grad_u = self.velocityModel.ebq_global[('grad(u)',0)]
                    grad_v = self.velocityModel.ebq_global[('grad(u)',1)]
                    self.c_grad_u[grad_u.shape] = grad_u
                    self.c_grad_v[grad_v.shape] = grad_v
                if self.bdf is int(2):
                    u_last = self.velocityModel.ebq_global[('u_last',0)]
                    v_last = self.velocityModel.ebq_global[('u_last',1)]
                    self.c_u_last[u_last.shape] = u_last
                    self.c_v_last[v_last.shape] = v_last
                    if self.useStabilityTerms:
                        grad_u_last = self.velocityModel.ebq_global[('grad(u)_last',0)]
                        grad_v_last = self.velocityModel.ebq_global[('grad(u)_last',1)]
                        self.c_grad_u_last[grad_u_last.shape] = grad_u_last
                        self.c_grad_v_last[grad_v_last.shape] = grad_v_last
    def initializeMesh(self,mesh):
        """
        Give the TC object access to the mesh for any mesh-dependent information.
        """
        pass
    def initializeElementQuadrature(self,t,cq):
        """
        Give the TC object access to the element quadrature storage
        """
        for ci in range(self.nc):
            cq[('u_last',ci)] = deepcopy(cq[('u',ci)])
            #cq[('grad(u)_last',ci)] = deepcopy(cq[('grad(u)',ci)])
            if self.bdf is int(2):
                cq[('u_lastlast',ci)] = deepcopy(cq[('u',ci)])
                # #cq[('grad(u)_lastlast',ci)] = deepcopy(cq[('grad(u)_last',ci)])
    def initializeElementBoundaryQuadrature(self,t,cebq,cebq_global):
        """
        Give the TC object access to the element boundary quadrature storage
        """
        # for ci in range(self.nc):
        #     cebq[('u_last',ci)] = deepcopy(cebq[('u',ci)])
        #     cebq[('grad(u)_last',ci)] = deepcopy(cebq[('grad(u)',ci)])
        #     cebq_global[('u_last',ci)] = deepcopy(cebq_global[('u',ci)])
        #     cebq_global[('grad(u)_last',ci)] = deepcopy(cebq_global[('grad(u)',ci)])
        pass
    def initializeGlobalExteriorElementBoundaryQuadrature(self,t,cebqe):
        """
        Give the TC object access to the exterior element boundary quadrature storage
        """
        for ci in range(self.nc):
            cebqe[('u_last',ci)] = deepcopy(cebqe[('u',ci)])
            #cebqe[('grad(u)_last',ci)] = deepcopy(cebqe[('grad(u)',ci)])
            if self.bdf is int(2):
                cebqe[('u_lastlast',ci)] = deepcopy(cebqe[('u',ci)])
                # #cebqe[('grad(u)_lastlast',ci)] = deepcopy(cebqe[('grad(u)',ci)])

    def initializeGeneralizedInterpolationPointQuadrature(self,t,cip):
        """
        Give the TC object access to the generalized interpolation point storage. These points are used  to project nonlinear potentials (phi).
        """
        pass
    def preStep(self,t,firstStep=False):
        """
        Give the TC object an opportunity to modify itself before the time step.
        """
        self.firstStep = firstStep # save for in evaluate

        for ci in range(self.nc):
            if not self.firstStep and self.bdf is int(2):
                self.model.q[('u_lastlast',ci)][:] = self.model.q[('u_last',ci)]
                # self.model.ebq[('u_lastlast',ci)][:] = self.model.ebq[('u_last',ci)]
                self.model.ebqe[('u_lastlast',ci)][:] = self.model.ebqe[('u_last',ci)]
                # self.model.ebq_global[('u_lastlast',ci)][:] = self.model.ebq_global[('u_last',ci)]

                # self.model.q[('grad(u)_lastlast',ci)][:] = self.model.q[('grad(u)_last',ci)]
                # self.model.ebq[('grad(u)_lastlast',ci)][:] = self.model.ebq[('grad(u)_last',ci)]
                # self.model.ebqe[('grad(u)_lastlast',ci)][:] = self.model.ebqe[('grad(u)_last',ci)]
                # self.model.ebq_global[('grad(u)_lastlast',ci)][:] = self.model.ebq_global[('grad(u)_last',ci)]

            self.model.q[('u_last',ci)][:] = self.model.q[('u',ci)]
            # self.model.ebq[('u_last',ci)][:] = self.model.ebq[('u',ci)]
            self.model.ebqe[('u_last',ci)][:] = self.model.ebqe[('u',ci)]
            # self.model.ebq_global[('u_last',ci)][:] = self.model.ebq_global[('u',ci)]

            # self.model.q[('grad(u)_last',ci)][:] = self.model.q[('grad(u)',ci)]
            # self.model.ebq[('grad(u)_last',ci)][:] = self.model.ebq[('grad(u)',ci)]
            # self.model.ebqe[('grad(u)_last',ci)][:] = self.model.ebqe[('grad(u)',ci)]
            # self.model.ebq_global[('grad(u)_last',ci)][:] = self.model.ebq_global[('grad(u)',ci)]
        copyInstructions = {}
        return copyInstructions
    def evaluate(self,t,c):
        """
        Evaluate the coefficients after getting the specified velocity
        """
        rho = c[('u',0)]
        # if self.bdf is int(2) and not self.firstStep:
        #     grad_rho = c[('grad(u)',0)]

        # If we use pressureIncrementModel.q[('velocity',)] for our velocity, then we must
        # adjust it to be scaled properly by multiplying by dt/chi.  Then it is physical velocity
        # hopefully with divergence free properties.
        dt = self.model.timeIntegration.dt  # 0 = densityModelIndex
        if self.bdf is int(2):
            dt_last = self.model.timeIntegration.dt_history[0] # note this only exists if we are using VBDF for Time integration

        # extract scaling terms for post processed velocity (see pressureIncrement model for description)
        if not self.useVelocityComponents and self.velocityFunction is None:
            chi = self.chiValue
            # beta_0 coefficient scalar
            if self.bdf is int(1) or self.firstStep:
                b0 = 1.0/dt
            elif self.bdf is int(2):
                r = dt/dt_last
                dtInv = 1.0/dt
                b0 = (1.0+2.0*r)/(1.0+r)*dtInv  # = self.model.timeIntegration.alpha_bdf  # = beta_0
            Invb0chi = 1.0/(chi*b0)


        # extract current velocity components
        if self.velocityFunction != None:
            u = self.velocityFunction(c['x'],t)[...,0]
            v = self.velocityFunction(c['x'],t)[...,1]
            if self.useStabilityTerms:
                div_vel = self.divVelocityFunction(c['x'],t)
        elif self.useVelocityComponents:
            u = self.c_u[c[('m',0)].shape]
            v = self.c_v[c[('m',0)].shape]
            if self.useStabilityTerms:
                div_vel = self.c_grad_u[c[('f',0)].shape][...,0] + self.c_grad_v[c[('f',0)].shape][...,1]
        else:
            u = Invb0chi*self.c_velocity[c[('f',0)].shape][...,0]  # make adjustment to physical values here by mult by dt/chi
            v = Invb0chi*self.c_velocity[c[('f',0)].shape][...,1]  # make adjustment to physical values here by mult by dt/chi
            if self.useStabilityTerms:
                div_vel = self.c_grad_u[c[('f',0)].shape][...,0] + self.c_grad_v[c[('f',0)].shape][...,1]


        # choose the velocity to be used for transport
        if self.bdf is int(1) or self.firstStep or self.velocityFunction != None or not self.useVelocityComponents: # TODO move useVelocityComponents to second order
            # use first order extrapolation of velocity
            u_star = u
            v_star = v
            if self.useStabilityTerms:
                div_vel_star = div_vel
        elif self.bdf is int(2):
            # extract the previous time step values as needed
            u_last = self.c_u_last[c[('m',0)].shape]
            v_last = self.c_v_last[c[('m',0)].shape]
            # use second order extrapolation of velocity
            u_star = u + dt/dt_last*( u - u_last )
            v_star = v + dt/dt_last*( v - v_last )
            if self.useStabilityTerms:
                div_vel_last = self.c_grad_u_last[c[('f',0)].shape][...,0] + self.c_grad_v_last[c[('f',0)].shape][...,1]
                div_vel_star = div_vel + dt/dt_last*(div_vel - div_vel_last )
        else:
            assert False, "Error: self.bdf = %f is not supported" %self.bdf

        #  bdf1:  rho_t + div( rho vel_star) - 0.5 rho div( vel_star ) = 0
        #  bdf2:  rho_t + vel_star \cdot grad_rho - 0.5 rho div( vel_star ) = 0
        c[('m',0)][:] = rho
        c[('dm',0,0)][:] = 1.0
        # if self.bdf is int(1) or self.firstStep:
        c[('f',0)][...,0] = rho*u_star
        c[('f',0)][...,1] = rho*v_star
        c[('df',0,0)][...,0] = u_star
        c[('df',0,0)][...,1] = v_star
        # elif self.bdf is int(2):
        #     c[('H',0)][:] = u_star*grad_rho[...,0] + v_star*grad_rho[...,1]
        #     c[('dH',0,0)][...,0] = u_star #  dH d(u_x)
        #     c[('dH',0,0)][...,1] = v_star #  dH d(u_y)
        # else:
        #     assert False, "Error: self.bdf = %f is not supported" %self.bdf
        if self.useStabilityTerms:
            c[('r',0)][:]    = -0.5*rho*div_vel_star
            c[('dr',0,0)][:] = -0.5*div_vel_star



class VelocityTransport2D(TransportCoefficients.TC_base):
    r"""
    The coefficients of the 2D Navier Stokes momentum equation with variable density.  This coefficient class
    will only represent the momentum equation but not the incompressibility equation and not the conservation of mass.

    .. math::
       :nowrap:

       \begin{equation}
       \begin{cases}
       p^{\#} = p^{k} + \phi^{k+1}
       \begin{split}
       \rho^{k}\frac{\tilde{\mathbf{u}^{k+1}}- \tilde{\mathbf{u}^{k}}}{\tau} + \rho^{k+1}\tilde{\mathbf{u}^{k}}\cdot\nabla\tilde{\mathbf{u}^{k+1}} &\\
        +  \nabla p^{\#}  - \nabla \cdot \left(\mu \nabla\tilde{\mathbf{u}^{k+1}}\right) &\\
        +\frac{1}{2}\left( \frac{\rho^{k+1} - \rho^{k}}{\tau} + \nabla\cdot\left(\rho^{k+1}\tilde{\mathbf{u}^{k+1}} \right) \right)\tilde{\mathbf{u}^{k+1}} &= \mathbf{f}(x,t)
       \end{split}
       \end{cases}
       \end{equation}

    where :math:`\rho^{k+1}>0` is the density at time :math:`t^{k+1}`, :math:`\mathbf{u}^{k+1}` is the velocity field,  :math:`p` is the pressure and :math:`\mu` is the dynamic
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
                 bdf=1,
                 f1ofx=None,
                 f2ofx=None,
                 mu=1.0,
                 currentModelIndex=1,
                 densityModelIndex=-1,
                 densityFunction=None,
                 densityGradFunction=None,
                 pressureIncrementModelIndex=-1,
                 pressureIncrementGradFunction=None,
                 pressureModelIndex=-1,
                 pressureGradFunction=None,
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
                         mass = {eu:{ui:'linear'},  # du/dt
                                 ev:{vi:'linear'}}, # dv/dt
                         hamiltonian = {eu:{ui:'linear'},  # rho*(u u_x + v u_y)   convection term
                                        ev:{vi:'linear'}}, # rho*(u v_x + v v_y)   convection term
                         diffusion = {eu:{ui:{ui:'constant'}},  # - \mu * \grad u
                                      ev:{vi:{vi:'constant'}}}, # - \mu * \grad v
                         potential = {eu:{ui:'u'},
                                      ev:{vi:'u'}}, # define the potential for the diffusion term to be the solution itself
                         reaction  = {eu:{ui:'constant'},  # -f1(x) + d/dx p^\# + (stability terms) * u
                                      ev:{vi:'constant'}}, # -f2(x) + d/dy p^\# + (stability terms) * v
                         sparseDiffusionTensors=sdInfo,
                         useSparseDiffusion = True),
        self.vectorComponents=[ui,vi]  # for plotting and hdf5 output only
        self.bdf=int(bdf)
        self.f1ofx=f1ofx
        self.f2ofx=f2ofx
        self.mu=mu
        self.currentModelIndex = currentModelIndex
        self.densityModelIndex = densityModelIndex
        self.densityFunction = densityFunction
        self.densityGradFunction = densityGradFunction
        self.pressureModelIndex = pressureModelIndex
        self.pressureGradFunction = pressureGradFunction
        self.pressureIncrementModelIndex = pressureIncrementModelIndex
        self.pressureIncrementGradFunction = pressureIncrementGradFunction
        self.useStabilityTerms = useStabilityTerms
        self.c_rho = {} # density
        self.c_rho_last = {} # density of cached values
        if self.bdf is int(2):
            self.c_rho_lastlast = {}
        self.c_p_last = {}  # pressure
        self.c_phi_last = {} # pressure increment phi
        if self.bdf is int(2):
            self.c_phi_lastlast = {}
        self.firstStep = True # manipulated in preStep()


    def attachModels(self,modelList):
        """
        Attach the models for density, pressure increment and pressure
        """
        self.model = modelList[self.currentModelIndex] # current model
        for ci in range(self.nc):
            self.model.points_quadrature.add(('u_last',ci))
            self.model.vectors_quadrature.add(('grad(u)_last',ci))
            self.model.numericalFlux.ebqe[('u_last',ci)]=self.model.ebqe[('u_last',ci)]
            self.model.points_elementBoundaryQuadrature.add(('u_last',ci))
            self.model.vectors_elementBoundaryQuadrature.add(('grad(u)_last',ci))
            self.model.numericalFlux.ebqe[('grad(u)_last',ci)]=self.model.ebqe[('grad(u)_last',ci)]
            if self.bdf is int(2):
                self.model.points_quadrature.add(('u_lastlast',ci))
                self.model.vectors_quadrature.add(('grad(u)_lastlast',ci))
                self.model.numericalFlux.ebqe[('u_lastlast',ci)]=self.model.ebqe[('u_lastlast',ci)]
                self.model.points_elementBoundaryQuadrature.add(('u_lastlast',ci))
                self.model.vectors_elementBoundaryQuadrature.add(('grad(u)_lastlast',ci))
                self.model.numericalFlux.ebqe[('grad(u)_lastlast',ci)]=self.model.ebqe[('grad(u)_lastlast',ci)]
        if self.densityModelIndex >= 0:
            assert self.densityModelIndex < len(modelList), \
                "density model index out of range 0," + repr(len(modelList))
            self.densityModel = modelList[self.densityModelIndex]
            if ('u',0) in self.densityModel.q:
                rho = self.densityModel.q[('u',0)]
                self.c_rho[rho.shape] = rho
                rho_last = self.densityModel.q[('u_last',0)]
                self.c_rho_last[rho_last.shape] = rho_last
                if self.useStabilityTerms:
                    grad_rho = self.densityModel.q[('grad(u)',0)]
                    self.c_rho[grad_rho.shape] = grad_rho
                if self.bdf is int(2):
                    rho_lastlast = self.densityModel.q[('u_lastlast',0)]
                    self.c_rho_lastlast[rho_lastlast.shape] = rho_lastlast
            if ('u',0) in self.densityModel.ebq:
                rho = self.densityModel.ebq[('u',0)]
                self.c_rho[rho.shape] = rho
                rho_last = self.densityModel.ebq[('u_last',0)]
                self.c_rho_last[rho_last.shape] = rho_last
                if self.useStabilityTerms:
                    grad_rho = self.densityModel.ebq[('grad(u)',0)]
                    self.c_rho[grad_rho.shape] = grad_rho
                if self.bdf is int(2):
                    rho_lastlast = self.densityModel.ebq[('u_lastlast',0)]
                    self.c_rho_lastlast[rho_lastlast.shape] = rho_lastlast
            if ('u',0) in self.densityModel.ebqe:
                rho = self.densityModel.ebqe[('u',0)]
                self.c_rho[rho.shape] = rho
                rho_last = self.densityModel.ebqe[('u_last',0)]
                self.c_rho_last[rho_last.shape] = rho_last
                if self.useStabilityTerms:
                    grad_rho = self.densityModel.ebqe[('grad(u)',0)]
                    self.c_rho[grad_rho.shape] = grad_rho
                if self.bdf is int(2):
                    rho_lastlast = self.densityModel.ebqe[('u_lastlast',0)]
                    self.c_rho_lastlast[rho_lastlast.shape] = rho_lastlast
            if ('u',0) in self.densityModel.ebq_global:
                rho = self.densityModel.ebq_global[('u',0)]
                self.c_rho[rho.shape] = rho
                rho_last = self.densityModel.ebq_global[('u_last',0)]
                self.c_rho_last[rho_last.shape] = rho_last
                if self.useStabilityTerms:
                    grad_rho = self.densityModel.ebq_global[('grad(u)',0)]
                    self.c_rho[grad_rho.shape] = grad_rho
                if self.bdf is int(2):
                    rho_lastlast = self.densityModel.ebq_global[('u_lastlast',0)]
                    self.c_rho_lastlast[rho_lastlast.shape] = rho_lastlast
        if self.pressureIncrementModelIndex >= 0:
            assert self.pressureIncrementModelIndex < len(modelList), \
                "pressure increment model index out of range 0," + repr(len(modelList))
            self.pressureIncrementModel = modelList[self.pressureIncrementModelIndex]
            if ('u',0) in self.pressureIncrementModel.q:
                grad_phi_last = self.pressureIncrementModel.q[('grad(u)',0)]  #grad(u) here is at time t^{k} so is _last now
                self.c_phi_last[grad_phi_last.shape] = grad_phi_last
                if self.bdf is int(2):
                    grad_phi_lastlast = self.pressureIncrementModel.q[('grad(u)_last',0)]  # grad(u)_last here is at time t^{k-1} so is _lastlast now
                    self.c_phi_lastlast[grad_phi_lastlast.shape] = grad_phi_lastlast
            if ('u',0) in self.pressureIncrementModel.ebq:
                grad_phi_last = self.pressureIncrementModel.ebq[('grad(u)',0)]
                self.c_phi_last[grad_phi_last.shape] = grad_phi_last
                if self.bdf is int(2):
                    grad_phi_lastlast = self.pressureIncrementModel.ebq[('grad(u)_last',0)]
                    self.c_phi_lastlast[grad_phi_lastlast.shape] = grad_phi_lastlast
            if ('u',0) in self.pressureIncrementModel.ebqe:
                grad_phi_last = self.pressureIncrementModel.ebqe[('grad(u)',0)]
                self.c_phi_last[grad_phi_last.shape] = grad_phi_last
                if self.bdf is int(2):
                    grad_phi_lastlast = self.pressureIncrementModel.ebqe[('grad(u)_last',0)]
                    self.c_phi_lastlast[grad_phi_lastlast.shape] = grad_phi_lastlast
            if ('u',0) in self.pressureIncrementModel.ebq_global:
                grad_phi_last = self.pressureIncrementModel.ebq_global[('grad(u)',0)]
                self.c_phi_last[grad_phi_last.shape] = grad_phi_last
                if self.bdf is int(2):
                    grad_phi_lastlast = self.pressureIncrementModel.ebq_global[('grad(u)_last',0)]
                    self.c_phi_lastlast[grad_phi_lastlast.shape] = grad_phi_lastlast
        if self.pressureModelIndex >= 0:
            assert self.pressureModelIndex < len(modelList), \
                "pressure model index out of range 0," + repr(len(modelList))
            self.pressureModel = modelList[self.pressureModelIndex]
            if ('u',0) in self.pressureModel.q:
                grad_p_last = self.pressureModel.q[('grad(u)',0)] #grad(u) here is at time t^{k} so is _last in this context
                self.c_p_last[grad_p_last.shape] = grad_p_last
            if ('u',0) in self.pressureModel.ebq:
                grad_p_last = self.pressureModel.ebq[('grad(u)',0)]
                self.c_p_last[grad_p_last.shape] = grad_p_last
            if ('u',0) in self.pressureModel.ebqe:
                grad_p_last = self.pressureModel.ebqe[('grad(u)',0)]
                self.c_p_last[grad_p_last.shape] = grad_p_last
            if ('u',0) in self.pressureModel.ebq_global:
                grad_p_last = self.pressureModel.ebq_global[('grad(u)',0)]
                self.c_p_last[grad_p_last.shape] = grad_p_last
    def initializeMesh(self,mesh):
        """
        Give the TC object access to the mesh for any mesh-dependent information.
        """
        pass
    def initializeElementQuadrature(self,t,cq):
        """
        Give the TC object access to the element quadrature storage
        """
        for ci in range(self.nc):
            cq[('u_last',ci)] = deepcopy(cq[('u',ci)])
            cq[('grad(u)_last',ci)] = deepcopy(cq[('grad(u)',ci)])
            if self.bdf is int(2):
                cq[('u_lastlast',ci)] = deepcopy(cq[('u',ci)])
                cq[('grad(u)_lastlast',ci)] = deepcopy(cq[('grad(u)',ci)])

    def initializeElementBoundaryQuadrature(self,t,cebq,cebq_global):
        """
        Give the TC object access to the element boundary quadrature storage
        """
        # for ci in range(self.nc):
        #     cebq[('u_last',ci)] = deepcopy(cebq[('u',ci)])
        #     cebq[('grad(u)_last',ci)] = deepcopy(cebq[('grad(u)',ci)])
        #     cebq_global[('u_last',ci)] = deepcopy(cebq_global[('u',ci)])
        #     cebq_global[('grad(u)_last',ci)] = deepcopy(cebq_global[('grad(u)',ci)])
        pass
    def initializeGlobalExteriorElementBoundaryQuadrature(self,t,cebqe):
        """
        Give the TC object access to the exterior element boundary quadrature storage
        """
        for ci in range(self.nc):
            cebqe[('u_last',ci)] = deepcopy(cebqe[('u',ci)])
            cebqe[('grad(u)_last',ci)] = deepcopy(cebqe[('grad(u)',ci)])
            if self.bdf is int(2):
                cebqe[('u_lastlast',ci)] = deepcopy(cebqe[('u',ci)])
                cebqe[('grad(u)_lastlast',ci)] = deepcopy(cebqe[('grad(u)',ci)])
    def initializeGeneralizedInterpolationPointQuadrature(self,t,cip):
        """
        Give the TC object access to the generalized interpolation point storage. These points are used  to project nonlinear potentials (phi).
        """
        pass
    def preStep(self,t,firstStep=False):
        """
        Give the TC object an opportunity to modify itself before the time step.
        """
        self.firstStep = firstStep # save for use in evaluate

        for ci in range(self.nc):
            if self.bdf is int(2) and not self.firstStep:
                # deep copy so we have a cached value instead of pointer to current values
                self.model.q[('u_lastlast',ci)][:] = self.model.q[('u_last',ci)]
                # self.model.ebq[('u_lastlast',ci)][:] = self.model.ebq[('u_last',ci)]
                self.model.ebqe[('u_lastlast',ci)][:] = self.model.ebqe[('u_last',ci)]
                # self.model.ebq_global[('u_lastlast',ci)][:] = self.model.ebq_global[('u_last',ci)]

                self.model.q[('grad(u)_lastlast',ci)][:] = self.model.q[('grad(u)_last',ci)]
                # self.model.ebq[('grad(u)_lastlast',ci)][:] = self.model.ebq[('grad(u)_last',ci)]
                self.model.ebqe[('grad(u)_lastlast',ci)][:] = self.model.ebqe[('grad(u)_last',ci)]
                # self.model.ebq_global[('grad(u)_lastlast',ci)][:] = self.model.ebq_global[('grad(u)_last',ci)]

            # deep copy so we have a cached value instead of pointer to current values
            self.model.q[('u_last',ci)][:] = self.model.q[('u',ci)]
            # self.model.ebq[('u_last',ci)][:] = self.model.ebq[('u',ci)]
            self.model.ebqe[('u_last',ci)][:] = self.model.ebqe[('u',ci)]
            # self.model.ebq_global[('u_last',ci)][:] = self.model.ebq_global[('u',ci)]

            self.model.q[('grad(u)_last',ci)][:] = self.model.q[('grad(u)',ci)]
            # self.model.ebq[('grad(u)_last',ci)][:] = self.model.ebq[('grad(u)',ci)]
            self.model.ebqe[('grad(u)_last',ci)][:] = self.model.ebqe[('grad(u)',ci)]
            # self.model.ebq_global[('grad(u)_last',ci)][:] = self.model.ebq_global[('grad(u)',ci)]

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

        # time management
        dt = self.model.timeIntegration.dt  # 0 = velocityModelIndex
        if self.bdf is int(2) and not self.firstStep:
            dt_last = self.model.timeIntegration.dt_history[0] # note this only exists if we are using VBDF for Time integration
            dtInv = 1.0/dt
            r = dt/dt_last
            # set coefficients  m_t  = b0*m^{k+1} - b1*m^{k} - b2*m^{k-1}
            b0 = (1.0+2.0*r)/(1.0+r)*dtInv   # is self.model.timeIntegration.alpha_bdf as set in calculateCoefs() of timeIntegration.py
            b1 = (1.0+r)*dtInv               # is -b0 as set in self.model.timeIntegration.calculateCoefs()
            b2 = -r*r/(1.0+r)*dtInv          # is -b1 as set in self.model.timeIntegration.calculateCoefs()

        tLast = self.model.timeIntegration.tLast
        if self.bdf is int(2) and not self.firstStep:
            tLastLast = tLast - dt_last

        # current velocity and grad velocity
        u = c[('u',ui)]
        v = c[('u',vi)]
        grad_u = c[('grad(u)',ui)]
        grad_v = c[('grad(u)',vi)]

        # previous velocity and grad velocity      # TODO:  decide whether or not to use post processed velocities here...
        u_last = c[('u_last',ui)]
        v_last = c[('u_last',vi)]
        grad_u_last = c[('grad(u)_last',ui)]
        grad_v_last = c[('grad(u)_last',vi)]

        # velocities *_lastlast  will be extracted as needed below

        # extract rho, rho_last, rho_lastlast and grad_rho as needed
        if self.densityFunction != None:
            rho = self.densityFunction(c['x'],t)
            rho_last =  self.densityFunction(c['x'],tLast)
            if self.useStabilityTerms:
                grad_rho = self.densityGradFunction(c['x'],t)
            if self.bdf is int(2) and not self.firstStep:
                rho_lastlast = self.densityFunction(c['x'],tLastLast)
        else:#use mass shape as key since it is same shape as density
            rho = self.c_rho[c[('m',0)].shape]
            rho_last = self.c_rho_last[c[('m',0)].shape]
            if self.useStabilityTerms:
                grad_rho = self.c_rho[c[('grad(u)',0)].shape] # use velocity shape since it is same shape as gradient
            if self.bdf is int(2) and not self.firstStep:
                rho_lastlast = self.c_rho_lastlast[c[('m',0)].shape]

        # extract grad_p_last  p^{k}
        if self.pressureGradFunction != None:
            grad_p_last = self.pressureGradFunction(c['x'],tLast)
        else:#use grad of component u shape as key since it is same shape as gradient of pressure
            grad_p_last = self.c_p_last[c[('grad(u)',0)].shape]

        # extract grad_phi_last phi^{k}
        if self.pressureIncrementGradFunction != None:
            grad_phi_last = self.pressureIncrementGradFunction(c['x'],tLast)
        else:#use velocity shape as key since it is same shape as gradient of pressure increment
            grad_phi_last = self.c_phi_last[c[('grad(u)',0)].shape]

        # extract grad_phi_lastlast  phi^{k-1}
        if self.bdf is int(2) and not self.firstStep:
            if self.pressureIncrementGradFunction != None:
                grad_phi_lastlast = self.pressureIncrementGradFunction(c['x'],tLastLast)
            else:#use velocity shape as key since it is same shape as gradient of pressure increment
                grad_phi_lastlast = self.c_phi_lastlast[c[('grad(u)',0)].shape]


        # choose the density to use on the mass term,  bdf1 is rho_last,  bdf2 is current rho
        # as well as the other various element (not velocity) that differ between bdf1 and bdf2
        if self.bdf is int(1) or self.firstStep:
            rho_sharp = rho_last
            rho_t = (rho - rho_last)/dt # bdf1 time derivative
            grad_psharp = grad_p_last + grad_phi_last
        elif self.bdf is int(2):
            rho_sharp = rho
            rho_t = b0*rho - b1*rho_last - b2*rho_lastlast #bdf2 time derivative  (see above for descriptions and definitions of b0 b1 and b2)
            grad_psharp = grad_p_last + b1/b0 * grad_phi_last + b2/b0 *grad_phi_lastlast

        # extrapolation of velocity
        if self.bdf is int(1) or self.firstStep:
            # first order extrapolation
            u_star = u_last
            v_star = v_last
            if self.useStabilityTerms:
                div_vel_star = grad_u_last[...,xi] + grad_v_last[...,yi]
                div_rho_vel_star = grad_rho[...,xi]*u_star + grad_rho[...,yi]*v_star + rho*div_vel_star
        elif self.bdf is int(2):
            # TODO:  decide whether or not to use post processed velocities here...
            u_lastlast = c[('u_lastlast',ui)]
            v_lastlast = c[('u_lastlast',vi)]
            # use second order extrapolation of velocity
            u_star = u_last + dt/dt_last*( u_last - u_lastlast ) # TODO: note could use r instead of dt/dt_last here to speed it up a bit
            v_star = v_last + dt/dt_last*( v_last - v_lastlast )
            if self.useStabilityTerms:
                grad_u_lastlast = c[('grad(u)_lastlast',ui)]
                grad_v_lastlast = c[('grad(u)_lastlast',vi)]
                div_vel_last = grad_u_last[...,xi] + grad_v_last[...,yi]
                div_vel_lastlast = grad_u_lastlast[...,xi] + grad_v_lastlast[...,yi]
                # use second order extrapolation of div velocity
                div_vel_star = div_vel_last + dt/dt_last*( div_vel_last - div_vel_lastlast)
                div_rho_vel_star = grad_rho[...,xi]*u_star + grad_rho[...,yi]*v_star + rho*div_vel_star


        #equation eu = 0
        # rho_sharp*u_t + rho(u_star u_x + v_star u_y ) + p_sharp_x - f1 + div(-mu grad(u))
        #            + 0.5( rho_t + rho_x u_star + rho_y v_star + rho div([u_star,v_star]) )u = 0
        c[('m',eu)][:] = rho_sharp*u    # d/dt ( rho_sharp * u) = d/dt (m_0)
        c[('dm',eu,ui)][:] = rho_sharp  # dm^0_du
        c[('r',eu)][:] = -self.f1ofx(c['x'][:],t) + grad_psharp[...,xi]
        c[('dr',eu,ui)][:] = 0.0
        if self.useStabilityTerms:
            c[('r',eu)][:] += 0.5*( rho_t + div_rho_vel_star )*u
            c[('dr',eu,ui)][:] += 0.5*( rho_t + div_rho_vel_star )
        c[('H',eu)][:] = rho*( u_star*grad_u[...,xi] + v_star*grad_u[...,yi] )
        c[('dH',eu,ui)][...,xi] = rho*u_star #  dH d(u_x)
        c[('dH',eu,ui)][...,yi] = rho*v_star #  dH d(u_y)
        c[('a',eu,ui)][...,0] = self.mu # -mu*\grad v :   tensor  [ mu  0;  0  mu] ordered [0 1; 2 3]  in our
        c[('a',eu,ui)][...,1] = self.mu # -mu*\grad v :       new diagonal notation from sDInfo above is [0 .; . 1] -> [0; 1]
        c[('da',eu,ui,ui)][...,0] = 0.0 # -(da/d ui)_0   # could leave these off since it is 0
        c[('da',eu,ui,ui)][...,1] = 0.0 # -(da/d ui)_1   # could leave these off since it is 0

        #equation ev = 1
        # rho_sharp*v_t + rho(u_star v_x + v_star v_y ) + p_sharp_y - f2 + div(-mu grad(v))
        #            + 0.5( rho_t + rho_x u_star + rho_y v_star + rho div([u_star,v_star]) )v = 0
        c[('m',ev)][:] = rho_sharp*v    # d/dt ( rho_sharp * v) = d/dt (m_0)
        c[('dm',ev,vi)][:] = rho_sharp  # dm^0_dv
        c[('r',ev)][:] = -self.f2ofx(c['x'][:],t) + grad_psharp[...,yi]
        c[('dr',ev,vi)][:] = 0.0
        if self.useStabilityTerms:
            c[('r',ev)][:] += 0.5*( rho_t + div_rho_vel_star )*v
            c[('dr',ev,vi)][:] += 0.5*( rho_t + div_rho_vel_star )
        c[('H',ev)][:] = rho*( u_star*grad_v[...,xi] + v_star*grad_v[...,yi] ) # add rho term
        c[('dH',ev,vi)][...,xi] = rho*u_star #  dH d(v_x)
        c[('dH',ev,vi)][...,yi] = rho*v_star #  dH d(v_y)
        c[('a',ev,vi)][...,0] = self.mu # -mu*\grad v :   tensor  [ mu  0;  0  mu] ordered [0 1; 2 3]  in our
        c[('a',ev,vi)][...,1] = self.mu # -mu*\grad v :       new diagonal notation from sDInfo above is [0 .; . 1] -> [0; 1]
        c[('da',ev,vi,vi)][...,0] = 0.0 # -(da/d vi)_0   # could leave these off since it is 0
        c[('da',ev,vi,vi)][...,1] = 0.0 # -(da/d vi)_1   # could leave these off since it is 0


class PressureIncrement2D(TransportCoefficients.TC_base):
    r"""
    The coefficients for pressure increment solution

    Update is given by

    .. math::

       \nabla\cdot( -\nabla \phi^{k+1} - \chi/dt\mathbf{u}^{k+1} ) = 0
    """
    def __init__(self,
                 bdf=1,
                 currentModelIndex=2,
                 velocityModelIndex=-1,
                 velocityFunction=None,
                 densityModelIndex=-1,
                 zeroMean=False,
                 chiValue=1.0):
        """Construct a coefficients object

        :param velocityModelIndex: The index into the proteus model list

        :param divVelocityFunction: A function taking as input an array of spatial
        locations :math: `x`, time :math: `t`, and velocity :math: `v`, setting
        the divVelocity parameter as a side effect.

        TODO: decide how the parameters interact. I think divVelocityFunction
        should override the velocity from another model

        """
        sdInfo  = {(0,0):(np.array([0,1,2],dtype='i'),  # sparse diffusion uses diagonal element for diffusion coefficient
                          np.array([0,1],dtype='i'))}

        TransportCoefficients.TC_base.__init__(self,
                                               nc = 1,
                                               variableNames = ['phi'],
                                               diffusion = {0:{0:{0:'constant'}}}, # - \grad phi
                                               potential = {0:{0:'u'}}, # define the potential for the diffusion term to be the solution itself
                                               advection = {0:{0:'constant'}}, # div (chi/dt velocity)
                                               sparseDiffusionTensors=sdInfo,
                                               useSparseDiffusion = True)
        self.bdf=int(bdf)
        self.chiValue = chiValue
        self.currentModelIndex = currentModelIndex
        self.velocityModelIndex = velocityModelIndex
        self.velocityFunction = velocityFunction
        self.densityModelIndex = densityModelIndex
        self.c_u = {}
        self.c_v = {}
        self.c_velocity = {}
        self.c_rho = {}
        self.firstStep = True # manipulated in preStep()
        self.zeroMean = zeroMean

    def attachModels(self,modelList):
        """
        Attach the model for velocity and density
        """
        self.model = modelList[self.currentModelIndex] # current model
        # will need grad_phi and grad_phi_last for bdf2 algorithm in velocity model
        # representing times t^{k} and t^{k-1} respectively.  Since velocity is before
        # pressure increment, we haven't had a chance to move the phi's to be consistent
        # notation on the new time step so we must handle this inconsistency here.
        if self.bdf is int(2):
            self.model.vectors_quadrature.add(('grad(u)_last',0))
            self.model.vectors_elementBoundaryQuadrature.add(('grad(u)_last',0))
            self.model.numericalFlux.ebqe[('grad(u)_last',0)] = self.model.ebqe[('grad(u)_last',0)]
        if self.velocityModelIndex >= 0:
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
    def initializeMesh(self,mesh):
        """
        Give the TC object access to the mesh for any mesh-dependent information.
        """
        pass
    def initializeElementQuadrature(self,t,cq):
        """
        Give the TC object access to the element quadrature storage
        """
        if self.bdf is int(2):
            for ci in range(self.nc):
                # cq[('u_last',ci)] = deepcopy(cq[('u',ci)])
                cq[('grad(u)_last',ci)] = deepcopy(cq[('grad(u)',ci)])

    def initializeElementBoundaryQuadrature(self,t,cebq,cebq_global):
        """
        Give the TC object access to the element boundary quadrature storage
        """
        # if self.bdf is int(2):
        #     for ci in range(self.nc):
        #         cebq[('u_last',ci)] = deepcopy(cebq[('u',ci)])
        #         cebq[('grad(u)_last',ci)] = deepcopy(cebq[('grad(u)',ci)])
        #         cebq_global[('u_last',ci)] = deepcopy(cebq_global[('u',ci)])
        #         cebq_global[('grad(u)_last',ci)] = deepcopy(cebq_global[('grad(u)',ci)])
        pass
    def initializeGlobalExteriorElementBoundaryQuadrature(self,t,cebqe):
        """
        Give the TC object access to the exterior element boundary quadrature storage
        """
        if self.bdf is int(2):
            for ci in range(self.nc):
                # cebqe[('u_last',ci)] = deepcopy(cebqe[('u',ci)])
                cebqe[('grad(u)_last',ci)] = deepcopy(cebqe[('grad(u)',ci)])


    def initializeGeneralizedInterpolationPointQuadrature(self,t,cip):
        """
        Give the TC object access to the generalized interpolation point storage. These points are used  to project nonlinear potentials (phi).
        """
        pass
    def preStep(self,t,firstStep=False):
        """
        Move the current values to values_last to keep cached set of values for bdf1 algorithm
        """
        self.firstStep = firstStep # save for use in evaluate()

        for ci in range(self.nc):
            # The only time we need phi^{k} and phi^{k-1} are when we are on step {k+1} in
            # the velocity update.  But then phi^{k} is the current solution 'u' to pressureIncrement
            # and phi^{k-1} is 'u_last' since we have moved ahead one time step but haven't reached
            # pressureIncrement yet to move them over.
            #
            # So for bdf1 we only need
            # pressureIncrementModel.q['grad(u)']      representing grad phi^{k}
            # and for bdf2 we also need
            # pressureIncrementModel.q['grad(u)_last']    representing grad phi^{k-1}
            #
            if not self.firstStep and self.bdf is int(2):
                # self.model.q[('u_last',ci)][:] = self.model.q[('u',ci)]
                # self.model.ebq[('u_last',ci)][:] = self.model.ebq[('u',ci)]
                # self.model.ebqe[('u_last',ci)][:] = self.model.ebqe[('u',ci)]
                # self.model.ebq_global[('u_last',ci)][:] = self.model.ebq_global[('u',ci)]

                self.model.q[('grad(u)_last',ci)][:] = self.model.q[('grad(u)',ci)]
                # self.model.ebq[('grad(u)_last',ci)][:] = self.model.ebq[('grad(u)',ci)]
                self.model.ebqe[('grad(u)_last',ci)][:] = self.model.ebqe[('grad(u)',ci)]
                # self.model.ebq_global[('grad(u)_last',ci)][:] = self.model.ebq_global[('grad(u)',ci)]

        copyInstructions = {}
        return copyInstructions
    def postStep(self,t,firstStep=False):
        """
        Calculate the mean value of phi and adjust to make mean value 0.
        """
        from math import fabs
        import proteus.Norms as Norms
        if self.zeroMean:
            meanvalue = Norms.scalarDomainIntegral(self.model.q['dV'],
                                                   self.model.q[('u',0)],
                                                   self.model.mesh.nElements_owned)/self.model.mesh.volume
            self.model.q[('u',0)][:] = self.model.q[('u',0)] - meanvalue
            self.model.ebqe[('u',0)] -= meanvalue
            self.model.u[0].dof -= meanvalue

            newmeanvalue = Norms.scalarDomainIntegral(self.model.q['dV'],
                                                      self.model.q[('u',0)],
                                                      self.model.mesh.nElements_owned)
            assert fabs(newmeanvalue) < 1.0e-8, "new mean should be zero but is "+`newmeanvalue`
        # add post processing adjustments here if possible.  They have already be solved for by this point.

        copyInstructions = {}
        return copyInstructions
    def evaluate(self,t,c):
        """
        Evaluate the coefficients after getting the specified velocity and density
        """
        # time management
        dt = self.model.timeIntegration.dt

        # beta_0 coefficient scalar
        if self.bdf is int(1) or self.firstStep:
            b0 = 1.0/dt
        elif self.bdf is int(2):
            b0 = self.model.timeIntegration.alpha_bdf  # = beta_0

        # find minimal density value set it to be chi
        if self.densityModelIndex>0:
            rho = self.c_rho[c[('m',0)].shape]
        else:
            rho = [self.chiValue] # just give it the self.chiValue so that test passes as we assume user has given correct chiValue in this case.

        # Extract minimum of density and compare to given chiValue.
        # Ideally, the density model is maximum preserving and will not drop values
        # below this but if it does we want to know.  Even still we will
        # use the user given value of chi.
        chi = np.min(rho)
        if self.chiValue < chi:  # raise warning but do not stop
            log("*** warning: minimum of density = %1.3e is below physical limit chiValue = %1.3e. ***" %(chi, self.chiValue),  level=1)
        chi = self.chiValue

        # extract velocity components   ** notice that the ('velocity',0) field corresponds to this model so is not available **  (why not ? srp July 11, 2015)
        if self.velocityFunction != None:
            u = self.velocityFunction(c['x'],t)[...,0]
            v = self.velocityFunction(c['x'],t)[...,1]
        else:
            u = self.c_u[c[('u',0)].shape]
            v = self.c_v[c[('u',0)].shape]

        # set coefficients  -div (grad phi) + chi b0 div (u) = 0
        c[('f',0)][...,0] = chi*b0*u
        c[('f',0)][...,1] = chi*b0*v
        c[('df',0,0)][...,0] = 0.0
        c[('df',0,0)][...,1] = 0.0
        c[('a',0,0)][...,0] = 1.0 # -\grad v :   tensor  [ 1.0  0;  0  1.0] ordered [0 1; 2 3]  in our
        c[('a',0,0)][...,1] = 1.0 # -\grad v :       new diagonal notation from sDInfo above is [0 .; . 1] -> [0; 1]
        c[('da',0,0,0)][...,0] = 0.0 # -(da/d vi)_0   # could leave these off since it is 0
        c[('da',0,0,0)][...,1] = 0.0 # -(da/d vi)_1   # could leave these off since it is 0



class Pressure2D(TransportCoefficients.TC_base):
    r"""
    The coefficients for pressure solution

    Update is given by

    .. math::

       p^{k+1} - p^{k} - phi^{k+1} + \nabla\cdot(\mu \mathbf{u}^{k+1}) = 0
    """
    def __init__(self,
                 bdf=1,
                 mu=1.0,
                 currentModelIndex=3,
                 velocityModelIndex=-1,
                 velocityFunction=None,
                 useVelocityComponents=True,
                 pressureIncrementModelIndex=-1,
                 pressureIncrementFunction=None,
                 useRotationalModel=True,
                 chiValue=1.0):
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
                                               reaction = {0:{0:'linear'}}, #  = p - p_last - phi
                                               advection = {0:{0:'constant'}})#, # div  (\mu velocity)
        self.bdf=int(bdf)
        self.mu=mu
        self.currentModelIndex = currentModelIndex
        self.velocityModelIndex = velocityModelIndex
        self.velocityFunction = velocityFunction
        self.useVelocityComponents = useVelocityComponents
        self.pressureIncrementModelIndex = pressureIncrementModelIndex
        self.pressureIncrementFunction = pressureIncrementFunction
        self.useRotationalModel = useRotationalModel
        self.chiValue = chiValue
        if self.useRotationalModel:
            self.c_u = {}
            self.c_v = {}
            self.c_velocity = {}
        self.c_phi = {}
        self.firstStep = True # manipulated in preStep()

    def attachModels(self,modelList):
        """
        Attach the model for velocity

        We are implementing the post processing in the pressureIncrement model which is
        essentially the divergence free velocity equation.  The velocity
        is then extracted from the pressureIncrement Model as ('velocity',0).  In order to
        get a physical velocity, we must then scale it by the constants dt \beta_0 since the pressure
        equation is  -div(  grad\phi - chi \beta_0 [u v] ) = 0  so that the flux F has local integrals matching chi \beta_0 [u v]
        and hopefully has locally divergence free velocity matching chi\beta_0 [u v].  Thus the scaling by 1/(\beta_0 chi)
        to get physical velocity.

        In pressureincrement_n.py, the following could be set.  we recommend the 'pwl-bdm' as the best
        for this current situation:

        conservativeFlux = {0:'point-eval'}  - will return computed velocities without change
                                               (since there is no diffusion in eqn (2) )
        conservativeFlux = {0:'pwl-bdm'}     - will return velocities projected onto the bdm space (CG
                                               Taylor-Hood enriched with DG pw linears on each element)
        conservativeFlux = {0:'pwl-bdm-opt'} - same as pwl-bdm but optimized in a special way to be more
                                               effective.  any additional comments ?

        """
        self.model = modelList[self.currentModelIndex] # current model
        self.model.points_quadrature.add(('u_last',0))
        self.model.points_elementBoundaryQuadrature.add(('u_last',0))
        self.model.numericalFlux.ebqe[('u_last',0)] = self.model.ebqe[('u_last',0)]  # why do we need this line? srp july 15, 2015
        if self.useRotationalModel and not self.useVelocityComponents and self.pressureIncrementModelIndex >= 0:
            assert self.pressureIncrementModelIndex < len(modelList), \
                "pressure increment model index out of range 0," + repr(len(modelList))
            self.pressureIncrementModel = modelList[self.pressureIncrementModelIndex]
            if ('velocity',0) in self.pressureIncrementModel.q:
                vel = self.pressureIncrementModel.q[('velocity',0)]
                self.c_velocity[vel.shape] = vel
            if ('velocity',0) in self.pressureIncrementModel.ebq:
                vel = self.pressureIncrementModel.ebq[('velocity',0)]
                self.c_velocity[vel.shape] = vel
            if ('velocity',0) in self.pressureIncrementModel.ebqe:
                vel = self.pressureIncrementModel.ebqe[('velocity',0)]
                self.c_velocity[vel.shape] = vel
            if ('velocity',0) in self.pressureIncrementModel.ebq_global:
                vel = self.pressureIncrementModel.ebq_global[('velocity',0)]
                self.c_velocity[vel.shape] = vel
        elif self.useRotationalModel and self.useVelocityComponents and self.velocityModelIndex >= 0:
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
    def initializeMesh(self,mesh):
        """
        Give the TC object access to the mesh for any mesh-dependent information.
        """
        pass
    def initializeElementQuadrature(self,t,cq):
        """
        Give the TC object access to the element quadrature storage
        """
        for ci in range(self.nc):
            cq[('u_last',ci)] = deepcopy(cq[('u',ci)])
            # cq[('grad(u)_last',ci)] = deepcopy(cq[('grad(u)',ci)])
    def initializeElementBoundaryQuadrature(self,t,cebq,cebq_global):
        """
        Give the TC object access to the element boundary quadrature storage
        """
        # for ci in range(self.nc):
        #     cebq[('u_last',ci)] = deepcopy(cebq[('u',ci)])
        #     cebq[('grad(u)_last',ci)] = deepcopy(cebq[('grad(u)',ci)])
        #     cebq_global[('u_last',ci)] = deepcopy(cebq_global[('u',ci)])
        #     cebq_global[('grad(u)_last',ci)] = deepcopy(cebq_global[('grad(u)',ci)])
        pass
    def initializeGlobalExteriorElementBoundaryQuadrature(self,t,cebqe):
        """
        Give the TC object access to the exterior element boundary quadrature storage
        """
        for ci in range(self.nc):
            cebqe[('u_last',ci)] = deepcopy(cebqe[('u',ci)])
            # cebqe[('grad(u)_last',ci)] = deepcopy(cebqe[('grad(u)',ci)])
    def initializeGeneralizedInterpolationPointQuadrature(self,t,cip):
        """
        Give the TC object access to the generalized interpolation point storage. These points are used  to project nonlinear potentials (phi).
        """
        pass
    def preStep(self,t,firstStep=False):
        """
        Give the TC object an opportunity to modify itself before the time step.
        """
        self.firstStep = firstStep

        for ci in range(self.nc):
            # don't need to store pressure_lastlast at all since it is never needed in algorithm

            self.model.q[('u_last',ci)][:] = self.model.q[('u',ci)]
            # self.model.ebq[('u_last',ci)][:] = self.model.ebq[('u',ci)]
            self.model.ebqe[('u_last',ci)][:] = self.model.ebqe[('u',ci)]
            # self.model.ebq_global[('u_last',ci)][:] = self.model.ebq_global[('u',ci)]

            # don't need grad(u)_last either since it will never be used.
            # when we do need grad_pressure^k, we are on velocity model on the next time
            # step so that 'grad(u)' is grad_pressure^k.


        copyInstructions = {}
        return copyInstructions
    def evaluate(self,t,c):
        """
        Evaluate the coefficients after getting the specified velocity and density
        """

        # current and previous pressure values
        p = c[('u',0)]
        p_last = c[('u_last',0)]

        # extract pressure increment
        if self.velocityFunction != None:
            phi = self.pressureIncrementFunction(c['x'],t)
        else:
            phi = self.c_phi[c[('u',0)].shape]

        if self.useRotationalModel:
            # extract scaling terms for post processed velocity (see pressureIncrement model for description)
            if not self.useVelocityComponents:
                dt = self.model.timeIntegration.dt
                chi = self.chiValue
                # beta_0 coefficient scalar
                if self.bdf is int(1) or self.firstStep:
                    b0 = 1.0/dt
                elif self.bdf is int(2):
                    b0 = self.model.timeIntegration.alpha_bdf  # = beta_0
                    # dt_last = self.model.timeIntegration.dt_history[0] # note this only exists if we are using VBDF for Time integration
                    # r = dt/dt_last
                    # dtInv = 1.0/dt
                    # b0 = (1.0+2.0*r)/(1.0+r)*dtInv  # = self.model.timeIntegration.alpha_bdf  # = beta_0
                Invb0chi = 1.0/(chi*b0)

            # extract velocity components
            if self.velocityFunction != None:
                u = self.velocityFunction(c['x'],t)[...,0]
                v = self.velocityFunction(c['x'],t)[...,1]
            elif self.useVelocityComponents:
                u = self.c_u[c[('u',0)].shape]
                v = self.c_v[c[('u',0)].shape]
            else:
                u = Invb0chi*self.c_velocity[c[('f',0)].shape][...,0] # adjust post processed velocity to be physical units by mult by dt/chi
                v = Invb0chi*self.c_velocity[c[('f',0)].shape][...,1] # adjust post processed velocity to be physical units by mult by dt/chi

        # set coefficients   p - p_last - phi + div (mu u) = 0
        #G&S11,p941,remark 5.5
        if self.useRotationalModel:
            c[('f',0)][...,0] = self.mu*u
            c[('f',0)][...,1] = self.mu*v
            c[('df',0,0)][...,0] = 0.0
            c[('df',0,0)][...,1] = 0.0
        else:
            c[('f',0)][...,0] = 0.0
            c[('f',0)][...,1] = 0.0
            c[('df',0,0)][...,0] = 0.0
            c[('df',0,0)][...,1] = 0.0
        #G&S11,p92, eq 3.10
        c[('r',0)][:] = p - p_last - phi
        c[('dr',0,0)][:] = 1.0
