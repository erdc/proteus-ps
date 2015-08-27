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



class HistoryManipulation:
    """
        Encapsulate the manipulation of time history storage.  This is called in
        the first model AttachModels() and then is run in the preStep at the
        beginning of each new time stage.

    """
    def __init__(self, modelList=None, bdf=1, useNumericalFluxEbqe=False):
        self.modelList = modelList
        self.bdf = int(bdf)
        self.useNumericalFluxEbqe = useNumericalFluxEbqe

    def initializeHistory(self):
        """
            Initialize the solutions before first step so that we will have

            u = solution at time t^{1}
            u_last = solution at time t^{0}
            u_lastlast = solution at time t^{0}
        """
        for model in self.modelList:
            for transfer_from_c, transfer_to_c in zip([model.q, model.numericalFlux.ebqe, model.numericalFlux.ebqe if self.useNumericalFluxEbqe else model.ebqe],
                                                      [model.q, model.numericalFlux.ebqe, model.ebqe]):
                for ci in range(model.nc):
                    if self.bdf is int(2):
                        transfer_to_c[('u_lastlast',ci)][:] = transfer_from_c[('u',ci)]
                        transfer_to_c[('grad(u)_lastlast',ci)][:] = transfer_from_c[('grad(u)',ci)]
                    transfer_to_c[('u_last',ci)][:] = transfer_from_c[('u',ci)]
                    transfer_to_c[('grad(u)_last',ci)][:] = transfer_from_c[('grad(u)',ci)]


            for transfer_from_c, transfer_to_c in zip([model.q, model.ebqe],
                                                      [model.q, model.ebqe]):
                if ('velocity_last',0) in transfer_to_c:
                    if self.bdf is int(2):
                        transfer_to_c[('velocity_lastlast',0)][:] = transfer_from_c[('velocity',0)]
                    transfer_to_c[('velocity_last',0)][:] = transfer_from_c[('velocity',0)]

    def updateHistory(self):
        """
            Transfer the solutions backward so that they represent the previous
            time step values.  After this call on time t^{n}, we have:

              u = solution at time t^{n}
              u_last = solution at time t^{n-1}
              u_lastlast = solution at time t^{n-2}
        """
        for model in self.modelList:
            for transfer_from_c, transfer_to_c in zip([model.q, model.numericalFlux.ebqe, model.numericalFlux.ebqe if self.useNumericalFluxEbqe else model.ebqe],
                                                      [model.q, model.numericalFlux.ebqe, model.ebqe]):
                for ci in range(model.nc):
                    if self.bdf is int(2):
                        transfer_to_c[('u_lastlast',ci)][:] = transfer_from_c[('u_last',ci)]
                        transfer_to_c[('grad(u)_lastlast',ci)][:] = transfer_from_c[('grad(u)_last',ci)]
                    transfer_to_c[('u_last',ci)][:] = transfer_from_c[('u',ci)]
                    transfer_to_c[('grad(u)_last',ci)][:] = transfer_from_c[('grad(u)',ci)]

            for transfer_from_c, transfer_to_c in zip([model.q, model.ebqe],
                                                      [model.q, model.ebqe]):
                if ('velocity_last',0) in transfer_to_c:
                    if self.bdf is int(2):
                        transfer_to_c[('velocity_lastlast',0)][:] = transfer_from_c[('velocity_last',0)]
                    transfer_to_c[('velocity_last',0)][:] = transfer_from_c[('velocity',0)]


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
                 densityFunction=None,
                 velocityModelIndex=-1,
                 velocityFunction=None,
                 divVelocityFunction=None,
                 useVelocityComponents=True,
                 chiValue=1.0,  # only needed for the scaling adjustment in case of post processed velocity
                 pressureIncrementModelIndex=-1,
                 useStabilityTerms=False,
                 setFirstTimeStepValues=True,
                 useNumericalFluxEbqe=False):
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
                                               diffusion = {0:{0:{0:'constant'}}},
                                               potential = {0:{0:'u'}},
                                               reaction = {0:{0:'linear'}}) # for the stability term
        self.bdf=int(bdf)
        self.currentModelIndex = currentModelIndex
        self.densityFunction = densityFunction
        self.velocityModelIndex = velocityModelIndex
        self.velocityFunction = velocityFunction
        self.divVelocityFunction = divVelocityFunction
        self.useVelocityComponents = useVelocityComponents
        self.chiValue = chiValue
        self.pressureIncrementModelIndex=pressureIncrementModelIndex
        self.c_u_last = {}
        self.c_v_last = {}
        self.c_u_lastlast = {}
        self.c_v_lastlast = {}
        self.c_velocity_last = {}
        self.c_velocity_lastlast = {}
        self.c_grad_u_last = {}
        self.c_grad_v_last = {}
        self.c_grad_u_lastlast = {}
        self.c_grad_v_lastlast = {}
        self.useStabilityTerms = useStabilityTerms
        self.firstStep = True # manipulated in preStep()
        self.setFirstTimeStepValues = setFirstTimeStepValues
        self.useNumericalFluxEbqe = useNumericalFluxEbqe

    def attachModels(self,modelList):
        """
        Attach the model for velocity

        We are implementing the post processing in the pressureIncrement model which is
        essentially the divergence free velocity equation.  The velocity
        is then extracted from the pressureIncrement Model as ('velocity',0).  In order to
        get a physical velocity, we must then scale it by the constants dt/chi*beta_0  since the pressure
        equation is  -div(  grad\phi - beta_0*chi/dt [u v] ) = 0  so that the flux F has local integrals matching beta_0*chi/dt [u v]
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

        #Initialize the HistoryManipulation class for all of the models
        self.HistoryManipulation = HistoryManipulation(modelList=modelList[0:4],
                                                       bdf=self.bdf,
                                                       useNumericalFluxEbqe=self.useNumericalFluxEbqe)

        # densiy model work
        self.model = modelList[self.currentModelIndex] # current model
        for ci in range(self.nc):
            self.model.points_quadrature.add(('u_last',ci))
            self.model.points_elementBoundaryQuadrature.add(('u_last',ci))
            self.model.numericalFlux.ebqe[('u_last',ci)]=deepcopy(self.model.ebqe[('u_last',ci)])
            self.model.vectors_quadrature.add(('grad(u)_last',ci))
            self.model.vectors_elementBoundaryQuadrature.add(('grad(u)_last',ci))
            self.model.numericalFlux.ebqe[('grad(u)_last',ci)]=deepcopy(self.model.ebqe[('grad(u)_last',ci)])
            if self.bdf is int(2):
                self.model.points_quadrature.add(('u_lastlast',ci))
                self.model.points_elementBoundaryQuadrature.add(('u_lastlast',ci))
                self.model.numericalFlux.ebqe[('u_lastlast',ci)]=deepcopy(self.model.ebqe[('u_lastlast',ci)])
                self.model.vectors_quadrature.add(('grad(u)_lastlast',ci))
                self.model.vectors_elementBoundaryQuadrature.add(('grad(u)_lastlast',ci))
                self.model.numericalFlux.ebqe[('grad(u)_lastlast',ci)]=deepcopy(self.model.ebqe[('grad(u)_lastlast',ci)])

        if (not self.useVelocityComponents and self.velocityModelIndex >=0
            and self.pressureIncrementModelIndex >= 0 and self.velocityFunction is None):
            assert self.pressureIncrementModelIndex < len(modelList), \
                "velocity model index out of  range 0," + repr(len(modelList))
            assert self.pressureIncrementModelIndex < len(modelList), \
                "pressureIncrement model index out of  range 0," + repr(len(modelList))

            self.velocityModel = modelList[self.velocityModelIndex]
            self.pressureIncrementModel = modelList[self.pressureIncrementModelIndex]
            if ('velocity',0) in self.pressureIncrementModel.q:
                vel_last = self.pressureIncrementModel.q[('velocity_last',0)]
                self.c_velocity_last[vel_last.shape] = vel_last
                if self.bdf is int(2):
                    vel_lastlast = self.pressureIncrementModel.q[('velocity_lastlast',0)]
                    self.c_velocity_lastlast[vel_last.shape] = vel_last
                if self.useStabilityTerms:
                    grad_u_last = self.velocityModel.q[('grad(u)',0)].copy()
                    grad_u_last[:] = 0.0 #hack--just used in calculating div since post processed velocity is div free, set = 0
                    grad_v_last = self.velocityModel.q[('grad(u)',1)].copy()
                    grad_v_last[:] = 0.0 #hack--just used in calculating div since pp velocity is div free, set = 0
                    self.c_grad_u_last[grad_u_last.shape] = grad_u_last
                    self.c_grad_v_last[grad_v_last.shape] = grad_v_last
                if self.bdf is int(2):
                    if self.useStabilityTerms:
                        grad_u_lastlast = self.velocityModel.q[('grad(u)_lastlast',0)].copy()
                        grad_u_lastlast[:] = 0.0
                        grad_v_lastlast = self.velocityModel.q[('grad(u)_lastlast',1)].copy()
                        grad_v_lastlast[:] = 0.0
                        self.c_grad_u_lastlast[grad_u_lastlast.shape] = grad_u_lastlast
                        self.c_grad_v_lastlast[grad_v_lastlast.shape] = grad_v_lastlast
            if ('velocity',0) in self.pressureIncrementModel.ebq:
                vel_last = self.pressureIncrementModel.ebq[('velocity_last',0)]
                self.c_velocity_last[vel_last.shape] = vel_last
                if self.bdf is int(2):
                    vel_lastlast = self.pressureIncrementModel.ebq[('velocity_lastlast',0)]
                    self.c_velocity_lastlast[vel_last.shape] = vel_last
                if self.useStabilityTerms:
                    grad_u_last = self.velocityModel.ebq[('grad(u)',0)].copy()
                    grad_u_last[:] = 0.0
                    grad_v_last = self.velocityModel.ebq[('grad(u)',1)].copy()
                    grad_v_last[:] = 0.0
                    self.c_grad_u_last[grad_u_last.shape] = grad_u_last
                    self.c_grad_v_last[grad_v_last.shape] = grad_v_last
                if self.bdf is int(2):
                    if self.useStabilityTerms:
                        grad_u_lastlast = self.velocityModel.ebq[('grad(u)_lastlast',0)].copy()
                        grad_u_lastlast[:] = 0.0
                        grad_v_lastlast = self.velocityModel.ebq[('grad(u)_lastlast',1)].copy()
                        grad_v_lastlast[:] = 0.0
                        self.c_grad_u_lastlast[grad_u_lastlast.shape] = grad_u_lastlast
                        self.c_grad_v_lastlast[grad_v_lastlast.shape] = grad_v_lastlast
            if ('velocity',0) in self.pressureIncrementModel.ebqe:
                vel_last = self.pressureIncrementModel.ebqe[('velocity_last',0)]
                self.c_velocity_last[vel_last.shape] = vel_last
                if self.bdf is int(2):
                    vel_lastlast = self.pressureIncrementModel.ebqe[('velocity_lastlast',0)]
                    self.c_velocity_lastlast[vel_last.shape] = vel_last
                if self.useStabilityTerms:
                    grad_u_last = self.velocityModel.ebqe[('grad(u)',0)].copy()
                    grad_u_last[:] = 0.0
                    grad_v_last = self.velocityModel.ebqe[('grad(u)',1)].copy()
                    grad_v_last[:] = 0.0
                    self.c_grad_u_last[grad_u_last.shape] = grad_u_last
                    self.c_grad_v_last[grad_v_last.shape] = grad_v_last
                if self.bdf is int(2):
                    if self.useStabilityTerms:
                        grad_u_lastlast = self.velocityModel.ebqe[('grad(u)_lastlast',0)].copy()
                        grad_u_lastlast[:] = 0.0
                        grad_v_lastlast = self.velocityModel.ebqe[('grad(u)_lastlast',1)].copy()
                        grad_v_lastlast[:] = 0.0
                        self.c_grad_u_lastlast[grad_u_lastlast.shape] = grad_u_lastlast
                        self.c_grad_v_lastlast[grad_v_lastlast.shape] = grad_v_lastlast
        elif (self.useVelocityComponents and self.velocityModelIndex >= 0 and
                self.velocityFunction is None):
            assert self.velocityModelIndex < len(modelList), \
                "velocity model index out of  range 0," + repr(len(modelList))
            self.velocityModel = modelList[self.velocityModelIndex]
            if ('u',0) in self.velocityModel.q:
                u_last = self.velocityModel.q[('u_last',0)]
                v_last = self.velocityModel.q[('u_last',1)]
                self.c_u_last[u_last.shape] = u_last
                self.c_v_last[v_last.shape] = v_last
                if self.useStabilityTerms:
                    grad_u_last = self.velocityModel.q[('grad(u)_last',0)]
                    grad_v_last = self.velocityModel.q[('grad(u)_last',1)]
                    self.c_grad_u_last[grad_u_last.shape] = grad_u_last
                    self.c_grad_v_last[grad_v_last.shape] = grad_v_last
                if self.bdf is int(2):
                    u_lastlast = self.velocityModel.q[('u_lastlast',0)]
                    v_lastlast = self.velocityModel.q[('u_lastlast',1)]
                    self.c_u_lastlast[u_lastlast.shape] = u_lastlast
                    self.c_v_lastlast[v_lastlast.shape] = v_lastlast
                    if self.useStabilityTerms:
                        grad_u_lastlast = self.velocityModel.q[('grad(u)_lastlast',0)]
                        grad_v_lastlast = self.velocityModel.q[('grad(u)_lastlast',1)]
                        self.c_grad_u_lastlast[grad_u_lastlast.shape] = grad_u_lastlast
                        self.c_grad_v_lastlast[grad_v_lastlast.shape] = grad_v_lastlast
            if ('u',0) in self.velocityModel.ebq:
                u_last = self.velocityModel.ebq[('u_last',0)]
                v_last = self.velocityModel.ebq[('u_last',1)]
                self.c_u_last[u_last.shape] = u_last
                self.c_v_last[v_last.shape] = v_last
                if self.useStabilityTerms:
                    grad_u_last = self.velocityModel.ebq[('grad(u)_last',0)]
                    grad_v_last = self.velocityModel.ebq[('grad(u)_last',1)]
                    self.c_grad_u_last[grad_u_last.shape] = grad_u_last
                    self.c_grad_v_last[grad_v_last.shape] = grad_v_last
                if self.bdf is int(2):
                    u_lastlast = self.velocityModel.ebq[('u_lastlast',0)]
                    v_lastlast = self.velocityModel.ebq[('u_lastlast',1)]
                    self.c_u_lastlast[u_lastlast.shape] = u_lastlast
                    self.c_v_lastlast[v_lastlast.shape] = v_lastlast
                    if self.useStabilityTerms:
                        grad_u_lastlast = self.velocityModel.ebq[('grad(u)_lastlast',0)]
                        grad_v_lastlast = self.velocityModel.ebq[('grad(u)_lastlast',1)]
                        self.c_grad_u_lastlast[grad_u_lastlast.shape] = grad_u_lastlast
                        self.c_grad_v_lastlast[grad_v_lastlast.shape] = grad_v_lastlast
            if ('u',0) in self.velocityModel.ebqe:
                u_last = self.velocityModel.ebqe[('u_last',0)]
                v_last = self.velocityModel.ebqe[('u_last',1)]
                self.c_u_last[u_last.shape] = u_last
                self.c_v_last[v_last.shape] = v_last
                if self.useStabilityTerms:
                    grad_u_last = self.velocityModel.ebqe[('grad(u)_last',0)]
                    grad_v_last = self.velocityModel.ebqe[('grad(u)_last',1)]
                    self.c_grad_u_last[grad_u_last.shape] = grad_u_last
                    self.c_grad_v_last[grad_v_last.shape] = grad_v_last
                if self.bdf is int(2):
                    u_lastlast = self.velocityModel.ebqe[('u_lastlast',0)]
                    v_lastlast = self.velocityModel.ebqe[('u_lastlast',1)]
                    self.c_u_lastlast[u_lastlast.shape] = u_lastlast
                    self.c_v_lastlast[v_lastlast.shape] = v_lastlast
                    if self.useStabilityTerms:
                        grad_u_lastlast = self.velocityModel.ebqe[('grad(u)_lastlast',0)]
                        grad_v_lastlast = self.velocityModel.ebqe[('grad(u)_lastlast',1)]
                        self.c_grad_u_lastlast[grad_u_lastlast.shape] = grad_u_lastlast
                        self.c_grad_v_lastlast[grad_v_lastlast.shape] = grad_v_lastlast
            if ('u',0) in self.velocityModel.ebq_global:
                u_last = self.velocityModel.ebq_global[('u_last',0)]
                v_last = self.velocityModel.ebq_global[('u_last',1)]
                self.c_u_last[u_last.shape] = u_last
                self.c_v_last[v_last.shape] = v_last
                if self.useStabilityTerms:
                    grad_u_last = self.velocityModel.ebq_global[('grad(u)_last',0)]
                    grad_v_last = self.velocityModel.ebq_global[('grad(u)_last',1)]
                    self.c_grad_u_last[grad_u_last.shape] = grad_u_last
                    self.c_grad_v_last[grad_v_last.shape] = grad_v_last
                if self.bdf is int(2):
                    u_lastlast = self.velocityModel.ebq_global[('u_lastlast',0)]
                    v_lastlast = self.velocityModel.ebq_global[('u_lastlast',1)]
                    self.c_u_lastlast[u_lastlast.shape] = u_lastlast
                    self.c_v_lastlast[v_lastlast.shape] = v_lastlast
                    if self.useStabilityTerms:
                        grad_u_lastlast = self.velocityModel.ebq_global[('grad(u)_lastlast',0)]
                        grad_v_lastlast = self.velocityModel.ebq_global[('grad(u)_lastlast',1)]
                        self.c_grad_u_lastlast[grad_u_lastlast.shape] = grad_u_lastlast
                        self.c_grad_v_lastlast[grad_v_lastlast.shape] = grad_v_lastlast
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
                cq[('grad(u)_lastlast',ci)] = deepcopy(cq[('grad(u)_last',ci)])
    def initializeElementBoundaryQuadrature(self,t,cebq,cebq_global):
        """
        Give the TC object access to the element boundary quadrature storage
        """
        for ci in range(self.nc):
            cebq[('u_last',ci)] = deepcopy(cebq[('u',ci)])
            cebq[('grad(u)_last',ci)] = deepcopy(cebq[('grad(u)',ci)])
            if self.bdf is int(2):
                cebq[('u_lastlast',ci)] = deepcopy(cebq[('u',ci)])
                cebq[('grad(u)_lastlast',ci)] = deepcopy(cebq[('grad(u)_last',ci)])
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
        self.firstStep = firstStep # save for in evaluate

        # Transfer the solutions to the new time step representation
        if self.firstStep:
            self.HistoryManipulation.initializeHistory()
        else:
            self.HistoryManipulation.updateHistory()

        copyInstructions = {}
        return copyInstructions
    def postStep(self,t,firstStep=False):
        """
        Give the TC object an opportunity to modify itself before the time step.
        """
        if (self.setFirstTimeStepValues and firstStep and t>0):
            for eN in range(self.model.mesh.nElements_global):
                for j in self.model.u[0].femSpace.referenceFiniteElement.localFunctionSpace.range_dim:
                    jg = self.model.u[0].femSpace.dofMap.l2g[eN,j]
                    x = self.model.u[0].femSpace.dofMap.lagrangeNodesArray[jg]
                    self.model.u[0].dof[jg]=self.densityFunction(x,t)
            self.model.calculateSolutionAtQuadrature()
            self.evaluate(t,self.model.q)
            self.evaluate(t,self.model.ebqe)
            self.model.timeIntegration.calculateElementCoefficients(self.model.q)


        copyInstructions = {}
        return copyInstructions
    def evaluate(self,t,c):
        """
        Evaluate the coefficients after getting the specified velocity
        """

        # precompute the shapes to extract things we need from self.c_name[] dictionaries
        u_shape = c[('u',0)].shape
        grad_shape = c[('f',0)].shape

        rho = c[('u',0)]
        # if self.bdf is int(2) and not self.firstStep:
        #     grad_rho = c[('grad(u)',0)]

        # If we use pressureIncrementModel.q[('velocity',)] for our velocity, then we must
        # adjust it to be scaled properly by multiplying by dt/chi.  Then it is physical velocity
        # hopefully with divergence free properties.
        dt = self.model.timeIntegration.dt  # 0 = densityModelIndex
        if self.bdf is int(2):
            if self.firstStep:
                dt_last = dt
            else:
                dt_last = self.model.timeIntegration.dt_history[0] # note this only exists if we are using VBDF for Time integration

        tLast = self.model.timeIntegration.tLast
        if self.bdf is int(2) and not self.firstStep:
            tLastLast = tLast - dt_last

        # extract last velocity components
        if self.velocityFunction != None:
            u_last = self.velocityFunction(c['x'],tLast)[...,0]
            v_last = self.velocityFunction(c['x'],tLast)[...,1]
            if self.useStabilityTerms:
                div_vel_last = self.divVelocityFunction(c['x'],tLast)
        elif self.useVelocityComponents:
            u_last = self.c_u_last[u_shape]
            v_last = self.c_v_last[u_shape]
            if self.useStabilityTerms:
                div_vel_last = self.c_grad_u_last[grad_shape][...,0] + self.c_grad_v_last[grad_shape][...,1]
        else:
            u_last = Invb0chi*self.c_velocity_last[grad_shape][...,0]
            v_last = Invb0chi*self.c_velocity_last[grad_shape][...,1]
            if self.useStabilityTerms:
                div_vel_last = self.c_grad_u_last[grad_shape][...,0] + self.c_grad_v_last[grad_shape][...,1]

        # extract the lastlast time step values as needed
        if self.bdf is int(2) and not self.firstStep:
            if self.velocityFunction != None:
                u_lastlast = self.velocityFunction(c['x'],tLastLast)[...,0]
                v_lastlast = self.velocityFunction(c['x'],tLastLast)[...,1]
                if self.useStabilityTerms:
                    div_vel_lastlast = self.divVelocityFunction(c['x'],tLastLast)
            elif self.useVelocityComponents:
                u_lastlast = self.c_u_lastlast[u_shape]
                v_lastlast = self.c_v_lastlast[u_shape]
                if self.useStabilityTerms:
                    div_vel_lastlast = self.c_grad_u_lastlast[grad_shape][...,0] + self.c_grad_v_lastlast[grad_shape][...,1]
            else:
                u_lastlast = self.c_velocity_lastlast[grad_shape][...,0]
                v_lastlast = self.c_velocity_lastlast[grad_shape][...,1]
                if self.useStabilityTerms:
                    div_vel_lastlast = self.c_grad_u_lastlast[grad_shape][...,0] + self.c_grad_v_lastlast[grad_shape][...,1]

        # choose the velocity to be used for transport
        if self.bdf is int(1) or self.firstStep:
            # use first order extrapolation of velocity
            u_star = u_last
            v_star = v_last
            if self.useStabilityTerms:
                div_vel_star = div_vel_last
        elif self.bdf is int(2):
            # use second order extrapolation of velocity
            u_star = u_last + dt/dt_last*( u_last - u_lastlast )
            v_star = v_last + dt/dt_last*( v_last - v_lastlast )
            if self.useStabilityTerms:
                div_vel_star = div_vel_last + dt/dt_last*(div_vel_last - div_vel_lastlast )
        else:
            assert False, "Error: self.bdf = %f is not supported" %self.bdf

        #  rho_t + div( rho vel_star) - 0.5 rho div( vel_star ) = 0
        c[('m',0)][:] = rho
        c[('dm',0,0)][:] = 1.0
        c[('f',0)][...,0] = rho*u_star
        c[('f',0)][...,1] = rho*v_star
        c[('df',0,0)][...,0] = u_star
        c[('df',0,0)][...,1] = v_star
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
                 uFunction=None,
                 vFunction=None,
                 pressureIncrementModelIndex=-1,
                 pressureIncrementFunction=None,
                 pressureIncrementGradFunction=None,
                 pressureModelIndex=-1,
                 pressureFunction=None,
                 pressureGradFunction=None,
                 useStabilityTerms=False,
                 setFirstTimeStepValues=True,
                 useNonlinearAdvection=False,
                 usePressureExtrapolations=False,
                 useConservativePressureTerm=False,
                 useVelocityComponents=True):

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
                         hamiltonian = {eu:{ui:'nonlinear'},  # rho*(u u_x + v u_y)   convection term
                                        ev:{vi:'nonlinear'}}  # rho*(u v_x + v v_y)   convection term
                                       if useNonlinearAdvection else
                                       {eu:{ui:'linear'},  # rho*(u u_x + v u_y)   convection term
                                        ev:{vi:'linear'}}, # rho*(u v_x + v v_y)   convection term
                         advection = {eu:{ui:'constant'},  #  < grad p^{\#}, w > = - < p^#, div w >
                                      ev:{vi:'constant'}}  #  < grad p^{\#}, w > = - < p^#, div w >
                                      if useConservativePressureTerm else
                                      {},
                         diffusion = {eu:{ui:{ui:'constant'}},  # - \mu * \grad u
                                      ev:{vi:{vi:'constant'}}}, # - \mu * \grad v
                         potential = {eu:{ui:'u'},
                                      ev:{vi:'u'}}, # define the potential for the diffusion term to be the solution itself
                         reaction  = {eu:{ui:'constant'},  # -f1(x) (+ (d/dx p^\#))
                                      ev:{vi:'constant'}}  # -f2(x) (+ (d/dy p^\#))
                                     if not useStabilityTerms else
                                     {eu:{ui:'linear'},  # -f1(x) (+ (d/dx p^\#)) + (stability terms u extrapolated * u
                                      ev:{vi:'linear'}}  # -f2(x) (+ (d/dy p^\#)) + (stability terms v extrapolated) * v
                                     if not useNonlinearAdvection else
                                     {eu:{ui:'nonlinear',vi:'linear'},  # -f1(x) (+ (d/dx p^\#)) + (stability terms u,v) * u
                                      ev:{ui:'linear',vi:'nonlinear'}}, # -f2(x) (+ (d/dy p^\#)) + (stability terms u,v) * v
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
        self.pressureFunction = pressureFunction
        self.pressureGradFunction = pressureGradFunction
        self.pressureIncrementModelIndex = pressureIncrementModelIndex
        self.pressureIncrementFunction = pressureIncrementFunction
        self.pressureIncrementGradFunction = pressureIncrementGradFunction
        self.useStabilityTerms = useStabilityTerms
        self.c_rho = {} # density
        self.c_rho_last = {} # density of cached values
        self.c_p_last = {}  # pressure
        self.c_phi_last = {} # pressure increment phi
        self.c_vel_last = {} # post-processed velocity
        if self.bdf is int(2):
            self.c_rho_lastlast = {}
            self.c_p_lastlast = {}
            self.c_phi_lastlast = {}
        self.firstStep = True # manipulated in preStep()
        self.setFirstTimeStepValues=setFirstTimeStepValues
        self.uFunction = uFunction # for initialization on firstStep if switched on
        self.vFunction = vFunction # for initialization on firstStep if switched on
        self.useNonlinearAdvection=useNonlinearAdvection
        self.usePressureExtrapolations=usePressureExtrapolations
        self.useConservativePressureTerm=useConservativePressureTerm
        self.useVelocityComponents = useVelocityComponents

    def attachModels(self,modelList):
        """
        Attach the models for density, pressure increment and pressure
        """
        self.model = modelList[self.currentModelIndex] # current model
        for ci in range(self.nc):
            self.model.points_quadrature.add(('u_last',ci))
            self.model.points_elementBoundaryQuadrature.add(('u_last',ci))
            self.model.numericalFlux.ebqe[('u_last',ci)]=deepcopy(self.model.ebqe[('u_last',ci)])
            self.model.vectors_quadrature.add(('grad(u)_last',ci))
            self.model.vectors_elementBoundaryQuadrature.add(('grad(u)_last',ci))
            self.model.numericalFlux.ebqe[('grad(u)_last',ci)]=deepcopy(self.model.ebqe[('grad(u)_last',ci)])
            if self.bdf is int(2):
                self.model.points_quadrature.add(('u_lastlast',ci))
                self.model.points_elementBoundaryQuadrature.add(('u_lastlast',ci))
                self.model.numericalFlux.ebqe[('u_lastlast',ci)]=deepcopy(self.model.ebqe[('u_lastlast',ci)])
                self.model.vectors_quadrature.add(('grad(u)_lastlast',ci))
                self.model.vectors_elementBoundaryQuadrature.add(('grad(u)_lastlast',ci))
                self.model.numericalFlux.ebqe[('grad(u)_lastlast',ci)]=deepcopy(self.model.ebqe[('grad(u)_lastlast',ci)])
        if (self.densityModelIndex >= 0 and self.densityFunction is None):
            assert self.densityModelIndex < len(modelList), \
                "density model index out of range 0," + repr(len(modelList))
            self.densityModel = modelList[self.densityModelIndex]
            if ('u',0) in self.densityModel.q:
                rho = self.densityModel.q[('u',0)]
                self.c_rho[rho.shape] = rho
                rho_last = self.densityModel.q[('u_last',0)]
                self.c_rho_last[rho_last.shape] = rho_last
                if self.bdf is int(2):
                    rho_lastlast = self.densityModel.q[('u_lastlast',0)]
                    self.c_rho_lastlast[rho_lastlast.shape] = rho_lastlast
                if self.useStabilityTerms:
                    grad_rho = self.densityModel.q[('grad(u)',0)]
                    self.c_rho[grad_rho.shape] = grad_rho
            if ('u',0) in self.densityModel.ebq:
                rho = self.densityModel.ebq[('u',0)]
                self.c_rho[rho.shape] = rho
                rho_last = self.densityModel.ebq[('u_last',0)]
                self.c_rho_last[rho_last.shape] = rho_last
                if self.bdf is int(2):
                    rho_lastlast = self.densityModel.ebq[('u_lastlast',0)]
                    self.c_rho_lastlast[rho_lastlast.shape] = rho_lastlast
                if self.useStabilityTerms:
                    grad_rho = self.densityModel.ebq[('grad(u)',0)]
                    self.c_rho[grad_rho.shape] = grad_rho
            if ('u',0) in self.densityModel.ebqe:
                rho = self.densityModel.numericalFlux.ebqe[('u',0)]  # why use numericalFlux.ebqe?
                self.c_rho[rho.shape] = rho
                rho_last = self.densityModel.numericalFlux.ebqe[('u_last',0)]
                self.c_rho_last[rho_last.shape] = rho_last
                if self.bdf is int(2):
                    rho_lastlast = self.densityModel.numericalFlux.ebqe[('u_lastlast',0)]
                    self.c_rho_lastlast[rho_lastlast.shape] = rho_lastlast
                if self.useStabilityTerms:
                    grad_rho = self.densityModel.ebqe[('grad(u)',0)]
                    self.c_rho[grad_rho.shape] = grad_rho
            if ('u',0) in self.densityModel.ebq_global:
                rho = self.densityModel.ebq_global[('u',0)]
                self.c_rho[rho.shape] = rho
                rho_last = self.densityModel.ebq_global[('u_last',0)]
                self.c_rho_last[rho_last.shape] = rho_last
                if self.bdf is int(2):
                    rho_lastlast = self.densityModel.ebq_global[('u_lastlast',0)]
                    self.c_rho_lastlast[rho_lastlast.shape] = rho_lastlast
                if self.useStabilityTerms:
                    grad_rho = self.densityModel.ebq_global[('grad(u)',0)]
                    self.c_rho[grad_rho.shape] = grad_rho
        if (self.pressureIncrementModelIndex >= 0 and self.pressureIncrementGradFunction is None
             and not self.useConservativePressureTerm):
            assert self.pressureIncrementModelIndex < len(modelList), \
                "pressure increment model index out of range 0," + repr(len(modelList))
            self.pressureIncrementModel = modelList[self.pressureIncrementModelIndex]
            if ('u',0) in self.pressureIncrementModel.q:
                grad_phi_last = self.pressureIncrementModel.q[('grad(u)_last',0)]
                self.c_phi_last[grad_phi_last.shape] = grad_phi_last
                if self.bdf is int(2):
                    grad_phi_lastlast = self.pressureIncrementModel.q[('grad(u)_lastlast',0)]
                    self.c_phi_lastlast[grad_phi_lastlast.shape] = grad_phi_lastlast
            if ('u',0) in self.pressureIncrementModel.ebq:
                grad_phi_last = self.pressureIncrementModel.ebq[('grad(u)_last',0)]
                self.c_phi_last[grad_phi_last.shape] = grad_phi_last
                if self.bdf is int(2):
                    grad_phi_lastlast = self.pressureIncrementModel.ebq[('grad(u)_lastlast',0)]
                    self.c_phi_lastlast[grad_phi_lastlast.shape] = grad_phi_lastlast
            if ('u',0) in self.pressureIncrementModel.ebqe:
                grad_phi_last = self.pressureIncrementModel.ebqe[('grad(u)_last',0)]
                self.c_phi_last[grad_phi_last.shape] = grad_phi_last
                if self.bdf is int(2):
                    grad_phi_lastlast = self.pressureIncrementModel.ebqe[('grad(u)_lastlast',0)]
                    self.c_phi_lastlast[grad_phi_lastlast.shape] = grad_phi_lastlast
            if ('u',0) in self.pressureIncrementModel.ebq_global:
                grad_phi_last = self.pressureIncrementModel.ebq_global[('grad(u)_last',0)]
                self.c_phi_last[grad_phi_last.shape] = grad_phi_last
                if self.bdf is int(2):
                    grad_phi_lastlast = self.pressureIncrementModel.ebq_global[('grad(u)_lastlast',0)]
                    self.c_phi_lastlast[grad_phi_lastlast.shape] = grad_phi_lastlast
        if (self.pressureIncrementModelIndex >= 0 and self.pressureIncrementFunction is None
                and self.useConservativePressureTerm):
            assert self.pressureIncrementModelIndex < len(modelList), \
                "pressure increment model index out of range 0," + repr(len(modelList))
            self.pressureIncrementModel = modelList[self.pressureIncrementModelIndex]
            if ('u',0) in self.pressureIncrementModel.q:
                phi_last = self.pressureIncrementModel.q[('u_last',0)]
                self.c_phi_last[phi_last.shape] = phi_last
                if self.bdf is int(2):
                    phi_lastlast = self.pressureIncrementModel.q[('u_lastlast',0)]
                    self.c_phi_lastlast[phi_lastlast.shape] = phi_lastlast
            if ('u',0) in self.pressureIncrementModel.ebq:
                phi_last = self.pressureIncrementModel.ebq[('u_last',0)]
                self.c_phi_last[phi_last.shape] = phi_last
                if self.bdf is int(2):
                    phi_lastlast = self.pressureIncrementModel.ebq[('u_lastlast',0)]
                    self.c_phi_lastlast[phi_lastlast.shape] = phi_lastlast
            if ('u',0) in self.pressureIncrementModel.ebqe:
                phi_last = self.pressureIncrementModel.ebqe[('u_last',0)]
                self.c_phi_last[phi_last.shape] = phi_last
                if self.bdf is int(2):
                    phi_lastlast = self.pressureIncrementModel.ebqe[('u_lastlast',0)]
                    self.c_phi_lastlast[phi_lastlast.shape] = phi_lastlast
            if ('u',0) in self.pressureIncrementModel.ebq_global:
                phi_last = self.pressureIncrementModel.ebq_global[('u_last',0)]
                self.c_phi_last[phi_last.shape] = phi_last
                if self.bdf is int(2):
                    phi_lastlast = self.pressureIncrementModel.ebq_global[('u_lastlast',0)]
                    self.c_phi_lastlast[phi_lastlast.shape] = phi_lastlast
        if (self.pressureIncrementModelIndex >= 0 and self.pressureIncrementFunction is None
                and not self.useVelocityComponents):
            assert self.pressureIncrementModelIndex < len(modelList), \
                "pressure increment model index out of range 0," + repr(len(modelList))
            self.pressureIncrementModel = modelList[self.pressureIncrementModelIndex]
            if ('velocity',0) in self.pressureIncrementModel.q:
                vel_last = self.pressureIncrementModel.q[('velocity_last',0)]
                self.c_vel_last[vel_last.shape] = vel_last
                if self.bdf is int(2):
                    vel_lastlast = self.pressureIncrementModel.q[('velocity_lastlast',0)]
                    self.c_vel_lastlast[vel_lastlast.shape] = vel_lastlast
            if ('velocity',0) in self.pressureIncrementModel.ebq:
                vel_last = self.pressureIncrementModel.ebq[('velocity_last',0)]
                self.c_vel_last[vel_last.shape] = vel_last
                if self.bdf is int(2):
                    vel_lastlast = self.pressureIncrementModel.ebq[('velocity_lastlast',0)]
                    self.c_vel_lastlast[vel_lastlast.shape] = vel_lastlast
            if ('velocity',0) in self.pressureIncrementModel.ebqe:
                vel_last = self.pressureIncrementModel.ebqe[('velocity_last',0)]
                self.c_vel_last[vel_last.shape] = vel_last
                if self.bdf is int(2):
                    vel_lastlast = self.pressureIncrementModel.ebqe[('velocity_lastlast',0)]
                    self.c_vel_lastlast[vel_lastlast.shape] = vel_lastlast
            if ('velocity',0) in self.pressureIncrementModel.ebq_global:
                vel_last = self.pressureIncrementModel.ebq_global[('velocity_last',0)]
                self.c_vel_last[vel_last.shape] = vel_last
                if self.bdf is int(2):
                    vel_lastlast = self.pressureIncrementModel.ebq_global[('velocity_lastlast',0)]
                    self.c_vel_lastlast[vel_lastlast.shape] = vel_lastlast
        if (self.pressureModelIndex >= 0 and self.pressureGradFunction is None
                    and not self.useConservativePressureTerm):
            assert self.pressureModelIndex < len(modelList), \
                "pressure model index out of range 0," + repr(len(modelList))
            self.pressureModel = modelList[self.pressureModelIndex]
            if ('u',0) in self.pressureModel.q:
                grad_p_last = self.pressureModel.q[('grad(u)_last',0)]
                self.c_p_last[grad_p_last.shape] = grad_p_last
                if self.bdf is int(2):
                    grad_p_lastlast = self.pressureModel.q[('grad(u)_lastlast',0)]
                    self.c_p_lastlast[grad_p_lastlast.shape] = grad_p_lastlast
            if ('u',0) in self.pressureModel.ebq:
                grad_p_last = self.pressureModel.ebq[('grad(u)_last',0)]
                self.c_p_last[grad_p_last.shape] = grad_p_last
                if self.bdf is int(2):
                    grad_p_lastlast = self.pressureModel.ebq[('grad(u)_lastlast',0)]
                    self.c_p_lastlast[grad_p_lastlast.shape] = grad_p_lastlast
            if ('u',0) in self.pressureModel.ebqe:
                grad_p_last = self.pressureModel.ebqe[('grad(u)_last',0)]
                self.c_p_last[grad_p_last.shape] = grad_p_last
                if self.bdf is int(2):
                    grad_p_lastlast = self.pressureModel.ebqe[('grad(u)_lastlast',0)]
                    self.c_p_lastlast[grad_p_lastlast.shape] = grad_p_lastlast
            if ('u',0) in self.pressureModel.ebq_global:
                grad_p_last = self.pressureModel.ebq_global[('grad(u)_last',0)]
                self.c_p_last[grad_p_last.shape] = grad_p_last
                if self.bdf is int(2):
                    grad_p_lastlast = self.pressureModel.ebq_global[('grad(u)_lastlast',0)]
                    self.c_p_lastlast[grad_p_lastlast.shape] = grad_p_lastlast
        if (self.pressureModelIndex >= 0 and self.pressureFunction is None
                 and self.useConservativePressureTerm):
            assert self.pressureModelIndex < len(modelList), \
                "pressure model index out of range 0," + repr(len(modelList))
            self.pressureModel = modelList[self.pressureModelIndex]
            if ('u',0) in self.pressureModel.q:
                p_last = self.pressureModel.q[('u_last',0)]
                self.c_p_last[p_last.shape] = p_last
                if self.bdf is int(2):
                    p_lastlast = self.pressureModel.q[('u_lastlast',0)]
                    self.c_p_lastlast[p_lastlast.shape] = p_lastlast
            if ('u',0) in self.pressureModel.ebq:
                p_last = self.pressureModel.ebq[('u_last',0)]
                self.c_p_last[p_last.shape] = p_last
                if self.bdf is int(2):
                    p_lastlast = self.pressureModel.ebq[('u_lastlast',0)]
                    self.c_p_lastlast[p_lastlast.shape] = p_lastlast
            if ('u',0) in self.pressureModel.ebqe:
                p_last = self.pressureModel.ebqe[('u_last',0)]
                self.c_p_last[p_last.shape] = p_last
                if self.bdf is int(2):
                    p_lastlast = self.pressureModel.ebqe[('u_lastlast',0)]
                    self.c_p_lastlast[p_lastlast.shape] = p_lastlast
            if ('u',0) in self.pressureModel.ebq_global:
                p_last = self.pressureModel.ebq_global[('u_last',0)]
                self.c_p_last[p_last.shape] = p_last
                if self.bdf is int(2):
                    p_lastlast = self.pressureModel.ebq_global[('u_lastlast',0)]
                    self.c_p_lastlast[p_lastlast.shape] = p_lastlast
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
        for ci in range(self.nc):
            cebq[('u_last',ci)] = deepcopy(cebq[('u',ci)])
            cebq[('grad(u)_last',ci)] = deepcopy(cebq[('grad(u)',ci)])
            if self.bdf is int(2):
                cebq[('u_lastlast',ci)] = deepcopy(cebq[('u',ci)])
                cebq[('grad(u)_lastlast',ci)] = deepcopy(cebq[('grad(u)',ci)])
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

        copyInstructions = {}
        return copyInstructions
    def postStep(self,t,firstStep=False):
        """
        Give the TC object an opportunity to modify itself before the time step.
        """
        if (self.setFirstTimeStepValues and firstStep and t>0):
            for eN in range(self.model.mesh.nElements_global):
                for j in self.model.u[0].femSpace.referenceFiniteElement.localFunctionSpace.range_dim:
                    jg = self.model.u[0].femSpace.dofMap.l2g[eN,j]
                    x = self.model.u[0].femSpace.dofMap.lagrangeNodesArray[jg]
                    self.model.u[0].dof[jg]=self.uFunction(x,t)
                    self.model.u[1].dof[jg]=self.vFunction(x,t)
            # self.model.u[0].dof[:] = self.uFunction(self.model.mesh.nodeArray,t)
            # self.model.u[1].dof[:] = self.vFunction(self.model.mesh.nodeArray,t)
            self.model.calculateSolutionAtQuadrature()
            self.evaluate(t,self.model.q)
            self.evaluate(t,self.model.ebqe)
            self.model.timeIntegration.calculateElementCoefficients(self.model.q)

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
        eu=0; ev=1; # equation numbers  momentum u, momentum v
        ui=0; vi=1; # variable name ordering

        # precompute the shapes to extract things we need from self.c_name[] dictionaries
        u_shape = c[('u',0)].shape
        grad_shape = c[('grad(u)',0)].shape


        # time management
        dt = self.model.timeIntegration.dt  # 0 = velocityModelIndex
        if self.bdf is int(2):
            if not self.firstStep:
                dt_last = self.model.timeIntegration.dt_history[0] # note this only exists if we are using VBDF for Time integration
            else:
                dt_last = dt
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

        # previous velocity and grad velocity and divergence of velocity
        u_last = c[('u_last',ui)]
        v_last = c[('u_last',vi)]
        grad_u_last = c[('grad(u)_last',ui)]
        grad_v_last = c[('grad(u)_last',vi)]

        if self.bdf is int(2):
            u_lastlast = c[('u_lastlast',ui)]
            v_lastlast = c[('u_lastlast',vi)]
            grad_u_lastlast = c[('grad(u)_lastlast',ui)]
            grad_v_lastlast = c[('grad(u)_lastlast',vi)]

        if self.useStabilityTerms:
            div_vel_last = grad_u_last[...,xi] + grad_v_last[...,yi]
            if self.bdf is int(2):
                div_vel_lastlast = grad_u_lastlast[...,xi] + grad_v_lastlast[...,yi]

        # extract rho, rho_last, rho_lastlast and grad_rho as needed
        if self.densityFunction != None:
            rho = self.densityFunction(c['x'],t)
            rho_last =  self.densityFunction(c['x'],tLast)
            if self.useStabilityTerms:
                grad_rho = self.densityGradFunction(c['x'],t)
            if self.bdf is int(2) and not self.firstStep:
                rho_lastlast = self.densityFunction(c['x'],tLastLast)
        else:
            rho = self.c_rho[u_shape]
            rho_last = self.c_rho_last[u_shape]
            if self.useStabilityTerms:
                grad_rho = self.c_rho[grad_shape]
            if self.bdf is int(2) and not self.firstStep:
                rho_lastlast = self.c_rho_lastlast[u_shape]

        # take care of pressure term  < grad p^#, w>  = - <p^#, grad w >  in conservative or non conservative form
        if self.useConservativePressureTerm:
            # extract grad_p_last  p^{k}
            if self.pressureGradFunction != None:
                p_last = self.pressureFunction(c['x'],tLast)
            else:
                p_last = self.c_p_last[u_shape]

            # extract grad_phi_last phi^{k}
            if self.pressureIncrementGradFunction != None:
                phi_last = self.pressureIncrementFunction(c['x'],tLast)
            else:
                phi_last = self.c_phi_last[u_shape]

            # extract grad_phi_lastlast  phi^{k-1}
            if self.bdf is int(2) and not self.firstStep:
                if self.pressureFunction != None:
                    p_lastlast = self.pressureFunction(c['x'],tLastLast)
                else:
                    p_lastlast = self.c_p_lastlast[u_shape]

                if self.pressureIncrementFunction != None:
                    phi_lastlast = self.pressureIncrementFunction(c['x'],tLastLast)
                else:
                    phi_lastlast = self.c_phi_lastlast[u_shape]

            # order of pressure extrapolation first or second order (should be the same as in pressure model)
            if self.bdf is int(1) or self.firstStep or not self.usePressureExtrapolations:
                p_star = p_last
            elif self.bdf is int(2) and self.usePressureExtrapolations:
                p_star = p_last + dt/dt_last*(p_last - p_lastlast)

            # choose the density to use on the mass term,  bdf1 is rho_last,  bdf2 is current rho
            # as well as the other various element (not velocity) that differ between bdf1 and bdf2
            if self.bdf is int(1) or self.firstStep:
                p_sharp = p_star + phi_last
            elif self.bdf is int(2):
                p_sharp = p_star + b1/b0 * phi_last + b2/b0 *phi_lastlast

            # if the pressure Gradient function is given, then we should ignore the
            # adjustment given by pressure increment and just use grad_p_exact(t)
            if self.pressureFunction is not None:
                p_sharp = self.pressureFunction(c['x'],t)

        else:  # add them to the reaction term as non conservative elements
            # extract grad_p_last  p^{k}
            if self.pressureGradFunction != None:
                grad_p_last = self.pressureGradFunction(c['x'],tLast)
            else:
                grad_p_last = self.c_p_last[grad_shape]

            # extract grad_phi_last phi^{k}
            if self.pressureIncrementGradFunction != None:
                grad_phi_last = self.pressureIncrementGradFunction(c['x'],tLast)
            else:
                grad_phi_last = self.c_phi_last[grad_shape]

            # extract grad_phi_lastlast  phi^{k-1}
            if self.bdf is int(2) and not self.firstStep:
                if self.pressureGradFunction != None:
                    grad_p_lastlast = self.pressureGradFunction(c['x'],tLastLast)
                else:
                    grad_p_lastlast = self.c_p_lastlast[grad_shape]

                if self.pressureIncrementGradFunction != None:
                    grad_phi_lastlast = self.pressureIncrementGradFunction(c['x'],tLastLast)
                else:
                    grad_phi_lastlast = self.c_phi_lastlast[grad_shape]

            # order of pressure extrapolation first or second order (should be the same as in pressure model)
            if self.bdf is int(1) or self.firstStep or not self.usePressureExtrapolations:
                grad_p_star = grad_p_last
            elif self.bdf is int(2) and self.usePressureExtrapolations:
                grad_p_star = grad_p_last + dt/dt_last*(grad_p_last - grad_p_lastlast)

            # choose the density to use on the mass term,  bdf1 is rho_last,  bdf2 is current rho
            # as well as the other various element (not velocity) that differ between bdf1 and bdf2
            if self.bdf is int(1) or self.firstStep:
                grad_p_sharp = grad_p_star + grad_phi_last
            elif self.bdf is int(2):
                grad_p_sharp = grad_p_star + b1/b0 * grad_phi_last + b2/b0 *grad_phi_lastlast

            # if the pressure Gradient function is given, then we should ignore the
            # adjustment given by pressure increment and just use grad_p_exact(t)
            if self.pressureGradFunction is not None:
                grad_p_sharp = self.pressureGradFunction(c['x'],t)



        # choose the density to use on the mass term,  bdf1 is rho_last,  bdf2 is current rho
        # as well as the other various element (not velocity) that differ between bdf1 and bdf2
        if self.bdf is int(1) or self.firstStep:
            if self.useStabilityTerms:
                rho_sharp = rho_last
            else:
                rho_sharp = rho
            rho_t = (rho - rho_last)/dt # bdf1 time derivative
        elif self.bdf is int(2):
            rho_sharp = rho
            rho_t = b0*rho - b1*rho_last - b2*rho_lastlast #bdf2 time derivative  (see above for descriptions and definitions of b0 b1 and b2)

        if self.densityFunction is not None:
            rho_sharp = self.densityFunction(c['x'],t)

        # choose velocity advection
        if self.useNonlinearAdvection:
            # nonlinear advection but not div velocity for stability terms
            u_star = u
            v_star = v
        elif not self.useVelocityComponents:
            if self.bdf is int(1) or self.firstStep:
                u_star = self.c_vel_last[grad_shap][...,0]
                v_star = self.c_vel_last[grad_shap][...,1]
            elif self.bdf is int(2):
                u_star = self.c_vel_last[grad_shap][...,0] + dt/dt_last*( self.c_vel_last[grad_shap][...,0] - self.c_vel_lastlast[grad_shap][...,0] )
                v_star = self.c_vel_last[grad_shap][...,1] + dt/dt_last*( self.c_vel_last[grad_shap][...,1] - self.c_vel_lastlast[grad_shap][...,1] )
        else: # use extrapolation of velocity
            if self.bdf is int(1) or self.firstStep:
                # first order extrapolation
                u_star = u_last
                v_star = v_last
            elif self.bdf is int(2):
                # use second order extrapolation of velocity
                u_star = u_last + dt/dt_last*( u_last - u_lastlast )
                v_star = v_last + dt/dt_last*( v_last - v_lastlast )

        # when doing nonlinear, we will use extrapolated div_vel_star instead of making
        # it also nonlinear.  Thus the only nonlinear parts are in terms of u and v.
        if self.useStabilityTerms:
            if self.bdf is int(1) or self.firstStep:
                # first order extrapolation
                div_vel_star = div_vel_last
            elif self.bdf is int(2):
                # use second order extrapolation of div velocity
                div_vel_star = div_vel_last + dt/dt_last*( div_vel_last - div_vel_lastlast)
            # set the stability div(rho_vel)
            div_rho_vel_star = grad_rho[...,xi]*u_star + grad_rho[...,yi]*v_star + rho*div_vel_star
        #equation eu = 0
        # rho_sharp*u_t + rho(u_star u_x + v_star u_y ) + p_sharp_x - f1 + div(-mu grad(u))
        #            + 0.5( rho_t + rho_x u_star + rho_y v_star + rho div([u_star,v_star]) )u = 0
        c[('m',eu)][:] = rho_sharp*u    # d/dt ( rho_sharp * u) = d/dt (m_0)
        c[('dm',eu,ui)][:] = rho_sharp  # dm^0_du
        if self.useConservativePressureTerm:
            c[('r',eu)][:] = -self.f1ofx(c['x'][:],t)
            c[('dr',eu,ui)][:] = 0.0
            c[('f',eu)][...,xi] = p_sharp
            c[('f',eu)][...,yi] = 0.0      #< div (p I), w >, so that f = [p_sharp, 0] for eu component
            c[('df',eu,ui)][...,xi] = 0.0
            c[('df',eu,ui)][...,yi] = 0.0
        else:
            c[('r',eu)][:] = -self.f1ofx(c['x'][:],t) + grad_p_sharp[...,xi]
            c[('dr',eu,ui)][:] = 0.0
        if self.useStabilityTerms:
            c[('r',eu)][:] += 0.5*( rho_t + div_rho_vel_star )*u
            if self.useNonlinearAdvection: # 0.5*(rho_t + [u v] * grad_rho + rho*div_vel_star)*u
                c[('dr',eu,ui)][:] += 0.5*( rho_t + div_rho_vel_star ) + 0.5*grad_rho[...,xi]*u
                c[('dr',eu,vi)][:] += 0.5*grad_rho[...,yi]*u
            else:
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
        if self.useConservativePressureTerm:
            c[('r',ev)][:] = -self.f2ofx(c['x'][:],t)
            c[('dr',ev,vi)][:] = 0.0
            c[('f',ev)][...,xi] = 0.0         #< div (p I), w >, so that f = [0 p_sharp] for ev component
            c[('f',ev)][...,yi] = p_sharp
            c[('df',ev,vi)][...,xi] = 0.0
            c[('df',ev,vi)][...,yi] = 0.0
        else:
            c[('r',ev)][:] = -self.f2ofx(c['x'][:],t) + grad_p_sharp[...,yi]
            c[('dr',ev,vi)][:] = 0.0
        if self.useStabilityTerms:
            c[('r',ev)][:] += 0.5*( rho_t + div_rho_vel_star )*v
            if self.useNonlinearAdvection: # 0.5*(rho_t + [u v] * grad_rho + rho*div_vel_star)*v
                c[('dr',ev,ui)][:] += 0.5*grad_rho[...,xi]*v
                c[('dr',ev,vi)][:] += 0.5*( rho_t + div_rho_vel_star ) + 0.5*grad_rho[...,yi]*v
            else:
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
                 densityModelIndex=-1,
                 velocityModelIndex=-1,
                 velocityFunction=None,
                 pressureFunction=None,
                 zeroMean=False,
                 chiValue=1.0,
                 setFirstTimeStepValues=True):
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
        self.densityModelIndex = densityModelIndex
        self.velocityModelIndex = velocityModelIndex
        self.velocityFunction = velocityFunction
        self.pressureFunction = pressureFunction
        self.c_u = {}
        self.c_v = {}
        self.c_velocity = {}
        self.c_rho = {}
        self.firstStep = True # manipulated in preStep()
        self.zeroMean = zeroMean
        self.setFirstTimeStepValues=setFirstTimeStepValues
        self.calculateMeshVolume = True # manipulated in postStep()

    def attachModels(self,modelList):
        """
        Attach the model for velocity and density to PressureIncrement model
        """
        self.model = modelList[self.currentModelIndex] # current model
        # will need grad_phi and grad_phi_last for bdf2 algorithm in velocity model
        # representing times t^{k} and t^{k-1} respectively.  Since velocity is before
        # pressure increment, we haven't had a chance to move the phi's to be consistent
        # notation on the new time step so we must handle this inconsistency here.
        for ci in range(self.nc):
            self.model.points_quadrature.add(('u_last',ci))
            self.model.points_elementBoundaryQuadrature.add(('u_last',ci))
            self.model.numericalFlux.ebqe[('u_last',ci)]=deepcopy(self.model.ebqe[('u_last',ci)])
            self.model.vectors_quadrature.add(('grad(u)_last',ci))
            self.model.vectors_elementBoundaryQuadrature.add(('grad(u)_last',ci))
            self.model.numericalFlux.ebqe[('grad(u)_last',ci)]=deepcopy(self.model.ebqe[('grad(u)_last',ci)])
            if self.bdf is int(2):
                self.model.points_quadrature.add(('u_lastlast',ci))
                self.model.points_elementBoundaryQuadrature.add(('u_lastlast',ci))
                self.model.numericalFlux.ebqe[('u_lastlast',ci)]=deepcopy(self.model.ebqe[('u_lastlast',ci)])
                self.model.vectors_quadrature.add(('grad(u)_lastlast',ci))
                self.model.vectors_elementBoundaryQuadrature.add(('grad(u)_lastlast',ci))
                self.model.numericalFlux.ebqe[('grad(u)_lastlast',ci)]=deepcopy(self.model.ebqe[('grad(u)_lastlast',ci)])

        if (self.velocityModelIndex >= 0 and self.velocityFunction is None):
            assert self.velocityModelIndex < len(modelList), \
                "velocity model index out of  range 0," + repr(len(modelList))
            self.velocityModel = modelList[self.velocityModelIndex]
            if ('u',0) in self.velocityModel.q:
                u = self.velocityModel.q[('u',0)]
                v = self.velocityModel.q[('u',1)]
                self.c_u[u.shape] = u
                self.c_v[v.shape] = v
                self.model.q[('velocity',0)][...,0] = u
                self.model.q[('velocity',0)][...,1] = v
            if ('u',0) in self.velocityModel.ebq:
                u = self.velocityModel.ebq[('u',0)]
                v = self.velocityModel.ebq[('u',1)]
                self.c_u[u.shape] = u
                self.c_v[v.shape] = v
                self.model.ebq[('velocity',0)][...,0] = u
                self.model.ebq[('velocity',0)][...,1] = v
            if ('u',0) in self.velocityModel.ebqe:
                u = self.velocityModel.numericalFlux.ebqe[('u',0)]
                v = self.velocityModel.numericalFlux.ebqe[('u',1)]
                self.c_u[u.shape] = u
                self.c_v[v.shape] = v
                self.model.ebqe[('velocity',0)][...,0] = u
                self.model.ebqe[('velocity',0)][...,1] = v
            if ('u',0) in self.velocityModel.ebq_global:
                u = self.velocityModel.ebq_global[('u',0)]
                v = self.velocityModel.ebq_global[('u',1)]
                self.c_u[u.shape] = u
                self.c_v[v.shape] = v
                self.model.ebq_global[('velocity',0)][...,0] = u
                self.model.ebq_global[('velocity',0)][...,1] = v
        elif (self.velocityModelIndex >= 0):
            assert self.velocityModelIndex < len(modelList), \
                "velocity model index out of  range 0," + repr(len(modelList))
            self.velocityModel = modelList[self.velocityModelIndex]
        if self.densityModelIndex >= 0:  # make this model available to test vs chi
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
                rho = self.densityModel.numericalFlux.ebqe[('u',0)]
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
        for ci in range(self.nc):
            cq[('u_last',ci)] = deepcopy(cq[('u',ci)])
            cq[('grad(u)_last',ci)] = deepcopy(cq[('grad(u)',ci)])
            if self.bdf is int(2):
                cq[('u_lastlast',ci)] = deepcopy(cq[('u',ci)])
                cq[('grad(u)_lastlast',ci)] = deepcopy(cq[('grad(u)',ci)])
        cq[('velocity_last',0)] = deepcopy(cq[('velocity',0)])
        if self.bdf is int(2):
            cq[('velocity_lastlast',0)] = deepcopy(cq[('velocity',0)])

    def initializeElementBoundaryQuadrature(self,t,cebq,cebq_global):
        """
        Give the TC object access to the element boundary quadrature storage
        """
        for ci in range(self.nc):
            cebq[('u_last',ci)] = deepcopy(cebq[('u',ci)])
            cebq[('grad(u)_last',ci)] = deepcopy(cebq[('grad(u)',ci)])
            if self.bdf is int(2):
                cebq[('u_lastlast',ci)] = deepcopy(cebq[('u',ci)])
                cebq[('grad(u)_lastlast',ci)] = deepcopy(cebq[('grad(u)',ci)])
        cebq[('velocity_last',0)] = deepcopy(cebq[('velocity',0)])
        if self.bdf is int(2):
            cebq[('velocity_lastlast',0)] = deepcopy(cebq[('velocity',0)])
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
        cebqe[('velocity_last',0)] = deepcopy(cebqe[('velocity',0)])
        if self.bdf is int(2):
            cebqe[('velocity_lastlast',0)] = deepcopy(cebqe[('velocity',0)])
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

        copyInstructions = {}
        return copyInstructions
    def postStep(self,t,firstStep=False):
        """
        Calculate the mean value of phi and adjust to make mean value 0.
        """
        from math import fabs
        import proteus.Norms as Norms
        from proteus.flcbdfWrappers import globalSum,globalMax


        # Do adjustment to get zero value for mean of pressure increment
        if self.zeroMean:
            # It appears that the mesh.volume term is not being correctly
            # computed in parallel so I implemented this work around for now.
            if self.calculateMeshVolume:
                onesVector = np.ones(self.model.q['u',0].shape)
                self.meshVolume = Norms.scalarDomainIntegral(self.model.q['dV'],
                                                       onesVector,
                                                       self.model.mesh.nElements_owned)
                self.calculateMeshVolume = False


            meanvalue = Norms.scalarDomainIntegral(self.model.q['dV'],
                                                   self.model.q[('u',0)],
                                                   self.model.mesh.nElements_owned)/self.meshVolume
            self.model.q[('u',0)][:] = self.model.q[('u',0)] - meanvalue
            self.model.ebqe[('u',0)] -= meanvalue
            self.model.u[0].dof -= meanvalue

            # test to see if we are in fact zero mean.  This can be removed in
            # an optimized code.
            newmeanvalue = Norms.scalarDomainIntegral(self.model.q['dV'],
                                                      self.model.q[('u',0)],
                                                      self.model.mesh.nElements_owned)/self.meshVolume
            assert fabs(newmeanvalue) < 1.0e-8, "new mean should be zero but is "+`newmeanvalue`

        # add post processing adjustments here if possible.  They have already be solved for by this point.

        # If self.initializeUsingPressureFunction (for debugging), then
        # set the first step of pressure increment to be p_h^1 - p_h^0
        if (self.setFirstTimeStepValues and firstStep and t>0):
            # for eN in range(self.model.mesh.nElements_global):
            #     for j in self.model.u[0].femSpace.referenceFiniteElement.localFunctionSpace.range_dim:
            #         jg = self.model.u[0].femSpace.dofMap.l2g[eN,j]
            #         x = self.model.u[0].femSpace.dofMap.lagrangeNodesArray[jg]
            #         self.model.u[0].dof[jg]=self.pressureFunction(x,t)-self.pressureFunction(x,0)
            self.model.u[0].dof[:] = self.pressureFunction(self.model.mesh.nodeArray,t)-self.pressureFunction(self.model.mesh.nodeArray,0)
            self.model.calculateSolutionAtQuadrature()
            self.evaluate(t,self.model.q)
            # self.evaluate(t,self.model.ebq)
            self.evaluate(t,self.model.ebqe)
            # self.evaluate(t,self.model.ebq_global)

        copyInstructions = {}
        return copyInstructions
    def evaluate(self,t,c):
        """
        Evaluate the coefficients after getting the specified velocity and density
        """

        # precompute the shapes to extract things we need from self.c_name[] dictionaries
        u_shape = c[('u',0)].shape
        # grad_shape = c[('grad(u)',0)].shape

        # time management
        dt = self.model.timeIntegration.dt

        # beta_0 coefficient scalar
        if self.bdf is int(1) or self.firstStep:
            b0 = 1.0/dt
            dt_last = dt
        elif self.bdf is int(2):
            dt_last = self.velocityModel.timeIntegration.dt_history[0] # velocity model has the timeIntegration all set up.
            dtInv = 1.0/dt
            r = dt/dt_last
            b0 = (1.0+2.0*r)/(1.0+r)*dtInv # use this instead of alpha_bdf since we have no timeIntegration in this model

        # find minimal density value set it to be chi
        if self.densityModelIndex>0:
            rho = self.c_rho[u_shape]
        else:
            rho = [self.chiValue] # just give it the self.chiValue so that test passes as we assume user has given correct chiValue in this case.

        # Extract minimum of density and compare to given chiValue.
        # Ideally, the density model is maximum preserving and will not drop values
        # below this but if it does we want to know.  Even still we will
        # use the user given value of chi.
        chi = np.min(rho)
        if chi < self.chiValue :  # raise warning but do not stop
            log("*** warning: minimum of density = %1.3e is below physical limit chiValue = %1.3e. ***" %(chi, self.chiValue),  level=1)
        chi = self.chiValue

        # Extract velocity components:  notice that the ('velocity',0) field
        # corresponds to this model so it has not been updated to reflect the
        # new information calculated.  Thus it is unavailable at this time.
        # The post processed velocity should be generated here from the newly
        # calculated velocity so we want to use the actual velocityModel data.
        if self.velocityFunction != None:
            u = self.velocityFunction(c['x'],t)[...,0]
            v = self.velocityFunction(c['x'],t)[...,1]
        else:
            u = self.c_u[u_shape]
            v = self.c_v[u_shape]

        # set coefficients  -div (grad phi) + chi b0 div (u) = 0
        #  div ( f - a grad phi  )  = div( chi b0 u - grad phi) = 0

        rescale = True  # True: makes the residuals (newton linear step, nonlinear)
                        # better scaled and gives 'f' the right velocity scale
                        # so that since the post processor takes from 'f' - 'a' grad 'phi'
                        # there is no need to rescale anything elsewhere if we user
                        # the pp velocity from the pressure increment 'f' ...  which will
                        # be divergence free.
        if rescale:
            c[('f',0)][...,0] = u
            c[('f',0)][...,1] = v
        else:
            c[('f',0)][...,0] = chi*b0*u
            c[('f',0)][...,1] = chi*b0*v
        c[('df',0,0)][...,0] = 0.0
        c[('df',0,0)][...,1] = 0.0
        if rescale:
            c[('a',0,0)][...,0] = 1.0/(chi*b0) # -\grad v :   tensor  [ 1.0  0;  0  1.0] ordered [0 1; 2 3]  in our
            c[('a',0,0)][...,1] = 1.0/(chi*b0) # -\grad v :       new diagonal notation from sDInfo above is [0 .; . 1] -> [0; 1]
        else:
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
                 pressureFunction=None,
                 useRotationalModel=True,
                 chiValue=1.0,
                 setFirstTimeStepValues=True,
                 usePressureExtrapolations=False):
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
        self.pressureFunction = pressureFunction
        self.useRotationalModel = useRotationalModel
        self.chiValue = chiValue
        if self.useRotationalModel:
            self.c_u = {}
            self.c_v = {}
            self.c_velocity = {}
        self.c_phi = {}
        self.firstStep = True # manipulated in preStep()
        self.setFirstTimeStepValues = setFirstTimeStepValues
        self.usePressureExtrapolations = usePressureExtrapolations

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
        for ci in range(self.nc):
            self.model.points_quadrature.add(('u_last',ci))
            self.model.points_elementBoundaryQuadrature.add(('u_last',ci))
            self.model.numericalFlux.ebqe[('u_last',ci)]=deepcopy(self.model.ebqe[('u_last',ci)])
            self.model.vectors_quadrature.add(('grad(u)_last',ci))
            self.model.vectors_elementBoundaryQuadrature.add(('grad(u)_last',ci))
            self.model.numericalFlux.ebqe[('grad(u)_last',ci)]=deepcopy(self.model.ebqe[('grad(u)_last',ci)])
            if self.bdf is int(2):
                self.model.points_quadrature.add(('u_lastlast',ci))
                self.model.points_elementBoundaryQuadrature.add(('u_lastlast',ci))
                self.model.numericalFlux.ebqe[('u_lastlast',ci)]=deepcopy(self.model.ebqe[('u_lastlast',ci)])
                self.model.vectors_quadrature.add(('grad(u)_lastlast',ci))
                self.model.vectors_elementBoundaryQuadrature.add(('grad(u)_lastlast',ci))
                self.model.numericalFlux.ebqe[('grad(u)_lastlast',ci)]=deepcopy(self.model.ebqe[('grad(u)_lastlast',ci)])
        if ((self.usePressureExtrapolations or not self.useVelocityComponents) and self.velocityModelIndex >= 0):
            self.velocityModel = modelList[self.velocityModelIndex]

        if ( self.useRotationalModel and not self.useVelocityComponents and
             self.pressureIncrementModelIndex >= 0 and self.velocityFunction is None ):
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
        elif (self.useRotationalModel and self.useVelocityComponents and
              self.velocityFunction is None and self.velocityModelIndex >= 0):
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
                u = self.velocityModel.numericalFlux.ebqe[('u',0)]
                v = self.velocityModel.numericalFlux.ebqe[('u',1)]
                self.c_u[u.shape] = u
                self.c_v[v.shape] = v
            if ('u',0) in self.velocityModel.ebq_global:
                u = self.velocityModel.ebq_global[('u',0)]
                v = self.velocityModel.ebq_global[('u',1)]
                self.c_u[u.shape] = u
                self.c_v[v.shape] = v
        if (self.pressureIncrementModelIndex >= 0 and self.pressureIncrementFunction is None):
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
            cq[('grad(u)_last',ci)] = deepcopy(cq[('grad(u)',ci)])
            if self.bdf is int(2):
                cq[('u_lastlast',ci)] = deepcopy(cq[('u',ci)])
                cq[('grad(u)_lastlast',ci)] = deepcopy(cq[('grad(u)',ci)])
    def initializeElementBoundaryQuadrature(self,t,cebq,cebq_global):
        """
        Give the TC object access to the element boundary quadrature storage
        """
        for ci in range(self.nc):
            cebq[('u_last',ci)] = deepcopy(cebq[('u',ci)])
            cebq[('grad(u)_last',ci)] = deepcopy(cebq[('grad(u)',ci)])
            if self.bdf is int(2):
                cebq[('u_lastlast',ci)] = deepcopy(cebq[('u',ci)])
                cebq[('grad(u)_lastlast',ci)] = deepcopy(cebq[('grad(u)',ci)])
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
        self.firstStep = firstStep

        copyInstructions = {}
        return copyInstructions
    def postStep(self,t,firstStep=False):
        """
        Give the TC object an opportunity to modify itself before the time step.
        """
        if (self.setFirstTimeStepValues and firstStep and t>0):
            # for eN in range(self.model.mesh.nElements_global):
            #     for j in self.model.u[0].femSpace.referenceFiniteElement.localFunctionSpace.range_dim:
            #         jg = self.model.u[0].femSpace.dofMap.l2g[eN,j]
            #         x = self.model.u[0].femSpace.dofMap.lagrangeNodesArray[jg]
            #         self.model.u[0].dof[jg]=self.pressureFunction(x,t)
            self.model.u[0].dof[:] = self.pressureFunction(self.model.mesh.nodeArray,t)
            self.model.calculateSolutionAtQuadrature()
            self.evaluate(t,self.model.q)
            # self.evaluate(t,self.model.ebq)
            self.evaluate(t,self.model.ebqe)
            # self.evaluate(t,self.model.ebq_global)
        else:
            if not self.usePressureExtrapolations and not self.useRotationalModel:
                assert ((self.model.q[('u',0)] - self.model.q[('u_last',0)] - self.pressureIncrementModel.q[('u',0)])**2 < 1.0e-16).all()
                assert ((self.model.ebqe[('u',0)] - self.model.ebqe[('u_last',0)] - self.pressureIncrementModel.ebqe[('u',0)])**2 < 1.0e-16).all()
        copyInstructions = {}
        return copyInstructions
    def evaluate(self,t,c):
        """
        Evaluate the coefficients after getting the specified velocity and density
        """
        # precompute the shapes to extract things we need from self.c_name[] dictionaries
        u_shape = c[('u',0)].shape
        grad_shape = c[('grad(u)',0)].shape

        # current and previous pressure values
        p = c[('u',0)]
        p_last = c[('u_last',0)]

        # order of pressure extrapolation first or second order (should be the same as in pressure model)
        if self.bdf is int(1) or self.firstStep or not self.usePressureExtrapolations:
            p_star = p_last
        elif self.bdf is int(2) and self.usePressureExtrapolations:
            dt = self.model.timeIntegration.dt
            dt_last = self.velocityModel.timeIntegration.dt_history[0]
            p_lastlast = c[('u_lastlast',0)]
            p_star = p_last + dt/dt_last*(p_last - p_lastlast)

        # extract pressure increment
        if self.pressureIncrementFunction != None:
            phi = self.pressureIncrementFunction(c['x'],t)
        else:
            phi = self.c_phi[u_shape]

        if self.useRotationalModel:
            # extract velocity components
            if self.velocityFunction != None:
                u = self.velocityFunction(c['x'],t)[...,0]
                v = self.velocityFunction(c['x'],t)[...,1]
            elif self.useVelocityComponents:
                u = self.c_u[u_shape]
                v = self.c_v[u_shape]
            else:
                u = self.c_velocity[grad_shape][...,0]
                v = self.c_velocity[grad_shape][...,1]


        # set coefficients   p - p_star - phi + div (mu u) = 0
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
        c[('r',0)][:] = p - p_star - phi
        c[('dr',0,0)][:] = 1.0

class L2Projection(TransportCoefficients.TC_base):
    def __init__(self,
                 projectTime=None,
                 toName='u',
                 myModelIndex=0,
                 toModelIndex=-1,
                 toModel_u_ci=0,
                 exactFunction=lambda x,t: 1.0):
        TransportCoefficients.TC_base.__init__(self,
                                               nc = 1,
                                               variableNames = ['pi_'+toName],
                                               reaction = {0:{0:'linear'}})
        self.myModelIndex = myModelIndex
        self.toModelIndex = toModelIndex
        self.toModel_u_ci = toModel_u_ci
        self.exactFunction = exactFunction
        self.projectTime=projectTime
    def attachModels(self,modelList):
        self.myModel = modelList[self.myModelIndex]
        self.toModel = modelList[self.toModelIndex]
    def evaluate(self,t,c):
        if self.projectTime is not None:
            T=self.projectTime
        else:
            T=t
        c[('r',0)][:] = c[('u',0)] - self.exactFunction(c['x'],0.0)
        c[('dr',0,0)][:] = 1.0
        if self.toModelIndex >= 0:
            self.toModel.u[self.toModel_u_ci].dof[:] = self.myModel.u[0].dof

class StokesProjection2D(TransportCoefficients.TC_base):
    def __init__(self,
                 grad_u_function,
                 grad_v_function,
                 p_function,
                 mu=1.0,
                 projectTime=0.0,
                 myModelIndex=0,
                 toModelIndex_v=-1,
                 toModelIndex_p=-1):
        sdInfo  = {(0,0):(np.array([0,1,2],dtype='i'),  # sparse diffusion uses diagonal element for diffusion coefficient
                          np.array([0,1],dtype='i')),
                   (1,1):(np.array([0,1,2],dtype='i'),
                          np.array([0,1],dtype='i'))}
        dim=2; # dimension of space
        xi=0; yi=1; # indices for first component or second component of dimension
        eu=0; ev=1; ediv=2; # equation numbers  momentum u, momentum v, divergencefree
        ui=0; vi=1; pi=2;  # variable name ordering
        TransportCoefficients.TC_base.__init__(self,
                         nc=dim+1, #number of components  u, v, p
                         variableNames=['pi_u','pi_v','pi_p'], # defines variable reference order [0, 1, 2]
                        advection = {eu:{ui:'constant'},
                                     ev:{vi:'constant'},
                                     ediv:{ui:'constant',
                                           vi:'constant'}},
                         hamiltonian = {eu:{pi:'linear'},   # p_x
                                        ev:{pi:'linear'}},  # p_y
                         diffusion = {eu:{ui:{ui:'constant'}},  # - \mu * \grad u
                                      ev:{vi:{vi:'constant'}}}, # - \mu * \grad v
                         potential = {eu:{ui:'u'},
                                      ev:{vi:'u'}}, # define the potential for the diffusion term to be the solution itself
                         sparseDiffusionTensors=sdInfo,
                         useSparseDiffusion = True),
        self.vectorComponents=[ui,vi]
        self.vectorName="pi_velocity"
        self.myModelIndex = myModelIndex
        self.toModelIndex_v = toModelIndex_v
        self.toModelIndex_p = toModelIndex_p
        self.grad_u_function = grad_u_function
        self.grad_v_function = grad_v_function
        self.p_function = p_function
        self.mu=mu
        self.T=projectTime
    def attachModels(self,modelList):
        self.myModel = modelList[self.myModelIndex]
        self.toModel_v = modelList[self.toModelIndex_v]
        self.toModel_p = modelList[self.toModelIndex_p]
    def postStep(self,t,firstStep=False):
        self.toModel_v.u[0].dof[:]=self.myModel.u[0].dof
        self.toModel_v.u[1].dof[:]=self.myModel.u[1].dof
        self.toModel_v.calculateSolutionAtQuadrature()
        self.toModel_v.numericalFlux.setDirichletValues(self.toModel_v.ebqe)
        self.toModel_p.u[0].dof[:]=self.myModel.u[2].dof
        self.toModel_p.calculateSolutionAtQuadrature()
        self.toModel_p.numericalFlux.setDirichletValues(self.toModel_p.ebqe)
    def evaluate(self,t,c):
        xi=0; yi=1; # indices for first component or second component of dimension
        eu=0; ev=1; ediv=2; # equation numbers  momentum u, momentum v, divergencefree
        ui=0; vi=1; pi=2;  # variable name ordering
        u = c[('u',ui)]
        v = c[('u',vi)]
        p = c[('u',pi)]
        grad_p = c[('grad(u)',pi)]


        c[('H',eu)][:] = grad_p[...,xi]
        c[('dH',eu,pi)][...,xi] = 1.0
        c[('a',eu,ui)][...,0] = self.mu
        c[('a',eu,ui)][...,1] = self.mu
        c[('f',eu)][...,0]  = self.mu*self.grad_u_function(c['x'],self.T)[...,0] - self.p_function(c['x'],self.T)
        c[('f',eu)][...,1]  = self.mu*self.grad_u_function(c['x'],self.T)[...,1]

        c[('H',ev)][:] = grad_p[...,yi]
        c[('dH',ev,pi)][...,yi] = 1.0
        c[('a',ev,vi)][...,0] = self.mu
        c[('a',ev,vi)][...,1] = self.mu
        c[('f',ev)][...,0]  = self.mu*self.grad_v_function(c['x'],self.T)[...,0]
        c[('f',ev)][...,1]  = self.mu*self.grad_v_function(c['x'],self.T)[...,1] - self.p_function(c['x'],self.T)

        c[('f',ediv)][...,xi] = u
        c[('f',ediv)][...,yi] = v
        c[('df',ediv,ui)][...,xi] = 1.0  # d_f_d_u [xi]
        c[('df',ediv,ui)][...,yi] = 0.0  # d_f_d_u [yi]
        c[('df',ediv,vi)][...,xi] = 0.0  # d_f_d_v [xi]
        c[('df',ediv,vi)][...,yi] = 1.0  # d_f_d_v [yi]
