"""
This module contains the coefficients classes for various NavierStokes equation definitions.
"""


from proteus.iproteus import TransportCoefficients


class NavierStokes1D(TransportCoefficients.TC_base):
    r"""
    The coefficients of the 1D Navier Stokes equation
    
    The compressible equations are given by
    
    .. math:: 
       :nowrap:
       
       \begin{align}
         \nabla \cdot \mathbf{v}  &= r(x,t)\\ 
         \frac{\partial (\rho\mathbf{v})}{\partial t} + \nabla\cdot\left(\rho\mathbf{v}\otimes\mathbf{v}\right) +  \nabla p  - \nabla \cdot \left(\mu \nabla\mathbf{v}\right) &= \rho\mathbf{f}(x,t)
       \end{align}
       
    with :math:`\rho` and :math:`\mu` the given density and viscosity as constants.
     
    """
    def __init__(self,rofx,fofx,rho=1.0,mu=1.0):
        TransportCoefficients.TC_base.__init__(self, 
                         nc=2, #number of components
                         variableNames=['p','v'],
                         mass = {1:{1:'linear'}}, # du/dt
                         advection = {0:{1:'linear'}, # \nabla\cdot v
                                      1:{1:'nonlinear'}}, # \nabla \cdot (v\otimes v)
                         hamiltonian = {1:{0:'linear'}}, # grad (p)
                         diffusion = {1:{1:{1:'constant'}}}, # - 1/Re * \grad v
                         potential = {1:{1:'u'}}, # define the potential for the diffusion term to be the solution itself
                         reaction  = {0:{0:'constant'}, # r(x)
                                      1:{1:'constant'}}) # f(x)
        self.rofx=rofx
        self.fofx=fofx
        self.rho=rho
        self.mu=mu
        
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
        p = c[('u',0)]
        v = c[('u',1)] #1D - x component of velocity
        grad_p = c[('grad(u)',0)]
        #equation 0  div(f) + r = 0  (proteus  notation)  div(velocity)=r(x) (our notation)
        c[('f',0)][...,0] = v
        c[('df',0,1)][...,0] = 1.0  # d_f^0_d_u^0
        c[('r',0)][:]     = -self.rofx(c['x'][:],t)
        c[('dr',0,0)][:]   = 0.0
        #equation 1   u_t + div (u^2) + grad(p) + div(-1/Re grad(v)) + f = 0
        c[('m',1)][:] = rho*v  # d/dt ( rho * v) = d/dt (m_1)
        c[('dm',1,1)][:] = rho  # d_m^1_d_u^1
        c[('f',1)][...,0] = rho*v*v # div ( v\otimes v)
        c[('df',1,1)][...,0] = 2.0*rho*v # d_f^1_d_u^1
        c[('H',1)][:] = grad_p[...,0] #grad(p)
        c[('dH',1,0)][...,0] = 1.0 # 1
        c[('r',1)][:]     = -rho*self.fofx(c['x'][:],t)
        c[('dr',1,1)][:]  = 0.0#0
        c[('a',1,1)][...,0] = mu # -mu*\grad v
        c[('da',1,1,1)][...,0] = 0.0 # -d_(1/Re)
        


class NavierStokes2D_Momentum(TransportCoefficients.TC_base):
    r"""
    The coefficients of the 2D Navier Stokes momentum equation with variable density.  This coefficient class
    will only represent the momentum equation and the incompressibility equation but not the conservation of mass.
    
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
    
    """
    def __init__(self,rhoofx,f1ofx,f2ofx,mu=1.0):
        
        sdInfo  = {(0,0):(numpy.array([0,1,2],dtype='i'),  # sparse diffusion uses diagonal element for diffusion coefficient
                          numpy.array([0,1],dtype='i')),
                   (1,1):(numpy.array([0,1,2],dtype='i'),
                          numpy.array([0,1],dtype='i'))}
        
        TransportCoefficients.TC_base.__init__(self, 
                         nc=2, #number of components
                         variableNames=['u','v','p'], # defines variable reference order [0, 1, 2]
                         mass = {0:{0:'linear'}, # du/dt
                                 1:{1:'linear'}}, # dv/dt
                         advection = {2:{0:'linear',   # \nabla\cdot [u v]
                                         1:'linear'}}, # \nabla\cdot [u v]
                         hamiltonian = {0:{0:'nonlinear', # u u_x + v u_y    convection term   
                                           2:'linear'},   # grad p
                                        1:{1:'nonlinear', # u v_x + v v_y   convection term
                                           2:'linear'}},  # grad (p)
                         diffusion = {0:{1:{1:'constant'}},  # - \mu * \grad u
                                      1:{2:{2:'constant'}}}, # - \mu * \grad v
                         potential = {0:{0:'u'},
                                      1:{1:'u'}}, # define the potential for the diffusion term to be the solution itself
                         reaction  = {0:{0:'constant'}, # f1(x)
                                      1:{1:'constant'}}), # f2(x)
                         sparseDiffusionTensors=sdInfo,
                         useSparseDiffusion = True
                        
                
        self.rhoofx=rhoofx
        self.f1ofx=f1ofx
        self.f2ofx=f2ofx
        self.mu=mu
        
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
        xi=0; yi=1; ui=0; vi=1; pi=2;  # indices for better readability
        p = c[('u',pi)]
        u = c[('u',ui)]
        v = c[('u',vi)]
        grad_p = c[('grad(u)',pi)]
        grad_u = c[('grad(u)',ui)]
        grad_v = c[('grad(u)',vi)]
        #equation 0  rho*(u_t + u ux + v uy ) + px + div(-mu grad(u)) - f1 = 0
        c[('m',ui)][:] = rho*u  # d/dt ( rho * u) = d/dt (m_0)
        c[('dm',ui,ui)][:] = rho  # dm^0_du
        c[('r',ui)][:] = -self.f1ofx(c['x'][:],t)
        c[('dr',ui,ui)][:] = 0.0
        c[('H',ui)][:] = grad_p[...,xi] + u*grad_u[...,xi] + v*grad_u[...,yi]  # add rho term
        c[('dH',ui,ui)][...,xi] = u # 
        c[('dH',ui,ui)][...,yi] = v # 
        c[('dH',ui,pi)][...,xi] = 1.0 #  p_x
        c[('a',ui,ui)][...,0] = mu # -mu*\grad v  tensor  [ mu  0;  0  mu]  [0 1; 2 3]  in our new notation is [0 .; . 1]
        c[('a',ui,ui)][...,1] = mu # -mu*\grad v  
        c[('da',ui,ui,ui)][...,0] = 0.0 # -(da/d ui)_0   # could leave these off since it is 0
        c[('da',ui,ui,ui)][...,1] = 0.0 # -(da/d ui)_1   # could leave these off since it is 0

        # equation 1  rho*(v_t + u vx + v vy ) + py + div(-mu grad(v)) - f2 = 0
        c[('m',vi)][:] = rho*v  # d/dt ( rho * v) = d/dt (m_1)
        c[('dm',vi,vi)][:] = rho  # dm^1_dv
        c[('r',vi)][:] = -self.f2ofx(c['x'][:],t)
        c[('dr',vi,vi)][:] = 0.0
        c[('H',vi)][:] = grad_p[...,xi] + u*grad_v[...,xi] + v*grad_v[...,yi]  # add rho term
        c[('dH',vi,vi)][...,xi] = u # 
        c[('dH',vi,vi)][...,yi] = v # 
        c[('dH',vi,pi)][...,yi] = 1.0 #  p_y
        c[('a',vi,vi)][...,0] = mu # -mu*\grad v  tensor  [ mu  0;  0  mu]  [0 1; 2 3]  in our new notation is [0 .; . 1]
        c[('a',vi,vi)][...,1] = mu # -mu*\grad v  
        c[('da',vi,vi,vi)][...,0] = 0.0 # -(da/d vi)_0   # could leave these off since it is 0
        c[('da',vi,vi,vi)][...,1] = 0.0 # -(da/d vi)_1   # could leave these off since it is 0

        #equation 2  div [u v] = 0
        c[('f',pi)][...,0] = u
        c[('f',pi)][...,1] = v
        c[('df',pi,ui)][...,0] = 1.0  # d_f^0_d_u^0
        c[('df',pi,ui)][...,1] = 0.0  # d_f^0_d_u^0
        c[('df',pi,vi)][...,0] = 0.0  # d_f^0_d_u^0
        c[('df',pi,vi)][...,1] = 1.0  # d_f^0_d_u^0
        

