from proteus.default_so import *
from proteus import Context

import navierstokes_vardensity
import SplitOperatorPS
Context.setFromModule(navierstokes_vardensity)
ctx = Context.get()


if ctx.useInitialConditions is int(0): # Interpolate initial conditions
    pnList = [("density_p", "density_n"),
              ("velocity_p", "velocity_n"),
              ("pressureincrement_p", "pressureincrement_n"),
              ("pressure_p", "pressure_n")]
    # modelSpinUpList = []
    systemStepControllerType = Sequential_FixedStep_Simple # uses time steps in so.tnList
if ctx.useInitialConditions is int(1): # use projection for everything but calculate pressure increment from velocity.
    pnList = [("density_p", "density_n"),
              ("velocity_p", "velocity_n"),
              ("pressureincrement_p", "pressureincrement_n"),
              ("pressure_p", "pressure_n"),
              ("L2_rho_p","L2_rho_n"),
              ("L2_u_p","L2_u_n"),
              ("L2_v_p","L2_v_n"),
              ("L2_p_p","L2_p_n")]
    modelSpinUpList = [4,5,6,7,2]# density through pressure then initialize pressure increment based on velocity
    systemStepControllerType = SplitOperatorPS.Sequential_FixedStep_SimplePS # uses time steps in so.tnList
elif ctx.useInitialConditions is int(2): # use Stokes projection for velocity and pressure and calculate pressure increment from velocity and use L2 proj for density
    pnList = [("density_p", "density_n"),
              ("velocity_p", "velocity_n"),
              ("pressureincrement_p", "pressureincrement_n"),
              ("pressure_p", "pressure_n"),
              ("L2_rho_p","L2_rho_n"),
              ("StokesProjection_p","StokesProjection_n")]
    modelSpinUpList = [4,5,2] # density, stokes projection, pressure increment,
    systemStepControllerType = SplitOperatorPS.Sequential_FixedStep_SimplePS # uses time steps in so.tnList



# name = "navierstokes_vardensity" + "_%3.0e_DT_BDF%1d_p" %(ctx.DT,int(float(ctx.globalBDFTimeOrder)))
name = "navierstokes_vardensity" + "_DT_0_%s_BDF%1d_p" %(ctx.DT_string,int(float(ctx.globalBDFTimeOrder)))

# modelSpinUpList = [1] # for model [1] take a step and then rewind time to time t^0 and proceed as usual
# systemStepControllerType = Sequential_MinAdaptiveModelStep  # uses minimal time step from each _n model
# systemStepControllerType = Sequential_FixedStep  # not sure what this one currently does but it should use the DT
# systemStepControllerType = Sequential_MinModelStep # uses DT set in _n.py files
# systemStepControllerType = Sequential_FixedStep_Simple # uses time steps in so.tnList
# systemStepControllerType = SplitOperatorPS.Sequential_FixedStep_SimplePS # uses time steps in so.tnList

if not ctx.useVelocityComponents:
    needEBQ_GLOBAL = True
    needEBQ = True
else:
    needEBQ_GLOBAL = False
    needEBQ = False

archiveFlag = ArchiveFlags.EVERY_USER_STEP

tnList = ctx.tnList
