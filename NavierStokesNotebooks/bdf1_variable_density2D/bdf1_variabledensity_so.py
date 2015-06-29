from proteus.default_so import *
from proteus import Context

import bdf1_variabledensity
Context.setFromModule(bdf1_variabledensity)
ctx = Context.get()

pnList = [("density_p", "density_n"),
          ("velocity_p", "velocity_n"),
          ("pressureincrement_p", "pressureincrement_n"),
          ("pressure_p", "pressure_n")]

name = "bdf1_variabledensity" + "_%3.0e_DT_BDF%1d_p" %(ctx.DT,int(float(ctx.globalTimeOrder)))

# modelSpinUpList = [1] # for model [1] take a step and then rewind time to time t^0 and proceed as usual
# systemStepControllerType = Sequential_MinAdaptiveModelStep  # uses minimal time step from each _n model
# systemStepControllerType = Sequential_FixedStep  # not sure what this one currently does but it should use the DT
# systemStepControllerType = Sequential_MinModelStep # uses DT set in _n.py files
systemStepControllerType = Sequential_FixedStep_Simple # uses time steps in so.tnList

needEBQ_GLOBAL = True
needEBQ = True

tnList = ctx.tnList

