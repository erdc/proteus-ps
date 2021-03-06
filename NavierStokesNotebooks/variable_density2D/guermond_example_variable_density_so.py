from proteus.default_so import *
from proteus import Context

import guermond_example_variable_density
Context.setFromModule(guermond_example_variable_density)
ctx = Context.get()

pnList = [("rho_p", "rho_n"),
          ("mom_p", "mom_n")]

name = "guermond_example_variable_density" + "_%3.0e_DT_BDF%1d_p" %(ctx.DT,int(float(ctx.globalTimeOrder)))

# modelSpinUpList = [1] # for model [1] take a step and then rewind time to time t^0 and proceed as usual
# systemStepControllerType = Sequential_MinAdaptiveModelStep  # uses minimal time step from each _n model
# systemStepControllerType = Sequential_FixedStep  # not sure what this one currently does but it should use the DT
# systemStepControllerType = Sequential_MinModelStep # uses DT set in _n.py files
systemStepControllerType = Sequential_FixedStep_Simple # uses time steps in so.tnList

needEBQ_GLOBAL = True
needEBQ = True

tnList = ctx.tnList

