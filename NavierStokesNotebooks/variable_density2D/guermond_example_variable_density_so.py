from proteus.default_so import *
from proteus import Context
import guermond_example_variable_density
Context.setFromModule(guermond_example_variable_density)
ctx = Context.get()

pnList = [("rho_p", "rho_n"),
          ("mom_p", "mom_n")]

#modelSpinUpList = [1]
name = "guermond_example_variable_density_p"

systemStepControllerType = Sequential_MinAdaptiveModelStep

needEBQ_GLOBAL = True
needEBQ = True

tnList = ctx.tnList
