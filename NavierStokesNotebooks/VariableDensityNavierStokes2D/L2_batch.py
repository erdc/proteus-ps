
from proteus import Context
from proteus import Comm
comm = Comm.get()
ctx = Context.get()


# simulation flags for error analysis
#
# simFlagsList is initialized in proteus.iproteus
#

# Density
simFlagsList[0]['errorQuantities']=['u']
simFlagsList[0]['errorTypes']= ['numericalSolution'] #compute error in soln and glob. mass bal
simFlagsList[0]['errorNorms']= ['L2'] #compute L2 norm in space
simFlagsList[0]['errorTimes']= ['All'] #'All', 'Last'
simFlagsList[0]['echo']=True
simFlagsList[0]['dataFile']       = simFlagsList[0]['simulationName'] + "_BDF%1d_dt_0_%s_%s.db" %(int(float(ctx.globalBDFTimeOrder)),ctx.DT_string,comm.rank())
simFlagsList[0]['dataDir']        = os.getcwd()+'/results'
simFlagsList[0]['storeQuantities']= ['simulationData','errorData'] #include errorData for mass bal
simFlagsList[0]['storeTimes']     = ['Last']

# Velocity
simFlagsList[1]['errorQuantities']=['u']
simFlagsList[1]['errorTypes']= ['numericalSolution'] #compute error in soln and glob. mass bal
simFlagsList[1]['errorNorms']= ['L2','H1'] #compute L2 norm in space or H1 or ...
simFlagsList[1]['errorTimes']= ['All'] #'All', 'Last'
simFlagsList[1]['echo']=True
simFlagsList[1]['dataFile']       = simFlagsList[1]['simulationName'] + "_BDF%1d_dt_0_%s_%s.db" %(int(float(ctx.globalBDFTimeOrder)),ctx.DT_string,comm.rank())
simFlagsList[1]['dataDir']        = os.getcwd()+'/results'
simFlagsList[1]['storeQuantities']= ['simulationData','errorData'] #include errorData for mass bal
simFlagsList[1]['storeTimes']     = ['Last']

# # Pressure Increment
# simFlagsList[2]['errorQuantities']=['u']
# simFlagsList[2]['errorTypes']= ['numericalSolution'] #compute error in soln and glob. mass bal
# simFlagsList[2]['errorNorms']= ['L2'] #compute L2 norm in space or H1 or ...
# simFlagsList[2]['errorTimes']= ['All'] #'All', 'Last'
# simFlagsList[2]['echo']=True
# simFlagsList[2]['dataFile']       = simFlagsList[2]['simulationName'] + "_BDF%1d_dt_0_%s_%s.db" %(int(float(ctx.globalBDFTimeOrder)),ctx.DT_string,comm.rank())
# simFlagsList[2]['dataDir']        = os.getcwd()+'/results'
# simFlagsList[2]['storeQuantities']= ['simulationData','errorData'] #include errorData for mass bal
# simFlagsList[2]['storeTimes']     = ['Last']

# Pressure
simFlagsList[3]['errorQuantities']=['u']
simFlagsList[3]['errorTypes']= ['numericalSolution'] #compute error in soln and glob. mass bal
simFlagsList[3]['errorNorms']= ['L2'] #compute 'L2' norm in space or 'H1' or 'H1semi' ...
simFlagsList[3]['errorTimes']= ['All'] #'All', 'Last'
simFlagsList[3]['echo']=True
simFlagsList[3]['dataFile']       = simFlagsList[3]['simulationName'] + "_BDF%1d_dt_0_%s_%s.db" %(int(float(ctx.globalBDFTimeOrder)),ctx.DT_string,comm.rank())
simFlagsList[3]['dataDir']        = os.getcwd()+'/results'
simFlagsList[3]['storeQuantities']= ['simulationData','errorData'] #include errorData for mass bal
simFlagsList[3]['storeTimes']     = ['Last']

#
start
quit
