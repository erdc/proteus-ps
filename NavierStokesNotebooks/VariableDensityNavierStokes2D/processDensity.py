#!/usr/bin/env python

import sys # to read in command line arguments
import os  # to check if file exists
import shelve # to open and read the database file

import numpy as np

usePlots = True
if usePlots:
    import matplotlib
    matplotlib.use('AGG')   # generate png output by default

	import matplotlib.pyplot as plt
	from matplotlib import rc
	rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
	## for Palatino and other serif fonts use:
	#rc('font',**{'family':'serif','serif':['Palatino']})
	rc('text', usetex=True)

	# create plots for time series of errors
	fig_L2, ax_L2 = plt.subplots(nrows=1, ncols=1)
	# fig_H1, ax_H1 = plt.subplots(nrows=1, ncols=1)

# read in file names and test if they can be found.  if so
# then add them to our list of filenames to be processed
filenames = {}
if len(sys.argv) > 1:
    counter = 0;
    for i in range(1,len(sys.argv)):
        test_filename = sys.argv[i]
        if os.path.exists(test_filename):
            print "File %s found and will be processed!" %test_filename
            filenames[counter] = test_filename
            counter+=1
        else:
            print "File %s not found!"%test_filename

num_filenames = len(filenames)
# print filenames

# base data that we will extract from all files given
baseshelf = {}
baseshelf['timeValues'] = {}
# baseshelf['simulationData'] = {}
# baseshelf['flags'] = {}
baseshelf['errorData'] = {}

# open shelves
shelfset = {}
database = {key : dict(baseshelf) for key in range(num_filenames)}
for i in range(num_filenames):
    shelfset[i] = shelve.open(filenames[i])
    try:
        database[i]['timeValues'] = shelfset[i]['timeValues']
        # database[i]['simulationData'] = shelfset[i]['simulationData']
        # database[i]['flags'] = shelfset[i]['flags']
        database[i]['errorData'] = shelfset[i]['errorData']
    finally:
        shelfset[i].close()


# extract the information we want under assumption that the files came in
# momentum0.db, momentum1.db, momentum2.db, etc

numTimeSteps = [None]*num_filenames
dt = [None]*num_filenames
rho_maxL2Norm =[None]*num_filenames
# rho_maxH1Norm = [None]*num_filenames
rho_ell2L2Norm = [None]*num_filenames
# rho_ell2H1Norm = [None]*num_filenames

for i in range(num_filenames):
    numTimeSteps[i] = len(database[i]['timeValues'])-1 # subtract 1 to get the number of steps not the number of time values
    dt[i] = database[i]['timeValues'][1] - database[i]['timeValues'][0]

    # extract data from database i
    rho_L2Error = np.array([0] + database[i]['errorData'][0][0]['error_u_L2'])
    # rho_H1Error = np.array([0] + database[i]['errorData'][0][0]['error_u_H1'])


    # calculate the maximum time E space norm
    rho_maxL2Norm[i] = np.max(rho_L2Error)
    # rho_maxH1Norm[i] = np.max(rho_H1Error)


    print "\nMaximum in time norms for database %1d with numTimeSteps = %05d:" %(i,numTimeSteps[i])
    print "  ||v||_{Linf-L2}\t= %2.4e" %rho_maxL2Norm[i]
    # print "  ||v||_{Linf-H1}\t= %2.4e" %rho_maxH1Norm[i]


    # calculate the \ell_2 time E space norm
    rho_ell2L2Norm[i] = 0
    # rho_ell2H1Norm[i] = 0
    tnList = database[i]['timeValues']
    for j,t in enumerate(tnList):
        if j == 0: continue  # skip first step
        dtn = tnList[j] - tnList[j-1]
        rho_ell2L2Norm[i] += dtn * rho_L2Error[j]**2
        # rho_ell2H1Norm[i] += dtn * rho_H1Error[j]**2

    rho_ell2L2Norm[i] = np.sqrt(rho_ell2L2Norm[i])
    # rho_ell2H1Norm[i] = np.sqrt(rho_ell2H1Norm[i])

    if usePlots:
    	print "\\ell_2 in time norms for database %1d with numTimeSteps = %05d:" %(i,numTimeSteps[i])
    	print "  ||\\rho||_{\\ell_2-L2}\t= %2.4g" %rho_ell2L2Norm[i]
    	# print "  ||v||_{\\ell_2-H1}\t= %2.4g" %rho_ell2H1Norm[i]

    	# plot time series of errors
    	ax_L2.plot(tnList, rho_L2Error,label='dt=%1.5f' %dt[i])
    	# ax_H1.plot(tnList, rho_H1Error,label='dt=%1.5f' %dt[i])

if usePlots:
	# format plots
	ax_L2.set_xlabel('Time')
	ax_L2.set_ylabel(r'$e_h(t)$')
	ax_L2.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
	fig_L2.savefig('plotDensityErrorL2.png', bbox_inches='tight')
	plt.close(fig_L2)

	# ax_H1.set_xlabel('Time')
	# ax_H1.set_ylabel(r'$e_h(t)$')
	# ax_H1.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
	# fig_H1.savefig('plotDensityErrorH1.png', bbox_inches='tight')
	# plt.close(fig_H1)


# calculate rates of convergence and make a table
if num_filenames > 1:

    rate_rho_maxL2Norm = [0]*num_filenames
    # rate_rho_maxH1Norm = [0]*num_filenames
    rate_rho_ell2L2Norm = [0]*num_filenames
    # rate_rho_ell2H1Norm = [0]*num_filenames
    for i in range(1,num_filenames):
        rate_rho_maxL2Norm[i] = -np.log(rho_maxL2Norm[i]/rho_maxL2Norm[i-1])/np.log(float(numTimeSteps[i])/numTimeSteps[i-1])
        # rate_rho_maxH1Norm[i] = -np.log(rho_maxH1Norm[i]/rho_maxH1Norm[i-1])/np.log(float(numTimeSteps[i])/numTimeSteps[i-1])

        rate_rho_ell2L2Norm[i] = -np.log(rho_ell2L2Norm[i]/rho_ell2L2Norm[i-1])/np.log(float(numTimeSteps[i])/numTimeSteps[i-1])
        # rate_rho_ell2H1Norm[i] = -np.log(rho_ell2H1Norm[i]/rho_ell2H1Norm[i-1])/np.log(float(numTimeSteps[i])/numTimeSteps[i-1])



    print "\nnumTS   rho_maxL2   rate    rho_l2L2    rate"
    for i in range(num_filenames):
        print "%05d   %3.3e  %+1.2f    %3.3e  %+1.2f"  %(numTimeSteps[i],\
                                                    rho_maxL2Norm[i],rate_rho_maxL2Norm[i],\
                                                    rho_ell2L2Norm[i],rate_rho_ell2L2Norm[i])


    # print "\nnumTS   rho_maxL2   rate    rho_maxH1   rate"
    # for i in range(num_filenames):
    #     print "%05d   %3.3e  %+1.2f    %3.3e  %+1.2f"  %(numTimeSteps[i],\
    #                                                 rho_maxL2Norm[i],rate_rho_maxL2Norm[i],\
    #                                                 rho_maxH1Norm[i],rate_rho_maxH1Norm[i])
    # print "\nnumTS   rho_l2L2    rate    rho_l2H1    rate"
    # for i in range(num_filenames):
    #     print "%05d   %3.3e  %+1.2f    %3.3e  %+1.2f"  %(numTimeSteps[i],\
    #             rho_ell2L2Norm[i],rate_rho_ell2L2Norm[i],\
    #             rho_ell2H1Norm[i],rate_rho_ell2H1Norm[i])
