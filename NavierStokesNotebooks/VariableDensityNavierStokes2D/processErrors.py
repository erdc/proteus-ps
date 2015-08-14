#!/usr/bin/env python

import sys # to read in command line arguments
import os  # to check if file exists
import shelve # to open and read the database file
import argparse # for options parsing
import numpy as np


parser = argparse.ArgumentParser(description='Process some error database files.')

parser.add_argument('-np','--num_proc', dest='num_proc',
                    action='store',  default=1,
                    type=int, choices=range(1, 17),
                    required=False, metavar='NUM_PROC',
                    help="""specify the number of processors which saved the data in results/*_p.db format and need to be analyzed. (default: 1)"""  )

parser.add_argument('-s','--short_name_string', dest='short_names', default=None, # type = string is default
                    action='append', required=False, metavar='SHORT_NAMES',
                    help="""Give the shortened names for multiple processor files, mainly the base name ex results/velocity_BDF2_dt_0_100000  which is short for in the first processor, results/velocity_BDF2_dt_0_100000_0.db.  Then use -np option to tell it how many to expect of that form""")

parser.add_argument('-p','--plot', dest='usePlots',
                    action='store_true', default=False,
                    help='turn on plotting of error time series')

parser.add_argument('-H1','--useH1Norm', dest='useH1Norm',
                    action='store_true', default=False,
                    help='Include calculations using H1 norms if available')

parser.add_argument('-t','--type', dest='type', default='variable',
                    choices=['velocity','pressure','density','variable'],
                    required=False, help="""Give variable indicator name for string outputting.""")

parser.add_argument('file_names', nargs=argparse.REMAINDER,
                    help="""If output is from a single processor, you can still list the files in order at the end without flags and it will pick them up and analyze them""")

args = parser.parse_args()


# type of variable name for short snippets
short_type = args.type[0:3] # extract the first three letters for short type name strings
if short_type == 'den':
    short_type = 'rho'





# read in file names and test if they can be found.  if so
# then add them to our list of filenames to be processed.
#
# In the case that the filenames are given exactly, then we simply add them to filenames
#
# but if the short names are given which is for use in the case of multiple processors
# then we only save the short name which then needs to have
#  +"_%d.db" %p  for p in range(0, args.num_proc)
# added to the short name to create the list of filenames.
if args.short_names is not None:
    use_parallel_ext = True
    filenames = {}
    if len(args.short_names) > 0:
        counter = 0
        for i in range(0, len(args.short_names)):
            addName = True
            for p in range(0,args.num_proc):
                test_filename = args.short_names[i] + "_%d.db" %p
                if os.path.exists(test_filename):
                    print "** File %s found!" %test_filename
                else:
                    print "** File %s not found!"%test_filename
                    addName = False
            if addName:
                print "%d: All *_p.db files found for %s and will be processed!" %(counter, args.short_names[i])
                filenames[counter] = args.short_names[i]
                counter+=1
else:
    use_parallel_ext = False
    filenames = {}
    if len(args.file_names) > 0:
        counter = 0
        for i in range(0,len(args.file_names)):
            test_filename = args.file_names[i]
            if os.path.exists(test_filename):
                print "%d: File %s found and will be processed!" %(counter, test_filename)
                filenames[counter] = test_filename
                counter+=1
            else:
                print "** File %s not found!"%test_filename

num_filenames = len(filenames)

if num_filenames is int(0):
    sys.exit("Error: No files to process!")


if args.usePlots:
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
    if args.useH1Norm:
        fig_H1, ax_H1 = plt.subplots(nrows=1, ncols=1)


# preallocate space for the results
numTimeSteps = [0.0]*num_filenames
dt = [0.0]*num_filenames
maxL2Norm =[0.0]*num_filenames
ell2L2Norm = [0.0]*num_filenames
if args.useH1Norm:
    maxH1Norm = [0.0]*num_filenames
    ell2H1Norm = [0.0]*num_filenames

# open shelves
shelfvalue_p = {}
for i in range(num_filenames):
    if use_parallel_ext:
        for p in range(0,args.num_proc):
            filename_p = filenames[i] + "_%d.db"%p
            shelfvalue_p = shelve.open(filename_p)
            try:
                if p is 0:
                    tnList = shelfvalue_p['timeValues']

                    numTimeSteps[i] = len(tnList)-1 # subtract 1 to get the number of steps not the number of time values
                    dt[i] = np.max(np.array(tnList[1:-1]-np.array(tnList[0:-2])))

                errorData_p =  shelfvalue_p['errorData']
                # test for existence of H1 norm data
                if args.useH1Norm:
                    for j in errorData_p:
                        if 'error_u_H1' not in errorData_p[j][0]:
                            print "Error: The key 'error_u_H1' is not in shelf['errorData'][%d][0] for %s, so setting useH1Norm = False." %(j,filename_p)
                            args.useH1Norm = False

                # cycle through the components if they exist and take l2 norm of components
                L2Error_p = np.array([0.0]*(len(errorData_p[0][0]['error_u_L2'])+1))
                if args.useH1Norm:
                    H1Error_p = np.array([0.0]*(len(errorData_p[0][0]['error_u_H1'])+1))

                for j in errorData_p:
                    L2Error_p += np.array([0] + errorData_p[j][0]['error_u_L2'])**2
                    if args.useH1Norm:
                        H1Error_p += np.array([0] + errorData_p[0][0]['error_u_H1'])**2

                # compute the maximum in time norms
                L2Error_p = np.sqrt(L2Error_p)
                maxL2Norm_p = np.max(L2Error_p)
                if args.useH1Norm:
                    H1Error_p = np.sqrt(H1Error_p)
                    maxH1Norm_p = np.max(H1Error_p)


                # calculate the \ell_2 time norms
                ell2L2Norm_p = 0
                if args.useH1Norm:
                    ell2H1Norm_p = 0
                for j,t in enumerate(tnList):
                    if j == 0: continue  # skip first step
                    dtn = tnList[j] - tnList[j-1]
                    ell2L2Norm_p += dtn * L2Error_p[j]**2
                    if args.useH1Norm:
                        ell2H1Norm_p += dtn * H1Error_p**2

                ell2L2Norm_p = np.sqrt(ell2L2Norm_p)
                if args.useH1Norm:
                    ell2H1Norm_p = np.sqrt(ell2H1Norm_p)

                # consolidate all the _p terms into global
                maxL2Norm[i] = np.max(maxL2Norm[i], maxL2Norm_p)
                ell2L2Norm[i] = np.sqrt(ell2L2Norm[i]**2 + ell2L2Norm_p**2)
                if args.useH1Norm:
                    maxH1Norm[i] = np.max(maxH1Norm[i], maxH1Norm_p)
                    ell2H1Norm[i] = np.sqrt(ell2H1Norm[i]**2 + ell2H1Norm_p**2)

            finally:
                shelfvalue_p.close()


        # output norm stuff
        print "\nMaximum in time norms for database %1d with numTimeSteps = %05d:" %(i,numTimeSteps[i])
        print "  ||\\%s||_{Linf-L2}\t= %2.4e" %(short_type,maxL2Norm[i])
        if args.useH1Norm:
            print "  ||\\%s||_{Linf-H1}\t= %2.4e" %(short_type,maxH1Norm[i])

        print "\\ell_2 in time norms for database %1d with numTimeSteps = %05d:" %(i,numTimeSteps[i])
        print "  ||\\%s||_{\\ell_2-L2}\t= %2.4g" %(short_type,ell2L2Norm[i])
        if args.useH1Norm:
            print "  ||\\%s||_{\\ell_2-H1}\t= %2.4g" %(short_type,ell2H1Norm[i])

        if args.usePlots:
            # plot time series of errors
            ax_L2.plot(tnList, L2Error,label='dt=%1.5f' %dt[i])
            if args.useH1Norm:
                ax_H1.plot(tnList, H1Error,label='dt=%1.5f' %dt[i])


    else:
        shelfvalue = shelve.open(filenames[i])
        try:
            tnList = shelfvalue['timeValues']
            numTimeSteps[i] = len(tnList)-1 # subtract 1 to get the number of steps not the number of time values
            dt[i] = np.max(np.array(tnList[1:-1]-np.array(tnList[0:-2])))

            errorData =  shelfvalue['errorData']

            # test for existence of H1 norm data
            if args.useH1Norm:
                for j in errorData:
                    if 'error_u_H1' not in errorData[j][0]:
                        print "Error: The key 'error_u_H1' is not in shelf['errorData'][%d][0] for %s, so setting useH1Norm = False." %(j,filenames[i])
                        args.useH1Norm = False

            # cycle through the components if they exist and take l2 norm of components
            L2Error = np.array([0.0]*(len(errorData[0][0]['error_u_L2'])+1))
            if args.useH1Norm:
                H1Error = np.array([0.0]*(len(errorData[0][0]['error_u_H1'])+1))

            for j in errorData:
                L2Error += np.array([0] + errorData[j][0]['error_u_L2'])**2
                if args.useH1Norm:
                    H1Error += np.array([0] + errorData[0][0]['error_u_H1'])**2

            # compute the maximum in time norms
            L2Error = np.sqrt(L2Error)
            maxL2Norm[i] = np.max(L2Error)
            if args.useH1Norm:
                H1Error = np.sqrt(H1Error)
                maxH1Norm[i] = np.max(H1Error)

            # calculate the \ell_2 time norms
            ell2L2Norm[i] = 0
            if args.useH1Norm:
                ell2H1Norm[i] = 0

            for j,t in enumerate(tnList):
                if j == 0: continue  # skip first step
                dtn = tnList[j] - tnList[j-1]
                ell2L2Norm[i] += dtn * L2Error[j]**2
                if args.useH1Norm:
                    ell2H1Norm[i] += dtn * H1Error[j]**2

            ell2L2Norm[i] = np.sqrt(ell2L2Norm[i])
            if args.useH1Norm:
                ell2H1Norm[i] = np.sqrt(ell2H1Norm[i])


            # output results
            print "\nMaximum in time norms for database %1d with numTimeSteps = %05d:" %(i,numTimeSteps[i])
            print "  ||\\%s||_{Linf-L2}\t= %2.4e" %(short_type,maxL2Norm[i])
            if args.useH1Norm:
                print "  ||\\%s||_{Linf-H1}\t= %2.4e" %(short_type,maxH1Norm[i])

            print "\\ell_2 in time norms for database %1d with numTimeSteps = %05d:" %(i,numTimeSteps[i])
            print "  ||\\%s||_{\\ell_2-L2}\t= %2.4g" %(short_type,ell2L2Norm[i])
            if args.useH1Norm:
                print "  ||\\%s||_{\\ell_2-H1}\t= %2.4g" %(short_type,ell2H1Norm[i])

            if args.usePlots:
            	# plot time series of errors
            	ax_L2.plot(tnList, L2Error,label='dt=%1.5f' %dt[i])
                if args.useH1Norm:
            	    ax_H1.plot(tnList, H1Error,label='dt=%1.5f' %dt[i])

        finally:
            shelfvalue.close()



# ------- Done extracting information and now time to calculate error rates -----------

if args.usePlots:
    # format plots
    ax_L2.set_xlabel('Time')
    ax_L2.set_ylabel(r'$e_h(t)$')
    ax_L2.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    fig_L2.savefig('%sTimeSeriesErrorL2.png' %args.type, bbox_inches='tight')
    plt.close(fig_L2)
    if args.useH1Norm:
        ax_H1.set_xlabel('Time')
        ax_H1.set_ylabel(r'$e_h(t)$')
        ax_H1.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        fig_H1.savefig('%sTimeSeriesErrorH1.png', bbox_inches='tight')
        plt.close(fig_H1)


# calculate rates of convergence and make a table
if num_filenames > 1:

    rate_maxL2Norm = [0]*num_filenames
    rate_ell2L2Norm = [0]*num_filenames
    if args.useH1Norm:
        rate_maxH1Norm = [0]*num_filenames
        rate_ell2H1Norm = [0]*num_filenames

    for i in range(1,num_filenames):
        rate_maxL2Norm[i] = -np.log(maxL2Norm[i]/maxL2Norm[i-1])/np.log(float(numTimeSteps[i])/numTimeSteps[i-1])
        rate_ell2L2Norm[i] = -np.log(ell2L2Norm[i]/ell2L2Norm[i-1])/np.log(float(numTimeSteps[i])/numTimeSteps[i-1])
        if args.useH1Norm:
            rate_maxH1Norm[i] = -np.log(maxH1Norm[i]/maxH1Norm[i-1])/np.log(float(numTimeSteps[i])/numTimeSteps[i-1])
            rate_ell2H1Norm[i] = -np.log(ell2H1Norm[i]/ell2H1Norm[i-1])/np.log(float(numTimeSteps[i])/numTimeSteps[i-1])

    print "\nnumTS   %s_maxL2   rate    %s_l2L2    rate" %(short_type, short_type)
    for i in range(num_filenames):
        print "%05d   %3.3e  %+1.2f    %3.3e  %+1.2f"  %(numTimeSteps[i],\
                                                    maxL2Norm[i],rate_maxL2Norm[i],\
                                                    ell2L2Norm[i],rate_ell2L2Norm[i])

    if args.useH1Norm:
        print "\nnumTS   %s_maxH1   rate    %s_l2H1    rate" %(short_type, short_type)
        for i in range(num_filenames):
            print "%05d   %3.3e  %+1.2f    %3.3e  %+1.2f"  %(numTimeSteps[i],\
                                                        maxH1Norm[i],rate_maxH1Norm[i],\
                                                        ell2H1Norm[i],rate_ell2H1Norm[i])

# calculate rates of convergence and make a table
if num_filenames > 1:
    rate_maxL2Norm = [0]*num_filenames
    rate_ell2L2Norm = [0]*num_filenames
    if args.useH1Norm:
        rate_maxH1Norm = [0]*num_filenames
        rate_ell2H1Norm = [0]*num_filenames

    for i in range(1,num_filenames):
        rate_maxL2Norm[i] = np.log(maxL2Norm[i]/maxL2Norm[i-1])/np.log(float(dt[i])/dt[i-1])
        rate_ell2L2Norm[i] = np.log(ell2L2Norm[i]/ell2L2Norm[i-1])/np.log(float(dt[i])/dt[i-1])
        if args.useH1Norm:
            rate_maxH1Norm[i] = -np.log(maxH1Norm[i]/maxH1Norm[i-1])/np.log(float(numTimeSteps[i])/numTimeSteps[i-1])
            rate_ell2H1Norm[i] = -np.log(ell2H1Norm[i]/ell2H1Norm[i-1])/np.log(float(numTimeSteps[i])/numTimeSteps[i-1])

    print "\nmax dt      %s_maxL2   rate    %s_l2L2    rate" %(short_type, short_type)
    for i in range(num_filenames):
        print "%1.3e   %3.3e  %+1.2f    %3.3e  %+1.2f"  %(dt[i],\
                                                    maxL2Norm[i],rate_maxL2Norm[i],\
                                                    ell2L2Norm[i],rate_ell2L2Norm[i])

    if args.useH1Norm:
        print "\nmax dt      %s_maxH1   rate    %s_l2H1    rate" %(short_type, short_type)
        for i in range(num_filenames):
            print "%1.3e   %3.3e  %+1.2f    %3.3e  %+1.2f"  %(dt[i],\
                                                        maxH1Norm[i],rate_maxH1Norm[i],\
                                                        ell2H1Norm[i],rate_ell2H1Norm[i])
