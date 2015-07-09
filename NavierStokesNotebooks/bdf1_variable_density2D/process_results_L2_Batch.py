#! /usr/bin/env python

import sys # to read in command line arguments
import os  # to check if file exists
import shelve # to open and read the database file
import copy
import numpy as np

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
database = {key : copy.deepcopy(baseshelf) for key in range(num_filenames)}
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
# density.dat  momentum.dat

rho_L2Error = np.array([0] + database[0]['errorData'][0][0]['error_u_L2'])
u_L2Error = np.array([0] + database[1]['errorData'][0][0]['error_u_L2'])
v_L2Error = np.array([0] + database[1]['errorData'][1][0]['error_u_L2'])
p_L2Error = np.array([0] + database[2]['errorData'][0][0]['error_u_L2'])
gradu_L2Error = np.array([0] + database[1]['errorData'][0][0]['error_u_H1'])
gradv_L2Error = np.array([0] + database[1]['errorData'][1][0]['error_u_H1'])
#gradp_L2Error = np.array([0] + database[2]['errorData'][0][0]['error_u_H1'])

#velpp_L2Error = np.array([0] + database[1]['errorData'][2][0]['error_velocity_L2'])

vel_L2Error = np.sqrt(u_L2Error**2 + v_L2Error**2)
vel_H1Error = np.sqrt(vel_L2Error**2 + gradu_L2Error**2 + gradv_L2Error**2)


# calculate the maximum time E space norm
vel_maxL2Norm = np.max(vel_L2Error)
vel_maxH1Norm = np.max(vel_H1Error)
#velpp_maxL2Norm = np.max(velpp_L2Error)
p_maxL2Norm = np.max(p_L2Error)
rho_maxL2Norm = np.max(rho_L2Error)

print "Maximum in time norms:"
print "  ||v||_{Linf-L2}\t= %2.4e" %vel_maxL2Norm
print "  ||v||_{Linf-H1}\t= %2.4e" %vel_maxH1Norm
#print "  ||vpp||_{Linf-L2}\t= %2.4e" %velpp_maxL2Norm
print "  ||p||_{Linf-L2}\t= %2.4e" %p_maxL2Norm
print "  ||rho||_{Linf-L2}\t= %2.4e" %rho_maxL2Norm


# calculate the \ell_2 time E space norm
rho_ell2L2Norm = p_ell2L2Norm = vel_ell2L2Norm = vel_ell2H1Norm = velpp_ell2L2Norm = 0
tnList = database[0]['timeValues']
for i,t in enumerate(tnList):
    if i > 0:
        dt = tnList[i] - tnList[i-1]
        vel_ell2L2Norm += dt * vel_L2Error[i]**2
        vel_ell2H1Norm += dt * vel_H1Error[i]**2
#        velpp_ell2L2Norm += dt * velpp_L2Error[i]**2
        p_ell2L2Norm   += dt * p_L2Error[i]**2
        rho_ell2L2Norm += dt * rho_L2Error[i]**2
#         print i, dt, rho_L2Error[i]**2, dt * rho_L2Error[i]**2,  rho_ell2L2Norm

vel_ell2L2Norm = np.sqrt(vel_ell2L2Norm)
vel_ell2H1Norm = np.sqrt(vel_ell2H1Norm)
#velpp_ell2L2Norm = np.sqrt(velpp_ell2L2Norm)
p_ell2L2Norm = np.sqrt(p_ell2L2Norm)
rho_ell2L2Norm = np.sqrt(rho_ell2L2Norm)


print "\n\\ell_2 in time norms:"
print "  ||v||_{\\ell_2-L2}\t= %2.4g" %vel_ell2L2Norm
print "  ||v||_{\\ell_2-H1}\t= %2.4g" %vel_ell2H1Norm
#print "  ||vpp||_{\\ell_2-L2}\t= %2.4g" %velpp_ell2L2Norm
print "  ||p||_{\\ell_2-L2}\t= %2.4g" %p_ell2L2Norm
print "  ||rho||_{\\ell_2-L2}\t= %2.4g" %rho_ell2L2Norm




# import matplotlib
# from matplotlib import  pyplot as plt
# from matplotlib import rc
# rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
# rc('text', usetex=True)
#
# # plot time series of errors
#
# fig = plt.figure(figsize=(8,6)) # w, h in inches
# ax1 = fig.add_subplot(221)
# ax2 = fig.add_subplot(222)
# ax3 = fig.add_subplot(223)
# ax4 = fig.add_subplot(224)
#
# ax1.set_title("$\\|\\rho(t)\\|_{L^2(\\Omega)}$")
# ax2.set_title("$\\|p(t)\\|_{L^2(\\Omega)}$")
# ax3.set_title("$\\|\mathbf{v}(t)\\|_{L^2(\\Omega)}$")
# ax4.set_title("$\\|\mathbf{v}(t)\\|_{H^1(\\Omega)}$")
#
# ax1.set_xlabel("t")
# ax2.set_xlabel("t")
# ax3.set_xlabel("t")
# ax4.set_xlabel("t")
#
# ax1.plot(so.ctx.tnList,rho_L2Error)
# ax2.plot(so.ctx.tnList,p_L2Error)
# ax3.plot(so.ctx.tnList,vel_L2Error)
# ax4.plot(so.ctx.tnList,vel_H1Error)
#
# fig.tight_layout() # spread out the plots so that they don't overlap
#
