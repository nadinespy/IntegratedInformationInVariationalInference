#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 11:41:04 2021

@author: nadinespy
"""

import numpy as np
import scipy.linalg as la
from matplotlib import pyplot as plt


all_rho = np.array([-0.9, -0.7, -0.5, -0.3,                                     # correlation coefficients
       0.0, 0.3, 0.5, 0.7, 0.9]) 

all_errvar = [0.0001, 0.001, 0.01, 0.1,                                         # error variance
         0.3, 0.5, 0.7, 0.9, 1.2]            

np.random.seed(10)
mu = np.random.randn(2)                                                         # mean of referene distribution
mu = np.array([1,-1])                                      

T = 2000
dt = 0.01                                                                       # integration step
dn = 10                                                                         # delay measured in integration steps
mx = np.zeros((2,T))                                                            # mean vector of variational parameters
mx[:,0] = np.random.rand(2)                                                     # initial mean at t=0

print('initial mean at t=0: ', mx)
   
all_rho_errvar_phi = np.zeros((len(all_rho), len(all_errvar), T))


for i in range(len(all_rho)):                                                   # loop over correlations

    S = np.zeros((2,2))                                                         # covariance of reference distribution
    S[0,0] = 1
    S[1,1] = 1
    
    rho = all_rho[i]
    
    S[0,1] = np.sqrt(S[0,0]*S[1,1])*rho
    S[1,0] = S[0,1]
    A = la.inv(S)                                                               # inverse of covariance
    
    for j in range(len(all_errvar)):                                            # loop over error variances
        errvar = all_errvar[j]
        
        Sx0 = np.zeros((2,2,T))                                                 # same-time covariance matrix of variational parameters
        Sx0[:,:,0] = np.eye(2)                                                  # initial covariance of parameters at t=0
        Sx1 = np.zeros((2,2,T))                                                 # time-delayed (one unit) covariance matrix of variational parameters

        # calculate time-lagged covariance matrices
        for n in range(T):                                                      # loop over time-points 
            t = n*dt
            mx[:,n] = (np.eye(2)-la.expm(-A*t)) @ mu + la.expm(-A*t) @ mx[:,0]
            Sx0[:,:,n] = la.expm(-A*t) @ Sx0[:,:,0]  @ la.expm(-A.T*t)  + 0.5*errvar**2 *la.inv(A)@(np.eye(2)- la.expm(-A*2*t))
            if n>dn:
                s = (n-dn)*dt
                Sx1[:,:,n] = la.expm(-A*(t+s)) @ Sx0[:,:,0]  + 0.5*errvar**2 * la.inv(A) @(la.expm(A*(s-t))- la.expm(A*(-t-s)))
        
        # calculate conditional covariance matrices
        Sx1_conditional = np.zeros((2,2,T))
        Sx1_conditional_part1 = np.zeros((T))
        Sx1_conditional_part2 = np.zeros((T))

        for m in range(1+dn,T):                                                 # loop over time-points
            cov_matrix = np.zeros((4,4))
            cov_matrix[np.ix_([0, 1], [0, 1])]= Sx0[:,:,m-dn]
            cov_matrix[np.ix_([2, 3], [2, 3])]= Sx0[:,:,m]
            cov_matrix[np.ix_([0, 1], [2, 3])]= Sx1[:,:,m].T
            cov_matrix[np.ix_([2, 3], [0, 1])]= Sx1[:,:,m]

            # Condtition matrix on x1_t x2_t
            cond_cov = cov_matrix[np.ix_([0, 1], [0, 1])] - cov_matrix[np.ix_([0, 1], [2, 3])] @ la.pinv(cov_matrix[np.ix_([2, 3], [2, 3])]) @ cov_matrix[np.ix_([2, 3], [0, 1])]
            # Condtition matrix on x1_t
            cond_cov_1 = cov_matrix[np.ix_([0, 1, 3], [0, 1, 3])] - cov_matrix[np.ix_([0, 1, 3], [2])] @ la.pinv(cov_matrix[np.ix_([2], [2])]) @ cov_matrix[np.ix_([2], [0, 1, 3])]
            # Condtition matrix on x2_t
            cond_cov_2 = cov_matrix[np.ix_([0, 1, 2], [0, 1, 2])] - cov_matrix[np.ix_([0, 1, 2], [3])] @ la.pinv(cov_matrix[np.ix_([3], [3])]) @ cov_matrix[np.ix_([3], [0, 1, 2])]
     
            Sx1_conditional[:,:,m] = cond_cov
            Sx1_conditional_part1[m] = cond_cov_1[0,0]  # get variance of x1_t-tau conditioned on x1_t
            Sx1_conditional_part2[m] = cond_cov_2[1,1]  # get variance of x2_t-tau conditioned on x2_t
        
        # calculate phi
        phi = np.zeros(T)
        for n in range(dn,T):                                                   # loop over time-points
            try:
                phi[n] = 0.5*np.log(np.linalg.det(Sx0[:,:,n-dn])/((np.linalg.det(Sx1_conditional[:,:,n]))+0j)\
                        /(Sx0[0,0,n-dn]/(Sx1_conditional[0,0,n]+0j)) / (Sx0[1,1,n-dn]/(Sx1_conditional[1,1,n]+0j)))
            except: 
                pass
        
        all_rho_errvar_phi[i,j,:] = phi
        print(i)
   
np.save('all_rho_errvar_phi.npy', all_rho_errvar_phi)
        
#%% plotting

       
# plots per correlation value (looping over error variances)
for i in range(len(all_rho)):
    fig, axs = plt.subplots(3, 3)
    fig.suptitle('phi time-courses for rho = {}'.format(all_rho[i]), fontsize = 10)
        
    plt.rcParams['xtick.labelsize'] = 6
    plt.rcParams['ytick.labelsize'] = 6
    
    if i == 4:
        plt.rcParams['lines.linewidth'] = 0.25
        custom_ylim = (-1E-14, 1E-14)
        plt.setp(axs, ylim=custom_ylim)
    
    else: 
        plt.rcParams['lines.linewidth'] = 1
        
    axs[0, 0].plot(all_rho_errvar_phi[i,0,:])
    axs[0, 0].set_title('error variance = {}'.format(all_errvar[0]), color = 'r', pad=10)
        
    axs[0, 1].plot(all_rho_errvar_phi[i,1,:])
    axs[0, 1].set_title('error variance = {}'.format(all_errvar[1]), color = 'r', pad=10)
        
    axs[0, 2].plot(all_rho_errvar_phi[i,2,:])
    axs[0, 2].set_title('error variance = {}'.format(all_errvar[2]), color = 'r', pad=10)
        
    axs[1, 0].plot(all_rho_errvar_phi[i,3,:])
    axs[1, 0].set_title('{}'.format(all_errvar[3]), color = 'r')
        
    axs[1, 1].plot(all_rho_errvar_phi[i,4,:])
    axs[1, 1].set_title('{}'.format(all_errvar[4]), color = 'r')
    
    axs[1, 2].plot(all_rho_errvar_phi[i,5,:])
    axs[1, 2].set_title('{}'.format(all_errvar[5]), color = 'r')

    axs[2, 0].plot(all_rho_errvar_phi[i,6,:])
    axs[2, 0].set_title('{}'.format(all_errvar[6]), color = 'r')
    axs[2, 0].set_xlabel('evolution over time', fontsize = 7, color = 'g')
        
    axs[2, 1].plot(all_rho_errvar_phi[i,7,:])
    axs[2, 1].set_title('{}'.format(all_errvar[7]), color = 'r')
    axs[2, 1].set_xlabel('evolution over time', fontsize = 7, color = 'g')
        
    axs[2, 2].plot(all_rho_errvar_phi[i,8,:])
    axs[2, 2].set_title('{}'.format(all_errvar[8]), color = 'r')
    axs[2, 2].set_xlabel('evolution over time', fontsize = 7, color = 'g')
        
    plt.subplots_adjust(hspace=0.5, wspace=0.4)
        
    fig.savefig('all_errvar_phi_rho'+
                str(all_rho[i]).replace('.', '')+'.png', dpi=300,
                bbox_inches='tight')  
    
    del fig
    plt.cla()


# plots per error variance (looping over correlations)
for j in range(len(all_errvar)):
    fig, axs = plt.subplots(3, 3)
    fig.suptitle('phi time-courses for error variance = {}'.format(all_errvar[j]), fontsize = 10)
        
    plt.rcParams['xtick.labelsize'] = 6
    plt.rcParams['ytick.labelsize'] = 6
        
    axs[0, 0].plot(all_rho_errvar_phi[0,j,:], linewidth = 1)
    axs[0, 0].set_title('rho = {}'.format(all_rho[0]), color = 'r')
        
    axs[0, 1].plot(all_rho_errvar_phi[1,j,:], linewidth = 1)
    axs[0, 1].set_title('rho = {}'.format(all_rho[1]), color = 'r')
        
    axs[0, 2].plot(all_rho_errvar_phi[2,j,:], linewidth = 1)
    axs[0, 2].set_title('rho = {}'.format(all_rho[2]), color = 'r')
        
    axs[1, 0].plot(all_rho_errvar_phi[3,j,:], linewidth = 1)
    axs[1, 0].set_title('{}'.format(all_rho[3]), color = 'r')
        
    axs[1, 1].plot(all_rho_errvar_phi[4,j,:], linewidth = 0.5)
    axs[1, 1].set_title('{}'.format(all_rho[4]), color = 'r')
    axs[1, 1].set_ylim([-1E-14, 1E-14])
    
    axs[1, 2].plot(all_rho_errvar_phi[5,j,:], linewidth = 1)
    axs[1, 2].set_title('{}'.format(all_rho[5]), color = 'r')

    axs[2, 0].plot(all_rho_errvar_phi[6,j,:], linewidth = 1)
    axs[2, 0].set_title('{}'.format(all_rho[6]), color = 'r')
    axs[2, 0].set_xlabel('evolution over time', fontsize = 7, color = 'g')
        
    axs[2, 1].plot(all_rho_errvar_phi[7,j,:], linewidth = 1)
    axs[2, 1].set_title('{}'.format(all_rho[7]), color = 'r')
    axs[2, 1].set_xlabel('evolution over time', fontsize = 7, color = 'g')
        
    axs[2, 2].plot(all_rho_errvar_phi[8,j,:], linewidth = 1)
    axs[2, 2].set_title('{}'.format(all_rho[8]), color = 'r')
    axs[2, 2].set_xlabel('evolution over time', fontsize = 7, color = 'g')
        
    plt.subplots_adjust(hspace=0.5, wspace=0.4)
        
    fig.savefig('all_rho_phi_errvar'+
                str(all_errvar[j]).replace('.', '')+'.png', dpi=300,
                bbox_inches='tight')  
    
    del fig
    plt.cla()
        
    # fig.savefig(r'\\media\\nadinespy\\NewVolume\\my_stuff\\work\\other_projects\\FEP_IIT_some_thoughts\\viiit_with_miguel\\results\\' 
    #             +r'all_errvar_phi_rho'+str(all_rho[i]).replace('.', '')+r'.png', dpi=300,
    #             bbox_inches='tight')  

