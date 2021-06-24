#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 11:41:04 2021

@author: nadinespy
"""

import numpy as np
import scipy.linalg as la
from matplotlib import pyplot as plt
from double_redundancy_mmi import double_redundancy_mmi
import os




path_out1 = '/media/nadinespy/NewVolume/my_stuff/work/other_projects/FEP_IIT_some_thoughts/viiit_with_miguel/IntegratedInformationInVariationalInference/results'
path_out2 = '/media/nadinespy/NewVolume/my_stuff/work/other_projects/FEP_IIT_some_thoughts/viiit_with_miguel/IntegratedInformationInVariationalInference/results/plots'

# parameters to loop over
all_rho = np.array([-0.9, -0.7, -0.5, -0.3,                                     # correlation coefficients
        0.0, 0.3, 0.5, 0.7, 0.9])       

# all_timelags = np.array([1, 5, 10, 25, 50, 75, 100, 150, 200])                  # delay measured in integration steps

# all_errvar = [0.0001, 0.001, 0.01, 0.1,                                         # error variance
#          0.3, 0.5, 0.7, 1.0, 1.2]            


#all_rho = np.array([-0.9])                                     
all_errvar = np.array([0.01])
all_timelags = np.array([10])

                            
# other parameters
np.random.seed(10)
mu = np.random.randn(2)                                                         # mean of referene distribution
mu = np.array([1,-1])                                      

T = 2000
dt = 0.01                                                                       # integration step
mx = np.zeros((2,T))                                                            # mean vector of variational parameters
mx[:,0] = np.random.rand(2)               
                                      # initial mean at t=0

print('initial mean at t=0: ', mx)

# initialize matrices to store variables   
all_rho_errvar_timelags_phi = np.zeros((len(all_rho), len(all_errvar), len(all_timelags), T))
all_rho_errvar_timelags_phiR = np.zeros((len(all_rho), len(all_errvar), len(all_timelags), T))
all_rho_errvar_timelags_kl_div = np.zeros((len(all_rho), len(all_errvar), len(all_timelags), T))
all_rho_errvar_timelags_double_red = np.zeros((len(all_rho), len(all_errvar), len(all_timelags), T))


for i in range(len(all_rho)):                                                   # loop over correlations

    S = np.zeros((2,2))                                                         # covariance of reference distribution
    S[0,0] = 1
    S[1,1] = 1
    
    rho = all_rho[i]
    
    S[0,1] = np.sqrt(S[0,0]*S[1,1])*rho
    S[1,0] = S[0,1]
    A = la.inv(S)                                                               # inverse of covariance
    B = np.diag(A)                                                             
    
    for j in range(len(all_errvar)):                                            # loop over error variances
        errvar = all_errvar[j]
        
        Sx0 = np.zeros((2,2,T))                                                 # same-time covariance matrix of variational parameters
        Sx0[:,:,0] = np.eye(2)                                                  # initial covariance of parameters at t=0
        Sx1 = np.zeros((2,2,T))                                                 # time-delayed (one unit) covariance matrix of variational parameters

        for k in range(len(all_timelags)):
            
            time_lag = all_timelags[k]
            kl_div = np.zeros(T)
            
            # calculate time-lagged covariance matrices
            for n in range(T):                                                      # loop over time-points 
                t = n*dt
                mx[:,n] = (np.eye(2)-la.expm(-A*t)) @ mu + la.expm(-A*t) @ mx[:,0]
                Sx0[:,:,n] = la.expm(-A*t) @ Sx0[:,:,0]  @ la.expm(-A.T*t)  + 0.5*errvar**2 *la.inv(A)@(np.eye(2)- la.expm(-A*2*t))
                if n>time_lag:
                    s = (n-time_lag)*dt
                    Sx1[:,:,n] = la.expm(-A*(t+s)) @ Sx0[:,:,0]  + 0.5*errvar**2 * la.inv(A) @(la.expm(A*(s-t))- la.expm(A*(-t-s)))
            
                kl_div[n] = np.sum(0.5*(1-np.log(2*np.pi/B))) + 0.5*np.log(2*np.pi*np.linalg.det(np.linalg.inv(A))) + np.sum(0.5*B*np.diag(A)) + 0.5* (mx[:,n]-mu) @ A @ (mx[:,n]-mu) + 0.5*np.sum(A*Sx0[:,:,n])
                
            all_rho_errvar_timelags_kl_div[i,j,k,:] = kl_div
            
            # calculate conditional covariance matrices
            Sx1_conditional = np.zeros((2,2,T))
            Sx1_conditional_part1 = np.zeros((T))
            Sx1_conditional_part2 = np.zeros((T))

            for m in range(1+time_lag,T):                                                 # loop over time-points
                cov_matrix = np.zeros((4,4))
                cov_matrix[np.ix_([0, 1], [0, 1])]= Sx0[:,:,m-time_lag]
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
        
            # calculate double_redundancy
            all_rho_errvar_timelags_double_red[i,j,k,:] = double_redundancy_mmi(Sx0, Sx1_conditional)
            
            # calculate phi & phiR
            phi = np.zeros(T)
            phiR = np.zeros(T)
           
            #breakpoint()
            
            for n in range(time_lag,T):                                                   # loop over time-points
                try:
                    phi[n] = 0.5*np.log(np.linalg.det(Sx0[:,:,n-time_lag])/((np.linalg.det(Sx1_conditional[:,:,n]))+0j)\
                            /(Sx0[0,0,n-time_lag]/(Sx1_conditional[0,0,n]+0j)) / (Sx0[1,1,n-time_lag]/(Sx1_conditional[1,1,n]+0j)))
                        
                    phiR[n] = 0.5*np.log(np.linalg.det(Sx0[:,:,n-time_lag])/((np.linalg.det(Sx1_conditional[:,:,n]))+0j)\
                            /(Sx0[0,0,n-time_lag]/(Sx1_conditional[0,0,n]+0j)) / (Sx0[1,1,n-time_lag]/(Sx1_conditional[1,1,n]+0j))) + all_rho_errvar_timelags_double_red[i,j,k,n]
                except: #RuntimeWarning
                    pass
                    
        
            all_rho_errvar_timelags_phi[i,j,k,:] = phi
            all_rho_errvar_timelags_phiR[i,j,k,:] = phiR
            
            
            
            print(i)
   
np.save(os.path.join(path_out1, 'all_rho_errvar_timelags_kldiv.npy'), all_rho_errvar_timelags_kl_div)
np.save(os.path.join(path_out1, 'all_rho_errvar_timelags_phiR.npy'), all_rho_errvar_timelags_phiR)
np.save(os.path.join(path_out1, 'all_rho_errvar_timelags_phi.npy'), all_rho_errvar_timelags_phi) 
np.save(os.path.join(path_out1, 'all_rho_errvar_timelags_double_red.npy'), all_rho_errvar_timelags_double_red)
        
#%% plotting

       
# plots per correlation value (looping over error variances)
for i in range(len(all_rho)):
    fig, axs = plt.subplots(3, 3)
    fig.suptitle('phi time-courses for rho = {}'.format(all_rho[i]), fontsize = 10)
    
    plt.rcParams['xtick.labelsize'] = 6
    plt.rcParams['ytick.labelsize'] = 6
    plt.rcParams['axes.titlesize'] = 7
    plt.rcParams['axes.titlecolor'] = 'r'
    plt.rcParams['axes.labelsize'] = 7
    plt.rcParams['axes.labelcolor'] = 'g'
    
    if i == 4:
        plt.rcParams['lines.linewidth'] = 0.25
        custom_ylim = (-1E-14, 1E-14)
        plt.setp(axs, ylim=custom_ylim)
    
    else: 
        plt.rcParams['lines.linewidth'] = 1
        
    axs[0, 0].plot(all_rho_errvar_timelags_phi[i,0,k,:])
    axs[0, 0].set_title('error variance = {}'.format(all_errvar[0]), color = 'r', pad=10)
        
    axs[0, 1].plot(all_rho_errvar_timelags_phi[i,1,k,:])
    axs[0, 1].set_title('error variance = {}'.format(all_errvar[1]), color = 'r', pad=10)
        
    axs[0, 2].plot(all_rho_errvar_timelags_phi[i,2,k,:])
    axs[0, 2].set_title('error variance = {}'.format(all_errvar[2]), color = 'r', pad=10)
        
    axs[1, 0].plot(all_rho_errvar_timelags_phi[i,3,k,:])
    axs[1, 0].set_title('{}'.format(all_errvar[3]))
        
    axs[1, 1].plot(all_rho_errvar_timelags_phi[i,4,k,:])
    axs[1, 1].set_title('{}'.format(all_errvar[4]))
    
    axs[1, 2].plot(all_rho_errvar_timelags_phi[i,5,k,:])
    axs[1, 2].set_title('{}'.format(all_errvar[5]))

    axs[2, 0].plot(all_rho_errvar_timelags_phi[i,6,k,:])
    axs[2, 0].set_title('{}'.format(all_errvar[6]))
    axs[2, 0].set_xlabel('evolution over time')
        
    axs[2, 1].plot(all_rho_errvar_timelags_phi[i,7,k,:])
    axs[2, 1].set_title('{}'.format(all_errvar[7]))
    axs[2, 1].set_xlabel('evolution over time')
        
    axs[2, 2].plot(all_rho_errvar_timelags_phi[i,8,k,:])
    axs[2, 2].set_title('{}'.format(all_errvar[8]))
    axs[2, 2].set_xlabel('evolution over time')
        
    plt.subplots_adjust(hspace=0.5, wspace=0.4)
        
    fig.savefig(path_out2 + '/' + 'all_errvar_phi_rho'+
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
    plt.rcParams['axes.titlesize'] = 7
    plt.rcParams['axes.titlecolor'] = 'r'
    plt.rcParams['axes.labelsize'] = 7
    plt.rcParams['axes.labelcolor'] = 'g'
    plt.rcParams['lines.linewidth'] = 1
        
    axs[0, 0].plot(all_rho_errvar_timelags_phi[0,j,k,:], label ='phi')
    axs[0, 0].plot(all_rho_errvar_timelags_phiR[0,j,k,:], label ='phiR')
    axs[0, 0].plot(all_rho_errvar_timelags_double_red[0,j,k,:], label ='double-red')
    axs[0, 0].plot(all_rho_errvar_timelags_kl_div[0,j,k,:], label ='kl-div')
    axs[0, 0].set_title('rho = {}'.format(all_rho[0]))
        
    axs[0, 1].plot(all_rho_errvar_timelags_phi[1,j,k,:], label ='phi')
    axs[0, 1].plot(all_rho_errvar_timelags_phiR[1,j,k,:], label ='phiR')
    axs[0, 1].plot(all_rho_errvar_timelags_double_red[1,j,k,:], label ='double-red')
    axs[0, 1].plot(all_rho_errvar_timelags_kl_div[1,j,k,:], label ='kl-div')
    axs[0, 1].set_title('rho = {}'.format(all_rho[1]))
        
    axs[0, 2].plot(all_rho_errvar_timelags_phi[2,j,k,:], label ='phi')
    axs[0, 2].plot(all_rho_errvar_timelags_phiR[2,j,k,:], label ='phiR')
    axs[0, 2].plot(all_rho_errvar_timelags_double_red[2,j,k,:], label ='double-red')
    axs[0, 2].plot(all_rho_errvar_timelags_kl_div[2,j,k,:], label ='kl-div')
    axs[0, 2].set_title('rho = {}'.format(all_rho[2]))
    axs[0, 2].legend(loc ='upper right', fontsize = 2.5)
        
    axs[1, 0].plot(all_rho_errvar_timelags_phi[3,j,k,:], label ='phi')
    axs[1, 0].plot(all_rho_errvar_timelags_phiR[3,j,k,:], label ='phiR')
    axs[1, 0].plot(all_rho_errvar_timelags_double_red[3,j,k,:], label ='double-red')
    axs[1, 0].plot(all_rho_errvar_timelags_kl_div[3,j,k,:], label ='kl-div')
    axs[1, 0].set_title('{}'.format(all_rho[3]))
        
    axs[1, 1].plot(all_rho_errvar_timelags_phi[4,j,k,:], linewidth = 0.25, label ='phi')
    axs[1, 1].plot(all_rho_errvar_timelags_phiR[4,j,k,:], linewidth = 0.25, label ='phiR')
    axs[1, 1].plot(all_rho_errvar_timelags_double_red[4,j,k,:], linewidth = 0.25, label ='double-red')
    axs[1, 1].plot(all_rho_errvar_timelags_kl_div[4,j,k,:], linewidth = 0.25, label ='kl-div')
    axs[1, 1].set_title('{}'.format(all_rho[4]))
    axs[1, 1].set_ylim([-1E-14, 1E-14])
    
    axs[1, 2].plot(all_rho_errvar_timelags_phi[5,j,k,:], label ='phi')
    axs[1, 2].plot(all_rho_errvar_timelags_phiR[5,j,k,:], label ='phiR')
    axs[1, 2].plot(all_rho_errvar_timelags_double_red[5,j,k,:], label ='double-red')
    axs[1, 2].plot(all_rho_errvar_timelags_kl_div[5,j,k,:], label ='kl-div')
    axs[1, 2].set_title('{}'.format(all_rho[5]))

    axs[2, 0].plot(all_rho_errvar_timelags_phi[6,j,k,:], label ='phi')
    axs[2, 0].plot(all_rho_errvar_timelags_phiR[6,j,k,:], label ='phiR')
    axs[2, 0].plot(all_rho_errvar_timelags_double_red[6,j,k,:], label ='double-red')
    axs[2, 0].plot(all_rho_errvar_timelags_kl_div[6,j,k,:], label ='kl-div')
    axs[2, 0].set_title('{}'.format(all_rho[6]))
    axs[2, 0].set_xlabel('evolution over time')
        
    axs[2, 1].plot(all_rho_errvar_timelags_phi[7,j,k,:], label ='phi')
    axs[2, 1].plot(all_rho_errvar_timelags_phiR[7,j,k,:], label ='phiR')
    axs[2, 1].plot(all_rho_errvar_timelags_double_red[7,j,k,:], label ='double-red')
    axs[2, 1].plot(all_rho_errvar_timelags_kl_div[7,j,k,:], label ='kl-div')
    axs[2, 1].set_title('{}'.format(all_rho[7]))
    axs[2, 1].set_xlabel('evolution over time')
        
    axs[2, 2].plot(all_rho_errvar_timelags_phi[8,j,k,:], label ='phi')
    axs[2, 2].plot(all_rho_errvar_timelags_phiR[8,j,k,:], label ='phiR')
    axs[2, 2].plot(all_rho_errvar_timelags_double_red[8,j,k,:], label ='double-red')
    axs[2, 2].plot(all_rho_errvar_timelags_kl_div[8,j,k,:], label ='kl-div')
    axs[2, 2].set_title('{}'.format(all_rho[8]))
    axs[2, 2].set_xlabel('evolution over time')
        
    plt.subplots_adjust(hspace=0.5, wspace=0.4)
        
    fig.savefig(path_out2 + '/' + 'all_phi_doublered_phiR_kldiv_rho_errvar'+
                str(all_errvar[j]).replace('.', '')+'.png', dpi=300,
                bbox_inches='tight')  
    
    del fig
    plt.cla()
        
    # fig.savefig(r'\\media\\nadinespy\\NewVolume\\my_stuff\\work\\other_projects\\FEP_IIT_some_thoughts\\viiit_with_miguel\\results\\' 
    #             +r'all_errvar_phi_rho'+str(all_rho[i]).replace('.', '')+r'.png', dpi=300,
    #             bbox_inches='tight')  

