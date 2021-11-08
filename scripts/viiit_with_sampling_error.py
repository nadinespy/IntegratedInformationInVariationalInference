#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 11:41:04 2021

@author: nadinespy
"""

import numpy as np
import scipy.linalg as la
from matplotlib import pyplot as plt
import os
from oct2py import octave as oc
import pandas as pd

main_path = '/media/nadinespy/NewVolume/my_stuff/work/other_projects/FEP_IIT_some_thoughts/viiit_with_miguel/IntegratedInformationInVariationalInference/scripts'
os.chdir(main_path)
from double_redundancy_mmi import double_redundancy_mmi

path_out1 = '/media/nadinespy/NewVolume/my_stuff/work/other_projects/FEP_IIT_some_thoughts/viiit_with_miguel/IntegratedInformationInVariationalInference/results'
path_out2 = '/media/nadinespy/NewVolume/my_stuff/work/other_projects/FEP_IIT_some_thoughts/viiit_with_miguel/IntegratedInformationInVariationalInference/results/plots'

#%%

# parameters to loop over
# all_rho = np.array([-0.9, -0.7, -0.5, -0.3,                                     # correlation coefficients
#        0.0, 0.3, 0.5, 0.7, 0.9])       

# all_timelags = np.array([1, 5, 10, 25, 50, 75, 100, 150, 200])                  # delay measured in integration steps

# all_errvar = [0.0001, 0.001, 0.01, 0.1,                                         # error variance
#          0.3, 0.5, 0.7, 1.0, 1.2]            


all_rho = np.array([0.0])                                     
all_errvar = np.array([0.01])
all_timelags = np.array([1])

                            
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

for i in range(len(all_rho)):                                                   # loop over correlations

    S = np.zeros((2,2))                                                         # covariance of reference distribution
    S[0,0] = 1
    S[1,1] = 1
    
    rho = all_rho[i]
    
    S[0,1] = np.sqrt(S[0,0]*S[1,1])*rho
    S[1,0] = S[0,1]
    A = la.inv(S)                                                               # inverse of covariance
    B = np.diag(A)                                                             
    
    for j in range(len(all_errvar)):   
                                                                                # loop over error variances
        errvar = all_errvar[j]
        
        Sx0 = np.zeros((2,2,T))                                                 # same-time covariance matrix of variational parameters
        Sx0[:,:,0] = np.eye(2)                                                  # initial covariance of parameters at t=0
        Sx1 = np.zeros((2,2,T))                                                 # time-delayed (one unit) covariance matrix of variational parameters

        for k in range(len(all_timelags)):
            
            time_lag = all_timelags[k]
            kldiv = np.zeros(T)
            
            # ----------------------------------------------------------------
            # CALCULATE TIME-LAGGED & CONDITIONAL COVARIANCE MATRICES
            # ----------------------------------------------------------------
            
            # time-lagged covariance matrices
            for n in range(T):                                                      # loop over time-points 
                t = n*dt
                mx[:,n] = (np.eye(2)-la.expm(-A*t)) @ mu + la.expm(-A*t) @ mx[:,0]
                Sx0[:,:,n] = la.expm(-A*t) @ Sx0[:,:,0]  @ la.expm(-A.T*t)  + 0.5*errvar**2 *la.inv(A)@(np.eye(2)- la.expm(-A*2*t))
                if n>time_lag:
                    s = (n-time_lag)*dt
                    Sx1[:,:,n] = la.expm(-A*(t+s)) @ Sx0[:,:,0]  + 0.5*errvar**2 * la.inv(A) @(la.expm(A*(s-t))- la.expm(A*(-t-s)))
            
            
            # conditional covariance matrices
            Sx1_conditional = np.zeros((2,2,T))
            Sx1_conditional_part11 = np.zeros((T))
            Sx1_conditional_part22 = np.zeros((T))
            Sx1_conditional_part12 = np.zeros((T))
            Sx1_conditional_part21 = np.zeros((T))

            for n in range(1+time_lag,T):                                                 # loop over time-points
                try:  
                    Sx1_conditional[:,:,n] = Sx0[:,:,n-1] - Sx1[:,:,n].T @ la.pinv(Sx0[:,:,n]) @ Sx1[:,:,n]         # get covariance of X_t-tau conditioned on X_t: \Sigma_(X(t-1),X(t-1)) - \Sigma(X(t-1),X(t)) * \Sigma(X(t),X(t))^(-1) * \Sigma(X(t),X(t-1))
                    Sx1_conditional_part11[n] = Sx0[0,0,n-1] - Sx1[0,0,n] * np.reciprocal(Sx0[0,0,n]) * Sx1[0,0,n]  # get variance of x1_t-tau conditioned on x1_t: \Sigma_(x1(t-tau),x1(t-tau)) - \Sigma_x1(t-tau)x1(t) * \Sigma_(x1(t),x1(t))^(-1) * \Sigma_x1(t)x1(t-tau)
                    Sx1_conditional_part22[n] = Sx0[1,1,n-1] - Sx1[1,1,n] * np.reciprocal(Sx0[1,1,n]) * Sx1[1,1,n]  # get variance of x2_t-tau conditioned on x2_t: \Sigma_(x2(t-tau),x2(t-tau)) - \Sigma_x2(t-tau)x2(t) * \Sigma_(x2(t),x2(t))^(-1) * \Sigma_x1(t)x2(t-tau)
                    Sx1_conditional_part12[n] = Sx0[0,0,n-1] - Sx1[1,0,n] * np.reciprocal(Sx0[1,1,n]) * Sx1[1,0,n]  # get variance of x1_t-tau conditioned on x2_t: \Sigma_(x1(t-tau),x1(t-tau)) - \Sigma_x1(t-tau)x2(t) * \Sigma_(x2(t),x2(t))^(-1) * \Sigma_x2(t)x1(t-tau)
                    Sx1_conditional_part21[n] = Sx0[1,1,n-1] - Sx1[0,1,n] * np.reciprocal(Sx0[0,0,n]) * Sx1[0,1,n]  # get variance of x2_t-tau conditioned on x1_t: \Sigma_(x2(t-tau),x2(t-tau)) - \Sigma_x2(t-tau)x1(t) * \Sigma_(x1(t),x1(t))^(-1) * \Sigma_x1(t)x2(t-tau)
                except:
                    pass 
        
            # ----------------------------------------------------------------
            # CALCULATE DOUBLE-REDUNDANCY
            # ----------------------------------------------------------------
            
            double_red = double_redundancy_mmi(Sx0, Sx1_conditional, Sx1_conditional_part11, Sx1_conditional_part22, Sx1_conditional_part12, Sx1_conditional_part21, time_lag)
            
            # ----------------------------------------------------------------
            # CALCULATE PHI, PHI-R, KL-DIVERGENCE, & PHIID-BASED QUANTITIES
            # ----------------------------------------------------------------
            
            
            oc.addpath(main_path)
            oc.javaaddpath(main_path+'/infodynamics.jar')
            oc.eval('pkg load statistics')
                
            for n in range(time_lag,T):                                                   # loop over time-points
            
                # ----------------------------------------------------------------
                # KL DIVERGENCE
                # ----------------------------------------------------------------
                
                kldiv = np.sum(0.5*(1-np.log(2*np.pi/B))) + 0.5*np.log(2*np.pi*np.linalg.det(np.linalg.inv(A))) + np.sum(0.5*B*np.diag(A)) + 0.5* (mx[:,n]-mu) @ A @ (mx[:,n]-mu) + 0.5*np.sum(A*Sx0[:,:,n])
                
                
                # ----------------------------------------------------------------
                # PHI & PHI-R
                # ----------------------------------------------------------------
                
                try:
                    phi = ((0.5*np.log(np.linalg.det(Sx0[:,:,n-time_lag])/((np.linalg.det(Sx1_conditional[:,:,n]))+0j))\
                            /(Sx0[0,0,n-time_lag]/(Sx1_conditional[0,0,n]+0j)) / (Sx0[1,1,n-time_lag]/(Sx1_conditional[1,1,n]+0j)))+0j).real
                        
                    phiR = phi + double_red[n]
                except: #RuntimeWarning
                    phi = float('NaN')
                    phiR = float('NaN')
                
                # ----------------------------------------------------------------
                # PHIID BASED QUANTITIES
                # ----------------------------------------------------------------
            
                # simulate time-series with given covariance matrix
            
                time_series = np.random.multivariate_normal(mx[:,n], Sx0[:,:,n], T).T
                phiid = oc.PhiIDFull(time_series, time_lag, 'mmi')

                phiid_dict = {'rtr': phiid.rtr, 'rtx': phiid.rtx, 'rty': phiid.rty, 'rts': phiid.rts, 'xtr': phiid.xtr, 'xtx': phiid.xtx, \
                        'xty': phiid.xty, 'xts': phiid.xts, 'ytr': phiid.ytr, 'ytx': phiid.ytx, 'yty': phiid.yty, 'yts': phiid.yts, \
                        'str': phiid.str, 'stx': phiid.stx, 'sty': phiid.sty, 'sts': phiid.sts}
    
                # -----------------------------------------------------------------------------
                # calculate synergistic/emergent capacity, downward causation, 
                # causal decoupling and store everything in nested dictionary
                # -----------------------------------------------------------------------------

                # Syn(X_t;X_t-1) (synergistic capacity of the system) 
                # Un (Vt;Xt'|Xt) (causal decoupling - the top term in the lattice) 
                # Un(Vt;Xt'α|Xt) (downward causation) 

                # synergy (only considering the synergy that the sources have, not the target): 
                # {12} --> {1}{2} + {12} --> {1} + {12} --> {2} + {12} --> {12} 
 
                # causal decoupling: {12} --> {12}

                # downward causation: 
                # {12} --> {1}{2} + {12} --> {1} + {12} --> {2}
                
                # phi =     - {1}{2}-->{1}{2}                                                   (double-redundancy)
                #           + {12}-->{12}                                                       (causal decoupling)
                #           + {12}-->{1} + {12}-->{2} + {12}-->{1}{2}                           (downward causation)
                #           + {1}{2}-->{12} + {1}-->{12} + {2}-->{12}                           (upward causation)
                #           + {1}-->{2} + {2}-->{1}                                             (transfer)

                # synergy = causal decoupling + downward causation + upward causation
                

                emergence_capacity_phiid = phiid_dict["str"] + phiid_dict["stx"] + phiid_dict["sty"] + phiid_dict["sts"]
                downward_causation_phiid = phiid_dict["str"] + phiid_dict["stx"] + phiid_dict["sty"]
                synergy_phiid = emergence_capacity_phiid + phiid_dict["rts"] + phiid_dict["xts"] + phiid_dict["yts"]
                transfer_phiid = phiid_dict["xty"] + phiid_dict["ytx"]
                phi_phiid = - phiid_dict["rtr"] + synergy_phiid + transfer_phiid
                phiR_phiid = phi_phiid + phiid_dict["rtr"]
                
            
            
                super_df = pd.DataFrame({'correlation': [all_rho[i]], 'error_variance': [all_errvar[j]], 'time_lag': [all_timelags[k]], 'time_point' : [n+1],
                                         'phi': [phi], 'phiR': [phiR], 'kldiv': [kldiv], 'double_red': [double_red[n]], 'rtr': [phiid.rtr], 'rtx': [phiid.rtx], 
                                         'rty': [phiid.rty], 'rts': [phiid.rts], 'xtr': [phiid.xtr], 'xtx': [phiid.xtx], 'xty': [phiid.xty], 'xts': [phiid.xts], 
                                         'ytr': [phiid.ytr], 'ytx': [phiid.ytx], 'yty': [phiid.yty], 'yts': [phiid.yts], 'str': [phiid.str], 'stx': [phiid.stx], 
                                         'sty': [phiid.sty], 'sts': [phiid.sts], 'synergy_phiid': [synergy_phiid], 'transfer_phiid': [transfer_phiid], 
                                         'emergence_capacity_phiid': [emergence_capacity_phiid], 'downward_causation_phiid': [downward_causation_phiid], 
                                         'phi_phiid': [phi_phiid], 'phiR_phiid': [phiR_phiid]})
                

np.save(os.path.join(path_out1, 'super_df.npy'), super_df)


# TO DO 
# save phiid_dict for each time-point, calculate combinations of atoms on the fly instead of storing everything in single variables
# find out why lines are not transparent for plot with atoms

        
#%% plotting

# ----------------------------------------------------------------------------
# plots for phi, phi-R, kl-div, double-red
# ----------------------------------------------------------------------------
      
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
    axs[0, 0].plot(all_rho_errvar_timelags_kldiv[0,j,k,:], label ='kl-div')
    axs[0, 0].set_title('rho = {}'.format(all_rho[0]))
        
    axs[0, 1].plot(all_rho_errvar_timelags_phi[1,j,k,:], label ='phi')
    axs[0, 1].plot(all_rho_errvar_timelags_phiR[1,j,k,:], label ='phiR')
    axs[0, 1].plot(all_rho_errvar_timelags_double_red[1,j,k,:], label ='double-red')
    axs[0, 1].plot(all_rho_errvar_timelags_kldiv[1,j,k,:], label ='kl-div')
    axs[0, 1].set_title('rho = {}'.format(all_rho[1]))
        
    axs[0, 2].plot(all_rho_errvar_timelags_phi[2,j,k,:], label ='phi')
    axs[0, 2].plot(all_rho_errvar_timelags_phiR[2,j,k,:], label ='phiR')
    axs[0, 2].plot(all_rho_errvar_timelags_double_red[2,j,k,:], label ='double-red')
    axs[0, 2].plot(all_rho_errvar_timelags_kldiv[2,j,k,:], label ='kl-div')
    axs[0, 2].set_title('rho = {}'.format(all_rho[2]))
    axs[0, 2].legend(loc ='upper right', fontsize = 2.5)
        
    axs[1, 0].plot(all_rho_errvar_timelags_phi[3,j,k,:], label ='phi')
    axs[1, 0].plot(all_rho_errvar_timelags_phiR[3,j,k,:], label ='phiR')
    axs[1, 0].plot(all_rho_errvar_timelags_double_red[3,j,k,:], label ='double-red')
    axs[1, 0].plot(all_rho_errvar_timelags_kldiv[3,j,k,:], label ='kl-div')
    axs[1, 0].set_title('{}'.format(all_rho[3]))
        
    axs[1, 1].plot(all_rho_errvar_timelags_phi[4,j,k,:], label ='phi')
    axs[1, 1].plot(all_rho_errvar_timelags_phiR[4,j,k,:], label ='phiR')
    axs[1, 1].plot(all_rho_errvar_timelags_double_red[4,j,k,:], label ='double-red')
    axs[1, 1].plot(all_rho_errvar_timelags_kldiv[4,j,k,:], label ='kl-div')
    axs[1, 1].set_title('{}'.format(all_rho[4]))
    #axs[1, 1].set_ylim([-1E-14, 1E-14])
    
    axs[1, 2].plot(all_rho_errvar_timelags_phi[5,j,k,:], label ='phi')
    axs[1, 2].plot(all_rho_errvar_timelags_phiR[5,j,k,:], label ='phiR')
    axs[1, 2].plot(all_rho_errvar_timelags_double_red[5,j,k,:], label ='double-red')
    axs[1, 2].plot(all_rho_errvar_timelags_kldiv[5,j,k,:], label ='kl-div')
    axs[1, 2].set_title('{}'.format(all_rho[5]))

    axs[2, 0].plot(all_rho_errvar_timelags_phi[6,j,k,:], label ='phi')
    axs[2, 0].plot(all_rho_errvar_timelags_phiR[6,j,k,:], label ='phiR')
    axs[2, 0].plot(all_rho_errvar_timelags_double_red[6,j,k,:], label ='double-red')
    axs[2, 0].plot(all_rho_errvar_timelags_kldiv[6,j,k,:], label ='kl-div')
    axs[2, 0].set_title('{}'.format(all_rho[6]))
    axs[2, 0].set_xlabel('evolution over time')
        
    axs[2, 1].plot(all_rho_errvar_timelags_phi[7,j,k,:], label ='phi')
    axs[2, 1].plot(all_rho_errvar_timelags_phiR[7,j,k,:], label ='phiR')
    axs[2, 1].plot(all_rho_errvar_timelags_double_red[7,j,k,:], label ='double-red')
    axs[2, 1].plot(all_rho_errvar_timelags_kldiv[7,j,k,:], label ='kl-div')
    axs[2, 1].set_title('{}'.format(all_rho[7]))
    axs[2, 1].set_xlabel('evolution over time')
        
    axs[2, 2].plot(all_rho_errvar_timelags_phi[8,j,k,:], label ='phi')
    axs[2, 2].plot(all_rho_errvar_timelags_phiR[8,j,k,:], label ='phiR')
    axs[2, 2].plot(all_rho_errvar_timelags_double_red[8,j,k,:], label ='double-red')
    axs[2, 2].plot(all_rho_errvar_timelags_kldiv[8,j,k,:], label ='kl-div')
    axs[2, 2].set_title('{}'.format(all_rho[8]))
    axs[2, 2].set_xlabel('evolution over time')
        
    plt.subplots_adjust(hspace=0.5, wspace=0.4)
        
    fig.savefig(path_out2 + '/' + 'all_phi_doublered_phiR_kldiv_rho_errvar'+
                str(all_errvar[j]).replace('.', '')+'.png', dpi=300,
                bbox_inches='tight')  
    
    del fig
    plt.cla()
    
# ----------------------------------------------------------------------------
# plots for phiid atoms & compositions
# ----------------------------------------------------------------------------

phiid_terms_dict = {'all_rho_errvar_timelags_synergy_phiid', 'all_rho_errvar_timelags_downward_causation_phiid',
              'all_rho_errvar_timelags_double_red_phiid', 'all_rho_errvar_timelags_transfer_phiid', 
              'all_rho_errvar_timelags_causal_decoupling_phiid', 'all_rho_errvar_timelags_emergence_capacity_phiid'}

plt.rcParams['xtick.labelsize'] = 6
plt.rcParams['ytick.labelsize'] = 6
plt.rcParams['axes.titlesize'] = 7
plt.rcParams['axes.titlecolor'] = 'r'
plt.rcParams['axes.labelsize'] = 7
plt.rcParams['axes.labelcolor'] = 'g'
plt.rcParams['lines.linewidth'] = 0.2
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=["b", "0.4"]) 
    
# plots per error variance (looping over correlations)
for j in range(len(all_errvar)):
    for h in phiid_terms_dict:

        for f in range(len(all_rho)):

            moving_average_window = 120
            moving_average = np.int(np.float(T/moving_average_window))                                         # adjust according to iteration length; 100 is good for num_iters_GD = 5000, 30 for num_iters_GD = 500 and 250; 40 for num_iters_GD = 1000; 50 for num_iters_GD = 2000; 160 for num_iters_GD = 12000
            moving_average_vector = np.array(range(0,len(super_dict[h][f,j,k,:]),1))
            moving_average_vector = moving_average_vector.astype(np.float64)
            moving_average_vector.fill(np.nan) 
            raw_vector = np.ma.array(super_dict[h][f,j,k,:], mask=np.isnan(super_dict[h][f,j,k,:]))

            for i in range(len(super_dict[h][f,j,k,:])):
                if i < moving_average:
                    number_of_numbers_in_sum = np.count_nonzero(~np.isnan(raw_vector[i:i+moving_average]))
                    moving_average_vector[i] = np.sum(raw_vector[i:i+moving_average])/number_of_numbers_in_sum
                elif i > (len(raw_vector)-moving_average):
                    number_of_numbers_in_sum = np.count_nonzero(~np.isnan(raw_vector[i-moving_average:i]))
                    moving_average_vector[i] = np.sum(raw_vector[i-moving_average:i])/number_of_numbers_in_sum
                else:    
                    number_of_numbers_in_sum = np.count_nonzero(~np.isnan(raw_vector[i-moving_average:i+moving_average]))
                    moving_average_vector[i] = np.sum(raw_vector[i-moving_average:i+moving_average])/number_of_numbers_in_sum #Originally, it was (moving_average*2), but we need to divide by the numbers that went into the calculation of the sum (which has ignored nan values).


            fig, axs = plt.subplots(1, 1) 
            fig.suptitle(h.replace('all_rho_errvar_timelags_',''), fontsize = 10)  
            axs.plot(super_dict[h][f,j,k,:])
            axs.plot(moving_average_vector, label ='moving-average', color = 'k', linewidth = 1)
            axs.set_title('rho = {}'.format(all_rho[f]))
            
            fig.savefig(path_out2 + '/' + h + '_' + str(all_errvar[j]).replace('.', '') + '_' + str(all_rho[f]).replace('.', '') + '.png', dpi=300, bbox_inches='tight')  
            plt.cla()
            del fig
        
    
    # fig.savefig(r'\\media\\nadinespy\\NewVolume\\my_stuff\\work\\other_projects\\FEP_IIT_some_thoughts\\viiit_with_miguel\\results\\' 
    #             +r'all_errvar_phi_rho'+str(all_rho[i]).replace('.', '')+r'.png', dpi=300,
    #             bbox_inches='tight')  

