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
from joblib import Parallel, delayed
from itertools import product

main_path = '/media/nadinespy/NewVolume/my_stuff/work/other_projects/FEP_IIT_some_thoughts/viiit_with_miguel/IntegratedInformationInVariationalInference/scripts'
os.chdir(main_path)

import iit_in_vi as iv

oc.addpath(main_path)
oc.javaaddpath(main_path+'/infodynamics.jar')
oc.eval('pkg load statistics')
    
path_out1 = '/media/nadinespy/NewVolume/my_stuff/work/other_projects/FEP_IIT_some_thoughts/viiit_with_miguel/IntegratedInformationInVariationalInference/results/analyses/'
path_out2 = '/media/nadinespy/NewVolume/my_stuff/work/other_projects/FEP_IIT_some_thoughts/viiit_with_miguel/IntegratedInformationInVariationalInference/results/plots/'

#%%

# parameters to loop over
#all_rho = np.array([-0.9, -0.7, -0.5, -0.3,                                     # correlation coefficients
#        0.0, 0.3, 0.5, 0.7, 0.9])       

#all_timelags = np.array([1, 5, 10, 25, 50, 75, 100, 150, 200])                  # delay measured in integration steps

#all_errvar = [0.0001, 0.001, 0.01, 0.1,                                         # error variance
#         0.3, 0.5, 0.7, 1.0, 1.2]            


all_rho = np.array([0.5])                                     
all_errvar = np.array([0.01])
all_timelags = np.array([10])

                            
# other parameters
np.random.seed(10)
mu = np.random.randn(2)                                                         # mean of referene distribution
mu = np.array([1,-1])                                      

T = 2000
dt = 0.01                                                                       # integration step
mx = np.zeros((2,T))                                                            # mean vector of variational parameters
mx[:,0] = np.random.rand(2)                                                     # initial random values for mean variational parameters at t = 1 
initial_mx = mx[:,0]         

print('initial mean at t = 0: ', mx)


def get_results_from_model(rho, errvar, time_lag, T, dt, mx, mu):
    """docstring"""
    
    df = []
    
    S = np.zeros((2,2))                                                         # covariance of reference distribution
    S[0,0] = 1
    S[1,1] = 1
    
    
    S[0,1] = np.sqrt(S[0,0]*S[1,1]) * rho
    S[1,0] = S[0,1]
    A = la.inv(S)                                                               # inverse of covariance
    B = np.diag(A)                                                             
     
    
    Sx0 = np.zeros((2,2,T))                                                     # same-time covariance matrix of variational parameters
    Sx0[:,:,0] = np.eye(2)  
    initial_covariance = Sx0[:,:,0]                                             # initial covariance of parameters at t = 0
    Sx1 = np.zeros((2,2,T))                                                     # time-delayed (one unit) covariance matrix of variational parameters
    
    kldiv = np.zeros(T)
    
    # ----------------------------------------------------------------
    # CALCULATE TIME-LAGGED & CONDITIONAL COVARIANCE MATRICES
    # ----------------------------------------------------------------
    
    # time-lagged covariance matrices
    for n in range(T):                                                          # loop over time-points 
        t = n * dt
        mx[:,n] = iv.get_var_mean(A, t, mu, initial_mx)
        Sx0[:,:,n] = iv.get_var_cov(A, t, errvar, initial_covariance)
    
        if n > time_lag:
            s = (n-time_lag) * dt
            Sx1[:,:,n] = iv.get_var_time_lagged_cov(A, t, s, errvar, initial_covariance)
    
    
    # conditional covariance matrices
    Sx1_conditional = np.zeros((2,2,T))
    Sx1_conditional_part11 = np.zeros((T))
    Sx1_conditional_part22 = np.zeros((T))
    Sx1_conditional_part12 = np.zeros((T))
    Sx1_conditional_part21 = np.zeros((T))
    
    for n in range(1+time_lag,T):                                                 # loop over time-points
        try:  
            var_cov_present = Sx0[:,:,n]
            var_cov_past = Sx0[:,:,n-time_lag]
            var_cov_present_parts11 = Sx0[0,0,n]
            var_cov_past_parts11 = Sx0[0,0,n-time_lag]
            var_cov_present_parts22 =  Sx0[1,1,n]
            var_cov_past_parts22 = Sx0[1,1,n-time_lag]
            
            var_time_lagged_cov_present = Sx1[:,:,n]
            var_time_lagged_cov_present_parts11 = Sx1[0,0,n]
            var_time_lagged_cov_present_parts21 = Sx1[1,0,n]
            var_time_lagged_cov_present_parts22 = Sx1[1,1,n]
            var_time_lagged_cov_present_parts12 = Sx1[0,1,n]
            
            Sx1_conditional[:,:,n] = iv.get_var_cond_cov_full(var_cov_past, var_cov_present, var_time_lagged_cov_present) 
            Sx1_conditional_part11[n] = iv.get_var_cond_cov_parts(var_cov_past_parts11, var_cov_present_parts11, var_time_lagged_cov_present_parts11)
            Sx1_conditional_part22[n] = iv.get_var_cond_cov_parts(var_cov_past_parts22, var_cov_present_parts22, var_time_lagged_cov_present_parts22)        
            Sx1_conditional_part12[n] = iv.get_var_cond_cov_parts(var_cov_past_parts11, var_cov_present_parts22, var_time_lagged_cov_present_parts21)
            Sx1_conditional_part21[n] = iv.get_var_cond_cov_parts(var_cov_past_parts22, var_cov_present_parts11, var_time_lagged_cov_present_parts12)
        except:
            pass 
    
    # ----------------------------------------------------------------
    # CALCULATE DOUBLE-REDUNDANCY
    # ----------------------------------------------------------------
    
    double_red = iv.get_double_red_mmi(Sx0, Sx1_conditional, Sx1_conditional_part11, Sx1_conditional_part22, Sx1_conditional_part12, Sx1_conditional_part21, time_lag)
    
    # ----------------------------------------------------------------
    # CALCULATE PHI, PHI-R, KL-DIVERGENCE, & PHIID-BASED QUANTITIES
    # ----------------------------------------------------------------
    
    
    for n in range(time_lag,T):                                                   # loop over time-points
        print(n) 
        
        # ----------------------------------------------------------------
        # KL DIVERGENCE
        # ----------------------------------------------------------------
        
        variational_covariance = Sx0[:,:,n]
        variational_mean = mx[:,n]
        kldiv = iv.get_kl_div(A, B, mu, variational_mean, variational_covariance)
        
        # ----------------------------------------------------------------
        # PHI & PHI-R
        # ----------------------------------------------------------------
        
        try:                    
            var_cov_past = Sx0[:,:,n-time_lag] 
            var_cond_cov_present_full = Sx1_conditional[:,:,n] 
            var_cov_past_parts11 = Sx0[0,0,n-time_lag] 
            var_cond_cov_present_parts11 = Sx1_conditional[0,0,n]
            var_cond_cov_past_parts22 = Sx0[1,1,n-time_lag]
            var_cond_cov_present_parts22 = Sx1_conditional[1,1,n]
            
            phi = iv.get_phi(var_cov_past, var_cond_cov_present_full, var_cov_past_parts11, var_cond_cov_present_parts11, var_cond_cov_past_parts22, var_cond_cov_present_parts22)
                
            phiR = phi + double_red[n]
            
        except: #RuntimeWarning
            phi = float('NaN')
            phiR = float('NaN')

        # ----------------------------------------------------------------
        # PHIID BASED QUANTITIES
        # ----------------------------------------------------------------
        
        # simulate time-series with given covariance matrix
        time_series = np.random.multivariate_normal(mx[:,n], Sx0[:,:,n], T).T
        phiid, emergence_capacity_phiid, downward_causation_phiid, synergy_phiid, transfer_phiid, phi_phiid, phiR_phiid = iv.get_phiid(time_series, time_lag, 'mmi')
        
        df_temp = pd.DataFrame({'correlation': [rho], 'error_variance': [errvar], 'time_lag': [time_lag], 'time_point' : [n],
                                 'phi': [phi], 'phiR': [phiR], 'kldiv': [kldiv], 'double_red': [double_red[n]], 'rtr': [phiid.rtr], 'rtx': [phiid.rtx], 
                                 'rty': [phiid.rty], 'rts': [phiid.rts], 'xtr': [phiid.xtr], 'xtx': [phiid.xtx], 'xty': [phiid.xty], 'xts': [phiid.xts], 
                                 'ytr': [phiid.ytr], 'ytx': [phiid.ytx], 'yty': [phiid.yty], 'yts': [phiid.yts], 'str': [phiid.str], 'stx': [phiid.stx], 
                                 'sty': [phiid.sty], 'sts': [phiid.sts], 'synergy_phiid': [synergy_phiid], 'transfer_phiid': [transfer_phiid], 
                                 'emergence_capacity_phiid': [emergence_capacity_phiid], 'downward_causation_phiid': [downward_causation_phiid], 
                                 'phi_phiid': [phi_phiid], 'phiR_phiid': [phiR_phiid]})
        
        df.append(df_temp)
        
    return df   


def phi_in_variational_inference(all_rho, all_errvar, all_timelags, T, dt, mx, mu):
    results_df = []

    # storing each dataframe in a list
    results = [get_results_from_model(rho, errvar, time_lag, T, dt, mx, mu)
                                    for rho, errvar, time_lag in product(all_rho, all_errvar, all_timelags)]

    # unpacking dataframes so that each row is not of type "list", but of type "dataframe"
    for dataframe in results:
        unpack_dataframe = pd.concat(dataframe, ignore_index = True)
        results_df.append(unpack_dataframe)
    
    # putting dataframe rows into one a single dataframe
    results_df = pd.concat(results_df, ignore_index = True)
        
    return results_df
    
# variable name: results_df_[correlation]_[error_variance]_[time-lag]
results_df = phi_in_variational_inference(all_rho, all_errvar, all_timelags, T, dt, mx, mu)

# super_result = Parallel(n_jobs=1)(delayed(phi_in_variational_inference)(all_rho, all_errvar, all_timelags, T, dt, mx, mu)
#     for rho, errvar, time_lag in product(all_rho, all_errvar, all_timelags))

results_df.to_pickle(path_out1+r'results_df_' + str(all_rho[0]).replace('.', '') + '_' + str(all_errvar[0]).replace('.', '')  + '_' + str(all_timelags[0]) +'.pkl')
# results_df_05_001_1 = pd.read_pickle(path_out1+r'results_df_05_001_1.pkl')



# for rho, errvar, time_lag in product(all_rho, all_errvar, all_timelags):
#     your_result = get_results_from_model(rho, errvar, time_lag, T, dt, mx, mu)

# super_df.to_pickle(path_out1+r'super_df.pkl')
# super_df = pd.read_pickle(path_out1+r'super_df.pkl')

#np.save(os.path.join(path_out1, 'super_df.npy'), super_df)


# TO DO 
# loop needs to be continued as of all_rho[1] (a few values for all_rho[1] that have already been calculated need to be eliminated from super_df)
# parallelize for loops

        
#%% plotting

# ----------------------------------------------------------------------------
# plots for phiid atoms & compositions
# ----------------------------------------------------------------------------

phiid_terms = ['rtr', 'sts', 'synergy_phiid', 'transfer_phiid', 'emergence_capacity_phiid', 'downward_causation_phiid', 'phi_phiid', 'phiR_phiid']
                
# plots per correlation, error variance, and time-lag
for correlation in all_rho:
    for error_variance in all_errvar:
        for time_lag in all_timelags:
            
            # index for the phiid terms to be plotted, time for x-axis
            time = np.arange(time_lag, T, 1).tolist()
             
            fig, axs = plt.subplots(4, 2, figsize=(8, 10))
            fig.suptitle('rho = {}'.format(correlation)+', error variance = {}'.format(error_variance)+', time-lag = {}'.format(time_lag), fontsize = 10)

            axs = axs.flatten()
                    
            for index, ax in enumerate(axs):
                temp_model = results_df.loc[((results_df.correlation == correlation) & (results_df.error_variance == error_variance) & (results_df.time_lag == time_lag)), phiid_terms[index]]

                # calculate moving average
                moving_average_window = 120
                moving_average = np.int(np.float(T/moving_average_window))                                         
                moving_average_vector = np.array(range(0,len(temp_model),1))
                moving_average_vector = moving_average_vector.astype(np.float64)
                moving_average_vector.fill(np.nan) 
                raw_vector = np.ma.array(temp_model, mask=np.isnan(temp_model))
        
                for l in range(len(temp_model)):
                    if l < moving_average:
                        number_of_numbers_in_sum = np.count_nonzero(~np.isnan(raw_vector[l:l+moving_average]))
                        moving_average_vector[l] = np.sum(raw_vector[l:l+moving_average])/number_of_numbers_in_sum
                    elif l > (len(raw_vector)-moving_average):
                        number_of_numbers_in_sum = np.count_nonzero(~np.isnan(raw_vector[l-moving_average:l]))
                        moving_average_vector[l] = np.sum(raw_vector[l-moving_average:l])/number_of_numbers_in_sum
                    else:    
                        number_of_numbers_in_sum = np.count_nonzero(~np.isnan(raw_vector[l-moving_average:l+moving_average]))
                        moving_average_vector[l] = np.sum(raw_vector[l-moving_average:l+moving_average])/number_of_numbers_in_sum                     
                
                ax.plot(time, temp_model, label = phiid_terms[index], color = 'b', alpha = 0.6)
                ax.plot(moving_average_vector, label ='moving-average', color = 'k', linewidth = 1)
                ax.set_title(phiid_terms[index], color = 'r', pad=10)
            fig.tight_layout()
                

            fig.savefig(path_out2 + 'phiid_quantities_' + str(correlation).replace('.', '') + '_' + str(error_variance).replace('.', '')  + '_' + str(time_lag) + '.png', dpi=300, bbox_inches='tight')  
            plt.cla()
            del fig

# ----------------------------------------------------------------------------
# plots for phi, phiR, KL-divergence, and double-redundancy
# ----------------------------------------------------------------------------

quantities = ['kldiv', 'phi', 'phiR', 'double_red']
                
# plots per correlation, error variance, and time-lag
for correlation in all_rho:
    for error_variance in all_errvar:
        for time_lag in all_timelags:
            
            # index for the phiid terms to be plotted, time for x-axis
            time = np.arange(time_lag, T, 1).tolist()
             
            fig, axs = plt.subplots(2, 2, figsize=(10, 8))
            fig.suptitle('rho = {}'.format(correlation)+', error variance = {}'.format(error_variance)+', time-lag = {}'.format(time_lag), fontsize = 10)

            axs = axs.flatten()
                    
            for index, ax in enumerate(axs):
                temp_model = results_df.loc[((results_df.correlation == correlation) & (results_df.error_variance == error_variance) & (results_df.time_lag == time_lag)), quantities[index]]

                # calculate moving average
                moving_average_window = 120
                moving_average = np.int(np.float(T/moving_average_window))                                         
                moving_average_vector = np.array(range(0,len(temp_model),1))
                moving_average_vector = moving_average_vector.astype(np.float64)
                moving_average_vector.fill(np.nan) 
                raw_vector = np.ma.array(temp_model, mask=np.isnan(temp_model))
        
                for l in range(len(temp_model)):
                    if l < moving_average:
                        number_of_numbers_in_sum = np.count_nonzero(~np.isnan(raw_vector[l:l+moving_average]))
                        moving_average_vector[l] = np.sum(raw_vector[l:l+moving_average])/number_of_numbers_in_sum
                    elif l > (len(raw_vector)-moving_average):
                        number_of_numbers_in_sum = np.count_nonzero(~np.isnan(raw_vector[l-moving_average:l]))
                        moving_average_vector[l] = np.sum(raw_vector[l-moving_average:l])/number_of_numbers_in_sum
                    else:    
                        number_of_numbers_in_sum = np.count_nonzero(~np.isnan(raw_vector[l-moving_average:l+moving_average]))
                        moving_average_vector[l] = np.sum(raw_vector[l-moving_average:l+moving_average])/number_of_numbers_in_sum                     
                
                ax.plot(time, temp_model, label = quantities[index], color = 'b', alpha = 0.6)
                ax.plot(moving_average_vector, label ='moving-average', color = 'k', linewidth = 1)
                ax.set_title(quantities[index], color = 'r', pad=10)
            fig.tight_layout()
                

            fig.savefig(path_out2 + 'kldiv_phiR_double_red_' + str(correlation).replace('.', '') + '_' + str(error_variance).replace('.', '')  + '_' + str(time_lag) + '.png', dpi=300, bbox_inches='tight')  
            plt.cla()
            del fig
        
#%% OLD  
# ----------------------------------------------------------------------------
# plots for phi, phi-R, kl-div, double-red
# ----------------------------------------------------------------------------

super_df_terms = {'phi', 'phiR', 'kldiv', 'double_red'}

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
    



