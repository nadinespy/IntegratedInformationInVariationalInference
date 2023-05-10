#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 15:15:47 2023

@author: nadinespy
"""
import numpy as np
import scipy.linalg as la
from itertools import product
from matplotlib import pyplot as plt
import os
import pandas as pd
import pickle
# from joblib import Parallel, delayed
# import scipy.io

main_path = '/media/nadinespy/NewVolume1/work/current_projects/viiit/viiit_with_miguel/' \
        'IntegratedInformationInVariationalInference/scripts/'
os.chdir(main_path)
import iit_in_vi as iv

# oc.addpath(main_path)
# oc.javaaddpath(main_path+'/infodynamics.jar')
# oc.eval('pkg load statistics')

path_out1 = '/media/nadinespy/NewVolume1/work/current_projects/viiit/viiit_with_miguel/' \
        'IntegratedInformationInVariationalInference/results/analyses/'
path_out2 = '/media/nadinespy/NewVolume1/work/current_projects/viiit/viiit_with_miguel/' \
        'IntegratedInformationInVariationalInference/results/plots/'

# %%
# parameters to loop over
# all_rho = np.array([-0.9, -0.7, -0.5, -0.3,       # correlation coefficients
#          0.0, 0.3, 0.5, 0.7, 0.9])

# all_timelags = np.array([1, 5, 10, 25, 50, 75,    # delays (integration steps)
#          100, 150, 200])

# all_errvar = [0.0001, 0.001, 0.01, 0.1,           # error variances
#          0.3, 0.5, 0.7, 1.0, 1.2]

# all_weights = [0.0, 0.125, 0.25, 0.375,           # weights
#          0.5, 0.625, 0.75, 0.875, 1.0]

# all_off_diag_covs = ([0])                         # off-diagonal covariances

# ----------------------------------------------------------------
# ADJUST PARAMETERS
# ----------------------------------------------------------------

n_rho = 100
n_weights = 11
all_rho = np.arange(0, n_rho) / n_rho
all_weights = np.linspace(0, 1, n_weights)
all_errvar = np.array([0.01])
all_timelags = np.array([1])
# all_rho = np.array([0.5])
# all_weights = np.array([1])

all_off_diag_covs = ([0])
case = "discrete"
gamma = 0.01
dt = 0.01                                           # integration step
# ----------------------------------------------------------------

# initialize initial means and covariance

np.random.seed(10)
initial_var_means = np.random.rand(2)               # initial means at t = 0
# %%


def get_results_from_model(rho, errvar, weight, time_lag, gamma):
    """docstring"""

    # ----------------------------------------------------------------
    # INITIALIZE COVARIANCES, & KL-DIVERGENCE
    # ----------------------------------------------------------------

    same_time_COV = np.zeros((2, 2))                # same-time covariance matrix
    time_lagged_COV = np.zeros((2, 2))              # time-lagged (one unit) covariance matrix
    time_lagged_COND_COV = np.zeros((2, 2))         # conditional time-lagged covariance matrices

    # ----------------------------------------------------------------
    # GET TRUE MEANS, AND TRUE (WEIGHTED) & MEAN-FIELD COVARIANCE
    # ----------------------------------------------------------------

    np.random.seed(10)
    true_means = np.random.randn(2)             # means of true distribution

    # covariance of true distribution
    true_cov = np.eye(2)
    true_cov[0, 1] = np.sqrt(true_cov[0, 0] * true_cov[1, 1]) * rho
    true_cov[1, 0] = true_cov[0, 1]

    inv_true_cov = la.inv(true_cov)             # inverse of covariance

    # weighted inverse of covariance
    weighted_inv_true_cov = inv_true_cov.copy()
    weighted_inv_true_cov[0, 1] = weight*inv_true_cov[0, 1]
    weighted_inv_true_cov[1, 0] = weight*inv_true_cov[1, 0]

    mean_field_inv_true_cov = np.diag(inv_true_cov)

    # print('inv_true_covariance: ', inv_true_cov)
    # print('weighted_inv_true_covariance: ', weighted_inv_true_cov)
    # print('mean_field_inv_true_covariance: ', mean_field_inv_true_cov)

    # in the limit, variational and true means will be the same
    var_means = true_means
    identity = np.eye(2)
    K = np.linalg.inv(weighted_inv_true_cov @ identity @ weighted_inv_true_cov *
                      errvar ** 2) / gamma ** 2

    same_time_COV = np.linalg.inv(K - (identity - gamma * weighted_inv_true_cov) @ K @
                                  (identity - gamma * weighted_inv_true_cov))

    time_lagged_COV = (identity - gamma * weighted_inv_true_cov) @ same_time_COV

    kl_div = iv.get_kl_div(weighted_inv_true_cov, mean_field_inv_true_cov,
                           true_means, var_means, same_time_COV)

    # kl_div = -np.sum(0.5*(1+np.log(1/mean_field_inv_true_cov))) + \
    #          0.5*np.log(1*np.linalg.det(np.linalg.inv(weighted_inv_true_cov))) + \
    #          np.sum(0.5*mean_field_inv_true_cov*np.diag(weighted_inv_true_cov)) + \
    #          0.5* (var_means-true_means) @ weighted_inv_true_cov @ (var_means-true_means) + \
    #          0.5*np.sum(weighted_inv_true_cov*same_time_COV)

    time_lagged_COND_COV = iv.get_cond_cov_full(same_time_COV,
                                                same_time_COV,
                                                time_lagged_COV)
    try:
        time_lagged_COND_COV_PART11 = iv.get_cond_cov_parts(same_time_COV[0, 0],
                                                            same_time_COV[0, 0],
                                                            time_lagged_COV[0, 0])
        time_lagged_COND_COV_PART22 = iv.get_cond_cov_parts(same_time_COV[1, 1],
                                                            same_time_COV[1, 1],
                                                            time_lagged_COV[1, 1])
        time_lagged_COND_COV_PART12 = iv.get_cond_cov_parts(same_time_COV[0, 0],
                                                            same_time_COV[1, 1],
                                                            time_lagged_COV[1, 0])
        time_lagged_COND_COV_PART21 = iv.get_cond_cov_parts(same_time_COV[1, 1],
                                                            same_time_COV[0, 0],
                                                            time_lagged_COV[0, 1])
    except:
        pass

    # try:
    #      time_lagged_COND_COV = same_time_COV - time_lagged_COV.T @ la.pinv(same_time_COV) @ time_lagged_COV       
    #      time_lagged_COND_COV_PART11 = same_time_COV[0,0] - time_lagged_COV[0,0] * \
    #          np.reciprocal(same_time_COV[0,0]) * time_lagged_COV[0,0]  
    #      time_lagged_COND_COV_PART22 = same_time_COV[1,1] - time_lagged_COV[1,1] * \
    #          np.reciprocal(same_time_COV[1,1]) * time_lagged_COV[1,1] 
    #      time_lagged_COND_COV_PART12 = same_time_COV[0,0] - time_lagged_COV[1,0] * \
    #          np.reciprocal(same_time_COV[1,1]) * time_lagged_COV[1,0]  
    #      time_lagged_COND_COV_PART21 = same_time_COV[1,1] - time_lagged_COV[0,1] * \
    #          np.reciprocal(same_time_COV[0,0]) * time_lagged_COV[0,1]  
    # except:
    #      pass 

    double_red = iv.get_double_red_mmi(same_time_COV[0, 0],
                                       same_time_COV[1, 1],
                                       time_lagged_COND_COV_PART11,
                                       time_lagged_COND_COV_PART22,
                                       time_lagged_COND_COV_PART12,
                                       time_lagged_COND_COV_PART21)

    phi = iv.get_phi(same_time_COV,
                     time_lagged_COND_COV,
                     same_time_COV[0, 0],
                     time_lagged_COND_COV_PART11,
                     same_time_COV[1, 1],
                     time_lagged_COND_COV_PART22)

    phiR = phi + double_red

    a = np.concatenate((same_time_COV,
                        time_lagged_COV), axis=1)
    b = np.concatenate((time_lagged_COV,
                        same_time_COV), axis=1)
    full_time_lagged_COV = np.concatenate([a, b])

    [phiid,
     emergence_capacity_phiid,
     downward_causation_phiid,
     synergy_phiid,
     transfer_phiid,
     phi_phiid,
     phiR_phiid] = iv.get_phiid_analytical(full_time_lagged_COV, 'mmi')

    df_temp = pd.DataFrame({'correlation': [rho],
                            'error_variance': [errvar],
                            'time_lag': [time_lag],
                            'weight': [weight],
                            'phi': [phi],
                            'phiR': [phiR],
                            'kldiv': [kl_div],
                            'double_red': [double_red],
                            'rtr': [phiid['rtr']],
                            'rtx': [phiid['rtx']],
                            'rty': [phiid['rty']],
                            'rts': [phiid['rts']],
                            'xtr': [phiid['xtr']],
                            'xtx': [phiid['xtx']],
                            'xty': [phiid['xty']],
                            'xts': [phiid['xts']],
                            'ytr': [phiid['ytr']],
                            'ytx': [phiid['ytx']],
                            'yty': [phiid['yty']],
                            'yts': [phiid['yts']],
                            'str': [phiid['str']],
                            'stx': [phiid['stx']],
                            'sty': [phiid['sty']],
                            'sts': [phiid['sts']],
                            'synergy_phiid': [synergy_phiid],
                            'transfer_phiid': [transfer_phiid],
                            'emergence_capacity_phiid': [emergence_capacity_phiid],
                            'downward_causation_phiid': [downward_causation_phiid],
                            'phi_phiid': [phi_phiid],
                            'phiR_phiid': [phiR_phiid]})

    print('rho: ', rho)

    return df_temp


def phi_in_variational_inference(all_rho, all_errvar, all_weights,
                                 all_timelags, gamma):
    results_df = []

    # storing each dataframe in a list
    results = [get_results_from_model(rho, errvar, weight, time_lag, gamma)
               for rho, errvar, weight, time_lag in
               product(all_rho, all_errvar, all_weights, all_timelags)]

    # putting dataframe rows into one a single dataframe
    results_df = pd.concat(results, ignore_index=True)

    return results_df


# variable name: results_df_[correlation]_[error_variance]_[time-lag]
results_df = phi_in_variational_inference(all_rho, all_errvar, all_weights,
                                          all_timelags, gamma)

# super_result = Parallel(n_jobs=1)(delayed(phi_in_variational_inference)
#     (all_rho, all_errvar, all_timelags, T, dt, var_means)
#     for rho, errvar, time_lag in product(all_rho, all_errvar, all_timelags))

results_df.to_pickle(path_out1 +
                     r'discrete_steady_state_df_' +
                     str(n_rho) + '_' +
                     str(n_weights) + '_' +
                     str(all_errvar[0]).replace('.', '') + '_' +
                     str(all_timelags[0]) + '_' +
                     str(all_off_diag_covs[0]).replace('.', '') + '_' +
                     str(gamma).replace('.', '') + '.pkl')

# results_df = open(path_out1+r'discrete_steady_state_df_00_001_00_1_0_001.pkl', 'rb')
# results_df = pickle.load(results_df)


# %% plotting

# ----------------------------------------------------------------------------
# plots for phiid atoms & compositions
# ----------------------------------------------------------------------------






#for index, ax in enumerate(axs):

#    for j, weight in zip(range(0, len(all_weights)), all_weights):
#        temp_df = results_df.loc[((results_df.weight == weight),
#                                 quantities[index])]
#        plt.plot(all_rho, temp_df, '-', label = r'$\alpha$ = {:.2f}'.format(weight))
#        pylab.show()
#        plt.legend(ncol=2,loc=0)
#        #ax.axis([0,all_rho[-1],0,np.max(temp_model)])
        #ax.set_xlabel(r'$\rho$', fontsize=18)
        #ax.set_ylabel(r'$\varphi$', rotation=0, fontsize=18,labelpad=25)
#        title = r'$\varphi$ for different correlations & weights'


#plt.savefig(path_out2+r'discrete_steady_state_df_00_001_00_1_0_001.pdf', bbox_inches='tight')

#%%
   
# plots per correlation, error variance, and time-lag
for correlation in all_rho:
    for error_variance in all_errvar:
        for time_lag in all_timelags:
            for weight in all_weights:
                for off_diag_covs in all_off_diag_covs:
            
                    # index for the phiid terms to be plotted, time for x-axis
                    time = np.arange(time_lag, T, 1).tolist()
                     
                    fig, axs = plt.subplots(4, 2, figsize=(8, 10))
                    fig.suptitle('rho = {}'.format(correlation)+', error variance = {}'.format(error_variance)+', time-lag = {}'.format(time_lag), fontsize = 10)
        
                    axs = axs.flatten()
                            
                    for index, ax in enumerate(axs):
                        
                        if case == 'continuous':
                            temp_model = results_df.loc[((results_df.correlation == correlation) & (results_df.error_variance == error_variance/np.sqrt(2/dt)) & (results_df.time_lag == time_lag)), phiid_terms[index]]
                        elif case == 'discrete':
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
                        
        
                    fig.savefig(path_out2 + 'phiid_quantities_' + 
                                str(correlation).replace('.', '') + '_' + 
                                str(error_variance).replace('.', '')  + '_' + 
                                str(time_lag) + '_' +
                                str(weight).replace('.', '') + '_' +
                                str(off_diag_covs).replace('.', '') + '_' + 
                                case + '_' +
                                str(gamma).replace('.', '') + 
                                '.png', dpi=300, bbox_inches='tight')  
                    plt.cla()
                    del fig

# %% 
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
    



