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
# from joblib import Parallel, delayed
from itertools import product


main_path = '/media/nadinespy/NewVolume/my_stuff/work/other_projects/FEP_IIT_some_thoughts/' \
    'viiit_with_miguel/IntegratedInformationInVariationalInference/scripts'
os.chdir(main_path)
import iit_in_vi as iv


oc.addpath(main_path)
oc.javaaddpath(main_path+'/infodynamics.jar')
oc.eval('pkg load statistics')

path_out1 = '/media/nadinespy/NewVolume/my_stuff/work/other_projects/FEP_IIT_some_thoughts/' \
    'viiit_with_miguel/IntegratedInformationInVariationalInference/results/analyses/'
path_out2 = '/media/nadinespy/NewVolume/my_stuff/work/other_projects/FEP_IIT_some_thoughts/' \
    'viiit_with_miguel/IntegratedInformationInVariationalInference/results/plots/'

# %%

# parameters to loop over
# all_rho = np.array([-0.9, -0.7, -0.5, -0.3,            # correlation coefficients
#          0.0, 0.3, 0.5, 0.7, 0.9])

# all_timelags = np.array([1, 5, 10, 25, 50, 75,         # delays (integration steps)
#          100, 150, 200])

# all_errvar = [0.0001, 0.001, 0.01, 0.1,                # error variances
#          0.3, 0.5, 0.7, 1.0, 1.2]

# all_weights = [0.0, 0.125, 0.25, 0.375,                # error variances
#          0.5, 0.625, 0.75, 0.875, 1.0]

# all_rho = np.array([-0.9, -0.7, -0.5, -0.3,            # off-diagonal covariances
#          0.0, 0.3, 0.5, 0.7, 0.9])

# ----------------------------------------------------------------
# ADJUST PARAMETERS
# ----------------------------------------------------------------

all_rho = np.array([0.5])
all_errvar = np.array([0.01])
all_timelags = np.array([10])
all_weights = np.array([1])
all_off_diag_covs = ([0])

# ----------------------------------------------------------------

# other parameters


T = 1000
dt = 0.01                                                # integration step

np.random.seed(10)
initial_mx = np.random.rand(2)                           # initial means at t = 0

# initial covariance at t = 0
# off_diag_cov = np.random.uniform(-1, 1, 1)
initial_cov = np.zeros((2, 2))
initial_cov[0, 1] = all_off_diag_covs
initial_cov[1, 0] = all_off_diag_covs

print('initial means at t = 0: ', initial_mx)

# %%


def get_results_from_model(rho, errvar, time_lag, T, dt, initial_mx,
                           initial_cov, weights):
    """docstring"""

    errvar = errvar/np.sqrt(2/dt)

    # ----------------------------------------------------------------
    # INITIALIZE VARATIONAL MEANS, COVARIANCES, & KL-DIVERGENCE
    # ----------------------------------------------------------------

    mx = np.zeros((2, T))                               # variational mean vector
    mx[:, 0] = initial_mx                               # initial means

    COV = np.zeros((2, 2, T))                           # same-time covariance matrix
    COV[:, :, 0] = initial_cov                          # initial covariance

    time_lagged_COV = np.zeros((2, 2, T))                           # time-lagged (one unit) covariance matrix

    # conditional time-lagged covariance matrices
    time_lagged_COND_COV = np.zeros((2, 2, T))
    time_lagged_COND_COV_part11 = np.zeros((T))
    time_lagged_COND_COV_part22 = np.zeros((T))
    time_lagged_COND_COV_part12 = np.zeros((T))
    time_lagged_COND_COV_part21 = np.zeros((T))

    kldiv = np.zeros(T)                                 # KL-divergence

    # ----------------------------------------------------------------
    # GET TRUE MEANS, AND TRUE (WEIGHTED) & MEAN-FIELD COVARIANCE
    # ----------------------------------------------------------------

    np.random.seed(10)
    true_means = np.random.randn(2)                     # means of true distribution

    # covariance of true distribution
    true_cov = np.eye(2)
    true_cov[0, 1] = np.sqrt(true_cov[0, 0] * true_cov[1, 1]) * rho
    true_cov[1, 0] = true_cov[0, 1]

    inv_true_cov = la.inv(true_cov)                     # inverse of covariance

    mean_field_inv_true_cov = np.diag(inv_true_cov)

    # weighted inverse of covariance
    weighted_inv_true_cov = inv_true_cov.copy()
    weighted_inv_true_cov[0, 1] = weights*inv_true_cov[0, 1]
    weighted_inv_true_cov[1, 0] = weights*inv_true_cov[1, 0]

    print('inv_true_covariance: ', inv_true_cov)
    print('weighted_inv_true_covariance: ', weighted_inv_true_cov)
    print('mean_field_inv_true_covariance: ', mean_field_inv_true_cov)

    df = []

    # ----------------------------------------------------------------
    # CALCULATE TIME-LAGGED & CONDITIONAL COVARIANCE MATRICES
    # ----------------------------------------------------------------

    # time-lagged covariance matrices
    for n in range(T):                                  # loop over time-points
        t = n * dt
        mx[:, n] = iv.get_mean(weighted_inv_true_cov, t, true_means, initial_mx)

        COV[:, :, n] = iv.get_cov(weighted_inv_true_cov, t, errvar, initial_cov)

        if n > time_lag:
            s = (n-time_lag) * dt
            time_lagged_COV[:, :, n] = iv.get_time_lagged_cov(weighted_inv_true_cov,
                                                  t, s, errvar, initial_cov)

    for n in range(1+time_lag, T):                       # loop over time-points
        try:
            cov_present = COV[:, :, n]
            cov_past = COV[:, :, n - time_lag]
            cov_present_parts11 = COV[0, 0, n]
            cov_past_parts11 = COV[0, 0, n-time_lag]
            cov_present_parts22 = COV[1, 1, n]
            cov_past_parts22 = COV[1, 1, n-time_lag]
            time_lagged_cov_present = time_lagged_COV[:, :, n]
            time_lagged_cov_present_parts11 = time_lagged_COV[0, 0, n]
            time_lagged_cov_present_parts21 = time_lagged_COV[1, 0, n]
            time_lagged_cov_present_parts22 = time_lagged_COV[1, 1, n]
            time_lagged_cov_present_parts12 = time_lagged_COV[0, 1, n]

            time_lagged_COND_COV[:, :, n] = iv.get_cond_cov_full(cov_past, cov_present,
                                                            time_lagged_cov_present)
            time_lagged_cond_cov_part11[n] = iv.get_cond_cov_parts(
                cov_past_parts11,
                cov_present_parts11,
                time_lagged_cov_present_parts11)
            time_lagged_cond_cov_part22[n] = iv.get_cond_cov_parts(
                cov_past_parts22,
                cov_present_parts22,
                time_lagged_cov_present_parts22)
            time_lagged_cond_cov_part12[n] = iv.get_cond_cov_parts(
                cov_past_parts11,
                cov_present_parts22,
                time_lagged_cov_present_parts21)
            time_lagged_cond_cov_part21[n] = iv.get_cond_cov_parts(
                cov_past_parts22,
                cov_present_parts11,
                time_lagged_cov_present_parts12)
        except ZeroDivisionError:
            print('Divided by zero')

    # ----------------------------------------------------------------
    # CALCULATE DOUBLE-REDUNDANCY
    # ----------------------------------------------------------------

    double_red = iv.get_double_red_mmi(COV,
                                       time_lagged_COND_COV,
                                       time_lagged_COND_COV_part11,
                                       time_lagged_COND_COV_part22,
                                       time_lagged_COND_COV_part12,
                                       time_lagged_COND_COV_part21,
                                       time_lag)

    # ----------------------------------------------------------------
    # CALCULATE PHI, PHI-R, KL-DIVERGENCE, & PHIID-BASED QUANTITIES
    # ----------------------------------------------------------------

    for n in range(time_lag, T):                         # loop over time-points
        print(n)

        # ----------------------------------------------------------------
        # KL DIVERGENCE
        # ----------------------------------------------------------------

        cov_temp = COV[:, :, n]
        means_temp = mx[:, n]
        kldiv = iv.get_kl_div(inv_true_cov, mean_field_inv_true_cov,
                              true_means, means_temp, cov_temp)

        # ----------------------------------------------------------------
        # PHI & PHI-R
        # ----------------------------------------------------------------

        try:
            # might need to assign other values (e. g., time_lagged_COND_COV_part12),
            # as here, we're not conditioning on single variables
            cov_past = COV[:, :, n-time_lag]
            cond_cov_present_full = time_lagged_COND_COV[:, :, n]
            cov_past_parts11 = COV[0, 0, n-time_lag]
            cond_cov_present_parts11 = time_lagged_COND_COV[0, 0, n]
            cov_past_parts22 = COV[1, 1, n-time_lag]
            cond_cov_present_parts22 = time_lagged_COND_COV[1, 1, n]

            phi = iv.get_phi(cov_past,
                             cond_cov_present_full,
                             cov_past_parts11,
                             cond_cov_present_parts11,
                             cov_past_parts22,
                             cond_cov_present_parts22)

            phiR = phi + double_red[n]

        except RuntimeError:
            phi = float('NaN')
            phiR = float('NaN')
            print('phi and phiR are assigned NaN')

        # ----------------------------------------------------------------
        # PHIID BASED QUANTITIES
        # ----------------------------------------------------------------

        # simulate time-series with given covariance matrix
        time_series = np.random.multivariate_normal(mx[:, n], COV[:, :, n], T).T

        [phiid,
         emergence_capacity_phiid,
         downward_causation_phiid,
         synergy_phiid,
         transfer_phiid,
         phi_phiid,
         phiR_phiid] = iv.get_phiid(time_series, time_lag, 'mmi')

        df_temp = pd.DataFrame({'correlation': [rho],
                                'error_variance': [errvar],
                                'time_lag': [time_lag],
                                'time_point': [n],
                                'phi': [phi],
                                'phiR': [phiR],
                                'kldiv': [kldiv],
                                'double_red': [double_red[n]],
                                'rtr': [phiid.rtr],
                                'rtx': [phiid.rtx],
                                'rty': [phiid.rty],
                                'rts': [phiid.rts],
                                'xtr': [phiid.xtr],
                                'xtx': [phiid.xtx],
                                'xty': [phiid.xty],
                                'xts': [phiid.xts],
                                'ytr': [phiid.ytr],
                                'ytx': [phiid.ytx],
                                'yty': [phiid.yty],
                                'yts': [phiid.yts],
                                'str': [phiid.str],
                                'stx': [phiid.stx],
                                'sty': [phiid.sty],
                                'sts': [phiid.sts],
                                'synergy_phiid': [synergy_phiid],
                                'transfer_phiid': [transfer_phiid],
                                'emergence_capacity_phiid': [emergence_capacity_phiid],
                                'downward_causation_phiid': [downward_causation_phiid],
                                'phi_phiid': [phi_phiid],
                                'phiR_phiid': [phiR_phiid]})

        df.append(df_temp)

    return df


def phi_in_variational_inference(all_rho, all_errvar, all_timelags, T, dt,
                                 initial_mx, initial_cov, weights):
    results_df = []

    # storing each dataframe in a list
    results = [get_results_from_model(rho, errvar, time_lag, T, dt,
                                      initial_mx, initial_cov, weights)
               for rho, errvar, time_lag in product(all_rho, all_errvar, all_timelags)]

    # unpacking dataframes so that each row is not of type "list", but of type "dataframe"
    for dataframe in results:
        unpack_dataframe = pd.concat(dataframe, ignore_index=True)
        results_df.append(unpack_dataframe)

    # putting dataframe rows into one a single dataframe
    results_df = pd.concat(results_df, ignore_index=True)

    return results_df


# variable name: results_df_[correlation]_[error_variance]_[time-lag]
results_df = phi_in_variational_inference(all_rho, all_errvar, all_timelags, T, dt,
                                          initial_mx, initial_cov, all_weights)

# super_result = Parallel(n_jobs=1)(delayed(phi_in_variational_inference)
#     (all_rho, all_errvar, all_timelags, T, dt, mx)
#     for rho, errvar, time_lag in product(all_rho, all_errvar, all_timelags))

results_df.to_pickle(path_out1 +
                     r'results_df_' +
                     str(all_rho[0]).replace('.', '') +
                     '_' +
                     str(all_errvar[0]).replace('.', '') +
                     '_' + str(all_timelags[0]) +
                     '.pkl')

# results_df_05_001_1 = pd.read_pickle(path_out1+r'results_df_05_001_1.pkl')

# for rho, errvar, time_lag in product(all_rho, all_errvar, all_timelags):
#     your_result = get_results_from_model(rho, errvar, time_lag, T, dt, mx)

# super_df.to_pickle(path_out1+r'super_df.pkl')
# super_df = pd.read_pickle(path_out1+r'super_df.pkl')

# np.save(os.path.join(path_out1, 'super_df.npy'), super_df)

# TO DO
# loop needs to be continued as of all_rho[1] (a few values for all_rho[1]
# that have already been calculated need to be eliminated from super_df)
# parallelize for loops


# %% plotting

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
    



