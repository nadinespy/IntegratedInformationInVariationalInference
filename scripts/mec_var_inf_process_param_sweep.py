#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 11:41:04 2021

@author: nadinespy
"""
import numpy as np
import scipy.linalg as la
from itertools import product
from matplotlib import pyplot as plt
import os
import pandas as pd
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

# # Matlab engine stuff
# eng = matlab.engine.start_matlab()

# # np.random.random(): return random floats in the half-open interval [0.0, 1.0)
# cov_matrix = np.random.random((4, 4))*3
# means = np.mean(cov_matrix, 1)

# # mdic = {'array': cov_matrix, 'label': 'cov_matrix'}
# # scipy.io.savemat('/media/nadinespy/NewVolume1/my_stuff/work/other_projects/FEP_IIT_some_thoughts/viiit_with_miguel/IntegratedInformationInVariationalInference/scripts/cov_matrix.mat', mdic)
# # scipy.io.loadmat('/media/nadinespy/NewVolume1/my_stuff/work/other_projects/FEP_IIT_some_thoughts/viiit_with_miguel/IntegratedInformationInVariationalInference/scripts/cov_matrix.mat')

# xx = matlab.double(cov_matrix.tolist())
# yy = matlab.double(means.tolist())

# blubb = eng.PhiIDFull_Analytical(xx, yy, 'MMI')


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

all_rho = np.array([0.5])
all_errvar = np.array([0.01])
all_timelags = np.array([1])
all_weights = np.array([1])
all_off_diag_covs = ([0])
case = "discrete"
gamma = 0.01
dt = 0.01                                           # integration step
T = 5
# ----------------------------------------------------------------

# initialize initial means and covariance

np.random.seed(10)
initial_var_means = np.random.rand(2)               # initial means at t = 0

# initial covariance at t = 0
initial_same_time_cov = np.zeros((2, 2))
initial_same_time_cov[0, 1] = all_off_diag_covs[0]
initial_same_time_cov[1, 0] = all_off_diag_covs[0]

print('initial means at t = 0: ', initial_var_means)

# %%


def get_results_from_model(rho, errvar, weight, time_lag, T, dt, initial_var_means,
                           initial_same_time_cov, case, gamma):
    """docstring"""

    # ----------------------------------------------------------------
    # INITIALIZE VARATIONAL MEANS, COVARIANCES, & KL-DIVERGENCE
    # ----------------------------------------------------------------

    var_means = np.zeros((2, T))                        # variational mean vector
    var_means[:, 0] = initial_var_means                 # initial means
    kldiv = np.zeros(T)

    same_time_COV = np.zeros((2, 2, T))                 # same-time covariance matrix
    same_time_COV[:, :, 0] = initial_same_time_cov      # initial covariance

    time_lagged_COV = np.zeros((2, 2, T))               # time-lagged (one unit) covariance matrix

    # conditional time-lagged covariance matrices
    time_lagged_COND_COV = np.zeros((2, 2, T))
    time_lagged_COND_COV_PART11 = np.zeros((T))
    time_lagged_COND_COV_PART22 = np.zeros((T))
    time_lagged_COND_COV_PART12 = np.zeros((T))
    time_lagged_COND_COV_PART21 = np.zeros((T))

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
    noise_cov = np.eye(2)*errvar**2

    mean_field_inv_true_cov = np.diag(inv_true_cov)

    # weighted inverse of covariance
    weighted_inv_true_cov = inv_true_cov.copy()
    weighted_inv_true_cov[0, 1] = weight*inv_true_cov[0, 1]
    weighted_inv_true_cov[1, 0] = weight*inv_true_cov[1, 0]

    # print('inv_true_covariance: ', inv_true_cov)
    # print('weighted_inv_true_covariance: ', weighted_inv_true_cov)
    # print('mean_field_inv_true_covariance: ', mean_field_inv_true_cov)

    df = []

    # ----------------------------------------------------------------
    # CALCULATE VARIATIONAL MEANS, SAME TIME, TIME-LAGGED &
    # CONDITIONAL COVARIANCE MATRICES
    # ----------------------------------------------------------------

    # variational means, same-time & time-lagged covariance matrices
    if case == 'continuous':
        errvar = errvar/np.sqrt(2/dt)
        for n in range(T):                                  # loop over time-points
            t = n * dt
            var_means[:, n] = iv.get_mean_continuous(weighted_inv_true_cov, t,
                                                     true_means, initial_var_means)
            same_time_COV[:, :, n] = iv.get_cov_continuous(weighted_inv_true_cov,
                                                           t, errvar, initial_same_time_cov)

            if n > time_lag:
                s = (n-time_lag) * dt
                time_lagged_COV[:, :, n] = iv.get_time_lagged_cov_continuous(weighted_inv_true_cov,
                                                                             t, s, errvar,
                                                                             initial_same_time_cov)
    elif case == "discrete":
        for n in range(1, T):                                  # loop over time-points
            var_means[:, n] = iv.get_mean_discrete(gamma, weighted_inv_true_cov,
                                                   var_means[:, n-1], true_means)
            same_time_COV[:, :, n] = iv.get_cov_discrete(gamma, weighted_inv_true_cov,
                                                         same_time_COV[:, :, n-1], noise_cov)
            # COV[:, :, 1] = (np.eye(2) - gamma *weighted_inv_true_cov) @ COV[:,:,0]
            # @ (np.eye(2) - gamma *weighted_inv_true_cov) + gamma**2 *
            # weighted_inv_true_cov @ gamma @ weighted_inv_true_cov

            if n > time_lag:
                time_lagged_COV[:, :, n] = iv.get_time_lagged_cov_discrete(gamma,
                                                                           weighted_inv_true_cov,
                                                                           same_time_COV[:, :, n-1])

    # conditional time-lagged covariances (for full set (2 x 2), and part of variables (1))
    for n in range(1+time_lag, T):                       # loop over time-points
        try:
            cov_present = same_time_COV[:, :, n]
            cov_past = same_time_COV[:, :, n - time_lag]
            cov_present_parts11 = same_time_COV[0, 0, n]
            cov_past_parts11 = same_time_COV[0, 0, n-time_lag]
            cov_present_parts22 = same_time_COV[1, 1, n]
            cov_past_parts22 = same_time_COV[1, 1, n-time_lag]
            time_lagged_cov_present = time_lagged_COV[:, :, n]
            time_lagged_cov_present_parts11 = time_lagged_COV[0, 0, n]
            time_lagged_cov_present_parts21 = time_lagged_COV[1, 0, n]
            time_lagged_cov_present_parts22 = time_lagged_COV[1, 1, n]
            time_lagged_cov_present_parts12 = time_lagged_COV[0, 1, n]

            time_lagged_COND_COV[:, :, n] = iv.get_cond_cov_full(cov_past, cov_present,
                                                                 time_lagged_cov_present)
            time_lagged_COND_COV_PART11[n] = iv.get_cond_cov_parts(
                cov_past_parts11,
                cov_present_parts11,
                time_lagged_cov_present_parts11)
            time_lagged_COND_COV_PART22[n] = iv.get_cond_cov_parts(
                cov_past_parts22,
                cov_present_parts22,
                time_lagged_cov_present_parts22)
            time_lagged_COND_COV_PART12[n] = iv.get_cond_cov_parts(
                cov_past_parts11,
                cov_present_parts22,
                time_lagged_cov_present_parts21)
            time_lagged_COND_COV_PART21[n] = iv.get_cond_cov_parts(
                cov_past_parts22,
                cov_present_parts11,
                time_lagged_cov_present_parts12)
        except ZeroDivisionError:
            print('Divided by zero')

    # ----------------------------------------------------------------
    # KL DIVERGENCE
    # ----------------------------------------------------------------

    for n in range(T):
        cov_temp = same_time_COV[:, :, n]
        means_temp = var_means[:, n]

        cov_temp = same_time_COV[:, :, n]
        means_temp = var_means[:, n]
        kldiv[n] = iv.get_kl_div(inv_true_cov, mean_field_inv_true_cov,
                                 true_means, means_temp, cov_temp)

    # ----------------------------------------------------------------
    # CALCULATE PHI, PHI-R, & PHIID-BASED QUANTITIES
    # ----------------------------------------------------------------

    # loop over time-points
    for n in range(time_lag, T):
        print(n)

        cov_temp = same_time_COV[:, :, n]
        means_temp = var_means[:, n]

        # ----------------------------------------------------------------
        # DOUBLE REDUNDANCY, PHI & PHI-R
        # ----------------------------------------------------------------

        try:
            # might need to assign other values (e. g., time_lagged_COND_COV_PART12),
            # as here, we're not conditioning on single variables
            cov_past_full = same_time_COV[:, :, n-time_lag]
            cond_cov_present_full = time_lagged_COND_COV[:, :, n]
            cov_past_parts11 = same_time_COV[0, 0, n-time_lag]
            cond_cov_present_parts11 = time_lagged_COND_COV[0, 0, n]
            cov_past_parts22 = same_time_COV[1, 1, n-time_lag]
            cond_cov_present_parts22 = time_lagged_COND_COV[1, 1, n]
            cond_cov_present_parts12 = time_lagged_COND_COV_PART12[n]
            cond_cov_present_parts21 = time_lagged_COND_COV_PART21[n]

            double_red = iv.get_double_red_mmi(cov_past_parts11,
                                               cov_past_parts22,
                                               cond_cov_present_parts11,
                                               cond_cov_present_parts22,
                                               cond_cov_present_parts12,
                                               cond_cov_present_parts21)

            phi = iv.get_phi(cov_past_full,
                             cond_cov_present_full,
                             cov_past_parts11,
                             cond_cov_present_parts11,
                             cov_past_parts22,
                             cond_cov_present_parts22)

            phiR = phi + double_red

        except RuntimeError:
            phi = float('NaN')
            phiR = float('NaN')
            double_red = float('NaN')
            print('phi and phiR and/or are assigned NaN')

        # ----------------------------------------------------------------
        # PHIID BASED QUANTITIES
        # ----------------------------------------------------------------

        # get analytical solution of phiid

        # concatenate respective matrices to get full_time_lagged_COV
        a = np.concatenate((time_lagged_COV[:, :, n],
                            time_lagged_COND_COV[:, :, n]), axis=1)
        b = np.concatenate((time_lagged_COND_COV[:, :, n],
                            time_lagged_COV[:, :, n-time_lag]), axis=1)
        full_time_lagged_COV = np.concatenate([a, b])

        [phiid,
         emergence_capacity_phiid,
         downward_causation_phiid,
         synergy_phiid,
         transfer_phiid,
         phi_phiid,
         phiR_phiid] = iv.get_phiid_analytical(full_time_lagged_COV, 'mmi')

        # --------------------------------------------------------------------

        # a = np.concatenate((time_lagged_COV[:, :, n],
        #                     time_lagged_COND_COV[:, :, n]), axis=1)
        # b = np.concatenate((time_lagged_COND_COV[:, :, n],
        #                     time_lagged_COV[:, :, n-time_lag]), axis=1)
        # full_time_lagged_COV = np.concatenate([a, b])

        # # building array of means corresponding to full time-lagged covariance matrix
        # all_means = np.concatenate((var_means[:, n], var_means[:, n-time_lag]))

        # # convert to matlab-compatible data type *double*
        # full_time_lagged_COV = matlab.double(full_time_lagged_COV.tolist())
        # all_means = matlab.double(all_means.tolist())

        # phiid = eng.PhiIDFull_Analytical(full_time_lagged_COV, all_means, 'mmi')

        # # synergy = causal decoupling + downward causation + upward causation

        # emergence_capacity_phiid = phiid['str'] + phiid['stx'] + \
        #     phiid['sty'] + phiid['sts']
        # downward_causation_phiid = phiid['str'] + phiid['stx'] + \
        #     phiid['sty']
        # synergy_phiid = emergence_capacity_phiid + phiid['rts'] + \
        #     phiid['xts'] + phiid['yts']
        # transfer_phiid = phiid['xty'] + phiid['ytx']
        # phi_phiid = - phiid['rtr'] + synergy_phiid + transfer_phiid
        # phiR_phiid = phi_phiid + phiid['rtr']

        # ---------------------------------------------------------------------

        df_temp = pd.DataFrame({'correlation': [rho],
                                'error_variance': [errvar],
                                'time_lag': [time_lag],
                                'weight': [weight],
                                'phi': [phi],
                                'phiR': [phiR],
                                'kldiv': [kldiv[n]],
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

        # # simulate time-series with given covariance matrix to get phiid
        # time_series = np.random.multivariate_normal(var_means[:, n], COV[:, :, n], T).T

        # [phiid,
        #  emergence_capacity_phiid,
        #  downward_causation_phiid,
        #  synergy_phiid,
        #  transfer_phiid,
        #  phi_phiid,
        #  phiR_phiid] = iv.get_phiid(time_series, time_lag, 'mmi')

        # df_temp = pd.DataFrame({'correlation': [rho],
        #                         'error_variance': [errvar],
        #                         'time_lag': [time_lag],
        #                         'time_point': [n],
        #                         'phi': [phi],
        #                         'phiR': [phiR],
        #                         'kldiv': [kldiv],
        #                         'double_red': [double_red[n]],
        #                         'rtr': [phiid.rtr],
        #                         'rtx': [phiid.rtx],
        #                         'rty': [phiid.rty],
        #                         'rts': [phiid.rts],
        #                         'xtr': [phiid.xtr],
        #                         'xtx': [phiid.xtx],
        #                         'xty': [phiid.xty],
        #                         'xts': [phiid.xts],
        #                         'ytr': [phiid.ytr],
        #                         'ytx': [phiid.ytx],
        #                         'yty': [phiid.yty],
        #                         'yts': [phiid.yts],
        #                         'str': [phiid.str],
        #                         'stx': [phiid.stx],
        #                         'sty': [phiid.sty],
        #                         'sts': [phiid.sts],
        #                         'synergy_phiid': [synergy_phiid],
        #                         'transfer_phiid': [transfer_phiid],
        #                         'emergence_capacity_phiid': [emergence_capacity_phiid],
        #                         'downward_causation_phiid': [downward_causation_phiid],
        #                         'phi_phiid': [phi_phiid],
        #                         'phiR_phiid': [phiR_phiid]})

        df.append(df_temp)

    return df


def phi_in_variational_inference(all_rho, all_errvar, all_weights, all_timelags, T, dt,
                                 initial_var_means, initial_same_time_cov, case, gamma):
    results_df = []

    # storing each dataframe in a list
    results = [get_results_from_model(rho, errvar, weight, time_lag, T, dt,
                                      initial_var_means, initial_same_time_cov,
                                      case, gamma)
               for rho, errvar, weight, time_lag in product(all_rho, all_errvar, all_weights, all_timelags)]

    # unpacking dataframes so that each row is not of type "list", but of type "dataframe"
    for dataframe in results:
        unpack_dataframe = pd.concat(dataframe, ignore_index=True)
        results_df.append(unpack_dataframe)

    # putting dataframe rows into one a single dataframe
    results_df = pd.concat(results_df, ignore_index=True)

    return results_df


# variable name: results_df_[correlation]_[error_variance]_[time-lag]
results_df = phi_in_variational_inference(all_rho, all_errvar, all_weights,
                                          all_timelags, T, dt, initial_var_means,
                                          initial_same_time_cov, case, gamma)

# super_result = Parallel(n_jobs=1)(delayed(phi_in_variational_inference)
#     (all_rho, all_errvar, all_timelags, T, dt, var_means)
#     for rho, errvar, time_lag in product(all_rho, all_errvar, all_timelags))

results_df.to_pickle(path_out1 +
                     r'results_df_' +
                     str(all_rho[0]).replace('.', '') +
                     '_' +
                     str(all_errvar[0]).replace('.', '') +
                     '_' + str(all_weights[0]).replace('.', '') +
                     '_' + str(all_timelags[0]) +
                     '_' + str(all_off_diag_covs[0]).replace('.', '') +
                     '_' + case +
                     '_' + str(gamma).replace('.', '') +
                     '.pkl')

# results_df_05_001_1 = pd.read_pickle(path_out1+r'results_df_05_001_1.pkl')

# for rho, errvar, time_lag in product(all_rho, all_errvar, all_timelags):
#     your_result = get_results_from_model(rho, errvar, time_lag, T, dt, var_means)

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
    



