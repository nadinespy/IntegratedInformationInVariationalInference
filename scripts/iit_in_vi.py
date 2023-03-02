#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 18 17:54:03 2021

@author: nadinespy
"""
import numpy as np
import scipy.linalg as la
from oct2py import octave as oc
import os
import matlab.engine


# logarithm of the determinant of matrix A
def logdet(A):
    B = 2*sum(np.log(np.diag(np.linalg.cholesky(A))))
    return B


def get_double_red_mmi(cov_matrix, cond_cov_matrix, cond_cov_part11,
                       cond_cov_part22, cond_cov_part12, cond_cov_part21, time_lag):

    double_redundancy_mmi = np.zeros(cov_matrix.shape[2])

    for i in range(time_lag, cov_matrix.shape[2]):
        all_mutual_info = (0.5 * np.log(np.array(
            [cov_matrix[0, 0, i-time_lag]/(cond_cov_part11[i]+0j),
             cov_matrix[1, 1, i-time_lag]/(cond_cov_part22[i]+0j),
             cov_matrix[1, 1, i-time_lag]/(cond_cov_part12[i]+0j),
             cov_matrix[0, 0, i-time_lag]/(cond_cov_part21[i]+0j)]))).real

        # terms in all_mutual_info can get negative (e. g., when we take the
        # log of a number that due to numerical instabilities is not exactly,
        # but slightly smaller than 1, in which case the result gets negative)
        # which is why we make these 0 (as the log of 1 is 0)
        for j in range(len(all_mutual_info)):
            if all_mutual_info[j] < 0:
                all_mutual_info[j] = 0

    double_redundancy_mmi[i] = np.min(np.nan_to_num(all_mutual_info))

    return double_redundancy_mmi


def get_mean_continuous(weighted_inv_true_cov, t, true_means, init_variational_means):
    means = (np.eye(2)-la.expm(-weighted_inv_true_cov * t)) @ true_means + \
        la.expm(-weighted_inv_true_cov * t) @ init_variational_means
    return means

def get_mean_discrete(gamma, weighted_inv_true_cov, previous_var_means, true_means):
    var_means = (np.eye(2)- gamma *weighted_inv_true_cov) @ previous_var_means + \
        gamma * weighted_inv_true_cov @ true_means
    return var_means

def get_cov_continuous(weighted_inv_true_cov, t, errvar, initial_covariance):
    # not sure yet which of the following two is correct
    covariance = la.expm(-weighted_inv_true_cov * t) @ initial_covariance \
        @ la.expm(-weighted_inv_true_cov * t) + 0.5 * errvar**2 * weighted_inv_true_cov \
            @ weighted_inv_true_cov @ la.inv(weighted_inv_true_cov) \
                @ (np.eye(2) - la.expm(-weighted_inv_true_cov * 2 * t))
    return covariance

def get_cov_discrete(gamma, weighted_inv_true_cov, previous_covariance):
    covariance = (np.eye(2) - gamma *weighted_inv_true_cov) @ previous_covariance \
        @ (np.eye(2) - gamma *weighted_inv_true_cov)  +  gamma**2 * weighted_inv_true_cov \
            @ gamma @ weighted_inv_true_cov
    return covariance

def get_time_lagged_cov_continuous(weighted_inv_true_cov, t, s, errvar, initial_covariance):
    time_lagged_covariance = la.expm(-weighted_inv_true_cov * (t+s)) @ initial_covariance + \
        0.5 * errvar ** 2 * la.inv(weighted_inv_true_cov) @ (la.expm(weighted_inv_true_cov * \
            (s - t)) - la.expm(weighted_inv_true_cov * (-t - s)))
    return time_lagged_covariance

def get_time_lagged_cov_discrete(gamma, weighted_inv_true_cov, previous_covariance):
    time_lagged_covariance = (np.eye(2)- gamma *weighted_inv_true_cov) @ previous_covariance
    return time_lagged_covariance

def get_cond_cov_full(cov_past, cov_present, time_lagged_cov_present):
    conditional_covariance_full = cov_past - time_lagged_cov_present.T \
        @ la.pinv(cov_present) @ time_lagged_cov_present
    return conditional_covariance_full


def get_cond_cov_parts(cov_past_parts, cov_present_parts, time_lagged_cov_present_parts):
    conditional_covariance_parts = cov_past_parts - time_lagged_cov_present_parts * \
        np.reciprocal(cov_present_parts) * time_lagged_cov_present_parts
    return conditional_covariance_parts


def get_kl_div(A, B, mu, mean, covariance):
    kldiv = np.sum(0.5 * (1-np.log(2 * np.pi/B))) + \
        0.5 * np.log(2 * np.pi * np.linalg.det(np.linalg.inv(A))) + \
        np.sum(0.5 * B * np.diag(A)) + 0.5 * (mean - mu) @ A @ (mean - mu) + \
        0.5 * np.sum(A * covariance)
    return kldiv


def get_phi(cov_past, cond_cov_present_full, cov_past_parts11, cond_cov_present_parts11,
            cov_past_parts22, cond_cov_present_parts22):
    phi = (0.5 * np.log(np.linalg.det(cov_past) / ((np.linalg.det(cond_cov_present_full))+0j) / 
        (cov_past_parts11/(cond_cov_present_parts11+0j)) / 
        (cov_past_parts22/(cond_cov_present_parts22+0j)))).real
    return phi


def get_phiid(time_series, time_lag, redundancy_func):

    main_path = '/media/nadinespy/NewVolume1/work/current_projects/viiit/viiit_with_miguel/' \
        'IntegratedInformationInVariationalInference/scripts/'
    os.chdir(main_path)
    oc.addpath(main_path)
    oc.javaaddpath(main_path+'/infodynamics.jar')
    oc.eval('pkg load statistics')

    phiid = oc.PhiIDFull(time_series, time_lag, redundancy_func)

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

    # phi =     - {1}{2}-->{1}{2}                            (double-redundancy)
    #           + {12}-->{12}                                (causal decoupling)
    #           + {12}-->{1} + {12}-->{2} + {12}-->{1}{2}    (downward causation)
    #           + {1}{2}-->{12} + {1}-->{12} + {2}-->{12}    (upward causation)
    #           + {1}-->{2} + {2}-->{1}                      (transfer)

    # synergy = causal decoupling + downward causation + upward causation

    emergence_capacity_phiid = phiid.str + phiid.stx + phiid.sty + phiid.sts
    downward_causation_phiid = phiid.str + phiid.stx + phiid.sty
    synergy_phiid = emergence_capacity_phiid + phiid.rts + phiid.xts + phiid.yts
    transfer_phiid = phiid.xty + phiid.ytx
    phi_phiid = - phiid.rtr + synergy_phiid + transfer_phiid
    phiR_phiid = phi_phiid + phiid.rtr

    return phiid, emergence_capacity_phiid, downward_causation_phiid, synergy_phiid,
    transfer_phiid, phi_phiid, phiR_phiid

def get_phiid_analytical(time_lagged_COV, time_lagged_COV_time_lag, time_lagged_COND_COV, mx, mx_time_lag, redundancy_func):

    # Matlab engine stuff 
    eng = matlab.engine.start_matlab()
    
    # building full time-lagged covariance matrix (2xD-by-2xD):
    # \Sigma(X) = [[\Sigma_x1(t)x1 (t)    \Sigma_x1(t)x2(t)         \Sigma_x1(t)x1(t-tau)         \Sigma_x1(t)x2(t-tau)]                        
    #         [\Sigma_x2(t)x1(t)          \Sigma_x2(t)x2(t)         \Sigma_x2(t)x1(t-tau)         \Sigma_x2(t)x2(t-tau)]
    #         [\Sigma_x1(t-tau)x1(t)      \Sigma_x1(t-tau)x2(t)     \Sigma_x1(t-tau)x1(t-tau)     \Sigma_x1(t-tau)x2(t-tau)]   
    #         [\Sigma_x2(t-tau)x1(t)      \Sigma_x2(t-tau)x2(t)     \Sigma_x2(t-tau)x1(t-tau)     \Sigma_x2(t-tau)x2(t-tau)]]

    a = np.concatenate((time_lagged_COV, 
                        time_lagged_COND_COV), axis=1)
    b = np.concatenate((time_lagged_COND_COV, 
                        time_lagged_COV_time_lag), axis=1)
    full_time_lagged_COV = np.concatenate([a, b])
    
    
    # building array of means corresponding to full time-lagged covariance matrix
    all_means = np.concatenate((mx, mx_time_lag))
        
    # convert to matlab-compatible data type *double*
    full_time_lagged_COV = matlab.double(full_time_lagged_COV.tolist())
    all_means = matlab.double(all_means.tolist())

    phiid = eng.PhiIDFull_Analytical(full_time_lagged_COV, all_means, redundancy_func)
    # synergy = causal decoupling + downward causation + upward causation
    
    emergence_capacity_phiid = phiid['str'] + phiid['stx'] + \
        phiid['sty'] + phiid['sts']
    downward_causation_phiid = phiid['str'] + phiid['stx'] + \
        phiid['sty']
    synergy_phiid = emergence_capacity_phiid + phiid['rts'] + \
        phiid['xts'] + phiid['yts']
    transfer_phiid = phiid['xty'] + phiid['ytx']
    phi_phiid = - phiid['rtr'] + synergy_phiid + transfer_phiid
    phiR_phiid = phi_phiid + phiid['rtr']
    
    return (phiid, emergence_capacity_phiid, downward_causation_phiid, synergy_phiid, 
            transfer_phiid, phi_phiid, phiR_phiid)
   