#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 18 17:54:03 2021

@author: nadinespy
"""
import numpy as np


#logarithm of the determinant of matrix A
def logdet(A):
    B = 2*sum(np.log(np.diag(np.linalg.cholesky(A))))
    return B

def double_redundancy_mmi(cov_matrix, cond_cov_matrix, cond_cov_part11, cond_cov_part22, cond_cov_part12, cond_cov_part21, time_lag):
    
    double_redundancy_mmi = np.zeros(cov_matrix.shape[2])
    
    for i in range(1, cov_matrix.shape[2]):
        all_mutual_info = (0.5 * np.log(np.array([cov_matrix[0,0,i-time_lag]/(cond_cov_part11[i]+0j), cov_matrix[1,1,i-time_lag]/(cond_cov_part22[i]+0j), cov_matrix[1,1,i-time_lag]/(cond_cov_part12[i]+0j), cov_matrix[0,0,i-time_lag]/(cond_cov_part21[i]+0j)]))).real
        
        # terms in all_mutual_info can get negative (e. g., when we take the log of a number that due to numerical instabilities is not exactly, but slightly smaller than 1, in which case the result gets negative)
        # which is why we make these 0 (as the log of 1 is 0)
        for j in range(len(all_mutual_info)):
            if all_mutual_info[j] < 0:
                all_mutual_info[j] = 0
        
        double_redundancy_mmi[i] = np.min(np.nan_to_num(all_mutual_info))
        
    return double_redundancy_mmi
            



# old (and wrong, because based on cond_cov_part1 etc. are conditioned on all variables) definition

# def double_redundancy_mmi(cov_matrix, cond_cov_matrix):
    
#     double_redundancy_mmi = np.zeros(cov_matrix.shape[2])
    
#     for i in range(1, cov_matrix.shape[2]):
#           cov_part1 = cov_matrix[0,0,i-1]
#           cov_part2 = cov_matrix[1,1,i-1]
        
#          # cond_cov_part1 = cond_cov_matrix[0,0,i]                                    # get variance of x1_t-tau conditioned on x1_t
#          # cond_cov_part2 = cond_cov_matrix[1,1,i]                                    # get variance of x2_t-tau conditioned on x2_t
#          # cond_cov_part12 = cond_cov_matrix[0,1,i]                                   # get variance of x1_t-tau conditioned on x2_t
#          # cond_cov_part21 = cond_cov_matrix[1,0,i]                                   # get variance of x2_t-tau conditioned on x1_t
        
#         try: 
#              mutual_info11 = 0.5 * np.log(((np.linalg.det(np.expand_dims(cov_part1, axis=(0,1)))+0j).real) / ((np.linalg.det(np.expand_dims(cond_cov_part11, axis=(0,1)))+0j).real))
#              mutual_info22 = 0.5 * np.log(((np.linalg.det(np.expand_dims(cov_part2, axis=(0,1)))+0j).real) / ((np.linalg.det(np.expand_dims(cond_cov_part22, axis=(0,1)))+0j).real))
#              mutual_info12 = 0.5 * np.log(((np.linalg.det(np.expand_dims(cov_part1, axis=(0,1)))+0j).real) / ((np.linalg.det(np.expand_dims(cond_cov_part12, axis=(0,1)))+0j).real))
#              mutual_info21 = 0.5 * np.log(((np.linalg.det(np.expand_dims(cov_part2, axis=(0,1)))+0j).real) / ((np.linalg.det(np.expand_dims(cond_cov_part21, axis=(0,1)))+0j).real))
        
#             all_mutual_info = np.array([mutual_info11, mutual_info22, mutual_info12, mutual_info21])
#             where_are_NaNs = np.isnan(all_mutual_info)
#             all_mutual_info[where_are_NaNs] = 0

#             double_redundancy_mmi[i] = np.nanmin(all_mutual_info)

#         except:
#             pass
        
#     return double_redundancy_mmi
            