#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 18 17:54:03 2021

@author: nadinespy
"""
import numpy as np
import scipy.linalg as la


#logarithm of the determinant of matrix A
def logdet(A):
    B = 2*sum(np.log(np.diag(np.linalg.cholesky(A))))
    return B


def double_redundancy_mmi(cov_matrix, cond_cov_matrix):
    
    double_redundancy_mmi = np.zeros(cov_matrix.shape[2])
    
    for i in range(1, cov_matrix.shape[2]):
        cov_part1 = cov_matrix[0,0,i-1]
        cov_part2 = cov_matrix[1,1,i-1]
        
        cond_cov_part1 = cond_cov_matrix[0,0,i]                                    # get variance of x1_t-tau conditioned on x1_t
        cond_cov_part2 = cond_cov_matrix[1,1,i]                                    # get variance of x2_t-tau conditioned on x2_t
        cond_cov_part12 = cond_cov_matrix[0,1,i]                                   # get variance of x1_t-tau conditioned on x2_t
        cond_cov_part21 = cond_cov_matrix[1,0,i]                                   # get variance of x2_t-tau conditioned on x1_t
        

        mutual_info11 = 0.5 * np.log(((np.linalg.det(np.expand_dims(cov_part1, axis=(0,1)))+0j).real) / ((np.linalg.det(np.expand_dims(cond_cov_part1, axis=(0,1)))+0j).real))
        mutual_info22 = 0.5 * np.log(((np.linalg.det(np.expand_dims(cov_part2, axis=(0,1)))+0j).real) / ((np.linalg.det(np.expand_dims(cond_cov_part2, axis=(0,1)))+0j).real))
        mutual_info12 = 0.5 * np.log(((np.linalg.det(np.expand_dims(cov_part1, axis=(0,1)))+0j).real) / ((np.linalg.det(np.expand_dims(cond_cov_part12, axis=(0,1)))+0j).real))
        mutual_info21 = 0.5 * np.log(((np.linalg.det(np.expand_dims(cov_part2, axis=(0,1)))+0j).real) / ((np.linalg.det(np.expand_dims(cond_cov_part21, axis=(0,1)))+0j).real))
        
        double_redundancy_mmi[i] = np.amin([mutual_info11, mutual_info22, mutual_info12, mutual_info21])
        
    return double_redundancy_mmi
            
        