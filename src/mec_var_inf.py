#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 18 17:54:03 2021

@author: nadinespy
"""
import numpy as np
import scipy.linalg as la
from itertools import combinations
import os
import matlab.engine

eng = matlab.engine.start_matlab()

# explicitly specify the path to your module directory
module_dir = '/media/nadinespy/NewVolume1/work/phd/projects/mec_var_inf/mec_var_inf/src'
# add directory to Matlab path for .m files
eng.addpath(module_dir)
# add Java file to Matlab's Java class path
jar_file = os.path.join(module_dir, 'infodynamics.jar')
eng.eval(f"javaaddpath('{jar_file}')", nargout=0)

# logarithm of the determinant of matrix A
def logdet(A):
    B = 2*sum(np.log(np.diag(np.linalg.cholesky(A))))
    return B

def get_true_cov(rho, n_var):
    """
    Construct N-dimensional true covariance matrix.

    Args:
        rho (float): Correlation parameter between 0 and 1
        n_var (int): Dimensionality of the system

    Returns:
        numpy.ndarray: Reference covariance matrix S of shape (n_var, n_var)
    """
    true_cov = np.eye(n_var)
    for i in range(n_var):
        for j in range(i+1, n_var):
            true_cov[i,j] = np.sqrt(true_cov[i,i] * true_cov[j,j]) * rho
            true_cov[j,i] = true_cov[i,j]
    return true_cov

def get_approx_cov(true_cov, weight):
    """
    Construct inverse of true covariance matrix, weighted inverse of true covariance matrix, 
    and inverse of mean-field true covariance matrix for N-dimensional system.

    Args:
        inv_true_cov (numpy.ndarray): Reference covariance matrix of shape (N, N)
        weight (float): Weighting factor for off-diagonal elements

    Returns:
        dict containing:
            'inv_true_cov': Inverse of covariance matrix
            'weighted_inv_true_cov': Weighted inverse of covariance matrix
            'mean_field_inv_true_cov': Inverse of true mean-field covariance matrix
            (diagonal matrix from inv_true_cov)
    """
    inv_true_cov = np.linalg.inv(true_cov)
    weighted_inv_true_cov = inv_true_cov.copy()

    # weight off-diagonal elements
    for i in range(true_cov.shape[0]):
        for j in range(true_cov.shape[0]):
            if i != j:
                weighted_inv_true_cov[i,j] = weight * inv_true_cov[i,j]

    # inverse of mean-field covariance (diagonal matrix from inv_true_cov)
    mean_field_inv_true_cov = np.diag(inv_true_cov)

    return inv_true_cov, weighted_inv_true_cov, mean_field_inv_true_cov

def get_K_with_noise_corr(weighted_inv_true_cov, gamma, errvar, mean_noise_corr=0.5, 
                          spread_factor=0.2, seed=None):
    """
    Computes K matrix with randomly correlated errors.

    Args:
        A2 (numpy.ndarray): Input inverse covariance matrix of shape (N, N)
        gamma (float): Integration step
        errvar (float): Error standard deviation
        mean_noise_corr (float): Target average correlation (0 to 1)
        seed (int, optional): Random seed for reproducibility

    Returns:
        numpy.ndarray: Computed K matrix of shape (N, N)
    """
    size = weighted_inv_true_cov.shape[0]
    corr_matrix = get_rand_corr_matrix(size, mean_noise_corr, spread_factor, seed)
    corr_noise_matrix = errvar**2 * corr_matrix

    return np.linalg.inv(weighted_inv_true_cov @ corr_noise_matrix 
                         @ weighted_inv_true_cov.T) / gamma**2

def get_rand_corr_matrix(size, mean_corr=0.5, spread_factor=0.2, seed=None):
    """
    Generates a random correlation matrix with a specified average correlation.

    Args:
        size (int): Size of the matrix
        mean_corr (float): Target average correlation, between 0 and 1
            0 means uncorrelated (diagonal matrix)
            1 means perfectly correlated (all 1s)
        seed (int, optional): Random seed for reproducibility

    Returns:
        numpy.ndarray: Valid correlation matrix of shape (size, size)
    """
    if seed is not None:
        np.random.seed(seed)

    # generate random matrix
    #random_matrix = np.random.rand(size, size)

    # alternative way:
    # np.random.rand(size, size) generates values in [0,1]
    #   - subtracting 0.5 centers these around 0 (range [-0.5,0.5])
    #   - multiplying by 2 gives range [-1,1]
    #   - multiplying by C controls the spread around mean_corr
    #   - adding mean_corr shifts the whole distribution to center around mean_corr
    random_matrix = mean_corr + (np.random.rand(size, size) - 0.5) * 2 * spread_factor

    # make it symmetric
    random_matrix = (random_matrix + random_matrix.T) / 2

    # set diagonal to 1
    np.fill_diagonal(random_matrix, 1)

    # set minimum value for eigenvalues of noise correlation matrix
    min_eigenvalues = 1e-05

    # scale off-diagonal elements to achieve target average correlation
    off_diagonal_mask = ~np.eye(size, dtype=bool)
    current_average = random_matrix[off_diagonal_mask].mean()

    # scale off-diagonal elements
    if mean_corr > 0:  # Only scale if we want correlation
        scale_factor = mean_corr / current_average
        
        #random_matrix[off_diagonal_mask] *= scale_factor

        # alternative below - advantages:
        #     - scaling (*=) can amplify any existing patterns/biases in the random matrix, 
        #       while addition shifts all values equally
        #     - with multiplication, if current_average is very small, scale_factor becomes very 
        #       large, which can lead to numerical instability
        #     - addition directly enforces the desired mean by adding the exact difference needed, 
        #       while multiplication assumes the pattern scales linearly
        #     - [+= mean_corr - current_average] simply adds whatever difference is needed to make 
        #       the current average equal the target mean_corr;
        #       it's a more direct way to achieve the desired mean correlation
        random_matrix[off_diagonal_mask] += mean_corr - current_average

    # ensure matrix is positive definite
    eigenvalues, eigenvectors = np.linalg.eigh(random_matrix)

    # force minimum eigenvalues
    eigenvalues = np.maximum(eigenvalues, min_eigenvalues)

    # reconstruct matrix
    corrected_corr_matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

    # ensure diagonal is exactly 1
    d = np.sqrt(np.diag(corrected_corr_matrix))
    corrected_corr_matrix = ((corrected_corr_matrix.T / d).T) / d

    return corrected_corr_matrix

def get_phi_for_min_bipartition(same_time_COV, time_lagged_COV):
    """
    Find the bipartition that minimizes phi/corrected phi.

    Args:
        same_time_COV (numpy.ndarray): Same-time covariance matrix
        time_lagged_COV (numpy.ndarray): Time-lagged covariance matrix

    Returns:
        dict containing:
            'partition': Tuple of (part1_indices, part2_indices) for minimum partition
            'phi': Minimum phi value
            'phi_corrected': Corresponding phiR value
            'results': Complete results dictionary for minimum partition
    """
    n_var = same_time_COV.shape[0]
    valid_bipartitions = get_valid_bipartitions(n_var)

    min_phi = float('inf')
    min_phi_corrected = float('inf')
    min_double_red_mmi = float('inf')
    min_partition = None

    for part1_indices, part2_indices in valid_bipartitions:

        phi, phi_corrected, double_red_mmi = get_phi_measures(same_time_COV, time_lagged_COV, \
                                                              part1_indices, part2_indices)

        if phi < min_phi:
            min_phi = phi
            min_phi_corrected = phi_corrected
            min_double_red_mmi = double_red_mmi
            min_partition = (part1_indices, part2_indices)

    return min_partition, min_phi, min_phi_corrected, min_double_red_mmi

def get_valid_bipartitions(n_var):
    """
    Generate valid bipartitions for N-dimensional system.
    A bipartition splits the system into two parts with sizes differing by at most 1.

    Args:
        n_var (int): System dimensionality

    Returns:
        list of tuples: Each tuple contains two lists representing the bipartition.
        Example for N=3: [([0, 1], [2])] represents variables 0 & 1 in first part, 2 in second part
        Example for N=4: [([0, 1], [2, 3])] represents equal split
    """
    if n_var == 2:
        bipartition = [([0], [1])]
        return bipartition

    # For N>2, determine size of smaller partition
    part1_size = n_var // 2
    sizes_to_try = [part1_size]
    if n_var % 2 == 1:
        sizes_to_try.append(part1_size + 1)

    valid_bipartitions = []
    indices = list(range(n_var))

    for size in sizes_to_try:
        # Generate all possible combinations for part1 of the given size
        for part1_indices in combinations(indices, size):
            part1 = list(part1_indices)
            part2 = [i for i in indices if i not in part1]
            valid_bipartitions.append((part1, part2))

    return valid_bipartitions

def get_cond_covs(same_time_COV, time_lagged_COV, part1_indices, part2_indices):
    """
    Compute conditional covariance matrices for a specific bipartition.

    Args:
    same_time_COV (numpy.ndarray): Same-time covariance matrix of shape (N, N)
    time_lagged_COV (numpy.ndarray): Time-lagged covariance matrix of shape (N, N)
    part1_indices (list): Indices of variables in first part
    part2_indices (list): Indices of variables in second part

    Returns:
    time_lagged_COND_COV_FULL: Full system conditional covariance
    same_time_COND_COV_PART1: Conditional covariance for part 1
    same_time_COND_COV_PART2: Conditional covariance for part 2

    Raises:
    ValueError: If any of the conditional covariance calculations fail
    """

    # full system conditional covariance
    time_lagged_COND_COV_FULL = same_time_COV - time_lagged_COV.T @ \
        la.pinv(same_time_COV) @ time_lagged_COV

    # Part 1 conditional covariance
    same_time_COV_PART1 = same_time_COV[np.ix_(part1_indices, part1_indices)]
    time_lagged_COV_PART1 = time_lagged_COV[np.ix_(part1_indices, part1_indices)]
    same_time_COND_COV_PART1 = same_time_COV_PART1 - time_lagged_COV_PART1.T @ \
        la.pinv(same_time_COV_PART1) @ time_lagged_COV_PART1

    # Part 2 conditional covariance
    same_time_COV_PART2 = same_time_COV[np.ix_(part2_indices, part2_indices)]
    time_lagged_COV_PART2 = time_lagged_COV[np.ix_(part2_indices, part2_indices)]
    same_time_COND_COV_PART2 = same_time_COV_PART2 - time_lagged_COV_PART2.T @ \
        la.pinv(same_time_COV_PART2) @ time_lagged_COV_PART2

    return time_lagged_COND_COV_FULL, same_time_COND_COV_PART1, same_time_COND_COV_PART2

def get_entropies(same_time_COV, time_lagged_COND_COV_FULL, \
                      time_lagged_COND_COV_PART1, time_lagged_COND_COV_PART2, \
                        part1_indices, part2_indices):
    """
    Compute entropy measures for bipartitioned system.

    Args:
    same_time_COV (numpy.ndarray): NxN same-time covariance matrix
    time_lagged_COV (numpy.ndarray): NxN time-lagged covariance matrix
    time_lagged_COND_COV_FULL: Full system conditional covariance
    time_lagged_COND_COV_PART1: Conditional covariance for part 1
    time_lagged_COND_COV_PART2: Conditional covariance for part 2
    part1_indices (list): Indices for first part of bipartition
    part2_indices (list): Indices for second part of bipartition

    Returns:
    entropy_PRESENT_PART1 (float): Entropy of first part
    entropy_PRESENT_PART2 (float): Entropy of second part
    entropy_PRESENT_FULL (float): Joint entropy of full system
    mi_SAME_TIME_FULL (float): Mutual information between parts at same time
    mi_PAST_PRESENT_PART1' (float): Mutual information between past and present for part 1
    mi_PAST_PRESENT_FULL (float): Mutual information between past and present for full system
    """

    # extract submatrices for part 1
    same_time_COV_PART1 = same_time_COV[np.ix_(part1_indices, part1_indices)]
    same_time_COV_PART2 = same_time_COV[np.ix_(part2_indices, part2_indices)]

    entropy_PRESENT_PART1 = 0.5 * same_time_COV_PART1.shape[0] * (1 + np.log(2 * np.pi)) + \
        0.5 * np.log(np.linalg.det(same_time_COV_PART1))
    entropy_PRESENT_PART2 = 0.5 * same_time_COV_PART2.shape[0] * (1 + np.log(2 * np.pi)) + \
        0.5 * np.log(np.linalg.det(same_time_COV_PART2))

    # calculate joint entropy for full system:
    # H(X) = n/2 * (1 + ln(2π)) + 1/2 * ln(det(Σ))
    # where n is the system dimension and Σ is the covariance matrix
    entropy_PRESENT_FULL = 0.5 * same_time_COV.shape[0] * (1 + np.log(2 * np.pi)) + \
        0.5 * np.log(np.linalg.det(same_time_COV))

    # calculate mutual information between parts at same time:
    # I(X1;X2) = H(X1) + H(X2) - H(X1,X2)
    mi_SAME_TIME_FULL = entropy_PRESENT_PART1 + entropy_PRESENT_PART2 - entropy_PRESENT_FULL

    # calculate mutual information between past and present for part 1:
    # I(X1(t);X1(t-τ)) = 1/2 * ln(det(Σ11)/det(Σ11|1))
    # where Σ11|1 is the conditional covariance of X1(t-τ) given X1(t)
    mi_PAST_PRESENT_PART1 = 0
    if time_lagged_COND_COV_PART1 is not None:
      mi_PAST_PRESENT_PART1 = 0.5 * np.log(np.linalg.det(same_time_COV_PART1)/np.linalg.det(time_lagged_COND_COV_PART1))

    # same for part 2
    mi_PAST_PRESENT_PART2 = 0
    if time_lagged_COND_COV_PART2 is not None:
      mi_PAST_PRESENT_PART2 = 0.5 * np.log(np.linalg.det(same_time_COV_PART2)/np.linalg.det(time_lagged_COND_COV_PART2))

    # calculate mutual information between past and present for full system:
    # I(X(t);X(t-τ)) = 1/2 * ln(det(Σ)/det(Σ|X))
    # where Σ|X is the conditional covariance of X(t-τ) given X(t)
    mi_PAST_PRESENT_FULL = 0
    if time_lagged_COND_COV_FULL is not None:
      mi_PAST_PRESENT_FULL = 0.5 * np.log(np.linalg.det(same_time_COV)/np.linalg.det(time_lagged_COND_COV_FULL))

    return np.real(entropy_PRESENT_PART1), np.real(entropy_PRESENT_PART2), \
        np.real(entropy_PRESENT_FULL), np.real(mi_PAST_PRESENT_PART1),  np.real(mi_PAST_PRESENT_PART2), \
        np.real(mi_PAST_PRESENT_FULL), np.real(mi_SAME_TIME_FULL)

def get_phi_measures(same_time_COV, time_lagged_COV, part1_indices, part2_indices):
    """
    Compute Φ and ΦR for a specific bipartition.

    Args:
        same_time_COV (numpy.ndarray): Same-time covariance matrix
        time_lagged_COV (numpy.ndarray): Time-lagged covariance matrix
        part1_indices (list): Indices for first part
        part2_indices (list): Indices for second part

    Returns:
        dict containing:
            'phi': Integrated information
            'phiR': Integrated information with redundancy
            'double_red': Double redundancy
    """

    try:

        # ----------------------------------------------------------------------
        # compute phi using I(X;Y) = H(X) - H(X|Y)
        # ----------------------------------------------------------------------

        time_lagged_COND_COV_FULL, time_lagged_COND_COV_PART1, time_lagged_COND_COV_PART2 = \
            get_cond_covs(same_time_COV, time_lagged_COV, part1_indices, part2_indices)
        
        # extract submatrices for part 1
        same_time_COV_PART1 = same_time_COV[np.ix_(part1_indices, part1_indices)]
        same_time_COV_PART2 = same_time_COV[np.ix_(part2_indices, part2_indices)]

        phi_FULL = 0.5 * np.log(np.linalg.det(same_time_COV)/ \
                                ((np.linalg.det(time_lagged_COND_COV_FULL)+0j)))
        phi_PART1 = 0.5 * np.log(np.linalg.det(same_time_COV_PART1)/ \
                                 ((np.linalg.det(time_lagged_COND_COV_PART1)+0j)))
        phi_PART2 = 0.5 * np.log(np.linalg.det(same_time_COV_PART2)/ \
                                 ((np.linalg.det(time_lagged_COND_COV_PART2)+0j)))

        # ----------------------------------------------------------------------
        # compute phi using I(X;Y) = H(X) + H(Y) - H(X,Y)
        # ----------------------------------------------------------------------
        # FULL SYSTEM

        # calculate entropies for the full system
        #entropy_PRESENT_FULL = 0.5 * np.log(np.linalg.det(same_time_COV))
        #entropy_PAST_FULL = 0.5 * np.log(np.linalg.det(same_time_COV))  # since it's the same 
                                                                        # time covariance

        # construct full time-lagged covariance matrix (block-toeplitz matrix) for full system
        #full_time_lagged_COV_FULL = np.block([[same_time_COV, time_lagged_COV],
        #                         [time_lagged_COV.T, same_time_COV]])
        #entropy_PRESENT_PAST_FULL = 0.5 * np.log(np.linalg.det(full_time_lagged_COV_FULL))

        # calculate MI for full system
        #phi_FULL = entropy_PRESENT_FULL + entropy_PAST_FULL - entropy_PRESENT_PAST_FULL

        # ----------------------------------------------------------------------
        # PART 1

        # extract submatrices for part 1
        #same_time_COV_PART1 = same_time_COV[np.ix_(part1_indices, part1_indices)]
        #time_lagged_COV_PART1 = time_lagged_COV[np.ix_(part1_indices, part1_indices)]

        # calculate entropies for part 1
        #entropy_PRESENT_PART1 = 0.5 * np.log(np.linalg.det(same_time_COV_PART1))
        #entropy_PAST_PART1 = entropy_PRESENT_PART1  # same time covariance

        # construct full time-lagged covariance matrix (block-toeplitz matrix) for part 1
        #full_time_lagged_COV_PART1 = np.block([[same_time_COV_PART1, time_lagged_COV_PART1],
        #                          [time_lagged_COV_PART1.T, same_time_COV_PART1]])
        #entropy_PRESENT_PAST_PART1 = 0.5 * np.log(np.linalg.det(full_time_lagged_COV_PART1))

        # calculate MI for part 1
        #phi_PART1 = entropy_PRESENT_PART1 + entropy_PAST_PART1 - entropy_PRESENT_PAST_PART1

        # ----------------------------------------------------------------------
        # PART 2

        # extract submatrices for part 2
        #same_time_COV_PART2 = same_time_COV[np.ix_(part2_indices, part2_indices)]
        #time_lagged_COV_PART2 = time_lagged_COV[np.ix_(part2_indices, part2_indices)]
        
        # calculate MI for part 2
        #entropy_PRESENT_PART2 = 0.5 * np.log(np.linalg.det(same_time_COV_PART2))
        #entropy_PAST_PART2 = entropy_PRESENT_PART2  # same time covariance

        # construct full time-lagged covariance matrix (block-toeplitz matrix) for part 2
        #full_time_lagged_COV_PART2 = np.block([[same_time_COV_PART2, time_lagged_COV_PART2],
        #                          [time_lagged_COV_PART2.T, same_time_COV_PART2]])
        #entropy_PRESENT_PAST_PART2 = 0.5 * np.log(np.linalg.det(full_time_lagged_COV_PART2))

        # calculate MI for part 2
        #phi_PART2 = entropy_PRESENT_PART2 + entropy_PAST_PART2 - entropy_PRESENT_PAST_PART2

        phi = np.real(phi_FULL - (phi_PART1 + phi_PART2))

        double_red_mmi = get_double_red_mmi(same_time_COV, time_lagged_COV, part1_indices, part2_indices)
        phi_corrected = phi + double_red_mmi

    except:
        print("Error in phi/phiR calculation.")  # Debug print
        phi = 0
        phi_corrected = 0
        double_red = 0

    return phi, phi_corrected, double_red_mmi

def get_double_red_mmi(same_time_COV, time_lagged_COV, part1_indices, part2_indices):
    """ 
    Compute double redundancy between two parts of a system by calculating mutual information 
    between past and present states of the parts and taking their minimum.
    
    This implements the formula for double redundancy as min(I(x₁(t-τ);x₁(t)), I(x₂(t-τ);x₂(t)), 
    I(x₁(t-τ);x₂(t)), I(x₂(t-τ);x₁(t))), where:
    - I(x_i(t-τ);x_j(t)) is the mutual information between past of part i and present of part j
    - Each I is calculated as 0.5 * log(det(Σ_i) / det(Σ_i|j))
    - Σ_i|j is the conditional covariance matrix of part i given part j
    
    Parameters
    ----------
    same_time_COV : numpy.ndarray
        Same-time covariance matrix of shape (N, N)
    time_lagged_COV : numpy.ndarray
        Time-lagged covariance matrix of shape (N, N)
    part1_indices : list
        Indices for the first part of the bipartition
    part2_indices : list
        Indices for the second part of the bipartition
        
    Returns
    -------
    float
        Double redundancy value, which is the minimum of the four mutual information terms.
        Returns 0 for negative values due to numerical instabilities.
        
    Notes
    -----
    The function computes four conditional covariances:
    1. time_lagged_COND_COV_PART11: Σ(x₁(t-τ)|x₁(t)) - past of part 1 given present of part 1
    2. time_lagged_COND_COV_PART22: Σ(x₂(t-τ)|x₂(t)) - past of part 2 given present of part 2
    3. time_lagged_COND_COV_PART12: Σ(x₁(t-τ)|x₂(t)) - past of part 1 given present of part 2
    4. time_lagged_COND_COV_PART21: Σ(x₂(t-τ)|x₁(t)) - past of part 2 given present of part 1
    
    Each conditional covariance is calculated using the formula:
    Σ(x_i(t-τ)|x_j(t)) = Σ_ii - Σ_ij * Σ_jj^(-1) * Σ_ji
    
    The function handles numerical instabilities by:
    - Adding a small complex component (0j) to prevent negative logarithms
    - Setting negative MI values to 0
    - Converting NaN values to 0 using np.nan_to_num
    """

    # For 11: part1 past with part1 present
    time_lagged_COND_COV_PART11 = same_time_COV[part1_indices[0], part1_indices[0]] - \
                      time_lagged_COV[part1_indices[0], part1_indices[0]] * \
                      np.reciprocal(same_time_COV[part1_indices[0], part1_indices[0]]) * \
                      time_lagged_COV[part1_indices[0], part1_indices[0]]

    # For 22: part2 past with part2 present
    time_lagged_COND_COV_PART22 = same_time_COV[part2_indices[0], part2_indices[0]] - \
                      time_lagged_COV[part2_indices[0], part2_indices[0]] * \
                      np.reciprocal(same_time_COV[part2_indices[0], part2_indices[0]]) * \
                      time_lagged_COV[part2_indices[0], part2_indices[0]]

    # For 12: part1 past with part2 present
    time_lagged_COND_COV_PART12 = same_time_COV[part1_indices[0], part1_indices[0]] - \
                      time_lagged_COV[part2_indices[0], part1_indices[0]] * \
                      np.reciprocal(same_time_COV[part2_indices[0], part2_indices[0]]) * \
                      time_lagged_COV[part2_indices[0], part1_indices[0]]

    # For 21: part2 past with part1 present
    time_lagged_COND_COV_PART21 = same_time_COV[part2_indices[0], part2_indices[0]] - \
                      time_lagged_COV[part1_indices[0], part2_indices[0]] * \
                      np.reciprocal(same_time_COV[part1_indices[0], part1_indices[0]]) * \
                      time_lagged_COV[part1_indices[0], part2_indices[0]]

    # Calculate all four MI terms
    all_mutual_info = 0.5 * np.log(np.array([
        same_time_COV[part1_indices[0], part1_indices[0]]/(time_lagged_COND_COV_PART11+0j),
        same_time_COV[part2_indices[0], part2_indices[0]]/(time_lagged_COND_COV_PART22+0j),
        same_time_COV[part2_indices[0], part2_indices[0]]/(time_lagged_COND_COV_PART12+0j),
        same_time_COV[part1_indices[0], part1_indices[0]]/(time_lagged_COND_COV_PART21+0j)
    ]))

    # Handle negative values as in original
    all_mutual_info = np.real(all_mutual_info)
    all_mutual_info[all_mutual_info < 0] = 0

    # Take minimum as double redundancy
    return np.min(np.nan_to_num(all_mutual_info))

def get_phiid_analytical(full_time_lagged_COV, redundancy_func):
#
#     # Syn(X_t;X_t-1) (synergistic capacity of the system)
#     # Un (Vt;Xt'|Xt) (causal decoupling - the top term in the lattice)
#     # Un(Vt;Xt'α|Xt) (downward causation)
#
#     # synergy (only considering the synergy that the sources have, not the target):
#     # {12} --> {1}{2} + {12} --> {1} + {12} --> {2} + {12} --> {12}
#
#     # causal decoupling: {12} --> {12}
#
#     # downward causation:
#     # {12} --> {1}{2} + {12} --> {1} + {12} --> {2}
#
#     # phi =     - {1}{2}-->{1}{2}                            (double-redundancy)
#     #           + {12}-->{12}                                (causal decoupling)
#     #           + {12}-->{1} + {12}-->{2} + {12}-->{1}{2}    (downward causation)
#     #           + {1}{2}-->{12} + {1}-->{12} + {2}-->{12}    (upward causation)
#     #           + {1}-->{2} + {2}-->{1}                      (transfer)
#
#     # synergy = causal decoupling + downward causation + upward causation

    # start Matlab engine 
    # eng = matlab.engine.start_matlab()
        
    # convert to matlab-compatible data type *double*
    full_time_lagged_COV = matlab.double(full_time_lagged_COV.tolist())

    phiid = eng.PhiIDFull_Analytical(full_time_lagged_COV, redundancy_func)

    emergence_capacity_phiid = phiid['str'] + phiid['stx'] + \
        phiid['sty'] + phiid['sts']
    downward_causation_phiid = phiid['str'] + phiid['stx'] + \
        phiid['sty']
    synergy_phiid = emergence_capacity_phiid + phiid['rts'] + \
        phiid['xts'] + phiid['yts']
    transfer_phiid = phiid['xty'] + phiid['ytx']
    phi_phiid = - phiid['rtr'] + synergy_phiid + transfer_phiid
    phiR_phiid = phi_phiid + phiid['rtr']
    
    return (phiid, emergence_capacity_phiid, downward_causation_phiid, \
            synergy_phiid, transfer_phiid, phi_phiid, \
                phiR_phiid)








#%% OUTDATED

def get_time_lagged_cov_continuous(weighted_inv_true_cov, t, s, errvar, initial_covariance):
    time_lagged_covariance = la.expm(-weighted_inv_true_cov * (t+s)) @ initial_covariance + \
        0.5 * errvar ** 2 * la.inv(weighted_inv_true_cov) @ (la.expm(weighted_inv_true_cov * \
            (s - t)) - la.expm(weighted_inv_true_cov * (-t - s)))
    return time_lagged_covariance


def get_time_lagged_cov_discrete(gamma, weighted_inv_true_cov, previous_same_time_cov):
    time_lagged_covariance = (np.eye(2)- gamma *weighted_inv_true_cov) @ previous_same_time_cov
    return time_lagged_covariance

def get_cond_cov_full(cov_past, cov_present, time_lagged_cov_present):
    conditional_covariance_full = cov_past - time_lagged_cov_present.T \
        @ la.pinv(cov_present) @ time_lagged_cov_present
    return conditional_covariance_full


def get_cond_cov_parts(cov_past_parts, cov_present_parts, time_lagged_cov_present_parts):
    conditional_covariance_parts = cov_past_parts - time_lagged_cov_present_parts * \
        np.reciprocal(cov_present_parts) * time_lagged_cov_present_parts
    return conditional_covariance_parts

def get_kl_div(inv_true_cov, mean_field_inv_true_cov, true_means, var_means, same_time_cov):
    kldiv = -np.sum(0.5*(1+np.log(1/mean_field_inv_true_cov))) + \
              0.5*np.log(1*np.linalg.det(inv_true_cov)) + \
              np.sum(0.5*mean_field_inv_true_cov*np.diag(inv_true_cov)) + \
              0.5 * (var_means-true_means) @ inv_true_cov @ (var_means-true_means) + \
              0.5*np.sum(inv_true_cov*same_time_cov)

    return kldiv
              

def get_phi(cov_past, cond_cov_present_full, cov_past_parts11, cond_cov_present_parts11,
            cov_past_parts22, cond_cov_present_parts22):
    phi = (0.5 * np.log(np.linalg.det(cov_past) / ((np.linalg.det(cond_cov_present_full))+0j) /
                        (cov_past_parts11/(cond_cov_present_parts11+0j)) /
                        (cov_past_parts22/(cond_cov_present_parts22+0j)))).real
    return phi

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

def get_cov_discrete(gamma, weighted_inv_true_cov, previous_same_time_cov, noise_cov):
    covariance = (np.eye(2) - gamma *weighted_inv_true_cov) @ previous_same_time_cov \
        @ (np.eye(2) - gamma *weighted_inv_true_cov)  +  gamma**2 * weighted_inv_true_cov \
            @ noise_cov @ weighted_inv_true_cov
    return covariance
