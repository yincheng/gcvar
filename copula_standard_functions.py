# -*- coding: utf-8 -*-
"""
Created on Sun May 24 22:38:31 2015

@author: yinchengng
"""
import numpy as np
import scipy as sci
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.special import expit as sigmoid
from scipy.stats import multivariate_normal
from scipy.misc import logsumexp

def build_b_mat_rapisarda(a_mat):
    (row, col) = np.shape(a_mat)
    b_mat = np.zeros((row, col))
    for j in np.arange(0, col):
        if(j==0):
            b_mat[:,j] = np.ones(row)
            next
        else:
            b_mat[:,j] = b_mat[:,j-1] * np.sin(a_mat[:,j-1])
    b_mat = b_mat * np.cos(a_mat)
    return b_mat

def build_a_mat_rapisarda(a_input_vec):
    n = len(a_input_vec)
    n_mat = np.ceil(n/2.)+1.
    head = 0
    a_mat = np.zeros((n_mat, n_mat))
    row = 1
    while(head<n):
        tail = head + row
        a_mat[row,0:row] = a_input_vec[head:tail]
        if(len(a_mat[row,0:row])!=len(a_input_vec[head:tail])):
            raise Exception('Mismatch in input array dimension!')
        head = tail
        row += 1
    return a_mat[0:row, 0:row]    

def compute_copula_corr_mat_rapisarda(x_input_vec):
    x_to_angle_fn = lambda x: np.pi/2. - np.arctan(x)
    a_vec = np.array(map(x_to_angle_fn, x_input_vec))
    a_mat = build_a_mat_rapisarda(a_vec)
    b_mat = build_b_mat_rapisarda(a_mat)
    return np.dot(b_mat, np.transpose(b_mat))

def compute_copula_corr_mat(corr_var_param_vec):
    return compute_copula_corr_mat_rapisarda(corr_var_param_vec)

def std_normal_derivative(w_tilde):
    return -1.0 * w_tilde * norm.pdf(w_tilde)

def log_mog_cdf(w, k_vec, mu_vec, sigma_vec):
    exp_term = norm.logcdf(w, loc = mu_vec, scale = sigma_vec)
    coefficients = k_vec
    return logsumexp(exp_term, b = coefficients)

def mog_cdf(w, k_vec, mu_vec, sigma_vec):
    return np.exp(log_mog_cdf(w, k_vec, mu_vec, sigma_vec))

def mog_pdf_derivative(w, k_vec, mu_vec, sigma_vec):
    return -1.0 * np.dot(k_vec * (w - mu_vec)/(sigma_vec**2), np.array(map(lambda mu, sigma: norm.pdf(w, mu, sigma), mu_vec, sigma_vec)))

def log_mog_pdf(w, k_vec, mu_vec, sigma_vec):
    coefficients = k_vec/(np.sqrt(2.*np.pi) * sigma_vec)
    exp_term = -0.5 * ((w - mu_vec)/sigma_vec)**2
    return logsumexp(exp_term, b = coefficients)

def mog_pdf(w, k_vec, mu_vec, sigma_vec):
    return np.exp(log_mog_pdf(w, k_vec, mu_vec, sigma_vec))

def mog_inv_cdf_find_bracket(q, k_vec, mu_vec, sigma_vec, startpoint = 0.0, inc = 1.):
    if(np.log(inc)>=50):
        print 'Stuck in mog_inv_cdf_find_bracket' + str(q)
        raw_input()
    qstart = log_mog_cdf(startpoint, k_vec, mu_vec, sigma_vec)
    if(qstart<=q):
        qend = log_mog_cdf(startpoint+inc, k_vec, mu_vec, sigma_vec)
        if(qend>=q):
            return (startpoint, startpoint+inc)
        else:

            return mog_inv_cdf_find_bracket(q, k_vec, mu_vec, sigma_vec, startpoint = startpoint+inc, inc = 10. * inc)  
    else:
        qend = log_mog_cdf(startpoint-inc, k_vec, mu_vec, sigma_vec)
        if(qend<=q):
            return (startpoint-inc, startpoint)
        else:
            return mog_inv_cdf_find_bracket(q, k_vec, mu_vec, sigma_vec, startpoint = startpoint-inc, inc = 10. * inc)  

def mog_inv_cdf(q, k_vec, mu_vec, sigma_vec):
    q = np.log(q)
    (grid_lo, grid_hi) = mog_inv_cdf_find_bracket(q, k_vec, mu_vec, sigma_vec)
    grid_mid = (grid_lo + grid_hi)/2.0
    q_mid = log_mog_cdf(grid_mid, k_vec, mu_vec, sigma_vec)
    eps = 1e-13
    itr = 0
    
    maxiter = 1e6
    while(abs(grid_lo-grid_hi)>eps):
        if(itr>=maxiter):
            print 'WARNING: mog_inv_cdf exceeded max_iter!', abs(grid_lo-grid_hi), grid_lo, grid_hi, q, q_mid
#            print k_vec
#            print mu_vec
#            print sigma_vec
#            plt.plot(np.arange(-1500., 1500., .1), map(lambda x: mog_pdf(x, k_vec, mu_vec, sigma_vec), np.arange(-1500., 1500., .1)))
 #           exit()
            return grid_mid # TODO: Remove this later on
        if(q>q_mid):
            grid_lo = grid_mid
        else:
            grid_hi = grid_mid
        grid_mid = (grid_lo + grid_hi)/2.0
        q_mid = log_mog_cdf(grid_mid, k_vec, mu_vec, sigma_vec)
        itr = itr + 1
    if(np.abs(q - log_mog_cdf(grid_mid, k_vec, mu_vec, sigma_vec))>1e-6):
        print 'ERROR: mog_inv_cdf seems wrong!', q, log_mog_cdf(grid_mid, k_vec, mu_vec, sigma_vec)
        print 'k_vec: ', k_vec
        print 'mu_vec: ', mu_vec
        print 'sigma_vec: ', sigma_vec
        print 'q: ', q
    return np.round(grid_mid, 20)

def log_gaussian_copula_pdf(w_vec, w_marginal_param_list, copula_corr_mat = 0, copula_corr_var_param = 0):
    assert (type(copula_corr_mat) == np.ndarray or type(copula_corr_var_param) == np.ndarray), 'Please provide either copula_corr_mat or copula_upper_cholesky_mat.'
    if type(copula_corr_var_param) == np.ndarray:
        copula_corr_mat = compute_copula_corr_mat(copula_corr_var_param)
    marginal_cdf_vec = np.array([])
    D = len(w_vec)
    logmarginal = 0.
    logunigaussian = 0.
    norminvcdf = lambda q: mog_inv_cdf(q, np.array([1.]), np.array([0.]), np.array([1.]))
    for d in np.arange(0, D):
        k_vec = w_marginal_param_list[d]['k_vec']
        mu_vec = w_marginal_param_list[d]['mu_vec']
        sigma_vec = w_marginal_param_list[d]['sigma_vec']
        logmarginal += np.log(mog_pdf(w_vec[d], k_vec, mu_vec, sigma_vec))
        tmp = mog_cdf(w_vec[d], k_vec, mu_vec, sigma_vec)
        marginal_cdf_vec = np.append(marginal_cdf_vec, norminvcdf(tmp))
        logunigaussian += norm.logpdf(marginal_cdf_vec[-1])
    multivariate_term = multivariate_normal.logpdf(marginal_cdf_vec, mean = np.zeros(D), cov = copula_corr_mat)
    return multivariate_term + logmarginal - logunigaussian

def gaussian_copula_pdf(w_vec, w_marginal_param_list, copula_corr_mat = 0, copula_upper_cholesky_mat = 0):
    return np.exp(log_gaussian_copula_pdf(w_vec, w_marginal_param_list, copula_corr_mat, copula_upper_cholesky_mat))
    
def mog_inv_cdf_derivative(q, k_vec, mu_vec, sigma_vec):
    deriv = np.exp(-1. * log_mog_pdf(mog_inv_cdf(q, k_vec, mu_vec, sigma_vec), k_vec, mu_vec, sigma_vec))
    return deriv

def mog_inv_cdf_derivative2(q, k_vec, mu_vec, sigma_vec):
    denom = np.exp(log_mog_pdf(mog_inv_cdf(q, k_vec, mu_vec, sigma_vec), k_vec, mu_vec, sigma_vec))
    fprime = mog_pdf_derivative(mog_inv_cdf(q, k_vec, mu_vec, sigma_vec), k_vec, mu_vec, sigma_vec)
    return -1.0 * fprime / (denom ** 3)

def sigmoid_term(w, y, x_vec):
    #return sigmoid(y * np.dot(w, x_vec))
    return np.exp(-1. * logsumexp(np.array([0., -1. * y * np.dot(w, x_vec)])))