# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 11:11:31 2015

@author: yinchengng
"""
from copula_helper import *
from multiprocessing import pool
#import numdifftools as nd
import sys

def log_sigmoid(w, y, x_vec, derivatives=False):
    s_term = sigmoid_term(w, y, x_vec)
    if(derivatives):
        return {'eval': -1. * logsumexp(np.array([0., -1. * y * np.dot(w, x_vec)])), 'gradient': y * x_vec * (1. - s_term), 'hessian': -1. * np.outer(x_vec, x_vec) * (1. - s_term) * (s_term)}
    else:
        return -1. * logsumexp(np.array([0., -1. * y * np.dot(w, x_vec)]))

def get_a_coeff(corr):
    eps = 1.e-16
    np.random.seed(1)
    corr_mat = np.array([[1.-np.random.randn(1)[0] * eps, corr], [corr, 1.-np.random.randn(1)[0] * eps]])
    a_mat = np.linalg.inv(corr_mat) - np.eye(2)
#    print corr_mat
    return (a_mat[0, 0], a_mat[0, 1])

def compute_g(w_i_param_dict, w_j_param_dict, corr):
    eps = 1.e-320
    w_i_param_dict['sigma_vec'] = w_i_param_dict['sigma_vec'] + eps
    w_j_param_dict['sigma_vec'] = w_j_param_dict['sigma_vec'] + eps
    (a11, a12) = get_a_coeff(corr)
    # i_term
    w_i_vec = w_i_param_dict['mu_vec']
    m = len(w_i_vec)
    w_i_vec = np.tile(w_i_vec, (m, 1)).reshape(1, m * m, order = 'C')[0]
    # j_term
    w_j_vec = w_j_param_dict['mu_vec']
    l = len(w_j_vec)
    w_j_vec = np.tile(w_j_vec, (l, 1)).reshape(1, l * l, order = 'F')[0]

    w_i_term = np.array(map(lambda w_i: mog_cdf(w_i, w_i_param_dict['k_vec'], w_i_param_dict['mu_vec'], w_i_param_dict['sigma_vec']), w_i_vec))
    w_i_term = norminvcdf(w_i_term)

    w_j_term = np.array(map(lambda w_j: mog_cdf(w_j, w_j_param_dict['k_vec'], w_j_param_dict['mu_vec'], w_j_param_dict['sigma_vec']), w_j_vec))
    w_j_term = norminvcdf(w_j_term)
    output = -0.5 * (a11 * (w_i_term ** 2 + w_j_term ** 2) + 2. * a12 * w_i_term * w_j_term)
    if(np.isnan(np.sum(output))):
        print 'NaN in output!'
    return output

# This function returns a 2 x (MxL) array, with each column corresponding to gradient of g_ml(w_i, w_j)
# To index a particular gradient vector, one can just use syntax output[i]
def compute_g_gradients(w_i_param_dict, w_j_param_dict, corr):
    eps = 1.e-320
    w_i_param_dict['sigma_vec'] = w_i_param_dict['sigma_vec'] + eps
    w_j_param_dict['sigma_vec'] = w_j_param_dict['sigma_vec'] + eps

    (a11, a12) = get_a_coeff(corr)
    # i_term
    w_i_vec = w_i_param_dict['mu_vec']
    m = len(w_i_vec)
    w_i_vec = np.tile(w_i_vec, (m, 1)).reshape(1, m * m, order = 'C')[0]
    # j_term
    w_j_vec = w_j_param_dict['mu_vec']
    l = len(w_j_vec)
    w_j_vec = np.tile(w_j_vec, (l, 1)).reshape(1, l * l, order = 'F')[0]
    w_i_term = np.array(map(lambda w_i: mog_cdf(w_i, w_i_param_dict['k_vec'], w_i_param_dict['mu_vec'], w_i_param_dict['sigma_vec']), w_i_vec))
    w_i_term = norminvcdf(w_i_term)
    w_j_term = np.array(map(lambda w_j: mog_cdf(w_j, w_j_param_dict['k_vec'], w_j_param_dict['mu_vec'], w_j_param_dict['sigma_vec']), w_j_vec))
    w_j_term = norminvcdf(w_j_term)
    
    i_mog_gaussian_ratio = np.array(map(lambda w_i, w_term: mog_normal_pdf_ratio(w_i, w_i_param_dict['k_vec'], w_i_param_dict['mu_vec'], w_i_param_dict['sigma_vec'], w_term), w_i_vec, w_i_term))
    j_mog_gaussian_ratio = np.array(map(lambda w_j, w_term: mog_normal_pdf_ratio(w_j, w_j_param_dict['k_vec'], w_j_param_dict['mu_vec'], w_j_param_dict['sigma_vec'], w_term), w_j_vec, w_j_term))
    w_i_grad_factor = -1. * (a11 * w_i_term + a12 * w_j_term) * i_mog_gaussian_ratio 
    w_j_grad_factor = -1. * (a11 * w_j_term + a12 * w_i_term) * j_mog_gaussian_ratio 
    return np.transpose(np.array([w_i_grad_factor, w_j_grad_factor]))

def compute_g_hessians(w_i_param_dict, w_j_param_dict, corr):
    eps = 1.e-320
    w_i_param_dict['sigma_vec'] = w_i_param_dict['sigma_vec'] + eps
    w_j_param_dict['sigma_vec'] = w_j_param_dict['sigma_vec'] + eps
    (a11, a12) = get_a_coeff(corr)
    # i_term
    w_i_vec = w_i_param_dict['mu_vec']
    m = len(w_i_vec)
    w_i_vec = np.tile(w_i_vec, (m, 1)).reshape(1, m * m, order = 'C')[0]
    # j_term
    w_j_vec = w_j_param_dict['mu_vec']
    l = len(w_j_vec)
    w_j_vec = np.tile(w_j_vec, (l, 1)).reshape(1, l * l, order = 'F')[0]
    
    w_i_term = np.array(map(lambda w_i: mog_cdf(w_i, w_i_param_dict['k_vec'], w_i_param_dict['mu_vec'], w_i_param_dict['sigma_vec']), w_i_vec))
    w_i_term = norminvcdf(w_i_term)
    w_j_term = np.array(map(lambda w_j: mog_cdf(w_j, w_j_param_dict['k_vec'], w_j_param_dict['mu_vec'], w_j_param_dict['sigma_vec']), w_j_vec))
    w_j_term = norminvcdf(w_j_term)
    
    i_mog_gaussian_ratio = np.array(map(lambda w_i, w_term: mog_normal_pdf_ratio(w_i, w_i_param_dict['k_vec'], w_i_param_dict['mu_vec'], w_i_param_dict['sigma_vec'], w_term), w_i_vec, w_i_term))
    j_mog_gaussian_ratio = np.array(map(lambda w_j, w_term: mog_normal_pdf_ratio(w_j, w_j_param_dict['k_vec'], w_j_param_dict['mu_vec'], w_j_param_dict['sigma_vec'], w_term), w_j_vec, w_j_term))

    i_mog_prime_gaussian_ratio = np.array(map(lambda w_i, w_term: mog_derivative_normal_pdf_ratio(w_i, w_i_param_dict['k_vec'], w_i_param_dict['mu_vec'], w_i_param_dict['sigma_vec'], w_term), w_i_vec, w_i_term))
    j_mog_prime_gaussian_ratio = np.array(map(lambda w_j, w_term: mog_derivative_normal_pdf_ratio(w_j, w_j_param_dict['k_vec'], w_j_param_dict['mu_vec'], w_j_param_dict['sigma_vec'], w_term), w_j_vec, w_j_term))
    
    ij_hess = - a12 * i_mog_gaussian_ratio * j_mog_gaussian_ratio
    i_hess_factor = a11 * i_mog_gaussian_ratio ** 2
    i_hess_factor += (a11 * w_i_term + a12 * w_j_term) * i_mog_prime_gaussian_ratio * (1. + w_i_term * i_mog_prime_gaussian_ratio)
    j_hess_factor = a11 * j_mog_gaussian_ratio ** 2
    j_hess_factor += (a11 * w_j_term + a12 * w_i_term) * j_mog_prime_gaussian_ratio * (1. + w_j_term * j_mog_prime_gaussian_ratio)

    # i_term
    inv_var_i_vec = (1./w_i_param_dict['sigma_vec']**2)
    m = len(inv_var_i_vec)
    inv_var_i_vec = np.tile(inv_var_i_vec, (m, 1)).reshape(1, m * m, order = 'C')[0]
    # j_term
    inv_var_j_vec = (1./w_j_param_dict['sigma_vec']**2)
    l = len(inv_var_j_vec)
    inv_var_j_vec = np.tile(inv_var_j_vec, (l, 1)).reshape(1, l * l, order = 'F')[0]
    enum_i_hess = - (inv_var_i_vec + i_hess_factor)
    enum_j_hess = - (inv_var_j_vec + j_hess_factor)
    enum_ij_hess = ij_hess
    if np.any(np.isnan(enum_i_hess)):
        print 'NaN in enum_i_hess'
    if np.any(np.isnan(enum_j_hess)):
        print 'NaN in enum_j_hess'
    if np.any(np.isnan(enum_ij_hess)):
        print 'NaN in enum_ij_hess'
    build_hessian_mat_fn = lambda i_term, j_term, ij_term: np.array([[i_term, ij_term], [ij_term, j_term]])
    output = np.array(map(build_hessian_mat_fn, enum_i_hess, enum_j_hess, enum_ij_hess))
    return output

def compute_w_second_moment_pair(w_i_param_dict, w_j_param_dict, corr):
    eps_mat = 1.e-15 * np.array([[0.01, 0.], [0., 0.02]])
    g = compute_g(w_i_param_dict, w_j_param_dict, corr)
    grad_g = compute_g_gradients(w_i_param_dict, w_j_param_dict, corr)
    hess_g = compute_g_hessians(w_i_param_dict, w_j_param_dict, corr)
    #hess_g = compute_g_hessians_new(w_i_param_dict, w_j_param_dict, corr)
    hess_g = np.array(map(lambda mat: mat - eps_mat, hess_g))
    try:
        inv_hess_g = np.linalg.inv(hess_g)
    except:
        print 'Error in hessian matrix!'
        print str(hess_g)
    inv_hess_g_det = np.linalg.det(inv_hess_g)
    approx_gaussian_mean_vec = -1. * np.array(map(lambda mat, vec: np.dot(mat, vec), inv_hess_g, grad_g))
    dvd_factor_vec = -0.5 * np.array(map(lambda mat, vec: np.dot(vec, np.dot(mat, vec)), inv_hess_g, grad_g))
    # i_term
    k_i_vec = w_i_param_dict['k_vec']
    mu_i_vec = w_i_param_dict['mu_vec']
    sigma_i_vec = w_i_param_dict['sigma_vec']
    m = len(k_i_vec)
    k_i_vec = np.tile(k_i_vec, (m, 1)).reshape(1, m * m, order = 'C')[0]
    mu_i_vec = np.tile(mu_i_vec, (m, 1)).reshape(1, m * m, order = 'C')[0]
    sigma_i_vec = np.tile(sigma_i_vec, (m, 1)).reshape(1, m * m, order = 'C')[0]
    # j_term
    k_j_vec = w_j_param_dict['k_vec']
    mu_j_vec = w_j_param_dict['mu_vec']
    sigma_j_vec = w_j_param_dict['sigma_vec']
    l = len(k_j_vec)
    k_j_vec = np.tile(k_j_vec, (l, 1)).reshape(1, l * l, order = 'F')[0]
    mu_j_vec = np.tile(mu_j_vec, (l, 1)).reshape(1, l * l, order = 'F')[0]
    sigma_j_vec = np.tile(sigma_j_vec, (l, 1)).reshape(1, l * l, order = 'F')[0]
    factor_vec = mu_i_vec * mu_j_vec
    factor_vec += mu_i_vec * approx_gaussian_mean_vec[:,1]
    factor_vec += mu_j_vec * approx_gaussian_mean_vec[:,0]
    factor_vec -= inv_hess_g[:,0,1]
    factor_vec *= 2. * np.pi * np.sqrt(inv_hess_g_det)
    factor_vec *= np.exp(dvd_factor_vec)
    factor_vec *= np.exp(g)
    output = np.sum(k_i_vec * k_j_vec * factor_vec/(sigma_i_vec * sigma_j_vec))/(2. * np.pi * np.sqrt(1. - corr ** 2)) 
#    if(np.isnan(output)):
#        print 'NaN second moment!'
    return output

def compute_w_second_moment(w_marginal_param_list, copula_corr_mat):
    output_mat = np.zeros(np.shape(copula_corr_mat))
    D = len(output_mat)
    for row in np.arange(0, D):
        for col in np.arange(row+1, D):
            output_mat[row, col] = compute_w_second_moment_pair(w_marginal_param_list[row], w_marginal_param_list[col], copula_corr_mat[row, col])
    output_mat += np.transpose(output_mat)
    for i in np.arange(0, D):
        output_mat[i, i] = np.dot(w_marginal_param_list[i]['k_vec'], w_marginal_param_list[i]['sigma_vec'] ** 2 + w_marginal_param_list[i]['mu_vec'] ** 2)
    return output_mat

def taylor_series_loglikelihood_factor(y, x_vec, w_marginal_param_list, copula_corr_mat):
    w_mean_vec = np.array(map(lambda param_dict: np.dot(param_dict['k_vec'], param_dict['mu_vec']), w_marginal_param_list))
    logsigmoid = log_sigmoid(w_mean_vec, y, x_vec, derivatives=True)
    const_term = logsigmoid['eval']
    grad = logsigmoid['gradient']
    hess = logsigmoid['hessian']
    second_moment = compute_w_second_moment(w_marginal_param_list, copula_corr_mat)
    output = const_term + 0.5 * np.trace(np.dot(hess, second_moment)) - 0.5 * np.dot(np.dot(w_mean_vec, hess), w_mean_vec)
#    if(np.isnan(output)):
#        print 'NaN in output!'
    return output

def log_gaussian_prior_factor(w_marginal_param_list, prior_mu_vec, prior_sigma):
    D = len(prior_mu_vec)
    w_expected_vec = np.array([])
    w_sq_expected_vec = np.array([])
    for param_dict in w_marginal_param_list:
        k_vec = param_dict['k_vec']
        mu_vec = param_dict['mu_vec']
        sigma_vec = param_dict['sigma_vec']
        w_expected = np.dot(k_vec, mu_vec)
        w_sq_expected = np.dot(k_vec, (mu_vec ** 2 + sigma_vec ** 2))
        w_expected_vec = np.append(w_expected_vec, w_expected)
        w_sq_expected_vec = np.append(w_sq_expected_vec, w_sq_expected)
    term_a = -0.5 * D * np.log(2.0 * np.pi * (prior_sigma**2))
    term_b = -0.5 * ((1/prior_sigma) ** 2) * (np.sum(w_sq_expected_vec) - 2.0 * np.dot(prior_mu_vec, w_expected_vec) + np.dot(prior_mu_vec, prior_mu_vec))
    return term_a + term_b

def sum_of_factors(y_vec, x_mat, w_marginal_param_list, copula_corr_mat, prior_mu_vec, prior_sigma):
    factor_sum = log_gaussian_prior_factor(w_marginal_param_list, prior_mu_vec, prior_sigma)
    batch_taylor = lambda y, x: taylor_series_loglikelihood_factor(y, x, w_marginal_param_list, copula_corr_mat)
    taylor_factors_vec = np.array(map(batch_taylor, y_vec, x_mat))
#    if(np.isnan(np.sum(taylor_factors_vec))):
#        print 'Nan in sum of factors!'
    return factor_sum + np.sum(taylor_factors_vec)

def mog_entropy_lb(k_vec, mu_vec, sigma_vec):
    eps = 1e-300
    output = 0.0
    for j in np.arange(0, len(k_vec)):
        log_factor_vec = np.array([])
        if(k_vec[j]==0.):
            continue
        for i in np.arange(0, len(k_vec)):
            tmp = k_vec[i] * norm.pdf(mu_vec[i], loc = mu_vec[j], scale = np.sqrt(sigma_vec[i] ** 2 + sigma_vec[j]**2)+eps)
            if(np.isnan(tmp)):
                print 'tmp is nan!'
                print str(k_vec[i]), str(mu_vec[i]), str(mu_vec[j]), str(sigma_vec[i]), str(sigma_vec[j])
            log_factor_vec = np.append(log_factor_vec, tmp)
        output += k_vec[j] *np.log(np.sum(log_factor_vec))
        if(np.isnan(output)):
            print 'Output is nan!'
    return -1.0 * output

def mog_entropy_lb_sum(w_marginal_param_list):
    return np.sum(np.array(map(lambda param_dict: mog_entropy_lb(param_dict['k_vec'], param_dict['mu_vec'], param_dict['sigma_vec']), w_marginal_param_list)))

def obj_fn_maximise(y_vec, x_mat, w_marginal_param_list, copula_corr_mat, prior_mu_vec, prior_sigma):
#    sys.stdout.write('.')
    #copula_corr_mat = compute_copula_corr_mat(copula_corr_var_param_vec)
    term_a = 0.5 * np.linalg.slogdet(copula_corr_mat)[1]
    term_b = mog_entropy_lb_sum(w_marginal_param_list)
    term_c = sum_of_factors(y_vec, x_mat, w_marginal_param_list, copula_corr_mat, prior_mu_vec, prior_sigma)
    output = term_a + term_b + term_c
    if np.isnan(output):
        print str(copula_corr_mat)
        print str(w_marginal_param_list)
        raise RuntimeError('Objective function returned NaN!')
    return output

def create_gaussian_copula_plot(w_marginal_param_list, copula_corr_mat = 0, copula_upper_cholesky_mat = 0, x_lo = -20.0, x_hi = 20.0, y_lo = -20.0, y_hi = 20.0, filename = '', grid_size = 0.5):
    w1, w2 = np.meshgrid(np.arange(x_lo, x_hi+grid_size,grid_size), np.arange(y_lo, y_hi+grid_size,grid_size))
    innerlambda = lambda w1_val, w2_val: gaussian_copula_pdf(np.array([w1_val, w2_val]), w_marginal_param_list, copula_corr_mat, copula_upper_cholesky_mat)
    outerlambda = lambda w1_vec, w2_vec: pool.mapstar((innerlambda, w1_vec, w2_vec))
    density = map(outerlambda, w1, w2)
    # Plot contours
    rcParams['figure.figsize'] = 20,20
    plt.contourf(w1, w2,density, 200)
    rcParams['figure.figsize']= 7,7
    plt.show()
    plt.grid()
    plt.axes().set_aspect('equal')
    if (filename==''):
        return 1
    else:
        plt.savefig(filename+'.png')
        plt.close()
        return 1
