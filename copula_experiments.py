# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 10:51:49 2015

@author: yinchengng
"""

from Gaussian_copula_free_energy_v3 import *
from scipy.optimize import fmin_bfgs as minimizer
from scipy.optimize import fmin_l_bfgs_b as minimizer_bounded
from scipy.optimize import minimize
import pickle

def obj_fn_min_bfgs_mog(params, y_vec, x_mat, prior_sigma = 5., fitcorr=True):
    (n, d) = np.shape(x_mat)
    param_tmp_list = unpack_param_array(params, d, fitcorr=fitcorr)
    copula_corr_mat = param_tmp_list[0]
    w_marginal_param_list = param_tmp_list[1]
    prior_mu_vec = np.zeros(d)
    try:
        output =-1.0 * obj_fn_maximise(y_vec, x_mat, w_marginal_param_list, copula_corr_mat, prior_mu_vec, prior_sigma)
    except RuntimeError:
        print 'Runtime error in obj_fn_maximise!'
        print 'params_arr:'
        print params
        raise RuntimeError('')
    return output

opt_itr = 0
x = np.array([])
y = np.array([])
csv_filename = ''
csv_str = ''
objfn = lambda x: x+1
#param_per_iter = ()
def callback_opt(params):
    global opt_itr
    global x
    global y
    global csv_str
    global objfn
    opt_itr += 1
    output_str = ''
    output_str = str(opt_itr) + ', '+ str(objfn(params))
    for val in params:
        output_str += ', ' + str(val)
    print output_str
    if(csv_filename == ''):
        return 1
    csv_str += output_str + '\n'
    if(False):
        with open(csv_filename+'.csv','a') as csvfile:
            csvfile.write(csv_str)
        csv_str = ''
      
def run_experiment(n = 5, d = 2, k = 2, nexperiment = 1, param0_array = 0, csv = '', debug = False, fitcorr = True, x_arg = None, y_arg = None, randomseed = 0):
    global opt_itr
    global x
    global y
    global csv_filename
    global debugmode
    global objfn
    debugmode = debug
    csv_filename = csv

    if(x_arg is None or y_arg is None):
        (x, y) = confabulate_logit_data(d=d, N=n, randomseed = 1)
    else:
        x = x_arg
        y = y_arg
    (n, d) = np.shape(x)
    objfn = lambda params: obj_fn_min_bfgs_mog(params, y, x, prior_sigma = 5., fitcorr = fitcorr)
    if(not(fitcorr)):
        outputstr = ''
        outputstr += 'Gaussian Copula Approximate Inference Experiments\n'
        outputstr += ' Number of Data Points: ' + str(n) + '\n'
        outputstr += ' Number of Dimensions: ' + str(d) + '\n'
        outputstr += ' Number of Mixture Components: ' + str(k) + '\n'
        outputstr += ' Number of Experiments to run: ' + str(nexperiment) + '\n'
        outputstr += '\nFits marginal variational parameters only\n'
        print outputstr
    else:
        outputstr = ''
        outputstr += 'Fits all parameters\n'
        print outputstr
    if(csv_filename != ''):
        with open(csv_filename+'.csv', 'a') as csvfile:
            csvfile.write(outputstr + '\n')
    outputstr = ''
    initial_param_type = type(param0_array)
    for exp_itr in np.arange(0, nexperiment, 1):
        np.random.seed(exp_itr + randomseed)
        if(initial_param_type == int):
            param0_array = np.array([])
            if(fitcorr):
                param0_array = np.append(param0_array, np.random.randn(d*(d-1)/2))
            for i in np.arange(0, d):
                param0_array = np.append(param0_array, np.random.randn(k))
                param0_array = np.append(param0_array, np.random.randn(k))
                param0_array = np.append(param0_array, np.random.randn(k))
        else:
            if(fitcorr):
                param0_array = np.append(np.random.randn(d*(d-1)/2), param0_array)
            else:
                param0_array = param0_array
        
        opt_itr = -1
        if(not(fitcorr)):
            outputstr += 'Experiment #, ' + str(exp_itr)
        print outputstr
        if(csv_filename != ''):
            with open(csv_filename+'.csv', 'a') as csvfile:
                csvfile.write(outputstr + '\n')
        outputstr = ''
        
        param_output = np.empty((0, len(param0_array)))
        callback_opt(param0_array)
        #minimizer_output = minimizer_bounded(objfn, param0_array, approx_grad=1, factr = 0. ,epsilon=1e-10, pgtol=1e-20, callback = callback_opt)
        minimizer_output = minimize(objfn, param0_array, method='l-bfgs-b', callback = callback_opt)
        param_output = np.vstack((param_output, minimizer_output.x))
        outputstr = '\n'
        outputstr += str(minimizer_output)
        print outputstr
        if(csv_filename != ''):
            with open(csv_filename+'.csv', 'a') as csvfile:
                csvfile.write(outputstr + '\n')
        outputstr = ''
    outputstr += '===================== End of Experiment ====================='
    print outputstr
    if(csv_filename != ''):
        with open(csv_filename+'.csv', 'a') as csvfile:
            csvfile.write(outputstr + '\n')
    return param_output

def two_step_fitting(n = 5, d = 2, k = 2, nexperiment = 1, csv = '', x_arg = None, y_arg = None):
    if(not(x_arg is None)):
        (n, d) = np.shape(x_arg)
    print 'Fitting marginals...'
    first_step_result_arr = run_experiment(n = n, d = d, k = k, nexperiment = nexperiment, csv = csv, fitcorr = False, x_arg=x_arg, y_arg=y_arg)
    print 'Fitting all parameters...'
    output_param_list = ()
    for i in np.arange(0, nexperiment, 1):
        second_step_result = run_experiment(n = n, d = d, k = k, nexperiment = 1, csv = csv, param0_array=first_step_result_arr[i,:], fitcorr=True, x_arg=x_arg, y_arg=y_arg)
        output_param_list = output_param_list + (unpack_param_array(second_step_result[0,:], d, fitcorr=True), )
    return output_param_list
    
def load_benchmark_data(name = 'titanic'):
    with open('./data/'+name+'.pickle') as f:
        (x_mat, y_vec) = pickle.load(f)
    return (x_mat, y_vec)
