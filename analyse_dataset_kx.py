# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 00:04:27 2015

@author: yinchengng
"""


from copula_experiments import *
from sklearn import linear_model
import pickle

# TODO: Update all these before running
dataset = 'breast_cancer'
n_train = 5
n_true_sample = 500000
n_actual_true_sample = 100000
sample_file_path = '../posterior_samples/'+'true_post_samples-'+dataset+'-n_train_'+str(n_train)+'-n_samples_'+str(n_true_sample)+'.pickle'
param_arr = np.array([ 0.12023302, -0.33145742,  0.20623538, -0.06990782,  0.03603671,
       -0.11781796,  0.08590104, -0.04427313,  0.14534273,  0.04522248,
       -0.01253702,  0.02365346, -0.03093277, -0.00823509,  0.0105534 ,
        0.10147485, -0.04764807,  0.22085441,  0.05904424, -0.07578508,
        0.01337241,  0.34884562, -0.24201861,  1.0926059 ,  0.32391414,
       -0.45554562,  0.122559  , -1.38680264,  0.07883034, -0.04061805,
        0.13315445,  0.04140643, -0.05302034,  0.01026004, -0.07381893,
       -1.74733113,  1.76405235,  3.64472391,  1.47379881,  2.2408932 ,
        0.80375013,  1.56436733,  0.95008842, -0.70256617,  1.39953349,
        0.4105985 ,  0.7732587 ,  1.60123428,  0.76103773, -0.94509234,
        1.59724146,  0.33367433, -0.64286014,  1.60590652,  0.3130677 ,
        2.29715678,  1.50008579,  0.6536186 ,  1.18174074,  1.40639357,
        2.26975462, -0.8693448 ,  1.59906931])

with open('../data/'+dataset+'.pickle', 'r') as f:
    (full_x, full_y) = pickle.load(f)

train_x = full_x[0:n_train,:]
train_y = full_y[0:n_train]
if(dataset == 'simulated'):
    test_x = full_x[(len(full_y) - 500):, :]
    test_y = full_y[(len(full_y) - 500):]
else:
    test_x = full_x[n_train:, :]
    test_y = full_y[n_train:]

with open(sample_file_path, 'r') as f:
        (true_post_samples) = pickle.load(f)

(n, d) = np.shape(train_x)         
k = (len(param_arr) - int(d*(d-1)/2))/(3 * d)
param_list = unpack_param_array(param_arr, d=d, fitcorr=True)

copula_post_samples = simulate_copula_posterior(param_list[0], param_list[1], n=n_actual_true_sample)
hist_bin = np.arange(-20.,20.,0.25)
for j in np.arange(0, d):
    plt.figure(j)
    title = dataset +  ' dim = '+str(j+1) + ' k = '+str(k)
    plt.title(title)
    plt.subplot(211)
    plt.hist(true_post_samples[0][:,j], bins = hist_bin)
    plt.subplot(212)
    plt.hist(copula_post_samples[0:len(copula_post_samples), j], bins=hist_bin)
    plt.savefig(dataset+'-k'+str(k)+'-w'+str(j+1)+'.png')
    plt.close()

print 'Correlation Matrix: '
print str(param_list[0])
print 'Marginals: '
print str(param_list[1])
