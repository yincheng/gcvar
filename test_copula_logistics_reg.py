# -*- coding: utf-8 -*-
"""
Created on Sun May 24 23:04:16 2015

@author: yinchengng
"""

from copula_experiments import *
from sklearn import linear_model
import pickle

dataset = 'simulated'
n_train = 20

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

# Generate some posterior samples to be used later on
true_post_samples = sample_true_posterior(train_x, train_y, prior_sigma = 5.0, nsamples=1000)

for k in (1, 5, 10):
    trained_params = two_step_fitting(n = n_train, k = k, nexperiment = 1, x_arg = train_x, y_arg = train_y)
    copula_post_samples = simulate_copula_posterior(trained_params[0][0], trained_params[0][1], n=10000)
    train_y_predicted = compute_predicted_label(train_x, train_y, trained_params[0][0], trained_params[0][1], copula_samples=copula_post_samples)
    test_y_predicted = compute_predicted_label(test_x, test_y, trained_params[0][0], trained_params[0][1], copula_samples=copula_post_samples)
    copula_train_err = compute_zerone_loss_metric(train_y_predicted, train_y)
    copula_test_err = compute_zerone_loss_metric(test_y_predicted, test_y)
    log_gcp_pdf_batch = lambda w_vec: log_gaussian_copula_pdf(w_vec, trained_params[0][1], copula_corr_mat = trained_params[0][0])
    lnq_expectation = np.average(np.array(map(log_gcp_pdf_batch, true_post_samples)))
    filename = 'dataset_'+dataset+'-copula_k_'+str(k)+'-ntrain_'+str(n_train)+'.pickle'
    with open(filename, 'w') as f:
        pickle.dump([trained_params, train_y_predicted, test_y_predicted, copula_train_err, copula_test_err], f)
    print 'Copula posterior with k = '+str(k)
    print '-----------------------------'
    print 'Training error         : '+str(copula_train_err)
    print 'Testing error          : '+str(copula_test_err)
    print 'Empirical Cross-entropy: '+str(-lnq_expectation)

(normal_mu, normal_cov) = laplace_posterior(train_x, train_y)
normal_post_samples = multivariate_normal.rvs(mean = normal_mu, cov = normal_cov, size = 10000)
get_label_fn = lambda x_vec: np.average(sigmoid(np.dot(normal_post_samples, x_vec)))>0.5
train_y_predicted_normal = np.array(map(get_label_fn, train_x))
train_y_predicted_normal = np.array(map(lambda label: +1 if(label) else -1, train_y_predicted_normal))
test_y_predicted_normal = np.array(map(get_label_fn, test_x))
test_y_predicted_normal = np.array(map(lambda label: +1 if(label) else -1, test_y_predicted_normal))
normal_train_err = compute_zerone_loss_metric(train_y_predicted_normal, train_y)
normal_test_err = compute_zerone_loss_metric(test_y_predicted_normal, test_y)
lnq_expectation = np.average(multivariate_normal.logpdf(true_post_samples, mean = normal_mu, cov = normal_cov))
print 'Laplace posterior'
print '-----------------------------'
print 'Training error         : '+str(normal_train_err)
print 'Testing error          : '+str(normal_test_err)
print 'Empirical Cross-entropy: '+str(-lnq_expectation)

logistic_regression = linear_model.LogisticRegression()
logistic_regression.fit(train_x, train_y)
test_y_predicted_sk = logistic_regression.predict(test_x)
train_y_predicted_sk = logistic_regression.predict(train_x)
sk_train_err = compute_zerone_loss_metric(train_y_predicted_sk, train_y)
sk_test_err = compute_zerone_loss_metric(test_y_predicted_sk, test_y)
print 'Sklearn Logistics Regression'
print '-----------------------------'
print 'Training error: '+str(sk_train_err)
print 'Testing error : '+str(sk_test_err)

