# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 10:44:54 2015

@author: yinchengng
"""

from copula_standard_functions import *
from pylab import rcParams
import pylab as pyl
import pymc

negloglikelihood = lambda x,y,w: np.sum(map(softplus, -1.0 * y * np.dot(x, w)))
logposterior = lambda x,y,w,priorsigma: -1.0 * negloglikelihood(x,y,w) + np.log(multivariate_normal.pdf(w, np.array([0.0,0.0]), (priorsigma**2)* np.eye(2)))
softplus = lambda x: np.log(1.+np.exp(x))
debugmode = False

def create_data_plots(x, y, priorsigma = 5.0, x_lo = -20.0, x_hi = 20.0, y_lo = -20., y_hi = 20., grid_size = 0.5, filename = ''):
    w1, w2 = np.meshgrid(np.arange(x_lo, x_hi+grid_size,grid_size), np.arange(y_lo, y_hi+grid_size,grid_size))
    Z_posterior = map(lambda w1_vec, w2_vec: map(lambda w1_val, w2_val: logposterior(x, y, np.array([w1_val, w2_val]), priorsigma), w1_vec, w2_vec), w1, w2)
    # Plot contours
    rcParams['figure.figsize'] = 20,20
    plt.contourf(w1, w2,Z_posterior, 200)
    plt.axes().set_aspect('equal')
    plt.grid()
    if (filename==''):
        return 1
    else:
        plt.savefig(filename+'.png')
        plt.close()
        return 1
 
def confabulate_logit_data(d=2, N=10, randomseed = 1):
    # Fabricate some testing data
    np.random.seed(randomseed)
    # Generate data and plot
    mu_x = np.random.uniform(low = -1., high = 1., size=d)
    #P_x = np.random.uniform(size=(d,d))
    #sigma_x = P_x * np.transpose(P_x)
    sigma_x = 0.5 * np.ones((d, d)) + 0.5 * np.eye(d)
    x = np.random.multivariate_normal(mean = mu_x, cov = sigma_x, size=N)
    #w_actual = np.random.multivariate_normal(mean = np.zeros(d), cov = np.eye(d), )
    w_actual_mu = np.array([-2, 9., 3., 10.5, -5.5, 7.5, -8.5])
    if(d>len(w_actual_mu)):
        w_actual_mu = np.append(w_actual_mu, np.random.randn(d-len(w_actual_mu)))
    w_actual_mu = w_actual_mu[0:d]
    w_actual_mu = w_actual_mu/np.sqrt(np.dot(w_actual_mu, w_actual_mu))
    #w_actual = np.random.multivariate_normal(mean = w_actual_mu, cov = 0.01*np.eye(d), size = N)
    w_actual = np.tile(w_actual_mu, (N, 1))
    y = sigmoid(np.diag(np.dot(x,np.transpose(w_actual))))
    y = y>np.random.uniform(size=N)
    y = np.array(map(lambda z: +1 if z else -1, y))
    return (x, y)

def unpack_param_array(params, d, fitcorr = True):
    #sigmatransform = lambda xin: 1000. * sigmoid(0.1 * xin) + 0.0001
    sigmatransform = lambda xin: np.exp(xin)
    n_corr_var_param = int(d*(d-1)/2)
    if(fitcorr):
        corr_var_param_vec = params[0:n_corr_var_param]
        marginal_param_vec = params[n_corr_var_param:]
        copula_corr_mat = compute_copula_corr_mat(corr_var_param_vec)
    else:
        copula_corr_mat = np.eye(d)
        marginal_param_vec = params

    k = len(marginal_param_vec)/(2 * d)
    
    w_marginal_param_list = list([])
    for i in np.arange(0, d):
        j = i * 2 * k
        param_dict = {'k_vec': 1./k * np.ones(k),
                      'mu_vec': marginal_param_vec[j:j+k],
                      'sigma_vec': np.array(map(sigmatransform, marginal_param_vec[j+k:j+2*k]))}
        w_marginal_param_list[len(w_marginal_param_list):] = [param_dict]
    return (copula_corr_mat, w_marginal_param_list)

def simulate_copula_posterior(copula_corr_mat, w_marginal_param_list, n=1):
    d = len(copula_corr_mat)
    gaussian_sample_mat = np.random.multivariate_normal(mean=np.zeros(d), cov = copula_corr_mat, size=n)
    uniform_sample_mat = norm.cdf(gaussian_sample_mat)
    output_mat = np.zeros(np.shape(uniform_sample_mat))
    for i in np.arange(0, d):
        k_vec = w_marginal_param_list[i]['k_vec']
        mu_vec = w_marginal_param_list[i]['mu_vec']
        sigma_vec = w_marginal_param_list[i]['sigma_vec']
        mog_inv_cdf_batch = lambda q: np.array(map(lambda q: mog_inv_cdf(q, k_vec, mu_vec, sigma_vec), q))
        output_mat[:, i] = mog_inv_cdf_batch(uniform_sample_mat[:,i])
    return output_mat
    
def compute_logistics_predictive_prob(x_vec, copula_corr_mat, w_marginal_param_list, n=100, copula_samples = None):
    if copula_samples is None:
        posterior_sample_mat = simulate_copula_posterior(copula_corr_mat, w_marginal_param_list, n=n)
    else:
        posterior_sample_mat = copula_samples
    return sigmoid(np.dot(posterior_sample_mat, x_vec))

def create_pymc_model(x_mat, y_vec, prior_sigma = 5.0):
    tau_mc = 1./(prior_sigma ** 2)
    (n, d) = np.shape(x_mat)
    w_mc = list([])
    x_mc = list([])
    for i in np.arange(0, d):
        w_mc[len(w_mc):] = [pymc.Normal('w'+str(i+1)+'_mc', 0.0, tau_mc)]
        x_mc[len(x_mc):] = [pymc.Normal('x'+str(i+1)+'_mc', 0.0, 1.0, value = x_mat[:,i], observed=True)]
    w_mc = np.array(w_mc, dtype=object)
    x_mc = np.array(x_mc, dtype=object)
    @pymc.deterministic
    def pred_mu(w_mc = w_mc, x_mc = x_mc):
        return sigmoid(np.dot(x_mc, np.transpose(w_mc)))
    y_mc = pymc.Bernoulli('y_mc', p = pred_mu, value = np.array(map(lambda val: 0 if val==-1 else +1, y_vec)), observed = True)
    return pymc.Model([pred_mu, pymc.Container(w_mc), pymc.Container(x_mc), y_mc])  

def sample_true_posterior(x_mat, y_vec, prior_sigma = 5.0, nsamples = 10000):
    n_thin = 5
    n_burn = 10000
    model_mc = create_pymc_model(x_mat, y_vec, prior_sigma=prior_sigma)
    mcmc = pymc.MCMC(model_mc)
    print 'Sampling from posterior...'
    mcmc.sample(iter = nsamples * n_thin + n_burn, burn = n_burn, thin = n_thin, verbose=-1)
    print 'Done!'
    (n, d) = np.shape(x_mat)
    for i in np.arange(0, d):
        param_name = 'w'+str(i+1)+'_mc'
        if(i==0):
            sample_mat = np.array([mcmc.trace(param_name)[:]])
        else:
            sample_mat = np.append(sample_mat, np.array([mcmc.trace(param_name)[:]]), 0)
    return np.transpose(sample_mat)
  
def laplace_posterior(x_mat, y_vec, prior_sigma = 5.0):
    (n, d) = np.shape(x_mat)
    model_mc = create_pymc_model(x_mat, y_vec, prior_sigma=prior_sigma)
    N = pymc.NormApprox(model_mc)
    N.fit()
    mu_vec = N.mu[N.containers[0]]
    cov_mat = np.array(N.C[N.containers[0]])
    return (mu_vec, cov_mat)
    
def compute_predicted_label(x_mat, y_vec, copula_corr_mat, w_marginal_param_list, copula_samples = None):
    if(copula_samples is None):
        predictive_batch = lambda x_vec: np.average(compute_logistics_predictive_prob(x_vec, copula_corr_mat, w_marginal_param_list, n=100))
    else:
        predictive_batch = lambda x_vec: np.average(compute_logistics_predictive_prob(x_vec, copula_corr_mat, w_marginal_param_list, copula_samples=copula_samples))
    train_y_predicted = np.array(map(predictive_batch, x_mat))>0.5
    train_y_predicted = np.array(map(lambda label: +1 if(label) else -1, train_y_predicted))
    return train_y_predicted

def compute_zerone_loss_metric(predicted_vec, actual_vec):
    return 100. * np.sum(abs(predicted_vec - actual_vec)/2.)/len(actual_vec)

def sample_mog(param_dict, nsamples = 100000):
    k_vec = param_dict['k_vec']
    mu_vec = param_dict['mu_vec']
    sigma_vec = param_dict['sigma_vec']
    comp_vec = np.array([])
    i = 0
    for mixture in k_vec:
        comp_vec = np.append(comp_vec, i * np.ones(mixture*nsamples))
        i += 1
    if(len(comp_vec)>=nsamples):
        comp_vec = comp_vec[0:nsamples]
    else:
        comp_vec = np.append(comp_vec, (i-1) * np.ones(nsamples - len(comp_vec)))
    sample_mu = np.array(map(lambda i: mu_vec[int(i)], comp_vec))
    sample_sigma = np.array(map(lambda i: sigma_vec[int(i)], comp_vec))
    return sample_sigma * np.random.randn(nsamples) + sample_mu