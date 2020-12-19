from __future__ import print_function

from builtins import object

import autograd.numpy as np
import time
import pandas as pd
import os
import sys
import math
from collections import defaultdict
import joblib
import data as data
import numpy as np
import kernel_utils
import test_utils as tst
import random
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV



def performance_comparisons(methods,num_runs,param_name,params,alpha=0.01,output='p_value',
                            size=300, mu = 0.5, var=1, dx=20, dy=20,prop=1):
    '''
    Code to compute full performance comparison runs
    
    methods: name of methods to be tested
    num_runs: number of run results are averaged over
    param: parameter vector to iterate over
    proportion: proportion of confounding variables observed
    alpha: significance level
    '''
    
    type_I_error = defaultdict(int)
    type_II_error = defaultdict(int)

    for param in params:
        
        # define which parameter to iterate over
        if param_name == 'mu':
            mu = param
        if param_name == 'var':
            var = param
        if param_name == 'dx':
            dx = param
        if param_name == 'dy':
            dy = param
        if param_name == 'prop':
            prop = param
        if param_name == 'size':
            size = param
                
        for n_run in range(num_runs):
            np.random.seed(n_run)
        
            # Create a null and aternative version of the dataset.
            x1_n, x2_n, y1_n, y2_n = data.generate_samples_random(size=size, mu = mu, var=var, dx=dx, dy=dy, 
                                                             noise ="gaussian",f1='linear', f2='linear')
            x1_a, x2_a, y1_a, y2_a = data.generate_samples_random(size=size, mu = 0, var=1, dx=dx, dy=dy, 
                                                             noise = "gaussian", f1='linear', f2='cos')
            # number of variables to be retained
            l = math.ceil(prop*x1_n.shape[1])
            # redefine confounding variables
            x1_n = x1_n[:,:l]; x2_n = x2_n[:,:l]; x1_a = x1_a[:,:l]; x2_a = x2_a[:,:l]
            
            if n_run % 100 == 0:
                print('=' * 70)
                print('Sample output for parameter:', param)
                print('=' * 70)

            # Run the tests on both data sets and compute type I and II errors.
            for method in methods:
                method_name = method.__name__
                key = 'method: {}; num exp: {}; param value: {} '.format(method_name, n_run, param)
                key2 = 'method: {}; param value: {} '.format(method_name, param)
                tic = time.time()
                pval_n = method(x1_n, x2_n, y1_n, y2_n,output=output)
                pval_a = method(x1_a, x2_a, y1_a, y2_a,output=output)
                toc = (time.time() - tic) / 2.

                type_I_error[key2] += int(pval_n < alpha) / num_runs
                type_II_error[key2] += int(pval_a > alpha) / num_runs

                if n_run % 100 == 0:
                    print('{}: time={:.2}s, p_null={:.4}, p_alt={:.4}.'.format( method_name, toc, pval_n, pval_a))
                    
    return type_I_error, type_II_error

def stat_comparisons(methods,param_name,params,size=300, mu = 0, var=1, dx=20, dy=20,output='stat'):
    '''
    Code to compute full performance comparison runs
    
    methods: name of methods to be tested
    param: parameter vector to iterate over
    '''
    
    stat_values = defaultdict(int)

    for param in params:
                
        # define which parameter to iterate over
        if param_name == 'mu':
            mu = param
        if param_name == 'var':
            var = param
        if param_name == 'dx':
            dx = param
        if param_name == 'dy':
            dy = param
        if param_name == 'prop':
            prop = param
        if param_name == 'size':
            size = param
            
        # Create a null and aternative version of the dataset.
        x1, x2, y1, y2 = data.generate_samples_random(size=size, mu = mu, var=var, dx=dx, dy=dy, 
                                                      noise ="gaussian",f1='linear', f2='linear',
                                                      seed = 1)        

            # Run the tests on both data sets and compute type I and II errors.
        for method in methods:
            method_name = method.__name__
            key = 'method: {}; param value: {} '.format(method_name, param)
            tic = time.time()
            stat = method(x1, x2, y1, y2,output=output)
            toc = (time.time() - tic) / 2.

            stat_values[key] = stat

    return stat_values

def stat_comparisons_with_error(methods,param_name,params,size=300, mu = 0, var=1, dx=20, dy=20,output='stat'):
    '''
    Code to compute full performance comparison runs
    
    methods: name of methods to be tested
    param: parameter vector to iterate over
    '''
    
    stat_values = defaultdict(int)

    for param in params:
                
        # define which parameter to iterate over
        if param_name == 'mu':
            mu = param
        if param_name == 'var':
            var = param
        if param_name == 'dx':
            dx = param
        if param_name == 'dy':
            dy = param
        if param_name == 'prop':
            prop = param
        if param_name == 'size':
            size = param
            
        # Create a null and aternative version of the dataset.
        x1, x2, y1, y2 = data.generate_samples_random(size=size, mu = mu, var=var, dx=dx, dy=dy, 
                                                      noise ="gaussian",f1='linear', f2='linear')        

            # Run the tests on both data sets and compute type I and II errors.
        for method in methods:
            method_name = method.__name__
            key = 'method: {}; param value: {} '.format(method_name, param)
            tic = time.time()
            stat = method(x1, x2, y1, y2,output=output)
            toc = (time.time() - tic) / 2.

            stat_values[key] = stat
            
        key = 'KMM approx. error; param value: {} '.format(param)
        wtst_data = data.WTSTData(x1, x2, y1, y2)
        x1x2 = wtst_data.stack_x1x2()
        sig2x = meddistance(x1x2, subsample=1000)
        kx = kernel_utils.KGauss(sig2x)
        mmd_test = tst.WQuadMMDTest(kx, kx, n_permute=200, alpha=0.01)
        error = mmd_test.print_objective_KMM(x1, x2, kx, B=5)
        
        stat_values[key] = error
        
    return stat_values





def constrain(val, min_val, max_val):
    return min(max_val, max(min_val, val))


def dist_matrix(X, Y):
    """
    Construct a pairwise Euclidean distance matrix of size X.shape[0] x Y.shape[0]
    """
    sx = np.sum(X ** 2, 1)
    sy = np.sum(Y ** 2, 1)
    D2 = sx[:, np.newaxis] - 2.0 * np.dot(X, Y.T) + sy[np.newaxis, :]
    # to prevent numerical errors from taking sqrt of negative numbers
    D2[D2 < 0] = 0
    D = np.sqrt(D2)
    return D


def meddistance(X, subsample=None, mean_on_fail=True):
    """
    Compute the median of pairwise distances (not distance squared) of points
    in the matrix.  Useful as a heuristic for setting Gaussian kernel's width.
    Parameters
    ----------
    X : n x d numpy array
    mean_on_fail: True/False. If True, use the mean when the median distance is 0.
        This can happen especially, when the data are discrete e.g., 0/1, and
        there are more slightly more 0 than 1. In this case, the m
    Return
    ------
    median distance
    """
    if subsample is None:
        D = dist_matrix(X, X)
        Itri = np.tril_indices(D.shape[0], -1)
        Tri = D[Itri]
        med = np.median(Tri)
        if med <= 0:
            # use the mean
            return np.mean(Tri)
        return med

    else:
        assert subsample > 0
        rand_state = np.random.get_state()
        np.random.seed(9827)
        n = X.shape[0]
        ind = np.random.choice(n, min(subsample, n), replace=False)
        np.random.set_state(rand_state)
        # recursion just one
        return meddistance(X[ind, :], None, mean_on_fail)


def is_real_num(x):
    """return true if x is a real number"""
    try:
        float(x)
        return not (np.isnan(x) or np.isinf(x))
    except ValueError:
        return False


def tr_te_indices(n, tr_proportion, seed=9282):
    """Get two logical vectors for indexing train/test points.
    Return (tr_ind, te_ind)
    """
    rand_state = np.random.get_state()
    np.random.seed(seed)

    Itr = np.zeros(n, dtype=bool)
    tr_ind = np.random.choice(n, int(tr_proportion * n), replace=False)
    Itr[tr_ind] = True
    Ite = np.logical_not(Itr)

    np.random.set_state(rand_state)
    return (Itr, Ite)


def subsample_ind(n, k, seed=28):
    """
    Return a list of indices to choose k out of n without replacement
    """
    rand_state = np.random.get_state()
    np.random.seed(seed)

    ind = np.random.choice(n, k, replace=False)
    np.random.set_state(rand_state)
    return ind


def fit_gaussian_draw(X, J, seed=28, reg=1e-7, eig_pow=1.0):
    """
    Fit a multivariate normal to the data X (n x d) and draw J points
    from the fit.
    - reg: regularizer to use with the covariance matrix
    - eig_pow: raise eigenvalues of the covariance matrix to this power to construct
        a new covariance matrix before drawing samples. Useful to shrink the spread
        of the variance.
    """
    with NumpySeedContext(seed=seed):
        d = X.shape[1]
        mean_x = np.mean(X, 0)
        cov_x = np.cov(X.T)
        if d == 1:
            cov_x = np.array([[cov_x]])
        [evals, evecs] = np.linalg.eig(cov_x)
        evals = np.maximum(0, np.real(evals))
        assert np.all(np.isfinite(evals))
        evecs = np.real(evecs)
        shrunk_cov = evecs.dot(np.diag(evals ** eig_pow)).dot(evecs.T) + reg * np.eye(d)
        V = np.random.multivariate_normal(mean_x, shrunk_cov, J)
    return V

def propensity_score(x1,x2):
    """
    Computes the propensity score given to sets of covariates.
    """
    n1 = len(x1)
    n2 = len(x2)
    y = np.concatenate((np.ones(n1),np.zeros(n2)))
    x = np.vstack((x1,x2))
        
    shuffle_ind = random.sample(range(n1+n2), len(range(n1+n2)))
    x = x[shuffle_ind]; y = y[shuffle_ind]  
    
    # classifier to estimate the propensity score
    cls = LogisticRegression(random_state=42)

    # calibration of the classifier
    cls = CalibratedClassifierCV(cls)

    cls.fit(x, y)
    propensity = pd.DataFrame(cls.predict_proba(x))
    
    return propensity[1].values

def stratify_propensity(propensity_score,x1,B):
    '''
    Outputs B mutually exclusive (equal sized) subgroups stratified based on the propensity score.
    B: number of subsets
    propensity_score = propensity values of the combined sample (x1,x2)
    '''
    ind_1 = np.argsort(propensity_score[:len(x1[:,0])])
    ind_2 = np.argsort(propensity_score[len(x1[:,0]):])
    split_list_x1 = np.array_split(ind_1, B)
    split_list_x2 = np.array_split(ind_2, B)
    
    return split_list_x1, split_list_x2