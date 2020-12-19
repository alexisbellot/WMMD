import numpy as np
from random import sample
from scipy.stats import describe
import math
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers
from sklearn.metrics.pairwise import rbf_kernel


def b_test(X,Y, kernel,blocksize):
    """
    B-Test is a fast maximum discrepancy (MMD) kernel two-sample test that has low sample complexity,
    tractable null distribution and is consistent.

    :param kernel: kernel that takes two samples and returns similarity matrix
    :param alpha: significance level
    :param blocksize: length of each block
    :return p-value for the hypothhesis of equal sample distribution
    """

    # of samples
    n = len(X[:, 0])

    # of blocks
    m = floor(n / blocksize)

    hh = 0
    hh_null = 0

    for x in range(blocksize):
        for y in range(x,blocksize):
            id_x = list(range(m * x, m * (x + 1)))
            id_y = list(range(m * y, m * (y + 1)))
            hh += kernel(X[id_x, :], X[id_y, :])
            hh += kernel(Y[id_x, :], Y[id_y, :])
            hh -= kernel(X[id_x, :], Y[id_y, :])
            hh -= kernel(Y[id_x, :], X[id_y, :])

            rid_x1 = sample(list(range(0, blocksize)),m)
            rid_x2 = sample(list(range(0, blocksize)),m)
            rid_y1 = sample(list(range(0, blocksize)),m)
            rid_y2 = sample(list(range(0, blocksize)),m)
            hh_null += kernel(X[rid_x1, :], X[rid_x2, :])
            hh_null += kernel(Y[rid_y1, :], Y[rid_y2, :])
            hh_null -= kernel(X[rid_x1, :], Y[rid_y2, :])
            hh_null -= kernel(Y[rid_y1, :], X[rid_x2, :])

    stat = np.mean(hh)
    null_var = np.cov(hh_null)
    p_value = 1 - stats.norm.cdf(stat,0,np.sqrt(null_var))
    return(p_value)



def kernel_mean_matching(X, Z, kern='rbf', B=1.0, eps=None):
    '''
    An implementation of Kernel Mean Matching
    Referenres:
    1. Gretton, Arthur, et al. "Covariate shift by kernel mean matching." Dataset shift in machine learning 3.4 (2009): 5.
    2. Huang, Jiayuan, et al. "Correcting sample selection bias by unlabeled data." Advances in neural information processing systems. 2006.
    :param X: two dimensional sample from population 1
    :param Z: two dimensional sample from population 2
    :param kern: kernel to be used, linear or rbf
    :param B: upperbound on the solution search space
    :param eps: normalization error
    :return: weight coefficients of training sample
    '''
    nx = X.shape[0]
    nz = Z.shape[0]
    if eps == None:
        eps = B / math.sqrt(nz)
    if kern == 'lin':
        K = np.dot(Z, Z.T)
        kappa = np.sum(np.dot(Z, X.T) * float(nz) / float(nx), axis=1)
    elif kern == 'rbf':
        #K = compute_rbf(Z, Z)
        K = rbf_kernel(Z,gamma=1/2)
        kappa = np.sum(rbf_kernel(Z, X, gamma=1/2), axis=1) * float(nz) / float(nx)
    else:
        raise ValueError('unknown kernel')

    K = matrix(K)
    kappa = matrix(kappa)
    G = matrix(np.r_[np.ones((1, nz)), -np.ones((1, nz)), np.eye(nz), -np.eye(nz)])
    h = matrix(np.r_[nz * (1 + eps), nz * (eps - 1), B * np.ones((nz,)), np.zeros((nz,))])

    sol = solvers.qp(K, -kappa, G, h)
    coef = np.array(sol['x'])
    return coef


def compute_rbf(X, Z, sigma=1.0):
    K = np.zeros((X.shape[0], Z.shape[0]), dtype=float)
    for i, vx in enumerate(X):
        K[i, :] = np.exp(-np.sum((vx - Z) ** 2, axis=1) / (2.0 * sigma))
    return K

def weighted_mmd(X, Y, weights = np.ones(len(X)), gamma = 1):

    X = X.reshape((len(X)), 1)
    Y = Y.reshape((len(Y)), 1)

    K_XX = rbf_kernel(X, gamma=gamma)
    K_YY = rbf_kernel(Y, gamma=gamma)
    K_XY = rbf_kernel(X, Y, gamma=gamma)

    K_XX = np.matmul(np.matmul(np.diag(weights),K_XX),np.diag(weights))
    K_XY = np.matmul(np.diag(weights), K_XY)
    n = K_XX.shape[0]
    m = K_YY.shape[0]

    mmd_squared = (np.sum(K_XX) - np.trace(K_XX)) / (n * (n - 1)) + (np.sum(K_YY) - np.trace(K_YY)) / (
                m * (m - 1)) - 2 * np.sum(K_XY) / (m * n)

    return mmd_squared


X = np.random.normal(0,2,500)
Y = np.random.normal(0,2,500)

def test_KMM():
    x = 11 * np.random.random(200) - 6.0
    y = x ** 2 + 10 * np.random.random(200) - 5
    Z = np.c_[x, y]

    x = 2 * np.random.random(200) - 6.0
    y = x ** 2 + 10 * np.random.random(200) - 5
    X = np.c_[x, y]

    coef = kernel_mean_matching(X, Z, kern='rbf', B=10)

    plt.close()
    plt.figure()
    plt.scatter(Z[:, 0], Z[:, 1], color='black', marker='x')
    plt.scatter(X[:, 0], X[:, 1], color='red')
    plt.scatter(Z[:, 0], Z[:, 1], color='green', s=coef * 10, alpha=0.5)

    np.sum(coef > 1e-2)