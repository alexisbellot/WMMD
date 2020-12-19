

"""Module containing various types of two sample test algorithms 
with the same structure as https://github.com/wittawatj/interpretable-test"""


from __future__ import print_function
from __future__ import division

from builtins import str
from builtins import range
from past.utils import old_div
from builtins import object
from future.utils import with_metaclass

from abc import ABCMeta, abstractmethod
import autograd
import autograd.numpy as np
import general_utils as general_utils
import kernel_utils as kernel_utils
import data as data
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

import matplotlib.pyplot as plt
import scipy
import scipy.stats as stats
import theano
import theano.tensor as tensor
import theano.tensor.nlinalg as nlinalg
import theano.tensor.slinalg as slinalg
from cvxopt import matrix, solvers
import math
from random import sample


class TwoSampleTest(with_metaclass(ABCMeta, object)):
    """Parent class for two sample tests."""

    def __init__(self, alpha=0.01):
        """
        alpha: significance level of the test
        """
        self.alpha = alpha

    @abstractmethod
    def perform_test(self, tst_data):
        """perform the two-sample test and return values computed in a dictionary:
        {alpha: 0.01, pvalue: 0.0002, test_stat: 2.3, h0_rejected: True, ...}
        tst_data: an instance of TSTData
        """
        raise NotImplementedError()

    @abstractmethod
    def compute_stat(self, tst_data):
        """Compute the test statistic"""
        raise NotImplementedError()

# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------

class HotellingT2Test(TwoSampleTest):
    """Two-sample test with Hotelling T-squared statistic.
    Technical details follow "Applied Multivariate Analysis" of Neil H. Timm.
    See page 156.

    """

    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def perform_test(self, tst_data):
        """Perform the two-sample test and return values computed in a dictionary:
        {alpha: 0.01, pvalue: 0.0002, test_stat: 2.3, h0_rejected: True, ...}
        tst_data: an instance of TSTData
        """
        d = tst_data.dim()
        chi2_stat = self.compute_stat(tst_data)
        pvalue = stats.chi2.sf(chi2_stat, d)
        alpha = self.alpha
        results = {'alpha': self.alpha, 'pvalue': pvalue, 'test_stat': chi2_stat,
                   'h0_rejected': pvalue < alpha}
        return results

    def compute_stat(self, tst_data):
        """Compute the test statistic"""
        X, Y = tst_data.xy()
        # if X.shape[0] != Y.shape[0]:
        #    raise ValueError('Require nx = ny for now. Will improve if needed.')
        nx = X.shape[0]
        ny = Y.shape[0]
        mx = np.mean(X, 0)
        my = np.mean(Y, 0)
        mdiff = mx - my
        sx = np.cov(X.T)
        sy = np.cov(Y.T)
        s = old_div(sx, nx) + old_div(sy, ny)
        chi2_stat = np.dot(np.linalg.solve(s, mdiff), mdiff)
        return chi2_stat

    
# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------

class BTest(TwoSampleTest):
    """Block Two-Sample test with MMD^2 statistic.
    """
    def __init__(self, alpha=0.01):
        """
        kernel: an instance of Kernel
        """
        self.alpha = alpha
    
    def perform_test(self, wtst_data):
        """perform the two-sample test and return values computed in a dictionary:
        {alpha: 0.01, pvalue: 0.0002, test_stat: 2.3, h0_rejected: True, ...}
        tst_data: an instance of TSTData
        """
        raise NotImplementedError()
        
    def block_indeces(wtst_data, B=None):
        """ Outputs list of indeces of B mutually exclusive (equal sized) subgroups stratified based on the                   propensity score.
            B = number of blocks
        """
        X1, X2, Y1, Y2 = wtst_data.x1x2y1y2()
        if B == None:
            blocksize = np.sqrt(len(X1[:, 0]))
            # of blocks
            B = np.floor(len(X1[:, 0]) / blocksize)
            
        propensity_scores = general_utils.propensity_score(X1,X2)
        split_list_x1, split_list_x2 = general_utils.stratify_propensity(propensity_scores,X1,B)
        
        return split_list_x1, split_list_x2
    
    @staticmethod
    def compute_stat(wtst_data):
        """ Output the statistic of the test
        """ 
        
        return BTest.Btest(wtst_data,B=None,output='stat')
                      
    @staticmethod
    def compute_pvalue(wtst_data):
        """ Output the p_value of the test
        """ 
        
        return BTest.Btest(wtst_data,B=None,output='p_value')
        
    @staticmethod
    def Btest(wtst_data,B=None,output='stat'):
    
        """
        B-Test is a fast maximum discrepancy (MMD) kernel two-sample test that has low sample complexity,
        tractable null distribution and is consistent.

        :param kernel: kernel that takes two samples and returns similarity matrix
        :param alpha: significance level
        :param B: number of blocks
        :return p-value for the hypothhesis of equal sample distribution
        """
        
        # list of indeces for each block
        split_list_x1, split_list_x2 = BTest.block_indeces(wtst_data=wtst_data, B=B)
        X1, X2, Y1, Y2 = wtst_data.x1x2y1y2()
        out = 0
                      
        for indeces_1, indeces_2 in zip(split_list_x1, split_list_x2):
            
            tst_data = data.TSTData(Y1[indeces_1,:], Y2[indeces_2,:])
            tr, te = tst_data.split_tr_te(tr_proportion=0.5, seed=10)
            xtr, ytr = tr.xy()
            xytr = tr.stack_xy()
            sig2 = general_utils.meddistance(xytr, subsample=1000)
            k = kernel_utils.KGauss(sig2)
    
            # choose the best parameter and perform a test with permutations
            med = general_utils.meddistance(tr.stack_xy(), 1000)
            list_gwidth = np.hstack( ( (med**2) *(2.0**np.linspace(-4, 4, 20) ) ) )
            list_gwidth.sort()

            list_kernels = [kernel_utils.KGauss(gw2) for gw2 in list_gwidth]

            # grid search to choose the best Gaussian width
            besti, powers = QuadMMDTest.grid_search_kernel(tr, list_kernels, alpha=0.01)
            # perform test 
            best_ker = list_kernels[besti]
    
            mmd_test = QuadMMDTest(best_ker, n_permute=200, alpha=0.01)
                      
            if output == 'stat':
                out += mmd_test.compute_stat(te)
    
            if output == 'p_value':
                out += mmd_test.compute_pvalue(te)

        return out / len(split_list_x1)              
    
def run_full_B_test(x1, x2, y1, y2,alpha=0.01,output='p_value'):
    '''
    Runs full WMMD test with all optimization procedures included
    output = desried output, one of 'p_value', 'stat', 'full'
    require same number of instances in both populations
    '''
    wtst_data = data.WTSTData(x1, x2, y1, y2)
    B_test = BTest(alpha=alpha)
                      
    if output == 'stat':
        return B_test.compute_stat(wtst_data)
    if output == 'p_value':
        return B_test.compute_pvalue(wtst_data)
    
# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------


class LinearMMDTest(TwoSampleTest):
    """Two-sample test with linear MMD^2 statistic.
    """

    def __init__(self, kernel, alpha=0.01):
        """
        kernel: an instance of Kernel
        """
        self.kernel = kernel
        self.alpha = alpha

    def perform_test(self, tst_data):
        """perform the two-sample test and return values computed in a dictionary:
        {alpha: 0.01, pvalue: 0.0002, test_stat: 2.3, h0_rejected: True, ...}
        tst_data: an instance of TSTData
        """
        X, Y = tst_data.xy()
        n = X.shape[0]
        stat, snd = LinearMMDTest.two_moments(X, Y, self.kernel)
        # var = snd - stat**2
        var = snd
        pval = stats.norm.sf(stat, loc=0, scale=(2.0 * var / n) ** 0.5)
        results = {'alpha': self.alpha, 'pvalue': pval, 'test_stat': stat,
                   'h0_rejected': pval < self.alpha}
        return results
    
    def compute_pvalue(self, tst_data):
        """perform the two-sample test and return p-values 
        tst_data: an instance of TSTData
        """
        X, Y = tst_data.xy()
        n = X.shape[0]
        stat, snd = LinearMMDTest.two_moments(X, Y, self.kernel)
        # var = snd - stat**2
        var = snd
        pval = stats.norm.sf(stat, loc=0, scale=(2.0 * var / n) ** 0.5)
        
        return pval

    def compute_stat(self, tst_data):
        """Compute unbiased linear mmd estimator."""
        X, Y = tst_data.xy()
        return LinearMMDTest.linear_mmd(X, Y, self.kernel)

    @staticmethod
    def linear_mmd(X, Y, kernel):
        """Compute linear mmd estimator. O(n)"""
        lin_mmd, _ = LinearMMDTest.two_moments(X, Y, kernel)
        return lin_mmd

    @staticmethod
    def two_moments(X, Y, kernel):
        """Compute linear mmd estimator and a linear estimate of
        the uncentred 2nd moment of h(z, z'). Total cost: O(n).
        return: (linear mmd, linear 2nd moment)
        """
        if X.shape[0] != Y.shape[0]:
            raise ValueError('Require sample size of X = size of Y')
        n = X.shape[0]
        if n % 2 == 1:
            # make it even by removing the last row
            X = np.delete(X, -1, axis=0)
            Y = np.delete(Y, -1, axis=0)

        Xodd = X[::2, :]
        Xeven = X[1::2, :]
        assert Xodd.shape[0] == Xeven.shape[0]
        Yodd = Y[::2, :]
        Yeven = Y[1::2, :]
        assert Yodd.shape[0] == Yeven.shape[0]
        # linear mmd. O(n)
        xx = kernel.pair_eval(Xodd, Xeven)
        yy = kernel.pair_eval(Yodd, Yeven)
        xo_ye = kernel.pair_eval(Xodd, Yeven)
        xe_yo = kernel.pair_eval(Xeven, Yodd)
        h = xx + yy - xo_ye - xe_yo
        lin_mmd = np.mean(h)
        """
        Compute a linear-time estimate of the 2nd moment of h = E_z,z' h(z, z')^2.
        Note that MMD = E_z,z' h(z, z').
        Require O(n). Same trick as used in linear MMD to get O(n).
        """
        lin_2nd = np.mean(h ** 2)
        return lin_mmd, lin_2nd

    @staticmethod
    def variance(X, Y, kernel, lin_mmd=None):
        """Compute an estimate of the variance of the linear MMD.
        Require O(n^2). This is the variance under H1.
        """
        if X.shape[0] != Y.shape[0]:
            raise ValueError('Require sample size of X = size of Y')
        n = X.shape[0]
        if lin_mmd is None:
            lin_mmd = LinearMMDTest.linear_mmd(X, Y, kernel)
        # compute uncentred 2nd moment of h(z, z')
        K = kernel.eval(X, X)
        L = kernel.eval(Y, Y)
        KL = kernel.eval(X, Y)
        snd_moment = old_div(np.sum((K + L - KL - KL.T) ** 2), (n * (n - 1)))
        var_mmd = 2.0 * (snd_moment - lin_mmd ** 2)
        return var_mmd

    @staticmethod
    def grid_search_kernel(tst_data, list_kernels, alpha):
        """
        Return from the list the best kernel that maximizes the test power.
        return: (best kernel index, list of test powers)
        """
        X, Y = tst_data.xy()
        n = X.shape[0]
        powers = np.zeros(len(list_kernels))
        for ki, kernel in enumerate(list_kernels):
            lin_mmd, snd_moment = LinearMMDTest.two_moments(X, Y, kernel)
            var_lin_mmd = (snd_moment - lin_mmd ** 2)
            # test threshold from N(0, var)
            thresh = stats.norm.isf(alpha, loc=0, scale=(2.0 * var_lin_mmd / n) ** 0.5)
            power = stats.norm.sf(thresh, loc=lin_mmd, scale=(2.0 * var_lin_mmd / n) ** 0.5)
            # power = lin_mmd/var_lin_mmd
            powers[ki] = power
        best_ind = np.argmax(powers)
        return best_ind, powers

# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------


class ANCOVA(TwoSampleTest):
    """ANCOVA two sample test
    """

    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def perform_test(self, wtst_data):
        """Perform the two-sample test and return values computed in a dictionary:
        {alpha: 0.01, pvalue: 0.0002, test_stat: 2.3, h0_rejected: True, ...}
        tst_data: an instance of TSTData
        """
        residuals1, residuals2 = self.compute_residuals(wtst_data)
        dim = wtst_data.dim_y()
        if dim ==1:
            stat, pvalue = stats.f_oneway(residuals1,residuals2)
        
        else:
            stat = self.compute_stat(wtst_data)
            pvalue = stats.chi2.sf(stat, d)
        
        alpha = self.alpha
        results = {'alpha': self.alpha, 'pvalue': pvalue, 'test_stat': stat,
                   'h0_rejected': pvalue < alpha}
        return results
    
    def compute_residuals(self, wtst_data):
        """Compute the test statistic"""
        x1, x2, y1, y2 = wtst_data.x1x2y1y2()
        
        rf1 = RandomForestRegressor(max_depth=2, random_state=0,n_estimators=100)
        rf1.fit(x1, y1)  
                
        rf2 = RandomForestRegressor(max_depth=2, random_state=0,n_estimators=100)
        rf2.fit(x2, y2)  
        
        if y1.shape[1] == 1:
            residuals1 = y1.flatten() - rf1.predict(x1)
            residuals2 = y2.flatten() - rf2.predict(x2)
        else:
            residuals1 = y1 - rf1.predict(x1)
            residuals2 = y2 - rf2.predict(x2)
            
        return residuals1, residuals2
        
    def compute_stat(self, wtst_data):
        """Compute the test statistic"""
        residuals1, residuals2 = self.compute_residuals(wtst_data)
        
        dim = wtst_data.dim_y()
        if dim == 1:
            stat, pvalue = stats.f_oneway(residuals1,residuals2)
            return stat
        
        else:
            n1 = residuals1.shape[0]
            n2 = residuals2.shape[0]
            m1 = np.mean(residuals1, 0)
            m2 = np.mean(residuals2, 0)
            mdiff = m1 - m2
            s1 = np.cov(residuals1.T)
            s2 = np.cov(residuals2.T)
            s = old_div(s1, n1) + old_div(s2, n2)
            chi2_stat = np.dot(np.linalg.solve(s, mdiff), mdiff)
            return chi2_stat
    
    def compute_pvalue(self, wtst_data):
        """Compute the test statistic"""
        residuals1, residuals2 = self.compute_residuals(wtst_data)
        dim = wtst_data.dim_y()
        if dim == 1:
            stat, pvalue = stats.f_oneway(residuals1,residuals2)
            return pvalue
        
        else:
            stat = self.compute_stat(wtst_data)
            pvalue = stats.chi2.sf(stat, dim)

        return pvalue

def run_full_ANCOVA_test(x1, x2, y1, y2,alpha=0.01,output='p_value'):
    '''
    Runs full WMMD test with all optimization procedures included
    output = desried output, one of 'p_value', 'stat', 'full'
    require same number of instances in both populations
    '''
    wtst_data = data.WTSTData(x1, x2, y1, y2)
    ancova_test = ANCOVA(alpha=alpha)
                      
    if output == 'stat':
        return ancova_test.compute_stat(wtst_data)
    if output == 'p_value':
        return ancova_test.compute_pvalue(wtst_data)
    if output == 'full':
        return ancova_test.perform_test(wtst_data)

# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------


class WQuadMMDTest(TwoSampleTest):
    """
    Weighted MMD test where the null distribution is computed by permutation.
    """

    def __init__(self, kernel, kernel_KMM, n_permute=400, alpha=0.01):
        """
        kernel:     an instance of the Kernel class in 'kernel_utils' to be used for defining a distance between y                     samples 
        kernel_KMM: an instance of the Kernel class in 'kernel_utils' to be used for defining a distance between x                     samples 
        n_permute:  number of times to do permutation
        alpha:      significance level for type I error tolerance
        """
        self.kernel = kernel
        self.kernel_KMM = kernel_KMM
        self.n_permute = n_permute
        self.alpha = alpha

    def perform_test(self, wtst_data):
        """
        Perform the two-sample test and return values computed in a dictionary:
        {alpha: 0.01, pvalue: 0.0002, test_stat: 2.3, h0_rejected: True, ...}
        
        wtst_data: an instance of WTSTData
        """
        d         = wtst_data.dim()
        alpha     = self.alpha
        k         = self.kernel
        kx        = self.kernel_KMM
        repeats   = self.n_permute
        
        # Compute test statistic from data
        mmd2_stat = self.compute_stat(wtst_data)
        
        # Compute null distribution with permutations
        X1, X2, Y1, Y2 = wtst_data.x1x2y1y2()
        list_mmd2 = WQuadMMDTest.permutation_list_mmd2(X1, X2, Y1, Y2, k, kx, repeats)
        
        # approximate p-value with permutations
        pvalue = np.mean(list_mmd2 > mmd2_stat)

        results = {'alpha': self.alpha, 'pvalue': pvalue, 'test_stat': mmd2_stat,
                   'h0_rejected': pvalue < alpha}#, 'list_permuted_mmd2': list_mmd2}
        return results

    def compute_pvalue(self, wtst_data):
        """
        Perform the two-sample test and return p-values 
        
        wtst_data: an instance of WTSTData
        """
        d         = wtst_data.dim()
        alpha     = self.alpha
        k         = self.kernel
        kx        = self.kernel_KMM
        repeats   = self.n_permute
        
        # Compute test statistic from data
        mmd2_stat = self.compute_stat(wtst_data)
        
        # Compute null distribution with permutations
        X1, X2, Y1, Y2 = wtst_data.x1x2y1y2()
        list_mmd2 = WQuadMMDTest.permutation_list_mmd2(X1, X2, Y1, Y2, k, kx, repeats)
        
        # approximate p-value with permutations
        pvalue = np.mean(list_mmd2 > mmd2_stat)
        
        return pvalue

    def compute_stat(self, wtst_data):
        """Compute the test statistic: empirical quadratic MMD^2"""
        X1, X2, Y1, Y2 = wtst_data.x1x2y1y2()
        nx1 = X1.shape[0]
        nx2 = X2.shape[0]

        if nx1 != nx2:
            raise ValueError('nx1 must be the same as nx2')

        k = self.kernel
        kx = self.kernel_KMM
        mmd2, var = WQuadMMDTest.h1_mean_var(X1, X2, Y1, Y2, k, kx, is_var_computed=False)
        return mmd2

    @staticmethod
    def permutation_list_mmd2(X1, X2, Y1, Y2, k, kx, n_permute=400):
        """
        Repeatedly mix, permute X,Y and compute MMD^2. This is intended to be
        used to approximate the null distritubion.
        TODO: This is a naive implementation where the kernel matrix is recomputed
        for each permutation. We might be able to improve this if needed.
        """
        return WQuadMMDTest.permutation_list_mmd2_gram(X1, X2, Y1, Y2, k, kx, n_permute)

    @staticmethod
    def permutation_list_mmd2_gram(X1, X2, Y1, Y2, k, kx, n_permute=400):
        """
        Repeatedly mix, permute X,Y and compute MMD^2. This is intended to be
        used to approximate the null distritubion.
        """
        Y1Y2 = np.vstack((Y1, Y2))
        Ky1y2y1y2 = k.eval(Y1Y2, Y1Y2)

        rand_state = np.random.get_state()
        np.random.seed()

        ny1y2 = Y1Y2.shape[0]
        ny1 = Y1.shape[0]
        ny2 = Y2.shape[0]
        list_mmd2 = np.zeros(n_permute)

        for r in range(n_permute):
            # print r
            ind = np.random.choice(ny1y2, ny1y2, replace=False)
            # divide into new y1, y2
            indy1 = ind[:ny1]
            # print(indy1)
            indy2 = ind[ny1:]
            Ky1 = Ky1y2y1y2[np.ix_(indy1, indy1)]
            # print(Ky1)
            Ky2 = Ky1y2y1y2[np.ix_(indy2, indy2)]
            Ky1y2 = Ky1y2y1y2[np.ix_(indy1, indy2)]
            
            weights, _ = WQuadMMDTest.kernel_mean_matching(X1, X2, kx)
            Ky1 = np.matmul(np.matmul(np.diag(weights[:,0]),Ky1),np.diag(weights[:,0]))
            Ky1y2 = np.matmul(np.diag(weights[:,0]), Ky1y2)
            
            mmd2r, var = WQuadMMDTest.h1_mean_var_gram(Ky1, Ky2, Ky1y2, is_var_computed=False)
            list_mmd2[r] = mmd2r

        np.random.set_state(rand_state)
        return list_mmd2

    @staticmethod
    def kernel_mean_matching(X1, X2, kx, B=10, eps=None):
        '''
        An implementation of Kernel Mean Matching, note that this implementation uses its own kernel parameter
        References:
        1. Gretton, Arthur, et al. "Covariate shift by kernel mean matching." 
        2. Huang, Jiayuan, et al. "Correcting sample selection bias by unlabeled data."
        
        :param X1: two dimensional sample from population 1
        :param X2: two dimensional sample from population 2
        :param kern: kernel to be used, an instance of class Kernel in kernel_utils
        :param B: upperbound on the solution search space 
        :param eps: normalization error
        :return: weight coefficients for instances x1 such that the distribution of weighted x1 matches x2
        '''
        nx1 = X1.shape[0]
        nx2 = X2.shape[0]
        if eps == None:
            eps = B / math.sqrt(nx1)
        K = kx.eval(X1, X1)
        kappa = np.sum(kx.eval(X1, X2), axis=1) * float(nx1) / float(nx2)
        
        K = matrix(K)
        kappa = matrix(kappa)
        G = matrix(np.r_[np.ones((1, nx1)), -np.ones((1, nx1)), np.eye(nx1), -np.eye(nx1)])
        h = matrix(np.r_[nx1 * (1 + eps), nx1 * (eps - 1), B * np.ones((nx1,)), np.zeros((nx1,))])

        solvers.options['show_progress'] = False
        sol = solvers.qp(K, -kappa, G, h)
        coef = np.array(sol['x'])
        objective_value = sol['primal objective'] * 2 / (nx1**2) + np.sum(kx.eval(X2, X2)) / (nx2**2)
        
        return coef, objective_value

    @staticmethod
    def print_objective_KMM(X1, X2, kx, B=5):
        '''
        Print the objective function of KMM to inspect the error in the optimization
        - should be as close to zero as possible
        '''
        _, objective = WQuadMMDTest.kernel_mean_matching(X1, X2, kx, B=B)
        return objective
        
    @staticmethod
    def h1_mean_var(X1, X2, Y1, Y2, k, kx, is_var_computed):
        """
        X1: nxdx numpy array; X2: nxdx numpy array
        Y1: nxdy numpy array; Y1: nxdy numpy array
        k: a Kernel object
        is_var_computed: if True, compute the variance. If False, return None.
        
        returns (MMD^2, var[MMD^2]) under H1
        """

        nx1 = X1.shape[0]
        nx2 = X2.shape[0]

        Ky1 = k.eval(Y1, Y1)
        Ky2 = k.eval(Y2, Y2)
        Ky1y2 = k.eval(Y1, Y2)

        weights, _ = WQuadMMDTest.kernel_mean_matching(X1, X2, kx)
        Ky1 = np.matmul(np.matmul(np.diag(weights[:,0]),Ky1),np.diag(weights[:,0]))
        Ky1y2 = np.matmul(np.diag(weights[:,0]), Ky1y2)
        
        return WQuadMMDTest.h1_mean_var_gram(Ky1, Ky2, Ky1y2, is_var_computed)
    
    @staticmethod
    def h1_mean_var_gram(Ky1, Ky2, Ky1y2, is_var_computed):
        """
        Same as h1_mean_var() but takes in Gram matrices directly.
        """

        ny1 = Ky1.shape[0]
        ny2 = Ky2.shape[0]
        
        y1y1 = old_div((np.sum(Ky1) - np.sum(np.diag(Ky1))), (ny1 * (ny1 - 1)))
        y2y2 = old_div((np.sum(Ky2) - np.sum(np.diag(Ky2))), (ny2 * (ny2 - 1)))
        y1y2 = old_div(np.sum(Ky1y2), (ny1 * ny2))
        mmd2 = y1y1 - 2 * y1y2 + y2y2

        if not is_var_computed:
            return mmd2, None

        # compute the variance
        Ky1d = Ky1 - np.diag(np.diag(Ky1))
        Ky2d = Ky2 - np.diag(np.diag(Ky2))
        m = ny1
        n = ny2
        v = np.zeros(11)

        Ky1d_sum = np.sum(Ky1d)
        Ky2d_sum = np.sum(Ky2d)
        Ky1y2_sum = np.sum(Ky1y2)
        Ky1y22_sum = np.sum(Ky1y2 ** 2)
        Ky1d0_red = np.sum(Ky1d, 1)
        Ky2d0_red = np.sum(Ky2d, 1)
        Ky1y21 = np.sum(Ky1y2, 1)
        Ky2y11 = np.sum(Ky1y2, 0)

        #  varEst = 1/m/(m-1)/(m-2)    * ( sum(Kxd,1)*sum(Kxd,2) - sum(sum(Kxd.^2)))  ...
        v[0] = 1.0 / m / (m - 1) / (m - 2) * (np.dot(Ky1d0_red, Ky1d0_red) - np.sum(Ky1d ** 2))
        #           -  (  1/m/(m-1)   *  sum(sum(Kxd))  )^2 ...
        v[1] = -(1.0 / m / (m - 1) * Ky1d_sum) ** 2
        #           -  2/m/(m-1)/n     *  sum(Kxd,1) * sum(Kxy,2)  ...
        v[2] = -2.0 / m / (m - 1) / n * np.dot(Ky1d0_red, Ky1y21)
        #           +  2/m^2/(m-1)/n   * sum(sum(Kxd))*sum(sum(Kxy)) ...
        v[3] = 2.0 / (m ** 2) / (m - 1) / n * Ky1d_sum * Ky1y2_sum
        #           +  1/(n)/(n-1)/(n-2) * ( sum(Kyd,1)*sum(Kyd,2) - sum(sum(Kyd.^2)))  ...
        v[4] = 1.0 / n / (n - 1) / (n - 2) * (np.dot(Ky2d0_red, Ky2d0_red) - np.sum(Ky2d ** 2))
        #           -  ( 1/n/(n-1)   * sum(sum(Kyd))  )^2	...
        v[5] = -(1.0 / n / (n - 1) * Ky2d_sum) ** 2
        #           -  2/n/(n-1)/m     * sum(Kyd,1) * sum(Kxy',2)  ...
        v[6] = -2.0 / n / (n - 1) / m * np.dot(Ky2d0_red, Ky2y11)
        #           +  2/n^2/(n-1)/m  * sum(sum(Kyd))*sum(sum(Kxy)) ...
        v[7] = 2.0 / (n ** 2) / (n - 1) / m * Ky2d_sum * Ky1y2_sum
        #           +  1/n/(n-1)/m   * ( sum(Kxy',1)*sum(Kxy,2) -sum(sum(Kxy.^2))  ) ...
        v[8] = 1.0 / n / (n - 1) / m * (np.dot(Ky1y21, Ky1y21) - Ky1y22_sum)
        #           - 2*(1/n/m        * sum(sum(Kxy))  )^2 ...
        v[9] = -2.0 * (1.0 / n / m * Ky1y2_sum) ** 2
        #           +   1/m/(m-1)/n   *  ( sum(Kxy,1)*sum(Kxy',2) - sum(sum(Kxy.^2)))  ;
        v[10] = 1.0 / m / (m - 1) / n * (np.dot(Ky2y11, Ky2y11) - Ky1y22_sum)

        # %additional low order correction made to some terms compared with ICLR submission
        # %these corrections are of the same order as the 2nd order term and will
        # %be unimportant far from the null.

        #   %Eq. 13 p. 11 ICLR 2016. This uses ONLY first order term
        #   varEst = 4*(m-2)/m/(m-1) *  varEst  ;
        varEst1st = 4.0 * (m - 2) / m / (m - 1) * np.sum(v)

        Ky1y2d = Ky1y2 - np.diag(np.diag(Ky1y2))
        #   %Eq. 13 p. 11 ICLR 2016: correction by adding 2nd order term
        #   varEst2nd = 2/m/(m-1) * 1/n/(n-1) * sum(sum( (Kxd + Kyd - Kxyd - Kxyd').^2 ));
        varEst2nd = 2.0 / m / (m - 1) * 1 / n / (n - 1) * np.sum((Ky1d + Ky2d - Ky1y2d - Ky1y2d.T) ** 2)

        #   varEst = varEst + varEst2nd;
        varEst = varEst1st + varEst2nd

        #   %use only 2nd order term if variance estimate negative
        if varEst < 0:
            varEst = varEst2nd
            
        return mmd2, varEst
    

    @staticmethod
    def grid_search_kernel(wtst_data, list_kernels, kx, alpha, reg=1e-3):
        """
        Return from the list the best kernel that maximizes the test power criterion.

        In principle, the test threshold depends on the null distribution, which
        changes with kernel. Thus, we need to recompute the threshold for each kernel
        (require permutations), which is expensive. However, asymptotically
        the threshold goes to 0. So, for each kernel, the criterion needed is
        the ratio mean/variance of the MMD^2. (Source: Arthur Gretton)
        This is an approximate to avoid doing permutations for each kernel
        candidate.
        - reg: regularization parameter
        return: (best kernel index, list of test power objective values)
        """
        import time
        X1, X2, Y1, Y2 = wtst_data.x1x2y1y2()
        n = X1.shape[0]
        obj_values = np.zeros(len(list_kernels))
        for ki, k in enumerate(list_kernels):
            start = time.time()
            mmd2, mmd2_var = WQuadMMDTest.h1_mean_var(X1, X2, Y1, Y2, k, kx, is_var_computed=True)
            obj = float(mmd2) / ((mmd2_var + reg) ** 0.5)
            obj_values[ki] = obj
            end = time.time()
            #print('(%d/%d) %s: mmd2: %.3g, var: %.3g, power obj: %g, took: %s' % (ki + 1,
            #                         len(list_kernels), str(k), mmd2, mmd2_var, obj, end - start))
        best_ind = np.argmax(obj_values)
        return best_ind, obj_values

def run_full_WMMD_test(x1, x2, y1, y2,alpha=0.01, output='full'):
    '''
    Runs full WMMD test with all optimization procedures included
    output: the desired output, one of 'stat', 'p_value', 'full'
    require same number of instances in both populations
    '''
    wtst_data = data.WTSTData(x1, x2, y1, y2)
    tr, te = wtst_data.split_tr_te(tr_proportion=0.5)
    y1y2 = tr.stack_y1y2()
    x1x2 = tr.stack_x1x2()
    sig2y = general_utils.meddistance(y1y2, subsample=1000)
    sig2x = general_utils.meddistance(x1x2, subsample=1000)
    #print(sig2y)
    #print(sig2x)
    k = kernel_utils.KGauss(sig2y)
    kx = kernel_utils.KGauss(sig2x)
    
    # choose the best parameter and perform a test with permutations
    med = general_utils.meddistance(tr.stack_y1y2(), 1000)
    alpha = 0.01
    list_gwidth = np.hstack( ( (med**2) *(2.0**np.linspace(-4, 4, 20) ) ) )
    list_gwidth.sort()

    list_kernels = [kernel_utils.KGauss(gw2) for gw2 in list_gwidth]

    # grid search to choose the best Gaussian width
    besti, powers = WQuadMMDTest.grid_search_kernel(tr, list_kernels, kx, alpha)
    # perform test 
    best_ker = list_kernels[besti]
    
    mmd_test = WQuadMMDTest(best_ker, kx, n_permute=200, alpha=alpha)
    if output == 'stat':
        return mmd_test.compute_stat(te)
    if output == 'p_value':
        return mmd_test.compute_pvalue(te)
    if output == 'full':
        return mmd_test.perform_test(te)

# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------


class QuadMMDTest(TwoSampleTest):
    """
    Quadratic MMD test where the null distribution is computed by permutation.
    - Use a single U-statistic i.e., remove diagonal from the Kxy matrix.
    - The code is based on a Matlab code of Arthur Gretton from the paper 
    A TEST OF RELATIVE SIMILARITY FOR MODEL SELECTION IN GENERATIVE MODELS
    ICLR 2016
    """

    def __init__(self, kernel, n_permute=400, alpha=0.01, use_1sample_U=False):
        """
        kernel: an instance of Kernel 
        n_permute: number of times to do permutation
        """
        self.kernel = kernel
        self.n_permute = n_permute
        self.alpha = alpha 
        self.use_1sample_U = use_1sample_U

    def perform_test(self, tst_data):
        """perform the two-sample test and return values computed in a dictionary:
        {alpha: 0.01, pvalue: 0.0002, test_stat: 2.3, h0_rejected: True, ...}
        tst_data: an instance of TSTData
        """
        d = tst_data.dim()
        alpha = self.alpha
        mmd2_stat = self.compute_stat(tst_data, use_1sample_U=self.use_1sample_U)

        X, Y = tst_data.xy()
        k = self.kernel
        repeats = self.n_permute
        list_mmd2 = QuadMMDTest.permutation_list_mmd2(X, Y, k, repeats)
        # approximate p-value with the permutations 
        pvalue = np.mean(list_mmd2 > mmd2_stat)

        results = {'alpha': self.alpha, 'pvalue': pvalue, 'test_stat': mmd2_stat,
                'h0_rejected': pvalue < alpha}#, 'list_permuted_mmd2': list_mmd2}
        return results
    
    def compute_pvalue(self, tst_data):
        """perform the two-sample test and return values computed in a dictionary:
        {alpha: 0.01, pvalue: 0.0002, test_stat: 2.3, h0_rejected: True, ...}
        tst_data: an instance of TSTData
        """
        d = tst_data.dim()
        alpha = self.alpha
        mmd2_stat = self.compute_stat(tst_data, use_1sample_U=self.use_1sample_U)

        X, Y = tst_data.xy()
        k = self.kernel
        repeats = self.n_permute
        list_mmd2 = QuadMMDTest.permutation_list_mmd2(X, Y, k, repeats)
        # approximate p-value with the permutations 
        pvalue = np.mean(list_mmd2 > mmd2_stat)

        return pvalue
    
    def compute_stat(self, tst_data, use_1sample_U=True):
        """Compute the test statistic: empirical quadratic MMD^2"""
        X, Y = tst_data.xy()
        nx = X.shape[0]
        ny = Y.shape[0]

        if nx != ny:
            raise ValueError('nx must be the same as ny')

        k = self.kernel
        mmd2, var = QuadMMDTest.h1_mean_var(X, Y, k, is_var_computed=False,
                use_1sample_U=use_1sample_U)
        return mmd2

    @staticmethod 
    def permutation_list_mmd2(X, Y, k, n_permute=400, seed=8273):
        """
        Repeatedly mix, permute X,Y and compute MMD^2. This is intended to be
        used to approximate the null distritubion.
        TODO: This is a naive implementation where the kernel matrix is recomputed 
        for each permutation. We might be able to improve this if needed.
        """
        return QuadMMDTest.permutation_list_mmd2_gram(X, Y, k, n_permute, seed)

    @staticmethod 
    def permutation_list_mmd2_gram(X, Y, k, n_permute=400, seed=8273):
        """
        Repeatedly mix, permute X,Y and compute MMD^2. This is intended to be
        used to approximate the null distritubion.
        """
        XY = np.vstack((X, Y))
        Kxyxy = k.eval(XY, XY)

        rand_state = np.random.get_state()
        np.random.seed(seed)

        nxy = XY.shape[0]
        nx = X.shape[0]
        ny = Y.shape[0]
        list_mmd2 = np.zeros(n_permute)

        for r in range(n_permute):
            #print r
            ind = np.random.choice(nxy, nxy, replace=False)
            # divide into new X, Y
            indx = ind[:nx]
            #print(indx)
            indy = ind[nx:]
            Kx = Kxyxy[np.ix_(indx, indx)]
            #print(Kx)
            Ky = Kxyxy[np.ix_(indy, indy)]
            Kxy = Kxyxy[np.ix_(indx, indy)]

            mmd2r, var = QuadMMDTest.h1_mean_var_gram(Kx, Ky, Kxy, is_var_computed=False)
            list_mmd2[r] = mmd2r

        np.random.set_state(rand_state)
        return list_mmd2

    @staticmethod
    def h1_mean_var_gram(Kx, Ky, Kxy, is_var_computed, use_1sample_U=True):
        """
        Same as h1_mean_var() but takes in Gram matrices directly.
        """

        nx = Kx.shape[0]
        ny = Ky.shape[0]
        xx = old_div((np.sum(Kx) - np.sum(np.diag(Kx))),(nx*(nx-1)))
        yy = old_div((np.sum(Ky) - np.sum(np.diag(Ky))),(ny*(ny-1)))
        # one-sample U-statistic.
        if use_1sample_U:
            xy = old_div((np.sum(Kxy) - np.sum(np.diag(Kxy))),(nx*(ny-1)))
        else:
            xy = old_div(np.sum(Kxy),(nx*ny))
        mmd2 = xx - 2*xy + yy

        if not is_var_computed:
            return mmd2, None

        # compute the variance
        Kxd = Kx - np.diag(np.diag(Kx))
        Kyd = Ky - np.diag(np.diag(Ky))
        m = nx 
        n = ny
        v = np.zeros(11)

        Kxd_sum = np.sum(Kxd)
        Kyd_sum = np.sum(Kyd)
        Kxy_sum = np.sum(Kxy)
        Kxy2_sum = np.sum(Kxy**2)
        Kxd0_red = np.sum(Kxd, 1)
        Kyd0_red = np.sum(Kyd, 1)
        Kxy1 = np.sum(Kxy, 1)
        Kyx1 = np.sum(Kxy, 0)

        #  varEst = 1/m/(m-1)/(m-2)    * ( sum(Kxd,1)*sum(Kxd,2) - sum(sum(Kxd.^2)))  ...
        v[0] = 1.0/m/(m-1)/(m-2)*( np.dot(Kxd0_red, Kxd0_red ) - np.sum(Kxd**2) )
        #           -  (  1/m/(m-1)   *  sum(sum(Kxd))  )^2 ...
        v[1] = -( 1.0/m/(m-1) * Kxd_sum )**2
        #           -  2/m/(m-1)/n     *  sum(Kxd,1) * sum(Kxy,2)  ...
        v[2] = -2.0/m/(m-1)/n * np.dot(Kxd0_red, Kxy1)
        #           +  2/m^2/(m-1)/n   * sum(sum(Kxd))*sum(sum(Kxy)) ...
        v[3] = 2.0/(m**2)/(m-1)/n * Kxd_sum*Kxy_sum
        #           +  1/(n)/(n-1)/(n-2) * ( sum(Kyd,1)*sum(Kyd,2) - sum(sum(Kyd.^2)))  ...
        v[4] = 1.0/n/(n-1)/(n-2)*( np.dot(Kyd0_red, Kyd0_red) - np.sum(Kyd**2 ) ) 
        #           -  ( 1/n/(n-1)   * sum(sum(Kyd))  )^2	...		       
        v[5] = -( 1.0/n/(n-1) * Kyd_sum )**2
        #           -  2/n/(n-1)/m     * sum(Kyd,1) * sum(Kxy',2)  ...
        v[6] = -2.0/n/(n-1)/m * np.dot(Kyd0_red, Kyx1)

        #           +  2/n^2/(n-1)/m  * sum(sum(Kyd))*sum(sum(Kxy)) ...
        v[7] = 2.0/(n**2)/(n-1)/m * Kyd_sum*Kxy_sum
        #           +  1/n/(n-1)/m   * ( sum(Kxy',1)*sum(Kxy,2) -sum(sum(Kxy.^2))  ) ...
        v[8] = 1.0/n/(n-1)/m * ( np.dot(Kxy1, Kxy1) - Kxy2_sum )
        #           - 2*(1/n/m        * sum(sum(Kxy))  )^2 ...
        v[9] = -2.0*( 1.0/n/m*Kxy_sum )**2
        #           +   1/m/(m-1)/n   *  ( sum(Kxy,1)*sum(Kxy',2) - sum(sum(Kxy.^2)))  ;
        v[10] = 1.0/m/(m-1)/n * ( np.dot(Kyx1, Kyx1) - Kxy2_sum )


        #%additional low order correction made to some terms compared with ICLR submission
        #%these corrections are of the same order as the 2nd order term and will
        #%be unimportant far from the null.

        #   %Eq. 13 p. 11 ICLR 2016. This uses ONLY first order term
        #   varEst = 4*(m-2)/m/(m-1) *  varEst  ;
        varEst1st = 4.0*(m-2)/m/(m-1) * np.sum(v)

        Kxyd = Kxy - np.diag(np.diag(Kxy))
        #   %Eq. 13 p. 11 ICLR 2016: correction by adding 2nd order term
        #   varEst2nd = 2/m/(m-1) * 1/n/(n-1) * sum(sum( (Kxd + Kyd - Kxyd - Kxyd').^2 ));
        varEst2nd = 2.0/m/(m-1) * 1/n/(n-1) * np.sum( (Kxd + Kyd - Kxyd - Kxyd.T)**2)

        #   varEst = varEst + varEst2nd;
        varEst = varEst1st + varEst2nd

        #   %use only 2nd order term if variance estimate negative
        if varEst<0:
            varEst =  varEst2nd
        return mmd2, varEst

    @staticmethod
    def h1_mean_var(X, Y, k, is_var_computed, use_1sample_U=True):
        """
        X: nxd numpy array 
        Y: nxd numpy array
        k: a Kernel object 
        is_var_computed: if True, compute the variance. If False, return None.
        use_1sample_U: if True, use one-sample U statistic for the cross term 
          i.e., k(X, Y).
        Code based on Arthur Gretton's Matlab implementation for
        Bounliphone et. al., 2016.
        return (MMD^2, var[MMD^2]) under H1
        """

        nx = X.shape[0]
        ny = Y.shape[0]

        Kx = k.eval(X, X)
        Ky = k.eval(Y, Y)
        Kxy = k.eval(X, Y)

        return QuadMMDTest.h1_mean_var_gram(Kx, Ky, Kxy, is_var_computed, use_1sample_U)

    @staticmethod
    def grid_search_kernel(tst_data, list_kernels, alpha, reg=1e-3):
        """
        Return from the list the best kernel that maximizes the test power criterion.
        
        In principle, the test threshold depends on the null distribution, which 
        changes with kernel. Thus, we need to recompute the threshold for each kernel
        (require permutations), which is expensive. However, asymptotically 
        the threshold goes to 0. So, for each kernel, the criterion needed is
        the ratio mean/variance of the MMD^2. (Source: Arthur Gretton)
        This is an approximate to avoid doing permutations for each kernel 
        candidate.
        - reg: regularization parameter
        return: (best kernel index, list of test power objective values)
        """
        import time
        X, Y = tst_data.xy()
        n = X.shape[0]
        obj_values = np.zeros(len(list_kernels))
        for ki, k in enumerate(list_kernels):
            start = time.time()
            mmd2, mmd2_var = QuadMMDTest.h1_mean_var(X, Y, k, is_var_computed=True)
            obj = float(mmd2)/((mmd2_var + reg)**0.5)
            obj_values[ki] = obj
            end = time.time()
            #print('(%d/%d) %s: mmd2: %.3g, var: %.3g, power obj: %g, took: %s'%(ki+1,
            #    len(list_kernels), str(k), mmd2, mmd2_var, obj, end-start))
        best_ind = np.argmax(obj_values)
        return best_ind, obj_values
    
    
def run_full_MMD_test(x1,x2,y1,y2,alpha=0.01,output = 'full'):
    '''
    Runs full test with all optimization procedures included
    output: the desired output, one of 'stat', 'p_value', 'full'
    '''
    tst_data = data.TSTData(y1, y2)
    tr, te = tst_data.split_tr_te(tr_proportion=0.5, seed=10)
    xtr, ytr = tr.xy()
    xytr = tr.stack_xy()
    sig2 = general_utils.meddistance(xytr, subsample=1000)
    k = kernel_utils.KGauss(sig2)
    
    # choose the best parameter and perform a test with permutations
    med = general_utils.meddistance(tr.stack_xy(), 1000)
    list_gwidth = np.hstack( ( (med**2) *(2.0**np.linspace(-4, 4, 20) ) ) )
    list_gwidth.sort()

    list_kernels = [kernel_utils.KGauss(gw2) for gw2 in list_gwidth]

    # grid search to choose the best Gaussian width
    besti, powers = QuadMMDTest.grid_search_kernel(tr, list_kernels, alpha)
    # perform test 
    best_ker = list_kernels[besti]
    
    mmd_test = QuadMMDTest(best_ker, n_permute=200, alpha=alpha)
    if output == 'stat':
        return mmd_test.compute_stat(te)
    if output == 'p_value':
        return mmd_test.compute_pvalue(te)
    if output == 'full':
        return mmd_test.perform_test(te)

def test_KMM():
    x = 11 * np.random.random(200) - 6.0 # x lies in [-6,5]
    y = x ** 2 + 10 * np.random.random(200) - 5
    x1 = np.c_[x, y]

    x = 2 * np.random.random(100) - 6.0 # x lies in [-6,-4]
    y = x ** 2 + 10 * np.random.random(100) - 5
    x2 = np.c_[x, y]

    x1x2 = np.vstack((x1, x2))
    sig2 = general_utils.meddistance(x1x2, subsample=1000)
    print(sig2)
    k = kernel_utils.KGauss(sig2)

    coef = tst.WQuadMMDTest.kernel_mean_matching(x1, x2, k, B=10)

    plt.close()
    plt.figure()
    plt.scatter(x1[:, 0], x1[:, 1], color='black', marker='x')
    plt.scatter(x2[:, 0], x2[:, 1], color='red')
    plt.scatter(x1[:, 0], x1[:, 1], color='green', s=coef * 10, alpha=0.5)

    np.sum(coef > 1e-2)
