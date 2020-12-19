from __future__ import print_function
from __future__ import division
from builtins import map
from builtins import str
from builtins import range
from past.utils import old_div
from builtins import object
from future.utils import with_metaclass

from abc import ABCMeta, abstractmethod
import math
import matplotlib.pyplot as plt
import autograd.numpy as np
import general_utils as general_utils

import scipy.stats as stats
import numpy as np
import random

class SampleSource(with_metaclass(ABCMeta, object)):
    """A data source where it is possible to resample. Subclasses may prefix
    class names with SS"""

    @abstractmethod
    def sample(self, n, seed):
        """Return a TSTData. Returned result should be deterministic given
        the input (n, seed)."""
        raise NotImplementedError()

    @abstractmethod
    def dim(self):
        """Return the dimension of the problem"""
        raise NotImplementedError()

    def visualize(self, n=400):
        """Visualize the data, assuming 2d. If not possible graphically,
        subclasses should print to the console instead."""
        data = self.sample(n, seed=1)
        y1, y2 = data.y1y2()
        d = y1.shape[1]

        if d==2:
            plt.plot(y1[:, 0], y1[:, 1], '.r', label='Y1')
            plt.plot(y2[:, 0], y2[:, 1], '.b', label='Y2')
            plt.legend(loc='best')
        else:
            # not 2d. Print stats to the console.
            print(data)

class TSTData(object):
    """Class representing data for two-sample test"""

    """
    properties:
    X, Y: numpy array 
    """

    def __init__(self, X, Y, label=None):
        """
        :param X: n x d numpy array for dataset X1 with confounders of population
        :param Y: n x d numpy array for dataset Y
        """
        self.X = X
        self.Y = Y
        # short description to be used as a plot label
        self.label = label

        nx, dx = X.shape
        ny, dy = Y.shape

        # if nx != ny:
        #    raise ValueError('Data sizes must be the same.')
        if dx != dy:
            raise ValueError('Dimension sizes of the two datasets must be the same.')

    def __str__(self):
        mean_x = np.mean(self.X, 0)
        std_x = np.std(self.X, 0)
        mean_y = np.mean(self.Y, 0)
        std_y = np.std(self.Y, 0)
        prec = 4
        desc = ''
        desc += 'E[x] = %s \n ' %(np.array_str(mean_x, precision=prec ) )
        desc += 'E[y] = %s \n ' %(np.array_str(mean_y, precision=prec ) )
        desc += 'Std[x] = %s \n' %(np.array_str(std_x, precision=prec))
        desc += 'Std[y] = %s \n' % (np.array_str(std_y, precision=prec))
        return desc

    def dimension(self):
        """Return the dimension of the data."""
        dx = self.X.shape[1]
        return dx

    def dim(self):
        """Same as dimension()"""
        return self.dimension()

    def stack_xy(self):
        """Stack the two datasets together"""
        return np.vstack((self.X, self.Y))

    def xy(self):
        """Return (X, Y) as a tuple"""
        return (self.X, self.Y)

    def mean_std(self):
        """Compute the average standard deviation """

        # Gaussian width = mean of stds of all dimensions
        X, Y = self.xy()
        stdx = np.mean(np.std(X, 0))
        stdy = np.mean(np.std(Y, 0))
        mstd = old_div((stdx + stdy), 2.0)
        return mstd
        # xy = self.stack_xy()
        # return np.mean(np.std(xy, 0)**2.0)**0.5

    def split_tr_te(self, tr_proportion=0.5, seed=820):
        """Split the dataset into training and test sets. Assume n is the same
        for both X, Y.

        Return (TSTData for tr, TSTData for te)"""
        X = self.X
        Y = self.Y
        nx, dx = X.shape
        ny, dy = Y.shape
        if nx != ny:
            raise ValueError('Require nx = ny')
        Itr, Ite = general_utils.tr_te_indices(nx, tr_proportion, seed)
        label = '' if self.label is None else self.label
        tr_data = TSTData(X[Itr, :], Y[Itr, :], 'tr_' + label)
        te_data = TSTData(X[Ite, :], Y[Ite, :], 'te_' + label)
        return (tr_data, te_data)

    def subsample(self, n, seed=87):
        """Subsample without replacement. Return a new TSTData """
        if n > self.X.shape[0] or n > self.Y.shape[0]:
            raise ValueError('n should not be larger than sizes of X, Y.')
        ind_x = subsample_ind(self.X.shape[0], n, seed)
        ind_y = subsample_ind(self.Y.shape[0], n, seed)
        return TSTData(self.X[ind_x, :], self.Y[ind_y, :], self.label)
    
class WTSTData(object):
    """Class representing data for weighted two-sample test"""
    """
    properties:
    X1, X2, Y1, Y2: numpy array with counfounder and outcome variables for populations 1 and 2 to be compared
    """

    def __init__(self, X1, X2, Y1, Y2, label=None):
        """
        :param X1: n x d numpy array for dataset X1 with confounders of population 1
        :param X2: n x d numpy array for dataset X2 with confounders of population 2
        :param Y1: n x d numpy array for dataset Y1 with outcomes of population 1
        :param Y2: n x d numpy array for dataset Y2 with outcomes of population 2
        """
        self.X1 = X1; self.X2 = X2 
        self.Y1 = Y1; self.Y2 = Y2
        # short description to be used as a plot label
        self.label = label

        nx1, dx1 = X1.shape; nx2, dx2 = X2.shape
        ny1, dy1 = Y1.shape; ny2, dy2 = Y2.shape

        if dx1 != dx2:
            raise ValueError('Dimension sizes of the two counfounder datasets must be the same.')
        if dy1 != dy2:
            raise ValueError('Dimension sizes of the two outcome datasets must be the same.')

    def __str__(self):
        mean_y1 = np.mean(self.Y1, 0);  std_y1 = np.std(self.Y1, 0)
        mean_y2 = np.mean(self.Y2, 0);  std_y2 = np.std(self.Y2, 0)
        prec = 4
        desc = ''
        desc += 'E[y1] = %s; ' %(np.array_str(mean_y1, precision=prec ) )
        desc += 'E[y2] = %s; ' %(np.array_str(mean_y2, precision=prec ) )
        desc += 'Std[y1] = %s; ' %(np.array_str(std_y1, precision=prec))
        desc += 'Std[y2] = %s; ' % (np.array_str(std_y2, precision=prec))
        return desc
    
    def view_selection_prob(self):
        """Show selection probabilities"""
        raise NotImplementedError()

    def dimension(self):
        """Return the dimension of the data."""
        dx = self.X1.shape[1]
        return dx

    def dim(self):
        """Same as dimension()"""
        return self.dimension()
    
    def dim_y(self):
        """Same as dimension()"""
        dy = self.Y1.shape[1]
        return dy

    def stack_y1y2(self):
        """Stack the two datasets together, 
        to be used for computation of median pairwise distances in actual test"""
        return np.vstack((self.Y1, self.Y2))
    
    def stack_x1x2(self):
        """Stack the two datasets together, 
        to be used for computation of median pairwise distances in KMM"""
        return np.vstack((self.X1, self.X2))

    def x1x2y1y2(self):
        """Return (X1, X2, Y1, Y2) as a tuple"""
        return (self.X1, self.X2, self.Y1, self.Y2)


    def split_tr_te(self, tr_proportion=0.5, seed=820):
        """Split the dataset into training and test sets. Assume n is the same
        for both X, Y.

        Return (TSTData for tr, TSTData for te)"""
        X1 = self.X1; X2 = self.X2
        Y1 = self.Y1; Y2 = self.Y2
        nx1, dx1 = X1.shape
        nx2, dx2 = X2.shape
        if nx1 != nx2:
            raise ValueError('Require size of two data sets to be the same.')
        Itr, Ite = general_utils.tr_te_indices(nx1, tr_proportion, seed)
        label = '' if self.label is None else self.label
        tr_data = WTSTData(X1[Itr, :], X2[Itr, :], Y1[Itr, :], Y2[Itr, :], 'tr_' + label)
        te_data = WTSTData(X1[Ite, :], X2[Ite, :], Y1[Ite, :], Y2[Ite, :], 'te_' + label)
        return (tr_data, te_data)

    def subsample(self, n, seed=87):
        """Subsample without replacement. Return a new TSTData """
        if n > self.X.shape[0] or n > self.Y.shape[0]:
            raise ValueError('n should not be larger than sizes of populations 1 or 2.')
        ind_1 = general_utils.subsample_ind(self.Y1.shape[0], n, seed)
        ind_2 = general_utils.subsample_ind(self.Y2.shape[0], n, seed)
        return WTSTData(self.X1[ind_1, :], self.X2[ind_2, :], self.Y1[ind_1, :], self.Y2[ind_2, :], self.label)
    
class SSSameGauss_with_SB(SampleSource):
    """Two same standard Gaussians for P, Q with selection bias. The null hypothesis
    H0: P=Q is true."""
    def __init__(self, d):
        """
        d: dimension of the confounding variables
        mu: amount of selection bias
        """
        self.d = d

    def dim(self):
        return self.d
    
    # TODO: Define sampling mechanism
    def sample(self, n, mu, seed):
        rstate = np.random.get_state()
        np.random.seed(seed)

        #x = 11 * np.random.random(200) - 6.0 # x lies in [-6,5]
        x1 = np.random.normal(np.repeat(0, self.d), 1.0,size=n)
        y1 = x1 + x1 ** 2 + np.random.random(200)

        #x = 2 * np.random.random(200) - 2.0 # x lies in [-2,0]
        x2 = np.random.normal(np.repeat(mu, self.d), 1.0,size=n)
        y2 = x2 + x2 ** 2 + np.random.random(200)
        
        return x1[:,np.newaxis], x2[:,np.newaxis], y1[:,np.newaxis], y2[:,np.newaxis]




def same(x):
    return x

def cube(x):
    return np.power(x, 3)

def generate_samples_random(size=1000, mu = 0, var=1, dx=1, dy=1, noise = "gaussian",
                            f1='linear', f2='linear',seed = None):
    '''Generate null or alternative nonlinear samples with different degrees of confounding
    1. X1 and X2 independent Gaussians - confounding variables
    2. Y = f1(X1) + noise and Y = f2(X2)
    Arguments:
        size : number of samples
        mu: mean of X
        var: variance of X
        dx: Dimension of X
        dy: Dimension of Y
        nstd: noise standard deviation
        noise: type of noise
        f1,f2 to be within {x,x^2,x^3,tanh x, cos x}

    Output:
        allsamples --> complete data-set
    '''
    if seed == None:
        np.random.seed()
    else:
        np.random.seed(seed)

    if f1 == 'linear':
        f1 = same
    if f1 == 'square':
        f1 = np.square
    if f1 == 'tanh':
        f1 = np.tanh
    if f1 == 'cos':
        f1 = np.cos
    if f1 == 'cube':
        f1 = cube

    if f2 == 'linear':
        f2 = same
    if f2 == 'square':
        f2 = np.square
    if f2 == 'tanh':
        f2 = np.tanh
    if f2 == 'cos':
        f2 = np.cos
    if f2 == 'cube':
        f2 = cube    

    # generate confounding variables for 1st sample
    mean = np.zeros(dx); cov = np.eye(dx)
    X1 = np.random.multivariate_normal(mean, cov, size)
    X1 = np.matrix(X1)
    
    # generate confounding variables for 2nd sample
    mean = np.zeros(dx) + mu; cov = np.eye(dx)*var
    X2 = np.random.multivariate_normal(mean, cov, size)
    X2 = np.matrix(X2)
    
    Axy = np.random.rand(dx, dy)
    Axy = np.matrix(Axy)
    
    if noise == 'gaussian':
        noise1 = np.random.multivariate_normal(np.zeros(dy), np.eye(dy)*0.5, size)
        noise2 = np.random.multivariate_normal(np.zeros(dy), np.eye(dy)*0.5, size)
        noise1 = np.matrix(noise1)
        noise2 = np.matrix(noise2)
        
    elif noise == 'exp':
        noise1 = numpy.random.exponential(scale=1.0, size=size)
        noise2 = numpy.random.exponential(scale=1.0, size=size)
        noise1 = np.matrix(noise1)
        noise2 = np.matrix(noise2)
    
    if dx == dy:
        Y1 = X1; Y2=X2
        Y1[:,0] = f1(X1[:,0]) + noise1[:,0]
        Y2[:,0] = f2(X2[:,0]) + noise2[:,0]
        Y1[:,1:] = f1(X1[:,1:]) + noise1[:,1:]
        Y2[:,1:] = f2(X2[:,1:]) + noise2[:,1:]
    else:
        Y1 = f1(X1 * Axy) + noise1
        Y2 = f2(X2 * Axy) + noise2
        
    return np.array(X1), np.array(X2), np.array(Y1), np.array(Y2)


def test():
    size=1000;mu = 2; var=1; dx=20; dy=20; f1=np.square; f2=np.square

    # generate confounding variables for 1st sample
    mean = np.zeros(dx); cov = np.eye(dx)
    X1 = np.random.multivariate_normal(mean, cov, size)
    X1 = np.matrix(X1)
    
    # generate confounding variables for 2nd sample
    mean = np.zeros(dx) + mu; cov = np.eye(dx)*var
    X2 = np.random.multivariate_normal(mean, cov, size)
    X2 = np.matrix(X2)

    # Define data generating process for y
    Axy = np.random.rand(dx, dy)
    for i in range(dy):
        Axy[:, i] = Axy[:, i] / np.linalg.norm(Axy[:, i], ord=1)
    Axy = np.matrix(Axy)
    
    print(Axy[:10,])
    print(f1(X1 * Axy).shape)
    print(f1(X1 * Axy)[:10])
    #print(min(f1(X1)))

    #print(max(f1(X1 * Axy)))
    #print(min(f1(X1 * Axy)))

    #print(max(f2(X2 * Axy)))
    #print(min(f2(X2 * Axy)))

    print(np.mean(np.abs(f1(X1 * Axy))))
    print(np.mean(np.abs(f2(X2 * Axy))))