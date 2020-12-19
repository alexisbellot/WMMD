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

'''This is exactly as in https://github.com/wittawatj/interpretable-test'''

class Kernel(with_metaclass(ABCMeta, object)):
    """Abstract class for kernels"""

    @abstractmethod
    def eval(self, X1, X2):
        """Evalute the kernel on data X1 and X2 """
        pass

    @abstractmethod
    def pair_eval(self, X, Y):
        """Evaluate k(x1, y1), k(x2, y2), ..."""
        pass

class KGauss(Kernel):

    def __init__(self, sigma2):
        assert sigma2 > 0, 'sigma2 must be > 0'
        self.sigma2 = sigma2

    def eval(self, X1, X2):
        """
        Evaluate the Gaussian kernel on the two 2d numpy arrays.
        Parameters
        ----------
        X1 : n1 x d numpy array
        X2 : n2 x d numpy array
        Return
        ------
        K : a n1 x n2 Gram matrix.
        """
        (n1, d1) = X1.shape
        (n2, d2) = X2.shape
        assert d1==d2, 'Dimensions of the two inputs must be the same'
        D2 = np.sum(X1**2, 1)[:, np.newaxis] - 2*np.dot(X1, X2.T) + np.sum(X2**2, 1)
        K = np.exp(old_div(-D2,self.sigma2))
        
        return K

    def pair_eval(self, X, Y):
        """
        Evaluate k(x1, y1), k(x2, y2), ...
        Parameters
        ----------
        X, Y : n x d numpy array
        Return
        -------
        a numpy array with length n
        """
        (n1, d1) = X.shape
        (n2, d2) = Y.shape
        assert n1==n2, 'Two inputs must have the same number of instances'
        assert d1==d2, 'Two inputs must have the same dimension'
        D2 = np.sum( (X-Y)**2, 1)
        Kvec = np.exp(old_div(-D2,self.sigma2))
        return Kvec

    def __str__(self):
        return "KGauss(w2=%.3f)"%self.sigma2