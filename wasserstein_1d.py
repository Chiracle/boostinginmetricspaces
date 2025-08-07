# Modified from
# https://github.com/matthieubulte/pyfrechet/blob/main/pyfrechet/metric_spaces/wasserstein_1d.py


import numpy as np
from .metric_space import MetricSpace
from sklearn.isotonic import isotonic_regression

class Wasserstein1D(MetricSpace):
#    GRID = np.linspace(0, 1, 100) 
    #GRID = np.linspace(0, 1, 3) 
    #GRID = np.linspace(0, 1, 24) # Changed by me from 100

    def __init__(self, dim_w):
        self.dim_w = dim_w
        pass

    @property
    def GRID(self):
        return np.linspace(0, 1, self.dim_w)

    def _d(self, x, y):
#        return np.sqrt(np.trapz((x - y)**2, Wasserstein1D.GRID))
        return np.sqrt(np.trapz((x - y)**2, self.GRID))
    
    def _frechet_mean(self, y, w):
        # Computed the (weighted) averaged quantiles, then project to make sure that q is increasing
        #if len(y.shape)==1:
        #    y = np.expand_dims(y, -1)
        #if len(w.shape) == 1:
        #    w = np.expand_dims(w,-1)
        return isotonic_regression(np.dot(w, y))

    def __str__(self):
        return 'Wasserstein'

def noise(J=5, grid=None):
    if grid is None:
        grid = self.GRID
    def _T(K): return grid if K == 0 else grid - np.sin(grid * np.pi * K) / (np.pi * np.abs(K))
    def _K(): return 2 * (np.random.binomial(1,0.5) - 1)*np.random.poisson(3)
    U = np.sort(np.random.uniform(size=J-1))
    T = np.array([ _T(_K()) for _ in range(J) ])
    return U[0] * T[0,:] + np.dot(U[1:] - U[:-1], T[1:-1, :]) + (1 - U[-1]) * T[-1,:]

def noise_2(x, l=4):
    k = (1 - 2*np.random.binomial(1,0.5)) * np.random.random_integers(1, l)
    return x - np.sin(np.pi * k * x) / (np.pi*np.abs(k))