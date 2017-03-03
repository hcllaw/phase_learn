# Auxillary Code
from numpy import exp, shape, reshape, sqrt, median
from numpy.random import permutation,randn
from scipy.spatial.distance import squareform, pdist, cdist
import numpy as np
import os
from scipy.io import loadmat

def get_sigma_median_heuristic(X, is_sparse = False):
    if is_sparse:
        X = X.todense()
    n=shape(X)[0]
    if n>1000:
        X=X[permutation(n)[:1000],:]
    dists=squareform(pdist(X, 'euclidean'))
    median_dist=median(dists[dists>0])
    sigma=median_dist/sqrt(2.)
    return sigma

def load_data(path):
	os.chdir(path)
	mat = loadmat('MISR1.mat')
	array = mat['MISR1']
	x = np.zeros((800, 100, 16))
	y = np.zeros((800,1))
	for i in range(800):
		for j in range(100):
			x[i, j ,:] = array[i * 100 + j, 1:17]
			y[i,:] = array[i * 100 + j, 17]
	return x, y