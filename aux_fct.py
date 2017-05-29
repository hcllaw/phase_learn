### Auxillary Functions ###
import os
import numpy as np
from numpy import exp, shape, reshape, sqrt, median
from numpy.random import permutation,randn

from scipy.io import loadmat
from scipy.spatial.distance import squareform, pdist, cdist
from sklearn.metrics.pairwise import euclidean_distances


# Median Heuristic method for the RBF Kernel
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

# Median Heuristic method for the RBF Kernel
def median_sqdist(feats, n_sub=1000):
    all_Xs = np.concatenate(feats)
    N = all_Xs.shape[0]
    sub = all_Xs[np.random.choice(N, min(n_sub, N), replace=False)]
    D2 = euclidean_distances(sub, squared=True)
    return np.median(D2[np.triu_indices_from(D2, k=1)], overwrite_input=True)

# Auxillary Dataset Loading 
def load_data(path, random = False, seed = 30):
	os.chdir(path)
	mat = loadmat('MISR1.mat')
	array = mat['MISR1']
	x = np.zeros((800, 100, 16))
	y = np.zeros((800,1))
	for i in range(800):
		for j in range(100):
			x[i, j ,:] = array[i * 100 + j, 1:17]
			y[i,:] = array[i * 100 + j, 17]
	if random:
		np.random.seed(seed)
		permute_seq = np.random.permutation(800)
		x = x[permute_seq,:,:]
		y = y[permute_seq]
	return x, y



	return x, y