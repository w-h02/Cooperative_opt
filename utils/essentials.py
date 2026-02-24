import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

def kernel(x, y):
    """
    x,y: 2 vectors
    returns k(x,y) = exp(-||x-y||^2)
    """
    return np.exp(-np.linalg.norm(x - y)**2)

def kernel_matrix(x):
    """
    x: vector
    returns matrix K = (k(x_i, x_j) for i,j in [| 1, n|]) )
    """
    sq_dist = cdist(x, x, 'sqeuclidean')
    return np.exp(-sq_dist)

def get_random_points(X, m):
    """
    X: vector 
    m: number of indices needed
    returns the list of m randomly picked points (X_m) and a list of their indices (M)
    """
    M, X_m = [], []
    n = len(X)
    while len(M) != m:
        p = np.random.randint(0, n)
        if p not in M:
            M.append(p)
            X_m.append(X[p])
    return X_m, M

