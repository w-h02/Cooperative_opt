import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist


# Build the weight matrix by the Metropolis Hastings method

def build_weight_matrix(A):
    """
    entries:
     A: communication graph (matrix)
    returns:
     W: weight matrix by the Metropolis Hastings method:
        W[i,j] = 1 / (1 + max(dᵢ, dⱼ)) for i!= j connected
        W[i,i] = 1 - Σⱼ W[i,j]
    """
    n = A.shape[0]
    degrees = A.sum(axis=1)
    W = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if A[i,j] == 1:
                W[i,j] = 1 / (1 + max(degrees[i], degrees[j]))
        W[i,i] = 1 - W[i,:].sum()
    return W
