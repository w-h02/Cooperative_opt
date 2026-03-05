import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist


def Cov(x):
    m = len(x)
    Kmm = np.eye(m)
    for ii in range(m):
        for jj in range(ii+1,m):
            Kmm[ii,jj] = np.exp(-(x[ii]-x[jj])**2 )
            Kmm[jj,ii] = Kmm[ii,jj]

    return Kmm

def Cov2(x1,x2):
    m = len(x2)
    n = len(x1)
    Knm = np.zeros([n,m])
    for ii in range(n):
        for jj in range(m):
            Knm[ii, jj] = np.exp(-(x1[ii] - x2[jj]) ** 2 )
    return Knm

# Build the mixing matrix by the Metropolis Hastings method

def build_mixing_matrix(A):
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

def local_gradient(alpha, agents_x_i, agents_y_i, x_m, sigma = 0.5, nu = 1.0, a = 5):
    """
    entries:
        agents_x_i, agent_y_i: data points of agent i
        x_m: set of m points
        sigma, nu, a = params
    returns:
        local gradient of f_i
    """
    Kmm = Cov(x_m)
    Knm_i = Cov2(agents_x_i, x_m)
    residual = agents_y_i - Knm_i @ alpha
    
    grad = (sigma**2 / a) * Kmm @ alpha \
           - Knm_i.T @ residual \
           + (nu / a) * alpha
    return grad

def local_gradient_vector(alpha, agents_x, agents_y, x_m, sigma=0.5, nu=1.0, a=5):
    """
    Compute the local gradients for all agents
    """
    n_agents = len(agents_x)
    m = len(x_m)
    
    grad_matrix = np.zeros((n_agents, m))

    for i in range(n_agents):
        grad = local_gradient(alpha[i, :], agents_x[i], agents_y[i], x_m, sigma=sigma, nu=nu, a=a)
        grad_matrix[i, :] = grad.flatten()  

    return grad_matrix

def is_double_stoch(W):
    """
    Verifies if the matrix W is double stochastic
    """
    return np.allclose(W.sum(axis=1), 1) and np.allclose(W.sum(axis=0), 1)