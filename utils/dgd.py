import numpy as np
from centralized_solution import *
from utils.essentials import *

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

# dgd with constant stepsize
def DGD(x_m, agents_x, W, agents_y, alpha_star, n_iter = 100, sigma = 0.5, nu = 1.0, eta = 0.01):
    m = len(x_m)
    n_agents = len(agents_x)
    # Initializing alphas to zero
    alphas = [np.zeros(m) for _ in range(n_agents)]
    gaps = []
    
    for i in range(n_agents):
        for j in range(n_iter):
            # Communication step
            new_alphas = [sum(W[i,j] * alphas[j] for j in range(n_agents)) for i in range(n_agents)]
            # Computation step
            new_alphas = [
            new_alphas[i] - eta * local_gradient(alphas[i], agents_x[i], agents_y[i], x_m)
            for i in range(n_agents)
            ]
            alphas = new_alphas

            # Optimality gap 
            gap = np.mean([np.linalg.norm(alphas[i] - alpha_star) for i in range(n_agents)])
            gaps.append(gap)

    return alphas, gaps

# Since the problem is quadratic, we can find the Lipschitz constant L s.t. eta = 1/L and L is he largest eigenvalue of the local Hessian

def get_best_L(agents_x, x_m, sigma=0.5, nu=1.0, a=5):

    Kmm = Cov(x_m)
    # sum local Hessians across agents
    H = (sigma**2 / a) * Kmm + (nu / a) * np.eye(len(x_m))
    for i in range(a):
        Knm_i = Cov2(agents_x[i], x_m)
        H += Knm_i.T @ Knm_i
    ei = np.linalg.eigvalsh(H)
    return max(ei)
