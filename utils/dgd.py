import numpy as np
from utils.essentials import *

# dgd with constant stepsize
def DGD(x_m, agents_x, W, agents_y, alpha_star, n_iter = 100, sigma = 0.5, nu = 1.0, eta = 0.01):
    m = len(x_m)
    n_agents = len(agents_x)

    # Initializing alphas to zero
    alpha = [np.zeros(m) for _ in range(n_agents)]  
    gaps = []
    alphas = [alpha]

    for t in range(n_iter):
        # Communication step
        alpha_matrix = np.array(alpha)
        new_alpha_matrix = W @ alpha_matrix
        
        # Gradient step
        alpha = [
            new_alpha_matrix[i] - eta * local_gradient(new_alpha_matrix[i], agents_x[i], agents_y[i], x_m)
            for i in range(n_agents)
        ]
        
        gap = [np.linalg.norm(alpha[i] - alpha_star) for i in range(n_agents)]
        gaps.append(gap)
        alphas.append(alpha)

    return alphas, np.array(gaps)

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
