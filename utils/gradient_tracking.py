import numpy as np
from utils.essentials import *

def gradient_tracking(x_m, agents_x, W, agents_y, alpha_star, n_iter = 100, sigma = 0.5, nu = 1.0, eta = 0.01):
    n_agents = len(agents_x)
    m = len(x_m)

    # initialization
    alpha = np.zeros((n_agents, m))
    g = local_gradient_vector(alpha, agents_x, agents_y, x_m)

    gaps = []
    alphas = [alpha.copy()]
    gs = [g.copy()]

    for t in range(n_iter):

        # Communication step
        alpha_mix = W @ alpha
        g_mix = W @ g

        # Gradient step
        new_alpha = alpha_mix - eta * g_mix

        # Gradient tracking update
        grad_new = local_gradient_vector(new_alpha, agents_x, agents_y, x_m)
        grad_old = local_gradient_vector(alpha, agents_x, agents_y, x_m)

        new_g = g_mix + (grad_new - grad_old)

        alpha = new_alpha
        g = new_g

        gap = [np.linalg.norm(alpha[i] - alpha_star) for i in range(n_agents)]
        gaps.append(gap)

        alphas.append(alpha.copy())
        gs.append(g.copy())
    
    return alphas, gaps, gs
