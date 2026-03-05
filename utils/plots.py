import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Plotting kernel comparaison
def plot_fit_comparison(x, y, alphas_dict, x_m, agent_idx, Cov2, folder, filename, nt=250, out_dir="figures"):

    x_prime = np.linspace(-1, 1, nt)
    plt.figure()
    plt.plot(x, y, 'o', label='Data', alpha=0.5)
    for label, alpha in alphas_dict.items():
        y_prime = Cov2(x_prime, x_m) @ alpha[agent_idx]
        plt.plot(x_prime, y_prime, '-', label=label)
    plt.xlabel(r'$x$ feature', fontsize=12)
    plt.ylabel(r'$y$ label', fontsize=12)
    plt.title(f'Agent {agent_idx+1}', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()

    path = Path("figures") / folder / filename
    plt.savefig(path)
    plt.show()

# Plotting gaps with alpha^*
def plot_gap(gaps, folder, filename='DGD_gap.pdf', out_dir="figures"):

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gaps = np.array(gaps) 
    plt.figure()
    for i in range(gaps.shape[1]):
        plt.loglog(range(1, len(gaps)+1), gaps[:, i], label=f'Agent {i+1}')
    plt.xlabel('Iterations $t$', fontsize=12)
    plt.ylabel(r'$\|\alpha_i^t - \alpha^*\|$', fontsize=12)
    plt.title('DGD Optimality Gap', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, which='both')
    plt.tight_layout()
    
    path = Path("figures") / folder / filename
    plt.savefig(path)
    plt.show()
