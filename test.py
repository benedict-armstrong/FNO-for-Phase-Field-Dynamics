import numpy as np


from lib.allen_cahn import generate_dataset


def convergence_ratio(epsilon, ic_type, t_grid):
    nx = 128
    x_grid = np.linspace(-1, 1, nx)

    t_grid = np.linspace(0, final_time, 5)

    data = generate_dataset(100, epsilon, x_grid, t_grid, ic_type=ic_type, seed=100)


ic_types = ["fourier", "gmm", "piecewise"]
epsilons = [0.1, 0.05, 0.02]

for ic_type in ic_types:
    for eps in epsilons:
        
    