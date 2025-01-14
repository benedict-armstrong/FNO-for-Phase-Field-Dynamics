import os
import numpy as np
from scipy.integrate import solve_ivp
import tqdm


def generate_fourier_ic(x, n_modes=2, seed=None):
    """Generate random Fourier series initial condition.
    Hints:
    1. Use random coefficients for sin and cos terms
    2. Ensure the result is normalized to [-1, 1]
    3. Consider using np.random.normal for coefficients
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate coefficients for Fourier series
    coeff = np.random.normal(size=(2 * n_modes + 1))

    # Compute the Fourier series
    res = np.zeros_like(x)
    for k in range(1, n_modes + 1):
        res += coeff[k] * np.sin(2 * np.pi * k * x)
        res += coeff[k + n_modes] * np.cos(2 * np.pi * k * x)
    res += coeff[0]

    # Normalize to [-1, 1]
    res -= np.min(res)
    res /= np.max(res)

    return 2 * res - 1


def generate_gmm_ic(x, n_components=None, seed=None):
    """Generate Gaussian mixture model initial condition.
    Hints:
    1. Random number of components if n_components is None
    2. Use random means, variances, and weights
    3. Ensure result is normalized to [-1, 1]
    """
    if seed is not None:
        np.random.seed(seed)

    if n_components is None:
        n_components = np.random.randint(2, 6)

    # make response 3 time the size of x
    res = np.zeros_like(x)
    res = np.tile(res, 3)

    # Generate means, variances, and weights
    for _ in range(n_components):
        mean = np.random.uniform(x.min(), x.max())
        variance = np.random.uniform(x.max() / 10, x.max() / 2)
        weight = np.random.uniform(-1.0, 1.0)

        # Compute GMM
        x_extended = np.concatenate([x - 2, x, x + 2])

        res += (
            weight / (2 * np.pi) * np.exp(-0.5 * ((x_extended - mean) / variance) ** 2)
        )

    # fold the response to the size of x
    res = res[len(x) : 2 * len(x)] + res[2 * len(x) :] + res[: len(x)]

    # Normalize to [-1, 1]
    res = 2 * (res - res.min()) / (res.max() - res.min()) - 1

    return res


def generate_piecewise_ic(x, n_pieces=None, seed=None):
    """Generate piecewise linear initial condition.
    Hints:
    1. Generate random breakpoints
    2. Create piecewise linear function
    3. Add occasional discontinuities
    """
    if seed is not None:
        np.random.seed(seed)

    if n_pieces is None:
        n_pieces = np.random.randint(3, 7)

    # Generate breakpoints
    breakpoints = np.sort(np.random.uniform(x.min(), x.max(), size=n_pieces))
    # Generate values at breakpoints
    values = np.random.uniform(-1, 1, size=n_pieces).tolist()
    values += [values[0]]
    # Create piecewise linear function with same start and end points (0)
    conditions = (
        [x < breakpoints[0]]
        + [
            np.logical_and(x >= breakpoints[i], x < breakpoints[i + 1])
            for i in range(n_pieces - 1)
        ]
        + [x >= breakpoints[-1]]
    )

    res = np.piecewise(x, conditions, values)
    return res


def generate_sawtooth_ic(x, n_sawteeth=None, seed=None):
    """Generate sawtooth initial condition.
    Hints:
    1. Generate random number of sawteeth if n_sawteeth is None
    2. Create sawtooth function with random amplitude and frequency
    """
    if seed is not None:
        np.random.seed(seed)

    if n_sawteeth is None:
        n_sawteeth = np.random.randint(2, 6)

    # Generate sawtooth function
    res = np.zeros_like(x)
    for i in range(1, n_sawteeth + 1):
        res += np.sin(2 * np.pi * i * x) / i

    # Normalize to [-1, 1]
    res = 2 * (res - res.min()) / (res.max() - res.min()) - 1

    return res


def allen_cahn_rhs(t, u, epsilon, x_grid):
    """TODO: Implement Allen-Cahn equation RHS:
    ∂u/∂t = Δu - (1/ε²)(u³ - u)
    """
    dx = x_grid[1] - x_grid[0]
    padding = 2

    u_pad = np.pad(u, padding, mode="wrap")

    # TODO: Compute Laplacian (Δu) with periodic boundary conditions
    u_x = np.gradient(u_pad, dx, axis=-1)
    u_xx = np.gradient(u_x, dx, axis=-1)[padding:-padding]
    u_x = u_x[padding:-padding]

    # TODO: Compute nonlinear term -(1/ε²)(u³ - u)
    nonlin = -(1 / epsilon**2) * (u**3 - u)
    # TODO: Return full RHS
    return u_xx + nonlin


def generate_dataset(n_samples, epsilon, x_grid, t_eval, ic_type="fourier", seed=None):
    """Generate dataset for Allen-Cahn equation."""
    if seed is not None:
        np.random.seed(seed)

    # Initialize dataset array
    # for each sample, we have 5 time points
    # each time point has len(x_grid) spatial points
    # each spatial point has 3 values (u(x, t), t, epsilon)
    dataset = np.zeros((n_samples, 5, len(x_grid), 3))

    t_eval = t_eval * epsilon**1.75
    # print(t_eval)

    # Generate samples
    for i in tqdm.trange(
        n_samples, desc=f"Generating {ic_type} dataset for ε={epsilon}"
    ):
        # Generate initial condition based on type
        if ic_type == "fourier":
            u0 = generate_fourier_ic(x_grid, seed=seed + i if seed else None)
        elif ic_type == "gmm":
            u0 = generate_gmm_ic(x_grid, seed=seed + i if seed else None)
        elif ic_type == "piecewise":
            u0 = generate_piecewise_ic(x_grid, seed=seed + i if seed else None)
        elif ic_type == "OOD":
            # TODO: Generate OOD initial condition
            u0 = generate_sawtooth_ic(x_grid, seed=seed + i if seed else None)
        else:
            raise ValueError(f"Unknown IC type: {ic_type}")

        # sample 5 random points in the domain for time evaluation
        time_eval_points = np.sort(np.random.choice(t_eval, 5, replace=False))

        # Solve PDE using solve_ivp
        sol = solve_ivp(
            allen_cahn_rhs,
            t_span=(t_eval[0], t_eval[-1]),
            y0=u0,
            t_eval=time_eval_points,
            args=(epsilon, x_grid),
            method="RK45",
            rtol=1e-6,
            atol=1e-6,
        )

        solution = sol.y.T

        epsilon_values = np.ones(solution.shape) * epsilon
        time_values = time_eval_points.reshape(5, 1).repeat(len(x_grid), axis=-1)

        dataset[i] = np.stack([solution, time_values, epsilon_values], axis=-1)

    return dataset


def main():
    """Generate all datasets."""
    # Set up spatial grid
    nx = 128
    x_grid = np.linspace(-1, 1, nx)

    # Set up temporal grid
    t_eval = np.array([0.0, 0.25, 0.50, 0.75, 1.0])
    # t_eval = np.linspace(0, 1, 100)

    # Parameters for datasets
    epsilons = [0.1, 0.05, 0.02]  # Different epsilon values
    n_train = 500  # Number of training samples per configuration
    n_test = 200  # Number of test samples
    base_seed = 42  # For reproducibility

    out_dir = "data"
    os.makedirs(out_dir, exist_ok=True)

    for ic in ["fourier", "gmm", "piecewise"]:
        datasets_train = []
        datasets_test = []
        for epsilon in epsilons:
            # print(f"Generating datasets for IC: {ic}, epsilon: {epsilon}")

            # Generate training datasets for each epsilon and IC type
            dataset = generate_dataset(
                n_train, epsilon, x_grid, t_eval, ic_type=ic, seed=base_seed
            )

            datasets_train.append(dataset)

            # Generate standard test dataset
            dataset = generate_dataset(
                n_test, epsilon, x_grid, t_eval, ic_type=ic, seed=base_seed + 100
            )

            datasets_test.append(dataset)

        np.save(f"{out_dir}/train_allen_cahn_{ic}.npy", np.concatenate(datasets_train))
        np.save(f"{out_dir}/test_allen_cahn_{ic}.npy", np.concatenate(datasets_test))
    # TODO: Generate OOD test datasets (high frequency, sharp transitions)
    # Save all datasets using np.save

    OOD_epsilons = [0.15, 0.1, 0.005, 0.001]
    OOD_datasets = []
    for epsilon in OOD_epsilons:
        dataset = generate_dataset(
            n_test, epsilon, x_grid, t_eval, ic_type="OOD", seed=base_seed + 200
        )

        OOD_datasets.append(dataset)

    np.save(f"{out_dir}/OOD_allen_cahn.npy", np.concatenate(OOD_datasets))


if __name__ == "__main__":
    main()
