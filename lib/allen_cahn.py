import json
import os
from matplotlib import pyplot as plt
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


def generate_piecewise_ic(x, n_pieces=None, seed=None, discontinuities=None):
    """Generate piecewise linear initial condition.
    Hints:
    1. Generate random breakpoints
    2. Create piecewise linear function
    3. Add occasional discontinuities
    """
    if seed is not None:
        np.random.seed(seed)

    if n_pieces is None:
        n_pieces = np.random.randint(3, 6)

    if discontinuities is None:
        discontinuities = np.random.randint(0, 2)

    # Generate breakpoints
    breakpoints = np.sort(np.random.uniform(x[0], x[-1], n_pieces))
    breakpoints = np.concatenate([[x[0]], breakpoints, [x[-1]]])

    # Generate random y values at breakpoints
    y_values = np.random.uniform(-1, 1, n_pieces)

    # add boundary value to incorporate the periodic boundary condition (first/last value should be linear between the last and first breakpoints)
    b_dx_1 = abs(breakpoints[0] - breakpoints[1])
    b_dx_2 = abs(breakpoints[-1] - breakpoints[-2])
    dy = y_values[0] - y_values[-1]

    first_value = y_values[0] - dy * b_dx_1 / (b_dx_1 + b_dx_2)
    last_value = y_values[-1] + dy * b_dx_2 / (b_dx_1 + b_dx_2)

    y_values = np.concatenate([[first_value], y_values, [last_value]])

    # Interpolate
    f = np.interp(x, breakpoints, y_values)

    # Add occasional discontinuities
    for _ in range(discontinuities):
        # Randomly select a segment
        idx = np.random.randint(1, len(breakpoints) - 1)
        f = np.where(
            (x >= breakpoints[idx - 1]) & (x < breakpoints[idx]),
            f + np.random.uniform(-0.5, 0.5),
            f,
        )

    # make sure the response is normalized to [-1, 1]
    f = 2 * (f - f.min()) / (f.max() - f.min()) - 1

    return f


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
    dataset = np.zeros((n_samples, 5, len(x_grid)))
    times = np.zeros((n_samples, 5))
    epsilons = np.zeros((n_samples))

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
            u0 = generate_piecewise_ic(
                x_grid,
                seed=seed + i if seed else None,
                discontinuities=np.random.randint(0, 1),
            )
        elif ic_type == "OOD":
            # TODO: Generate OOD initial condition
            if i % 3 == 0:
                u0 = generate_fourier_ic(
                    x_grid,
                    seed=seed + i if seed else None,
                    # n_modes=np.random.randint(1, 3),
                )
            elif i % 3 == 1:
                u0 = generate_gmm_ic(
                    x_grid,
                    seed=seed + i if seed else None,
                    n_components=np.random.randint(2, 8),
                )
            else:
                u0 = generate_piecewise_ic(
                    x_grid,
                    seed=seed + i if seed else None,
                    discontinuities=np.random.randint(0, 3),
                )
        elif ic_type == "HF":
            u0 = generate_fourier_ic(
                x_grid,
                seed=seed + i if seed else None,
                n_modes=np.random.randint(6, 10),
            )
        else:
            raise ValueError(f"Unknown IC type: {ic_type}")

        # sample 5 random points in the domain for time evaluation
        time_eval_points = np.sort(np.random.choice(t_eval, 5, replace=False))

        if time_eval_points[0] != 0:
            time_eval_points[0] = 0

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

        dataset[i] = solution
        times[i] = time_eval_points
        epsilons[i] = epsilon

    return dataset, times, epsilons


def plot_time_eval(train_t_eval, test_t_eval):
    t_end = train_t_eval[-1]
    plt.figure(figsize=(6, 1))
    plt.xlim(-0.1, t_end + 0.1)
    plt.ylim(-0.1, 0.1)

    # turn of y-axis
    plt.yticks([])

    # remove frame
    plt.box(False)

    # remove padding around data so that the frame is tight around the data
    plt.subplots_adjust(left=0.05, right=0.95, top=0.6, bottom=0.4)

    plt.xticks([0, t_end / 2, t_end])
    plt.plot(train_t_eval, np.zeros_like(train_t_eval), "o")
    plt.plot(test_t_eval, np.zeros_like(test_t_eval), "x")
    plt.savefig("figures/eval_points.png", dpi=300)


def create_dataset(
    epsilons,
    time_eval,
    ic,
    n,
    x_grid,
    seed,
    save_path,
):
    data = {
        "dataset": [],
        "time": [],
        "epsilon": [],
    }

    for epsilon in epsilons:
        time_scaling = epsilon

        time_grid = time_eval * time_scaling

        print(f"Generating datasets for ε={epsilon}, IC={ic} at {time_grid}")

        # Generate training datasets for each epsilon and IC type
        dataset, ts, eps = generate_dataset(
            n,
            epsilon,
            x_grid,
            time_grid,
            ic_type=ic,
            seed=seed,
        )

        data["dataset"].append(dataset)
        data["time"].append(ts)
        data["epsilon"].append(eps)

    np.savez(
        save_path,
        data=np.concatenate(data["dataset"]),
        time=np.concatenate(data["time"]),
        epsilon=np.concatenate(data["epsilon"]),
    )

    return data


def main():
    """Generate all datasets."""
    # Set up spatial grid
    nx = 128
    x_grid = np.linspace(-1, 1, nx)

    # Set up temporal grid
    t_end = 1
    train_t_eval = np.logspace(0, t_end, 30, base=3) - 1
    train_t_eval = train_t_eval / train_t_eval[-1] * t_end
    test_t_eval = np.linspace(0, t_end, 5)

    print(f"Train t_eval: {train_t_eval}")
    print(f"Test t_eval: {test_t_eval}")

    plot_time_eval(train_t_eval, test_t_eval)

    config = {}

    # Parameters for datasets
    config["epsilons"] = [0.1, 0.05, 0.02]  # Different epsilon values
    config["ic_types"] = ["fourier", "gmm", "piecewise"]  # Different IC types
    config["n_train"] = 400  # Number of training samples per configuration
    config["n_test"] = 100  # Number of test samples
    config["base_seed"] = 42  # For reproducibility
    config["time_scaling"] = {}
    config["train_time"] = train_t_eval.tolist()
    config["test_time"] = test_t_eval.tolist()

    out_dir = "data"
    os.makedirs(out_dir, exist_ok=True)

    # for ic in config["ic_types"]:
    # create_dataset(
    #     config["epsilons"],
    #     train_t_eval,
    #     ic,
    #     config["n_train"],
    #     x_grid,
    #     config["base_seed"],
    #     f"{out_dir}/train_allen_cahn_{ic}.npz",
    # )

    # create_dataset(
    #     config["epsilons"],
    #     test_t_eval,
    #     ic,
    #     config["n_test"],
    #     x_grid,
    #     config["base_seed"] + 100,
    #     f"{out_dir}/test_allen_cahn_{ic}.npz",
    # )

    config["OOD_epsilons"] = [0.2, 0.15, 0.075, 0.035, 0.005]
    create_dataset(
        config["OOD_epsilons"],
        test_t_eval,
        "OOD",
        config["n_test"],
        x_grid,
        config["base_seed"] + 200,
        f"{out_dir}/OOD_allen_cahn.npz",
    )

    create_dataset(
        config["OOD_epsilons"],
        test_t_eval,
        "OOD",
        30,
        x_grid,
        config["base_seed"] + 123,
        f"{out_dir}/OOD_fine_tune_allen_cahn.npz",
    )

    create_dataset(
        config["epsilons"],
        test_t_eval,
        "OOD",
        config["n_test"],
        x_grid,
        config["base_seed"] + 300,
        f"{out_dir}/OOD_IC_allen_cahn.npz",
    )

    create_dataset(
        config["epsilons"],
        test_t_eval,
        "HF",
        20,
        x_grid,
        config["base_seed"] + 12,
        f"{out_dir}/OOD_HF_allen_cahn.npz",
    )

    create_dataset(
        [0.005, 0.001, 0.0001],
        test_t_eval,
        "fourier",
        20,
        x_grid,
        config["base_seed"] + 543,
        f"{out_dir}/OOD_LOW_E_allen_cahn.npz",
    )

    # save config to file
    with open(f"{out_dir}/config.json", "w") as f:
        json.dump(config, f)


if __name__ == "__main__":
    main()
