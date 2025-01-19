from typing import List, Tuple
from matplotlib import pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset


class PDEDatasetAll2All(Dataset):
    def __init__(
        self,
        path: str,
        device: str = "cpu",
        time_pairs: List[Tuple[int, int]] = None,
    ):
        super(PDEDatasetAll2All, self).__init__()

        self.numpy_data = np.load(path)

        self.data = torch.tensor(self.numpy_data["data"]).type(torch.float32).to(device)

        self.epsilons = (
            torch.tensor(self.numpy_data["epsilon"]).type(torch.float32).to(device)
        )
        self.time_grid = (
            torch.tensor(self.numpy_data["time"]).type(torch.float32).to(device)
        )

        self.samples = self.data.shape[0]
        self.time_steps = self.data.shape[1]
        self.spacial_res = self.data.shape[2]

        if time_pairs is not None:
            self.time_pairs = time_pairs
        else:
            # Precompute all possible (t_initial, t_final) pairs within the specified range.
            self.time_pairs = [
                (i, j)
                for i in range(0, self.time_steps)
                for j in range(i + 1, self.time_steps)
            ]

        self.len_times = len(self.time_pairs)

        self.total_samples = self.len_times * self.samples

        self.x_grid = torch.linspace(-1, 1, self.spacial_res).to(device)

        # Compute mean and std of the data
        self.mean = self.data.mean()
        self.std = self.data.std()

    def __len__(self):
        return self.total_samples

    def __getitem__(self, index):
        sample_idx = index // self.len_times
        sample_eps = self.epsilons[sample_idx]
        sample_time_grid = self.time_grid[sample_idx]
        time_pair_idx = index % self.len_times
        t_inp, t_out = self.time_pairs[time_pair_idx]
        time_delta = torch.abs(sample_time_grid[t_out] - sample_time_grid[t_inp]).item()
        epsilons = sample_eps.item()

        input = self.data[sample_idx, t_inp]

        epsilon_tensor = torch.ones(input.shape) * sample_eps
        time_tensor = torch.ones(input.shape) * time_delta

        input = torch.stack([input, self.x_grid, epsilon_tensor, time_tensor], axis=-1)

        target = self.data[sample_idx, t_out]

        return float(time_delta), epsilons, input, target


if __name__ == "__main__":
    dataset = PDEDatasetAll2All("data/test_allen_cahn_fourier.npz")
    dataset[0]
    raw_data = np.load("data/test_allen_cahn_fourier.npz")

    # torch.manual_seed(0)
    # np.random.seed(0)

    # sample_id = 0

    # trajectories = []
    # start = dataset.len_times * sample_id
    # for i in range(start, start + dataset.len_times):
    #     print(i)
    #     dt, sample, target, times = dataset[i]
    #     x = np.linspace(0, 1, sample.shape[-1])

    #     if times[0] not in trajectories:
    #         plt.plot(x, sample[0], label=f"$u(t = {times[0]})$")
    #         trajectories.append(times[0])
    #     if times[1] not in trajectories:
    #         plt.plot(x, target[0], label=f"$u_t(t = {times[1]})$")
    #         trajectories.append(times[1])

    # plt.plot(x, raw_data[sample_id, 0], label="$u(t = 0)$")
    # plt.plot(x, raw_data[sample_id, 1], label="$u(t = 0.25)$")
    # plt.plot(x, raw_data[sample_id, 2], label="$u(t = 0.5)$")
    # plt.plot(x, raw_data[sample_id, 3], label="$u(t = 0.75)$")
    # plt.plot(x, raw_data[sample_id, 4], label="$u(t = 1)$")
    # plt.grid(True, which="both", ls=":")
    # plt.legend()

    # plt.show()
