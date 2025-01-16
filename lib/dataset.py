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

        self.data = torch.tensor(np.load(path)).type(torch.float32).to(device)

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
                for j in range(i, self.time_steps)
            ]

        self.len_times = len(self.time_pairs)

        self.total_samples = self.len_times * self.samples

        self.x_grid = torch.linspace(0, 1, self.spacial_res).to(device).reshape(1, -1)

        # Compute mean and std of the data
        self.mean = self.data.mean()
        self.std = self.data.std()

    def __len__(self):
        return self.total_samples

    def __getitem__(self, index):
        sample_idx = index // self.len_times
        time_pair_idx = index % self.len_times
        t_inp, t_out = self.time_pairs[time_pair_idx]

        input = self.data[sample_idx, t_inp]
        target = self.data[sample_idx, t_out]

        time_delta = abs((input[0, 1] - target[0, 1]).item())

        return float(time_delta), input, target[..., 0]


if __name__ == "__main__":
    dataset = PDEDatasetAll2All()
    raw_data = np.load("data/train_sol.npy")

    torch.manual_seed(0)
    np.random.seed(0)

    sample_id = 0

    trajectories = []
    start = dataset.len_times * sample_id
    for i in range(start, start + dataset.len_times):
        print(i)
        dt, sample, target, times = dataset[i]
        x = np.linspace(0, 1, sample.shape[-1])

        if times[0] not in trajectories:
            plt.plot(x, sample[0], label=f"$u(t = {times[0]})$")
            trajectories.append(times[0])
        if times[1] not in trajectories:
            plt.plot(x, target[0], label=f"$u_t(t = {times[1]})$")
            trajectories.append(times[1])

    plt.plot(x, raw_data[sample_id, 0], label="$u(t = 0)$")
    plt.plot(x, raw_data[sample_id, 1], label="$u(t = 0.25)$")
    plt.plot(x, raw_data[sample_id, 2], label="$u(t = 0.5)$")
    plt.plot(x, raw_data[sample_id, 3], label="$u(t = 0.75)$")
    plt.plot(x, raw_data[sample_id, 4], label="$u(t = 1)$")
    plt.grid(True, which="both", ls=":")
    plt.legend()

    plt.show()
