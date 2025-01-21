from typing import Tuple
import torch
import tqdm
from torch.utils.data import DataLoader
import pandas as pd


def relative_l2_error(
    output: torch.Tensor, target: torch.Tensor, dim=None
) -> torch.Tensor:
    return torch.norm(output - target, dim=dim) / torch.norm(target, dim=dim)


def fourier_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Fourier transformed loss function.
    """
    return torch.mean(
        (torch.fft.fft(output, dim=-1) - torch.fft.fft(target, dim=-1)).real ** 2
    )


def calculate_errors(
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
) -> Tuple[pd.DataFrame, float]:
    errors = {
        "dt": [],
        "l2_error": [],
        "epsilon": [],
    }

    data_loader = DataLoader(dataset, batch_size=64, shuffle=True)

    model.eval()
    progress_bar = tqdm.tqdm(data_loader)

    with torch.no_grad():
        average = 0.0
        for i, (dt, eps, input, target) in enumerate(progress_bar):
            prediction = model(input, dt, eps).squeeze(-1)

            loss = relative_l2_error(prediction, target, dim=-1)

            for j in range(len(dt)):
                errors["dt"].append(dt[j].item())
                errors["l2_error"].append(loss[j].item())
                errors["epsilon"].append(eps[j].item())

            average += loss.sum().item()

        average /= len(dataset)

    return pd.DataFrame(errors), average
