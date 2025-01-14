import torch


def relative_l2_error(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.norm(output - target) / torch.norm(target)


def fourier_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Fourier transformed loss function.
    """
    return torch.mean(
        (torch.fft.fft(output, dim=-1) - torch.fft.fft(target, dim=-1)).real ** 2
    )
