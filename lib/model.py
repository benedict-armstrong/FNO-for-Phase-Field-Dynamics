import torch
import torch.nn as nn

from lib.layers import ResidualBlock, SpectralConv1d


class FNO1d(nn.Module):
    def __init__(self, modes: int, width: int, layers: int = 4):
        """
        The overall network. It contains `layers` layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        input: the solution of the initial condition and location (a(x), x, t)
        input shape: (batchsize, x=s, c=3)
        output: the solution of a later timestep
        output shape: (batchsize, x=s, c=1)
        """
        super(FNO1d, self).__init__()
        self.modes = modes
        self.width = width
        self.padding = 1  # pad the domain if input is non-periodic
        self.last_layer_width = 32

        self.linear_p = nn.Linear(
            3, self.width
        )  # input channel is 2: (u0(x), x, t) --> GRID IS INCLUDED!

        self.spectral_layers = nn.ModuleList(
            [SpectralConv1d(self.width, self.width, self.modes) for _ in range(layers)]
        )

        self.linear_conv_layers = nn.ModuleList(
            [ResidualBlock(self.width) for _ in range(layers)]
        )

        self.linear_q = nn.Linear(self.width, self.last_layer_width)

        self.output_layer = nn.Linear(self.last_layer_width, 1)

        self.last_activation = torch.nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def fourier_layer(self, x, time_delta, epsilons):
        for f, c in zip(self.spectral_layers, self.linear_conv_layers):
            x = f(x, time_delta, epsilons) + c(x, time_delta, epsilons)
            x = self.dropout(x)
        return x

    def forward(
        self, x: torch.Tensor, time_delta: torch.Tensor, epsilons: torch.Tensor
    ):
        x = self.linear_p(x)
        # swap the channel dimension to the last dimension
        x = x.permute(0, 2, 1)
        x = self.fourier_layer(x, time_delta, epsilons)
        x = x.permute(0, 2, 1)
        x = self.linear_q(x)
        x = self.last_activation(x)
        x = self.output_layer(x)
        return x
