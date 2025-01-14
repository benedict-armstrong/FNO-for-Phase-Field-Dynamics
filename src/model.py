import torch
import torch.nn as nn

from .layers import ResidualBlock, SpectralConv1d


class FNO1d(nn.Module):
    def __init__(self, modes, width):
        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        input: the solution of the initial condition and location (a(x), x)
        input shape: (batchsize, x=s, c=2)
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
        )  # input channel is 2: (u0(x), x, epsilon) --> GRID IS INCLUDED!

        self.spectral_layers = nn.ModuleList(
            [SpectralConv1d(self.width, self.width, self.modes) for _ in range(4)]
        )

        self.linear_conv_layers = nn.ModuleList(
            [ResidualBlock(self.width) for _ in range(4)]
        )

        self.linear_q = nn.Linear(self.width, self.last_layer_width)

        self.output_layer = nn.Linear(self.last_layer_width, 1)

        self.activation = torch.nn.Tanh()
        self.last_activation = torch.nn.ReLU()
        # self.dropout = nn.Dropout(0.2)

    def fourier_layer(self, x, time_delta):
        ##########################################
        # TODO: Implement the Fourier layer:
        ##########################################
        for f, c in zip(self.spectral_layers, self.linear_conv_layers):
            x = f(x, time_delta) + c(x, time_delta)
            x = self.activation(x)
            # x = self.dropout(x)
        return x

    def rescale(self, x: torch.Tensor):
        x_v = x[..., 0]
        scale = torch.max(torch.abs(x_v), dim=1).values

        x_rescaled = x_v / scale.unsqueeze(-1)
        x[..., :1] = x_rescaled.unsqueeze(-1)
        return x, scale

    def unscale(self, x: torch.Tensor, scale: torch.Tensor):
        return x.squeeze(-1) * scale.unsqueeze(-1)

    # def rescale(self, x: torch.Tensor):
    #     x_v = x[..., 0]
    #     min = torch.min(x_v, dim=1).values
    #     max = torch.max(x_v, dim=1).values
    #     x_rescaled = 2 * (x_v - min.unsqueeze(1)) / (max - min).unsqueeze(1) - 1
    #     x[..., :1] = x_rescaled.unsqueeze(-1)
    #     return x, min, max

    # def unscale(self, x: torch.Tensor, max: torch.Tensor, min: torch.Tensor):
    #     return (
    #         (x + 1).squeeze(-1) * (max - min).unsqueeze(1) / 2 + min.unsqueeze(1)
    #     ).unsqueeze(-1)

    def forward(self, x: torch.Tensor, time_delta: torch.Tensor):
        #################################################
        # TODO: Implement the forward method
        #        using the Fourier and the Linear layer:
        #################################################
        # swap the channel dimension to the last dimension
        # x, scale = self.rescale(x)
        x = self.linear_p(x)
        x = x.permute(0, 2, 1)
        x = self.fourier_layer(x, time_delta)
        x = x.permute(0, 2, 1)
        x = self.linear_q(x)
        x = self.last_activation(x)
        x = self.output_layer(x)
        # x = self.unscale(x, scale)
        return x
