import torch
import torch.nn as nn


class FILM(torch.nn.Module):
    def __init__(self, channels, use_bn=True):
        super(FILM, self).__init__()
        self.channels = channels

        self.inp2scale = nn.Linear(in_features=1, out_features=channels, bias=True)
        self.inp2bias = nn.Linear(in_features=1, out_features=channels, bias=True)

        self.inp2scale.weight.data.fill_(0)
        self.inp2scale.bias.data.fill_(1)
        self.inp2bias.weight.data.fill_(0)
        self.inp2bias.bias.data.fill_(0)

        if use_bn:
            self.norm = nn.BatchNorm1d(channels)
        else:
            self.norm = nn.Identity()

    def forward(self, x: torch.Tensor, time: torch.Tensor):
        x = self.norm(x)
        time = time.reshape(-1, 1).type_as(x)
        scale = self.inp2scale(time)
        bias = self.inp2bias(time)
        scale = scale.unsqueeze(2).expand_as(x)
        bias = bias.unsqueeze(2).expand_as(x)

        return x * scale + bias


class ResidualBlock(nn.Module):
    def __init__(self, channels, use_bn=True):
        super(ResidualBlock, self).__init__()

        self.channels = channels

        self.convolution1 = torch.nn.Conv1d(
            in_channels=self.channels,
            out_channels=self.channels,
            kernel_size=5,
            padding=2,
            padding_mode="circular",
        )
        self.convolution2 = torch.nn.Conv1d(
            in_channels=self.channels,
            out_channels=self.channels,
            kernel_size=11,
            padding=5,
            padding_mode="circular",
        )

        self.batch_norm1 = FILM(self.channels, use_bn)
        self.batch_norm2 = FILM(self.channels, use_bn)
        self.batch_norm3 = FILM(self.channels, use_bn)

        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)

        self.act = torch.nn.ReLU()

    def forward(self, x: torch.Tensor, time: torch.Tensor):
        out = self.convolution1(x)
        out = self.batch_norm1(out, time)
        out = self.act(out)
        # out = self.dropout1(out)

        out = self.convolution2(out)
        out = self.batch_norm2(out, time)
        out = self.act(out)
        # out = self.dropout2(out)

        return self.batch_norm3(x, time) + out


class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes, use_bn=True):
        super(SpectralConv1d, self).__init__()
        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes
        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale
            * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat)
        )

        self.batch_norm1 = FILM(self.in_channels, use_bn)

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x, time_delta):
        batchsize = x.shape[0]
        # x.shape == [batch_size, in_channels, number of grid points]
        # hint: use torch.fft library torch.fft.rfft
        # use DFT to approximate the fourier transform
        # Compute Fourier coefficients
        x_ft = torch.fft.rfft(x)
        # Multiply relevant Fourier modes
        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            x.size(-1) // 2 + 1,
            device=x.device,
            dtype=torch.cfloat,
        )
        out_ft[:, :, : self.modes1] = self.compl_mul1d(
            x_ft[:, :, : self.modes1], self.weights1
        )

        # Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))

        x = self.batch_norm1(x, time_delta)

        return x
