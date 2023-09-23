import torch
from torch import nn


from torchdiffeq import odeint, odeint_adjoint
import torchdyn
from torchdyn.numerics import odeint as torchdyn_odeint, odeint_hybrid
from torchdyn.nn import DepthCat

from .depthwise_convolution import DepthWiseSeparableConvolution


class ODEBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        end_time: float = 1.0,
        adjoint: bool = True,
        error_tolerance: float = 5e-3,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.adjoint = adjoint

        self.linear_mapping = nn.Conv2d(in_channels, out_channels, 1)
        self.time = nn.Parameter(torch.tensor([0.0, end_time]).float(), requires_grad=False)

        self.func = ODEFunction(out_channels)
        self.error_tolerance = error_tolerance

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        integrator = torchdyn_odeint

        time = torch.linspace(0.0, 1.0, 5)

        x = integrator(
            f=self.func,
            x=self.linear_mapping(x),
            t_span=time,
            rtol=self.error_tolerance,
            atol=self.error_tolerance,
            solver="euler",
        )

        return x[1][-1]


class ODEFunction(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv_1 = TimeAugmentedConv2d(channels, channels, 3, stride=1, padding=1)
        self.norm_1 = nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-6)
        self.act_1 = nn.SiLU()

        self.norm_2 = nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-6)
        self.act_2 = nn.SiLU()
        self.conv_2 = nn.Conv2d(channels, channels, 1)

    def forward(self, t: torch.Tensor | float, x: torch.Tensor) -> None:
        x = self.norm_1(x)
        x = self.act_1(x)
        x = self.conv_1(x, t)

        x = self.norm_2(x)
        x = self.act_2(x)
        x = self.conv_2(x)

        return x


class TimeAugmentedConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=0,
        time_encoding_frequencies: int = 6,
        max_time_encoding_period: int = 5000,
    ):
        super(TimeAugmentedConv2d, self).__init__()

        encoding_weights = torch.arange(0, time_encoding_frequencies)
        encoding_weights = 1 / (max_time_encoding_period) ** (encoding_weights * 2 / time_encoding_frequencies)

        self.register_buffer("encoding_weights", encoding_weights.reshape(1, -1, 1, 1))

        self._layer = DepthWiseSeparableConvolution(
            in_channels + 1 + 2 * time_encoding_frequencies,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor | float) -> torch.Tensor:
        original_time_encoding = torch.ones_like(x[:, :1, :, :]) * t
        sine_time_encoding = torch.sin(self.encoding_weights * original_time_encoding)
        cosine_time_encoding = torch.cos(self.encoding_weights * original_time_encoding)

        time_encoding = torch.cat(
            [
                original_time_encoding,
                sine_time_encoding,
                cosine_time_encoding,
            ],
            dim=1,
        )

        time_encoded_x = torch.cat([x, time_encoding], 1)
        return self._layer(time_encoded_x)
