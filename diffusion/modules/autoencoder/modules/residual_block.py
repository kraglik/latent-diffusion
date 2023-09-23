import torch
from torch import nn

from .depthwise_convolution import DepthWiseSeparableConvolution


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_groups: int = 4):
        super().__init__()

        self.block = nn.Sequential(
            nn.GroupNorm(num_groups=4, num_channels=in_channels, eps=1e-6),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1),
            nn.GroupNorm(num_groups=4, num_channels=out_channels, eps=1e-6),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1),
        )

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor):
        return self.shortcut(x) + self.block(x)
