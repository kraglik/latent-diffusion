import torch
from torch import nn
from torch.nn import functional as fun

from .attention_block import AttentionBlock


class UpSample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()

        self.conv_1 = nn.Conv2d(channels, channels, 3, padding=1)
        # self.conv_2 = nn.Sequential(
        #     nn.GroupNorm(num_groups=3, num_channels=channels * 3, eps=1e-6),
        #     nn.SiLU(),
        #     nn.Conv2d(channels * 3, channels, 3, padding=1),
        # )
        #
        # self.folding_conv = nn.Conv2d(channels, channels, 4, stride=4)
        # self.attention_block = AttentionBlock(channels)
        # self.expanding_conv = nn.ConvTranspose2d(channels, channels, 4, stride=4)

    def forward(self, x: torch.Tensor, scale_factor: float = 2.0):
        interpolated = fun.interpolate(x, scale_factor=scale_factor, mode="nearest")

        x = self.conv_1(interpolated)

        return x

        # y = self.folding_conv(x)
        # y = self.attention_block(y)
        # y = self.expanding_conv(y)
        #
        # return interpolated + self.conv_2(torch.cat([interpolated, x, y], dim=1))
