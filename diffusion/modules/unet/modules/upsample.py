import torch
from torch import nn
import torch.nn.functional as fun


class UpSample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()

        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor):
        x = fun.interpolate(x, scale_factor=2, mode="nearest")

        return self.conv(x)
