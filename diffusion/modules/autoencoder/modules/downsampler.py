import torch
from torch import nn
from torch.nn import functional as fun


class DownSample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()

        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=0)

    def forward(self, x: torch.Tensor):
        x = fun.pad(x, (0, 1, 0, 1), mode="constant", value=0)

        return self.conv(x)
