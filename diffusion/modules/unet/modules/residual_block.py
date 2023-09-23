import torch
from torch import nn

from .group_norm_32 import GroupNorm32


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, d_t_emb: int, out_channels=None):
        super().__init__()

        if out_channels is None:
            out_channels = channels

        self.in_layers = nn.Sequential(
            GroupNorm32(32, channels),
            nn.SiLU(),
            nn.Conv2d(channels, out_channels, 3, padding=1),
        )

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_t_emb, out_channels),
        )

        self.out_layers = nn.Sequential(
            GroupNorm32(32, out_channels),
            nn.SiLU(),
            nn.Dropout(0.),
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
        )

        if out_channels == channels:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Conv2d(channels, out_channels, 1)

    def forward(self, x: torch.Tensor, time_embedding: torch.Tensor):
        h = self.in_layers(x)

        time_embedding = self.emb_layers(time_embedding).type(h.dtype)

        h = h + time_embedding[:, :, None, None]

        return self.skip(x) + self.out_layers(h)
