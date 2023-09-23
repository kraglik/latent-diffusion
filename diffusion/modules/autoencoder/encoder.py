import torch
from torch import nn

from .modules import (
    AttentionBlock,
    ODEBlock,
    DownSample,
    ResidualBlock,
)


class Encoder(nn.Module):
    def __init__(
        self,
        channels: int,
        channel_multipliers: list[int],
        in_channels: int,
        z_channels: int,
        residual_blocks_count: int = 2,
        block: type = ResidualBlock,
        neural_ode_end_time: float = 1.0,
    ) -> None:

        super().__init__()

        n_resolutions = len(channel_multipliers)

        self.conv_in = nn.Conv2d(in_channels, channels, 3, stride=1, padding=1)

        channels_list = [m * channels for m in [1] + channel_multipliers]

        self.encode = nn.ModuleList()

        for i in range(n_resolutions):
            self.encode.extend(
                [
                    *(
                        [block(channels_list[i], channels_list[i + 1])]
                        + [
                            block(channels_list[i + 1], channels_list[i + 1])
                            for _ in range(residual_blocks_count - 1)
                        ]
                    ),
                    (DownSample(channels_list[i + 1]) if i != n_resolutions - 1 else nn.Identity()),
                ]
            )

        channels = channels_list[-1]

        self.mid = nn.Sequential(
            ResidualBlock(channels, channels),
            AttentionBlock(channels),
            ResidualBlock(channels, channels)
        )

        self.out = nn.Sequential(
            nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-6),
            nn.SiLU(),
            nn.Conv2d(channels, 2 * z_channels, 3, stride=1, padding=1)
        )

    def forward(self, img: torch.Tensor):
        x = self.conv_in(img)

        for block in self.encode:
            x = block(x)

        x = self.mid(x)
        x = self.out(x)

        return x
