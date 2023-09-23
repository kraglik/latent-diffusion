import torch
from torch import nn

from .modules import (
    AttentionBlock,
    ODEBlock,
    UpSample, ResidualBlock,
)


class Decoder(nn.Module):
    def __init__(
        self,
        channels: int,
        channel_multipliers: list[int],
        out_channels: int,
        z_channels: int,
        residual_blocks_count: int = 2,
        block: type = ResidualBlock,
        neural_ode_end_time: float = 1.0,
    ) -> None:
        super().__init__()

        num_resolutions = len(channel_multipliers)

        channels_list = [m * channels for m in channel_multipliers]

        channels = channels_list[-1]

        self.conv_in = nn.Conv2d(z_channels, channels, 3, stride=1, padding=1)

        self.mid = nn.Sequential(
            ResidualBlock(channels, channels),
            AttentionBlock(channels),
            ResidualBlock(channels, channels),
            # AttentionBlock(channels),
        )

        self.decode = nn.ModuleList()

        for i, next_channels in enumerate(reversed(channels_list)):
            self.decode.extend(
                [
                    *(
                        [block(channels, next_channels)]
                        + [
                            block(next_channels, next_channels)
                            for _ in range(residual_blocks_count - 1)
                        ]
                    ),
                    (UpSample(next_channels) if i < num_resolutions - 1 else nn.Identity()),
                ]
            )
            channels = next_channels

        self.out = nn.Sequential(
            # ResidualBlock(channels, channels),
            nn.GroupNorm(num_groups=4, num_channels=channels, eps=1e-6),
            nn.SiLU(),
            nn.Conv2d(channels, out_channels, 5, stride=1, padding=2)
        )

    def forward(self, z: torch.Tensor):
        h = self.conv_in(z)
        h = self.mid(h)

        for i, block in enumerate(self.decode):
            h = block(h)

        img = self.out(h)

        return img
