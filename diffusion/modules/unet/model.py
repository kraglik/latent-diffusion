import math
from typing import Optional

import torch
from torch import nn

from .modules import (
    ImageTransformer,
    GroupNorm32,
    ResidualBlock,
    UpSample,
    DownSample,
)


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        channels: int,
        # conditional_channels: int,
        attention_levels: list[int],
        channel_multipliers: list[int],
        num_unet_blocks: int,
        transformer_heads: int,
        transformer_layers: int = 1,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.channels = channels

        levels = len(channel_multipliers)

        time_embedding_dims = channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(channels, time_embedding_dims),
            nn.SiLU(),
            nn.Linear(time_embedding_dims, time_embedding_dims),
        )

        self.in_conv = nn.Conv2d(in_channels, channels, 3, padding=1)

        self.input_blocks = nn.ModuleList()

        channels_list = [channels * m for m in channel_multipliers]
        input_channels = [channels]

        for i in range(levels):
            residual_blocks = [
                ResidualBlock(
                    channels if j == 0 else channels_list[i],
                    time_embedding_dims,
                    out_channels=channels_list[i],
                )
                for j in range(num_unet_blocks)
            ]
            transformer_blocks = None
            channels = channels_list[i]

            if i in attention_levels:
                transformer_blocks = [
                    ImageTransformer(channels, transformer_heads, transformer_layers)
                    for _ in range(num_unet_blocks)
                ]

            self.input_blocks.append(
                UNetBlock(
                    residual_blocks,
                    transformer_blocks,
                    nn.Identity() if i == levels - 1 else DownSample(channels),
                )
            )
            input_channels.append(channels)

        self.middle_block_residual_1 = ResidualBlock(channels, time_embedding_dims)
        self.middle_block_transformer = ImageTransformer(channels, transformer_heads, transformer_layers)
        self.middle_block_residual_2 = ResidualBlock(channels, time_embedding_dims)

        self.output_blocks = nn.ModuleList([])

        for i in reversed(range(levels)):
            layer_input_channels = input_channels.pop()
            residual_blocks = [
                ResidualBlock(
                    (channels if j == 0 else channels_list[i]) + layer_input_channels,
                    time_embedding_dims,
                    out_channels=channels_list[i],
                )
                for j in range(num_unet_blocks + 1)
            ]
            channels = channels_list[i]

            transformer_blocks = None
            
            if i in attention_levels:
                transformer_blocks = [
                    ImageTransformer(channels, transformer_heads, transformer_layers)
                    for _ in range(num_unet_blocks)
                ]
                
            self.output_blocks.append(
                UNetBlock(
                    residual_blocks,
                    transformer_blocks,
                    nn.Identity() if i == 0 else UpSample(channels),
                )
            )

        self.out = nn.Sequential(
            GroupNorm32(32, channels),
            nn.SiLU(),
            nn.Conv2d(channels, out_channels, 3, padding=1),
        )

    def forward(
        self,
        x: torch.Tensor,
        time_steps: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        x_input_block = []

        time_embedding = self.time_step_embedding(time_steps)
        time_embedding = self.time_embed(time_embedding)

        x = self.in_conv(x)

        for module in self.input_blocks:
            out_xs, x = module(x, time_embedding, cond)
            x_input_block.append(out_xs)

        x = self.middle_block_residual_1(x, time_embedding)
        x = self.middle_block_transformer(x, cond)
        x = self.middle_block_residual_2(x, time_embedding)

        for module in self.output_blocks:
            xs_in = x_input_block.pop()
            _, x = module(x, time_embedding, cond, xs_in=xs_in)

        return self.out(x)

    def time_step_embedding(self, time_steps: torch.Tensor, max_period: int = 10000):
        half = self.channels // 2

        frequencies = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=time_steps.device)

        args = time_steps[:, None].float() * frequencies[None]

        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class UNetBlock(nn.Module):
    def __init__(
        self,
        residual_blocks: list[ResidualBlock],
        image_transformers: Optional[list[ImageTransformer]],
        sample: nn.Module,
    ) -> None:
        super().__init__()

        self.residual_blocks = nn.ModuleList(residual_blocks)
        self.image_transformers = nn.ModuleList(image_transformers) if image_transformers else None
        self.sample = sample

    def forward(
        self,
        x: torch.Tensor,
        time_embedding: torch.Tensor,
        cond: torch.Tensor,
        xs_in: Optional[list[torch.Tensor]] = None,
    ) -> (torch.Tensor, torch.Tensor):
        out_xs: list[torch.Tensor] = []

        if self.image_transformers is None:
            for block in self.residual_blocks:
                if xs_in is not None:
                    x = torch.cat([x, xs_in.pop()], dim=1)

                x = block(x, time_embedding)
                out_xs.append(x)

        else:
            for block, transformer in zip(self.residual_blocks, self.image_transformers):
                if xs_in is not None:
                    x = torch.cat([x, xs_in.pop()], dim=1)

                x = block(x, time_embedding)
                x = transformer(x, cond)
                out_xs.append(x)

        return out_xs, self.sample(x)
