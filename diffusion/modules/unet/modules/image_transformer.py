import torch
from torch import nn

from diffusion.utils.positional_encoding import PositionalEncodingPermute2D
from .transformer_block import TransformerBlock


class ImageTransformer(nn.Module):
    def __init__(self, channels: int, num_heads: int, num_layers: int):
        super().__init__()
        self.norm = torch.nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-6, affine=True)
        self.map_in = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)

        self.positional_encoding = PositionalEncodingPermute2D(channels=channels)

        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(channels, num_heads, channels // num_heads) for _ in range(num_layers)]
        )

        self.map_out = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor, cond: torch.Tensor):
        b, c, h, w = x.shape
        x_in = x

        x = self.normalize(x)
        x = self.map_in(x)

        x = x + self.positional_encoding(x)

        x = x.permute(0, 2, 3, 1).reshape(b, h * w, c)

        for block in self.transformer_blocks:
            x = block(x, cond)

        x = x.view(b, h, w, c).permute(0, 3, 1, 2)
        x = self.map_out(x)

        return x + x_in

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)

        return x
