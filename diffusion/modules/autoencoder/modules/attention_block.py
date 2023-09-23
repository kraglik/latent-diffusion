import torch
from torch import nn
from torch.nn import functional as fun

from .swiglu import SwiGLU


class AttentionBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=4, num_channels=channels, eps=1e-6)

        self.q_map = nn.Conv2d(channels, channels, 1)
        self.k_map = nn.Conv2d(channels, channels, 1)
        self.v_map = nn.Conv2d(channels, channels, 1)

        self.proj_out = nn.Conv2d(channels, channels, 1)

        self.mlp = nn.Sequential(
            nn.GroupNorm(num_groups=4, num_channels=channels, eps=1e-6),
            nn.Conv2d(channels, channels * 2, 1),
            SwiGLU(),
            nn.Conv2d(channels, channels, 1),
        )

        self.scale = channels ** -0.5

    def forward(self, x: torch.Tensor):
        x_norm = self.norm(x)

        q = self.q_map(x_norm)
        k = self.k_map(x_norm)
        v = self.v_map(x_norm)

        b, c, h, w = q.shape

        q = q.view(b, c, h * w)
        k = k.view(b, c, h * w)
        v = v.view(b, c, h * w)

        attn = torch.einsum('bci,bcj->bij', q, k) * self.scale
        attn = fun.softmax(attn, dim=2)

        out = torch.einsum('bij,bcj->bci', attn, v)

        out = out.view(b, c, h, w)
        # out = self.proj_out(out)
        out = self.mlp(out)

        return x + out
