import torch
from torch import nn

from .swiglu import SwiGLU


class MLP(nn.Module):
    def __init__(self, d_model: int, d_mult: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            SwiGLU(d_model, d_model * d_mult),
            nn.Dropout(0.),
            nn.Linear(d_model * d_mult, d_model)
        )

    def forward(self, x: torch.Tensor):
        return self.net(x)
