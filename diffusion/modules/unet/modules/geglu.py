import torch
from torch import nn
from torch.nn import functional as fun


class GeGLU(nn.Module):
    def __init__(self, d_in: int, d_out: int):
        super().__init__()

        self.map_out = nn.Linear(d_in, d_out * 2)

    def forward(self, x: torch.Tensor):
        x, gate = self.map_out(x).chunk(2, dim=-1)
        return x * fun.gelu(gate)
