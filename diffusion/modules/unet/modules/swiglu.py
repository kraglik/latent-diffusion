from torch import nn
from torch.nn import functional as fun


class SwiGLU(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        self.proj = nn.Linear(in_channels, out_channels * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)

        return fun.silu(gate) * x
