from torch import nn
from torch.nn import functional as fun


class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=1)

        return fun.silu(gate) * x
