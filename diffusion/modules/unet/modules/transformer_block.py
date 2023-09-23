import torch
from torch import nn

from .cross_attention import CrossAttention
from .mlp import MLP


class TransformerBlock(nn.Module):
    def __init__(self, dims_model: int, num_heads: int, dims_head: int) -> None:
        super().__init__()
        self.attn1 = CrossAttention(dims_model, num_heads, dims_head)
        self.norm1 = nn.LayerNorm(dims_model)
        self.attn2 = CrossAttention(dims_model, num_heads, dims_head)
        self.norm2 = nn.LayerNorm(dims_model)
        self.mlp = MLP(dims_model)
        self.norm3 = nn.LayerNorm(dims_model)

    def forward(self, x: torch.Tensor, cond: torch.Tensor):
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), cond=cond) + x
        x = self.mlp(self.norm3(x)) + x

        return x
