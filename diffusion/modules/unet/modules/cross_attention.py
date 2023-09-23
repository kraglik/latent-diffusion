import torch
from torch import nn


class CrossAttention(nn.Module):
    def __init__(self, dims_model: int, num_heads: int, head_channels: int, is_inplace: bool = True):
        super().__init__()

        self.is_inplace = is_inplace
        self.num_heads = num_heads
        self.head_channels = head_channels

        self.scale = head_channels ** -0.5

        attention_channels = head_channels * num_heads

        self.to_q = nn.Linear(dims_model, attention_channels, bias=False)
        self.to_k = nn.Linear(dims_model, attention_channels, bias=False)
        self.to_v = nn.Linear(dims_model, attention_channels, bias=False)

        self.to_out = nn.Sequential(nn.Linear(attention_channels, dims_model))

        try:
            from flash_attn.flash_attention import FlashAttention

            self.flash = FlashAttention()
            self.flash.softmax_scale = self.scale

        except ImportError:
            self.flash = None

    def forward(self, x: torch.Tensor, cond=None):
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        if self.flash is not None and self.head_channels <= 128:
            return self.flash_attention(q, k, v)
        else:
            return self.normal_attention(q, k, v)

    def flash_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        batch_size, seq_len, _ = q.shape

        qkv = torch.stack((q, k, v), dim=2)
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_channels)

        if self.head_channels <= 32:
            pad = 32 - self.head_channels
        elif self.head_channels <= 64:
            pad = 64 - self.head_channels
        elif self.head_channels <= 128:
            pad = 128 - self.head_channels
        else:
            raise ValueError(f'Head size ${self.head_channels} too large for Flash Attention')

        if pad:
            qkv = torch.cat((qkv, qkv.new_zeros(batch_size, seq_len, 3, self.num_heads, pad)), dim=-1)

        out, _ = self.flash(qkv)
        out = out[:, :, :, :self.head_channels]
        out = out.reshape(batch_size, seq_len, self.num_heads * self.head_channels)

        return self.to_out(out)

    def normal_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        q = q.view(*q.shape[:2], self.num_heads, -1)
        k = k.view(*k.shape[:2], self.num_heads, -1)
        v = v.view(*v.shape[:2], self.num_heads, -1)

        attn = torch.einsum('bihd,bjhd->bhij', q, k) * self.scale

        if self.is_inplace:
            half = attn.shape[0] // 2
            attn[half:] = attn[half:].softmax(dim=-1)
            attn[:half] = attn[:half].softmax(dim=-1)
        else:
            attn = attn.softmax(dim=-1)

        out = torch.einsum('bhij,bjhd->bihd', attn, v)
        out = out.reshape(*out.shape[:2], -1)

        return self.to_out(out)
