import torch
import torch.nn as nn
import math
from torch.nn import functional as F
from .config import GPTConfig


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        assert config.embd_size % config.n_heads == 0, "embedding size not divisible by heads"

        self.c_attn = nn.Linear(config.embd_size, 3 * config.embd_size, bias=True)
        self.c_proj = nn.Linear(config.embd_size, config.embd_size, bias=True)
        setattr(self.c_proj, "resid_scale", True)
        self.dropout = nn.Dropout(config.dropout)

        # (1, 1, block_size, block_size)
        self.register_buffer(
            "causal_mask",
            torch.full((config.block_size, config.block_size), float("-inf"))
            .triu(diagonal=1)
            .view(1, 1, config.block_size, config.block_size),
        )

    def forward(self, x: torch.Tensor):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.config.embd_size, dim=-1)
        q = q.view(B, T, self.config.n_heads, C // self.config.n_heads).transpose(1, 2)
        k = k.view(B, T, self.config.n_heads, C // self.config.n_heads).transpose(1, 2)
        v = v.view(B, T, self.config.n_heads, C // self.config.n_heads).transpose(1, 2)

        att = (q @ k.transpose(-1, -2)) / math.sqrt(C // self.config.n_heads)
        att = att + self.causal_mask[:, :, :T, :T] # type: ignore[attr-defined]
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        y = self.dropout(self.c_proj(y))
        return y

