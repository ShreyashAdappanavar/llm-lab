import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

class CausalSelfAttentionWithKVCache(nn.Module):
    def __init__(self, embd_size, n_heads, dropout, block_size) -> None:
        super().__init__()
        self.embd_size, self.n_heads, self.dropout, self.block_size = embd_size, n_heads, dropout, block_size
        assert embd_size%n_heads==0, "embedding size not divisible by heads"
        self.dk = self.embd_size // self.n_heads

        self.c_attn = nn.Linear(self.embd_size, 3*self.embd_size)
        self.c_proj = nn.Linear(self.embd_size,self.embd_size)
        self.dropout = nn.Dropout(self.dropout)

        # (1, 1, block_size, block_size)
        self.register_buffer(
            "causal_mask",
            torch.full((self.block_size, self.block_size), float("-inf"))
            .triu(diagonal=1)
            .view(1, 1, self.block_size, self.block_size),
        )
    
    def forward(self, x: torch.Tensor, old_kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.embd_size, dim=-1) # (B, T, embd_size)
        q = q.view(B, T, self.n_heads, self.dk).transpose(1, 2) # (B, H, T, dk)
        k = k.view(B, T, self.n_heads, self.dk).transpose(1, 2) # (B, H, T, dk)
        v = v.view(B, T, self.n_heads, self.dk).transpose(1, 2) # (B, H, T, dk)

        if old_kv_cache:
            old_k, old_v = old_kv_cache
            k = torch.cat((old_k, k), dim=1)
            v = torch.cat((old_v, v), dim=1)

        else:
            att = (q @ k.transpose(-1, -2)) / math.sqrt(self.dk) # (B, H, T, T)
            att = att + self.causal_mask[:, :, :T, :T] # type: ignore[attr-defined]
            att = F.softmax(att, dim=-1) # (B, H, T, T)
            att = self.dropout(att) # (B, H, T, T)
            y = att @ v # (B, H, T, dk)
            y = y.transpose(1, 2).contiguous().view(B, T, C) # (B, T, embd_size)

            y = self.dropout(self.c_proj(y)) # (B, T, embd_size)
            return y, (k, v)