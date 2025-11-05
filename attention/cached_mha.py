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
        """
        T_new   : The sequence length of the new prompt. Can either be length of input prompt or 1 depending on whether old_kv_cache exists or is None.
        T_old   : The TOTAL sequence length upto the last generated token
        T_total : T_old + T_new
        """
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.embd_size, dim=-1) # (B, T_new, embd_size)
        q = q.view(B, T, self.n_heads, self.dk).transpose(1, 2) # (B, H, T_new, dk)
        k = k.view(B, T, self.n_heads, self.dk).transpose(1, 2) # (B, H, T_new, dk)
        v = v.view(B, T, self.n_heads, self.dk).transpose(1, 2) # (B, H, T_new, dk)

        if old_kv_cache is not None:
            """
            T_new = 1
            q : (B, H, 1, dk)
            """
            old_k, old_v = old_kv_cache # (B, H, T_old, dk)
            k = torch.cat((old_k, k), dim=2) # (B, H, T_total, dk)
            v = torch.cat((old_v, v), dim=2) # (B, H, T_total, dk)
            att = (q @ k.transpose(-1, -2)) / math.sqrt(self.dk) # (B, H, 1, T_total)

        else:
            att = (q @ k.transpose(-1, -2)) / math.sqrt(self.dk) # (B, H, T_new, T_new)
            att = att + self.causal_mask[:, :, :T, :T] # type: ignore[attr-defined]

        att = F.softmax(att, dim=-1) # (B, H, 1, T_total) if old_kv_cache and else: (B, H, T_new, T_new)
        att = self.dropout(att) # (B, H, 1, T_total) if old_kv_cache and else: (B, H, T_new, T_new)
        y = att @ v # (B, H, 1, dk) if old_kv_cache and else: (B, H, T_new, dk)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # (B, 1, embd_size) if old_kv_cache and else: (B, T_new, embd_size)

        y = self.dropout(self.c_proj(y)) # (B, 1, embd_size) if old_kv_cache and else: (B, T_new, embd_size)
        return y, (k, v)