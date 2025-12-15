import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

class GQACausalSelfAttention(nn.Module):
    def __init__(self, embd_size, n_heads, dropout, block_size, n_groups) -> None:
        super().__init__()
        self.embd_size, self.n_heads, self.dropout, self.block_size, self.n_groups = embd_size, n_heads, dropout, block_size, n_groups
        assert embd_size%n_heads==0, "embedding size not divisible by heads"
        assert n_heads%n_groups==0, "number of heads should be divisible by the number of groups"
        self.dk = self.embd_size // self.n_heads

        self.q_proj = nn.Linear(self.embd_size, self.embd_size)
        self.kv_proj = nn.Linear(self.embd_size, 2 * self.n_groups * self.dk)

        self.c_proj = nn.Linear(self.embd_size, self.embd_size)
        self.dropout = nn.Dropout(self.dropout)

        self.register_buffer(
            "causal_mask", 
            torch.full((self.block_size, self.block_size), float("-inf"))
            .triu(diagonal=1)
            .view(1, 1, self.block_size, self.block_size))
        

    def forward(self, x: torch.Tensor, old_kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        """
        T_new   : The sequence length of the new prompt. Can either be length of input prompt or 1 depending on whether old_kv_cache exists or is None.
        T_old   : The TOTAL sequence length upto the last generated token
        T_total : T_old + T_new
        """
        B, T, _ = x.size()
        q = self.q_proj(x) # (B, T_new, embd_size)
        q = q.view(B, T, self.n_heads, self.dk).transpose(1, 2) # (B, H, T_new, dk)

        k, v = self.kv_proj(x).split(self.n_groups * self.dk, dim=-1) # (B, T_new, n_groups * dk)
        k = k.view(B, T, self.n_groups, self.dk).transpose(1, 2) # (B, n_groups, T_new, dk)
        v = v.view(B, T, self.n_groups, self.dk).transpose(1, 2) # (B, n_groups, T_new, dk)

        # WRONG?? SHOULDNT IT BE REPEAT_INTERLEAVE?
        k = k.repeat(1, self.n_heads//self.n_groups, 1, 1) # (B, H, T_new, dk)
        v = v.repeat(1, self.n_heads//self.n_groups, 1, 1) # (B, H, T_new, dk)

        if old_kv_cache is not None:
            k_old, v_old = old_kv_cache # (B, H, T_old, dk)
            k = torch.cat((k_old, k), dim=2) # (B, H, T_total, dk)
            v = torch.cat((v_old, v), dim=2) # (B, H, T_total, dk)
            attn = (q @ k.transpose(-1, -2)) / math.sqrt(self.dk) # (B, H, T_new, T_total)
        
        else:
            attn = (q @ k.transpose(-1, -2)) / math.sqrt(self.dk) # (B, H, T_new, T_new)
            attn = attn + self.causal_mask[:, :, :T, :T] # type: ignore[attr-defined]
        
        attn = F.softmax(attn, dim=-1) # (B, H, 1, T_total) if old_kv_cache and else: (B, H, T_new, T_new)
        attn = self.dropout(attn) # (B, H, 1, T_total) if old_kv_cache and else: (B, H, T_new, T_new)
        y = attn @ v # (B, H, 1, dk) if old_kv_cache and else: (B, H, T_new, dk)
        y = y.transpose(1, 2).contiguous().view(B, T, self.embd_size) # (B, 1, embd_size) if old_kv_cache and else: (B, T_new, embd_size)

        y = self.dropout(self.c_proj(y)) # (B, 1, embd_size) if old_kv_cache and else: (B, T_new, embd_size)
        return y, (k, v)