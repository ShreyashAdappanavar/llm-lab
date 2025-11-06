import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

class MQACausalSelfAttention(nn.Module):
    def __init__(self, embd_size, n_heads, dropout, block_size) -> None:
        super().__init__()
        self.embd_size, self.n_heads, self.dropout, self.block_size = embd_size, n_heads, dropout, block_size
        assert embd_size%n_heads==0, "embedding size not divisible by heads"
        self.dk = self.embd_size // self.n_heads
        self.q_proj = nn.Linear(self.embd_size, self.embd_size)
        self.kv_proj = nn.Linear(self.embd_size, 2 * self.dk)

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
        B, T, C = x.size()