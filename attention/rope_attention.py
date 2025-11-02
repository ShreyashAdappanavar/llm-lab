import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RopeCausalSelfAttention(nn.Module):
    def __init__(self, embd_size, n_heads, dropout, block_size, rope_base)->None:
        super().__init__
        self.embd_size, self.n_heads, self.dropout, self.block_size, self.rope_base = embd_size, n_heads, dropout, block_size, rope_base
        assert embd_size%n_heads==0, "embedding size not divisible by heads"

        self.c_attn = nn.Linear(self.embd_size, 3*self.embd_size)
        self.c_proj = nn.Linear(self.embd_size,self.embd_size)
        self.dropout = nn.Dropout(self.dropout)