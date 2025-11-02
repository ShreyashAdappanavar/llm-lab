import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RopeCausalSelfAttention(nn.Module):
    def __init__(self, embd_size, n_heads, dropout, block_size, rope_base)->None:
        super().__init__()
        self.embd_size, self.n_heads, self.dropout, self.block_size, self.rope_base = embd_size, n_heads, dropout, block_size, rope_base
        assert embd_size%n_heads==0, "embedding size not divisible by heads"
        self.dk = self.embd_size // self.n_heads
        assert self.dk % 2 == 0, "(embd_size // n_heads) MUST be even"


        self.c_attn = nn.Linear(self.embd_size, 3*self.embd_size)
        self.c_proj = nn.Linear(self.embd_size,self.embd_size)
        self.dropout = nn.Dropout(self.dropout)

        theta_indx = torch.arange(0, self.dk//2)
        freq_idx = torch.pow(self.rope_base, -2*theta_indx/self.dk).unsqueeze(0) # (1 , dk//2)

        pos_idx = torch.arange(0, self.block_size).unsqueeze(1) # (block_size, 1)
        angles = pos_idx * freq_idx # (block_size, dk//2)

        freqs_complex = torch.polar(torch.tensor(1.0), angles)
        self.register_buffer("frequencies_complex", freqs_complex)