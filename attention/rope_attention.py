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

    def forward(self, x: torch.Tensor):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.embd_size, dim=-1) # (B, T, embd_size)
        q = q.view(B, T, self.n_heads, self.embd_size//self.n_heads).transpose(1, 2) # (B, H, T, embd_size//H)
        k = k.view(B, T, self.n_heads, self.embd_size//self.n_heads).transpose(1, 2) # (B, H, T, embd_size//H)
        v = v.view(B, T, self.n_heads, self.embd_size//self.n_heads).transpose(1, 2) # (B, H, T, embd_size//H)

        q = q.view(B, self.n_heads, T, 2, self.embd_size//(self.n_heads * 2))
        k = k.view(B, self.n_heads, T, 2, self.embd_size//(self.n_heads * 2))