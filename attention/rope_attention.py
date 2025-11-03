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
        rope_cos = torch.cos(angles) # (block_size, dk//2)
        rope_sin = torch.sin(angles) # (block_size, dk//2)

        causal_mask = torch.full((self.block_size, self.block_size), float("-inf")).triu(diagonal=1).view(1, 1, self.block_size, self.block_size)
        
        self.register_buffer("cos_angles", rope_cos)
        self.register_buffer("sin_angles", rope_sin)
        self.register_buffer("causal_mask", causal_mask)


    def forward(self, x: torch.Tensor):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.embd_size, dim=-1) # (B, T, embd_size)
        q = q.view(B, T, self.n_heads, self.dk).transpose(1, 2) # (B, H, T, dk)
        k = k.view(B, T, self.n_heads, self.dk).transpose(1, 2) # (B, H, T, dk)
        v = v.view(B, T, self.n_heads, self.dk).transpose(1, 2) # (B, H, T, dk)

        cos_angles = self.cos_angles[:T, :].unsqueeze(0).unsqueeze(0) # type: ignore[attr-defined] # (1, 1, T, dk//2)
        sin_angles = self.sin_angles[:T, :].unsqueeze(0).unsqueeze(0) # type: ignore[attr-defined] # (1, 1, T, dk//2)

        q_even = q[:, :, :, 0::2] # (B, H, T, dk//2)
        q_odd = q[:, :, :, 1::2] # (B, H, T, dk//2)
        k_even = k[:, :, :, 0::2] # (B, H, T, dk//2)
        k_odd = k[:, :, :, 1::2] # (B, H, T, dk//2)

        q_rotated_even = q_even*cos_angles - q_odd*sin_angles # (B, H, T, dk//2)
        q_rotated_odd = q_even*sin_angles + q_odd*cos_angles # (B, H, T, dk//2)
    
        k_rotated_even = k_even*cos_angles - k_odd*sin_angles # (B, H, T, dk//2)
        k_rotated_odd = k_even*sin_angles + k_odd*cos_angles # (B, H, T, dk//2)

        q_rotated = torch.zeros((B, self.n_heads, T, self.dk)) # (B, H, T, dk)
        k_rotated = torch.zeros((B, self.n_heads, T, self.dk)) # (B, H, T, dk)

        q_rotated[:, :, :, 0::2] = q_rotated_even # (B, H, T, dk)
        q_rotated[:, :, :, 1::2] = q_rotated_odd # (B, H, T, dk)

        k_rotated[:, :, :, 0::2] = k_rotated_even # (B, H, T, dk)
        k_rotated[:, :, :, 1::2] = k_rotated_odd # (B, H, T, dk)

        attn = (q_rotated @ k_rotated.transpose(-1, -2)) / math.sqrt(self.dk) # (B, H, T, T)
        attn = attn + self.causal_mask[:, :, :T, :T] # type: ignore[attr-defined]
        attn = F.softmax(attn, dim=-1) # (B, H, T, T)
        attn = self.dropout(attn) # (B, H, T, T)

        scores = attn @ v # (B, H, T, dk)
        scores = scores.transpose(1, 2).contiguous().view(B, T, self.embd_size) # (B, T, embd_size)

        y = self.dropout(self.c_proj(scores)) # (B, T, embd_size)

        return y