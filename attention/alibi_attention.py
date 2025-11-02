import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class AlibiCausalSelfAttention(nn.Module):
    def __init__(self, embd_size, n_heads, dropout, block_size) -> None:
        super().__init__()
        self.embd_size, self.n_heads, self.dropout, self.block_size = embd_size, n_heads, dropout, block_size
        assert embd_size%n_heads==0, "embedding size not divisible by heads"

        self.c_attn = nn.Linear(self.embd_size, 3*self.embd_size) # bias=True by default!
        self.c_proj = nn.Linear(self.embd_size, self.embd_size)
        self.dropout = nn.Dropout(self.dropout)

        slopes = 1/torch.pow(2, torch.arange(1, self.n_heads + 1))

        alibi_mask = (torch.arange(0, self.block_size).unsqueeze(0) - torch.arange(0, self.block_size).unsqueeze(1)).tril()
        causal_mask = torch.full((self.block_size, self.block_size), float("-inf")).triu(diagonal=1)
        mask = (alibi_mask+causal_mask).unsqueeze(0)
        slopes = slopes.unsqueeze(1).unsqueeze(1)
        mask = mask*slopes # (H, T, T)
        mask = mask.unsqueeze(0) # (1, H, T, T)

        self.register_buffer("alibi_causal_mask", mask)
    
    def forward(self, x: torch.Tensor):
        B, T, C = x.size()
        q,k,v = self.c_attn(x).split(self.embd_size, dim=-1)
        q = q.view(B, T, self.n_heads, self.embd_size//self.n_heads).transpose(1, 2) # (B, H, T, embd_size//H)
        k = k.view(B, T, self.n_heads, self.embd_size//self.n_heads).transpose(1, 2) # (B, H, T, embd_size//H)
        v = v.view(B, T, self.n_heads, self.embd_size//self.n_heads).transpose(1, 2) # (B, H, T, embd_size//H)

        att = (q @ k.transpose(-1, -2)) / math.sqrt(self.embd_size//self.n_heads) # (B, H, T, T)
        att = att + self.alibi_causal_mask[:, :, :T, :T] # type: ignore[attr-defined] # (B, H, T, T)
        att = self.dropout(F.softmax(att, dim=-1)) # (B, H, T, T)
        y = att @ v # (B, H, T, embd_size//H)
        y = y.transpose(1, 2).contiguous().view(B, T, self.embd_size)
        y = self.dropout(self.c_proj(y))
        return y