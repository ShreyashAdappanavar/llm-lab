import torch
import torch.nn as nn

class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, block_size, embd_size, freq=10_000) -> None:
        super().__init__()
        self.block_size=block_size
        self.embd_size=embd_size
        assert self.embd_size % 2 == 0, "Embedding size must be a multiple of 2"
        self.freq = float(freq) # integer bases with float exponents are slower in PyTorch compared to float bases with float exponents

        pos=torch.arange(0, block_size, dtype=torch.float32)
        theta=torch.arange(0, embd_size//2, dtype=torch.float32)
        theta=torch.pow(self.freq, (2*theta/self.embd_size))

        angles = pos.unsqueeze(1) / theta.unsqueeze(0)
        pe=torch.zeros((self.block_size, self.embd_size))
        pe[:, 0::2] = torch.sin(angles)
        pe[:, 1::2] = torch.cos(angles)

        self.register_buffer("positional_embedding", pe)
    
    def forward(self, idx: torch.Tensor):
        B, T = idx.size()
        assert T <= self.block_size, "The sequence length is higher than the block size"

        pe = self.positional_embedding[:T, :].to(idx.device) # type: ignore[attr-defined]
        return pe.unsqueeze(0).expand((B, -1, -1)) # expands the tensor to (B, T, embd_size) explicitly to avoid surprises during broadcasting