import torch
import torch.nn as nn

class AbsoluteLearnedPositionalEmbeddings(nn.Module):
    def __init__(self, block_size, embd_size):
        super().__init__()
        self.block_size, self.embd_size = block_size, embd_size
        self.wpe = nn.Embedding(self.block_size, self.embd_size)
    
    def forward(self, idx: torch.Tensor):
        B, T = idx.size()
        pos = torch.arange(0, T, device=idx.device)
        pos = self.wpe(pos).unsqueeze(0)
        return pos