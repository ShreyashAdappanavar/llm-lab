import torch.nn as nn
from .config import GPTConfig


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.embd_size, 4 * config.embd_size, bias=True)
        self.c_proj = nn.Linear(4 * config.embd_size, config.embd_size, bias=True)
        setattr(self.c_proj, "resid_scale", True)
        self.gelu = nn.GELU(approximate="tanh")
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))
