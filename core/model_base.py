import torch
import torch.nn as nn
import math
from torch.nn import functional as F
from typing import Optional
from .config import GPTConfig
from .transformer_block import TransformerBlock


class GPTModel(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.wte = nn.Embedding(config.vocab_size, config.embd_size)
        self.wpe = nn.Embedding(config.block_size, config.embd_size)
        self.dropout = nn.Dropout(config.dropout)
        self.h = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.embd_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
            if hasattr(module, "resid_scale"):
                module.weight.data /= math.sqrt(2 * self.config.n_layer)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _compute_loss(self, logits, targets):
        B, T = targets.size()
        logits = logits.view(B * T, logits.size(-1))
        targets = targets.view(B * T)
        return F.cross_entropy(logits, targets, reduction="mean")

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None):
        B, T = idx.size()
        assert T <= self.config.block_size, "sequence length > block size"

        pos = torch.arange(0, T, device=idx.device)
        x = self.dropout(self.wte(idx) + self.wpe(pos).unsqueeze(0))

        for block in self.h:
            x = block(x)

        x = self.ln_f(x)
        logits = x @ self.wte.weight.T
        loss = None
        if targets is not None:
            loss = self._compute_loss(logits, targets)
        return logits, loss

    @torch.no_grad()
    def generate(self, input_idx, max_new_tokens, temp=1.0, top_k=1):
        for _ in range(max_new_tokens):
            idx_cond = input_idx[:, -self.config.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temp
            if top_k == 1:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                topk_vals, topk_idx = torch.topk(logits, k=top_k, dim=-1)
                probs = F.softmax(topk_vals, dim=-1)
                next_token = topk_idx.gather(-1, torch.multinomial(probs, 1))
            input_idx = torch.cat((input_idx, next_token), dim=-1)
        return input_idx


def count_parameters(model: GPTModel) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_required_tokens(model: GPTModel, recommended_tk_per_param: int = 20) -> int:
    return count_parameters(model) * recommended_tk_per_param
