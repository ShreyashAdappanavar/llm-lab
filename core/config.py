from dataclasses import dataclass

@dataclass
class GPTConfig:
    vocab_size: int   # V
    block_size: int   # Blk_Size
    embd_size: int    # C
    n_heads: int      # H
    n_layer: int      # N
    dropout: float    # P
